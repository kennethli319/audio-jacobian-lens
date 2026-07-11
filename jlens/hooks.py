# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn


class ActivationRecorder:
    """Captures residual-stream tensors at the given block indices.

    Registers a forward hook on each requested block on ``__enter__`` and
    removes them on ``__exit__``. On the next forward pass each block's output
    is stored in :attr:`activations`, keyed by block index. Stored tensors are
    not detached, so they can be passed straight to :func:`torch.autograd.grad`.

    Args:
        blocks: The sequence of residual blocks (e.g. ``model.layers``).
        at: Block indices to record at.
        start_graph_at: If given, the captured tensor at this index is marked
            ``requires_grad_(True)`` before downstream blocks see it. When the
            model's parameters all have ``requires_grad=False``, this makes the
            captured residual the leaf that roots the autograd graph, so the
            retained graph spans only this block onward.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ResidualAdder:
    """Temporarily add a fixed residual tensor after one transformer block.

    This is an intervention hook, not an activation recorder. The tensor must
    have the same ``[batch, position, width]`` shape as the selected block's
    output. Tuple-shaped Hugging Face block outputs retain every non-residual
    entry unchanged.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        layer: int,
        delta: torch.Tensor,
    ) -> None:
        if layer < 0 or layer >= len(blocks):
            raise ValueError(f"layer {layer} is outside the supplied block range")
        if delta.ndim != 3:
            raise ValueError("delta must have shape [batch, positions, width]")
        self._block = blocks[layer]
        self._delta = delta
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _hook(self, module: nn.Module, inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        delta = self._delta.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(delta.shape) != tuple(hidden.shape):
            raise ValueError(
                "intervention delta shape "
                f"{tuple(delta.shape)} does not match residual shape "
                f"{tuple(hidden.shape)}"
            )
        shifted = hidden + delta
        if torch.is_tensor(output):
            return shifted
        return (shifted, *output[1:])

    def __enter__(self) -> ResidualAdder:
        self._handle = self._block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class DecoderResidualScheduleAdder:
    """Add decoder vectors by absolute position, including with a KV cache.

    Whisper generation normally feeds one new decoder token at a time after the
    first call. A decoder pre-hook records the absolute positions represented by
    that call; block hooks then gather the requested vectors for those positions.
    The same schedule therefore works for full teacher forcing and cached free
    generation without accidentally treating every one-token call as position 0.
    """

    def __init__(
        self,
        decoder: nn.Module | None,
        blocks: Sequence[nn.Module],
        *,
        vectors_by_layer: Mapping[int, Mapping[int, torch.Tensor]],
    ) -> None:
        if not vectors_by_layer:
            raise ValueError("vectors_by_layer cannot be empty")
        checked: dict[int, dict[int, torch.Tensor]] = {}
        for layer, vectors_by_position in vectors_by_layer.items():
            if layer < 0 or layer >= len(blocks):
                raise ValueError(f"layer {layer} is outside the supplied block range")
            if not vectors_by_position:
                raise ValueError("vectors_by_position cannot be empty")
            checked_positions: dict[int, torch.Tensor] = {}
            for position, vector in vectors_by_position.items():
                if position < 0:
                    raise ValueError("intervention positions must be nonnegative")
                if vector.ndim != 1:
                    raise ValueError(
                        "each intervention vector must be one-dimensional"
                    )
                checked_positions[int(position)] = vector
            checked[int(layer)] = checked_positions
        self._decoder = decoder
        self._blocks = blocks
        self._vectors_by_layer = checked
        self._active_positions: torch.Tensor | None = None
        self._pre_handle: torch.utils.hooks.RemovableHandle | None = None
        self._block_handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _past_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            return int(get_seq_length())
        try:
            first_key = past_key_values[0][0]
        except (IndexError, TypeError):
            return 0
        return int(first_key.shape[-2])

    def _decoder_pre_hook(self, module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is None and args:
            input_ids = args[0]
        sequence = input_ids if input_ids is not None else inputs_embeds
        if sequence is None or sequence.ndim < 2:
            raise ValueError("decoder intervention could not determine input positions")
        batch_size, sequence_length = sequence.shape[:2]

        position_ids = kwargs.get("position_ids")
        cache_position = kwargs.get("cache_position")
        if position_ids is not None:
            positions = position_ids
        elif cache_position is not None:
            positions = cache_position
        else:
            start = self._past_length(kwargs.get("past_key_values"))
            positions = torch.arange(
                start,
                start + sequence_length,
                device=sequence.device,
                dtype=torch.long,
            )
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[-1] != sequence_length:
            raise ValueError("decoder position metadata does not match sequence length")
        if positions.shape[0] == 1 and batch_size > 1:
            positions = positions.expand(batch_size, -1)
        if positions.shape[0] != batch_size:
            raise ValueError("decoder position metadata does not match batch size")
        self._active_positions = positions

    def _make_block_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple]:
        vectors_by_position = self._vectors_by_layer[layer]

        def hook(module: nn.Module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            positions = self._active_positions
            if positions is None:
                positions = torch.arange(
                    hidden.shape[1], device=hidden.device, dtype=torch.long
                ).unsqueeze(0).expand(hidden.shape[0], -1)
            if hidden.ndim != 3 or positions.shape != hidden.shape[:2]:
                raise ValueError("decoder positions do not align with residual output")
            shifted = hidden.clone()
            for absolute_position, vector in vectors_by_position.items():
                if vector.numel() != hidden.shape[-1]:
                    raise ValueError(
                        f"intervention width {vector.numel()} does not match "
                        f"residual width {hidden.shape[-1]}"
                    )
                mask = positions.eq(absolute_position).unsqueeze(-1)
                shifted += mask * vector.to(
                    device=hidden.device, dtype=hidden.dtype
                ).view(1, 1, -1)
            if torch.is_tensor(output):
                return shifted
            return (shifted, *output[1:])

        return hook

    def __enter__(self) -> DecoderResidualScheduleAdder:
        try:
            if self._decoder is not None:
                self._pre_handle = self._decoder.register_forward_pre_hook(
                    self._decoder_pre_hook, with_kwargs=True
                )
            for layer in sorted(self._vectors_by_layer):
                self._block_handles.append(
                    self._blocks[layer].register_forward_hook(
                        self._make_block_hook(layer)
                    )
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        for handle in self._block_handles:
            handle.remove()
        self._block_handles = []
        self._active_positions = None
