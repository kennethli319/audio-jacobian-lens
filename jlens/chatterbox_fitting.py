# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Projected fitted Jacobian lenses for Chatterbox-Turbo's T3 backbone.

The interactive Chatterbox trace differentiates one realized speech-code log
probability with respect to input-text positions.  This module implements the
separate, paper-style fitted lens: it averages Jacobians from an intermediate
T3 residual at speech-prediction positions to the final T3 residual at speech-
prediction positions over a corpus of teacher-forced generated trajectories.

For a target probe ``r`` and source layer ``l``, the per-example estimator is::

    mean_i d <r, reduce_j H_target[q_j]> / d H_l[q_i]

where ``q_j`` are the positions that predict ordinary speech codes.  ``reduce``
is either ``sum`` (the released J-lens convention) or ``mean``.  A zero
additive perturbation is inserted after every requested source block, allowing
one MLX VJP to return the gradients for all source layers simultaneously.

The 1,024-wide T3 stream is fitted with the same seeded Hadamard projection and
``ProjectedCrossJacobianLens`` artifact used by the local LFM implementation.
At projection rank 1,024 the orthogonal basis reconstructs the dense estimator
exactly; lower ranks are explicitly approximate.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from jlens.mlx_fitting import PROJECTION_METHOD, hadamard_target_factors
from jlens.projected_lens import ProjectedCrossJacobianLens

TargetReduction = Literal["sum", "mean"]
CHATTERBOX_FIT_PREPROCESSING_VERSION = 1
CHATTERBOX_SOURCE_STREAM = "t3_speech_position"
CHATTERBOX_TARGET_STREAM = "t3_speech_position"
JOINT_VJP_STRATEGY = "post_block_zero_delta_injection"
CHATTERBOX_CAPTURE_CONVENTION = "t3_post_block_speech_prediction_v1"
DEFAULT_CHATTERBOX_FITTED_LAYERS = (0, 4, 8, 12, 16, 20, 22)


def _integer_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    converted = tuple(values)
    if not converted:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in converted):
        raise TypeError(f"{name} must contain integers")
    if any(value < 0 for value in converted):
        raise ValueError(f"{name} must contain nonnegative integers")
    return converted


@dataclass(frozen=True)
class ChatterboxLensExample:
    """Exact teacher-forced T3 inputs used by one fitted-lens example.

    Waveform decoding is deliberately absent: S3Gen and the vocoder are not on
    the fitted T3 residual-to-residual path.
    """

    text_token_ids: tuple[int, ...]
    speech_code_ids: tuple[int, ...]
    selected_code_indices: tuple[int, ...] | None = None
    raw_text: str = ""
    normalized_text: str = ""
    record_id: str = ""

    def __post_init__(self) -> None:
        text_ids = _integer_tuple(self.text_token_ids, name="text_token_ids")
        code_ids = _integer_tuple(self.speech_code_ids, name="speech_code_ids")
        if text_ids != self.text_token_ids or code_ids != self.speech_code_ids:
            raise TypeError("token IDs must be supplied as tuples")
        selected = self.selected_code_indices
        if selected is not None:
            selected = _integer_tuple(selected, name="selected_code_indices")
            if selected != self.selected_code_indices:
                raise TypeError("selected_code_indices must be supplied as a tuple")
            if tuple(sorted(set(selected))) != selected:
                raise ValueError(
                    "selected_code_indices must be sorted and contain no duplicates"
                )
            if selected[-1] >= len(code_ids):
                raise ValueError("selected_code_indices exceed speech_code_ids")
        for name in ("raw_text", "normalized_text", "record_id"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")

    @property
    def code_indices(self) -> tuple[int, ...]:
        """Resolved relative speech-code indices selected for fitting."""
        return (
            tuple(range(len(self.speech_code_ids)))
            if self.selected_code_indices is None
            else self.selected_code_indices
        )

    @classmethod
    def from_captured_run(
        cls,
        run: Any,
        *,
        max_speech_positions: int | None = None,
        record_id: str = "",
    ) -> ChatterboxLensExample:
        """Create a compact fit example from a ``ChatterboxCapturedRun``."""
        speech_codes = tuple(int(value) for value in run.speech_code_ids)
        if max_speech_positions is not None:
            if max_speech_positions <= 0:
                raise ValueError("max_speech_positions must be positive")
            selected: tuple[int, ...] | None = tuple(
                range(min(max_speech_positions, len(speech_codes)))
            )
        else:
            selected = None
        return cls(
            text_token_ids=tuple(int(value) for value in run.text_token_ids),
            speech_code_ids=speech_codes,
            selected_code_indices=selected,
            raw_text=str(run.raw_text),
            normalized_text=str(run.normalized_text),
            record_id=record_id,
        )


@dataclass(frozen=True)
class ChatterboxProjectedExample:
    """Projected VJP responses and activation means for one T3 trajectory."""

    source_factors: dict[int, torch.Tensor]
    source_means: dict[int, torch.Tensor]
    target_mean: torch.Tensor
    projection_seconds: float
    n_target_positions: int
    speech_start: int


def chatterbox_examples_fingerprint(
    examples: Sequence[ChatterboxLensExample],
) -> str:
    """Hash exact teacher-forced token arrays, selections, and their order."""
    if not examples:
        raise ValueError("examples fingerprint needs at least one example")
    digest = hashlib.sha256()
    digest.update(
        f"mlx-chatterbox-fit-v{CHATTERBOX_FIT_PREPROCESSING_VERSION}".encode()
    )
    digest.update(len(examples).to_bytes(8, "big"))
    for example_index, example in enumerate(examples):
        digest.update(example_index.to_bytes(8, "big"))
        for name, values in (
            ("text_token_ids", example.text_token_ids),
            ("speech_code_ids", example.speech_code_ids),
            ("selected_code_indices", example.code_indices),
        ):
            digest.update(name.encode())
            array = np.asarray(values, dtype=np.int64)
            digest.update(len(array).to_bytes(8, "big"))
            digest.update(array.tobytes())
        digest.update(example.raw_text.encode("utf-8"))
        digest.update(example.normalized_text.encode("utf-8"))
        digest.update(example.record_id.encode("utf-8"))
    return digest.hexdigest()


def _resolve_layers(
    model: Any,
    source_layers: Sequence[int] | None,
    target_layer: int | None,
) -> tuple[list[int], int]:
    n_layers = int(model.n_layers)
    target = n_layers - 1 if target_layer is None else int(target_layer)
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(
            f"target_layer={target_layer} is outside {n_layers} T3 layers"
        )
    if source_layers is None:
        default_layers = getattr(model, "default_lens_layers", None)
        if default_layers is None:
            sources = [
                layer
                for layer in DEFAULT_CHATTERBOX_FITTED_LAYERS
                if layer < target
            ]
            if not sources:
                sources = list(range(target))
        else:
            sources = sorted({int(layer) for layer in default_layers})
    else:
        sources = sorted(
            {
                int(layer) + n_layers if int(layer) < 0 else int(layer)
                for layer in source_layers
            }
        )
    if not sources:
        raise ValueError("select at least one T3 source layer")
    if sources[0] < 0 or sources[-1] >= target:
        raise ValueError(f"source layers must be in [0, {target}); got {sources}")
    return sources, target


def _teacher_forced_initial_residual(
    model: Any, example: ChatterboxLensExample
) -> tuple[Any, int, Any]:
    """Build the exact T3 teacher-forced residual and selected positions."""
    import mlx.core as mx

    t3 = model.model.t3
    text_ids = mx.array([list(example.text_token_ids)], dtype=mx.int32)
    codes = mx.array([list(example.speech_code_ids)], dtype=mx.int32)
    bos = mx.full((1, 1), t3.hp.start_speech_token, dtype=mx.int32)
    speech_inputs = mx.concatenate([bos, codes[:, :-1]], axis=1)
    embeds, condition_length = t3.prepare_input_embeds(
        model.model._conds.t3, text_ids, speech_inputs
    )
    transformer = t3.tfmr
    initial = embeds + transformer.wpe(mx.arange(embeds.shape[1]))
    speech_start = int(condition_length) + len(example.text_token_ids)
    positions = mx.array(
        [speech_start + index for index in example.code_indices], dtype=mx.int32
    )
    mx.eval(initial, positions)
    return initial, speech_start, positions


def projected_vjps_for_chatterbox_example(
    model: Any,
    example: ChatterboxLensExample,
    target_factors: torch.Tensor,
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    target_reduction: TargetReduction = "sum",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ChatterboxProjectedExample:
    """Compute ``R @ J_l`` for all requested T3 source layers jointly.

    The VJP primals are zero perturbations inserted immediately after the
    requested T3 blocks.  Their gradients therefore equal gradients with
    respect to the corresponding post-block residuals, while one reverse pass
    returns every requested source layer.
    """
    import mlx.core as mx

    sources, target = _resolve_layers(model, source_layers, target_layer)
    if target_reduction not in ("sum", "mean"):
        raise ValueError("target_reduction must be 'sum' or 'mean'")
    hidden_size = int(model.hidden_size)
    if (
        not torch.is_tensor(target_factors)
        or not torch.is_floating_point(target_factors)
        or target_factors.ndim != 2
        or target_factors.shape[0] <= 0
        or target_factors.shape[1] != hidden_size
    ):
        raise ValueError(
            f"target_factors must have shape [projection_dim, {hidden_size}]"
        )
    if not bool(torch.isfinite(target_factors).all()):
        raise ValueError("target_factors contain non-finite values")
    model_metadata = dict(model.metadata())
    valid_speech_codes = int(
        model_metadata.get(
            "valid_speech_codes",
            model_metadata.get(
                "speech_vocab_size", getattr(model, "speech_vocab_size", 0)
            ),
        )
    )
    if valid_speech_codes <= 0:
        raise ValueError("model metadata must report valid ordinary speech codes")
    if any(code_id >= valid_speech_codes for code_id in example.speech_code_ids):
        raise ValueError("fit examples must contain only ordinary speech-code IDs")

    initial, speech_start, positions = _teacher_forced_initial_residual(
        model, example
    )
    transformer = model.model.t3.tfmr
    residual = initial
    activations: dict[int, Any] = {}
    for layer_index in range(target + 1):
        residual, _ = transformer.h[layer_index](residual, cache=None)
        if layer_index in sources or layer_index == target:
            activations[layer_index] = residual
    mx.eval(*activations.values())

    selected_target = activations[target][:, positions, :]
    target_mean_mx = (
        selected_target.sum(axis=1)[0]
        if target_reduction == "sum"
        else selected_target.mean(axis=1)[0]
    )
    source_means = {
        layer: torch.from_numpy(
            np.asarray(
                activations[layer][:, positions, :].mean(axis=1)[0],
                dtype=np.float32,
            ).copy()
        )
        for layer in sources
    }
    target_mean = torch.from_numpy(
        np.asarray(target_mean_mx, dtype=np.float32).copy()
    )

    zero_deltas = [mx.zeros_like(initial) for _ in sources]
    source_to_delta = {layer: index for index, layer in enumerate(sources)}

    def target_from_deltas(*deltas: Any) -> Any:
        hidden = initial
        for layer_index in range(target + 1):
            hidden, _ = transformer.h[layer_index](hidden, cache=None)
            delta_index = source_to_delta.get(layer_index)
            if delta_index is not None:
                hidden = hidden + deltas[delta_index]
        selected = hidden[:, positions, :]
        return (
            selected.sum(axis=1)[0]
            if target_reduction == "sum"
            else selected.mean(axis=1)[0]
        )

    mlx_probes = mx.array(target_factors.detach().float().cpu().numpy())
    responses: dict[int, list[torch.Tensor]] = {layer: [] for layer in sources}
    started = time.perf_counter()
    for probe_index in range(target_factors.shape[0]):
        _, gradients = mx.vjp(
            target_from_deltas,
            zero_deltas,
            [mlx_probes[probe_index]],
        )
        for layer, gradient in zip(sources, gradients, strict=True):
            response = gradient[:, positions, :].mean(axis=1)[0]
            mx.eval(response)
            responses[layer].append(
                torch.from_numpy(np.asarray(response, dtype=np.float32).copy())
            )
        if progress is not None:
            progress(
                {
                    "event": "probe",
                    "probe_index": probe_index,
                    "probe_count": int(target_factors.shape[0]),
                    "source_layers": sources,
                }
            )
        del gradients
    mx.clear_cache()

    return ChatterboxProjectedExample(
        source_factors={
            layer: torch.stack(layer_responses)
            for layer, layer_responses in responses.items()
        },
        source_means=source_means,
        target_mean=target_mean,
        projection_seconds=time.perf_counter() - started,
        n_target_positions=len(example.code_indices),
        speech_start=speech_start,
    )


def fit_mlx_chatterbox_speech_lens(
    model: Any,
    examples: Sequence[ChatterboxLensExample],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    projection_dim: int = 128,
    projection_seed: int = 0,
    target_reduction: TargetReduction = "sum",
    center: bool = False,
    artifact_metadata: Mapping[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ProjectedCrossJacobianLens:
    """Fit an averaged projected T3 speech-position Jacobian lens.

    Per-example factors are accumulated immediately, so working memory is
    independent of corpus size.
    """
    if not examples:
        raise ValueError("fit needs at least one Chatterbox example")
    sources, target = _resolve_layers(model, source_layers, target_layer)
    hidden_size = int(model.hidden_size)
    target_factors = hadamard_target_factors(
        hidden_size, projection_dim, seed=projection_seed
    )
    factor_sums = {
        layer: torch.zeros(projection_dim, hidden_size, dtype=torch.float32)
        for layer in sources
    }
    source_mean_sums = {
        layer: torch.zeros(hidden_size, dtype=torch.float32) for layer in sources
    }
    target_mean_sum = torch.zeros(hidden_size, dtype=torch.float32)
    total_target_positions = 0
    projection_seconds = 0.0

    for example_index, example in enumerate(examples):
        if progress is not None:
            progress(
                {
                    "event": "example_start",
                    "example_index": example_index,
                    "example_count": len(examples),
                }
            )
        result = projected_vjps_for_chatterbox_example(
            model,
            example,
            target_factors,
            source_layers=sources,
            target_layer=target,
            target_reduction=target_reduction,
            progress=progress,
        )
        for layer in sources:
            factor_sums[layer] += result.source_factors[layer]
            source_mean_sums[layer] += result.source_means[layer]
        target_mean_sum += result.target_mean
        total_target_positions += result.n_target_positions
        projection_seconds += result.projection_seconds
        if progress is not None:
            progress(
                {
                    "event": "example_complete",
                    "example_index": example_index,
                    "example_count": len(examples),
                    "projection_seconds": result.projection_seconds,
                    "target_positions": result.n_target_positions,
                }
            )

    n_examples = len(examples)
    model_metadata = dict(model.metadata())
    speech_vocab_size = int(
        model_metadata.get(
            "speech_vocab_size", getattr(model, "speech_vocab_size", 0)
        )
    )
    if speech_vocab_size <= 0:
        raise ValueError("model metadata must report a positive speech_vocab_size")
    valid_speech_codes = int(
        model_metadata.get("valid_speech_codes", speech_vocab_size)
    )
    if not 0 < valid_speech_codes <= speech_vocab_size:
        raise ValueError(
            "valid_speech_codes must be positive and no larger than the speech head"
        )
    metadata = {
        **model_metadata,
        "estimator": "projected_average_jacobian",
        "estimator_version": CHATTERBOX_FIT_PREPROCESSING_VERSION,
        "projection_method": PROJECTION_METHOD,
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
        "dense_at_full_rank": projection_dim == hidden_size,
        "vjp_source_strategy": JOINT_VJP_STRATEGY,
        "source_layers": sources,
        "target_layer": target,
        "capture_convention": CHATTERBOX_CAPTURE_CONVENTION,
        "source_residual": "post_block_pre_final_norm",
        "target_residual": "post_block_pre_final_norm",
        "target_head": {
            "name": "t3.speech_head",
            "semantic_kind": "speech_code",
            "vocab_size": speech_vocab_size,
            "valid_ordinary_codes": valid_speech_codes,
        },
        "position_axis": "teacher_forced_speech_prediction_positions",
        "source_reduction": "selected_position_mean",
        "target_reduction": target_reduction,
        "corpus_reduction": "example_mean",
        "centered": center,
        "examples_fingerprint": chatterbox_examples_fingerprint(examples),
        "preprocessing_version": CHATTERBOX_FIT_PREPROCESSING_VERSION,
        "target_positions": total_target_positions,
        "projection_seconds": projection_seconds,
        "artifact_metadata": dict(artifact_metadata or {}),
    }
    return ProjectedCrossJacobianLens(
        target_factors,
        {
            layer: factor_sum / n_examples
            for layer, factor_sum in factor_sums.items()
        },
        n_examples=n_examples,
        source_dim=hidden_size,
        target_dim=hidden_size,
        source_stream=CHATTERBOX_SOURCE_STREAM,
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
        source_means=(
            None
            if not center
            else {
                layer: value / n_examples
                for layer, value in source_mean_sums.items()
            }
        ),
        target_mean=(None if not center else target_mean_sum / n_examples),
        metadata=metadata,
    )


def validate_chatterbox_speech_lens(
    model: Any, lens: ProjectedCrossJacobianLens
) -> None:
    """Reject a fitted artifact for another T3 checkpoint or residual stream."""
    model_metadata = dict(model.metadata())
    expected_fingerprint = model_metadata.get("model_fingerprint")
    fitted_fingerprint = lens.metadata.get("model_fingerprint")
    if fitted_fingerprint != expected_fingerprint:
        raise ValueError(
            "lens/model fingerprint mismatch: "
            f"lens={fitted_fingerprint!r}, model={expected_fingerprint!r}"
        )
    if (
        lens.source_stream != CHATTERBOX_SOURCE_STREAM
        or lens.target_stream != CHATTERBOX_TARGET_STREAM
    ):
        raise ValueError("Chatterbox lens must map T3 speech positions to T3 speech positions")
    hidden_size = int(model.hidden_size)
    if lens.source_dim != hidden_size or lens.target_dim != hidden_size:
        raise ValueError("projected lens dimensions do not match the T3 backbone")
    target_layer = lens.metadata.get("target_layer")
    if target_layer != int(model.n_layers) - 1:
        raise ValueError("projected lens does not target the final T3 block")
    if any(layer < 0 or layer >= target_layer for layer in lens.source_layers):
        raise ValueError("projected lens source layers must precede the target block")
    if lens.metadata.get("source_layers") != lens.source_layers:
        raise ValueError("artifact source-layer metadata does not match its factors")
    if lens.metadata.get("capture_convention") != CHATTERBOX_CAPTURE_CONVENTION:
        raise ValueError("unsupported Chatterbox residual capture convention")
    is_centered = lens.source_means is not None and lens.target_mean is not None
    if lens.metadata.get("centered") is not is_centered:
        raise ValueError("artifact centering metadata disagrees with its tensors")
    target_head = lens.metadata.get("target_head")
    if not isinstance(target_head, Mapping):
        raise ValueError("artifact is missing target-head metadata")
    model_speech_vocab = int(
        model_metadata.get(
            "speech_vocab_size", getattr(model, "speech_vocab_size", 0)
        )
    )
    expected_head = {
        "name": "t3.speech_head",
        "semantic_kind": "speech_code",
        "vocab_size": model_speech_vocab,
        "valid_ordinary_codes": int(
            model_metadata.get("valid_speech_codes", model_speech_vocab)
        ),
    }
    if dict(target_head) != expected_head:
        raise ValueError("artifact target-head metadata does not match the model")
    if lens.projection_method != PROJECTION_METHOD:
        raise ValueError("unsupported Chatterbox projection method")


def chatterbox_speech_lens_logits(
    model: Any,
    lens: ProjectedCrossJacobianLens,
    run: Any,
    *,
    layers: Sequence[int] | None = None,
    speech_code_indices: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Read fitted speech-head logits from a captured Chatterbox run.

    Returned tensors have shape ``[positions, speech_head_vocabulary]`` and use
    the raw T3 head before repetition-penalty or sampling transforms.
    """
    import mlx.core as mx

    validate_chatterbox_speech_lens(model, lens)
    selected_layers = lens.source_layers if layers is None else list(layers)
    unknown_layers = sorted(set(selected_layers) - set(lens.source_layers))
    if unknown_layers:
        raise ValueError(f"layers are not present in the fitted lens: {unknown_layers}")
    indices = (
        tuple(range(len(run.speech_code_ids)))
        if speech_code_indices is None
        else tuple(speech_code_indices)
    )
    if not indices:
        raise ValueError("select at least one speech-code position")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(run.speech_code_ids)
        for index in indices
    ):
        raise ValueError("speech_code_indices are outside the captured run")
    sequence_positions = [run.speech_start + index for index in indices]
    transformer = model.model.t3.tfmr
    output: dict[int, torch.Tensor] = {}
    for layer in selected_layers:
        source = np.asarray(
            run.post_block_residuals[layer][0, sequence_positions, :],
            dtype=np.float32,
        ).copy()
        transported = lens.transport(torch.from_numpy(source), layer)
        target_residual = mx.array(transported.detach().float().cpu().numpy())
        logits = model.model.t3.speech_head(transformer.ln_f(target_residual))
        mx.eval(logits)
        output[layer] = torch.from_numpy(
            np.asarray(logits, dtype=np.float32).copy()
        )
    return output


def example_manifest_record(example: ChatterboxLensExample) -> dict[str, Any]:
    """Return the stable JSON-compatible portion of a prepared fit example."""
    return json.loads(
        json.dumps(
            {
                "record_id": example.record_id,
                "raw_text": example.raw_text,
                "normalized_text": example.normalized_text,
                "text_token_ids": list(example.text_token_ids),
                "speech_code_ids": list(example.speech_code_ids),
                "selected_code_indices": list(example.code_indices),
            },
            sort_keys=True,
        )
    )
