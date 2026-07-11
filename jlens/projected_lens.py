# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Factorized cross-stream Jacobian transports.

This module stores a projected approximation rather than a dense
``[target_dim, source_dim]`` Jacobian for every layer.  Given shared target
factors ``T`` and per-layer source factors ``S_l``, it represents

``J_l = T.T @ S_l / projection_dim``.

The artifact is independent of how a model runtime produced the probe
responses.  The factors can come from target-side VJPs, source-side JVPs, or a
separate matrix factorization; ``projection_method`` records which construction
was used.  The artifact deliberately contains no model hooks or autograd
integration.  PyTorch is used only as the canonical tensor and serialization
format.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _safe_value(value: Any, *, path: str) -> Any:
    """Return a deterministic, weights-only-safe copy of metadata."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return {
            key: _safe_value(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only None, booleans, numbers, strings, "
        "lists, and string-keyed mappings"
    )


def _checked_layer_mapping(
    values: Mapping[int, torch.Tensor],
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must contain at least one source layer")
    checked: dict[int, torch.Tensor] = {}
    for layer, tensor in values.items():
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise TypeError(f"{name} layer keys must be nonnegative integers")
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name}[{layer}] must be a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name}[{layer}] has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name}[{layer}] must be floating point")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name}[{layer}] contains non-finite values")
        checked[layer] = tensor.detach().float().cpu().contiguous()
    return {layer: checked[layer] for layer in sorted(checked)}


class ProjectedCrossJacobianLens:
    """A factorized Jacobian between named residual streams.

    ``target_factors`` has shape ``[projection_dim, target_dim]`` and every
    ``source_factors[layer]`` has shape ``[projection_dim, source_dim]``.  The
    represented dense map is ``target_factors.T @ source_factors[layer] /
    projection_dim``.  Source and target activation means optionally turn that
    linear map into the affine transport ``J_l @ (h - source_mean_l) +
    target_mean``.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        target_factors: torch.Tensor,
        source_factors: Mapping[int, torch.Tensor],
        *,
        n_examples: int,
        source_dim: int,
        target_dim: int,
        source_stream: str,
        target_stream: str,
        projection_method: str,
        source_means: Mapping[int, torch.Tensor] | None = None,
        target_mean: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(n_examples, bool) or not isinstance(n_examples, int):
            raise TypeError("n_examples must be an integer")
        if n_examples <= 0:
            raise ValueError(f"n_examples must be positive, got {n_examples}")
        if (
            isinstance(source_dim, bool)
            or not isinstance(source_dim, int)
            or isinstance(target_dim, bool)
            or not isinstance(target_dim, int)
        ):
            raise TypeError("source_dim and target_dim must be integers")
        if source_dim <= 0 or target_dim <= 0:
            raise ValueError("source_dim and target_dim must be positive")
        if not isinstance(source_stream, str) or not source_stream:
            raise ValueError("source_stream must be a non-empty string")
        if not isinstance(target_stream, str) or not target_stream:
            raise ValueError("target_stream must be a non-empty string")
        if not isinstance(projection_method, str) or not projection_method:
            raise ValueError("projection_method must be a non-empty string")
        if not torch.is_tensor(target_factors):
            raise TypeError("target_factors must be a torch.Tensor")
        if target_factors.ndim != 2 or target_factors.shape[0] <= 0:
            raise ValueError(
                "target_factors must have shape [projection_dim, target_dim]"
            )
        if target_factors.shape[1] != target_dim:
            raise ValueError(
                f"target_factors have shape {tuple(target_factors.shape)}, "
                f"expected [projection_dim, {target_dim}]"
            )
        if not torch.is_floating_point(target_factors):
            raise TypeError("target_factors must be floating point")
        if not bool(torch.isfinite(target_factors).all()):
            raise ValueError("target_factors contain non-finite values")

        self.target_factors = (
            target_factors.detach().float().cpu().contiguous()
        )
        self.projection_dim = int(target_factors.shape[0])
        self.source_factors = _checked_layer_mapping(
            source_factors,
            name="source_factors",
            expected_shape=(self.projection_dim, source_dim),
        )
        self.source_layers = sorted(self.source_factors)
        self.n_examples = n_examples
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.source_stream = source_stream
        self.target_stream = target_stream
        self.projection_method = projection_method

        if (source_means is None) != (target_mean is None):
            raise ValueError(
                "source_means and target_mean must either both be provided or "
                "both be omitted"
            )
        if source_means is None:
            self.source_means: dict[int, torch.Tensor] | None = None
            self.target_mean: torch.Tensor | None = None
        else:
            self.source_means = _checked_layer_mapping(
                source_means,
                name="source_means",
                expected_shape=(source_dim,),
            )
            if set(self.source_means) != set(self.source_layers):
                raise ValueError(
                    "source_means must have exactly the same layers as source_factors"
                )
            if not torch.is_tensor(target_mean):
                raise TypeError("target_mean must be a torch.Tensor")
            if tuple(target_mean.shape) != (target_dim,):
                raise ValueError(
                    f"target_mean has shape {tuple(target_mean.shape)}, "
                    f"expected ({target_dim},)"
                )
            if not torch.is_floating_point(target_mean):
                raise TypeError("target_mean must be floating point")
            if not bool(torch.isfinite(target_mean).all()):
                raise ValueError("target_mean contains non-finite values")
            self.target_mean = target_mean.detach().float().cpu().contiguous()

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        checked_metadata = _safe_value(dict(metadata or {}), path="metadata")
        recorded_method = checked_metadata.get("projection_method")
        if recorded_method is not None and recorded_method != projection_method:
            raise ValueError(
                "metadata projection_method disagrees with projection_method"
            )
        self.metadata = _safe_value(
            {**checked_metadata, "projection_method": projection_method},
            path="metadata",
        )

    def __repr__(self) -> str:
        layer_range = f"{self.source_layers[0]}..{self.source_layers[-1]}"
        return (
            "ProjectedCrossJacobianLens("
            f"{self.source_stream}->{self.target_stream}, "
            f"shape={self.target_dim}x{self.source_dim}, "
            f"projection_dim={self.projection_dim}, "
            f"projection_method={self.projection_method!r}, "
            f"n_examples={self.n_examples}, "
            f"source_layers=[{layer_range}] ({len(self.source_layers)} layers))"
        )

    def state_dict(self, *, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
        """Return a sorted tensor-and-primitive serialization payload."""
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("serialization dtype must be floating point")
        return {
            "format": "projected-cross-jacobian-lens",
            "format_version": self.FORMAT_VERSION,
            "target_factors": self.target_factors.to(dtype).clone(),
            "source_factors": {
                layer: self.source_factors[layer].to(dtype).clone()
                for layer in self.source_layers
            },
            "n_examples": self.n_examples,
            "source_layers": list(self.source_layers),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "source_stream": self.source_stream,
            "target_stream": self.target_stream,
            "projection_method": self.projection_method,
            "source_means": (
                None
                if self.source_means is None
                else {
                    layer: self.source_means[layer].to(dtype).clone()
                    for layer in self.source_layers
                }
            ),
            "target_mean": (
                None
                if self.target_mean is None
                else self.target_mean.to(dtype).clone()
            ),
            "metadata": _safe_value(self.metadata, path="metadata"),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> ProjectedCrossJacobianLens:
        """Construct an artifact from :meth:`state_dict` output."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if state.get("format") != "projected-cross-jacobian-lens":
            raise ValueError("payload is not a ProjectedCrossJacobianLens state")
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported ProjectedCrossJacobianLens format version "
                f"{state.get('format_version')!r}; expected {cls.FORMAT_VERSION}"
            )
        required = {
            "target_factors",
            "source_factors",
            "n_examples",
            "source_layers",
            "source_dim",
            "target_dim",
            "source_stream",
            "target_stream",
            "projection_method",
        }
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"projected lens state is missing fields: {missing}")
        source_factors = state["source_factors"]
        if not isinstance(source_factors, Mapping):
            raise TypeError("state['source_factors'] must be a mapping")
        factor_layers = list(source_factors)
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in factor_layers
        ):
            raise TypeError(
                "state['source_factors'] layer keys must be nonnegative integers"
            )
        reported_layers = state["source_layers"]
        if (
            not isinstance(reported_layers, list)
            or reported_layers != sorted(factor_layers)
        ):
            raise ValueError(
                "source_layers does not match the sorted source-factor layers"
            )
        return cls(
            state["target_factors"],
            source_factors,
            n_examples=state["n_examples"],
            source_dim=state["source_dim"],
            target_dim=state["target_dim"],
            source_stream=state["source_stream"],
            target_stream=state["target_stream"],
            projection_method=state["projection_method"],
            source_means=state.get("source_means"),
            target_mean=state.get("target_mean"),
            metadata=state.get("metadata", {}),
        )

    def save(self, path: str | os.PathLike[str], *, dtype: torch.dtype = torch.float16) -> None:
        """Atomically save a weights-only-safe artifact."""
        destination = os.fspath(path)
        temporary = f"{destination}.tmp.{os.getpid()}"
        try:
            torch.save(self.state_dict(dtype=dtype), temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ProjectedCrossJacobianLens:
        return cls.from_state_dict(
            torch.load(os.fspath(path), map_location="cpu", weights_only=True)
        )

    @classmethod
    def merge(
        cls, lenses: Sequence[ProjectedCrossJacobianLens]
    ) -> ProjectedCrossJacobianLens:
        """Merge source factors with identical target factors and metadata."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        compatibility = (
            first.source_layers,
            first.source_dim,
            first.target_dim,
            first.source_stream,
            first.target_stream,
            first.projection_method,
            first.metadata,
            first.source_means is not None,
        )
        for other in lenses[1:]:
            other_compatibility = (
                other.source_layers,
                other.source_dim,
                other.target_dim,
                other.source_stream,
                other.target_stream,
                other.projection_method,
                other.metadata,
                other.source_means is not None,
            )
            if other_compatibility != compatibility:
                raise ValueError(
                    "projected lenses disagree on shape, streams, layers, "
                    "projection method, centering, or metadata"
                )
            if not torch.equal(other.target_factors, first.target_factors):
                raise ValueError("projected lenses use different target factors")

        n_total = sum(lens.n_examples for lens in lenses)
        source_factors = {
            layer: sum(
                lens.source_factors[layer] * lens.n_examples for lens in lenses
            )
            / n_total
            for layer in first.source_layers
        }
        source_means = (
            None
            if first.source_means is None
            else {
                layer: sum(
                    lens.source_means[layer] * lens.n_examples
                    for lens in lenses
                    if lens.source_means is not None
                )
                / n_total
                for layer in first.source_layers
            }
        )
        target_mean = (
            None
            if first.target_mean is None
            else sum(
                lens.target_mean * lens.n_examples
                for lens in lenses
                if lens.target_mean is not None
            )
            / n_total
        )
        return cls(
            first.target_factors,
            source_factors,
            n_examples=n_total,
            source_dim=first.source_dim,
            target_dim=first.target_dim,
            source_stream=first.source_stream,
            target_stream=first.target_stream,
            projection_method=first.projection_method,
            source_means=source_means,
            target_mean=target_mean,
            metadata=first.metadata,
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply ``(h @ S_l.T) @ T / k`` without materializing ``J_l``."""
        if not torch.is_tensor(residual) or not torch.is_floating_point(residual):
            raise TypeError("residual must be a floating-point torch.Tensor")
        if residual.shape[-1] != self.source_dim:
            raise ValueError(
                f"residual width {residual.shape[-1]} != source_dim {self.source_dim}"
            )
        if layer not in self.source_factors:
            raise KeyError(f"layer {layer} is not present in this projected lens")
        target_factors = self.target_factors.to(
            device=residual.device, dtype=residual.dtype
        )
        source_factors = self.source_factors[layer].to(
            device=residual.device, dtype=residual.dtype
        )
        source = residual
        if self.source_means is not None:
            source = source - self.source_means[layer].to(
                device=residual.device, dtype=residual.dtype
            )
        transported = (
            (source @ source_factors.T) @ target_factors / self.projection_dim
        )
        if self.target_mean is not None:
            transported = transported + self.target_mean.to(
                device=residual.device, dtype=residual.dtype
            )
        return transported

    def vocabulary_directions(
        self, unembedding_weight: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Return ``W_U @ (T.T @ S_l / k)`` in source coordinates."""
        if not torch.is_tensor(unembedding_weight) or not torch.is_floating_point(
            unembedding_weight
        ):
            raise TypeError(
                "unembedding_weight must be a floating-point torch.Tensor"
            )
        if unembedding_weight.ndim != 2:
            raise ValueError("unembedding_weight must be a 2-D matrix")
        if unembedding_weight.shape[1] != self.target_dim:
            raise ValueError(
                f"unembedding width {unembedding_weight.shape[1]} != "
                f"target_dim {self.target_dim}"
            )
        if layer not in self.source_factors:
            raise KeyError(f"layer {layer} is not present in this projected lens")
        target_factors = self.target_factors.to(
            device=unembedding_weight.device, dtype=unembedding_weight.dtype
        )
        source_factors = self.source_factors[layer].to(
            device=unembedding_weight.device, dtype=unembedding_weight.dtype
        )
        return (
            (unembedding_weight @ target_factors.T)
            @ source_factors
            / self.projection_dim
        )
