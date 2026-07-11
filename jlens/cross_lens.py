# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Rectangular Jacobian transports between two residual streams.

The decoder-only :class:`jlens.lens.JacobianLens` stores square maps because
its source and target residuals have the same width. Encoder-decoder models do
not generally have that property. This module keeps the same readout idea while
making the source and target dimensions, and their semantic stream names,
explicit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch

_SHARD_PROVENANCE_FIELD = "shard_provenance"
_SHARD_PROVENANCE_KEYS = frozenset(
    {
        "corpus_fingerprint",
        "examples_fingerprint",
        "manifest_name",
        "requested_examples",
        "shard_count",
        "shard_id",
        "shard_index",
    }
)


def _merge_shard_metadata(
    items: Sequence[tuple[Mapping[str, Any], int]],
) -> dict[str, Any]:
    """Merge stable metadata and retain varying corpus provenance per shard."""
    if not items:
        raise ValueError("metadata merge needs at least one item")

    stable_items = [
        {
            key: value
            for key, value in metadata.items()
            if key not in _SHARD_PROVENANCE_KEYS and key != _SHARD_PROVENANCE_FIELD
        }
        for metadata, _ in items
    ]
    first_stable = stable_items[0]
    if any(stable != first_stable for stable in stable_items[1:]):
        raise ValueError("lenses disagree on stable estimator metadata")

    provenance: list[dict[str, Any]] = []
    for metadata, n_examples in items:
        existing = metadata.get(_SHARD_PROVENANCE_FIELD)
        if existing is not None:
            if not isinstance(existing, list) or any(
                not isinstance(entry, Mapping) for entry in existing
            ):
                raise ValueError("shard_provenance must be a list of mappings")
            provenance.extend(dict(entry) for entry in existing)
            continue
        entry = {
            key: metadata[key] for key in _SHARD_PROVENANCE_KEYS if key in metadata
        }
        entry["n_examples"] = n_examples
        provenance.append(entry)

    return {
        **first_stable,
        _SHARD_PROVENANCE_FIELD: provenance,
    }


class CrossJacobianLens:
    """A fitted average Jacobian from a named source stream to a target stream.

    ``jacobians[layer]`` has shape ``[target_dim, source_dim]``. A source
    activation ``h`` is transported with ``h @ J.T`` into the target residual
    basis, after which the target model's normal final norm and output head can
    be applied. When fitted activation means are present, transport is the
    affine first-order approximation ``J @ (h - mu_source) + mu_target``.
    """

    FORMAT_VERSION = 2

    def __init__(
        self,
        jacobians: Mapping[int, torch.Tensor],
        *,
        n_examples: int,
        source_dim: int,
        target_dim: int,
        source_stream: str,
        target_stream: str,
        source_means: Mapping[int, torch.Tensor] | None = None,
        target_mean: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if n_examples <= 0:
            raise ValueError(f"n_examples must be positive, got {n_examples}")
        if source_dim <= 0 or target_dim <= 0:
            raise ValueError("source_dim and target_dim must be positive")
        if not jacobians:
            raise ValueError("jacobians must contain at least one source layer")

        checked: dict[int, torch.Tensor] = {}
        expected = (target_dim, source_dim)
        for layer, matrix in jacobians.items():
            if matrix.ndim != 2 or tuple(matrix.shape) != expected:
                raise ValueError(
                    f"layer {layer} has shape {tuple(matrix.shape)}, expected {expected}"
                )
            checked[int(layer)] = matrix.detach().float().cpu()

        self.jacobians = checked
        self.source_layers = sorted(checked)
        self.n_examples = int(n_examples)
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.source_stream = source_stream
        self.target_stream = target_stream
        if (source_means is None) != (target_mean is None):
            raise ValueError(
                "source_means and target_mean must either both be provided or "
                "both be omitted"
            )
        if source_means is None:
            self.source_means: dict[int, torch.Tensor] | None = None
            self.target_mean: torch.Tensor | None = None
        else:
            source_mean_layers = {int(layer) for layer in source_means}
            if source_mean_layers != set(self.source_layers):
                raise ValueError(
                    "source_means must have exactly the same layers as jacobians"
                )
            checked_means: dict[int, torch.Tensor] = {}
            for layer in self.source_layers:
                mean = source_means[layer]
                if mean.ndim != 1 or mean.shape[0] != source_dim:
                    raise ValueError(
                        f"source mean for layer {layer} has shape "
                        f"{tuple(mean.shape)}, expected ({source_dim},)"
                    )
                checked_means[layer] = mean.detach().float().cpu()
            if target_mean.ndim != 1 or target_mean.shape[0] != target_dim:
                raise ValueError(
                    f"target_mean has shape {tuple(target_mean.shape)}, "
                    f"expected ({target_dim},)"
                )
            self.source_means = checked_means
            self.target_mean = target_mean.detach().float().cpu()
        self.metadata = dict(metadata or {})

    def __repr__(self) -> str:
        layer_range = f"{self.source_layers[0]}..{self.source_layers[-1]}"
        return (
            "CrossJacobianLens("
            f"{self.source_stream}->{self.target_stream}, "
            f"shape={self.target_dim}x{self.source_dim}, "
            f"n_examples={self.n_examples}, "
            f"source_layers=[{layer_range}] ({len(self.source_layers)} layers))"
        )

    def state_dict(self, *, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
        """Return a safe, tensor-and-primitive serialization payload."""
        return {
            "format": "cross-jacobian-lens",
            "format_version": self.FORMAT_VERSION,
            "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
            "n_examples": self.n_examples,
            "source_layers": self.source_layers,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "source_stream": self.source_stream,
            "target_stream": self.target_stream,
            "source_means": (
                None
                if self.source_means is None
                else {
                    layer: mean.to(dtype) for layer, mean in self.source_means.items()
                }
            ),
            "target_mean": (
                None if self.target_mean is None else self.target_mean.to(dtype)
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CrossJacobianLens:
        """Construct a lens from :meth:`state_dict` output."""
        if state.get("format") != "cross-jacobian-lens" or "J" not in state:
            raise ValueError("payload is not a CrossJacobianLens state")
        version = state.get("format_version")
        if version not in (1, cls.FORMAT_VERSION):
            raise ValueError(
                f"unsupported CrossJacobianLens format version {version!r}; "
                f"expected 1 or {cls.FORMAT_VERSION}"
            )
        source_means = state.get("source_means") if version >= 2 else None
        target_mean = state.get("target_mean") if version >= 2 else None
        return cls(
            state["J"],
            n_examples=state["n_examples"],
            source_dim=state["source_dim"],
            target_dim=state["target_dim"],
            source_stream=state["source_stream"],
            target_stream=state["target_stream"],
            source_means=source_means,
            target_mean=target_mean,
            metadata=state.get("metadata", {}),
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(self.state_dict(dtype=dtype), path)

    @classmethod
    def load(cls, path: str) -> CrossJacobianLens:
        return cls.from_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "cross_lens.pt",
        revision: str | None = None,
    ) -> CrossJacobianLens:
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[CrossJacobianLens]) -> CrossJacobianLens:
        """Merge lenses fitted on disjoint examples with a weighted mean."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        compatibility = (
            first.source_layers,
            first.source_dim,
            first.target_dim,
            first.source_stream,
            first.target_stream,
        )
        for other in lenses[1:]:
            other_compatibility = (
                other.source_layers,
                other.source_dim,
                other.target_dim,
                other.source_stream,
                other.target_stream,
            )
            if other_compatibility != compatibility:
                raise ValueError("lenses disagree on shape, streams, or layers")
            if (other.source_means is None) != (first.source_means is None):
                raise ValueError("cannot merge centered and uncentered lenses")

        merged_metadata = _merge_shard_metadata(
            [(lens.metadata, lens.n_examples) for lens in lenses]
        )

        n_total = sum(lens.n_examples for lens in lenses)
        merged = {
            layer: sum(lens.jacobians[layer] * lens.n_examples for lens in lenses)
            / n_total
            for layer in first.source_layers
        }
        merged_source_means = (
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
        merged_target_mean = (
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
            merged,
            n_examples=n_total,
            source_dim=first.source_dim,
            target_dim=first.target_dim,
            source_stream=first.source_stream,
            target_stream=first.target_stream,
            source_means=merged_source_means,
            target_mean=merged_target_mean,
            metadata=merged_metadata,
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport ``[..., source_dim]`` residuals into the target basis."""
        if residual.shape[-1] != self.source_dim:
            raise ValueError(
                f"residual width {residual.shape[-1]} != source_dim {self.source_dim}"
            )
        matrix = self.jacobians[layer].to(device=residual.device, dtype=residual.dtype)
        if self.source_means is None:
            return residual @ matrix.T
        assert self.target_mean is not None
        source_mean = self.source_means[layer].to(
            device=residual.device, dtype=residual.dtype
        )
        target_mean = self.target_mean.to(device=residual.device, dtype=residual.dtype)
        return (residual - source_mean) @ matrix.T + target_mean

    def vocabulary_directions(
        self, unembedding_weight: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Return token directions ``W_U @ J`` in source-layer coordinates."""
        if unembedding_weight.ndim != 2:
            raise ValueError("unembedding_weight must be a 2-D matrix")
        if unembedding_weight.shape[1] != self.target_dim:
            raise ValueError(
                f"unembedding width {unembedding_weight.shape[1]} != "
                f"target_dim {self.target_dim}"
            )
        matrix = self.jacobians[layer].to(
            device=unembedding_weight.device, dtype=unembedding_weight.dtype
        )
        return unembedding_weight @ matrix
