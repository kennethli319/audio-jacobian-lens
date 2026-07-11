# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Projected Jacobian fitting for the MLX LFM2.5 language backbone.

An exact dense LFM Jacobian has 2048 x 2048 entries per source layer and would
require one reverse pass per target dimension.  This module instead uses a
seeded subset of an orthogonal Hadamard basis ``R[k, target_dim]``.  Native MLX
VJPs compute ``R @ J_l`` and
:class:`~jlens.projected_lens.ProjectedCrossJacobianLens` stores the unbiased
rank-``k`` estimate ``R.T @ (R @ J_l) / k``.  At ``k == target_dim`` the basis
is complete and reconstructs the dense estimator exactly.

This changes estimator variance, not the underlying Jacobian definition.  The
artifact records the projection dimension, seed, target reduction, runtime,
and exact quantized checkpoint fingerprint so the approximation cannot be
mistaken for the dense reference estimator or applied to a different model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from jlens.mlx_lfm import LFMLensInputs, MLXLFMModel, _as_float32_numpy
from jlens.projected_lens import ProjectedCrossJacobianLens

TargetReduction = Literal["sum", "mean"]
PROJECTION_METHOD = "subsampled_hadamard_output_probe_vjp"


@dataclass(frozen=True)
class LFMProjectedExample:
    """Projected VJP responses and timing for one generated conversation."""

    source_factors: dict[int, torch.Tensor]
    source_means: dict[int, torch.Tensor]
    target_mean: torch.Tensor
    projection_seconds: float
    n_target_positions: int


def rademacher_target_factors(
    target_dim: int, projection_dim: int, *, seed: int
) -> torch.Tensor:
    """Return deterministic ±1 target probes without touching global RNG state."""
    if target_dim <= 0 or projection_dim <= 0:
        raise ValueError("target_dim and projection_dim must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(
            0,
            2,
            (projection_dim, target_dim),
            generator=generator,
            dtype=torch.int8,
        ).float()
        * 2
        - 1
    )


def hadamard_target_factors(
    target_dim: int, projection_dim: int, *, seed: int
) -> torch.Tensor:
    """Return a seeded subset of an orthogonal ±1 Hadamard basis.

    LFM2.5's 2048-wide residual stream is a power of two.  Sampling rows without
    replacement reduces redundant probes, while the full basis satisfies
    ``H.T @ H / target_dim == I`` exactly.
    """
    if target_dim <= 0 or target_dim & (target_dim - 1):
        raise ValueError("Hadamard target_dim must be a positive power of two")
    if not 0 < projection_dim <= target_dim:
        raise ValueError("projection_dim must be in [1, target_dim]")
    factors = torch.ones(1, 1)
    while factors.shape[0] < target_dim:
        factors = torch.cat(
            (
                torch.cat((factors, factors), dim=1),
                torch.cat((factors, -factors), dim=1),
            ),
            dim=0,
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randperm(target_dim, generator=generator)[:projection_dim]
    return factors[rows].contiguous()


def _resolve_language_layers(
    model: MLXLFMModel,
    source_layers: Sequence[int] | None,
    target_layer: int | None,
) -> tuple[list[int], int]:
    target = model.n_language_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += model.n_language_layers
    if not 0 <= target < model.n_language_layers:
        raise ValueError(
            f"target_layer={target_layer} is outside {model.n_language_layers} layers"
        )
    if source_layers is None:
        sources = list(range(target))
    else:
        sources = sorted(
            {
                layer + model.n_language_layers if layer < 0 else layer
                for layer in source_layers
            }
        )
    if not sources:
        raise ValueError("select at least one language source layer")
    if sources[0] < 0 or sources[-1] >= target:
        raise ValueError(
            f"source layers must be in [0, {target}); got {sources}"
        )
    return sources, target


def lfm_examples_fingerprint(examples: Sequence[LFMLensInputs]) -> str:
    """Hash exact teacher-forced arrays, target positions, and their order."""
    if not examples:
        raise ValueError("examples fingerprint needs at least one example")
    digest = hashlib.sha256()
    digest.update(b"mlx-lfm-projected-fit-v1")
    digest.update(len(examples).to_bytes(8, "big"))
    for example_index, example in enumerate(examples):
        digest.update(example_index.to_bytes(8, "big"))
        for name in ("text_tokens", "audio_features", "audio_codes", "modalities"):
            value = getattr(example, name)
            digest.update(name.encode())
            if value is None:
                digest.update(b"none")
                continue
            array = _as_float32_numpy(value)
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(
            json.dumps(
                {
                    "prediction_positions": example.prediction_positions,
                    "target_token_ids": example.target_token_ids,
                    "duration_seconds": example.duration_seconds,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def projected_vjps_for_lfm_example(
    model: MLXLFMModel,
    inputs: LFMLensInputs,
    target_factors: torch.Tensor,
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    target_reduction: TargetReduction = "sum",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> LFMProjectedExample:
    """Compute ``R @ J_l`` for every requested language source layer."""
    import mlx.core as mx

    sources, target = _resolve_language_layers(model, source_layers, target_layer)
    if target_reduction not in ("sum", "mean"):
        raise ValueError("target_reduction must be 'sum' or 'mean'")
    if target_factors.ndim != 2 or tuple(target_factors.shape[1:]) != (
        model.language_dim,
    ):
        raise ValueError(
            "target_factors must have shape "
            f"[projection_dim, {model.language_dim}]"
        )
    if not bool(torch.isfinite(target_factors).all()):
        raise ValueError("target_factors contain non-finite values")

    trace = model.forward_trace(inputs)
    prediction_positions = mx.array(inputs.prediction_positions, dtype=mx.int32)
    n_positions = len(inputs.prediction_positions)
    selected_target = trace.activations[target][:, prediction_positions, :]
    target_mean_mx = (
        selected_target.sum(axis=1)[0]
        if target_reduction == "sum"
        else selected_target.mean(axis=1)[0]
    )
    mx.eval(target_mean_mx)

    mlx_probes = mx.array(target_factors.detach().float().cpu().numpy())
    source_factors: dict[int, torch.Tensor] = {}
    source_means: dict[int, torch.Tensor] = {}
    started = time.perf_counter()
    for source_index, source_layer in enumerate(sources):
        source = trace.activations[source_layer]
        selected_source = source[:, prediction_positions, :]
        source_means[source_layer] = torch.from_numpy(
            _as_float32_numpy(selected_source.mean(axis=1)[0])
        )

        def target_from_source(
            source_residual: Any, start_layer: int = source_layer
        ) -> Any:
            hidden = source_residual
            for layer_index in range(start_layer + 1, target + 1):
                layer = model.model.lfm.layers[layer_index]
                mask = (
                    trace.attention_mask
                    if layer.is_attention_layer
                    else trace.convolution_mask
                )
                hidden = layer(hidden, mask, cache=None)
            selected = hidden[:, prediction_positions, :]
            return (
                selected.sum(axis=1)[0]
                if target_reduction == "sum"
                else selected.mean(axis=1)[0]
            )

        responses: list[torch.Tensor] = []
        for probe_index in range(target_factors.shape[0]):
            _, gradients = mx.vjp(
                target_from_source,
                [source],
                [mlx_probes[probe_index]],
            )
            response = gradients[0][:, prediction_positions, :].mean(axis=1)[0]
            responses.append(torch.from_numpy(_as_float32_numpy(response)))
            if progress is not None:
                progress(
                    {
                        "event": "probe",
                        "source_layer": source_layer,
                        "source_index": source_index,
                        "source_count": len(sources),
                        "probe_index": probe_index,
                        "probe_count": int(target_factors.shape[0]),
                    }
                )
        source_factors[source_layer] = torch.stack(responses)
        mx.clear_cache()

    return LFMProjectedExample(
        source_factors=source_factors,
        source_means=source_means,
        target_mean=torch.from_numpy(_as_float32_numpy(target_mean_mx)),
        projection_seconds=time.perf_counter() - started,
        n_target_positions=n_positions,
    )


def fit_mlx_lfm_language_lens(
    model: MLXLFMModel,
    examples: Sequence[LFMLensInputs],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    projection_dim: int = 512,
    projection_seed: int = 0,
    target_reduction: TargetReduction = "sum",
    center: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ProjectedCrossJacobianLens:
    """Fit an averaged projected language->language lens for LFM2.5 Audio."""
    if not examples:
        raise ValueError("fit needs at least one prepared LFM example")
    sources, target = _resolve_language_layers(
        model, source_layers, target_layer
    )
    target_factors = hadamard_target_factors(
        model.language_dim, projection_dim, seed=projection_seed
    )
    results: list[LFMProjectedExample] = []
    for example_index, example in enumerate(examples):
        if progress is not None:
            progress(
                {
                    "event": "example_start",
                    "example_index": example_index,
                    "example_count": len(examples),
                }
            )
        result = projected_vjps_for_lfm_example(
            model,
            example,
            target_factors,
            source_layers=sources,
            target_layer=target,
            target_reduction=target_reduction,
            progress=progress,
        )
        results.append(result)
        if progress is not None:
            progress(
                {
                    "event": "example_complete",
                    "example_index": example_index,
                    "example_count": len(examples),
                    "projection_seconds": result.projection_seconds,
                }
            )

    averaged_factors = {
        layer: torch.stack(
            [result.source_factors[layer] for result in results]
        ).mean(dim=0)
        for layer in sources
    }
    source_means = (
        None
        if not center
        else {
            layer: torch.stack(
                [result.source_means[layer] for result in results]
            ).mean(dim=0)
            for layer in sources
        }
    )
    target_mean = (
        None
        if not center
        else torch.stack([result.target_mean for result in results]).mean(dim=0)
    )
    metadata = {
        **model.lens_metadata(),
        "estimator": "projected_average_jacobian",
        "projection_method": PROJECTION_METHOD,
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
        "target_layer": target,
        "target_reduction": target_reduction,
        "centered": center,
        "examples_fingerprint": lfm_examples_fingerprint(examples),
        "target_positions": sum(result.n_target_positions for result in results),
        "projection_seconds": sum(result.projection_seconds for result in results),
    }
    return ProjectedCrossJacobianLens(
        target_factors,
        averaged_factors,
        n_examples=len(examples),
        source_dim=model.language_dim,
        target_dim=model.language_dim,
        source_stream="language",
        target_stream="language",
        projection_method=PROJECTION_METHOD,
        source_means=source_means,
        target_mean=target_mean,
        metadata=metadata,
    )
