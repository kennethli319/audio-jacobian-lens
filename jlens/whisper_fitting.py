# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fit decoder and encoder-to-decoder Jacobian lenses for Whisper.

The estimator preserves the released J-lens implementation's efficient trick:
each replicated batch element carries a different output-dimension cotangent,
so one backward pass produces several rows of every requested Jacobian.

For decoder sources, causal self-attention makes gradients from earlier target
positions exactly zero. For encoder sources there is no causal relationship
between audio and text axes; independent source and target masks specify the
reduction. A globally masked example produces an all-output audio lens, while
an example with one target token and its aligned encoder window contributes one
sample to an aligned estimator.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.hooks import ActivationRecorder
from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_lens import WhisperJacobianLens

logger = logging.getLogger(__name__)

TargetReduction = Literal["sum", "mean"]
WHISPER_FIT_PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class WhisperExampleJacobians:
    encoder: dict[int, torch.Tensor]
    decoder: dict[int, torch.Tensor]
    encoder_source_means: dict[int, torch.Tensor]
    target_activation_mean: torch.Tensor
    n_encoder_positions: int
    n_decoder_source_positions: int
    n_decoder_target_positions: int


def _examples_fingerprint(examples: Sequence[WhisperLensInputs]) -> str:
    """Hash exact prepared tensors and their order for safe checkpoint resume."""
    digest = hashlib.sha256()
    digest.update(f"whisper-fit-v{WHISPER_FIT_PREPROCESSING_VERSION}".encode())
    digest.update(len(examples).to_bytes(8, "big"))
    for example_index, example in enumerate(examples):
        digest.update(example_index.to_bytes(8, "big"))
        for name in (
            "input_features",
            "decoder_input_ids",
            "decoder_target_ids",
            "encoder_position_mask",
            "decoder_position_mask",
        ):
            tensor = getattr(example, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        duration = example.duration_seconds
        digest.update(b"duration_seconds")
        digest.update(b"none" if duration is None else float(duration).hex().encode())
    return digest.hexdigest()


def _resolve_layers(
    model: HFWhisperLensModel,
    encoder_source_layers: Sequence[int] | None,
    decoder_source_layers: Sequence[int] | None,
    target_decoder_layer: int | None,
) -> tuple[list[int], list[int], int]:
    target = (
        model.n_decoder_layers - 1
        if target_decoder_layer is None
        else target_decoder_layer
    )
    if target < 0:
        target += model.n_decoder_layers
    if not 0 <= target < model.n_decoder_layers:
        raise ValueError(
            f"target_decoder_layer={target_decoder_layer} out of range for "
            f"{model.n_decoder_layers} decoder layers"
        )

    if encoder_source_layers is None:
        encoder = list(range(model.n_encoder_layers))
    else:
        encoder = sorted(
            {
                layer + model.n_encoder_layers if layer < 0 else layer
                for layer in encoder_source_layers
            }
        )
    if encoder and (encoder[0] < 0 or encoder[-1] >= model.n_encoder_layers):
        raise ValueError(
            f"encoder source layers {encoder_source_layers} out of range for "
            f"{model.n_encoder_layers} layers"
        )

    if decoder_source_layers is None:
        decoder = list(range(target))
    else:
        decoder = sorted(
            {
                layer + model.n_decoder_layers if layer < 0 else layer
                for layer in decoder_source_layers
            }
        )
    if decoder and (decoder[0] < 0 or decoder[-1] >= model.n_decoder_layers):
        raise ValueError(
            f"decoder source layers {decoder_source_layers} out of range for "
            f"{model.n_decoder_layers} layers"
        )
    if decoder and decoder[-1] >= target:
        raise ValueError(
            "decoder source layers must be before target_decoder_layer="
            f"{target}; got max={decoder[-1]}"
        )
    if not encoder and not decoder:
        raise ValueError("select at least one encoder or decoder source layer")
    return encoder, decoder, target


def jacobians_for_whisper_example(
    model: HFWhisperLensModel,
    inputs: WhisperLensInputs,
    *,
    encoder_source_layers: Sequence[int] | None = None,
    decoder_source_layers: Sequence[int] | None = None,
    target_decoder_layer: int | None = None,
    dim_batch: int = 4,
    target_reduction: TargetReduction = "sum",
) -> WhisperExampleJacobians:
    """Compute all requested Jacobian matrices for one prepared audio example.

    ``target_reduction='sum'`` matches the public J-lens estimator. ``'mean'``
    divides the injected cotangent by the number of selected decoder targets,
    reducing transcript-length scale differences. Final LayerNorm removes much
    of this scalar difference during readout, but the choice is still recorded
    in fitted-artifact metadata.
    """
    if inputs.batch_size != 1:
        raise ValueError("jacobians_for_whisper_example expects one example")
    if dim_batch <= 0:
        raise ValueError(f"dim_batch must be positive, got {dim_batch}")
    if target_reduction not in ("sum", "mean"):
        raise ValueError("target_reduction must be 'sum' or 'mean'")
    encoder_layers, decoder_layers, target_layer = _resolve_layers(
        model,
        encoder_source_layers,
        decoder_source_layers,
        target_decoder_layer,
    )

    inputs = inputs.to(model.input_device)
    n_encoder_positions = int(inputs.encoder_position_mask[0].sum())
    n_decoder_positions = int(inputs.decoder_position_mask[0].sum())
    if inputs.encoder_position_mask.shape[1] != model.max_source_positions:
        raise ValueError(
            "encoder_position_mask length does not match model max_source_positions"
        )
    if n_encoder_positions == 0 or n_decoder_positions == 0:
        raise ValueError("position masks must select at least one position")

    encoder_jacobians = {
        layer: torch.zeros(model.decoder_dim, model.encoder_dim, dtype=torch.float32)
        for layer in encoder_layers
    }
    decoder_jacobians = {
        layer: torch.zeros(model.decoder_dim, model.decoder_dim, dtype=torch.float32)
        for layer in decoder_layers
    }

    n_passes = math.ceil(model.decoder_dim / dim_batch)
    decoder_at = sorted({*decoder_layers, target_layer})
    decoder_graph_root = (
        min(decoder_layers) if decoder_layers and not encoder_layers else None
    )

    encoder_recorder = ActivationRecorder(
        model.encoder_layers,
        at=encoder_layers,
        start_graph_at=min(encoder_layers) if encoder_layers else None,
    )
    decoder_recorder = ActivationRecorder(
        model.decoder_layers,
        at=decoder_at,
        start_graph_at=decoder_graph_root,
    )

    with encoder_recorder, decoder_recorder, torch.enable_grad():
        if (
            not encoder_layers
            and hasattr(model, "encode_audio")
            and hasattr(model, "forward_decoder")
        ):
            # Decoder-layer Jacobians do not require gradients through the
            # encoder. Compute the 1500-position audio stack once rather than
            # once per output-dimension batch element.
            with torch.no_grad():
                encoded_audio = model.encode_audio(inputs)
            replicated = inputs.expand_batch(dim_batch)
            model.forward_decoder(
                replicated,
                encoded_audio.expand(dim_batch, *encoded_audio.shape[1:]),
            )
        else:
            replicated = inputs.expand_batch(dim_batch)
            model.forward(replicated)
        target_activation = decoder_recorder.activations[target_layer]
        if target_activation.shape[-1] != model.decoder_dim:
            raise ValueError("captured decoder target width disagrees with config")

        named_sources: list[tuple[str, int, torch.Tensor]] = [
            ("encoder", layer, encoder_recorder.activations[layer])
            for layer in encoder_layers
        ] + [
            ("decoder", layer, decoder_recorder.activations[layer])
            for layer in decoder_layers
        ]
        source_tensors = [tensor for _, _, tensor in named_sources]

        target_positions = (
            inputs.decoder_position_mask[0]
            .nonzero(as_tuple=True)[0]
            .to(target_activation.device)
        )
        encoder_positions = (
            inputs.encoder_position_mask[0]
            .nonzero(as_tuple=True)[0]
            .to(target_activation.device)
        )
        encoder_source_means = {
            layer: encoder_recorder.activations[layer][0, encoder_positions]
            .detach()
            .float()
            .mean(dim=0)
            .cpu()
            for layer in encoder_layers
        }
        selected_target_activations = (
            target_activation[0, target_positions].detach().float()
        )
        # Match the target aggregate whose derivative defines the Jacobian.
        # The aligned estimator selects one target, so sum and mean coincide;
        # the distinction matters for the global all-targets estimator.
        target_activation_mean = (
            selected_target_activations.sum(dim=0)
            if target_reduction == "sum"
            else selected_target_activations.mean(dim=0)
        ).cpu()
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)
        target_scale = 1.0 if target_reduction == "sum" else 1.0 / n_decoder_positions

        for pass_index, dim_start in enumerate(range(0, model.decoder_dim, dim_batch)):
            n_dims = min(dim_batch, model.decoder_dim - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                target_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = target_scale

            gradients = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_tensors,
                grad_outputs=cotangent,
                retain_graph=(pass_index < n_passes - 1),
            )
            for (stream, layer, _), gradient in zip(
                named_sources, gradients, strict=True
            ):
                source_mask = (
                    inputs.encoder_position_mask[0]
                    if stream == "encoder"
                    else inputs.decoder_position_mask[0]
                )
                source_positions = source_mask.nonzero(as_tuple=True)[0].to(
                    gradient.device
                )
                rows = gradient[:n_dims, source_positions, :].float().mean(dim=1)
                destination = (
                    encoder_jacobians[layer]
                    if stream == "encoder"
                    else decoder_jacobians[layer]
                )
                destination[dim_start : dim_start + n_dims] = rows.cpu()
            del gradients

    return WhisperExampleJacobians(
        encoder=encoder_jacobians,
        decoder=decoder_jacobians,
        encoder_source_means=encoder_source_means,
        target_activation_mean=target_activation_mean,
        n_encoder_positions=n_encoder_positions,
        n_decoder_source_positions=n_decoder_positions,
        n_decoder_target_positions=n_decoder_positions,
    )


def _atomic_save(obj: object, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit_whisper(
    model: HFWhisperLensModel,
    examples: Sequence[WhisperLensInputs],
    *,
    encoder_source_layers: Sequence[int] | None = None,
    decoder_source_layers: Sequence[int] | None = None,
    target_decoder_layer: int | None = None,
    dim_batch: int = 4,
    target_reduction: TargetReduction = "sum",
    estimator_name: str = "global",
    artifact_metadata: Mapping[str, Any] | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> WhisperJacobianLens:
    """Average Whisper Jacobians over prepared examples with resume support."""
    if not examples:
        raise ValueError("fit_whisper needs at least one example")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive or None")
    encoder_layers, decoder_layers, target_layer = _resolve_layers(
        model,
        encoder_source_layers,
        decoder_source_layers,
        target_decoder_layer,
    )
    examples_fingerprint = _examples_fingerprint(examples)

    fit_config = {
        "encoder_source_layers": encoder_layers,
        "decoder_source_layers": decoder_layers,
        "target_decoder_layer": target_layer,
        "target_reduction": target_reduction,
        "estimator_name": estimator_name,
        "model_fingerprint": model.fingerprint,
        "examples_fingerprint": examples_fingerprint,
        "preprocessing_version": WHISPER_FIT_PREPROCESSING_VERSION,
        "encoder_transport": (
            "affine-activation-mean-v1" if encoder_layers else "linear"
        ),
        "artifact_metadata": dict(artifact_metadata or {}),
    }
    encoder_sum = {
        layer: torch.zeros(model.decoder_dim, model.encoder_dim, dtype=torch.float32)
        for layer in encoder_layers
    }
    decoder_sum = {
        layer: torch.zeros(model.decoder_dim, model.decoder_dim, dtype=torch.float32)
        for layer in decoder_layers
    }
    encoder_source_mean_sum = {
        layer: torch.zeros(model.encoder_dim, dtype=torch.float32)
        for layer in encoder_layers
    }
    encoder_target_mean_sum = torch.zeros(model.decoder_dim, dtype=torch.float32)
    n_done = 0
    next_index = 0

    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state.get("fit_config") != fit_config:
            raise ValueError(
                "checkpoint estimator/model configuration does not match this fit; "
                "pass resume=False to replace it"
            )
        encoder_sum = state["encoder_sum"]
        decoder_sum = state["decoder_sum"]
        encoder_source_mean_sum = state["encoder_source_mean_sum"]
        encoder_target_mean_sum = state["encoder_target_mean_sum"]
        n_done = int(state["n_done"])
        next_index = int(state["next_index"])
        logger.info("resuming Whisper fit at example %d", next_index)

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "format": "whisper-jacobian-fit-checkpoint",
                    "fit_config": fit_config,
                    "encoder_sum": encoder_sum,
                    "decoder_sum": decoder_sum,
                    "encoder_source_mean_sum": encoder_source_mean_sum,
                    "encoder_target_mean_sum": encoder_target_mean_sum,
                    "n_done": n_done,
                    "next_index": next_index,
                },
                checkpoint_path,
            )

    for example_index, example in enumerate(examples):
        if example_index < next_index:
            continue
        started = time.perf_counter()
        result = jacobians_for_whisper_example(
            model,
            example,
            encoder_source_layers=encoder_layers,
            decoder_source_layers=decoder_layers,
            target_decoder_layer=target_layer,
            dim_batch=dim_batch,
            target_reduction=target_reduction,
        )
        for layer, matrix in result.encoder.items():
            encoder_sum[layer] += matrix
            encoder_source_mean_sum[layer] += result.encoder_source_means[layer]
        for layer, matrix in result.decoder.items():
            decoder_sum[layer] += matrix
        if result.encoder:
            encoder_target_mean_sum += result.target_activation_mean
        n_done += 1
        next_index = example_index + 1
        logger.info(
            "Whisper fit example %d/%d: encoder_positions=%d "
            "decoder_positions=%d %.1fs",
            next_index,
            len(examples),
            result.n_encoder_positions,
            result.n_decoder_target_positions,
            time.perf_counter() - started,
        )
        if checkpoint_every is not None and next_index % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no examples were fitted")

    shared_metadata = {
        "target_decoder_layer": target_layer,
        "target_reduction": target_reduction,
        "estimator_name": estimator_name,
        "encoder_centering": (
            None
            if not encoder_layers
            else {
                "source_reduction": "masked-position-mean",
                "target_reduction": target_reduction,
                "corpus_reduction": "example-mean",
            }
        ),
        "model_id": model.model_id,
        "model_fingerprint": model.fingerprint,
        "examples_fingerprint": examples_fingerprint,
        "preprocessing_version": WHISPER_FIT_PREPROCESSING_VERSION,
        **dict(artifact_metadata or {}),
    }
    encoder_lens = (
        None
        if not encoder_layers
        else CrossJacobianLens(
            {layer: value / n_done for layer, value in encoder_sum.items()},
            n_examples=n_done,
            source_dim=model.encoder_dim,
            target_dim=model.decoder_dim,
            source_stream="encoder",
            target_stream="decoder",
            source_means={
                layer: value / n_done
                for layer, value in encoder_source_mean_sum.items()
            },
            target_mean=encoder_target_mean_sum / n_done,
            metadata=shared_metadata,
        )
    )
    decoder_lens = (
        None
        if not decoder_layers
        else CrossJacobianLens(
            {layer: value / n_done for layer, value in decoder_sum.items()},
            n_examples=n_done,
            source_dim=model.decoder_dim,
            target_dim=model.decoder_dim,
            source_stream="decoder",
            target_stream="decoder",
            metadata=shared_metadata,
        )
    )
    return WhisperJacobianLens(
        encoder=encoder_lens,
        decoder=decoder_lens,
        model_metadata=model.lens_metadata(),
        estimator_metadata={
            "target_decoder_layer": target_layer,
            "target_reduction": target_reduction,
            "estimator_name": estimator_name,
            "examples_fingerprint": examples_fingerprint,
            "preprocessing_version": WHISPER_FIT_PREPROCESSING_VERSION,
            "encoder_centering": (
                None
                if not encoder_layers
                else {
                    "source_reduction": "masked-position-mean",
                    "target_reduction": target_reduction,
                    "corpus_reduction": "example-mean",
                }
            ),
            **dict(artifact_metadata or {}),
        },
    )
