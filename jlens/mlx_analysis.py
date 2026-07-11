# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Explorer payloads for the MLX LFM2.5 text-head Jacobian lens."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jlens.mlx_lfm import LFMLensInputs, MLXLFMModel
from jlens.projected_lens import ProjectedCrossJacobianLens
from jlens.whisper_analysis import (
    _cells_for_layer,
    _decode_token,
    display_token_mask,
    waveform_envelope,
    waveform_wav_data_url,
)


def _text_output_payload(
    model: MLXLFMModel,
    inputs: LFMLensInputs,
    actual_logits: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, Any]:
    targets = torch.tensor(inputs.target_token_ids, dtype=torch.long)
    tokens: list[dict[str, Any]] = []
    for position, target_id in enumerate(targets.tolist()):
        log_probs = actual_logits[position].float().log_softmax(dim=-1)
        probabilities = log_probs.exp()
        values, ids = probabilities.topk(top_k)
        entropy = float(-(probabilities * log_probs).sum())
        tokens.append(
            {
                "id": target_id,
                "text": _decode_token(model.tokenizer, target_id),
                "is_special": False,
                "start_seconds": None,
                "end_seconds": None,
                "probability": float(probabilities[target_id]),
                "log_probability": float(log_probs[target_id]),
                "entropy": entropy,
                "top_tokens": [
                    {
                        "id": int(token_id),
                        "text": _decode_token(model.tokenizer, int(token_id)),
                        "probability": float(value),
                    }
                    for token_id, value in zip(
                        ids.tolist(), values.tolist(), strict=True
                    )
                ],
            }
        )
    return {
        "text": inputs.generated_text
        or "".join(token["text"] for token in tokens),
        "tokens": tokens,
        "timing_source": "unavailable",
        "timing_quality": "unavailable",
        "semantic_role": "generated_response_text",
    }


@torch.no_grad()
def analyze_mlx_lfm_run(
    model: MLXLFMModel,
    lens: ProjectedCrossJacobianLens,
    inputs: LFMLensInputs,
    input_waveform: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Apply a projected language lens to one generated speech response."""
    model.validate_projected_lens(lens)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    captured = model.capture(inputs)
    transcription = _text_output_payload(
        model, inputs, captured.actual_logits, top_k=top_k
    )
    token_mask = display_token_mask(model.tokenizer, model.vocab_size)
    positions = [
        {
            "index": index,
            "sequence_position": sequence_position,
            "token_id": token["id"],
            "text": token["text"],
            "start_seconds": None,
            "end_seconds": None,
        }
        for index, (sequence_position, token) in enumerate(
            zip(
                inputs.prediction_positions,
                transcription["tokens"],
                strict=True,
            )
        )
    ]
    decoder_cells = []
    selected_positions = list(inputs.prediction_positions)
    for layer in lens.source_layers:
        layer_residuals = captured.language_residuals[layer][selected_positions]
        decoder_cells.append(
            _cells_for_layer(
                model,
                lens,
                layer_residuals,
                layer=layer,
                top_k=top_k,
                token_mask=token_mask,
            )
        )

    output_audio = (
        None
        if inputs.generated_audio is None
        else waveform_wav_data_url(
            inputs.generated_audio,
            sampling_rate=inputs.generated_audio_sample_rate,
        )
    )
    metadata = model.lens_metadata()
    artifact_generation = lens.metadata.get("generation")
    serving_generation = inputs.metadata.get("generation")
    generation_diagnostics = inputs.metadata.get("generation_diagnostics")
    metadata.update(
        {
            "schema_version": 2,
            "streams": ["decoder"],
            "stream_labels": {
                "decoder": "LFM language backbone",
                "output": "Tied text head",
            },
            "lens_examples": lens.n_examples,
            "artifact_generation": (
                dict(artifact_generation)
                if isinstance(artifact_generation, dict)
                else None
            ),
            "serving_generation": (
                dict(serving_generation)
                if isinstance(serving_generation, dict)
                else None
            ),
            "generation_diagnostics": (
                dict(generation_diagnostics)
                if isinstance(generation_diagnostics, dict)
                else None
            ),
            "estimator": (
                f"Projected average Jacobian · {lens.projection_method} · "
                f"rank {lens.projection_dim}/{lens.target_dim}"
            ),
            "projection": {
                "method": lens.projection_method,
                "rank": lens.projection_dim,
                "target_dim": lens.target_dim,
                "seed": lens.metadata.get("projection_seed"),
                # A complete orthogonal Hadamard basis reconstructs the dense
                # estimator.  The older independent-Rademacher pilot does not:
                # a square random probe matrix is not generally orthogonal.
                "dense_at_full_rank": (
                    lens.projection_method
                    == "subsampled_hadamard_output_probe_vjp"
                ),
            },
            "display_vocabulary": {
                "policy": "alphanumeric_lexical_tokens",
                "full_vocabulary_size": model.vocab_size,
                "display_vocabulary_size": int(token_mask.sum()),
            },
            "decoder_token_length_filter": {
                "policy": "unavailable_for_projected_lfm_pilot",
                "eligible_source_layers": [],
            },
            "capabilities": {
                "input_audio": True,
                "generated_text": True,
                "generated_audio": output_audio is not None,
                "language_jlens": True,
                "audio_encoder_jlens": False,
                "audio_codebook_jlens": False,
            },
            "warnings": [
                "This is a projected rank-limited estimate of the average Jacobian, not the dense reference estimator. Compare projection ranks before interpreting fine ordering.",
                "The displayed lens targets LFM's tied text head. It does not yet explain the eight generated audio-codebook heads or the final waveform.",
                "Text probabilities are raw teacher-forced probabilities on the model's own generated interleaved path; they are not calibrated confidence.",
                "Language-backbone readouts are ranked lexical diagnostics, not signed causal effects or a complete account of speech generation.",
            ],
        }
    )
    return {
        "audio": {
            "duration_seconds": inputs.duration_seconds,
            "waveform": waveform_envelope(input_waveform),
            "model_input_wav": waveform_wav_data_url(
                input_waveform, sampling_rate=model.input_sample_rate
            ),
            "model_input_format": (
                f"mono {model.input_sample_rate / 1000:g} kHz PCM"
            ),
            "model_output_wav": output_audio,
            "model_output_format": (
                None
                if output_audio is None
                else f"mono {inputs.generated_audio_sample_rate / 1000:g} kHz PCM"
            ),
            "model_output_duration_seconds": (
                None
                if inputs.generated_audio is None
                else float(
                    inputs.generated_audio.size
                    / inputs.generated_audio_sample_rate
                )
            ),
        },
        "transcription": transcription,
        "encoder": {"layers": [], "positions": [], "cells": []},
        "decoder": {
            "layers": lens.source_layers,
            "positions": positions,
            "cells": decoder_cells,
            "score_kind": "raw_readout_logit",
            "stream_kind": "causal_language_backbone",
            "target_layer": lens.metadata.get("target_layer"),
            "projection_rank": lens.projection_dim,
        },
        "metadata": metadata,
    }
