"""Run a reproducible Whisper residual causal trace on one local audio file."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from jlens.audio_io import AUDIO_PREPROCESSING_VERSION, decode_audio_bytes
from jlens.fit_whisper_cli import _device
from jlens.hooks import DecoderResidualScheduleAdder, ResidualAdder
from jlens.whisper import HFWhisperLensModel
from jlens.whisper_causal import (
    DecoderIntervention,
    DecoderInterventionSchedule,
    EncoderIntervention,
    EncoderInterventionSchedule,
    candidate_text_token_ids,
    decoder_lens_contrast_direction,
    encoder_lens_contrast_direction,
    prepare_candidate_inputs,
    random_decoder_direction,
    random_encoder_direction,
    run_decoder_intervention_schedule,
    run_encoder_intervention_schedule,
    score_candidate_text,
    vocabulary_token_ids_starting_with,
)
from jlens.whisper_lens import WhisperJacobianLens


def _span_from_seconds(
    *, start_seconds: float, end_seconds: float, valid_positions: int
) -> tuple[int, int]:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than a nonnegative start_seconds")
    start = max(0, math.floor(start_seconds / 0.02))
    end = min(valid_positions, math.ceil(end_seconds / 0.02))
    if end <= start:
        raise ValueError("selected seconds contain no valid 20 ms encoder positions")
    return start, end


def _score_payload(score) -> dict[str, Any]:
    return {
        "text": score.text,
        "token_ids": list(score.token_ids),
        "token_log_probabilities": list(score.token_log_probabilities),
        "token_probabilities": [
            math.exp(value) for value in score.token_log_probabilities
        ],
        "total_log_probability": score.total_log_probability,
        "mean_log_probability": score.mean_log_probability,
    }


def _parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers:
        raise ValueError("layers cannot be empty")
    if layers != sorted(layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be unique and ordered, for example 1,2,3")
    return layers


def _parse_segments(values: list[str] | None) -> list[tuple[float, float, str]]:
    """Parse repeated ``start:end:text`` target-piece specifications."""
    segments: list[tuple[float, float, str]] = []
    for value in values or []:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError("segments must use start:end:text, for example 0:0.3: Y")
        start, end, text = float(parts[0]), float(parts[1]), parts[2]
        if not text:
            raise ValueError("segment text cannot be empty")
        segments.append((start, end, text))
    return segments


def _tokenizer_faithful_segments(
    model: HFWhisperLensModel,
    text: str,
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[tuple[float, float, str]]:
    """Divide one selected span across the target's actual BPE token pieces."""
    token_ids = candidate_text_token_ids(model, text)
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("end_seconds must be greater than start_seconds")
    segments: list[tuple[float, float, str]] = []
    for index, token_id in enumerate(token_ids):
        piece = model.tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        if not piece:
            raise ValueError(f"target token {token_id} decodes to an empty piece")
        piece_start = start_seconds + duration * index / len(token_ids)
        piece_end = start_seconds + duration * (index + 1) / len(token_ids)
        segments.append((piece_start, piece_end, piece))
    return segments


def _candidate_outcomes(
    model: HFWhisperLensModel,
    inputs,
    *,
    positive_text: str,
    negative_text: str,
    comparison_texts: list[str],
    logits: torch.Tensor,
    first_prediction_position: int,
    schedule: EncoderInterventionSchedule | None = None,
    deltas: dict[int, torch.Tensor] | None = None,
    decoder_vectors: dict[int, dict[int, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    scores = {
        text: score_candidate_text(
            model,
            inputs,
            text,
            schedule=schedule,
            deltas=deltas,
            decoder_vectors=decoder_vectors,
        )
        for text in comparison_texts
    }
    positive = scores[positive_text]
    negative = scores[negative_text]
    mean_comparison_scores = torch.tensor(
        [scores[text].mean_log_probability for text in comparison_texts]
    )
    total_comparison_scores = torch.tensor(
        [scores[text].total_log_probability for text in comparison_texts]
    )
    restricted_mean_probabilities = mean_comparison_scores.softmax(dim=0).tolist()
    restricted_total_probabilities = total_comparison_scores.softmax(dim=0).tolist()
    first_token_probabilities = logits[0, first_prediction_position].float().softmax(
        dim=-1
    )
    return {
        "positive": _score_payload(positive),
        "negative": _score_payload(negative),
        "mean_log_probability_positive_minus_negative": (
            positive.mean_log_probability - negative.mean_log_probability
        ),
        "total_log_probability_positive_minus_negative": (
            positive.total_log_probability - negative.total_log_probability
        ),
        "comparison_primary_metric": (
            "restricted_total_candidate_token_log_probability_softmax"
        ),
        "candidate_token_paths_include_eos": False,
        "comparison_set": [
            {
                **_score_payload(scores[text]),
                "restricted_mean_log_probability_softmax": mean_probability,
                "restricted_total_log_probability_softmax": total_probability,
                "first_token_id": int(scores[text].token_ids[0]),
                "first_token_softmax_probability": float(
                    first_token_probabilities[scores[text].token_ids[0]].cpu()
                ),
            }
            for text, mean_probability, total_probability in zip(
                comparison_texts,
                restricted_mean_probabilities,
                restricted_total_probabilities,
                strict=True,
            )
        ],
        "first_prediction_position": first_prediction_position,
    }


def _prepare_audio_run(model: HFWhisperLensModel, audio_path: Path):
    """Decode audio, run baseline generation, and prepare captured inputs."""
    decoded = decode_audio_bytes(audio_path.read_bytes())
    features = model.processor.feature_extractor(
        decoded.waveform,
        sampling_rate=decoded.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = features.input_features.to(model.input_device)
    attention_mask = features.attention_mask.to(model.input_device)
    generated = model.generate(
        input_features,
        attention_mask=attention_mask,
        return_dict_in_generate=True,
    )
    sequence_ids = generated["sequences"].cpu()
    special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
    has_ordinary_target = any(
        int(token_id) not in special_ids for token_id in sequence_ids[0, 1:]
    )
    inputs = model.prepare_audio(
        decoded.waveform,
        sampling_rate=decoded.sampling_rate,
        sequence_ids=sequence_ids,
        include_eos_target=not has_ordinary_target,
        duration_seconds=decoded.duration_seconds,
    )
    return decoded, input_features, attention_mask, sequence_ids, inputs


def _validate_causal_lens(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    *,
    stream: str,
) -> None:
    lens.validate_model(model)
    if getattr(lens, stream) is None:
        raise ValueError(f"causal {stream} steering requires a {stream} lens")
    preprocessing_versions = {
        lens_stream.metadata.get("audio_preprocessing_version")
        for lens_stream in (lens.encoder, lens.decoder)
        if lens_stream is not None
        and lens_stream.metadata.get("audio_preprocessing_version") is not None
    }
    if preprocessing_versions and preprocessing_versions != {
        AUDIO_PREPROCESSING_VERSION
    }:
        raise ValueError(
            "lens audio preprocessing is incompatible with this causal runner: "
            f"lens={sorted(preprocessing_versions)!r}, "
            f"runner={AUDIO_PREPROCESSING_VERSION!r}"
        )


@torch.no_grad()
def _steered_generation(
    model: HFWhisperLensModel,
    *,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    schedule: EncoderInterventionSchedule,
    deltas: dict[int, torch.Tensor],
) -> torch.Tensor:
    with ExitStack() as stack:
        # Multiple time segments at the same layer are already summed in
        # ``deltas``. Register that combined layer delta exactly once; applying
        # it once per segment would multiply the requested intervention during
        # free generation while candidate scoring applies it only once.
        for layer in sorted(deltas):
            stack.enter_context(
                ResidualAdder(
                    model.encoder_layers,
                    layer=layer,
                    delta=deltas[layer],
                )
            )
        return model.generate(input_features, attention_mask=attention_mask)


@torch.no_grad()
def _steered_decoder_generation(
    model: HFWhisperLensModel,
    *,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    vectors: dict[int, dict[int, torch.Tensor]],
) -> torch.Tensor:
    """Generate with an absolute-position, open-loop decoder edit schedule."""
    with DecoderResidualScheduleAdder(
        model.decoder,
        model.decoder_layers,
        vectors_by_layer=vectors,
    ):
        return model.generate(input_features, attention_mask=attention_mask)


def run_causal_trace(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    *,
    audio_path: Path,
    positive_text: str,
    negative_text: str,
    layers: list[int],
    start_seconds: float | None,
    end_seconds: float | None,
    strength: float,
    random_control_seed: int | None = None,
    comparison_texts: list[str] | None = None,
    segments: list[tuple[float, float, str]] | None = None,
    prefix_segments: list[tuple[float, float, str]] | None = None,
    negative_prefix: str | None = None,
) -> dict[str, Any]:
    """Run and serialize a J-lens-proposed encoder intervention on one clip."""
    _validate_causal_lens(model, lens, stream="encoder")

    decoded, input_features, attention_mask, sequence_ids, inputs = (
        _prepare_audio_run(model, audio_path)
    )
    valid_positions = int(inputs.encoder_position_mask[0].sum())
    if segments is not None and prefix_segments is not None:
        raise ValueError("pass exact segments or prefix-family segments, not both")
    if (segments is not None or prefix_segments is not None) and (
        start_seconds is not None or end_seconds is not None
    ):
        raise ValueError("pass a selected span or explicit segments, not both")
    if negative_prefix is not None and prefix_segments is None:
        raise ValueError("--negative-prefix requires --prefix-segment")
    segment_strategy = "explicit"
    segment_specs: list[dict[str, Any]] = []
    if prefix_segments is not None:
        segment_strategy = "vocabulary_prefix_family"
        for segment_start, segment_end, prefix in prefix_segments:
            prefix = prefix.lstrip()
            token_ids = vocabulary_token_ids_starting_with(model, prefix)
            segment_specs.append(
                {
                    "start_seconds": segment_start,
                    "end_seconds": segment_end,
                    "text": f"{prefix}*",
                    "prefix": prefix,
                    "token_ids": token_ids,
                    "target_kind": "decoded_prefix_family",
                }
            )
    elif segments is None:
        if start_seconds is None or end_seconds is None:
            raise ValueError("start_seconds and end_seconds are required without segments")
        segments = _tokenizer_faithful_segments(
            model,
            positive_text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        segment_strategy = "tokenizer_faithful_default"
    if segments is not None:
        segment_specs.extend(
            {
                "start_seconds": segment_start,
                "end_seconds": segment_end,
                "text": segment_text,
                "token_ids": candidate_text_token_ids(model, segment_text),
                "target_kind": "exact_tokenized_text",
            }
            for segment_start, segment_end, segment_text in segments
        )
    resolved_segments = []
    for segment in segment_specs:
        start_position, end_position = _span_from_seconds(
            start_seconds=segment["start_seconds"],
            end_seconds=segment["end_seconds"],
            valid_positions=valid_positions,
        )
        token_examples = [
            model.tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
            for token_id in segment["token_ids"][:12]
        ]
        resolved_segments.append(
            {
                **segment,
                "start_position": start_position,
                "end_position": end_position,
                "token_examples": token_examples,
            }
        )
    if not resolved_segments:
        raise ValueError("at least one encoder segment is required")
    normalized_negative_prefix = (
        None if negative_prefix is None else negative_prefix.lstrip()
    )
    negative_token_ids = (
        candidate_text_token_ids(model, negative_text)
        if normalized_negative_prefix is None
        else vocabulary_token_ids_starting_with(model, normalized_negative_prefix)
    )
    negative_token_set = set(negative_token_ids)
    for segment in resolved_segments:
        positive_token_set = set(segment["token_ids"])
        if (
            segment_strategy == "vocabulary_prefix_family"
            and positive_token_set == negative_token_set
        ):
            raise ValueError("positive and negative vocabulary families must differ")
        overlap_ids = (
            sorted(positive_token_set & negative_token_set)
            if segment_strategy == "vocabulary_prefix_family"
            else []
        )
        segment["negative_overlap_token_ids"] = overlap_ids
    default_comparisons = [" Lily!", " Yay!", positive_text, negative_text]
    comparison_texts = list(
        dict.fromkeys(comparison_texts or default_comparisons)
    )
    if positive_text not in comparison_texts or negative_text not in comparison_texts:
        raise ValueError("comparison texts must include the positive and negative targets")
    if not layers:
        raise ValueError("at least one encoder layer is required")
    per_layer_strength = strength / math.sqrt(len(layers) * len(resolved_segments))
    schedule = EncoderInterventionSchedule(
        tuple(
            EncoderIntervention(
                layer=layer,
                start_position=segment["start_position"],
                end_position=segment["end_position"],
                direction=encoder_lens_contrast_direction(
                    model,
                    lens,
                    layer=layer,
                    positive_token_ids=segment["token_ids"],
                    negative_token_ids=negative_token_ids,
                ),
                strength=per_layer_strength,
            )
            for layer in layers
            for segment in resolved_segments
        )
    )
    trace = run_encoder_intervention_schedule(model, inputs, schedule)
    first_prediction_position = int(
        inputs.decoder_position_mask[0].nonzero(as_tuple=True)[0][0]
    )
    baseline_candidates = _candidate_outcomes(
        model,
        inputs,
        positive_text=positive_text,
        negative_text=negative_text,
        comparison_texts=comparison_texts,
        logits=trace.baseline_logits,
        first_prediction_position=first_prediction_position,
    )
    steered_candidates = _candidate_outcomes(
        model,
        inputs,
        positive_text=positive_text,
        negative_text=negative_text,
        comparison_texts=comparison_texts,
        logits=trace.steered_logits,
        first_prediction_position=first_prediction_position,
        schedule=schedule,
        deltas=trace.deltas,
    )
    steered_sequence = _steered_generation(
        model,
        input_features=input_features,
        attention_mask=attention_mask,
        schedule=schedule,
        deltas=trace.deltas,
    )

    random_control = None
    if random_control_seed is not None:
        control_schedule = EncoderInterventionSchedule(
            tuple(
                EncoderIntervention(
                    layer=layer,
                    start_position=segment["start_position"],
                    end_position=segment["end_position"],
                    direction=random_encoder_direction(
                        model.encoder_dim,
                        seed=random_control_seed + layer * 101 + segment_index,
                        device=model.input_device,
                    ),
                    strength=per_layer_strength,
                )
                for layer in layers
                for segment_index, segment in enumerate(resolved_segments)
            )
        )
        control_trace = run_encoder_intervention_schedule(
            model, inputs, control_schedule
        )
        control_candidates = _candidate_outcomes(
            model,
            inputs,
            positive_text=positive_text,
            negative_text=negative_text,
            comparison_texts=comparison_texts,
            logits=control_trace.steered_logits,
            first_prediction_position=first_prediction_position,
            schedule=control_schedule,
            deltas=control_trace.deltas,
        )
        control_sequence = _steered_generation(
            model,
            input_features=input_features,
            attention_mask=attention_mask,
            schedule=control_schedule,
            deltas=control_trace.deltas,
        )
        random_control = {
            "seed": random_control_seed,
            "intervention_delta_l2_norm_by_layer": {
                str(layer): float(delta.float().norm().cpu())
                for layer, delta in control_trace.deltas.items()
            },
            "generated_transcript": model.tokenizer.decode(
                control_sequence[0].cpu(), skip_special_tokens=True
            ),
            "candidates": control_candidates,
        }

    return {
        "format": "audio-jlens-causal-trace-v1",
        "stream": "encoder",
        "audio": {
            "filename": audio_path.name,
            "duration_seconds": decoded.duration_seconds,
            "sampling_rate": decoded.sampling_rate,
        },
        "model_id": model.model_id,
        "generated_transcript": {
            "baseline": model.tokenizer.decode(
                sequence_ids[0], skip_special_tokens=True
            ),
            "steered": model.tokenizer.decode(
                steered_sequence[0].cpu(), skip_special_tokens=True
            ),
        },
        "intervention": {
            "method": (
                "encoder_jlens_vocabulary_prefix_mean_contrast"
                if segment_strategy == "vocabulary_prefix_family"
                else "encoder_jlens_candidate_token_mean_contrast"
            ),
            "segment_strategy": segment_strategy,
            "prefix_matching": (
                None
                if segment_strategy != "vocabulary_prefix_family"
                else {
                    "case_sensitive": True,
                    "strip_leading_whitespace": True,
                    "negative_prefix": normalized_negative_prefix,
                    "negative_token_count": len(negative_token_ids),
                }
            ),
            "layers": layers,
            "total_strength": strength,
            "per_layer_strength": per_layer_strength,
            "segments": [
                {
                    "text": segment["text"],
                    "target_kind": segment["target_kind"],
                    "prefix": segment.get("prefix"),
                    "token_ids": segment["token_ids"],
                    "token_count": len(segment["token_ids"]),
                    "token_examples": segment["token_examples"],
                    "negative_overlap_token_count": len(
                        segment["negative_overlap_token_ids"]
                    ),
                    "start_position": segment["start_position"],
                    "end_position": segment["end_position"],
                    "start_seconds": segment["start_position"] * 0.02,
                    "end_seconds": min(
                        decoded.duration_seconds, segment["end_position"] * 0.02
                    ),
                }
                for segment in resolved_segments
            ],
            "delta_l2_norm_by_layer": {
                str(layer): float(delta.float().norm().cpu())
                for layer, delta in trace.deltas.items()
            },
        },
        "candidates": {
            "baseline": baseline_candidates,
            "steered": steered_candidates,
        },
        "random_direction_control": random_control,
        "propagation": {
            "encoder_l2_change_by_layer": {
                str(layer_index): values[0].float().cpu().tolist()
                for layer_index, values in trace.encoder_change_norms().items()
            },
            "decoder_l2_change_by_layer": {
                str(layer_index): values[0].float().cpu().tolist()
                for layer_index, values in trace.decoder_change_norms().items()
            },
            "output_logit_l2_change_by_decoder_position": (
                (trace.steered_logits - trace.baseline_logits)
                .float()
                .norm(dim=-1)[0]
                .cpu()
                .tolist()
            ),
        },
        "warnings": [
            "This is a causal intervention in Whisper, not evidence of a human perceptual mechanism.",
            "The J-lens direction is a corpus-averaged proposed edit; evaluate matched-norm random directions before making functional claims.",
            "Candidate scores are teacher-forced full-text continuation scores, not calibrated probabilities or generation-time scores.",
        ]
        + (
            [
                "Vocabulary-prefix families are decoded-token groups, not phoneme classes; family size and tokenizer composition affect the averaged direction."
            ]
            if segment_strategy == "vocabulary_prefix_family"
            else []
        )
        + (
            [
                "Positive and negative vocabulary-prefix families overlap; shared tokens contribute to both equal-weight means."
            ]
            if segment_strategy == "vocabulary_prefix_family" and any(
                segment["negative_overlap_token_ids"]
                for segment in resolved_segments
            )
            else []
        ),
    }


def run_decoder_causal_trace(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    *,
    audio_path: Path,
    positive_text: str,
    negative_text: str,
    layers: list[int],
    strength: float,
    piece_indices: list[int] | None = None,
    random_control_seed: int | None = None,
    comparison_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Run and serialize a decoder-residual J-lens intervention."""
    _validate_causal_lens(model, lens, stream="decoder")
    assert lens.decoder is not None
    decoded, input_features, attention_mask, sequence_ids, inputs = (
        _prepare_audio_run(model, audio_path)
    )
    positive_inputs, positive_token_ids_tuple, first_prediction_position = (
        prepare_candidate_inputs(model, inputs, positive_text)
    )
    positive_token_ids = list(positive_token_ids_tuple)
    if piece_indices is None:
        piece_indices = list(range(len(positive_token_ids)))
    if not piece_indices:
        raise ValueError("at least one decoder piece index is required")
    if piece_indices != sorted(set(piece_indices)):
        raise ValueError("decoder piece indices must be unique and ordered")
    if piece_indices[0] < 0 or piece_indices[-1] >= len(positive_token_ids):
        raise ValueError(
            f"decoder piece index is outside the {len(positive_token_ids)}-piece target"
        )
    if not layers:
        raise ValueError("at least one decoder layer is required")
    missing_layers = sorted(set(layers) - set(lens.decoder.source_layers))
    if missing_layers:
        raise ValueError(
            f"decoder lens has no source layers {missing_layers}; "
            f"available={lens.decoder.source_layers}"
        )

    negative_token_ids = candidate_text_token_ids(model, negative_text)
    per_edit_strength = strength / math.sqrt(len(layers) * len(piece_indices))
    schedule = DecoderInterventionSchedule(
        tuple(
            DecoderIntervention(
                layer=layer,
                position=first_prediction_position + piece_index,
                direction=decoder_lens_contrast_direction(
                    model,
                    lens,
                    layer=layer,
                    positive_token_ids=[positive_token_ids[piece_index]],
                    negative_token_ids=negative_token_ids,
                ),
                strength=per_edit_strength,
            )
            for layer in layers
            for piece_index in piece_indices
        )
    )
    trace = run_decoder_intervention_schedule(model, positive_inputs, schedule)
    default_comparisons = [" Lily!", " Yay!", positive_text, negative_text]
    comparison_texts = list(dict.fromkeys(comparison_texts or default_comparisons))
    if positive_text not in comparison_texts or negative_text not in comparison_texts:
        raise ValueError("comparison texts must include the positive and negative targets")

    baseline_candidates = _candidate_outcomes(
        model,
        inputs,
        positive_text=positive_text,
        negative_text=negative_text,
        comparison_texts=comparison_texts,
        logits=trace.baseline_logits,
        first_prediction_position=first_prediction_position,
    )
    steered_candidates = _candidate_outcomes(
        model,
        inputs,
        positive_text=positive_text,
        negative_text=negative_text,
        comparison_texts=comparison_texts,
        logits=trace.steered_logits,
        first_prediction_position=first_prediction_position,
        decoder_vectors=trace.vectors,
    )
    steered_sequence = _steered_decoder_generation(
        model,
        input_features=input_features,
        attention_mask=attention_mask,
        vectors=trace.vectors,
    )

    random_control = None
    if random_control_seed is not None:
        control_schedule = DecoderInterventionSchedule(
            tuple(
                DecoderIntervention(
                    layer=layer,
                    position=first_prediction_position + piece_index,
                    direction=random_decoder_direction(
                        model.decoder_dim,
                        seed=random_control_seed + layer * 101 + piece_index,
                        device=model.input_device,
                    ),
                    strength=per_edit_strength,
                )
                for layer in layers
                for piece_index in piece_indices
            )
        )
        control_trace = run_decoder_intervention_schedule(
            model, positive_inputs, control_schedule
        )
        control_candidates = _candidate_outcomes(
            model,
            inputs,
            positive_text=positive_text,
            negative_text=negative_text,
            comparison_texts=comparison_texts,
            logits=control_trace.steered_logits,
            first_prediction_position=first_prediction_position,
            decoder_vectors=control_trace.vectors,
        )
        control_sequence = _steered_decoder_generation(
            model,
            input_features=input_features,
            attention_mask=attention_mask,
            vectors=control_trace.vectors,
        )
        random_control = {
            "seed": random_control_seed,
            "intervention_vector_l2_norm_by_layer_position": {
                str(layer): {
                    str(position): float(vector.float().norm().cpu())
                    for position, vector in layer_vectors.items()
                }
                for layer, layer_vectors in control_trace.vectors.items()
            },
            "generated_transcript": model.tokenizer.decode(
                control_sequence[0].cpu(), skip_special_tokens=True
            ),
            "candidates": control_candidates,
        }

    pieces = [
        {
            "piece_index": piece_index,
            "text": model.tokenizer.decode(
                [positive_token_ids[piece_index]],
                clean_up_tokenization_spaces=False,
            ),
            "token_id": positive_token_ids[piece_index],
            "prediction_position": first_prediction_position + piece_index,
        }
        for piece_index in piece_indices
    ]
    return {
        "format": "audio-jlens-causal-trace-v1",
        "stream": "decoder",
        "audio": {
            "filename": audio_path.name,
            "duration_seconds": decoded.duration_seconds,
            "sampling_rate": decoded.sampling_rate,
        },
        "model_id": model.model_id,
        "generated_transcript": {
            "baseline": model.tokenizer.decode(
                sequence_ids[0], skip_special_tokens=True
            ),
            "steered": model.tokenizer.decode(
                steered_sequence[0].cpu(), skip_special_tokens=True
            ),
            "policy": "open_loop_absolute_prediction_positions",
        },
        "intervention": {
            "method": "decoder_jlens_candidate_token_mean_contrast",
            "reference": "teacher_forced_positive_path",
            "layers": layers,
            "total_strength": strength,
            "per_edit_strength": per_edit_strength,
            "positive_token_ids": positive_token_ids,
            "selected_piece_indices": piece_indices,
            "pieces": pieces,
            "vector_l2_norm_by_layer_position": {
                str(layer): {
                    str(position): float(vector.float().norm().cpu())
                    for position, vector in layer_vectors.items()
                }
                for layer, layer_vectors in trace.vectors.items()
            },
        },
        "candidates": {
            "baseline": baseline_candidates,
            "steered": steered_candidates,
        },
        "random_direction_control": random_control,
        "propagation": {
            "decoder_l2_change_by_layer": {
                str(layer_index): values[0].float().cpu().tolist()
                for layer_index, values in trace.decoder_change_norms().items()
            },
            "output_logit_l2_change_by_decoder_position": (
                (trace.steered_logits - trace.baseline_logits)
                .float()
                .norm(dim=-1)[0]
                .cpu()
                .tolist()
            ),
        },
        "warnings": [
            "Decoder J-lens source layers are L0-L2; L3 is the downstream target layer, not an edited source layer.",
            "Candidate scores use the positive candidate as a teacher-forced norm reference and are not calibrated probabilities.",
            "Free generation uses an open-loop absolute-position schedule: a later piece edit still runs if an earlier generated piece differs.",
            "Encoder and decoder percentage budgets are normalized within different residual streams and are not identical causal doses.",
            "One matched random direction is exploratory, not a control distribution.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Whisper encoder- or decoder-residual causal trace"
    )
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--revision", help="Hugging Face model revision or commit")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument(
        "--stream",
        choices=("encoder", "decoder"),
        default="encoder",
        help="residual stream to intervene on (default: encoder)",
    )
    parser.add_argument(
        "--layers",
        help=(
            "ordered source layers; defaults to encoder L1 or every available "
            "decoder J-lens source layer"
        ),
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        help="selected audio span start; without --segment, split across positive BPE pieces",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        help="selected audio span end; without --segment, split across positive BPE pieces",
    )
    parser.add_argument(
        "--segment",
        action="append",
        help=(
            "explicit start:end:text target-piece override; without this flag, "
            "the selected span is split across the positive text's actual BPE pieces"
        ),
    )
    parser.add_argument(
        "--prefix-segment",
        action="append",
        help=(
            "encoder-only start:end:prefix family; matches every ordinary "
            "decoded token after stripping leading whitespace from both sides"
        ),
    )
    parser.add_argument(
        "--negative-prefix",
        help=(
            "encoder-only decoded-token prefix family for the contrast side; "
            "case-sensitive after stripping leading whitespace"
        ),
    )
    parser.add_argument(
        "--piece-index",
        action="append",
        type=int,
        help=(
            "zero-based positive-target BPE piece to edit in decoder mode; "
            "repeatable and defaults to all pieces"
        ),
    )
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument(
        "--random-control-seed",
        type=int,
        help="run a matched-norm random-direction control with this seed",
    )
    parser.add_argument("--positive", default=" Yanny")
    parser.add_argument("--negative", default=" Laurel")
    parser.add_argument(
        "--comparison-candidate",
        action="append",
        help="candidate text to include in the restricted comparison set; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.audio.is_file():
        raise ValueError(f"audio file not found: {args.audio}")
    segments = _parse_segments(args.segment)
    prefix_segments = _parse_segments(args.prefix_segment)
    if args.stream == "encoder":
        if segments and prefix_segments:
            raise ValueError("pass --segment or --prefix-segment, not both")
        if (segments or prefix_segments) and (
            args.start_seconds is not None or args.end_seconds is not None
        ):
            raise ValueError("pass seconds or explicit segments, not both")
        if args.negative_prefix is not None and not prefix_segments:
            raise ValueError("--negative-prefix requires --prefix-segment")
        if not segments and not prefix_segments and (
            args.start_seconds is None or args.end_seconds is None
        ):
            raise ValueError(
                "encoder mode needs seconds, --segment, or --prefix-segment"
            )
        if args.piece_index is not None:
            raise ValueError("--piece-index is only valid in decoder mode")
    elif (
        segments
        or prefix_segments
        or args.negative_prefix is not None
        or args.start_seconds is not None
        or args.end_seconds is not None
    ):
        raise ValueError(
            "decoder mode uses token positions; do not pass encoder segment options"
        )
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision
    )
    device = _device(args.device)
    hf_model.to(device)
    model = HFWhisperLensModel(hf_model, processor, model_id=args.model)
    lens = WhisperJacobianLens.load(args.lens)
    if args.layers is None:
        if args.stream == "encoder":
            layers = [1]
        else:
            if lens.decoder is None:
                raise ValueError("decoder mode requires a decoder lens")
            layers = lens.decoder.source_layers
    else:
        layers = _parse_layers(args.layers)
    if args.stream == "encoder":
        result = run_causal_trace(
            model,
            lens,
            audio_path=args.audio,
            positive_text=args.positive,
            negative_text=args.negative,
            layers=layers,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            strength=args.strength,
            random_control_seed=args.random_control_seed,
            comparison_texts=args.comparison_candidate,
            segments=segments or None,
            prefix_segments=prefix_segments or None,
            negative_prefix=args.negative_prefix,
        )
    else:
        result = run_decoder_causal_trace(
            model,
            lens,
            audio_path=args.audio,
            positive_text=args.positive,
            negative_text=args.negative,
            layers=layers,
            strength=args.strength,
            piece_indices=args.piece_index,
            random_control_seed=args.random_control_seed,
            comparison_texts=args.comparison_candidate,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
