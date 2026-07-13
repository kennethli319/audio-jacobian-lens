# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Turn a held-out Whisper run into the localhost explorer's JSON schema."""

from __future__ import annotations

import base64
import io
import math
import wave
from typing import Any

import numpy as np
import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.phonetic_signatures import PhoneSignaturePrototypes
from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_lens import (
    LensTopK,
    WhisperJacobianLens,
    lens_topk,
    lens_topk_grouped,
    pool_sequence_residuals,
)


def waveform_envelope(waveform: np.ndarray, *, n_bins: int = 400) -> list[float]:
    """Return a compact absolute-peak envelope for the browser canvas."""
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return []
    n_bins = max(1, min(n_bins, mono.size))
    edges = np.linspace(0, mono.size, n_bins + 1, dtype=np.int64)
    envelope = [
        float(np.max(np.abs(mono[edges[i] : max(edges[i] + 1, edges[i + 1])])))
        for i in range(n_bins)
    ]
    maximum = max(envelope, default=0.0)
    if maximum > 0:
        envelope = [value / maximum for value in envelope]
    return envelope


def waveform_wav_data_url(waveform: np.ndarray, *, sampling_rate: int = 16_000) -> str:
    """Encode the exact mono model-input waveform for synchronized playback."""
    samples = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(samples * 32767.0).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sampling_rate)
        output.writeframes(pcm.tobytes())
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


_DISPLAY_TOKEN_MASKS: dict[tuple[int, int], torch.Tensor] = {}
_DISPLAY_TOKEN_LENGTHS: dict[tuple[int, int], torch.Tensor] = {}
_DECODER_TOKEN_LENGTH_FILTER_LAYERS = frozenset({0, 1})
_RANK_TIE_POLICY = "1_plus_count_strictly_greater"
_LEXICAL_RANK_SPACE = "lexical_display_vocabulary"
_LENGTH_BUCKET_RANK_SPACE = "exact_decoded_character_length_bucket"
_MAX_LENGTH_RANK_SPACE = "maximum_decoded_character_length_vocabulary"
_FULL_VOCABULARY_RANK_SPACE = "full_model_vocabulary"


def display_token_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Vocabulary mask that hides Whisper control/timestamp/empty tokens."""
    key = (id(tokenizer), vocab_size)
    if key in _DISPLAY_TOKEN_MASKS:
        return _DISPLAY_TOKEN_MASKS[key]
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        stripped = text.strip()
        if not stripped or "<|" in stripped or "|>" in stripped:
            continue
        if stripped == "�":
            continue
        # The raw vocabulary often ranks quotation marks and separator-only
        # fragments highly in early layers. Keep lexical fragments (in any
        # script) for the default human-readable grid; actual model
        # probabilities still use the full unfiltered vocabulary.
        if not any(character.isalnum() for character in stripped):
            continue
        mask[token_id] = True
    _DISPLAY_TOKEN_MASKS[key] = mask
    return mask


def display_token_lengths(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Decoded character count for every lexical display token."""
    key = (id(tokenizer), vocab_size)
    if key in _DISPLAY_TOKEN_LENGTHS:
        return _DISPLAY_TOKEN_LENGTHS[key]
    lexical_mask = display_token_mask(tokenizer, vocab_size)
    lengths = torch.zeros(vocab_size, dtype=torch.long)
    for token_id in lexical_mask.nonzero(as_tuple=True)[0].tolist():
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()
        lengths[token_id] = len(text)
    _DISPLAY_TOKEN_LENGTHS[key] = lengths
    return lengths


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _top_tokens(
    result: LensTopK,
    tokenizer: Any,
    *,
    score_kind: str = "raw_readout_logit",
    token_lengths: torch.Tensor,
    rank_space: str,
    character_length_constraint: int | None = None,
) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for (
        position_ids,
        position_scores,
        position_ranks,
        display_ranks,
        full_ranks,
    ) in zip(
        result.token_ids,
        result.scores,
        result.ranks,
        result.display_vocabulary_ranks,
        result.full_vocabulary_ranks,
        strict=True,
    ):
        output.append(
            [
                {
                    "id": int(token_id),
                    "text": _decode_token(tokenizer, int(token_id)),
                    "score": float(score),
                    "rank": int(rank),
                    "rank_denominator": result.rank_denominator,
                    "rank_space": rank_space,
                    "display_vocabulary_rank": int(display_rank),
                    "display_vocabulary_denominator": (
                        result.display_vocabulary_denominator
                    ),
                    "full_vocabulary_rank": int(full_rank),
                    "full_vocabulary_denominator": (
                        result.full_vocabulary_denominator
                    ),
                    "rank_tie_policy": _RANK_TIE_POLICY,
                    "score_kind": score_kind,
                    "vocabulary_filter": {
                        "display_lexical_filter_applied": True,
                        "character_length_filter_applied": (
                            character_length_constraint is not None
                        ),
                        "decoded_character_length": int(token_lengths[token_id]),
                        "character_length_constraint": (
                            None
                            if character_length_constraint is None
                            else {
                                "operator": "exact",
                                "value": character_length_constraint,
                            }
                        ),
                    },
                }
                for token_id, score, rank, display_rank, full_rank in zip(
                    position_ids.tolist(),
                    position_scores.tolist(),
                    position_ranks.tolist(),
                    display_ranks.tolist(),
                    full_ranks.tolist(),
                    strict=True,
                )
            ]
        )
    return output


def _realized_token_payloads(
    result: LensTopK,
    tokenizer: Any,
    *,
    score_kind: str,
    token_lengths: torch.Tensor | None,
) -> list[dict[str, Any]]:
    """Serialize exact selected-token readouts computed from full logits."""
    selected = result.selected_readouts
    if selected is None:
        return []
    payloads: list[dict[str, Any]] = []
    for token_id, score, display_rank, display_eligible, full_rank in zip(
        selected.token_ids.tolist(),
        selected.scores.tolist(),
        selected.display_vocabulary_ranks.tolist(),
        selected.display_vocabulary_eligible.tolist(),
        selected.full_vocabulary_ranks.tolist(),
        strict=True,
    ):
        eligible = bool(display_eligible)
        primary_rank = int(display_rank) if eligible else int(full_rank)
        primary_denominator = (
            selected.display_vocabulary_denominator
            if eligible
            else selected.full_vocabulary_denominator
        )
        primary_space = _LEXICAL_RANK_SPACE if eligible else _FULL_VOCABULARY_RANK_SPACE
        text = _decode_token(tokenizer, int(token_id))
        payloads.append(
            {
                "id": int(token_id),
                "text": text,
                "score": float(score),
                "rank": primary_rank,
                "rank_denominator": primary_denominator,
                "rank_space": primary_space,
                "display_vocabulary_rank": (int(display_rank) if eligible else None),
                "display_vocabulary_denominator": (
                    selected.display_vocabulary_denominator
                ),
                "full_vocabulary_rank": int(full_rank),
                "full_vocabulary_denominator": (selected.full_vocabulary_denominator),
                "rank_tie_policy": _RANK_TIE_POLICY,
                "score_kind": score_kind,
                "vocabulary_filter": {
                    "display_lexical_filter_applied": eligible,
                    "display_lexical_eligible": eligible,
                    "character_length_filter_applied": False,
                    "decoded_character_length": (
                        int(token_lengths[token_id])
                        if eligible and token_lengths is not None
                        else len(text.strip())
                    ),
                    "character_length_constraint": None,
                },
            }
        )
    return payloads


def _realized_ranks_by_max_length(
    result: LensTopK,
    *,
    token_lengths: torch.Tensor,
) -> list[dict[str, int | None]]:
    """Return exact selected-token ranks for every cumulative length filter.

    A null value means the selected token is outside that filtered vocabulary,
    not that its score or rank was unavailable.
    """
    selected = result.selected_readouts
    if (
        selected is None
        or selected.group_strictly_greater_counts is None
        or selected.group_denominators is None
    ):
        return []
    positions = int(selected.token_ids.shape[0])
    greater_counts = torch.zeros(positions, dtype=torch.long)
    output: list[dict[str, int | None]] = [{} for _ in range(positions)]
    maximum_length = int(token_lengths.max())
    for maximum in range(1, maximum_length + 1):
        group_counts = selected.group_strictly_greater_counts.get(maximum)
        if group_counts is not None:
            greater_counts += group_counts.to(dtype=torch.long)
        for position, token_id_tensor in enumerate(selected.token_ids):
            token_id = int(token_id_tensor)
            eligible = bool(selected.display_vocabulary_eligible[position])
            decoded_length = int(token_lengths[token_id]) if eligible else 0
            if not eligible or decoded_length > maximum:
                output[position][str(maximum)] = None
                continue
            output[position][str(maximum)] = int(greater_counts[position]) + 1
    return output


def _cells_for_layer(
    model: HFWhisperLensModel,
    lens: CrossJacobianLens,
    residuals: torch.Tensor,
    *,
    layer: int,
    top_k: int,
    token_mask: torch.Tensor,
    token_lengths: torch.Tensor | None = None,
    realized_token_ids: torch.Tensor | None = None,
    subtract_target_baseline: bool = False,
    score_kind: str = "raw_readout_logit",
) -> list[dict[str, Any]]:
    grouped_top = None
    if token_lengths is not None:
        length_masks = {
            int(length): token_lengths == length
            for length in torch.unique(token_lengths[token_lengths > 0]).tolist()
        }
        grouped_top = lens_topk_grouped(
            model,
            lens,
            residuals,
            layer=layer,
            top_k=top_k,
            token_mask=token_mask,
            token_groups=length_masks,
            selected_token_ids=realized_token_ids,
            subtract_target_baseline=subtract_target_baseline,
        )
        top = grouped_top.overall
    else:
        top = lens_topk(
            model,
            lens,
            residuals,
            layer=layer,
            top_k=top_k,
            token_mask=token_mask,
            selected_token_ids=realized_token_ids,
            subtract_target_baseline=subtract_target_baseline,
        )
    if token_lengths is None:
        token_lengths = display_token_lengths(model.tokenizer, model.vocab_size)
    decoded = _top_tokens(
        top,
        model.tokenizer,
        score_kind=score_kind,
        token_lengths=token_lengths,
        rank_space=_LEXICAL_RANK_SPACE,
    )
    cells = [
        {
            "selected_score": tokens[0]["score"],
            "top_tokens": tokens,
        }
        for tokens in decoded
    ]
    if realized_token_ids is not None:
        realized = _realized_token_payloads(
            top,
            model.tokenizer,
            score_kind=score_kind,
            token_lengths=token_lengths,
        )
        if len(realized) != len(cells):
            raise ValueError("realized readouts and lens cells must align")
        for cell, realized_token in zip(cells, realized, strict=True):
            cell["realized_token"] = realized_token
        if grouped_top is not None:
            filtered_realized = _realized_ranks_by_max_length(
                top,
                token_lengths=token_lengths,
            )
            if len(filtered_realized) != len(cells):
                raise ValueError("filtered realized readouts and lens cells must align")
            for cell, by_maximum in zip(cells, filtered_realized, strict=True):
                cell["realized_rank_by_max_length"] = by_maximum
    if grouped_top is not None:
        decoded_groups = {
            str(length): _top_tokens(
                result,
                model.tokenizer,
                score_kind=score_kind,
                token_lengths=token_lengths,
                rank_space=_LENGTH_BUCKET_RANK_SPACE,
                character_length_constraint=length,
            )
            for length, result in grouped_top.groups.items()
        }
        for position, cell in enumerate(cells):
            cell["top_tokens_by_length"] = {
                length: tokens[position] for length, tokens in decoded_groups.items()
            }
    return cells


def _attach_cell_context(
    cells: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    *,
    timing_source: str,
    display_vocabulary_size: int,
) -> None:
    if len(cells) != len(contexts):
        raise ValueError("lens cells and position contexts must align")
    for position_index, (cell, context) in enumerate(
        zip(cells, contexts, strict=True)
    ):
        length_filter_available = "top_tokens_by_length" in cell
        cell["position_index"] = position_index
        cell["time_window"] = {
            "start_seconds": context.get("start_seconds"),
            "end_seconds": context.get("end_seconds"),
            "timing_source": timing_source,
        }
        cell["candidate_space"] = {
            "primary_rank_space": _LEXICAL_RANK_SPACE,
            "primary_rank_denominator": display_vocabulary_size,
            "display_lexical_filter_applied": True,
            "character_length_filter_available": length_filter_available,
            "character_length_filter_policy": (
                "exact_decoded_character_length_buckets"
                if length_filter_available
                else None
            ),
        }


def _head_candidate_provenance(
    *,
    token_id: int,
    text: str,
    probability: float,
    log_probability: float,
    rank: int,
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "id": token_id,
        "text": text,
        "probability": probability,
        "log_probability": log_probability,
        "rank": rank,
        "rank_denominator": vocab_size,
        "rank_space": _FULL_VOCABULARY_RANK_SPACE,
        "full_vocabulary_rank": rank,
        "full_vocabulary_denominator": vocab_size,
        "rank_tie_policy": _RANK_TIE_POLICY,
        "score_kind": "raw_teacher_forced_probability",
        "vocabulary_filter": {
            "display_lexical_filter_applied": False,
            "character_length_filter_applied": False,
            "decoded_character_length": len(text.strip()),
            "character_length_constraint": None,
        },
    }


def _next_later_timestamp(
    timestamps: list[float], start_index: int, start: float, duration: float
) -> float:
    for value in timestamps[start_index + 1 :]:
        if value > start + 1e-4:
            return min(duration, value)
    return duration


def _transcript_and_confidence(
    model: HFWhisperLensModel,
    inputs: WhisperLensInputs,
    actual_logits: torch.Tensor,
    *,
    token_timestamps: torch.Tensor | None,
    duration_seconds: float,
    top_k: int,
) -> tuple[dict[str, Any], list[int]]:
    mask = inputs.decoder_position_mask[0].cpu()
    positions = mask.nonzero(as_tuple=True)[0].tolist()
    targets = inputs.decoder_target_ids[0].cpu()
    logits = actual_logits[0].float().cpu()
    special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
    timing_source = (
        "unavailable" if token_timestamps is None else "whisper_cross_attention_dtw"
    )
    full_timestamps = (
        None
        if token_timestamps is None
        else token_timestamps.reshape(-1).float().cpu().tolist()
    )

    tokens: list[dict[str, Any]] = []
    for decoder_position in positions:
        target_id = int(targets[decoder_position])
        log_probs = logits[decoder_position].log_softmax(dim=-1)
        probabilities = log_probs.exp()
        entropy = float(-(probabilities * log_probs).sum())
        values, ids = probabilities.topk(top_k)
        candidate_ranks = [
            int((logits[decoder_position] > logits[decoder_position, token_id]).sum())
            + 1
            for token_id in ids.tolist()
        ]
        target_rank = (
            int((logits[decoder_position] > logits[decoder_position, target_id]).sum())
            + 1
        )
        full_sequence_index = decoder_position + 1
        if full_timestamps is None or full_sequence_index >= len(full_timestamps):
            start_seconds = None
            end_seconds = None
        else:
            start = full_timestamps[full_sequence_index]
            end = _next_later_timestamp(
                full_timestamps, full_sequence_index, start, duration_seconds
            )
            start_seconds = float(max(0.0, min(start, duration_seconds)))
            end_seconds = float(max(start_seconds, min(end, duration_seconds)))
        token_text = _decode_token(model.tokenizer, target_id)
        is_special = target_id in special_ids
        vocab_size = int(logits.shape[-1])
        target_provenance = _head_candidate_provenance(
            token_id=target_id,
            text=token_text,
            probability=float(probabilities[target_id]),
            log_probability=float(log_probs[target_id]),
            rank=target_rank,
            vocab_size=vocab_size,
        )
        tokens.append(
            {
                **target_provenance,
                "is_special": is_special,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "entropy": entropy,
                "top_tokens": [
                    _head_candidate_provenance(
                        token_id=int(token_id),
                        text=_decode_token(model.tokenizer, int(token_id)),
                        probability=float(value),
                        log_probability=float(log_probs[token_id]),
                        rank=rank,
                        vocab_size=vocab_size,
                    )
                    for token_id, value, rank in zip(
                        ids.tolist(),
                        values.tolist(),
                        candidate_ranks,
                        strict=True,
                    )
                ],
                "candidate_space": {
                    "primary_rank_space": _FULL_VOCABULARY_RANK_SPACE,
                    "primary_rank_denominator": vocab_size,
                    "display_lexical_filter_applied": False,
                    "character_length_filter_available": False,
                    "character_length_filter_policy": None,
                },
            }
        )
    text = "".join(token["text"] for token in tokens if not token["is_special"])
    if not text and tokens:
        text = "(no ordinary text generated)"
    return {
        "text": text,
        "tokens": tokens,
        "timing_source": timing_source,
        "timing_quality": (
            "model_derived" if full_timestamps is not None else "unavailable"
        ),
    }, positions


def _encoder_realized_token_alignment(
    time_bins: list[dict[str, Any]],
    transcript_tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map each encoder window to its greatest-overlap approximate output token.

    Ties prefer the token whose interval midpoint is closest to the encoder
    window midpoint, then the lower transcript position. The mapping uses
    Whisper's cross-attention/DTW timestamps and is therefore explicitly
    approximate rather than a word boundary or causal attribution.
    """
    timed_tokens: list[tuple[int, float, float]] = []
    for token_position, token in enumerate(transcript_tokens):
        start = token.get("start_seconds")
        end = token.get("end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if not math.isfinite(float(start)) or not math.isfinite(float(end)):
            continue
        if float(end) < float(start):
            continue
        timed_tokens.append((token_position, float(start), float(end)))
    if not timed_tokens:
        return []

    alignments: list[dict[str, Any]] = []
    for time_bin in time_bins:
        start = float(time_bin["start_seconds"])
        end = float(time_bin["end_seconds"])
        midpoint = (start + end) / 2
        scored_tokens = [
            (
                max(0.0, min(end, token_end) - max(start, token_start)),
                token_position,
                token_start,
                token_end,
            )
            for token_position, token_start, token_end in timed_tokens
        ]
        overlap, token_position, token_start, token_end = min(
            scored_tokens,
            key=lambda candidate: (
                -candidate[0],
                abs(midpoint - (candidate[2] + candidate[3]) / 2),
                candidate[1],
            ),
        )
        window_duration = max(0.0, end - start)
        alignments.append(
            {
                "token_position": token_position,
                "match": "overlapping" if overlap > 0 else "nearest",
                "window_midpoint_seconds": midpoint,
                "token_start_seconds": token_start,
                "token_end_seconds": token_end,
                "overlap_seconds": overlap,
                "overlap_fraction_of_window": (
                    overlap / window_duration if window_duration > 0 else 0.0
                ),
            }
        )
    return alignments


@torch.no_grad()
def analyze_whisper_run(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    inputs: WhisperLensInputs,
    waveform: np.ndarray,
    *,
    token_timestamps: torch.Tensor | None = None,
    top_k: int = 5,
    time_bin_seconds: float = 0.1,
    time_bin_overlap_seconds: float = 0.02,
    max_time_bins: int = 100,
    phone_signature_prototypes: PhoneSignaturePrototypes | None = None,
) -> dict[str, Any]:
    """Compute actual-output diagnostics and both lens grids for one clip."""
    lens.validate_model(model)
    if top_k <= 0 or top_k > model.vocab_size:
        raise ValueError(f"top_k must be in [1, {model.vocab_size}]")
    if time_bin_seconds <= 0:
        raise ValueError("time_bin_seconds must be positive")
    if time_bin_overlap_seconds < 0:
        raise ValueError("time_bin_overlap_seconds cannot be negative")
    if time_bin_overlap_seconds >= time_bin_seconds:
        raise ValueError(
            "time_bin_overlap_seconds must be smaller than time_bin_seconds"
        )
    if max_time_bins <= 0:
        raise ValueError("max_time_bins must be positive")
    duration = inputs.duration_seconds
    if duration is None:
        raise ValueError("inputs.duration_seconds is required for audio analysis")
    if phone_signature_prototypes is not None:
        if lens.encoder is None:
            raise ValueError("phone signatures require an encoder lens")
        phone_signature_prototypes.validate(
            model=model,
            encoder_lens=lens.encoder,
        )

    encoder_layers = [] if lens.encoder is None else lens.encoder.source_layers
    decoder_layers = [] if lens.decoder is None else lens.decoder.source_layers
    encoder_activations, decoder_activations, actual_logits = model.capture(
        inputs,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    transcription, decoder_positions = _transcript_and_confidence(
        model,
        inputs,
        actual_logits,
        token_timestamps=token_timestamps,
        duration_seconds=duration,
        top_k=top_k,
    )
    token_mask = display_token_mask(model.tokenizer, model.vocab_size)
    token_lengths = display_token_lengths(model.tokenizer, model.vocab_size)
    display_vocabulary_size = int(token_mask.sum())
    length_bucket_counts = {
        str(length): int((token_lengths == length).sum())
        for length in torch.unique(token_lengths[token_lengths > 0]).tolist()
    }
    cumulative_length_counts: dict[str, int] = {}
    running_count = 0
    maximum_token_length = int(token_lengths.max())
    for length in range(1, maximum_token_length + 1):
        running_count += length_bucket_counts.get(str(length), 0)
        cumulative_length_counts[str(length)] = running_count

    encoder_payload: dict[str, Any] = {
        "layers": encoder_layers,
        "time_bins": [],
        "cells": [],
        "score_kind": "target_mean_relative_logit_delta",
        "realized_token_alignment": {
            "method": "maximum_token_interval_overlap",
            "tie_break": "closest_interval_midpoint_then_lower_token_position",
            "timing_source": transcription["timing_source"],
            "timing_quality": transcription["timing_quality"],
            "interpretation": "approximate_non_causal_synchronization",
        },
    }
    if lens.encoder is not None:
        valid_positions = int(inputs.encoder_position_mask[0].sum())
        requested_positions = max(1, round(time_bin_seconds / 0.02))
        overlap_positions = max(0, round(time_bin_overlap_seconds / 0.02))
        if overlap_positions >= requested_positions:
            raise ValueError(
                "the requested encoder overlap rounds to at least one full window"
            )
        # Cover the entire valid span in at most max_time_bins. Short clips keep
        # the requested window exactly; long clips widen it while preserving the
        # requested overlap and reporting the effective geometry below.
        positions_per_bin = max(
            requested_positions,
            math.ceil(
                (valid_positions + (max_time_bins - 1) * overlap_positions)
                / max_time_bins
            ),
        )
        stride_positions = positions_per_bin - overlap_positions
        pooled_by_layer = {
            layer: pool_sequence_residuals(
                encoder_activations[layer][0],
                inputs.encoder_position_mask[0].to(encoder_activations[layer].device),
                positions_per_bin=positions_per_bin,
                stride_positions=stride_positions,
            )
            for layer in encoder_layers
        }
        boundaries = pooled_by_layer[encoder_layers[0]]
        encoder_payload["time_bins"] = [
            {
                "start_seconds": min(duration, start * 0.02),
                "end_seconds": min(duration, end * 0.02),
            }
            for start, end in zip(
                boundaries.start_positions, boundaries.end_positions, strict=True
            )
        ]
        encoder_payload["pooling"] = {
            "requested_window_seconds": time_bin_seconds,
            "requested_overlap_seconds": time_bin_overlap_seconds,
            "effective_window_seconds": positions_per_bin * 0.02,
            "effective_overlap_seconds": overlap_positions * 0.02,
            "effective_hop_seconds": stride_positions * 0.02,
            "adaptive_for_max_bins": positions_per_bin > requested_positions,
            "max_time_bins": max_time_bins,
        }
        encoder_realized_alignment = _encoder_realized_token_alignment(
            encoder_payload["time_bins"], transcription["tokens"]
        )
        encoder_realized_ids = (
            None
            if not encoder_realized_alignment
            else torch.tensor(
                [
                    transcription["tokens"][alignment["token_position"]]["id"]
                    for alignment in encoder_realized_alignment
                ],
                dtype=torch.long,
            )
        )
        encoder_payload["cells"] = [
            _cells_for_layer(
                model,
                lens.encoder,
                pooled_by_layer[layer].residuals,
                layer=layer,
                top_k=top_k,
                token_mask=token_mask,
                token_lengths=token_lengths,
                realized_token_ids=encoder_realized_ids,
                subtract_target_baseline=True,
                score_kind="target_mean_relative_logit_delta",
            )
            for layer in encoder_layers
        ]
        if phone_signature_prototypes is not None:
            for layer_index, layer in enumerate(encoder_layers):
                phone_rows = phone_signature_prototypes.score_layer(
                    model,
                    lens.encoder,
                    pooled_by_layer[layer].residuals,
                    layer=layer,
                    top_n=top_k,
                )
                layer_cells = encoder_payload["cells"][layer_index]
                if len(phone_rows) != len(layer_cells):
                    raise ValueError(
                        "phone-signature rows and encoder cells must align"
                    )
                for cell, candidates in zip(
                    layer_cells, phone_rows, strict=True
                ):
                    cell["phone_signatures"] = candidates
                    cell["phone_signature_usable"] = bool(candidates)
                    cell["phone_signature_margin"] = (
                        None
                        if len(candidates) < 2
                        else candidates[0]["similarity"]
                        - candidates[1]["similarity"]
                    )
        for layer_cells in encoder_payload["cells"]:
            if encoder_realized_alignment:
                for cell, alignment in zip(
                    layer_cells, encoder_realized_alignment, strict=True
                ):
                    cell["realized_token_position"] = alignment["token_position"]
                    cell["realized_token_alignment"] = {
                        key: value
                        for key, value in alignment.items()
                        if key != "token_position"
                    }
            _attach_cell_context(
                layer_cells,
                encoder_payload["time_bins"],
                timing_source="encoder_pooling_window",
                display_vocabulary_size=display_vocabulary_size,
            )

    decoder_payload: dict[str, Any] = {
        "layers": decoder_layers,
        "positions": [],
        "cells": [],
        "score_kind": "raw_readout_logit",
    }
    if lens.decoder is not None:
        decoder_payload["positions"] = [
            {
                "index": index,
                "token_id": token["id"],
                "text": token["text"],
                "start_seconds": token["start_seconds"],
                "end_seconds": token["end_seconds"],
            }
            for index, token in enumerate(transcription["tokens"])
        ]
        decoder_payload["cells"] = [
            _cells_for_layer(
                model,
                lens.decoder,
                decoder_activations[layer][0, decoder_positions],
                layer=layer,
                top_k=top_k,
                token_mask=token_mask,
                token_lengths=(
                    token_lengths
                    if layer in _DECODER_TOKEN_LENGTH_FILTER_LAYERS
                    else None
                ),
                realized_token_ids=inputs.decoder_target_ids[
                    0, decoder_positions
                ].detach().cpu(),
                score_kind="raw_readout_logit",
            )
            for layer in decoder_layers
        ]
        for layer_cells in decoder_payload["cells"]:
            _attach_cell_context(
                layer_cells,
                decoder_payload["positions"],
                timing_source=transcription["timing_source"],
                display_vocabulary_size=display_vocabulary_size,
            )

    counts: list[str] = []
    estimators: list[str] = []
    if lens.encoder is not None:
        counts.append(f"encoder {lens.encoder.n_examples}")
        estimators.append(
            f"encoder {lens.encoder.metadata.get('estimator_name', 'unknown')}"
        )
    if lens.decoder is not None:
        counts.append(f"decoder {lens.decoder.n_examples}")
        estimators.append(
            f"decoder {lens.decoder.metadata.get('estimator_name', 'unknown')}"
        )

    return {
        "audio": {
            "duration_seconds": duration,
            "waveform": waveform_envelope(waveform),
            "model_input_wav": waveform_wav_data_url(waveform),
            "model_input_format": "mono 16 kHz PCM",
        },
        "transcription": transcription,
        "encoder": encoder_payload,
        "decoder": decoder_payload,
        "metadata": {
            "model_id": model.model_id,
            "streams": [
                stream
                for stream, value in (
                    ("encoder", lens.encoder),
                    ("decoder", lens.decoder),
                )
                if value is not None
            ],
            "lens_examples": " · ".join(counts),
            "estimator": " · ".join(estimators),
            "display_vocabulary": {
                "policy": "alphanumeric_lexical_tokens",
                "full_vocabulary_size": model.vocab_size,
                "display_vocabulary_size": display_vocabulary_size,
                "exact_decoded_character_length_counts": length_bucket_counts,
                "maximum_decoded_character_length_counts": (
                    cumulative_length_counts
                ),
            },
            "candidate_rank_semantics": {
                "method": _RANK_TIE_POLICY,
                "ties": "equal scores share the same competition rank",
                "lens_primary_space": _LEXICAL_RANK_SPACE,
                "output_head_primary_space": _FULL_VOCABULARY_RANK_SPACE,
                "character_filter_merge": (
                    "merge disjoint exact-length buckets, sort by score, and "
                    "rank by strictly greater scores"
                ),
                "realized_maximum_length_space": _MAX_LENGTH_RANK_SPACE,
                "realized_maximum_length_denominator_source": (
                    "display_vocabulary.maximum_decoded_character_length_counts"
                ),
            },
            "encoder_token_length_filter": {
                "policy": "exact_decoded_character_length_buckets",
                "maximum_available_length": int(token_lengths.max()),
                "character_count_ignores_surrounding_whitespace": True,
            },
            "phone_signature": (
                {"available": False}
                if phone_signature_prototypes is None
                else {
                    **phone_signature_prototypes.public_metadata(),
                    "effective_display_window_seconds": encoder_payload.get(
                        "pooling", {}
                    ).get("effective_window_seconds"),
                    "effective_display_hop_seconds": encoder_payload.get(
                        "pooling", {}
                    ).get("effective_hop_seconds"),
                }
            ),
            "decoder_token_length_filter": {
                "policy": "exact_decoded_character_length_buckets",
                "eligible_source_layers": [
                    layer
                    for layer in decoder_layers
                    if layer in _DECODER_TOKEN_LENGTH_FILTER_LAYERS
                ],
                "maximum_available_length": int(token_lengths.max()),
                "character_count_ignores_surrounding_whitespace": True,
            },
            "warnings": [
                "Model probabilities are raw teacher-forced Whisper probabilities before generation-time suppression and timestamp rules.",
                "Encoder grid scores are target-mean-relative readout-logit changes; decoder grid scores are raw readout logits. Neither is a calibrated causal effect or probability.",
                "The interactive grids use lexical-display rank as their primary lens rank; each serialized candidate also carries exact display and full-vocabulary rank provenance. Held-out evaluation still ranks the full vocabulary.",
                "Encoder states are bidirectional and may contain information from later audio in the same 30-second window.",
                "An encoder cell's realized-token rank uses the model-derived Whisper cross-attention/DTW token interval with greatest temporal overlap (ties use closest interval midpoint, then lower token position). This is approximate synchronization, not a phoneme/word boundary or causal attribution.",
                "Encoder-to-decoder J-lens is an experimental extension not validated by the source paper.",
                "The optional encoder token-length filter reranks over lexical tokens up to a user-selected decoded character count. It is a phoneme-oriented exploration aid, not a phoneme classifier, and is not part of the source J-lens method.",
                "The optional decoder token-length filter reranks the lexical display vocabulary for decoder L0 and L1 only. Decoder L2 remains character-length-unfiltered but still uses the lexical display vocabulary; the output head is fully unfiltered.",
                *(
                    [
                        "The optional phone-signature view ranks frozen training-phone prototypes by sparse top-100 J-signature cosine similarity. It is not probability, model confidence, a literal phone head, or causal attribution; pooled display windows can contain multiple phones and there is no fitted silence/unknown class."
                    ]
                    if phone_signature_prototypes is not None
                    else []
                ),
            ],
        },
    }
