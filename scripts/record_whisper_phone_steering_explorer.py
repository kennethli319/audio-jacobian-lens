#!/usr/bin/env python3
"""Record complete Explorer analyses for the Laurel/Yanny steering replay.

The code is public, but its default inputs intentionally live under the
gitignored ``artifacts/private`` tree.  It records three real Whisper runs:
the unedited baseline, the frozen equal-strength Yanny intervention, and the
frozen target-conditioned Laurel coefficient ray at scale 0.7.  No private
path is serialized into the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from jlens.audio_io import DecodedAudio, decode_audio_bytes
from jlens.fit_whisper_cli import _device
from jlens.hooks import ResidualAdder
from jlens.phonetic_signatures import PhoneSignaturePrototypes
from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_analysis import (
    WhisperAnalysisCapture,
    analyze_whisper_run,
)
from jlens.whisper_causal import (
    EncoderIntervention,
    EncoderInterventionSchedule,
)
from jlens.whisper_lens import WhisperJacobianLens

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
SCHEMA_ID = "audio-jacobian-lens.recorded-asr-steering"
SCHEMA_VERSION = 1
ORIGINAL_POST_URL = "https://hrbosker.github.io/demos/laurel-yanny/"

DEFAULT_AUDIO = (
    ROOT / "artifacts/private/phonetic_encoder/causal/laurel-yanny-original.mp3"
)
DEFAULT_ENCODER_LENS = (
    ROOT / "artifacts/private/phonetic_encoder/lenses/encoder_a2_global_mean.pt"
)
DEFAULT_DECODER_LENS = ROOT / "artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt"
DEFAULT_DISPLAY_PHONES = (
    ROOT / "artifacts/private/phonetic_encoder/lenses/"
    "a2_phone_signature_prototypes_v1.pt"
)
DEFAULT_STEERING_PHONES = (
    ROOT / "artifacts/private/phonetic_encoder/lenses/"
    "a2_phone_signature_prototypes_y35_v1.pt"
)
DEFAULT_LAUREL_RECIPE = (
    ROOT / "artifacts/private/phonetic_encoder/results/"
    "laurel_phone_basis_optimization_active_v1.json"
)
DEFAULT_PUBLIC_CHECKPOINTS = ROOT / "data/static_phone_steering_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/private/phonetic_encoder/causal/"
    "recorded_asr_steering_explorer_v1.json"
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recommended_checkpoint(target: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = target.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("public steering target has no checkpoint list")
    matches = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict) and checkpoint.get("id") == "recommended"
    ]
    if len(matches) != 1:
        raise ValueError("public steering target needs one recommended checkpoint")
    checkpoint = matches[0]
    if (
        checkpoint.get("recorded") is not True
        or checkpoint.get("interpolated") is not False
    ):
        raise ValueError(
            "recommended public checkpoint must be recorded, not interpolated"
        )
    return checkpoint


def _canonical_condition_metadata(
    public_payload: Mapping[str, Any], condition_id: str
) -> dict[str, Any]:
    """Copy the evidence language already reviewed for the public replay."""

    if condition_id == "baseline":
        baseline = public_payload.get("baseline")
        if not isinstance(baseline, Mapping):
            raise ValueError("public steering payload has no baseline")
        return {
            "id": "baseline",
            "label": str(baseline.get("label", "No intervention")),
            "recorded": True,
            "interpolated": False,
            "generated": deepcopy(baseline["generated"]),
            "budget_fraction": 0.0,
            "coefficient_scale": 0.0,
            "evidence": {
                "tier": "observed_baseline",
                "badge": "Baseline · no intervention",
                "tone": "neutral",
                "summary": "The same pinned model and source clip with no residual edit.",
            },
            "method": {
                "kind": "unmodified_whisper_forward_pass",
                "label": "No residual intervention",
                "description": "Generate and teacher-force the model's ordinary baseline output without adding any encoder residual.",
                "coefficient_policy": "No fitted phone direction is applied.",
            },
            "layers": [],
            "schedule": [],
        }

    targets = public_payload.get("targets")
    if not isinstance(targets, Mapping) or not isinstance(
        targets.get(condition_id), Mapping
    ):
        raise ValueError(f"public steering payload has no {condition_id!r} target")
    target = targets[condition_id]
    checkpoint = _recommended_checkpoint(target)
    return {
        "id": condition_id,
        "label": str(target["label"]),
        "recorded": True,
        "interpolated": False,
        "generated": deepcopy(checkpoint["generated"]),
        "budget_fraction": float(checkpoint["budget_fraction"]),
        "coefficient_scale": float(checkpoint["coefficient_scale"]),
        "evidence": deepcopy(target["evidence"]),
        "method": deepcopy(target["method"]),
        "layers": [int(layer) for layer in target["layers"]],
        "schedule": deepcopy(target["schedule"]),
    }


def _equal_coefficients(
    condition: Mapping[str, Any], coefficient: float
) -> list[dict[str, Any]]:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("equal steering coefficient must be finite and nonnegative")
    return [
        {
            "layer": int(layer),
            "segment_index": segment_index,
            "phone": str(segment["phone"]),
            "start_position": int(segment["start_position"]),
            "end_position": int(segment["end_position"]),
            "coefficient": coefficient,
        }
        for layer in condition["layers"]
        for segment_index, segment in enumerate(condition["schedule"])
    ]


def _scaled_laurel_coefficients(
    recipe: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    scale: float,
) -> list[dict[str, Any]]:
    """Validate and scale the frozen optimized coefficient ray."""

    if recipe.get("format") != "audio-jlens-private-laurel-phone-basis-search-v1":
        raise ValueError("Laurel recipe has an unexpected format")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Laurel coefficient scale must be finite and nonnegative")
    raw_coefficients = (recipe.get("best") or {}).get("coefficients")
    if not isinstance(raw_coefficients, list):
        raise ValueError("Laurel recipe has no frozen best coefficients")

    expected = {
        (int(layer), segment_index): (
            str(segment["phone"]),
            int(segment["start_position"]),
            int(segment["end_position"]),
        )
        for layer in condition["layers"]
        for segment_index, segment in enumerate(condition["schedule"])
    }
    if len(raw_coefficients) != len(expected):
        raise ValueError("Laurel recipe does not cover every layer-by-phone coordinate")

    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in raw_coefficients:
        if not isinstance(raw, Mapping):
            raise ValueError("Laurel recipe coefficient must be an object")
        key = (int(raw["layer"]), int(raw["segment_index"]))
        if key not in expected or key in seen:
            raise ValueError(
                "Laurel recipe contains an unknown or duplicate coordinate"
            )
        expected_phone, expected_start, expected_end = expected[key]
        actual_coordinate = (
            str(raw["phone"]),
            int(raw["start_position"]),
            int(raw["end_position"]),
        )
        if actual_coordinate != (expected_phone, expected_start, expected_end):
            raise ValueError("Laurel recipe coordinate disagrees with public schedule")
        source = float(raw["coefficient"])
        if not math.isfinite(source) or source < 0:
            raise ValueError("Laurel recipe coefficient must be finite and nonnegative")
        output.append(
            {
                "layer": key[0],
                "segment_index": key[1],
                "phone": expected_phone,
                "start_position": expected_start,
                "end_position": expected_end,
                "source_coefficient": source,
                "coefficient": source * scale,
            }
        )
        seen.add(key)
    output.sort(key=lambda item: (item["layer"], item["segment_index"]))
    return output


def _build_schedule(
    model: HFWhisperLensModel,
    encoder_lens,
    steering_phones: PhoneSignaturePrototypes,
    baseline_encoder: Mapping[int, torch.Tensor],
    coefficients: Sequence[Mapping[str, Any]],
) -> tuple[EncoderInterventionSchedule, list[dict[str, Any]]]:
    """Pull frozen phone objectives back at the baseline span means."""

    interventions: list[EncoderIntervention] = []
    pullbacks: list[dict[str, Any]] = []
    for coordinate in coefficients:
        layer = int(coordinate["layer"])
        start = int(coordinate["start_position"])
        end = int(coordinate["end_position"])
        phone = str(coordinate["phone"])
        coefficient = float(coordinate["coefficient"])
        if layer not in baseline_encoder:
            raise ValueError(f"baseline capture has no encoder layer {layer}")
        residual = baseline_encoder[layer]
        if residual.ndim != 3 or residual.shape[0] != 1:
            raise ValueError("baseline encoder residuals must have one batch")
        if not 0 <= start < end <= residual.shape[1]:
            raise ValueError("steering coordinate is outside the encoder residual span")
        reference = residual[0, start:end].float().mean(dim=0)
        pullback = steering_phones.prototype_logit_pullback(
            model,
            encoder_lens,
            reference,
            layer=layer,
            target_phone=phone,
        )
        interventions.append(
            EncoderIntervention(
                layer=layer,
                start_position=start,
                end_position=end,
                direction=pullback.direction,
                strength=coefficient,
            )
        )
        pullbacks.append(
            {
                "layer": layer,
                "segment_index": int(coordinate["segment_index"]),
                "phone": phone,
                "start_position": start,
                "end_position": end,
                "coefficient": coefficient,
                "reference_span_mean_l2_norm": float(
                    residual[0, start:end].float().norm(dim=-1).mean().cpu()
                ),
                "objective_value": pullback.objective_value,
                "target_readout_score": pullback.target_readout_score,
                "other_mean_readout_score": pullback.other_mean_readout_score,
                "competing_phone_count": pullback.competing_phone_count,
                "gradient_l2_norm": pullback.gradient_l2_norm,
                "objective_kind": pullback.objective_kind,
                "is_probability": pullback.is_probability,
            }
        )
    return EncoderInterventionSchedule(tuple(interventions)), pullbacks


@contextmanager
def _residual_edits(
    model: HFWhisperLensModel, deltas: Mapping[int, torch.Tensor] | None
):
    with ExitStack() as stack:
        for layer, delta in sorted((deltas or {}).items()):
            stack.enter_context(
                ResidualAdder(model.encoder_layers, layer=layer, delta=delta)
            )
        yield


def _capture(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    inputs: WhisperLensInputs,
    *,
    deltas: Mapping[int, torch.Tensor] | None = None,
) -> WhisperAnalysisCapture:
    encoder_layers = [] if lens.encoder is None else lens.encoder.source_layers
    decoder_layers = [] if lens.decoder is None else lens.decoder.source_layers
    with _residual_edits(model, deltas):
        encoder, decoder, logits = model.capture(
            inputs,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
    return WhisperAnalysisCapture(encoder, decoder, logits)


@torch.no_grad()
def _generate(
    model: HFWhisperLensModel,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    deltas: Mapping[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    """Generate under edits, retaining DTW token timestamps when supported."""

    timestamp_error: str | None = None
    with _residual_edits(model, deltas):
        try:
            generated = model.generate(
                input_features,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                return_token_timestamps=True,
            )
        except (TypeError, ValueError) as error:
            timestamp_error = f"{type(error).__name__}: {error}"
            generated = model.generate(
                input_features,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
            )
    sequences = generated["sequences"].detach().cpu()
    timestamps = generated.get("token_timestamps")
    if timestamps is not None:
        timestamps = timestamps.detach().cpu()
    return sequences, timestamps, timestamp_error


def _prepare_inputs(
    model: HFWhisperLensModel,
    decoded: DecodedAudio,
    sequence_ids: torch.Tensor,
    *,
    condition_id: str,
) -> WhisperLensInputs:
    special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
    has_ordinary_target = any(
        int(token_id) not in special_ids for token_id in sequence_ids[0, 1:]
    )
    return model.prepare_audio(
        decoded.waveform,
        sampling_rate=decoded.sampling_rate,
        sequence_ids=sequence_ids,
        include_eos_target=not has_ordinary_target,
        duration_seconds=decoded.duration_seconds,
        metadata={"recorded_condition": condition_id},
    )


def _generation_summary(
    model: HFWhisperLensModel, sequence_ids: torch.Tensor
) -> dict[str, Any]:
    special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
    ordinary = [
        int(token_id)
        for token_id in sequence_ids[0].tolist()
        if int(token_id) not in special_ids
    ]
    return {
        "text": model.tokenizer.decode(
            ordinary,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip(),
        "token_ids": ordinary,
    }


def _assert_generation_matches(
    condition: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    expected = condition["generated"]
    if actual.get("text") != expected.get("text") or actual.get(
        "token_ids"
    ) != expected.get("token_ids"):
        raise RuntimeError(
            f"recorded {condition['id']} generation changed: expected "
            f"{expected.get('text')!r} {expected.get('token_ids')!r}, got "
            f"{actual.get('text')!r} {actual.get('token_ids')!r}"
        )


def _serialize_analysis(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    display_phones: PhoneSignaturePrototypes,
    decoded: DecodedAudio,
    inputs: WhisperLensInputs,
    capture: WhisperAnalysisCapture,
    timestamps: torch.Tensor | None,
    *,
    top_k: int,
    time_bin_seconds: float,
    time_bin_overlap_seconds: float,
) -> dict[str, Any]:
    return analyze_whisper_run(
        model,
        lens,
        inputs,
        decoded.waveform,
        token_timestamps=timestamps,
        top_k=top_k,
        time_bin_seconds=time_bin_seconds,
        time_bin_overlap_seconds=time_bin_overlap_seconds,
        phone_signature_prototypes=display_phones,
        captured=capture,
    )


def _condition_with_analysis(
    condition: dict[str, Any],
    *,
    analysis: dict[str, Any],
    generation: Mapping[str, Any],
    token_timestamps_available: bool,
    timestamp_error: str | None,
    coefficients: Sequence[Mapping[str, Any]],
    pullbacks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _assert_generation_matches(condition, generation)
    return {
        **condition,
        "recording": {
            "capture_convention": "post_block_residuals_after_registered_encoder_edits_then_untouched_downstream_recomputation",
            "teacher_forced_on_realized_sequence": True,
            "token_timestamps_available": token_timestamps_available,
            "token_timestamp_fallback": timestamp_error,
            "coefficients": [dict(item) for item in coefficients],
            "pullback_audit": [dict(item) for item in pullbacks],
        },
        "analysis": analysis,
    }


def record(args: argparse.Namespace) -> dict[str, Any]:
    """Run all three recorded conditions and return the raw JSON payload."""

    required = {
        "source audio": args.audio,
        "A2 encoder lens": args.encoder_lens,
        "decoder lens": args.decoder_lens,
        "public 34-phone prototypes": args.display_phones,
        "private Y35 steering prototypes": args.steering_phones,
        "Laurel recipe": args.laurel_recipe,
        "public steering checkpoints": args.public_checkpoints,
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(f"{label} not found: {path}")

    public = _load_json(args.public_checkpoints, label="public steering checkpoints")
    if public.get("schema_id") != "audio-jacobian-lens.phone-steering":
        raise ValueError("public steering checkpoints have an unexpected schema")
    conditions = {
        condition_id: _canonical_condition_metadata(public, condition_id)
        for condition_id in ("baseline", "yanny", "laurel")
    }
    if conditions["yanny"]["coefficient_scale"] != 0.035:
        raise ValueError("public Yanny checkpoint is no longer the 3.5% recipe")
    if conditions["laurel"]["coefficient_scale"] != 0.7:
        raise ValueError("public Laurel checkpoint is no longer the scale-0.7 recipe")

    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = _device(args.device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    hf_model.to(device)
    model = HFWhisperLensModel(hf_model, processor, model_id=MODEL_ID)

    encoder_bundle = WhisperJacobianLens.load(str(args.encoder_lens))
    decoder_bundle = WhisperJacobianLens.load(str(args.decoder_lens))
    lens = WhisperJacobianLens.combine_streams(
        encoder_bundle=encoder_bundle,
        decoder_bundle=decoder_bundle,
    )
    lens.validate_model(model)
    if lens.encoder is None or lens.decoder is None:
        raise ValueError(
            "the recorded Explorer requires both encoder and decoder lenses"
        )
    display_phones = PhoneSignaturePrototypes.load(args.display_phones)
    steering_phones = PhoneSignaturePrototypes.load(args.steering_phones)
    display_phones.validate(model=model, encoder_lens=lens.encoder)
    steering_phones.validate(model=model, encoder_lens=lens.encoder)
    if "Y" in display_phones.labels:
        raise ValueError("the public display bank must remain the frozen 34-phone bank")
    if "Y" not in steering_phones.labels:
        raise ValueError("the private steering bank must contain the Y extension")

    decoded = decode_audio_bytes(args.audio.read_bytes())
    features = model.processor.feature_extractor(
        decoded.waveform,
        sampling_rate=decoded.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = features.input_features.to(device)
    attention_mask = features.attention_mask.to(device)

    baseline_ids, baseline_timestamps, baseline_timestamp_error = _generate(
        model, input_features, attention_mask
    )
    baseline_inputs = _prepare_inputs(
        model, decoded, baseline_ids, condition_id="baseline"
    )
    baseline_capture = _capture(model, lens, baseline_inputs)
    baseline_analysis = _serialize_analysis(
        model,
        lens,
        display_phones,
        decoded,
        baseline_inputs,
        baseline_capture,
        baseline_timestamps,
        top_k=args.top_k,
        time_bin_seconds=args.time_bin_seconds,
        time_bin_overlap_seconds=args.time_bin_overlap_seconds,
    )
    baseline_generation = _generation_summary(model, baseline_ids)
    recorded_conditions: list[dict[str, Any]] = [
        _condition_with_analysis(
            conditions["baseline"],
            analysis=baseline_analysis,
            generation=baseline_generation,
            token_timestamps_available=baseline_timestamps is not None,
            timestamp_error=baseline_timestamp_error,
            coefficients=[],
            pullbacks=[],
        )
    ]

    laurel_recipe = _load_json(args.laurel_recipe, label="Laurel recipe")
    coefficient_sets = {
        "yanny": _equal_coefficients(
            conditions["yanny"], conditions["yanny"]["coefficient_scale"]
        ),
        "laurel": _scaled_laurel_coefficients(
            laurel_recipe,
            conditions["laurel"],
            scale=conditions["laurel"]["coefficient_scale"],
        ),
    }
    for condition_id in ("yanny", "laurel"):
        schedule, pullbacks = _build_schedule(
            model,
            lens.encoder,
            steering_phones,
            baseline_capture.encoder_activations,
            coefficient_sets[condition_id],
        )
        deltas = schedule.make_deltas(dict(baseline_capture.encoder_activations))
        sequence_ids, timestamps, timestamp_error = _generate(
            model,
            input_features,
            attention_mask,
            deltas=deltas,
        )
        inputs = _prepare_inputs(
            model, decoded, sequence_ids, condition_id=condition_id
        )
        capture = _capture(model, lens, inputs, deltas=deltas)
        analysis = _serialize_analysis(
            model,
            lens,
            display_phones,
            decoded,
            inputs,
            capture,
            timestamps,
            top_k=args.top_k,
            time_bin_seconds=args.time_bin_seconds,
            time_bin_overlap_seconds=args.time_bin_overlap_seconds,
        )
        recorded_conditions.append(
            _condition_with_analysis(
                conditions[condition_id],
                analysis=analysis,
                generation=_generation_summary(model, sequence_ids),
                token_timestamps_available=timestamps is not None,
                timestamp_error=timestamp_error,
                coefficients=coefficient_sets[condition_id],
                pullbacks=pullbacks,
            )
        )

    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "default": "baseline",
        "source": {
            "title": "Laurel or Yanny?",
            "original_post_url": ORIGINAL_POST_URL,
            "audio_sha256": _sha256(args.audio),
            "duration_seconds": decoded.duration_seconds,
        },
        "provenance": {
            "recorded_only": True,
            "interpolated": False,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_fingerprint": model.fingerprint,
            "device": str(device),
            "display_phone_inventory_size": len(display_phones.labels),
            "steering_phone_inventory_size": len(steering_phones.labels),
            "artifact_sha256": {
                "encoder_lens": _sha256(args.encoder_lens),
                "decoder_lens": _sha256(args.decoder_lens),
                "display_phone_prototypes": _sha256(args.display_phones),
                "steering_phone_prototypes": _sha256(args.steering_phones),
                "laurel_recipe": _sha256(args.laurel_recipe),
                "public_checkpoint_metadata": _sha256(args.public_checkpoints),
            },
        },
        "conditions": recorded_conditions,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record baseline, Yanny, and Laurel raw ASR Explorer analyses"
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--encoder-lens", type=Path, default=DEFAULT_ENCODER_LENS)
    parser.add_argument("--decoder-lens", type=Path, default=DEFAULT_DECODER_LENS)
    parser.add_argument("--display-phones", type=Path, default=DEFAULT_DISPLAY_PHONES)
    parser.add_argument("--steering-phones", type=Path, default=DEFAULT_STEERING_PHONES)
    parser.add_argument("--laurel-recipe", type=Path, default=DEFAULT_LAUREL_RECIPE)
    parser.add_argument(
        "--public-checkpoints", type=Path, default=DEFAULT_PUBLIC_CHECKPOINTS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--time-bin-seconds", type=float, default=0.1)
    parser.add_argument("--time-bin-overlap-seconds", type=float, default=0.02)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = record(args)
    _write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
