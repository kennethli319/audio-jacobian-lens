#!/usr/bin/env python3
"""Publish the recorded Laurel/Yanny runs as one static ASR Explorer report.

The recorder intentionally writes a private, audit-heavy capture.  This script
is the only bridge from that capture to the public site: it reduces each raw
analysis through the ordinary Explorer allowlists, keeps the unedited run as
the report payload, stores only the two alternate bounded matrix views in the
replay extension, and rebuilds the ASR manifest without the displaced
``impossible`` report.  The Speech explorer and its ``impossible`` report are
never touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from jlens.static_explorer_catalog import load_static_explorer_catalog
    from scripts import export_static_explorer as exporter
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jlens.static_explorer_catalog import load_static_explorer_catalog
    from scripts import export_static_explorer as exporter


CAPTURE_SCHEMA_ID = "audio-jacobian-lens.recorded-asr-steering"
REPLAY_SCHEMA_ID = "audio-jacobian-lens.recorded-asr-intervention-replay"
REPLAY_MODE = "static_recorded_analyses"
SCHEMA_VERSION = 1
SAMPLE_SLUG = "laurel-yanny"
DISPLACED_SLUG = "impossible"
CONDITION_IDS = ("baseline", "yanny", "laurel")
EXPECTED_CONDITIONS = {
    "baseline": {
        "text": "Lily!",
        "token_ids": [20037, 0],
        "target_match": False,
        "budget_fraction": 0.0,
        "coefficient_scale": 0.0,
        "evidence_tier": "observed_baseline",
        "layers": [],
        "schedule_count": 0,
    },
    "yanny": {
        "text": "Yanny!",
        "token_ids": [575, 7737, 0],
        "target_match": True,
        "budget_fraction": 0.035,
        "coefficient_scale": 0.035,
        "evidence_tier": "open_loop_cross_fit_reproduced",
        "layers": [0, 1, 2, 3],
        "schedule_count": 4,
    },
    "laurel": {
        "text": "Laurel",
        "token_ids": [43442],
        "target_match": True,
        "budget_fraction": 0.1452915875040831,
        "coefficient_scale": 0.7,
        "evidence_tier": "target_conditioned_clip_specific_existence",
        "layers": [0, 1, 2, 3],
        "schedule_count": 5,
    },
}
PINNED_PRIVATE_ARTIFACT_SHA256 = {
    # These values bind the private capture to the exact audited inputs used by
    # the recorder.  They deliberately live in tracked publisher code rather
    # than being accepted from the untrusted capture envelope.
    "steering_phone_prototypes": (
        "f894e17952ea18cebf57d1d76c84991d3196dfb7ccb4d47d530f28578ad6214f"
    ),
    "laurel_recipe": (
        "cbc875c05274a5b92f34101be0a854ec1888b3537f4261978a2fc74e261e9c46"
    ),
    "public_checkpoint_metadata": (
        "ef0499b1dce6dfe241284d58aa508ad266507a118a456c4fb41edeb3f6eddea7"
    ),
}
CONDITION_PUBLIC_FIELDS = (
    "id",
    "label",
    "recorded",
    "interpolated",
    "generated",
    "budget_fraction",
    "coefficient_scale",
    "evidence",
    "method",
    "layers",
    "schedule",
)
CAPTURE_CONDITION_FIELDS = set(CONDITION_PUBLIC_FIELDS) | {"analysis", "recording"}
GENERATED_FIELDS = ("text", "token_ids", "target_match")
EVIDENCE_FIELDS = ("tier", "badge", "tone", "summary")
METHOD_FIELDS = ("kind", "label", "description", "coefficient_policy")
SCHEDULE_FIELDS = (
    "phone",
    "start_seconds",
    "end_seconds",
    "start_position",
    "end_position",
)
REPLAY_ANALYSIS_FIELDS = ("metadata", "transcription", "encoder", "decoder")
REPLAY_FORBIDDEN_KEYS = {
    "analysis_id",
    "audio",
    "audio_data_url",
    "audio_url",
    "branch_analysis_id",
    "coefficients",
    "delta",
    "deltas",
    "direction",
    "generated_audio",
    "model_input_wav",
    "output_waveform",
    "parent_analysis_id",
    "pullback_audit",
    "recording",
    "repository_path",
    "residual",
    "residuals",
    "waveform",
}
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PHONE_RE = re.compile(r"^[A-Z]{1,3}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _canonical_json_value(value: Any) -> Any:
    """Stabilize hashes despite exporter allowlists implemented as sets."""

    if isinstance(value, Mapping):
        return {
            key: _canonical_json_value(value[key]) for key in sorted(value, key=str)
        }
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    return value


def _write_canonical_json(
    path: Path, value: Mapping[str, Any], *, compact: bool = False
) -> None:
    exporter._write_json(path, _canonical_json_value(value), compact=compact)


def _close(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-8)
    except (TypeError, ValueError, OverflowError):
        return False


def _require_exact_fields(
    value: Any, fields: tuple[str, ...] | set[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        unexpected = sorted(str(key) for key in actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            f"{label} has unexpected or missing fields "
            f"(unexpected={unexpected}, missing={missing})"
        )
    return value


def _require_string(value: Any, *, label: str, minimum: int = 1, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{label} must be a string with length {minimum}..{maximum}")
    return value


def _require_finite_number(
    value: Any, *, label: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} is outside {minimum}..{maximum}")
    return numeric


def _validate_generated(value: Any, *, label: str) -> Mapping[str, Any]:
    generated = _require_exact_fields(value, GENERATED_FIELDS, label=label)
    _require_string(generated["text"], label=f"{label}.text", maximum=128)
    token_ids = generated["token_ids"]
    if not isinstance(token_ids, list) or not 1 <= len(token_ids) <= 16:
        raise ValueError(f"{label}.token_ids must contain 1..16 token ids")
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int or not 0 <= token_id < 51_864:
            raise ValueError(f"{label}.token_ids[{index}] is outside the vocabulary")
    if type(generated["target_match"]) is not bool:
        raise ValueError(f"{label}.target_match must be boolean")
    return generated


def _validate_evidence(value: Any, *, label: str) -> Mapping[str, Any]:
    evidence = _require_exact_fields(value, EVIDENCE_FIELDS, label=label)
    tier = _require_string(evidence["tier"], label=f"{label}.tier", maximum=64)
    tone = _require_string(evidence["tone"], label=f"{label}.tone", maximum=64)
    if not _SLUG_RE.fullmatch(tier) or not _SLUG_RE.fullmatch(tone):
        raise ValueError(f"{label} tier and tone must be bounded lowercase slugs")
    _require_string(evidence["badge"], label=f"{label}.badge", maximum=160)
    _require_string(evidence["summary"], label=f"{label}.summary", maximum=1000)
    return evidence


def _validate_method(value: Any, *, label: str) -> Mapping[str, Any]:
    method = _require_exact_fields(value, METHOD_FIELDS, label=label)
    kind = _require_string(method["kind"], label=f"{label}.kind", maximum=80)
    if not _SLUG_RE.fullmatch(kind):
        raise ValueError(f"{label}.kind must be a bounded lowercase slug")
    _require_string(method["label"], label=f"{label}.label", maximum=160)
    _require_string(method["description"], label=f"{label}.description", maximum=1000)
    _require_string(
        method["coefficient_policy"],
        label=f"{label}.coefficient_policy",
        maximum=1000,
    )
    return method


def _validate_schedule(value: Any, *, condition_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"recorded {condition_id} schedule must be a list")
    expected_count = EXPECTED_CONDITIONS[condition_id]["schedule_count"]
    if len(value) != expected_count:
        raise ValueError(
            f"recorded {condition_id} schedule must have {expected_count} segments"
        )

    result: list[Mapping[str, Any]] = []
    previous_end_seconds = 0.0
    previous_end_position = 0
    for index, raw in enumerate(value):
        label = f"recorded {condition_id} schedule[{index}]"
        item = _require_exact_fields(raw, SCHEDULE_FIELDS, label=label)
        phone = _require_string(item["phone"], label=f"{label}.phone", maximum=3)
        if not _PHONE_RE.fullmatch(phone):
            raise ValueError(f"{label}.phone must be a bounded ARPAbet-style label")
        start_seconds = _require_finite_number(
            item["start_seconds"],
            label=f"{label}.start_seconds",
            minimum=0.0,
            maximum=30.0,
        )
        end_seconds = _require_finite_number(
            item["end_seconds"],
            label=f"{label}.end_seconds",
            minimum=0.0,
            maximum=30.0,
        )
        start_position = item["start_position"]
        end_position = item["end_position"]
        if (
            type(start_position) is not int
            or type(end_position) is not int
            or not 0 <= start_position < end_position <= 1500
        ):
            raise ValueError(f"{label} positions are outside the encoder bounds")
        if start_seconds >= end_seconds:
            raise ValueError(f"{label} has a non-positive time interval")
        if (
            start_seconds < previous_end_seconds - 1e-8
            or start_position < previous_end_position
        ):
            raise ValueError(f"recorded {condition_id} schedule is not ordered")
        if not (
            math.isclose(
                start_seconds, start_position * 0.02, rel_tol=0.0, abs_tol=1e-8
            )
            and math.isclose(
                end_seconds, end_position * 0.02, rel_tol=0.0, abs_tol=1e-8
            )
        ):
            raise ValueError(f"{label} time does not match its native 20 ms positions")
        previous_end_seconds = end_seconds
        previous_end_position = end_position
        result.append(item)
    return result


def _condition_map(capture: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = capture.get("conditions")
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("recorded capture has no condition list")
    ids = [str(item.get("id") or "") for item in raw]
    if ids != list(CONDITION_IDS):
        raise ValueError("recorded capture conditions must be baseline, Yanny, Laurel")
    return {str(item["id"]): item for item in raw}


def _validate_capture(capture: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        capture.get("schema_id") != CAPTURE_SCHEMA_ID
        or capture.get("schema_version") != SCHEMA_VERSION
        or capture.get("default") != "baseline"
    ):
        raise ValueError("recorded capture has an unsupported envelope")
    provenance = capture.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("recorded_only") is not True
        or provenance.get("interpolated") is not False
    ):
        raise ValueError("recorded capture is not an all-recorded bundle")
    conditions = _condition_map(capture)
    for condition_id, condition in conditions.items():
        _require_exact_fields(
            condition,
            CAPTURE_CONDITION_FIELDS,
            label=f"recorded {condition_id} condition",
        )
        expected = EXPECTED_CONDITIONS[condition_id]
        generated = _validate_generated(
            condition["generated"], label=f"recorded {condition_id}.generated"
        )
        evidence = _validate_evidence(
            condition["evidence"], label=f"recorded {condition_id}.evidence"
        )
        _validate_method(condition["method"], label=f"recorded {condition_id}.method")
        schedule = _validate_schedule(condition["schedule"], condition_id=condition_id)
        analysis = condition["analysis"]
        _require_string(
            condition["label"],
            label=f"recorded {condition_id}.label",
            maximum=160,
        )
        layers = condition["layers"]
        if not isinstance(layers, list) or any(
            type(layer) is not int or not 0 <= layer <= 3 for layer in layers
        ):
            raise ValueError(f"recorded {condition_id} layers are invalid")
        if len(layers) != len(set(layers)) or layers != sorted(layers):
            raise ValueError(
                f"recorded {condition_id} layers are not unique and ordered"
            )
        if layers != expected["layers"]:
            raise ValueError(f"recorded {condition_id} layers changed")
        _require_finite_number(
            condition["budget_fraction"],
            label=f"recorded {condition_id}.budget_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            condition["coefficient_scale"],
            label=f"recorded {condition_id}.coefficient_scale",
            minimum=0.0,
            maximum=10.0,
        )
        if (
            condition["id"] != condition_id
            or condition["recorded"] is not True
            or condition["interpolated"] is not False
            or generated.get("text") != expected["text"]
            or generated.get("token_ids") != expected["token_ids"]
            or generated.get("target_match") is not expected["target_match"]
            or not _close(condition.get("budget_fraction"), expected["budget_fraction"])
            or not _close(
                condition.get("coefficient_scale"), expected["coefficient_scale"]
            )
            or not isinstance(evidence, Mapping)
            or evidence.get("tier") != expected["evidence_tier"]
            or not isinstance(analysis, Mapping)
            or not isinstance(condition["recording"], Mapping)
        ):
            raise ValueError(f"recorded {condition_id} condition changed")
        if len(schedule) != expected["schedule_count"]:  # defensive narrowing
            raise ValueError(f"recorded {condition_id} steering schedule changed")
        tokens = analysis.get("transcription", {}).get("tokens")
        if not isinstance(tokens, list) or [
            token.get("id") for token in tokens
        ] != list(expected["token_ids"]):
            raise ValueError(f"recorded {condition_id} token sequence changed")
        if (
            str(analysis.get("transcription", {}).get("text") or "").strip()
            != expected["text"]
        ):
            raise ValueError(f"recorded {condition_id} transcript changed")
    return conditions


def _validate_capture_provenance(
    capture: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_audio_sha256: str,
) -> None:
    source = capture.get("source")
    provenance = capture.get("provenance")
    public = manifest.get("provenance")
    if not all(isinstance(value, Mapping) for value in (source, provenance, public)):
        raise ValueError("capture or manifest provenance is missing")
    if source.get("audio_sha256") != expected_audio_sha256:
        raise ValueError("recorded capture uses a different source-audio hash")
    model = public.get("model")
    lens = public.get("lens")
    if not isinstance(model, Mapping) or not isinstance(lens, Mapping):
        raise ValueError("ASR manifest lacks model or lens provenance")
    for field, manifest_field in (
        ("model_id", "id"),
        ("model_revision", "revision"),
        ("model_fingerprint", "model_fingerprint"),
    ):
        if provenance.get(field) != model.get(manifest_field):
            raise ValueError(f"capture {field} disagrees with the ASR manifest")
    artifacts = provenance.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise ValueError("recorded capture has no artifact hashes")
    expected_hashes = {
        "encoder_lens": (lens.get("encoder") or {}).get("sha256"),
        "decoder_lens": (lens.get("decoder") or {}).get("sha256"),
        "display_phone_prototypes": (lens.get("phone_signature") or {}).get("sha256"),
        **PINNED_PRIVATE_ARTIFACT_SHA256,
    }
    if set(artifacts) != set(expected_hashes):
        raise ValueError("recorded capture artifact hash map changed")
    for field, expected in expected_hashes.items():
        captured = artifacts.get(field)
        if (
            not isinstance(expected, str)
            or not _SHA256_RE.fullmatch(expected)
            or not isinstance(captured, str)
            or not _SHA256_RE.fullmatch(captured)
            or captured != expected
        ):
            raise ValueError(f"capture {field} hash disagrees with the ASR manifest")


def _public_condition(
    condition: Mapping[str, Any], *, include_analysis: bool
) -> dict[str, Any]:
    published = {
        "id": condition["id"],
        "label": condition["label"],
        "recorded": condition["recorded"],
        "interpolated": condition["interpolated"],
        "generated": {
            field: copy.deepcopy(condition["generated"][field])
            for field in GENERATED_FIELDS
        },
        "budget_fraction": condition["budget_fraction"],
        "coefficient_scale": condition["coefficient_scale"],
        "evidence": {field: condition["evidence"][field] for field in EVIDENCE_FIELDS},
        "method": {field: condition["method"][field] for field in METHOD_FIELDS},
        "layers": copy.deepcopy(condition["layers"]),
        "schedule": [
            {field: item[field] for field in SCHEDULE_FIELDS}
            for item in condition["schedule"]
        ],
    }
    # Normalize the insignificant measurement noise in the 3.5% public budget.
    published["budget_fraction"] = EXPECTED_CONDITIONS[str(condition["id"])][
        "budget_fraction"
    ]
    if include_analysis:
        reduced = exporter._reduce_payload(condition["analysis"])
        published["analysis"] = {
            field: reduced[field] for field in REPLAY_ANALYSIS_FIELDS
        }
    return published


def _encoder_geometry(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoder = payload.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("recorded condition has no encoder matrix")
    cells = encoder.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("recorded condition has an empty encoder matrix")
    return {
        "layers": copy.deepcopy(encoder.get("layers")),
        "pooling": copy.deepcopy(encoder.get("pooling")),
        "positions": copy.deepcopy(encoder.get("positions")),
        "time_bins": copy.deepcopy(encoder.get("time_bins")),
        "widths": [len(row) if isinstance(row, list) else -1 for row in cells],
        "coordinates": [
            [
                {
                    "position_index": cell.get("position_index"),
                    "time_window": cell.get("time_window"),
                }
                for cell in row
            ]
            for row in cells
            if isinstance(row, list)
        ],
    }


def _validate_replay_safe(
    value: Any, *, path: str = "$.recorded_intervention_replay"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in REPLAY_FORBIDDEN_KEYS or str(key).endswith(
                "_analysis_id"
            ):
                raise ValueError(f"forbidden replay field {path}.{key}")
            _validate_replay_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_replay_safe(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.strip().lower().replace("\\", "/")
        if (
            "/users/" in lowered
            or "artifacts/private/" in lowered
            or lowered.startswith("data:audio/")
            or lowered.endswith(
                (".pt", ".pth", ".npy", ".npz", ".wav", ".flac", ".mp3")
            )
        ):
            raise ValueError(f"private artifact or audio reference at {path}")


def _validate_bounded_payload(payload: Mapping[str, Any], *, label: str) -> None:
    tokens = payload.get("transcription", {}).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{label} has no realized output sequence")
    for token in tokens:
        candidates = token.get("top_tokens")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
            raise ValueError(f"{label} HEAD candidates are not bounded to top five")
    for stream_name in ("encoder", "decoder"):
        cells = payload.get(stream_name, {}).get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"{label} {stream_name} matrix is incomplete")
        for row in cells:
            for cell in row:
                candidates = cell.get("top_tokens")
                if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
                    raise ValueError(
                        f"{label} {stream_name} candidates are not bounded to top five"
                    )
                phones = cell.get("phone_signatures")
                if phones is not None and (
                    not isinstance(phones, list) or not 1 <= len(phones) <= 5
                ):
                    raise ValueError(f"{label} phone candidates are not bounded")


def _compose_report(
    report: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    composed = copy.deepcopy(dict(report))
    analysis = condition.get("analysis")
    if isinstance(analysis, Mapping):
        composed["payload"].update(copy.deepcopy(dict(analysis)))
    return composed


def _validate_published_report(report: Mapping[str, Any]) -> None:
    replay = report.get("recorded_intervention_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("published report has no replay extension")
    if set(replay) != {
        "schema_id",
        "schema_version",
        "mode",
        "default_condition",
        "conditions",
    }:
        raise ValueError("published replay root has unexpected fields")
    if (
        replay.get("schema_id") != REPLAY_SCHEMA_ID
        or replay.get("schema_version") != SCHEMA_VERSION
        or replay.get("mode") != REPLAY_MODE
        or replay.get("default_condition") != "baseline"
    ):
        raise ValueError("published replay has an invalid envelope")
    conditions = replay.get("conditions")
    if not isinstance(conditions, list) or [
        item.get("id") for item in conditions
    ] != list(CONDITION_IDS):
        raise ValueError("published replay condition order changed")
    _validate_replay_safe(replay)

    base_geometry = _encoder_geometry(report["payload"])
    for condition in conditions:
        condition_id = str(condition["id"])
        expected_fields = set(CONDITION_PUBLIC_FIELDS)
        if condition_id != "baseline":
            expected_fields.add("analysis")
        _require_exact_fields(
            condition,
            expected_fields,
            label=f"published {condition_id} condition",
        )
        expected = EXPECTED_CONDITIONS[condition_id]
        generated = _validate_generated(
            condition["generated"], label=f"published {condition_id}.generated"
        )
        evidence = _validate_evidence(
            condition["evidence"], label=f"published {condition_id}.evidence"
        )
        _validate_method(condition["method"], label=f"published {condition_id}.method")
        _validate_schedule(condition["schedule"], condition_id=condition_id)
        if (
            generated["text"] != expected["text"]
            or generated["token_ids"] != expected["token_ids"]
            or generated["target_match"] is not expected["target_match"]
            or evidence["tier"] != expected["evidence_tier"]
            or condition["layers"] != expected["layers"]
            or not _close(condition["budget_fraction"], expected["budget_fraction"])
            or not _close(condition["coefficient_scale"], expected["coefficient_scale"])
        ):
            raise ValueError(f"published {condition_id} metadata changed")
        composed = _compose_report(report, condition)
        if condition_id == "baseline" and "analysis" in condition:
            raise ValueError("baseline replay must reuse the report payload")
        if condition_id != "baseline" and set(condition.get("analysis", {})) != set(
            REPLAY_ANALYSIS_FIELDS
        ):
            raise ValueError(f"{condition_id} replay analysis is incomplete")
        if _encoder_geometry(composed["payload"]) != base_geometry:
            raise ValueError(f"{condition_id} encoder geometry changed")
        exporter._validate_matrix(composed)
        _validate_bounded_payload(composed["payload"], label=condition_id)


def _existing_entry(
    entries: Mapping[str, Mapping[str, Any]],
    *,
    sample_id: str,
    site_root: Path,
) -> dict[str, Any]:
    entry = entries.get(sample_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"existing ASR manifest has no {sample_id} entry")
    report_path = site_root / str(entry.get("report_url", "")).removeprefix(
        "/audio-jacobian-lens/"
    )
    filter_reference = entry.get("character_length_filter_cache")
    if not isinstance(filter_reference, Mapping):
        raise ValueError(f"existing {sample_id} entry has no filter sidecar")
    filter_path = site_root / str(filter_reference.get("url", "")).removeprefix(
        "/audio-jacobian-lens/"
    )
    if (
        not report_path.is_file()
        or exporter._sha256(report_path) != entry.get("sha256")
        or report_path.stat().st_size != entry.get("bytes")
        or not filter_path.is_file()
        or exporter._sha256(filter_path) != filter_reference.get("sha256")
        or filter_path.stat().st_size != filter_reference.get("bytes")
    ):
        raise ValueError(f"existing {sample_id} files are not manifest-bound")
    return copy.deepcopy(dict(entry))


def _resolve_site_asset(site_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"site manifest path escapes the site root: {relative_path}")
    resolved = (site_root / relative).resolve()
    if not resolved.is_relative_to(site_root):
        raise ValueError(f"site manifest path escapes the site root: {relative_path}")
    return resolved


def _validate_site_manifest_hashes(
    payload: Mapping[str, Any],
    *,
    site_root: Path,
    staged_overrides: Mapping[str, Path] | None = None,
) -> None:
    recorded = payload.get("sha256")
    if not isinstance(recorded, Mapping) or not recorded:
        raise ValueError("site manifest has no SHA-256 map")
    overrides = staged_overrides or {}
    for relative_path, expected in recorded.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ValueError("site manifest SHA-256 entries must be strings")
        asset = overrides.get(relative_path)
        if asset is None:
            asset = _resolve_site_asset(site_root, relative_path)
        if not asset.is_file():
            raise ValueError(f"site manifest asset is missing: {relative_path}")
        if exporter._sha256(asset) != expected:
            raise ValueError(f"site manifest hash is stale: {relative_path}")


def _stage_site_manifest(
    *, site_root: Path, destination: Path, staged_asr_manifest: Path
) -> dict[str, Any]:
    path = site_root / "site-manifest.json"
    payload = _load(path, label="site manifest")
    recorded = payload.get("sha256")
    if not isinstance(recorded, dict):
        raise ValueError("site manifest has no SHA-256 map")
    asr_manifest_key = "explorer/data/asr/manifest.json"
    if asr_manifest_key not in recorded:
        raise ValueError("site manifest does not bind the ASR manifest")
    overrides = {asr_manifest_key: staged_asr_manifest}
    for relative_path in list(recorded):
        asset = overrides.get(relative_path)
        if asset is None:
            asset = _resolve_site_asset(site_root, relative_path)
        if not asset.is_file():
            raise ValueError(f"site manifest asset is missing: {relative_path}")
        recorded[relative_path] = exporter._sha256(asset)
    payload["media_policy"] = (
        "The ASR explorer includes nine CC BY 4.0 LibriSpeech inputs and the "
        "separately attributed CC BY 4.0 Laurel/Yanny Audio S7 source; the Speech "
        "explorer retains its ten LibriSpeech inputs. Generated audio is excluded."
    )
    payload["payload_policy"] = (
        "Each family manifest hash-pins ten reduced reports. The Laurel/Yanny ASR "
        "report adds three recorded, non-interpolated conditions with bounded "
        "encoder, decoder, and HEAD views; residual edits, coefficient audits, "
        "prototype tensors, private paths, and embedded audio are excluded."
    )
    _write_canonical_json(destination, payload)
    _validate_site_manifest_hashes(
        payload, site_root=site_root, staged_overrides=overrides
    )
    return payload


def _validate_manifest_entry_file(
    entry: Mapping[str, Any],
    *,
    site_root: Path,
    staged_overrides: Mapping[str, Path],
) -> None:
    report_url = entry.get("report_url")
    if not isinstance(report_url, str) or not report_url.startswith(
        "/audio-jacobian-lens/"
    ):
        raise ValueError("ASR report has an invalid public URL")
    report_relative = report_url.removeprefix("/audio-jacobian-lens/")
    report_path = staged_overrides.get(report_relative)
    if report_path is None:
        report_path = _resolve_site_asset(site_root, report_relative)
    if (
        not report_path.is_file()
        or exporter._sha256(report_path) != entry.get("sha256")
        or report_path.stat().st_size != entry.get("bytes")
    ):
        raise ValueError(f"ASR report is not manifest-bound: {entry.get('id')}")

    filter_reference = entry.get("character_length_filter_cache")
    if not isinstance(filter_reference, Mapping):
        raise ValueError(f"ASR report has no filter sidecar: {entry.get('id')}")
    filter_url = filter_reference.get("url")
    if not isinstance(filter_url, str) or not filter_url.startswith(
        "/audio-jacobian-lens/"
    ):
        raise ValueError("ASR filter sidecar has an invalid public URL")
    filter_relative = filter_url.removeprefix("/audio-jacobian-lens/")
    filter_path = staged_overrides.get(filter_relative)
    if filter_path is None:
        filter_path = _resolve_site_asset(site_root, filter_relative)
    if (
        not filter_path.is_file()
        or exporter._sha256(filter_path) != filter_reference.get("sha256")
        or filter_path.stat().st_size != filter_reference.get("bytes")
    ):
        raise ValueError(f"ASR filter is not manifest-bound: {entry.get('id')}")


def _validate_staged_publication(
    *,
    site_root: Path,
    staged_files: Mapping[str, Path],
    deleted_paths: set[str],
    rebuilt_manifest: Mapping[str, Any],
    expected_audio_sha256: str,
) -> None:
    required = {
        f"audio/{SAMPLE_SLUG}.mp3",
        f"explorer/data/asr/{SAMPLE_SLUG}.json",
        f"explorer/data/asr/{SAMPLE_SLUG}.filters.json",
        "explorer/data/asr/manifest.json",
        "site-manifest.json",
    }
    if set(staged_files) != required:
        raise ValueError("publication transaction has an unexpected staged-file set")
    expected_deleted = {
        f"explorer/data/asr/{DISPLACED_SLUG}.json",
        f"explorer/data/asr/{DISPLACED_SLUG}.filters.json",
    }
    if deleted_paths != expected_deleted:
        raise ValueError("publication transaction has an unexpected deletion set")
    staged_audio = staged_files[f"audio/{SAMPLE_SLUG}.mp3"]
    if exporter._sha256(staged_audio) != expected_audio_sha256:
        raise ValueError("staged Laurel/Yanny audio hash changed")

    report = _load(
        staged_files[f"explorer/data/asr/{SAMPLE_SLUG}.json"],
        label="staged Laurel/Yanny report",
    )
    filter_cache = _load(
        staged_files[f"explorer/data/asr/{SAMPLE_SLUG}.filters.json"],
        label="staged Laurel/Yanny filter cache",
    )
    _validate_published_report(report)
    exporter._validate_filter_cache(filter_cache, report)

    staged_manifest = _load(
        staged_files["explorer/data/asr/manifest.json"], label="staged ASR manifest"
    )
    if staged_manifest != _canonical_json_value(rebuilt_manifest):
        raise ValueError("staged ASR manifest differs from the validated rebuild")
    reports = staged_manifest.get("reports")
    if (
        not isinstance(reports, list)
        or staged_manifest.get("report_count") != len(reports)
        or any(entry.get("id") == f"asr-{DISPLACED_SLUG}" for entry in reports)
    ):
        raise ValueError("staged ASR manifest report set is invalid")
    for entry in reports:
        if not isinstance(entry, Mapping):
            raise ValueError("staged ASR manifest contains a non-object report")
        _validate_manifest_entry_file(
            entry, site_root=site_root, staged_overrides=staged_files
        )

    site_manifest = _load(
        staged_files["site-manifest.json"], label="staged site manifest"
    )
    _validate_site_manifest_hashes(
        site_manifest, site_root=site_root, staged_overrides=staged_files
    )


def _validate_committed_publication(
    *,
    site_root: Path,
    expected_files: Mapping[str, tuple[str, int]],
    deleted_paths: set[str],
) -> None:
    for relative_path, (expected_hash, expected_bytes) in expected_files.items():
        live = _resolve_site_asset(site_root, relative_path)
        if (
            not live.is_file()
            or exporter._sha256(live) != expected_hash
            or live.stat().st_size != expected_bytes
        ):
            raise ValueError(f"committed publication changed: {relative_path}")
    for relative_path in deleted_paths:
        if _resolve_site_asset(site_root, relative_path).exists():
            raise ValueError(f"committed deletion failed: {relative_path}")
    report = _load(
        site_root / "explorer" / "data" / "asr" / f"{SAMPLE_SLUG}.json",
        label="committed Laurel/Yanny report",
    )
    _validate_published_report(report)
    site_manifest = _load(site_root / "site-manifest.json", label="site manifest")
    _validate_site_manifest_hashes(site_manifest, site_root=site_root)


def _promote_publication_transaction(
    *,
    site_root: Path,
    staged_files: Mapping[str, Path],
    deleted_paths: set[str],
) -> None:
    """Promote every public mutation as one rollback-capable transaction."""

    replacements = sorted(staged_files.items(), key=lambda item: item[0])
    deletions = sorted(deleted_paths)
    expected_files = {
        relative_path: (exporter._sha256(staged), staged.stat().st_size)
        for relative_path, staged in replacements
    }
    with tempfile.TemporaryDirectory(
        prefix=".asr-replay-backup-", dir=site_root.parent
    ) as temporary:
        backup_root = Path(temporary)
        prepared: list[tuple[str, bool]] = []
        try:
            for relative_path, staged in replacements:
                live = _resolve_site_asset(site_root, relative_path)
                live.parent.mkdir(parents=True, exist_ok=True)
                existed = live.is_file()
                if existed:
                    backup = backup_root / relative_path
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(live, backup)
                prepared.append((relative_path, existed))
                os.replace(staged, live)
            for relative_path in deletions:
                live = _resolve_site_asset(site_root, relative_path)
                existed = live.is_file()
                if existed:
                    backup = backup_root / relative_path
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(live, backup)
                prepared.append((relative_path, existed))
                live.unlink(missing_ok=True)
            _validate_committed_publication(
                site_root=site_root,
                expected_files=expected_files,
                deleted_paths=deleted_paths,
            )
        except Exception as error:
            rollback_errors: list[str] = []
            for relative_path, existed in reversed(prepared):
                live = _resolve_site_asset(site_root, relative_path)
                try:
                    if existed:
                        backup = backup_root / relative_path
                        live.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(backup, live)
                    else:
                        live.unlink(missing_ok=True)
                except OSError as rollback_error:  # pragma: no cover - catastrophic IO
                    rollback_errors.append(f"{relative_path}: {rollback_error}")
            if rollback_errors:  # pragma: no cover - catastrophic IO
                raise RuntimeError(
                    "publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from error
            raise


def publish(
    *,
    capture_path: Path,
    site_root: Path,
    catalog_path: Path,
    samples_dir: Path,
) -> dict[str, Any]:
    """Publish the capture and return the rebuilt ASR manifest."""

    site_root = site_root.resolve()
    capture = _load(capture_path, label="recorded steering capture")
    conditions = _validate_capture(capture)
    catalog = load_static_explorer_catalog(catalog_path)
    samples = catalog.audio_samples_for_family("asr")
    matches = [sample for sample in samples if sample.slug == SAMPLE_SLUG]
    if len(matches) != 1:
        raise ValueError("ASR catalog must contain exactly one Laurel/Yanny sample")
    sample = matches[0]
    if sample.source_override is None:
        raise ValueError(
            "Laurel/Yanny catalog sample has no per-report rights metadata"
        )

    asr_dir = site_root / "explorer" / "data" / "asr"
    manifest_path = asr_dir / "manifest.json"
    manifest = _load(manifest_path, label="ASR Explorer manifest")
    if (
        manifest.get("schema_id") != exporter.MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != exporter.SCHEMA_VERSION
        or manifest.get("family") != "asr"
        or manifest.get("mode") != "static_cached_explorer"
    ):
        raise ValueError("ASR Explorer manifest has an unsupported envelope")
    _validate_capture_provenance(capture, manifest, expected_audio_sha256=sample.sha256)

    source_audio = samples_dir / sample.filename
    exporter._validate_sample_file(source_audio, sample)

    source_payload = exporter._source_payload(sample, catalog.audio_source)
    # This source is licensed by the original post rather than by the default
    # LibriSpeech collection, so keep the distinction explicit in public data.
    source_payload["rights_status"] = "attributed_under_source_page_license"

    raw_baseline = conditions["baseline"]["analysis"]
    report = {
        "schema_id": exporter.SCHEMA_ID,
        "schema_version": exporter.SCHEMA_VERSION,
        "family": "asr",
        "example_id": f"asr-{SAMPLE_SLUG}",
        "title": sample.title,
        "source": source_payload,
        "cache_policy": {
            "inference": "precomputed",
            "waveform": "1024-point uniform preview of the attributed input",
            "generated_audio_included": False,
            "character_length_bucket_cache_included": True,
            "unsupported_static_controls": [],
        },
        "payload": exporter._reduce_payload(raw_baseline),
        "recorded_intervention_replay": {
            "schema_id": REPLAY_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "mode": REPLAY_MODE,
            "default_condition": "baseline",
            "conditions": [
                _public_condition(
                    conditions[condition_id],
                    include_analysis=condition_id != "baseline",
                )
                for condition_id in CONDITION_IDS
            ],
        },
    }

    filter_cache = exporter._filter_cache(raw_baseline, example_id=f"asr-{SAMPLE_SLUG}")
    exporter._validate_safe(filter_cache)
    exporter._validate_filter_cache(filter_cache, report)
    exporter._validate_safe(report)
    _validate_published_report(report)

    existing_reports = manifest.get("reports")
    if not isinstance(existing_reports, list):
        raise ValueError("ASR Explorer manifest has no report list")
    by_id = {
        str(entry.get("id")): entry
        for entry in existing_reports
        if isinstance(entry, Mapping)
    }
    allowed_old_ids = {
        f"asr-{item.slug}" for item in samples if item.slug != SAMPLE_SLUG
    }
    if f"asr-{DISPLACED_SLUG}" in allowed_old_ids:
        raise ValueError("ASR catalog still contains the displaced impossible sample")

    with tempfile.TemporaryDirectory(
        prefix=".asr-replay-publish-", dir=site_root.parent
    ) as temporary:
        stage = Path(temporary) / "stage"
        audio_path = stage / "audio" / sample.filename
        report_path = stage / "explorer" / "data" / "asr" / f"{SAMPLE_SLUG}.json"
        filter_path = (
            stage / "explorer" / "data" / "asr" / f"{SAMPLE_SLUG}.filters.json"
        )
        staged_manifest_path = stage / "explorer" / "data" / "asr" / "manifest.json"
        staged_site_manifest_path = stage / "site-manifest.json"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_audio, audio_path)
        _write_canonical_json(filter_path, filter_cache, compact=True)
        filter_reference = exporter._filter_manifest(
            family="asr", sample=sample, filter_path=filter_path
        )
        report["cache_policy"]["character_length_filter_cache"] = filter_reference
        _write_canonical_json(report_path, report)
        new_entry = exporter._manifest_entry(
            family="asr",
            sample=sample,
            report_path=report_path,
            filter_manifest=filter_reference,
        )

        ordered_entries: list[dict[str, Any]] = []
        for catalog_sample in samples:
            sample_id = f"asr-{catalog_sample.slug}"
            if catalog_sample.slug == SAMPLE_SLUG:
                ordered_entries.append(new_entry)
            else:
                ordered_entries.append(
                    _existing_entry(by_id, sample_id=sample_id, site_root=site_root)
                )
        rebuilt = copy.deepcopy(manifest)
        rebuilt["report_count"] = len(ordered_entries)
        rebuilt["catalog"] = {
            "schema_id": exporter.CATALOG_SCHEMA_ID,
            "schema_version": exporter.CATALOG_SCHEMA_VERSION,
            "sha256": exporter._sha256(catalog_path),
            "reports_per_family": catalog.reports_per_family,
        }
        rebuilt["provenance"] = exporter._detailed_provenance(
            manifest["provenance"], catalog, family="asr"
        )
        rebuilt["reports"] = ordered_entries
        exporter._validate_safe(rebuilt)
        _write_canonical_json(staged_manifest_path, rebuilt)
        _stage_site_manifest(
            site_root=site_root,
            destination=staged_site_manifest_path,
            staged_asr_manifest=staged_manifest_path,
        )
        staged_files = {
            f"audio/{sample.filename}": audio_path,
            f"explorer/data/asr/{SAMPLE_SLUG}.json": report_path,
            f"explorer/data/asr/{SAMPLE_SLUG}.filters.json": filter_path,
            "explorer/data/asr/manifest.json": staged_manifest_path,
            "site-manifest.json": staged_site_manifest_path,
        }
        # These names are deliberately scoped to the ASR family directory. The
        # Speech family keeps its own impossible report and shared input audio.
        deleted_paths = {
            f"explorer/data/asr/{DISPLACED_SLUG}.json",
            f"explorer/data/asr/{DISPLACED_SLUG}.filters.json",
        }
        _validate_staged_publication(
            site_root=site_root,
            staged_files=staged_files,
            deleted_paths=deleted_paths,
            rebuilt_manifest=rebuilt,
            expected_audio_sha256=sample.sha256,
        )
        _promote_publication_transaction(
            site_root=site_root,
            staged_files=staged_files,
            deleted_paths=deleted_paths,
        )
    return rebuilt


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Publish the recorded Laurel/Yanny ASR Explorer replay"
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("site_root", type=Path)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "data" / "static_explorer_catalog_v2.json",
    )
    parser.add_argument("--samples-dir", type=Path, default=root / "samples")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = publish(
        capture_path=args.capture,
        site_root=args.site_root,
        catalog_path=args.catalog,
        samples_dir=args.samples_dir,
    )
    print(
        "published recorded ASR replay: "
        f"{manifest['report_count']} reports; {SAMPLE_SLUG} replaces {DISPLACED_SLUG}"
    )


if __name__ == "__main__":
    main()
