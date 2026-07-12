#!/usr/bin/env python3
"""Export rights-safe, reduced ASR and LFM explorer payloads.

The exporter calls the local single-worker servers sequentially. It builds each
published payload from explicit allowlists, never copies generated audio, and
fails if an ephemeral analysis identifier or embedded audio URI survives.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from jlens.static_explorer_catalog import (
        CATALOG_SCHEMA_ID,
        CATALOG_SCHEMA_VERSION,
        StaticAudioSample,
        StaticAudioSource,
        StaticExplorerCatalog,
        load_static_explorer_catalog,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jlens.static_explorer_catalog import (  # noqa: E402
        CATALOG_SCHEMA_ID,
        CATALOG_SCHEMA_VERSION,
        StaticAudioSample,
        StaticAudioSource,
        StaticExplorerCatalog,
        load_static_explorer_catalog,
    )

SCHEMA_ID = "audio-jacobian-lens.cached-explorer-report"
MANIFEST_SCHEMA_ID = "audio-jacobian-lens.cached-explorer-manifest"
FILTER_SCHEMA_ID = "audio-jacobian-lens.cached-explorer-filter-cache"
SCHEMA_VERSION = 1
WAVEFORM_PREVIEW_POINTS = 1024
FORBIDDEN_KEYS = {
    "analysis_id",
    "parent_analysis_id",
    "branch_analysis_id",
    "model_input_wav",
    "model_output_wav",
    "generated_audio",
    "output_waveform",
}

METADATA_FIELDS = {
    "artifact_generation",
    "backend",
    "capabilities",
    "candidate_rank_semantics",
    "capture_convention",
    "decoder_dim",
    "decoder_layers",
    "decoder_token_length_filter",
    "deferred_heads",
    "device",
    "display_vocabulary",
    "encoder_dim",
    "encoder_layers",
    "encoder_seconds_per_position",
    "encoder_token_length_filter",
    "estimator",
    "generation",
    "generation_diagnostics",
    "input_sample_rate",
    "language_dim",
    "language_layers",
    "lens_examples",
    "message",
    "model_config_fingerprint",
    "model_family",
    "model_fingerprint",
    "model_id",
    "model_revision",
    "output_sample_rate",
    "projection",
    "quantization",
    "ready",
    "runtime_versions",
    "schema_version",
    "serving_generation",
    "stream_labels",
    "streams",
    "target_head",
    "tokenizer_fingerprint",
    "vocab_size",
    "warnings",
    "weights_fingerprint",
}

CANDIDATE_FIELDS = {
    "display_vocabulary_denominator",
    "display_vocabulary_rank",
    "full_vocabulary_denominator",
    "full_vocabulary_rank",
    "id",
    "log_probability",
    "probability",
    "rank",
    "rank_denominator",
    "rank_space",
    "rank_tie_policy",
    "score",
    "score_kind",
    "text",
    "vocabulary_filter",
}

TOKEN_FIELDS = CANDIDATE_FIELDS | {
    "candidate_space",
    "end_seconds",
    "entropy",
    "is_special",
    "start_seconds",
}

CELL_FIELDS = {
    "candidate_space",
    "position_index",
    "realized_token_alignment",
    "realized_token_position",
    "selected_score",
    "time_window",
}

STREAM_FIELDS = {
    "layers",
    "pooling",
    "positions",
    "projection_rank",
    "realized_token_alignment",
    "score_kind",
    "stream_kind",
    "target_layer",
    "time_bins",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick(source: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {key: source[key] for key in fields if key in source}


def _candidate(source: Mapping[str, Any]) -> dict[str, Any]:
    return _pick(source, CANDIDATE_FIELDS)


def _token(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(source, TOKEN_FIELDS)
    result["top_tokens"] = [_candidate(item) for item in source.get("top_tokens", [])]
    return result


def _cell(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(source, CELL_FIELDS)
    result["top_tokens"] = [_candidate(item) for item in source.get("top_tokens", [])]
    realized = source.get("realized_token")
    if isinstance(realized, Mapping):
        result["realized_token"] = _candidate(realized)
    return result


def _stream(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not source:
        return {"layers": [], "positions": [], "cells": []}
    result = _pick(source, STREAM_FIELDS)
    result["cells"] = [
        [_cell(cell) for cell in layer_cells] for layer_cells in source.get("cells", [])
    ]
    return result


def _waveform_preview(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "kind": "uniform_samples",
            "source_sample_count": 0,
            "values": [],
        }
    count = min(WAVEFORM_PREVIEW_POINTS, len(numeric))
    if count == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(numeric) - 1) / (count - 1)) for index in range(count)
        ]
    return {
        "kind": "uniform_samples",
        "source_sample_count": len(numeric),
        "values": [numeric[index] for index in indices],
    }


def _reduce_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    audio = source.get("audio") or {}
    transcription = source.get("transcription") or {}
    metadata = _pick(source.get("metadata") or {}, METADATA_FIELDS)
    if isinstance(metadata.get("capabilities"), dict):
        metadata["capabilities"] = {
            key: value
            for key, value in metadata["capabilities"].items()
            if key != "generated_audio"
        }
    return {
        "metadata": metadata,
        "audio": {
            "duration_seconds": audio.get("duration_seconds"),
            "model_input_format": audio.get("model_input_format"),
            "waveform_preview": _waveform_preview(audio.get("waveform") or []),
        },
        "transcription": {
            "semantic_role": transcription.get("semantic_role"),
            "text": transcription.get("text"),
            "timing_quality": transcription.get("timing_quality"),
            "timing_source": transcription.get("timing_source"),
            "tokens": [_token(token) for token in transcription.get("tokens", [])],
        },
        "encoder": _stream(source.get("encoder")),
        "decoder": _stream(source.get("decoder")),
    }


def _filter_stream(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not source:
        return {"layers": [], "cells": []}
    retained_layers: list[Any] = []
    retained_cells: list[list[dict[str, Any]]] = []
    for layer, layer_cells in zip(
        source.get("layers", []), source.get("cells", []), strict=True
    ):
        if not any("top_tokens_by_length" in cell for cell in layer_cells):
            continue
        cells: list[dict[str, Any]] = []
        for cell in layer_cells:
            buckets = cell.get("top_tokens_by_length")
            if not isinstance(buckets, dict):
                raise ValueError(f"layer {layer} has a partial filter cache")
            realized_rank_by_length = cell.get("realized_rank_by_max_length")
            if (
                not isinstance(realized_rank_by_length, dict)
                or not realized_rank_by_length
            ):
                raise ValueError(
                    f"layer {layer} has no exact filtered realized-token ranks"
                )
            cells.append(
                {
                    "position_index": cell.get("position_index"),
                    "time_window": cell.get("time_window"),
                    "top_tokens_by_length": {
                        str(length): [_candidate(item) for item in candidates]
                        for length, candidates in buckets.items()
                    },
                    "realized_rank_by_max_length": {
                        str(length): (None if rank is None else int(rank))
                        for length, rank in realized_rank_by_length.items()
                    },
                }
            )
        retained_layers.append(layer)
        retained_cells.append(cells)
    return {"layers": retained_layers, "cells": retained_cells}


def _filter_cache(source: Mapping[str, Any], *, example_id: str) -> dict[str, Any]:
    return {
        "schema_id": FILTER_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": "asr",
        "example_id": example_id,
        "merge_semantics": {
            "bucket_kind": "exact_decoded_character_length",
            "maximum_filter_operator": "less_than_or_equal",
            "merge": (
                "Merge exact-length buckets whose numeric key is no greater "
                "than the requested limit, sort by score descending, and "
                "rerank by 1 + count of strictly greater scores."
            ),
            "tie_policy": "1_plus_count_strictly_greater",
        },
        "streams": {
            "encoder": _filter_stream(source.get("encoder")),
            "decoder": _filter_stream(source.get("decoder")),
        },
    }


def _multipart(audio_path: Path, fields: Mapping[str, str]) -> tuple[bytes, str]:
    boundary = "audio-jacobian-lens-static-export-v1"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    content_type = (
        mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="audio"; '
                f'filename="{audio_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            audio_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _analyze(url: str, audio_path: Path, fields: Mapping[str, str]) -> dict[str, Any]:
    body, content_type = _multipart(audio_path, fields)
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Accept": "application/json", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {error.code}: {detail}") from error
    if not isinstance(payload, dict) or "transcription" not in payload:
        raise RuntimeError(f"{url} returned an invalid analysis payload")
    return payload


def _validate_safe(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS or key.endswith("_analysis_id"):
                raise ValueError(f"forbidden field {path}.{key}")
            _validate_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("data:audio/"):
        raise ValueError(f"embedded audio URI at {path}")


def _validate_exact_rank_candidate(
    candidate: Mapping[str, Any] | None,
    *,
    label: str,
    expected_id: Any | None = None,
    require_score: bool = False,
) -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{label} has no exact realized-token provenance")
    required = {
        "id",
        "text",
        "rank",
        "rank_denominator",
        "rank_space",
        "rank_tie_policy",
        "score_kind",
    }
    if require_score:
        required.add("score")
    missing = sorted(required - candidate.keys())
    if missing:
        raise ValueError(
            f"{label} realized-token provenance lacks {', '.join(missing)}"
        )
    if expected_id is not None and candidate["id"] != expected_id:
        raise ValueError(f"{label} realized-token ID does not match the output token")
    try:
        rank = int(candidate["rank"])
        denominator = int(candidate["rank_denominator"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} realized-token rank is invalid") from error
    if rank < 1 or denominator < rank:
        raise ValueError(f"{label} realized-token rank is outside its rank space")
    if not str(candidate["rank_space"]).strip():
        raise ValueError(f"{label} realized-token rank space is empty")
    if candidate["rank_tie_policy"] != "1_plus_count_strictly_greater":
        raise ValueError(f"{label} realized-token tie policy is unsupported")


def _aligned_transcription_token(
    tokens: list[Mapping[str, Any]], time_window: Mapping[str, Any]
) -> tuple[int, Mapping[str, Any]]:
    """Match the backend's overlap-first encoder/token synchronization."""
    try:
        window_start = float(time_window["start_seconds"])
        window_end = float(time_window["end_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ASR encoder cell has no usable time window") from error
    if not math.isfinite(window_start) or not math.isfinite(window_end):
        raise ValueError("ASR encoder cell has a non-finite time window")
    if window_end < window_start:
        raise ValueError("ASR encoder cell has a reversed time window")
    midpoint = (window_start + window_end) / 2
    choices: list[tuple[tuple[float, float, int], int, Mapping[str, Any]]] = []
    for index, token in enumerate(tokens):
        start = token.get("start_seconds")
        end = token.get("end_seconds")
        if start is None or end is None:
            continue
        try:
            token_start = float(start)
            token_end = float(end)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(token_start)
            or not math.isfinite(token_end)
            or token_end < token_start
        ):
            continue
        overlap = max(
            0.0,
            min(window_end, token_end) - max(window_start, token_start),
        )
        midpoint_distance = abs(midpoint - (token_start + token_end) / 2)
        choices.append(((-overlap, midpoint_distance, index), index, token))
    if not choices:
        raise ValueError("ASR encoder cells cannot be aligned to untimed output tokens")
    _, best_index, best_token = min(choices, key=lambda choice: choice[0])
    return best_index, best_token


def _validate_encoder_alignment_provenance(
    alignment: Any,
    *,
    time_window: Mapping[str, Any],
    token: Mapping[str, Any],
) -> None:
    if not isinstance(alignment, Mapping):
        raise ValueError("asr encoder cell has no realized-token alignment provenance")
    required = {
        "match",
        "window_midpoint_seconds",
        "token_start_seconds",
        "token_end_seconds",
        "overlap_seconds",
        "overlap_fraction_of_window",
    }
    if required - alignment.keys():
        raise ValueError("asr encoder cell has incomplete alignment provenance")
    window_start = float(time_window["start_seconds"])
    window_end = float(time_window["end_seconds"])
    token_start = float(token["start_seconds"])
    token_end = float(token["end_seconds"])
    overlap = max(
        0.0,
        min(window_end, token_end) - max(window_start, token_start),
    )
    duration = max(0.0, window_end - window_start)
    expected = {
        "window_midpoint_seconds": (window_start + window_end) / 2,
        "token_start_seconds": token_start,
        "token_end_seconds": token_end,
        "overlap_seconds": overlap,
        "overlap_fraction_of_window": overlap / duration if duration > 0 else 0.0,
    }
    if alignment.get("match") != ("overlapping" if overlap > 0 else "nearest"):
        raise ValueError("asr encoder cell has an invalid alignment match kind")
    for field, value in expected.items():
        try:
            recorded = float(alignment[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"asr encoder alignment {field} is invalid") from error
        if not math.isclose(recorded, value, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"asr encoder alignment {field} does not match timing")


def _validate_matrix(report: Mapping[str, Any]) -> None:
    payload = report["payload"]
    family = report.get("family")
    transcription_tokens = payload.get("transcription", {}).get("tokens", [])
    if family == "asr":
        alignment_metadata = payload.get("encoder", {}).get("realized_token_alignment")
        if (
            not isinstance(alignment_metadata, Mapping)
            or alignment_metadata.get("method") != "maximum_token_interval_overlap"
        ):
            raise ValueError("asr encoder stream has no alignment-method provenance")
    for stream_name in ("encoder", "decoder"):
        stream = payload[stream_name]
        layers = stream.get("layers", [])
        cells = stream.get("cells", [])
        if len(cells) != len(layers):
            raise ValueError(f"{stream_name} layer/cell matrix mismatch")
        if not cells:
            if family == "speech" and stream_name == "decoder":
                raise ValueError("speech decoder matrix is empty")
            continue
        widths = {len(row) for row in cells}
        if len(widths) != 1:
            raise ValueError(f"{stream_name} matrix is ragged")
        if stream_name == "decoder" and family in {"asr", "speech"}:
            if widths != {len(transcription_tokens)}:
                raise ValueError(f"{family} decoder/output-token width mismatch")
        for layer_index, row in enumerate(cells):
            for position, cell in enumerate(row):
                if not cell.get("top_tokens"):
                    raise ValueError(f"{stream_name} cell has no candidates")
                if family in {"asr", "speech"} and stream_name == "decoder":
                    if position >= len(transcription_tokens):
                        raise ValueError(
                            f"{family} decoder is wider than its output tokens"
                        )
                    _validate_exact_rank_candidate(
                        cell.get("realized_token"),
                        label=(
                            f"{family} decoder layer {layer_index}, position {position}"
                        ),
                        expected_id=transcription_tokens[position].get("id"),
                        require_score=True,
                    )
                if family == "asr" and stream_name == "encoder":
                    token_index, token = _aligned_transcription_token(
                        transcription_tokens, cell.get("time_window") or {}
                    )
                    if cell.get("realized_token_position") != token_index:
                        raise ValueError(
                            "asr encoder realized-token position does not match "
                            "overlap-first output-token synchronization"
                        )
                    _validate_encoder_alignment_provenance(
                        cell.get("realized_token_alignment"),
                        time_window=cell.get("time_window") or {},
                        token=token,
                    )
                    _validate_exact_rank_candidate(
                        cell.get("realized_token"),
                        label=(
                            f"asr encoder layer {layer_index}, position {position} "
                            f"(aligned output token {token_index})"
                        ),
                        expected_id=token.get("id"),
                        require_score=True,
                    )
    if family in {"asr", "speech"}:
        for position, token in enumerate(transcription_tokens):
            _validate_exact_rank_candidate(
                token,
                label=f"{family} HEAD position {position}",
                expected_id=token.get("id"),
            )


def _validate_filter_cache(cache: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if cache["example_id"] != report["example_id"]:
        raise ValueError("filter cache/report example mismatch")
    payload = report["payload"]
    denominators_by_length = (
        payload.get("metadata", {})
        .get("display_vocabulary", {})
        .get("maximum_decoded_character_length_counts", {})
    )
    if not isinstance(denominators_by_length, Mapping) or not denominators_by_length:
        raise ValueError("report has no cumulative character-filter denominators")
    for stream_name in ("encoder", "decoder"):
        cache_stream = cache["streams"][stream_name]
        report_stream = payload[stream_name]
        report_width = (
            len(report_stream.get("cells", [])[0]) if report_stream.get("cells") else 0
        )
        report_layers = report_stream.get("layers", [])
        for layer, row in zip(
            cache_stream["layers"], cache_stream["cells"], strict=True
        ):
            if layer not in report_layers:
                raise ValueError(f"filter layer {layer} missing from {stream_name}")
            if len(row) != report_width:
                raise ValueError(f"filter/report width mismatch in {stream_name}")
            for position, cell in enumerate(row):
                if cell["position_index"] != position:
                    raise ValueError(
                        f"filter cell coordinate mismatch in {stream_name}"
                    )
                if not cell["top_tokens_by_length"]:
                    raise ValueError(f"empty filter buckets in {stream_name}")
                realized_rank_by_length = cell.get("realized_rank_by_max_length")
                if (
                    not isinstance(realized_rank_by_length, Mapping)
                    or not realized_rank_by_length
                ):
                    raise ValueError(
                        f"missing exact filtered realized ranks in {stream_name}"
                    )
                report_layer_index = report_layers.index(layer)
                base_cell = report_stream["cells"][report_layer_index][position]
                target_filter = base_cell.get("realized_token", {}).get(
                    "vocabulary_filter", {}
                )
                target_length = target_filter.get("decoded_character_length")
                target_eligible = target_filter.get("display_lexical_eligible")
                if not isinstance(target_eligible, bool):
                    raise ValueError(
                        f"realized token has no lexical eligibility in {stream_name}"
                    )
                if set(realized_rank_by_length) != set(denominators_by_length):
                    raise ValueError(
                        f"filtered realized ranks do not cover every limit in {stream_name}"
                    )
                for limit, rank in realized_rank_by_length.items():
                    try:
                        numeric_limit = int(limit)
                        denominator = int(denominators_by_length[str(limit)])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid realized-rank filter limit in {stream_name}"
                        ) from error
                    if rank is None:
                        if (
                            target_eligible
                            and target_length is not None
                            and int(target_length) <= numeric_limit
                        ):
                            raise ValueError(
                                f"eligible realized token is missing in {stream_name}"
                            )
                        continue
                    try:
                        numeric_rank = int(rank)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid filtered realized rank in {stream_name}"
                        ) from error
                    if numeric_rank != rank or not 1 <= numeric_rank <= denominator:
                        raise ValueError(
                            f"filtered realized rank is out of bounds in {stream_name}"
                        )
                    if not target_eligible or (
                        target_length is not None and int(target_length) > numeric_limit
                    ):
                        raise ValueError(
                            f"excluded realized token has a rank in {stream_name}"
                        )


def _write_json(
    path: Path, payload: Mapping[str, Any], *, compact: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
    else:
        rendered = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _source_payload(
    sample: StaticAudioSample, source: StaticAudioSource
) -> dict[str, Any]:
    return {
        "audio_url": f"/audio-jacobian-lens/audio/{sample.filename}",
        "repository_path": f"samples/{sample.filename}",
        "sha256": sample.sha256,
        "utterance_id": sample.utterance_id,
        "reference_transcript": sample.reference_transcript,
        "license": source.license,
        "rights_status": "cleared_with_attribution",
        "attribution": source.attribution,
        "source_url": source.source_url,
        "license_url": source.license_url,
        "lfm_fit_relationship": sample.lfm_fit_relationship,
    }


def _detailed_provenance(
    source: Mapping[str, Any], catalog: StaticExplorerCatalog
) -> dict[str, Any]:
    """Replace count-specific curated rights text for the ten-report bundle."""

    provenance = copy.deepcopy(dict(source))
    audio_source = catalog.audio_source
    count = catalog.reports_per_family
    rights = provenance.get("rights")
    if isinstance(rights, dict):
        rights.update(
            {
                "source_collection": audio_source.upstream_collection,
                "license": audio_source.license,
                "license_url": audio_source.license_url,
                "source_url": audio_source.source_url,
                "attribution": audio_source.attribution,
                "modification_notice": (
                    f"The {count} published input FLAC files are copied "
                    "without modification from the pinned catalog source."
                ),
            }
        )
    rights_policy = provenance.get("rights_policy")
    if isinstance(rights_policy, dict):
        rights_policy.update(
            {
                "included_audio": (
                    f"The {count} source-input FLAC files are distributed "
                    f"under {audio_source.license} with attribution."
                ),
                "attribution": audio_source.attribution,
                "source_url": audio_source.source_url,
                "license_url": audio_source.license_url,
            }
        )
    return provenance


def _computational_provenance(
    provenance: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    """Select fields that bind cached values to a model/lens computation."""

    keys = ("model", "lens", "rank_semantics", "timing_semantics")
    result = {key: copy.deepcopy(provenance[key]) for key in keys if key in provenance}
    if family == "speech" and "evaluation_generation" in provenance:
        generation = copy.deepcopy(provenance["evaluation_generation"])
        if not isinstance(generation, dict):
            raise ValueError("speech evaluation_generation must be an object")
        # The serving cap is derived from the complete report set below. It may
        # intentionally differ from the older curated-provenance snapshot.
        generation.pop("max_new_tokens", None)
        generation.pop("system_prompt", None)
        result["evaluation_generation"] = generation
    return result


def _load_reuse_manifest(
    *,
    family: str,
    output_dir: Path,
    requested_provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, dict[str, Mapping[str, Any]]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid existing manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema_id") != MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("family") != family
    ):
        raise ValueError(f"existing {family} manifest has an invalid identity")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"existing {family} manifest has no provenance")
    if _computational_provenance(
        provenance, family=family
    ) != _computational_provenance(requested_provenance, family=family):
        raise ValueError(
            f"existing {family} reports were produced by a different "
            "model, lens, or generation contract"
        )
    raw_entries = manifest.get("reports")
    if not isinstance(raw_entries, list):
        raise ValueError(f"existing {family} manifest has no report list")
    entries: dict[str, Mapping[str, Any]] = {}
    for value in raw_entries:
        if not isinstance(value, Mapping):
            raise ValueError(f"existing {family} manifest entry is not an object")
        example_id = value.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"existing {family} manifest entry has no ID")
        if example_id in entries:
            raise ValueError(f"duplicate existing {family} report ID {example_id}")
        entries[example_id] = value
    return provenance, entries


def _validate_report_computation(
    report: Mapping[str, Any],
    *,
    family: str,
    provenance: Mapping[str, Any],
) -> None:
    metadata = report.get("payload", {}).get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{family} report has no model metadata")
    model = provenance.get("model", {})
    if isinstance(model, Mapping):
        field_map = {
            "id": "model_id",
            "revision": "model_revision",
            "model_fingerprint": "model_fingerprint",
            "weights_fingerprint": "weights_fingerprint",
            "model_config_fingerprint": "model_config_fingerprint",
            "tokenizer_fingerprint": "tokenizer_fingerprint",
        }
        for provenance_key, metadata_key in field_map.items():
            expected = model.get(provenance_key)
            recorded = metadata.get(metadata_key)
            required = provenance_key == "id" or (
                family == "speech" and provenance_key == "revision"
            )
            if expected is not None and (
                (required and recorded != expected)
                or (recorded is not None and recorded != expected)
            ):
                raise ValueError(
                    f"{family} report {metadata_key} disagrees with provenance"
                )
    lens = provenance.get("lens", {})
    if not isinstance(lens, Mapping):
        return
    payload = report["payload"]
    if family == "asr":
        expected_encoder = lens.get("encoder_source_layers")
        expected_decoder = lens.get("decoder_source_layers")
        if (
            expected_encoder is not None
            and payload["encoder"].get("layers") != expected_encoder
        ):
            raise ValueError("ASR encoder layers disagree with lens provenance")
        if (
            expected_decoder is not None
            and payload["decoder"].get("layers") != expected_decoder
        ):
            raise ValueError("ASR decoder layers disagree with lens provenance")
        return
    expected_layers = lens.get("source_layers")
    decoder = payload["decoder"]
    if expected_layers is not None and decoder.get("layers") != expected_layers:
        raise ValueError("speech source layers disagree with lens provenance")
    if lens.get("target_layer") is not None and decoder.get("target_layer") != lens.get(
        "target_layer"
    ):
        raise ValueError("speech target layer disagrees with lens provenance")
    if lens.get("projection_rank") is not None and decoder.get(
        "projection_rank"
    ) != lens.get("projection_rank"):
        raise ValueError("speech projection rank disagrees with lens provenance")
    if lens.get("fit_examples") is not None and metadata.get(
        "lens_examples"
    ) != lens.get("fit_examples"):
        raise ValueError("speech fit-example count disagrees with lens provenance")


def _derive_speech_generation_provenance(
    provenance: dict[str, Any], reports: list[Mapping[str, Any]]
) -> None:
    """Require one serving policy and record its actual cap in the manifest."""

    serving_policies: list[dict[str, Any]] = []
    for report in reports:
        metadata = report.get("payload", {}).get("metadata", {})
        serving = metadata.get("serving_generation")
        if not isinstance(serving, Mapping):
            raise ValueError("speech report has no serving-generation policy")
        policy = {
            "system_prompt": serving.get("system_prompt"),
            "max_new_tokens": serving.get("max_new_tokens"),
            "temperature": serving.get("temperature"),
            "top_k": serving.get("top_k"),
            "audio_temperature": serving.get("audio_temperature"),
            "audio_top_k": serving.get("audio_top_k"),
        }
        if (
            not isinstance(policy["max_new_tokens"], int)
            or isinstance(policy["max_new_tokens"], bool)
            or policy["max_new_tokens"] <= 0
        ):
            raise ValueError("speech report has an invalid serving cap")
        generation = metadata.get("generation")
        if not isinstance(generation, Mapping) or any(
            generation.get(key) != value for key, value in policy.items()
        ):
            raise ValueError(
                "speech report generation and serving-generation policies disagree"
            )
        diagnostics = metadata.get("generation_diagnostics")
        if (
            not isinstance(diagnostics, Mapping)
            or diagnostics.get("max_new_tokens") != policy["max_new_tokens"]
        ):
            raise ValueError("speech diagnostics disagree with the serving cap")
        serving_policies.append(policy)
    if not serving_policies or any(
        policy != serving_policies[0] for policy in serving_policies[1:]
    ):
        raise ValueError("speech reports do not share one serving-generation policy")

    recorded = provenance.get("evaluation_generation")
    if not isinstance(recorded, dict):
        raise ValueError("speech provenance has no evaluation_generation object")
    common = serving_policies[0]
    comparisons = {
        "text_temperature": "temperature",
        "text_top_k": "top_k",
        "audio_temperature": "audio_temperature",
        "audio_top_k": "audio_top_k",
    }
    for provenance_key, serving_key in comparisons.items():
        if recorded.get(provenance_key) != common[serving_key]:
            raise ValueError(
                f"speech {provenance_key} disagrees with report generation"
            )
    recorded["max_new_tokens"] = common["max_new_tokens"]
    recorded["system_prompt"] = common["system_prompt"]


def _validate_sample_file(path: Path, sample: StaticAudioSample) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {sample.slug} input: {path}; run "
            "scripts/materialize_static_audio_samples.py"
        )
    actual = _sha256(path)
    if actual != sample.sha256:
        raise ValueError(
            f"{sample.slug} input hash mismatch: expected {sample.sha256}, "
            f"found {actual}"
        )


def _filter_manifest(
    *, family: str, sample: StaticAudioSample, filter_path: Path
) -> dict[str, Any] | None:
    if family != "asr":
        return None
    return {
        "url": (f"/audio-jacobian-lens/explorer/data/asr/{sample.slug}.filters.json"),
        "sha256": _sha256(filter_path),
        "bytes": filter_path.stat().st_size,
    }


def _manifest_entry(
    *,
    family: str,
    sample: StaticAudioSample,
    report_path: Path,
    filter_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    entry = {
        "id": f"{family}-{sample.slug}",
        "title": sample.title,
        "summary": sample.description,
        "reference_transcript": sample.reference_transcript,
        "utterance_id": sample.utterance_id,
        "audio_url": f"/audio-jacobian-lens/audio/{sample.filename}",
        "report_url": (
            f"/audio-jacobian-lens/explorer/data/{family}/{sample.slug}.json"
        ),
        "sha256": _sha256(report_path),
        "bytes": report_path.stat().st_size,
    }
    if filter_manifest is not None:
        entry["character_length_filter_cache"] = dict(filter_manifest)
    return entry


def _validate_existing_report(
    *,
    family: str,
    sample: StaticAudioSample,
    source: StaticAudioSource,
    output_dir: Path,
    provenance: Mapping[str, Any],
    manifest_entry: Mapping[str, Any] | None = None,
    require_manifest_binding: bool = False,
) -> dict[str, Any] | None:
    report_path = output_dir / f"{sample.slug}.json"
    if not report_path.is_file():
        return None
    if require_manifest_binding and manifest_entry is None:
        raise ValueError(
            f"cached {family}-{sample.slug} is not bound to the existing manifest"
        )
    if manifest_entry is not None:
        expected_url = f"/audio-jacobian-lens/explorer/data/{family}/{sample.slug}.json"
        if (
            manifest_entry.get("id") != f"{family}-{sample.slug}"
            or manifest_entry.get("report_url") != expected_url
            or manifest_entry.get("sha256") != _sha256(report_path)
            or manifest_entry.get("bytes") != report_path.stat().st_size
        ):
            raise ValueError(
                f"cached {family}-{sample.slug} does not match its published manifest"
            )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid cached report JSON: {report_path}") from error
    if not isinstance(report, dict):
        raise ValueError(f"cached report must be an object: {report_path}")
    expected_source = _source_payload(sample, source)
    if (
        report.get("schema_id") != SCHEMA_ID
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("family") != family
        or report.get("example_id") != f"{family}-{sample.slug}"
    ):
        raise ValueError(f"cached report identity mismatch: {report_path}")
    recorded_source = report.get("source")
    if not isinstance(recorded_source, Mapping):
        raise ValueError(f"cached report has no source metadata: {report_path}")
    for key, expected in expected_source.items():
        if key == "lfm_fit_relationship" and key not in recorded_source:
            # The original three reports predate this catalog field.
            continue
        if recorded_source.get(key) != expected:
            raise ValueError(f"cached report source mismatch for {sample.slug}.{key}")
    duration = report.get("payload", {}).get("audio", {}).get("duration_seconds")
    try:
        duration_matches = math.isclose(
            float(duration), sample.duration_seconds, rel_tol=0.0, abs_tol=1e-4
        )
    except (TypeError, ValueError):
        duration_matches = False
    if not duration_matches:
        raise ValueError(f"cached report duration mismatch: {report_path}")
    _validate_safe(report)
    _validate_matrix(report)
    _validate_report_computation(report, family=family, provenance=provenance)

    filter_manifest: dict[str, Any] | None = None
    if family == "asr":
        filter_path = output_dir / f"{sample.slug}.filters.json"
        if not filter_path.is_file():
            raise ValueError(f"cached ASR report has no filter sidecar: {filter_path}")
        try:
            filter_cache = json.loads(filter_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid filter cache JSON: {filter_path}") from error
        _validate_safe(filter_cache)
        _validate_filter_cache(filter_cache, report)
        filter_manifest = _filter_manifest(
            family=family, sample=sample, filter_path=filter_path
        )
        recorded_filter = report.get("cache_policy", {}).get(
            "character_length_filter_cache"
        )
        if recorded_filter != filter_manifest:
            raise ValueError(f"cached ASR filter hash metadata mismatch: {filter_path}")
        if (
            manifest_entry is not None
            and manifest_entry.get("character_length_filter_cache") != filter_manifest
        ):
            raise ValueError(
                f"cached ASR filter for {sample.slug} does not match its manifest"
            )
    return _manifest_entry(
        family=family,
        sample=sample,
        report_path=report_path,
        filter_manifest=filter_manifest,
    )


def _write_fresh_report(
    *,
    family: str,
    endpoint: str,
    sample: StaticAudioSample,
    source: StaticAudioSource,
    samples_dir: Path,
    output_dir: Path,
) -> None:
    audio_path = samples_dir / sample.filename
    _validate_sample_file(audio_path, sample)
    example_id = f"{family}-{sample.slug}"
    raw = _analyze(
        endpoint,
        audio_path,
        {"time_bin_overlap_seconds": "0.02"} if family == "asr" else {},
    )
    report = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "example_id": example_id,
        "title": sample.title,
        "source": _source_payload(sample, source),
        "cache_policy": {
            "inference": "precomputed",
            "waveform": "1024-point uniform preview of the cleared input",
            "generated_audio_included": False,
            "character_length_bucket_cache_included": family == "asr",
            "unsupported_static_controls": (
                []
                if family == "asr"
                else [
                    {
                        "control": "character_length_reranking",
                        "scope": "LFM projected language readouts",
                        "reason": (
                            "The projected LFM pilot does not provide "
                            "token-length buckets."
                        ),
                    }
                ]
            ),
        },
        "payload": _reduce_payload(raw),
    }
    if family == "asr":
        filter_cache = _filter_cache(raw, example_id=example_id)
        _validate_safe(filter_cache)
        _validate_filter_cache(filter_cache, report)
        filter_path = output_dir / f"{sample.slug}.filters.json"
        _write_json(filter_path, filter_cache, compact=True)
        report["cache_policy"]["character_length_filter_cache"] = _filter_manifest(
            family=family, sample=sample, filter_path=filter_path
        )
    _validate_safe(report)
    _validate_matrix(report)
    report_path = output_dir / f"{sample.slug}.json"
    _write_json(report_path, report)
    print(f"wrote {report_path} ({report_path.stat().st_size:,} bytes)")


def _stage_existing_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _promote_staged_family(
    *,
    stage_dir: Path,
    output_dir: Path,
    changed_filenames: set[str],
    replace_file=os.replace,
) -> None:
    """Promote validated files atomically and roll back ordinary failures."""

    names = sorted(name for name in changed_filenames if name != "manifest.json")
    names.append("manifest.json")
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-backup-", dir=output_dir.parent
    ) as temporary:
        backup_dir = Path(temporary)
        existed: dict[str, bool] = {}
        promoted: list[str] = []
        try:
            for name in names:
                staged = stage_dir / name
                if not staged.is_file():
                    raise FileNotFoundError(f"staged export file is missing: {staged}")
                live = output_dir / name
                existed[name] = live.is_file()
                if existed[name]:
                    shutil.copy2(live, backup_dir / name)
                replace_file(staged, live)
                promoted.append(name)
        except Exception:
            for name in reversed(promoted):
                live = output_dir / name
                if existed[name]:
                    backup = backup_dir / name
                    try:
                        os.replace(backup, live)
                    except OSError:
                        shutil.copy2(backup, live)
                else:
                    live.unlink(missing_ok=True)
            raise


def export_family(
    *,
    family: str,
    endpoint: str,
    samples_dir: Path,
    output_dir: Path,
    provenance: Mapping[str, Any],
    catalog: StaticExplorerCatalog,
    selected_slugs: set[str],
    resume: bool,
    catalog_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in catalog.audio_samples:
        _validate_sample_file(samples_dir / sample.filename, sample)
    requested_provenance = _detailed_provenance(provenance, catalog)
    all_slugs = {sample.slug for sample in catalog.audio_samples}
    needs_reuse = resume or selected_slugs != all_slugs
    if needs_reuse:
        reuse_provenance, reuse_entries = _load_reuse_manifest(
            family=family,
            output_dir=output_dir,
            requested_provenance=requested_provenance,
        )
    else:
        reuse_provenance, reuse_entries = None, {}

    with tempfile.TemporaryDirectory(
        prefix=f".{family}-export-", dir=output_dir.parent
    ) as temporary:
        stage_dir = Path(temporary)
        changed_filenames: set[str] = set()
        missing: list[str] = []
        for sample in catalog.audio_samples:
            example_id = f"{family}-{sample.slug}"
            manifest_entry = reuse_entries.get(example_id)
            should_reuse = sample.slug not in selected_slugs or (
                resume and manifest_entry is not None
            )
            if should_reuse:
                if reuse_provenance is None:
                    missing.append(sample.slug)
                    continue
                existing = _validate_existing_report(
                    family=family,
                    sample=sample,
                    source=catalog.audio_source,
                    output_dir=output_dir,
                    provenance=reuse_provenance,
                    manifest_entry=manifest_entry,
                    require_manifest_binding=True,
                )
                if existing is None:
                    if sample.slug not in selected_slugs:
                        missing.append(sample.slug)
                        continue
                else:
                    _stage_existing_file(
                        output_dir / f"{sample.slug}.json",
                        stage_dir / f"{sample.slug}.json",
                    )
                    if family == "asr":
                        _stage_existing_file(
                            output_dir / f"{sample.slug}.filters.json",
                            stage_dir / f"{sample.slug}.filters.json",
                        )
                    print(f"kept {example_id} (manifest-bound report)")
                    continue
            _write_fresh_report(
                family=family,
                endpoint=endpoint,
                sample=sample,
                source=catalog.audio_source,
                samples_dir=samples_dir,
                output_dir=stage_dir,
            )
            changed_filenames.add(f"{sample.slug}.json")
            if family == "asr":
                changed_filenames.add(f"{sample.slug}.filters.json")

        if missing:
            raise RuntimeError(
                "refusing to replace the publication manifest before every "
                f"catalog report exists; missing {family}: {', '.join(missing)}"
            )

        reports: list[dict[str, Any]] = []
        report_payloads: list[Mapping[str, Any]] = []
        for sample in catalog.audio_samples:
            entry = _validate_existing_report(
                family=family,
                sample=sample,
                source=catalog.audio_source,
                output_dir=stage_dir,
                provenance=requested_provenance,
            )
            if entry is None:
                raise RuntimeError(f"staged {family}-{sample.slug} is missing")
            reports.append(entry)
            report_payloads.append(
                json.loads(
                    (stage_dir / f"{sample.slug}.json").read_text(encoding="utf-8")
                )
            )
        if len(reports) != catalog.reports_per_family:
            raise RuntimeError(f"{family} report count does not match the catalog")
        final_provenance = _detailed_provenance(provenance, catalog)
        if family == "speech":
            _derive_speech_generation_provenance(final_provenance, report_payloads)
        for report in report_payloads:
            _validate_report_computation(
                report, family=family, provenance=final_provenance
            )
        manifest = {
            "schema_id": MANIFEST_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "mode": "static_cached_explorer",
            "report_count": len(reports),
            "catalog": {
                "schema_id": CATALOG_SCHEMA_ID,
                "schema_version": CATALOG_SCHEMA_VERSION,
                "sha256": catalog_sha256,
                "reports_per_family": catalog.reports_per_family,
            },
            "provenance": final_provenance,
            "reports": reports,
        }
        _validate_safe(manifest)
        _write_json(stage_dir / "manifest.json", manifest)
        _promote_staged_family(
            stage_dir=stage_dir,
            output_dir=output_dir,
            changed_filenames=changed_filenames,
        )
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/static_explorer_catalog_v2.json"),
    )
    parser.add_argument(
        "--provenance-source",
        type=Path,
        default=Path("data/static_public_reports_v1.json"),
    )
    parser.add_argument("--asr-url", default="http://127.0.0.1:8000/api/analyze")
    parser.add_argument("--speech-url", default="http://127.0.0.1:8001/api/analyze")
    parser.add_argument(
        "--family",
        action="append",
        choices=("asr", "speech"),
        help="Export one family; repeat to export both. Defaults to both.",
    )
    parser.add_argument(
        "--only",
        "--example-id",
        action="append",
        dest="only",
        metavar="SLUG",
        help=(
            "Regenerate only these comma-separated sample slugs or full "
            "family example IDs. Existing unselected reports are revalidated "
            "and retained. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and retain selected reports that already exist.",
    )
    return parser


def _selected_slugs(
    catalog: StaticExplorerCatalog, requested: list[str] | None
) -> set[str]:
    available = {sample.slug for sample in catalog.audio_samples}
    if not requested:
        return available
    selected: set[str] = set()
    for group in requested:
        for raw_value in group.split(","):
            value = raw_value.strip()
            for prefix in ("asr-", "speech-"):
                if value.startswith(prefix):
                    value = value[len(prefix) :]
                    break
            if value not in available:
                raise ValueError(f"unknown static explorer sample: {raw_value.strip()}")
            selected.add(value)
    if not selected:
        raise ValueError("--only did not select any static explorer samples")
    return selected


def main() -> None:
    args = _parser().parse_args()
    catalog = load_static_explorer_catalog(args.catalog)
    selected_slugs = _selected_slugs(catalog, args.only)
    families = args.family or ["asr", "speech"]
    endpoints = {"asr": args.asr_url, "speech": args.speech_url}
    provenance_source = json.loads(args.provenance_source.read_text(encoding="utf-8"))
    for family in families:
        provenance = provenance_source["families"][family]["provenance"]
        export_family(
            family=family,
            endpoint=endpoints[family],
            samples_dir=args.samples_dir,
            output_dir=args.site_root / "explorer" / "data" / family,
            provenance=provenance,
            catalog=catalog,
            selected_slugs=selected_slugs,
            resume=args.resume,
            catalog_sha256=_sha256(args.catalog),
        )


if __name__ == "__main__":
    main()
