#!/usr/bin/env python3
"""Export rights-safe, reduced ASR and LFM explorer payloads.

The exporter calls the local single-worker servers sequentially. It builds each
published payload from explicit allowlists, never copies generated audio, and
fails if an ephemeral analysis identifier or embedded audio URI survives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "selected_score",
    "time_window",
}

STREAM_FIELDS = {
    "layers",
    "pooling",
    "positions",
    "projection_rank",
    "score_kind",
    "stream_kind",
    "target_layer",
    "time_bins",
}


@dataclass(frozen=True)
class Sample:
    slug: str
    title: str
    utterance_id: str
    transcript: str


SAMPLES = (
    Sample(
        "question",
        "A short question",
        "1272-135031-0012",
        "Where is my brother now?",
    ),
    Sample(
        "universe",
        "Sir, I exist",
        "1272-141231-0000",
        "A man said to the universe, sir, I exist.",
    ),
    Sample(
        "buzzer",
        "The buzzer's whirr",
        "1272-141231-0006",
        "The buzzer's whirr triggered his muscles into complete relaxation.",
    ),
)


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
    result["top_tokens"] = [
        _candidate(item) for item in source.get("top_tokens", [])
    ]
    return result


def _cell(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(source, CELL_FIELDS)
    result["top_tokens"] = [
        _candidate(item) for item in source.get("top_tokens", [])
    ]
    realized = source.get("realized_token")
    if isinstance(realized, Mapping):
        result["realized_token"] = _candidate(realized)
    return result


def _stream(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not source:
        return {"layers": [], "positions": [], "cells": []}
    result = _pick(source, STREAM_FIELDS)
    result["cells"] = [
        [_cell(cell) for cell in layer_cells]
        for layer_cells in source.get("cells", [])
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
        indices = [round(index * (len(numeric) - 1) / (count - 1)) for index in range(count)]
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
            cells.append(
                {
                    "position_index": cell.get("position_index"),
                    "time_window": cell.get("time_window"),
                    "top_tokens_by_length": {
                        str(length): [
                            _candidate(item) for item in candidates
                        ]
                        for length, candidates in buckets.items()
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
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
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
        raise ValueError(f"{label} realized-token provenance lacks {', '.join(missing)}")
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


def _validate_matrix(report: Mapping[str, Any]) -> None:
    payload = report["payload"]
    family = report.get("family")
    transcription_tokens = payload.get("transcription", {}).get("tokens", [])
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
        if (
            family == "speech"
            and stream_name == "decoder"
            and widths != {len(transcription_tokens)}
        ):
            raise ValueError("speech decoder/output-token width mismatch")
        for layer_index, row in enumerate(cells):
            for position, cell in enumerate(row):
                if not cell.get("top_tokens"):
                    raise ValueError(f"{stream_name} cell has no candidates")
                if family == "speech" and stream_name == "decoder":
                    if position >= len(transcription_tokens):
                        raise ValueError("speech decoder is wider than its output tokens")
                    _validate_exact_rank_candidate(
                        cell.get("realized_token"),
                        label=f"speech decoder layer {layer_index}, position {position}",
                        expected_id=transcription_tokens[position].get("id"),
                        require_score=True,
                    )
    if family == "speech":
        for position, token in enumerate(transcription_tokens):
            _validate_exact_rank_candidate(
                token,
                label=f"speech HEAD position {position}",
                expected_id=token.get("id"),
            )


def _validate_filter_cache(
    cache: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    if cache["example_id"] != report["example_id"]:
        raise ValueError("filter cache/report example mismatch")
    payload = report["payload"]
    for stream_name in ("encoder", "decoder"):
        cache_stream = cache["streams"][stream_name]
        report_stream = payload[stream_name]
        report_width = len(report_stream.get("cells", [])[0]) if report_stream.get("cells") else 0
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
                    raise ValueError(f"filter cell coordinate mismatch in {stream_name}")
                if not cell["top_tokens_by_length"]:
                    raise ValueError(f"empty filter buckets in {stream_name}")


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
    path.write_text(
        rendered + "\n",
        encoding="utf-8",
    )


def export_family(
    *,
    family: str,
    endpoint: str,
    samples_dir: Path,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for sample in SAMPLES:
        audio_path = samples_dir / f"{sample.slug}.flac"
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
            "source": {
                "audio_url": f"/audio-jacobian-lens/audio/{sample.slug}.flac",
                "repository_path": f"samples/{sample.slug}.flac",
                "sha256": _sha256(audio_path),
                "utterance_id": sample.utterance_id,
                "reference_transcript": sample.transcript,
                "license": "CC BY 4.0",
                "rights_status": "cleared_with_attribution",
                "attribution": (
                    "LibriSpeech dev-clean, prepared by Vassil Panayotov, "
                    "Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur; "
                    "derived from public-domain LibriVox recordings."
                ),
                "source_url": "https://www.openslr.org/12",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
            },
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
        filter_manifest: dict[str, Any] | None = None
        if family == "asr":
            filter_cache = _filter_cache(raw, example_id=example_id)
            _validate_safe(filter_cache)
            _validate_filter_cache(filter_cache, report)
            filter_path = output_dir / f"{sample.slug}.filters.json"
            _write_json(filter_path, filter_cache, compact=True)
            filter_manifest = {
                "url": (
                    "/audio-jacobian-lens/explorer/data/asr/"
                    f"{sample.slug}.filters.json"
                ),
                "sha256": _sha256(filter_path),
                "bytes": filter_path.stat().st_size,
            }
            report["cache_policy"]["character_length_filter_cache"] = (
                filter_manifest
            )
        _validate_safe(report)
        _validate_matrix(report)
        report_path = output_dir / f"{sample.slug}.json"
        _write_json(report_path, report)
        manifest_entry = {
            "id": report["example_id"],
            "title": sample.title,
            "audio_url": report["source"]["audio_url"],
            "report_url": (
                f"/audio-jacobian-lens/explorer/data/{family}/{sample.slug}.json"
            ),
            "sha256": _sha256(report_path),
            "bytes": report_path.stat().st_size,
        }
        if filter_manifest is not None:
            manifest_entry["character_length_filter_cache"] = filter_manifest
        reports.append(manifest_entry)
        print(f"wrote {report_path} ({report_path.stat().st_size:,} bytes)")
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "mode": "static_cached_explorer",
        "provenance": provenance,
        "reports": reports,
    }
    _validate_safe(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--site-root", type=Path, required=True)
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
    return parser


def main() -> None:
    args = _parser().parse_args()
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
        )


if __name__ == "__main__":
    main()
