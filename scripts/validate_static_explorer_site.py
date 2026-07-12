#!/usr/bin/env python3
"""Validate the publishable, backend-free Audio Jacobian Lens site bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SITE_PREFIX = "/audio-jacobian-lens/"
PUBLIC_BASE = "https://kennethli319.github.io/audio-jacobian-lens/"
FAMILIES = ("asr", "speech", "tts")
EXPECTED_REPORT_COUNT = 10
EXPLORER_ASSET_VERSION = "20260711-11"
CANONICAL_DETAILED_ROUTES = {
    "asr": SITE_PREFIX,
    "speech": f"{SITE_PREFIX}speech/",
    "tts": f"{SITE_PREFIX}tts/",
}
FINDINGS_ROUTES = {
    "asr": f"{SITE_PREFIX}findings/",
    "speech": f"{SITE_PREFIX}findings/speech/",
    "tts": f"{SITE_PREFIX}findings/tts/",
}
LEGACY_EXPLORER_ROUTES = {
    family: f"{SITE_PREFIX}explorer/{family}/" for family in FAMILIES
}
FORBIDDEN_KEYS = {
    "analysis_id",
    "parent_analysis_id",
    "branch_analysis_id",
    "audio_data_url",
    "generated_audio",
    "output_waveform",
}
SPEECH_TERMINATION_SCRIPT_MARKERS = (
    "function renderSpeechTerminationStatus()",
    'data-speech-termination="budget-exhausted"',
    "response may be truncated",
)
SPEECH_TERMINATION_CSS_MARKERS = (".generation-status.capped",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _site_path(site_root: Path, url: str) -> Path:
    if not url.startswith(SITE_PREFIX):
        raise ValueError(f"published URL is outside {SITE_PREFIX}: {url}")
    return site_root / url.removeprefix(SITE_PREFIX)


def _manifest_reports(
    manifest: Mapping[str, Any], *, family: str
) -> list[Mapping[str, Any]]:
    reports = manifest.get("reports")
    if manifest.get("report_count") != EXPECTED_REPORT_COUNT:
        raise ValueError(
            f"{family} manifest report_count must be {EXPECTED_REPORT_COUNT}"
        )
    if not isinstance(reports, list) or len(reports) != EXPECTED_REPORT_COUNT:
        raise ValueError(
            f"{family} must publish exactly {EXPECTED_REPORT_COUNT} reports"
        )
    if not all(isinstance(entry, Mapping) for entry in reports):
        raise ValueError(f"{family} manifest contains a non-object report entry")
    ids = [str(entry.get("id") or "") for entry in reports]
    report_urls = [str(entry.get("report_url") or "") for entry in reports]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{family} manifest report IDs are empty or duplicated")
    if any(not value for value in report_urls) or len(set(report_urls)) != len(
        report_urls
    ):
        raise ValueError(f"{family} manifest report URLs are empty or duplicated")
    if family == "tts":
        if any(entry.get("audio_url") is not None for entry in reports):
            raise ValueError("TTS manifest must not publish generated-audio URLs")
    else:
        audio_urls = [str(entry.get("audio_url") or "") for entry in reports]
        if any(not value for value in audio_urls) or len(set(audio_urls)) != len(
            audio_urls
        ):
            raise ValueError(f"{family} manifest audio URLs are empty or duplicated")
    if family == "asr":
        filter_urls: list[str] = []
        for entry in reports:
            reference = entry.get("character_length_filter_cache")
            if not isinstance(reference, Mapping) or not reference.get("url"):
                raise ValueError(
                    "ASR manifest entry is missing its character-filter URL"
                )
            filter_urls.append(str(reference["url"]))
        if len(set(filter_urls)) != len(filter_urls):
            raise ValueError("ASR manifest character-filter URLs are duplicated")
    return reports


def _validate_safe(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS or key.endswith("_analysis_id"):
                raise ValueError(f"forbidden cached field {path}.{key}")
            _validate_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.lower().startswith("data:audio/"):
        raise ValueError(f"embedded audio URI at {path}")


def _validate_stream(
    stream: Mapping[str, Any], *, label: str, allow_empty: bool
) -> None:
    layers = stream.get("layers")
    cells = stream.get("cells")
    if not isinstance(layers, list) or not isinstance(cells, list):
        raise ValueError(f"{label} needs layer and cell arrays")
    if len(layers) != len(cells):
        raise ValueError(f"{label} layer/cell mismatch")
    if not cells:
        if allow_empty:
            return
        raise ValueError(f"{label} matrix is empty")
    widths = {len(row) for row in cells if isinstance(row, list)}
    if len(widths) != 1 or 0 in widths:
        raise ValueError(f"{label} matrix is empty or ragged")
    for row in cells:
        for cell in row:
            if not isinstance(cell.get("top_tokens"), list) or not cell["top_tokens"]:
                raise ValueError(f"{label} cell has no cached candidates")


def _validate_exact_realized_rank(
    candidate: Any,
    *,
    label: str,
    expected_id: Any,
    require_score: bool,
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
    if candidate["id"] != expected_id:
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


def _validate_speech_generation_diagnostics(report: Mapping[str, Any]) -> None:
    diagnostics = (
        report.get("payload", {}).get("metadata", {}).get("generation_diagnostics")
    )
    if not isinstance(diagnostics, Mapping):
        raise ValueError("speech report has no generation-termination diagnostics")
    values: dict[str, int] = {}
    for field in (
        "generated_steps",
        "max_new_tokens",
        "text_tokens",
        "audio_frames",
    ):
        value = diagnostics.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"speech generation diagnostic {field} is invalid")
        values[field] = value
    generated_steps = values["generated_steps"]
    max_new_tokens = values["max_new_tokens"]
    if (
        generated_steps < 1
        or max_new_tokens < generated_steps
        or values["text_tokens"] < 0
        or values["audio_frames"] < 0
    ):
        raise ValueError("speech generation diagnostic counts are out of bounds")
    audio_eos_seen = diagnostics.get("audio_eos_seen")
    budget_exhausted = diagnostics.get("budget_exhausted")
    if not isinstance(audio_eos_seen, bool) or not isinstance(budget_exhausted, bool):
        raise ValueError("speech generation diagnostic flags are invalid")
    eos_step = 1 if audio_eos_seen else 0
    if generated_steps != values["text_tokens"] + values["audio_frames"] + eos_step:
        raise ValueError("speech generation diagnostic step accounting is invalid")
    termination_reason = diagnostics.get("termination_reason")
    natural = (
        termination_reason == "audio_eos" and audio_eos_seen and not budget_exhausted
    )
    capped = (
        termination_reason == "budget_exhausted"
        and not audio_eos_seen
        and budget_exhausted
        and generated_steps == max_new_tokens
    )
    if not natural and not capped:
        raise ValueError("speech generation termination state is inconsistent")


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
        raise ValueError("ASR encoder cell has no realized-token alignment provenance")
    required = {
        "match",
        "window_midpoint_seconds",
        "token_start_seconds",
        "token_end_seconds",
        "overlap_seconds",
        "overlap_fraction_of_window",
    }
    if required - alignment.keys():
        raise ValueError("ASR encoder cell has incomplete alignment provenance")
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
        raise ValueError("ASR encoder cell has an invalid alignment match kind")
    for field, value in expected.items():
        try:
            recorded = float(alignment[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"ASR encoder alignment {field} is invalid") from error
        if not math.isclose(recorded, value, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"ASR encoder alignment {field} does not match timing")


def _validate_asr_or_speech(report: Mapping[str, Any], *, family: str) -> None:
    if family == "speech":
        _validate_speech_generation_diagnostics(report)
    payload = report["payload"]
    tokens = payload.get("transcription", {}).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{family} report has no output tokens")
    _validate_stream(payload["decoder"], label=f"{family} decoder", allow_empty=False)
    if len(payload["decoder"]["cells"][0]) != len(tokens):
        raise ValueError(f"{family} decoder/token width mismatch")
    if family in {"asr", "speech"}:
        for position, token in enumerate(tokens):
            _validate_exact_realized_rank(
                token,
                label=f"{family} HEAD position {position}",
                expected_id=token.get("id"),
                require_score=False,
            )
        for layer_index, row in enumerate(payload["decoder"]["cells"]):
            for position, cell in enumerate(row):
                _validate_exact_realized_rank(
                    cell.get("realized_token"),
                    label=(
                        f"{family} decoder layer {layer_index}, position {position}"
                    ),
                    expected_id=tokens[position].get("id"),
                    require_score=True,
                )
    _validate_stream(
        payload["encoder"], label=f"{family} encoder", allow_empty=family == "speech"
    )
    if family == "asr":
        alignment_metadata = payload["encoder"].get("realized_token_alignment")
        if (
            not isinstance(alignment_metadata, Mapping)
            or alignment_metadata.get("method") != "maximum_token_interval_overlap"
        ):
            raise ValueError("ASR encoder stream has no alignment-method provenance")
        for layer_index, row in enumerate(payload["encoder"]["cells"]):
            for position, cell in enumerate(row):
                token_index, token = _aligned_transcription_token(
                    tokens, cell.get("time_window") or {}
                )
                if cell.get("realized_token_position") != token_index:
                    raise ValueError(
                        "ASR encoder realized-token position does not match "
                        "overlap-first output-token synchronization"
                    )
                _validate_encoder_alignment_provenance(
                    cell.get("realized_token_alignment"),
                    time_window=cell.get("time_window") or {},
                    token=token,
                )
                _validate_exact_realized_rank(
                    cell.get("realized_token"),
                    label=(
                        f"asr encoder layer {layer_index}, position {position} "
                        f"(aligned output token {token_index})"
                    ),
                    expected_id=token.get("id"),
                    require_score=True,
                )
    preview = payload.get("audio", {}).get("waveform_preview", {})
    values = preview.get("values")
    if not isinstance(values, list) or not 0 < len(values) <= 1024:
        raise ValueError(f"{family} input waveform preview is invalid")
    if report.get("source", {}).get("rights_status") != "cleared_with_attribution":
        raise ValueError(f"{family} input audio is not rights-cleared")


def _validate_tts(report: Mapping[str, Any]) -> None:
    payload = report["payload"]
    codes = payload.get("output", {}).get("speech_codes")
    head = payload.get("output", {}).get("speech_head_candidates", {})
    fitted = payload.get("fitted_speech_code_jlens", {})
    traces = payload.get("traces_by_position")
    if not isinstance(codes, list) or not codes:
        raise ValueError("TTS report has no speech-code positions")
    width = len(codes)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("TTS report has no saved model metadata")
    generation = model.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("TTS report has no saved generation settings")
    generation_cap = generation.get("max_speech_tokens")
    if (
        isinstance(generation_cap, bool)
        or not isinstance(generation_cap, int)
        or generation_cap <= 0
    ):
        raise ValueError("TTS report has an invalid speech-code generation cap")
    if width >= generation_cap:
        raise ValueError(
            "TTS report reached its saved speech-code generation cap and may "
            "be truncated"
        )
    if len(head.get("positions", [])) != width:
        raise ValueError("TTS HEAD width mismatch")
    rows = fitted.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("TTS fitted rows are missing")
    if any(len(row.get("positions", [])) != width for row in rows):
        raise ValueError("TTS fitted matrix is ragged")
    if not isinstance(traces, Mapping) or set(traces) != {
        str(index) for index in range(width)
    }:
        raise ValueError("TTS trace cache does not cover every speech position")
    if payload.get("generated_audio_included") is not False:
        raise ValueError("TTS report must explicitly exclude generated audio")


def _validate_filter_cache(
    site_root: Path,
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    reference = entry.get("character_length_filter_cache")
    if not isinstance(reference, Mapping):
        raise ValueError("ASR report is missing its lazy character-filter cache")
    path = _site_path(site_root, str(reference["url"]))
    if not path.is_file() or _sha256(path) != reference["sha256"]:
        raise ValueError(f"ASR filter cache hash mismatch: {path}")
    if path.stat().st_size != int(reference.get("bytes", -1)):
        raise ValueError(f"ASR filter cache byte count mismatch: {path}")
    report_reference = report.get("cache_policy", {}).get(
        "character_length_filter_cache"
    )
    if report_reference != reference:
        raise ValueError("ASR report and manifest disagree on the filter cache")
    cache = _load(path)
    if cache.get("example_id") != report.get("example_id"):
        raise ValueError("ASR filter cache/report ID mismatch")
    denominators_by_length = (
        report.get("payload", {})
        .get("metadata", {})
        .get("display_vocabulary", {})
        .get("maximum_decoded_character_length_counts", {})
    )
    if not isinstance(denominators_by_length, Mapping) or not denominators_by_length:
        raise ValueError("ASR report has no character-filter denominators")
    for stream_name in ("encoder", "decoder"):
        filtered = cache["streams"][stream_name]
        base = report["payload"][stream_name]
        if any(layer not in base["layers"] for layer in filtered["layers"]):
            raise ValueError(f"ASR {stream_name} filter layer mismatch")
        width = len(base["cells"][0])
        if any(len(row) != width for row in filtered["cells"]):
            raise ValueError(f"ASR {stream_name} filter width mismatch")
        for layer, row in zip(filtered["layers"], filtered["cells"], strict=True):
            report_layer_index = base["layers"].index(layer)
            for position, cell in enumerate(row):
                if not cell.get("top_tokens_by_length"):
                    raise ValueError(f"ASR {stream_name} filter cell is empty")
                realized_rank_by_length = cell.get("realized_rank_by_max_length")
                if (
                    not isinstance(realized_rank_by_length, Mapping)
                    or not realized_rank_by_length
                ):
                    raise ValueError(
                        f"ASR {stream_name} filter cell has no exact realized ranks"
                    )
                base_cell = base["cells"][report_layer_index][position]
                target_filter = base_cell.get("realized_token", {}).get(
                    "vocabulary_filter", {}
                )
                target_length = target_filter.get("decoded_character_length")
                target_eligible = target_filter.get("display_lexical_eligible")
                if not isinstance(target_eligible, bool):
                    raise ValueError(
                        f"ASR {stream_name} realized token has no lexical eligibility"
                    )
                if set(realized_rank_by_length) != set(denominators_by_length):
                    raise ValueError(
                        f"ASR {stream_name} filtered ranks do not cover every limit"
                    )
                for limit, rank in realized_rank_by_length.items():
                    try:
                        numeric_limit = int(limit)
                        denominator = int(denominators_by_length[str(limit)])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"ASR {stream_name} filter limit is invalid"
                        ) from error
                    if rank is None:
                        if (
                            target_eligible
                            and target_length is not None
                            and int(target_length) <= numeric_limit
                        ):
                            raise ValueError(
                                f"ASR {stream_name} eligible realized rank is missing"
                            )
                        continue
                    try:
                        numeric_rank = int(rank)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"ASR {stream_name} filtered realized rank is invalid"
                        ) from error
                    if numeric_rank != rank or not 1 <= numeric_rank <= denominator:
                        raise ValueError(
                            f"ASR {stream_name} filtered realized rank is out of bounds"
                        )
                    if not target_eligible or (
                        target_length is not None and int(target_length) > numeric_limit
                    ):
                        raise ValueError(
                            f"ASR {stream_name} excluded realized token has a rank"
                        )
    _validate_safe(cache)


def _validate_audio_reference(
    site_root: Path,
    *,
    family: str,
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> Path:
    entry_url = str(entry.get("audio_url") or "")
    source = report.get("source")
    if not isinstance(source, Mapping) or source.get("audio_url") != entry_url:
        raise ValueError(f"{family} report and manifest audio URLs disagree")
    audio_path = _site_path(site_root, entry_url)
    if not audio_path.is_file():
        raise ValueError(f"missing cleared input audio: {audio_path}")
    expected_hash = source.get("sha256")
    if not isinstance(expected_hash, str) or _sha256(audio_path) != expected_hash:
        raise ValueError(f"{family} input audio hash mismatch: {audio_path}")
    return audio_path


def _validate_site_manifest_integrity(
    site_root: Path, *, counts: Mapping[str, int]
) -> None:
    manifest = _load(site_root / "site-manifest.json")
    if manifest.get("report_counts") != dict(counts):
        raise ValueError("site manifest report counts do not match family manifests")
    recorded_hashes = manifest.get("sha256")
    if not isinstance(recorded_hashes, Mapping) or not recorded_hashes:
        raise ValueError("site manifest has no asset hashes")
    required = {
        "assets/explorer.js",
        "assets/explorer.css",
        *(f"explorer/data/{family}/manifest.json" for family in FAMILIES),
    }
    if not required.issubset(recorded_hashes):
        raise ValueError("site manifest does not hash every explorer contract file")
    for relative_path, expected_hash in recorded_hashes.items():
        path = (site_root / str(relative_path)).resolve()
        if not path.is_relative_to(site_root) or not path.is_file():
            raise ValueError(f"site manifest hash path is missing or unsafe: {path}")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"site manifest hash mismatch: {path}")


def _require_page(site_root: Path, relative_path: str, *, label: str) -> str:
    path = site_root / relative_path
    if not path.is_file():
        raise ValueError(f"missing {label} page: {relative_path}")
    html = path.read_text(encoding="utf-8")
    if 'name="robots" content="noindex,nofollow"' not in html:
        raise ValueError(f"{label} page is not marked noindex")
    return html


def _require_markers(html: str, markers: tuple[str, ...], *, label: str) -> None:
    for marker in markers:
        if marker not in html:
            raise ValueError(f"{label} page is missing {marker!r}")


def _validate_route_contract(site_root: Path) -> None:
    detailed_pages = {
        "asr": (
            "index.html",
            "./assets/explorer.js",
            "./explorer/data/asr/manifest.json",
            ('href="./"', 'href="./speech/"', 'href="./tts/"'),
        ),
        "speech": (
            "speech/index.html",
            "../assets/explorer.js",
            "../explorer/data/speech/manifest.json",
            ('href="../"', 'href="./"', 'href="../tts/"'),
        ),
        "tts": (
            "tts/index.html",
            "../assets/explorer.js",
            "../explorer/data/tts/manifest.json",
            ('href="../"', 'href="../speech/"', 'href="./"'),
        ),
    }
    findings_pages = {
        "asr": (
            "findings/index.html",
            "../assets/app.js",
            "../data/reports.json",
        ),
        "speech": (
            "findings/speech/index.html",
            "../../assets/app.js",
            "../../data/reports.json",
        ),
        "tts": (
            "findings/tts/index.html",
            "../../assets/app.js",
            "../../data/reports.json",
        ),
    }
    alias_pages = {
        family: (
            f"explorer/{family}/index.html",
            "../../assets/explorer.js",
            f"../data/{family}/manifest.json",
        )
        for family in FAMILIES
    }

    for family, (path, script, manifest, nav_links) in detailed_pages.items():
        label = f"canonical {family} detailed explorer"
        html = _require_page(site_root, path, label=label)
        _require_markers(
            html,
            (
                'class="detailed-explorer"',
                f'data-family="{family}"',
                f'data-manifest-url="{manifest}"',
                f'src="{script}?v={EXPLORER_ASSET_VERSION}"',
                (
                    f'href="{script.removesuffix("explorer.js")}explorer.css'
                    f'?v={EXPLORER_ASSET_VERSION}"'
                ),
                f'<link rel="canonical" href="{PUBLIC_BASE}{CANONICAL_DETAILED_ROUTES[family].removeprefix(SITE_PREFIX)}">',
                *nav_links,
            ),
            label=label,
        )
        if "assets/app.js" in html:
            raise ValueError(f"{label} uses the findings renderer")

    for family, (path, script, data_url) in findings_pages.items():
        label = f"{family} findings"
        html = _require_page(site_root, path, label=label)
        _require_markers(
            html,
            (
                f'data-family="{family}"',
                f'data-data-url="{data_url}"',
                f'src="{script}"',
            ),
            label=label,
        )
        if 'class="detailed-explorer"' in html or "assets/explorer.js" in html:
            raise ValueError(f"{label} uses the detailed-explorer renderer")

    canonical_alias_nav = (
        'href="../../"',
        'href="../../speech/"',
        'href="../../tts/"',
    )
    for family, (path, script, manifest) in alias_pages.items():
        label = f"legacy {family} explorer alias"
        html = _require_page(site_root, path, label=label)
        canonical_suffix = CANONICAL_DETAILED_ROUTES[family].removeprefix(SITE_PREFIX)
        _require_markers(
            html,
            (
                'class="detailed-explorer"',
                f'data-family="{family}"',
                f'data-manifest-url="{manifest}"',
                f'src="{script}?v={EXPLORER_ASSET_VERSION}"',
                (
                    f'href="{script.removesuffix("explorer.js")}explorer.css'
                    f'?v={EXPLORER_ASSET_VERSION}"'
                ),
                f'<link rel="canonical" href="{PUBLIC_BASE}{canonical_suffix}">',
                *canonical_alias_nav,
            ),
            label=label,
        )
        if "assets/app.js" in html:
            raise ValueError(f"{label} uses the findings renderer")

    site_manifest = _load(site_root / "site-manifest.json")
    expected_routes = {
        "detailed_cached_explorers": list(CANONICAL_DETAILED_ROUTES.values()),
        "findings": list(FINDINGS_ROUTES.values()),
        "legacy_explorer_aliases": list(LEGACY_EXPLORER_ROUTES.values()),
    }
    routes = site_manifest.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("site manifest has no route map")
    for name, expected in expected_routes.items():
        if routes.get(name) != expected:
            raise ValueError(f"site manifest has an invalid {name} route list")


def validate_site(site_root: Path) -> dict[str, int]:
    site_root = site_root.resolve()
    counts: dict[str, int] = {}
    referenced_media: set[Path] = set()
    audio_urls_by_family: dict[str, list[str]] = {}
    for asset in (
        "assets/explorer.js",
        "assets/explorer.css",
        "assets/app.js",
        "assets/styles.css",
    ):
        if not (site_root / asset).is_file():
            raise ValueError(f"missing static-site asset: {asset}")
    for asset in ("assets/explorer.js", "assets/app.js"):
        script = (site_root / asset).read_text(encoding="utf-8")
        if (
            "/api/" in script
            or 'method: "POST"' in script
            or "method: 'POST'" in script
        ):
            raise ValueError(f"published {asset} contains a live API call")
    explorer_script = (site_root / "assets/explorer.js").read_text(encoding="utf-8")
    if 'URLSearchParams(window.location.search).get("sample")' not in explorer_script:
        raise ValueError("static explorer does not preserve ?sample selection")
    for marker in (
        "function renderSpeechRows()",
        "const windowSize = 8",
        'class="speech-matrix-window"',
        'family === "speech" ? renderSpeechRows()',
        "cell?.realized_token",
        'class="realized-rank-badge"',
        "showASRRealizedRank",
        "realized_rank_by_max_length",
        "expectedReportCount = 10",
        "payload.report_count !== expectedReportCount",
        'id="sample-search"',
        'class="sample-button-grid"',
        *SPEECH_TERMINATION_SCRIPT_MARKERS,
    ):
        if marker not in explorer_script:
            raise ValueError(
                "static explorer is missing the readable speech-band contract: "
                f"{marker}"
            )
    explorer_css = (site_root / "assets/explorer.css").read_text(encoding="utf-8")
    for marker in (
        ".position-timeline.speech-readable",
        ".speech-matrix-window",
        ".speech-matrix-grid",
        ".realized-rank-badge",
        '[data-family="asr"] .matrix-cell .realized-rank-badge',
        ".sample-picker-tools",
        ".sample-button-grid",
        "overflow-x: hidden",
        *SPEECH_TERMINATION_CSS_MARKERS,
    ):
        if marker not in explorer_css:
            raise ValueError(
                "static explorer CSS is missing the readable speech-band "
                f"contract: {marker}"
            )

    _validate_route_contract(site_root)

    for family in FAMILIES:
        manifest_path = site_root / "explorer" / "data" / family / "manifest.json"
        manifest = _load(manifest_path)
        if (
            manifest.get("schema_id") != "audio-jacobian-lens.cached-explorer-manifest"
            or manifest.get("family") != family
            or manifest.get("mode") != "static_cached_explorer"
        ):
            raise ValueError(f"invalid {family} manifest envelope")
        reports = _manifest_reports(manifest, family=family)
        if not manifest.get("provenance", {}).get("lens"):
            raise ValueError(f"{family} manifest has no pinned lens provenance")
        lens_provenance = manifest["provenance"]["lens"]
        if family == "asr":
            if "source_layers" in lens_provenance:
                raise ValueError(
                    "ASR lens provenance has an ambiguous source_layers field"
                )
            if not isinstance(lens_provenance.get("encoder_source_layers"), list):
                raise ValueError("ASR lens provenance has no encoder source layers")
            if not isinstance(lens_provenance.get("decoder_source_layers"), list):
                raise ValueError("ASR lens provenance has no decoder source layers")

        expected_family_files = {manifest_path.resolve()}
        for entry in reports:
            report_path = _site_path(site_root, str(entry["report_url"]))
            expected_parent = (site_root / "explorer" / "data" / family).resolve()
            if report_path.resolve().parent != expected_parent:
                raise ValueError(f"{family} report URL is outside its family directory")
            expected_family_files.add(report_path.resolve())
            if not report_path.is_file() or _sha256(report_path) != entry["sha256"]:
                raise ValueError(f"{family} report hash mismatch: {report_path}")
            if report_path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"{family} report byte count mismatch: {report_path}")
            report = _load(report_path)
            if (
                report.get("family") != family
                or report.get("example_id") != entry["id"]
            ):
                raise ValueError(f"{family} report/manifest identity mismatch")
            _validate_safe(report)
            if family == "tts":
                _validate_tts(report)
            else:
                _validate_asr_or_speech(report, family=family)
                if family == "asr":
                    if (
                        report["payload"]["encoder"]["layers"]
                        != lens_provenance["encoder_source_layers"]
                    ):
                        raise ValueError("ASR encoder layers disagree with provenance")
                    if (
                        report["payload"]["decoder"]["layers"]
                        != lens_provenance["decoder_source_layers"]
                    ):
                        raise ValueError("ASR decoder layers disagree with provenance")
                audio_path = _validate_audio_reference(
                    site_root,
                    family=family,
                    report=report,
                    entry=entry,
                )
                referenced_media.add(audio_path.resolve())
                if family == "asr":
                    _validate_filter_cache(site_root, report, entry)
                    filter_path = _site_path(
                        site_root,
                        str(entry["character_length_filter_cache"]["url"]),
                    )
                    if filter_path.resolve().parent != expected_parent:
                        raise ValueError(
                            "ASR filter URL is outside its family directory"
                        )
                    expected_family_files.add(filter_path.resolve())
        family_files = {
            path.resolve()
            for path in manifest_path.parent.glob("*.json")
            if path.is_file()
        }
        if family_files != expected_family_files:
            raise ValueError(
                f"{family} explorer data contains unreferenced or missing JSON files"
            )
        if family != "tts":
            audio_urls_by_family[family] = [
                str(entry["audio_url"]) for entry in reports
            ]
        counts[family] = len(reports)

    if audio_urls_by_family.get("asr") != audio_urls_by_family.get("speech"):
        raise ValueError(
            "ASR and speech manifests must use the same ordered input-audio set"
        )
    media = [
        path
        for path in site_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}
    ]
    if {path.resolve() for path in media} != referenced_media:
        raise ValueError(
            "the static site media set does not match the cleared manifest inputs"
        )
    _validate_site_manifest_integrity(site_root, counts=counts)
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "site_root",
        type=Path,
        help="Path to the published audio-jacobian-lens directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    counts = validate_site(args.site_root)
    print(
        "validated static explorer: "
        + ", ".join(f"{family}={count}" for family, count in counts.items())
    )


if __name__ == "__main__":
    main()
