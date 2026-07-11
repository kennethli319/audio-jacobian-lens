#!/usr/bin/env python3
"""Validate the publishable, backend-free Audio Jacobian Lens site bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SITE_PREFIX = "/audio-jacobian-lens/"
PUBLIC_BASE = "https://kennethli319.github.io/audio-jacobian-lens/"
FAMILIES = ("asr", "speech", "tts")
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


def _validate_asr_or_speech(
    report: Mapping[str, Any], *, family: str
) -> None:
    payload = report["payload"]
    tokens = payload.get("transcription", {}).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{family} report has no output tokens")
    _validate_stream(
        payload["decoder"], label=f"{family} decoder", allow_empty=False
    )
    if len(payload["decoder"]["cells"][0]) != len(tokens):
        raise ValueError(f"{family} decoder/token width mismatch")
    _validate_stream(
        payload["encoder"], label=f"{family} encoder", allow_empty=family == "speech"
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
    cache = _load(path)
    if cache.get("example_id") != report.get("example_id"):
        raise ValueError("ASR filter cache/report ID mismatch")
    for stream_name in ("encoder", "decoder"):
        filtered = cache["streams"][stream_name]
        base = report["payload"][stream_name]
        if any(layer not in base["layers"] for layer in filtered["layers"]):
            raise ValueError(f"ASR {stream_name} filter layer mismatch")
        width = len(base["cells"][0])
        if any(len(row) != width for row in filtered["cells"]):
            raise ValueError(f"ASR {stream_name} filter width mismatch")
        for row in filtered["cells"]:
            if any(not cell.get("top_tokens_by_length") for cell in row):
                raise ValueError(f"ASR {stream_name} filter cell is empty")
    _validate_safe(cache)


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
            ("href=\"./\"", "href=\"./speech/\"", "href=\"./tts/\""),
        ),
        "speech": (
            "speech/index.html",
            "../assets/explorer.js",
            "../explorer/data/speech/manifest.json",
            ("href=\"../\"", "href=\"./\"", "href=\"../tts/\""),
        ),
        "tts": (
            "tts/index.html",
            "../assets/explorer.js",
            "../explorer/data/tts/manifest.json",
            ("href=\"../\"", "href=\"../speech/\"", "href=\"./\""),
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
                f'src="{script}"',
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
                f'src="{script}"',
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
            or "method: \"POST\"" in script
            or "method: 'POST'" in script
        ):
            raise ValueError(f"published {asset} contains a live API call")
    explorer_script = (site_root / "assets/explorer.js").read_text(
        encoding="utf-8"
    )
    if 'URLSearchParams(window.location.search).get("sample")' not in explorer_script:
        raise ValueError("static explorer does not preserve ?sample selection")
    for marker in (
        "function renderSpeechRows()",
        "const windowSize = 8",
        'class="speech-matrix-window"',
        'family === "speech" ? renderSpeechRows()',
    ):
        if marker not in explorer_script:
            raise ValueError(
                "static explorer is missing the readable speech-band contract: "
                f"{marker}"
            )
    explorer_css = (site_root / "assets/explorer.css").read_text(
        encoding="utf-8"
    )
    for marker in (
        ".position-timeline.speech-readable",
        ".speech-matrix-window",
        ".speech-matrix-grid",
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
            manifest.get("schema_id")
            != "audio-jacobian-lens.cached-explorer-manifest"
            or manifest.get("family") != family
            or manifest.get("mode") != "static_cached_explorer"
        ):
            raise ValueError(f"invalid {family} manifest envelope")
        reports = manifest.get("reports")
        if not isinstance(reports, list) or len(reports) != 3:
            raise ValueError(f"{family} must publish exactly three reports")
        if not manifest.get("provenance", {}).get("lens"):
            raise ValueError(f"{family} manifest has no pinned lens provenance")

        for entry in reports:
            report_path = _site_path(site_root, str(entry["report_url"]))
            if not report_path.is_file() or _sha256(report_path) != entry["sha256"]:
                raise ValueError(f"{family} report hash mismatch: {report_path}")
            if report_path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"{family} report byte count mismatch: {report_path}")
            report = _load(report_path)
            if report.get("family") != family or report.get("example_id") != entry["id"]:
                raise ValueError(f"{family} report/manifest identity mismatch")
            _validate_safe(report)
            if family == "tts":
                _validate_tts(report)
            else:
                _validate_asr_or_speech(report, family=family)
                audio_path = _site_path(site_root, str(entry["audio_url"]))
                if not audio_path.is_file():
                    raise ValueError(f"missing cleared input audio: {audio_path}")
                if family == "asr":
                    _validate_filter_cache(site_root, report, entry)
        counts[family] = len(reports)

    media = [
        path
        for path in site_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}
    ]
    expected = {
        site_root / "audio" / "question.flac",
        site_root / "audio" / "universe.flac",
        site_root / "audio" / "buzzer.flac",
    }
    if set(media) != expected:
        raise ValueError("the static site contains media outside the three cleared inputs")
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
