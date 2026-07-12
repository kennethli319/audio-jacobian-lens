from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_static_explorer_site as validator


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _noindex(body: str) -> str:
    return (
        '<meta name="robots" content="noindex,nofollow">'
        f"<html><body {body}></body></html>"
    )


def _build_route_fixture(site_root: Path) -> None:
    canonical = {
        "asr": (
            "index.html",
            "./assets/explorer.js",
            "./explorer/data/asr/manifest.json",
            validator.PUBLIC_BASE,
            ('href="./"', 'href="./speech/"', 'href="./tts/"'),
        ),
        "speech": (
            "speech/index.html",
            "../assets/explorer.js",
            "../explorer/data/speech/manifest.json",
            f"{validator.PUBLIC_BASE}speech/",
            ('href="../"', 'href="./"', 'href="../tts/"'),
        ),
        "tts": (
            "tts/index.html",
            "../assets/explorer.js",
            "../explorer/data/tts/manifest.json",
            f"{validator.PUBLIC_BASE}tts/",
            ('href="../"', 'href="../speech/"', 'href="./"'),
        ),
    }
    for family, (path, script, manifest, canonical_url, links) in canonical.items():
        _write(
            site_root / path,
            _noindex(
                f'class="detailed-explorer" data-family="{family}" '
                f'data-manifest-url="{manifest}"'
            )
            + f'<link rel="canonical" href="{canonical_url}">'
            + f'<script src="{script}"></script>'
            + "".join(links),
        )

    findings = {
        "asr": ("findings/index.html", "../", "../"),
        "speech": ("findings/speech/index.html", "../../", "../../"),
        "tts": ("findings/tts/index.html", "../../", "../../"),
    }
    for family, (path, asset_prefix, data_prefix) in findings.items():
        _write(
            site_root / path,
            _noindex(
                f'data-family="{family}" '
                f'data-data-url="{data_prefix}data/reports.json"'
            )
            + f'<script src="{asset_prefix}assets/app.js"></script>',
        )

    for family in validator.FAMILIES:
        suffix = "" if family == "asr" else f"{family}/"
        _write(
            site_root / "explorer" / family / "index.html",
            _noindex(
                f'class="detailed-explorer" data-family="{family}" '
                f'data-manifest-url="../data/{family}/manifest.json"'
            )
            + (
                f'<link rel="canonical" href="{validator.PUBLIC_BASE}{suffix}">'
                '<script src="../../assets/explorer.js"></script>'
                'href="../../" href="../../speech/" href="../../tts/"'
            ),
        )

    routes = {
        "detailed_cached_explorers": list(
            validator.CANONICAL_DETAILED_ROUTES.values()
        ),
        "findings": list(validator.FINDINGS_ROUTES.values()),
        "legacy_explorer_aliases": list(
            validator.LEGACY_EXPLORER_ROUTES.values()
        ),
    }
    _write(site_root / "site-manifest.json", json.dumps({"routes": routes}))


def test_route_contract_keeps_explorers_primary_and_findings_secondary(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)

    validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_alias_navigation_to_legacy_routes(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)
    alias = tmp_path / "explorer" / "speech" / "index.html"
    html = alias.read_text(encoding="utf-8").replace(
        'href="../../speech/"', 'href="../speech/"'
    )
    alias.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="legacy speech explorer alias"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_findings_with_detailed_renderer(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)
    findings = tmp_path / "findings" / "index.html"
    html = findings.read_text(encoding="utf-8").replace(
        "../assets/app.js", "../assets/explorer.js"
    )
    findings.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="asr findings"):
        validator._validate_route_contract(tmp_path)


def _ranked_token(token_id: int, *, rank: int, score: float | None = None) -> dict:
    token = {
        "id": token_id,
        "text": " the",
        "rank": rank,
        "rank_denominator": 61_690,
        "rank_space": "lexical_display_vocabulary",
        "rank_tie_policy": "1_plus_count_strictly_greater",
        "score_kind": "raw_readout_logit" if score is not None else "raw_head_probability",
    }
    if score is not None:
        token["score"] = score
    return token


def _speech_report() -> dict:
    head = {
        **_ranked_token(42, rank=2),
        "probability": 0.2,
        "log_probability": -1.609,
        "top_tokens": [{"id": 99, "text": " a", "probability": 0.4}],
    }
    cell = {
        "top_tokens": [{"id": 99, "text": " a", "score": 0.8}],
        "realized_token": _ranked_token(42, rank=15, score=0.25),
    }
    return {
        "source": {"rights_status": "cleared_with_attribution"},
        "payload": {
            "audio": {"waveform_preview": {"values": [0.0, 0.1]}},
            "transcription": {"tokens": [head]},
            "encoder": {"layers": [], "cells": []},
            "decoder": {"layers": [0, 1], "cells": [[cell], [dict(cell)]]},
        },
    }


def test_speech_site_validation_accepts_exact_realized_rank_provenance() -> None:
    validator._validate_asr_or_speech(_speech_report(), family="speech")


def test_speech_site_validation_rejects_missing_layer_realized_rank() -> None:
    report = _speech_report()
    del report["payload"]["decoder"]["cells"][1][0]["realized_token"]

    with pytest.raises(ValueError, match="no exact realized-token provenance"):
        validator._validate_asr_or_speech(report, family="speech")
