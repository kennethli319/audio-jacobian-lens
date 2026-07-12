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
        "vocabulary_filter": {
            "decoded_character_length": 3,
            "display_lexical_eligible": True,
        },
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


def _asr_report() -> dict:
    head = {
        **_ranked_token(42, rank=2),
        "probability": 0.2,
        "log_probability": -1.609,
        "start_seconds": 0.0,
        "end_seconds": 0.2,
        "top_tokens": [{"id": 99, "text": " a", "probability": 0.4}],
    }
    decoder_cell = {
        "top_tokens": [{"id": 99, "text": " a", "score": 0.8}],
        "realized_token": _ranked_token(42, rank=15, score=0.25),
    }
    encoder_cell = {
        **decoder_cell,
        "time_window": {"start_seconds": 0.0, "end_seconds": 0.2},
        "realized_token_position": 0,
        "realized_token_alignment": {
            "match": "overlapping",
            "window_midpoint_seconds": 0.1,
            "token_start_seconds": 0.0,
            "token_end_seconds": 0.2,
            "overlap_seconds": 0.2,
            "overlap_fraction_of_window": 1.0,
        },
    }
    return {
        "example_id": "asr-example",
        "source": {"rights_status": "cleared_with_attribution"},
        "payload": {
            "metadata": {
                "display_vocabulary": {
                    "maximum_decoded_character_length_counts": {
                        "1": 10,
                        "2": 20,
                        "3": 30,
                    }
                }
            },
            "audio": {"waveform_preview": {"values": [0.0, 0.1]}},
            "transcription": {"tokens": [head]},
            "encoder": {
                "layers": [0],
                "realized_token_alignment": {
                    "method": "maximum_token_interval_overlap",
                    "tie_break": (
                        "closest_interval_midpoint_then_lower_token_position"
                    ),
                },
                "cells": [[encoder_cell]],
            },
            "decoder": {"layers": [0], "cells": [[decoder_cell]]},
        },
    }


def test_asr_site_validation_accepts_exact_realized_rank_provenance() -> None:
    validator._validate_asr_or_speech(_asr_report(), family="asr")


def test_asr_site_validation_rejects_shifted_encoder_token_mapping() -> None:
    report = _asr_report()
    report["payload"]["encoder"]["cells"][0][0][
        "realized_token_position"
    ] = 1

    with pytest.raises(ValueError, match="overlap-first"):
        validator._validate_asr_or_speech(report, family="asr")


def test_asr_filter_cache_requires_exact_compact_realized_ranks(
    tmp_path: Path,
) -> None:
    report = _asr_report()
    cache = {
        "example_id": "asr-example",
        "streams": {
            "encoder": {
                "layers": [0],
                "cells": [[{
                    "top_tokens_by_length": {"1": [{"id": 99}]},
                    "realized_rank_by_max_length": {
                        "1": None,
                        "2": None,
                        "3": 7,
                    },
                }]],
            },
            "decoder": {
                "layers": [0],
                "cells": [[{
                    "top_tokens_by_length": {"1": [{"id": 99}]},
                    "realized_rank_by_max_length": {
                        "1": None,
                        "2": None,
                        "3": 8,
                    },
                }]],
            },
        },
    }
    cache_path = tmp_path / "filters.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    entry = {
        "character_length_filter_cache": {
            "url": "/audio-jacobian-lens/filters.json",
            "sha256": validator._sha256(cache_path),
        }
    }
    validator._validate_filter_cache(tmp_path, report, entry)

    del cache["streams"]["encoder"]["cells"][0][0][
        "realized_rank_by_max_length"
    ]
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    entry["character_length_filter_cache"]["sha256"] = validator._sha256(
        cache_path
    )
    with pytest.raises(ValueError, match="no exact realized ranks"):
        validator._validate_filter_cache(tmp_path, report, entry)
