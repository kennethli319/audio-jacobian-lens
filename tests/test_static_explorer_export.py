from __future__ import annotations

import json

import pytest

from scripts import export_static_explorer as exporter


def _candidate(token_id: int, score: float) -> dict[str, object]:
    return {
        "id": token_id,
        "text": f" token-{token_id}",
        "score": score,
        "score_kind": "raw_readout_logit",
        "rank": 1,
        "rank_denominator": 10,
        "rank_space": "lexical_display_vocabulary",
    }


def _cell(position: int, *, with_filter: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "position_index": position,
        "time_window": {
            "start_seconds": position * 0.2,
            "end_seconds": (position + 1) * 0.2,
        },
        "top_tokens": [_candidate(100 + position, 1.0 - position)],
    }
    if with_filter:
        result["top_tokens_by_length"] = {
            "1": [_candidate(10 + position, 0.9 - position)],
            "2": [_candidate(20 + position, 0.8 - position)],
        }
    return result


def _raw_payload() -> dict[str, object]:
    return {
        "metadata": {
            "model_id": "example/model",
            "capabilities": {
                "generated_text": True,
                "generated_audio": True,
            },
        },
        "audio": {
            "duration_seconds": 0.4,
            "model_input_format": "mono_16khz",
            "waveform": [float(index) for index in range(2048)],
        },
        "transcription": {
            "text": " token",
            "tokens": [
                {
                    **_candidate(42, 0.5),
                    "probability": 0.25,
                    "top_tokens": [_candidate(42, 0.5)],
                }
            ],
        },
        "encoder": {
            "layers": [0, 1],
            "cells": [
                [_cell(0, with_filter=True), _cell(1, with_filter=True)],
                [_cell(0, with_filter=True), _cell(1, with_filter=True)],
            ],
        },
        "decoder": {
            "layers": [0, 1, 2],
            "cells": [
                [_cell(0, with_filter=True), _cell(1, with_filter=True)],
                [_cell(0, with_filter=True), _cell(1, with_filter=True)],
                [_cell(0, with_filter=False), _cell(1, with_filter=False)],
            ],
        },
    }


def test_reduced_payload_keeps_bounded_views_and_drops_generated_audio() -> None:
    reduced = exporter._reduce_payload(_raw_payload())

    assert reduced["metadata"]["model_id"] == "example/model"
    assert reduced["metadata"]["capabilities"] == {"generated_text": True}
    assert "waveform" not in reduced["audio"]
    preview = reduced["audio"]["waveform_preview"]
    assert preview["source_sample_count"] == 2048
    assert len(preview["values"]) == exporter.WAVEFORM_PREVIEW_POINTS
    assert "top_tokens_by_length" not in reduced["encoder"]["cells"][0][0]


def test_filter_cache_is_separate_and_aligned_with_base_report() -> None:
    raw = _raw_payload()
    report = {
        "example_id": "asr-example",
        "payload": exporter._reduce_payload(raw),
    }
    cache = exporter._filter_cache(raw, example_id="asr-example")

    exporter._validate_matrix(report)
    exporter._validate_filter_cache(cache, report)

    assert cache["streams"]["encoder"]["layers"] == [0, 1]
    assert cache["streams"]["decoder"]["layers"] == [0, 1]
    assert cache["streams"]["decoder"]["cells"][0][0][
        "top_tokens_by_length"
    ]["2"][0]["id"] == 20


def test_filter_cache_validation_rejects_shifted_coordinates() -> None:
    raw = _raw_payload()
    report = {
        "example_id": "asr-example",
        "payload": exporter._reduce_payload(raw),
    }
    cache = exporter._filter_cache(raw, example_id="asr-example")
    cache["streams"]["encoder"]["cells"][0][1]["position_index"] = 99

    with pytest.raises(ValueError, match="coordinate mismatch"):
        exporter._validate_filter_cache(cache, report)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"analysis_id": "ephemeral"},
        {"nested": {"parent_analysis_id": "ephemeral"}},
        {"audio": "data:audio/wav;base64,AAAA"},
    ],
)
def test_safety_validator_rejects_ephemeral_ids_and_embedded_audio(
    unsafe: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        exporter._validate_safe(unsafe)


def test_compact_json_writer_has_stable_single_line_payload(tmp_path) -> None:
    path = tmp_path / "filters.json"
    payload = {"schema_id": exporter.FILTER_SCHEMA_ID, "values": [1, 2, 3]}

    exporter._write_json(path, payload, compact=True)

    rendered = path.read_text(encoding="utf-8")
    assert rendered.count("\n") == 1
    assert json.loads(rendered) == payload
