from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import export_static_tts_explorer as exporter

ROOT = Path(__file__).resolve().parent.parent


def test_pinned_tts_catalog_has_exact_public_examples_and_provenance() -> None:
    catalog = exporter.load_catalog(ROOT / "data" / "static_public_reports_v1.json")

    assert tuple(example.example_id for example in catalog.examples) == (
        "tts-bridge-s9",
        "tts-turtles-monotonic",
        "tts-music-nonmonotonic",
    )
    assert catalog.examples[0].prompt == (
        "A bright red train crossed the narrow bridge."
    )
    assert catalog.provenance["model"]["revision"] == (
        "2f2e21a03863f86a1274d1060dcc188e7cde77e1"
    )
    assert catalog.provenance["lens"]["projection_rank"] == 128


def test_candidate_reducer_records_log_probability_not_absolute_logit() -> None:
    candidate = exporter._candidate({"id": 4133, "probability": 0.07596})

    assert candidate["id"] == 4133
    assert candidate["probability"] == pytest.approx(0.07596)
    assert candidate["log_probability"] == pytest.approx(math.log(0.07596))

    with pytest.raises(ValueError, match="probability"):
        exporter._candidate({"id": 1, "probability": 1.1})


def test_trace_sanitizer_drops_ephemeral_handle_and_keeps_diagnostics() -> None:
    trace = exporter.sanitize_trace(
        {
            "analysis_id": "a" * 32,
            "selection": {
                "speech_code_index": 0,
                "speech_code_id": 4133,
                "start_seconds": 0.0,
                "end_seconds": 0.04,
                "raw_probability": 0.07596,
            },
            "text_tokens": [
                {"index": 0, "id": 32, "text": "A", "char_start": 0, "char_end": 1}
            ],
            "layers": [0],
            "gradient_l2": [[0.5]],
            "gradient_share": [[1.0]],
            "gradient_text_mass": [0.2],
            "attention_share": [[1.0]],
            "attention_text_mass": [0.3],
            "target_log_probability": -2.0,
            "score_kind": "raw_log_probability_gradient_l2",
            "attention_kind": "text_prefix_causal_self_attention",
            "warnings": [],
        }
    )

    assert "analysis_id" not in trace
    assert trace["selection"]["speech_code_id"] == 4133
    assert trace["gradient_share"] == [[1.0]]
    exporter.validate_safe(trace)


@pytest.mark.parametrize(
    "unsafe",
    [
        {"analysis_id": "a" * 32},
        {"output": {"audio_data_url": "data:audio/wav;base64,AAAA"}},
        {"output": {"waveform": [0.0]}},
        {"generated_audio": "artifact.wav"},
    ],
)
def test_tts_safety_validator_rejects_handles_and_generated_audio(
    unsafe: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        exporter.validate_safe(unsafe)


def test_curated_bridge_merge_checks_live_s9_values() -> None:
    catalog = exporter.load_catalog(ROOT / "data" / "static_public_reports_v1.json")
    bridge = catalog.examples[0]
    baseline_probability = 0.139585892242929
    candidate_probability = 0.12639334319818746
    codes = [{"id": index} for index in range(9)]
    codes[8] = {"id": 4106}
    payload = {
        "output": {
            "speech_codes": codes,
            "speech_head_candidates": {
                "target_probabilities": [0.1] * 8 + [baseline_probability],
                "target_log_probabilities": [-math.log(10)] * 8
                + [math.log(baseline_probability)],
                "target_ranks": [1] * 9,
                "top_codes": [[{"id": index, "probability": 0.1}] for index in range(8)]
                + [[
                    {"id": 4106, "probability": baseline_probability},
                    {"id": 4358, "probability": candidate_probability},
                ]],
            },
        }
    }

    comparison = exporter.merge_curated_bridge_intervention(bridge, payload)

    assert comparison is not None
    assert comparison["baseline"]["realized_code_id"] == 4106
    assert comparison["candidate"]["code_id"] == 4358
    assert comparison["residual_steered"][
        "chosen_relative_residual_norm"
    ] == pytest.approx(0.001708984375)
    assert comparison["direct_forced_sequence_exact_match"] is True
