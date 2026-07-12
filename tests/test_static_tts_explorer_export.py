from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import export_static_tts_explorer as exporter

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "static_explorer_catalog_v2.json"
CURATED_PATH = ROOT / "data" / "static_public_reports_v1.json"


def test_pinned_tts_catalog_has_exact_public_examples_and_provenance() -> None:
    catalog = exporter.load_catalog(CATALOG_PATH, CURATED_PATH)

    assert tuple(example.example_id for example in catalog.examples) == (
        "tts-bridge-s9",
        "tts-turtles-monotonic",
        "tts-music-nonmonotonic",
        "tts-voice-question",
        "tts-record-in-context",
        "tts-code-nine-zero-four",
        "tts-ruby-ribbon",
        "tts-door-two-clause",
        "tts-linh-nguyen",
        "tts-first-then",
    )
    assert catalog.report_count == 10
    assert catalog.examples[0].prompt == (
        "A bright red train crossed the narrow bridge."
    )
    assert catalog.examples[0].intervention is not None
    assert all(example.intervention is None for example in catalog.examples[1:])
    assert catalog.provenance["model"]["revision"] == (
        "2f2e21a03863f86a1274d1060dcc188e7cde77e1"
    )
    assert catalog.provenance["lens"]["projection_rank"] == 128


def test_tts_exporter_direct_script_help_works_without_editable_install() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_static_tts_explorer.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--example-id" in result.stdout
    assert "--resume-valid" in result.stdout


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
    catalog = exporter.load_catalog(CATALOG_PATH, CURATED_PATH)
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
                + [
                    [
                        {"id": 4106, "probability": baseline_probability},
                        {"id": 4358, "probability": candidate_probability},
                    ]
                ],
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


def test_generation_cap_guard_rejects_possibly_truncated_run() -> None:
    source = {"model": {"generation": {"max_speech_tokens": 3}}}

    with pytest.raises(ValueError, match="possibly truncated"):
        exporter.reject_generation_cap(
            source,
            {"output": {"speech_codes": [{}, {}, {}]}},
        )

    exporter.reject_generation_cap(
        source,
        {"output": {"speech_codes": [{}, {}]}},
    )


def test_resume_validation_rejects_cached_report_at_generation_cap(
    tmp_path: Path,
) -> None:
    example = _example("tts-capped")
    report = {
        "schema_id": exporter.SCHEMA_ID,
        "schema_version": exporter.SCHEMA_VERSION,
        "family": "tts",
        "example_id": example.example_id,
        "title": example.title,
        "teaching_role": example.teaching_role,
        "teaching_purpose": example.teaching_purpose,
        "provenance": {},
        "source": {"prompt": example.prompt},
        "payload": {
            "model": {"generation": {"max_speech_tokens": 2}},
            "output": {"speech_codes": [{}, {}]},
        },
    }
    path = tmp_path / "tts-capped.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="possibly truncated"):
        exporter._validate_existing_report(
            path,
            example=example,
            provenance={},
        )


def _example(example_id: str) -> exporter.TTSExample:
    return exporter.TTSExample(
        example_id=example_id,
        title=f"Title {example_id}",
        teaching_role=f"role_{example_id}",
        teaching_purpose=f"Purpose {example_id}",
        prompt=f"Prompt {example_id}.",
        selected_position=None,
        intervention=None,
    )


def test_partial_export_reuses_unselected_report_and_writes_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _example("tts-first")
    second = _example("tts-second")
    catalog = exporter.TTSCatalog(
        label="TTS",
        description="Two reports",
        provenance={},
        examples=(first, second),
        report_count=2,
    )
    first_path = tmp_path / "tts-first.json"
    first_path.write_text("preserved\n", encoding="utf-8")
    original = first_path.read_bytes()

    def validate_existing(path: Path, **_: object) -> tuple[str, int]:
        assert path == first_path
        return "a" * 64, len(original)

    calls: list[str] = []

    def post_json(url: str, *_: object, **__: object) -> dict:
        calls.append(url)
        return {"analysis_id": "run", "model": {}}

    monkeypatch.setattr(exporter, "_validate_existing_report", validate_existing)
    monkeypatch.setattr(exporter, "_post_json", post_json)
    monkeypatch.setattr(
        exporter,
        "sanitize_generation",
        lambda _: {"output": {"speech_codes": []}},
    )
    monkeypatch.setattr(exporter, "validate_report", lambda _: None)
    monkeypatch.setattr(
        exporter,
        "_build_report",
        lambda example, provenance, payload: {"example_id": example.example_id},
    )

    manifest = exporter.export_tts_family(
        base_url="http://example.test",
        output_dir=tmp_path,
        catalog=catalog,
        timeout=1.0,
        example_ids={"tts-second"},
    )

    assert first_path.read_bytes() == original
    assert manifest["report_count"] == 2
    assert [entry["id"] for entry in manifest["reports"]] == [
        "tts-first",
        "tts-second",
    ]
    assert manifest["reports"][1]["prompt"] == second.prompt
    assert manifest["reports"][1]["summary"] == second.teaching_purpose
    assert calls == ["http://example.test/api/chatterbox/generate"]


def test_partial_export_rejects_missing_unselected_report(tmp_path: Path) -> None:
    first = _example("tts-first")
    second = _example("tts-second")
    catalog = exporter.TTSCatalog(
        label="TTS",
        description="Two reports",
        provenance={},
        examples=(first, second),
        report_count=2,
    )

    with pytest.raises(ValueError, match="unselected report tts-first is missing"):
        exporter.export_tts_family(
            base_url="http://example.test",
            output_dir=tmp_path,
            catalog=catalog,
            timeout=1.0,
            example_ids={"tts-second"},
        )


def test_resume_valid_reuses_selected_reports_without_api_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = _example("tts-existing")
    catalog = exporter.TTSCatalog(
        label="TTS",
        description="One report",
        provenance={},
        examples=(example,),
        report_count=1,
    )
    report_path = tmp_path / "tts-existing.json"
    report_path.write_text("preserved\n", encoding="utf-8")
    monkeypatch.setattr(
        exporter,
        "_validate_existing_report",
        lambda *args, **kwargs: ("b" * 64, report_path.stat().st_size),
    )
    monkeypatch.setattr(
        exporter,
        "_post_json",
        lambda *args, **kwargs: pytest.fail("resume should not call the API"),
    )

    manifest = exporter.export_tts_family(
        base_url="http://example.test",
        output_dir=tmp_path,
        catalog=catalog,
        timeout=1.0,
        resume_valid=True,
    )

    assert manifest["report_count"] == 1
    assert manifest["reports"][0]["sha256"] == "b" * 64
    assert json.loads((tmp_path / "manifest.json").read_text())["report_count"] == 1
