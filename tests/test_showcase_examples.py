from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_PATH = ROOT / "web" / "data" / "showcase-examples.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_lfm_showcase_evidence_is_self_consistent_and_rights_scoped():
    payload = json.loads(SHOWCASE_PATH.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    layers = provenance["lens"]["source_layers"]
    rank_semantics = provenance["rank_semantics"]
    rank_denominator = rank_semantics["rank_denominator"]

    assert payload["schema_version"] == 1
    assert layers == [0, 4, 8, 12, 14]
    assert rank_semantics["tie_policy"] == "1_plus_count_strictly_greater"
    assert provenance["rights_policy"]["excluded_audio"].startswith(
        "Generated LFM speech"
    )
    lens_path = ROOT / provenance["lens"]["source_path"]
    assert _sha256(lens_path) == provenance["lens"]["sha256"]

    examples = payload["examples"]
    assert [example["id"] for example in examples] == [
        "question-in-sample",
        "universe-held-out",
        "buzzer-held-out-budget-failure",
    ]
    assert {example["fit_relationship"]["kind"] for example in examples} == {
        "in_sample_input",
        "held_out_input",
    }

    for example in examples:
        source = ROOT / example["input"]["source_path"]
        assert source.is_file()
        assert _sha256(source) == example["input"]["sha256"]
        assert example["input"]["license"] == "CC BY 4.0"
        assert example["input"]["rights_status"] == "cleared_with_attribution"
        assert example["generated"]["audio_included"] is False

        termination = example["generated"]["termination"]
        eos_steps = int(termination["audio_eos_seen"])
        assert termination["generated_steps"] == (
            termination["text_tokens_including_controls"]
            + termination["audio_frames_excluding_eos"]
            + eos_steps
        )
        expected_duration = (
            termination["audio_frames_excluding_eos"]
            * provenance["evaluation_generation"]["audio_frame_seconds"]
        )
        assert abs(
            example["generated"]["decoded_audio_duration_seconds"]
            - expected_duration
        ) < 1e-9
        assert example["analysis_target_count"] == (
            termination["text_tokens_including_controls"] - 1
        )

        metrics = example["aggregate_rank_metrics"]
        assert [metric["layer"] for metric in metrics] == layers
        for metric in metrics:
            counts = [
                metric["top_1_count"],
                metric["top_5_count"],
                metric["top_10_count"],
                metric["top_100_count"],
                example["analysis_target_count"],
            ]
            assert counts == sorted(counts)
            assert 1 <= metric["median_rank"] <= rank_denominator
            assert 0 < metric["mean_reciprocal_rank"] <= 1

        assert example["tracked_tokens"]
        for token in example["tracked_tokens"]:
            assert len(token["ranks"]) == len(layers)
            assert all(1 <= rank <= rank_denominator for rank in token["ranks"])
            assert token["head_rank"] == 1
            assert 0 < token["head_probability"] <= 1
            assert "<|" not in token["text"]

    serialized = json.dumps(payload)
    assert "analysis_id" not in serialized
    assert "data:audio" not in serialized
    assert "macOS-TTS" in serialized
