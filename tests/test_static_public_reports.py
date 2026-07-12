from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_PATH = ROOT / "data" / "static_public_reports_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_static_public_report_bundle_is_complete_and_rights_scoped():
    payload = json.loads(REPORTS_PATH.read_text(encoding="utf-8"))

    assert payload["schema_id"] == "audio-jacobian-lens.cached-reports"
    assert payload["schema_version"] == 1
    assert payload["mode"] == "static_cached_reports"
    assert payload["family_order"] == ["asr", "speech", "tts"]

    families = payload["families"]
    asr_lens = families["asr"]["provenance"]["lens"]
    assert "source_layers" not in asr_lens
    assert asr_lens["encoder_source_layers"] == [0, 1, 2, 3]
    assert asr_lens["decoder_source_layers"] == [0, 1, 2]
    assert asr_lens["target_layer"] == 3
    expected_ids = {
        "asr": {
            "asr-question-late-emergence",
            "asr-universe-late-content",
            "asr-buzzer-model-error",
        },
        "speech": {
            "question-in-sample",
            "universe-held-out",
            "buzzer-held-out-budget-failure",
        },
        "tts": {
            "tts-bridge-s9",
            "tts-turtles-monotonic",
            "tts-music-nonmonotonic",
        },
    }

    for family_name, family_ids in expected_ids.items():
        examples = families[family_name]["examples"]
        assert len(examples) == 3
        assert {example["id"] for example in examples} == family_ids

        for example in examples:
            layers = example["layers"]
            assert layers
            assert example["tracks"]
            for track in example["tracks"]:
                ranks = track["ranks"]
                denominators = track["rank_denominators"]
                assert len(ranks) == len(layers)
                assert len(denominators) == len(layers)
                assert len(track["rank_spaces"]) == len(layers)
                assert all(
                    1 <= int(rank) <= int(denominator)
                    for rank, denominator in zip(
                        ranks, denominators, strict=True
                    )
                )

            source_path = example.get("input", {}).get("source_path")
            if source_path:
                source = ROOT / source_path
                assert source.is_file()
                assert _sha256(source) == example["input"]["sha256"]
                assert example["input"]["license"] == "CC BY 4.0"
                assert (
                    example["input"]["rights_status"]
                    == "cleared_with_attribution"
                )

    tts_examples = families["tts"]["examples"]
    assert all(
        example["input"].get("audio_asset") is None
        for example in tts_examples
    )
    assert tts_examples[0]["intervention"]["kind"] == (
        "replayed_residual_steering"
    )

    serialized = json.dumps(payload)
    assert "analysis_id" not in serialized
    assert "data:audio" not in serialized
    assert "generated_audio_url" not in serialized
