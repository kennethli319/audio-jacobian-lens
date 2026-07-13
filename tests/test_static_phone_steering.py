from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "static_phone_steering_v1.json"
RUNTIME_PATH = ROOT / "web" / "data" / "phone-steering-results.json"


def _load() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _checkpoint(target: dict, checkpoint_id: str) -> dict:
    return next(
        item for item in target["checkpoints"] if item["id"] == checkpoint_id
    )


def test_runtime_phone_steering_bundle_is_byte_identical() -> None:
    assert RUNTIME_PATH.read_bytes() == DATA_PATH.read_bytes()


def test_static_phone_steering_schema_and_recorded_only_contract() -> None:
    payload = _load()

    assert set(payload) == {
        "schema_id",
        "schema_version",
        "mode",
        "title",
        "source",
        "model",
        "rank_semantics",
        "baseline",
        "targets",
        "caveats",
    }
    assert payload["schema_id"] == "audio-jacobian-lens.phone-steering"
    assert payload["schema_version"] == 1
    assert payload["mode"] == "static_recorded_checkpoints"
    assert payload["source"]["media_included"] is False
    assert set(payload["targets"]) == {"yanny", "laurel"}

    baseline = payload["baseline"]
    assert baseline["recorded"] is True
    assert baseline["interpolated"] is False
    assert baseline["generated"]["text"] == "Lily!"
    assert baseline["generated"]["token_ids"] == [20037, 0]
    assert baseline["budget_fraction"] == 0.0

    for target in payload["targets"].values():
        assert target["layers"] == [0, 1, 2, 3]
        assert [row["layer"] for row in target["coefficient_heatmap"]] == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            len(row["values"]) == len(target["schedule"])
            for row in target["coefficient_heatmap"]
        )
        assert [item["id"] for item in target["checkpoints"]] == [
            "last_failure",
            "first_success",
            "recommended",
        ]
        for item in target["checkpoints"]:
            assert item["recorded"] is True
            assert item["interpolated"] is False
            assert item["budget_fraction"] > 0
            assert item["decisions"]
            for decision in item["decisions"]:
                assert 1 <= decision["rank"] <= decision["rank_denominator"]
                assert 0.0 <= decision["probability"] <= 1.0
                assert [candidate["rank"] for candidate in decision["top_candidates"]] == [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]


def test_static_phone_steering_locks_verified_yanny_metrics() -> None:
    payload = _load()
    target = payload["targets"]["yanny"]
    baseline = payload["baseline"]["decisions"]["yanny"]

    assert target["evidence"]["tier"] == "open_loop_cross_fit_reproduced"
    assert "No coefficient is optimized" in target["method"]["coefficient_policy"]
    assert [item["phone"] for item in target["schedule"]] == [
        "Y",
        "AE",
        "N",
        "IY",
    ]
    assert baseline[0]["rank"] == 3
    assert math.isclose(baseline[0]["probability"], 0.12333051860332489)
    assert baseline[1]["rank"] == 42
    assert math.isclose(baseline[1]["probability"], 0.002656369237229228)
    assert math.isclose(
        payload["baseline"]["sequence_probability_products"]["yanny"],
        0.0003276113420724869,
    )

    failure = _checkpoint(target, "last_failure")
    success = _checkpoint(target, "first_success")
    recommended = _checkpoint(target, "recommended")
    assert failure["generated"]["text"] == "Yelly!"
    assert failure["decisions"][1]["rank"] == 2
    assert math.isclose(failure["budget_fraction"], 0.03188415535774474)
    assert success["generated"]["token_ids"] == [575, 7737, 0]
    assert success["decisions"][1]["rank"] == 1
    assert math.isclose(success["budget_fraction"], 0.03188476638358459)
    assert recommended["generated"] == {
        "text": "Yanny!",
        "token_ids": [575, 7737, 0],
        "target_match": True,
    }
    assert [item["rank"] for item in recommended["decisions"]] == [1, 1]
    assert math.isclose(
        recommended["decisions"][0]["probability"], 0.5159616470336914
    )
    assert math.isclose(
        recommended["decisions"][1]["probability"], 0.0737132579088211
    )
    assert math.isclose(
        recommended["sequence_probability_product"], 0.03803320601582527
    )
    assert "0 / 10" in json.dumps(target["controls"])


def test_static_phone_steering_locks_verified_laurel_metrics_and_asymmetry() -> None:
    payload = _load()
    target = payload["targets"]["laurel"]
    baseline = payload["baseline"]["decisions"]["laurel"][0]

    assert target["evidence"]["tier"] == (
        "target_conditioned_clip_specific_existence"
    )
    assert "optimized directly" in target["method"]["coefficient_policy"]
    assert [item["phone"] for item in target["schedule"]] == [
        "L",
        "AO",
        "R",
        "AH",
        "L",
    ]
    assert baseline["rank"] == 2463
    assert math.isclose(baseline["probability"], 0.00001089832676370861)

    failure = _checkpoint(target, "last_failure")
    success = _checkpoint(target, "first_success")
    recommended = _checkpoint(target, "recommended")
    assert failure["generated"]["text"] == "L'Oreal"
    assert failure["decisions"][0]["rank"] == 2
    assert math.isclose(failure["budget_fraction"], 0.14280088907298913)
    assert success["generated"]["token_ids"] == [43442]
    assert success["decisions"][0]["rank"] == 1
    assert math.isclose(success["budget_fraction"], 0.14300845446353724)
    assert recommended["generated"] == {
        "text": "Laurel",
        "token_ids": [43442],
        "target_match": True,
    }
    assert math.isclose(recommended["budget_fraction"], 0.1452915875040831)
    assert math.isclose(recommended["coefficient_scale"], 0.7)
    assert recommended["decisions"][0]["rank"] == 1
    assert math.isclose(
        recommended["decisions"][0]["probability"], 0.10604292899370193
    )
    controls = json.dumps(target["controls"])
    assert "failed_exact_transfer" in controls
    assert "#10 / 1.65882%" in controls


def test_static_phone_steering_contains_no_private_or_media_payload() -> None:
    payload = _load()
    serialized = json.dumps(payload)
    lowered = serialized.lower()

    forbidden_fragments = [
        "/users/",
        "artifacts/private",
        "file://",
        "http://",
        "https://",
        "data:audio",
        "audio s7",
        ".mp3",
        ".flac",
        ".wav",
        ".pt\"",
        "source_path",
        "audio_url",
        "generated_audio_url",
    ]
    assert all(fragment not in lowered for fragment in forbidden_fragments)
    assert len(DATA_PATH.read_bytes()) < 100_000

    def visit(value) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            assert len(value) <= 20
            for child in value:
                visit(child)

    visit(payload)
