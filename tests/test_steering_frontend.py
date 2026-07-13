from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "web" / "steering.html"
SCRIPT_PATH = ROOT / "web" / "steering.js"
STYLE_PATH = ROOT / "web" / "steering.css"
CANONICAL_DATA_PATH = ROOT / "data" / "static_phone_steering_v1.json"
RUNTIME_DATA_PATH = ROOT / "web" / "data" / "phone-steering-results.json"


def test_phone_steering_page_is_static_accessible_and_has_no_audio_embed() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'data-results-url="./data/phone-steering-results.json"' in html
    assert 'data-target="yanny" aria-pressed="true"' in html
    assert 'data-target="laurel" aria-pressed="false"' in html
    assert 'id="checkpoint-range" type="range"' in html
    assert 'aria-live="polite" aria-atomic="true"' in html
    assert 'role="table" aria-label="Steering coefficients' in html
    assert 'href="#steering-main"' in html
    assert "No LM-head bias" in html
    assert "no browser interpolation" not in html.lower()  # supplied dynamically
    assert "<audio" not in html
    assert "<video" not in html
    assert "<iframe" not in html


def test_phone_steering_renderer_requires_recorded_checkpoints_and_full_head() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'data.mode !== "static_recorded_checkpoints"' in script
    assert "checkpoint.recorded !== true || checkpoint.interpolated !== false" in script
    assert "baseline.decisions[targetKey]" in script
    assert "baseline.sequence_probability_products?.[targetKey]" in script
    assert "target.evidence?.tone" in script
    assert "target.method?.coefficient_policy" in script
    assert "checkpoint.coefficient_scale" in script
    assert ".slice(0, 5)" in script
    assert "rank_denominator" in script
    assert "sequence_probability_product" in script
    assert "fetch(resultsUrl" in script
    assert 'method: "POST"' not in script
    assert "/api/" not in script
    assert "AudioContext" not in script
    assert "MediaRecorder" not in script


def test_phone_steering_styles_are_isolated_responsive_and_keyboard_visible() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")

    assert ".coefficient-heatmap" in css
    assert ".heatmap-cell" in css
    assert ".decision-grid" in css
    assert ".candidate-row.target" in css
    assert ".evidence-badge.strong" in css
    assert ".evidence-badge.limited" in css
    assert ":focus-visible" in css
    assert "@media (max-width: 560px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_runtime_steering_data_is_the_validated_canonical_derivative() -> None:
    canonical = json.loads(CANONICAL_DATA_PATH.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_DATA_PATH.read_text(encoding="utf-8"))

    assert runtime == canonical
    assert runtime["mode"] == "static_recorded_checkpoints"
    assert runtime["source"]["media_included"] is False
    assert runtime["targets"]["yanny"]["evidence"]["tier"] != runtime["targets"]["laurel"]["evidence"]["tier"]
    assert all(
        checkpoint["recorded"] is True and checkpoint["interpolated"] is False
        for target in runtime["targets"].values()
        for checkpoint in target["checkpoints"]
    )


def test_phone_steering_assets_are_packaged() -> None:
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for path in (
        "web/steering.css",
        "web/steering.html",
        "web/steering.js",
        "web/data/phone-steering-results.json",
    ):
        assert f'"{path}"' in packaging
