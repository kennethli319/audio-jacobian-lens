from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from jlens.static_explorer_catalog import load_static_explorer_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "static_explorer_catalog_v2.json"


def test_real_catalog_has_ten_ordered_examples_per_family() -> None:
    catalog = load_static_explorer_catalog(CATALOG_PATH)

    assert catalog.reports_per_family == 10
    assert [sample.slug for sample in catalog.audio_samples_for_family("speech")] == [
        "question",
        "universe",
        "buzzer",
        "raps",
        "one-minute",
        "ten-seconds",
        "oh-no",
        "metal-forest",
        "inexhaustible",
        "impossible",
    ]
    assert [sample.slug for sample in catalog.audio_samples] == [
        "question",
        "universe",
        "buzzer",
        "raps",
        "one-minute",
        "ten-seconds",
        "oh-no",
        "metal-forest",
        "inexhaustible",
        "impossible",
    ]
    assert [sample.slug for sample in catalog.audio_samples_for_family("asr")] == [
        "question",
        "universe",
        "buzzer",
        "raps",
        "one-minute",
        "ten-seconds",
        "oh-no",
        "metal-forest",
        "inexhaustible",
        "laurel-yanny",
    ]
    laurel_yanny = catalog.audio_samples_for_family("asr")[-1]
    assert laurel_yanny.filename == "laurel-yanny.mp3"
    assert laurel_yanny.utterance_id == "bosker-audio-s7"
    assert laurel_yanny.source_override is not None
    assert laurel_yanny.source_override.source_url == (
        "https://hrbosker.github.io/demos/laurel-yanny/"
    )
    assert len(catalog.tts_examples) == 10
    assert catalog.tts_examples[0].example_id == "tts-bridge-s9"
    assert [
        sample.slug
        for sample in catalog.audio_samples
        if sample.lfm_fit_relationship == "in_sample_integration"
    ] == ["question"]
    assert all(
        not sample.featured_views
        for family in ("asr", "speech")
        for sample in catalog.audio_samples_for_family(family)
    )


def test_catalog_audio_assets_and_local_sample_manifest_are_aligned() -> None:
    catalog = load_static_explorer_catalog(CATALOG_PATH)
    local_manifest = json.loads(
        (ROOT / "samples" / "samples.json").read_text(encoding="utf-8")
    )["samples"]
    by_filename = {entry["file"]: entry for entry in local_manifest}

    assert len(by_filename) == catalog.reports_per_family
    for sample in catalog.audio_samples_for_family("asr"):
        audio_path = ROOT / "samples" / sample.filename
        assert audio_path.is_file()
        assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == sample.sha256
        entry = by_filename[sample.filename]
        assert entry["title"] == sample.title
        assert entry["transcript"] == sample.reference_transcript
        assert entry["duration_seconds"] == sample.duration_seconds

    for sample in catalog.audio_samples_for_family("speech"):
        audio_path = ROOT / "samples" / sample.filename
        assert audio_path.is_file()
        assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == sample.sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__("reports_per_family", 9),
            "audio_samples count",
        ),
        (
            lambda value: value["audio_source"].__setitem__(
                "parquet_path", "../untrusted.parquet"
            ),
            "safe relative Parquet path",
        ),
        (
            lambda value: value["audio_samples"][1].__setitem__(
                "slug", value["audio_samples"][0]["slug"]
            ),
            "duplicate audio sample slug",
        ),
        (
            lambda value: value["asr_audio_samples"][-1].__setitem__(
                "filename", "different-name.mp3"
            ),
            "must match the sample slug",
        ),
    ],
)
def test_catalog_rejects_invalid_global_contracts(
    tmp_path: Path, mutation, message: str
) -> None:
    value = copy.deepcopy(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
    mutation(value)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_static_explorer_catalog(path)
