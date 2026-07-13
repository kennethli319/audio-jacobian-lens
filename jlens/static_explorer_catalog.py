"""Validated metadata for the backend-free explorer report catalog.

The detailed static explorers use a separate catalog from the three-example
curated findings bundle.  Keeping this loader dependency-free lets both export
scripts and integrity tests share the same ordering, identifiers, rights
metadata, and expected family size.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CATALOG_SCHEMA_ID = "audio-jacobian-lens.static-explorer-catalog"
CATALOG_SCHEMA_VERSION = 2

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SOURCE_ITEM_ID = re.compile(r"^(?:[0-9]+-[0-9]+-[0-9]+|[a-z0-9]+(?:-[a-z0-9]+)*)$")
_AUDIO_SUFFIXES = {".flac", ".mp3"}
_LFM_FIT_RELATIONSHIPS = {
    "held_out_from_one_clip_fit",
    "in_sample_integration",
}


@dataclass(frozen=True)
class StaticAudioSource:
    """Pinned source and redistribution metadata for the default input set."""

    dataset_id: str
    dataset_revision: str
    parquet_path: str
    upstream_collection: str
    license: str
    license_url: str
    source_url: str
    attribution: str


@dataclass(frozen=True)
class StaticAudioSourceOverride:
    """Per-input rights metadata for a sample outside the default dataset."""

    license: str
    license_url: str
    source_url: str
    attribution: str
    modification_notice: str


@dataclass(frozen=True)
class StaticAudioSample:
    """One ordered explorer input with a pinned local filename and hash."""

    slug: str
    title: str
    description: str
    utterance_id: str
    filename: str
    reference_transcript: str
    duration_seconds: float
    sha256: str
    lfm_fit_relationship: str
    source_override: StaticAudioSourceOverride | None = None
    featured_views: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticTTSExample:
    """One ordered TTS prompt for the detailed cached explorer."""

    example_id: str
    title: str
    prompt: str
    teaching_role: str
    teaching_purpose: str
    curated_source_id: str | None


@dataclass(frozen=True)
class StaticExplorerCatalog:
    """Complete, ordered v2 catalog shared by all detailed explorers."""

    reports_per_family: int
    curated_findings_policy: str
    audio_source: StaticAudioSource
    audio_samples: tuple[StaticAudioSample, ...]
    tts_examples: tuple[StaticTTSExample, ...]
    asr_audio_samples: tuple[StaticAudioSample, ...] | None = None

    def audio_samples_for_family(self, family: str) -> tuple[StaticAudioSample, ...]:
        """Return the ordered input set for one public explorer family."""

        if family == "asr":
            return self.asr_audio_samples or self.audio_samples
        if family == "speech":
            return self.audio_samples
        raise ValueError(f"unsupported static explorer family: {family}")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _safe_relative_posix_path(value: Any, *, label: str) -> str:
    path_text = _text(value, label=label)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
        raise ValueError(f"{label} must be a safe relative Parquet path")
    return path_text


def _audio_source(value: Any) -> StaticAudioSource:
    source = _mapping(value, label="audio_source")
    revision = _text(
        source.get("dataset_revision"), label="audio_source.dataset_revision"
    )
    if not _HEX_40.fullmatch(revision):
        raise ValueError("audio_source.dataset_revision must be a 40-character hash")
    license_name = _text(source.get("license"), label="audio_source.license")
    if license_name != "CC BY 4.0":
        raise ValueError("audio_source.license must remain CC BY 4.0")
    return StaticAudioSource(
        dataset_id=_text(source.get("dataset_id"), label="audio_source.dataset_id"),
        dataset_revision=revision,
        parquet_path=_safe_relative_posix_path(
            source.get("parquet_path"), label="audio_source.parquet_path"
        ),
        upstream_collection=_text(
            source.get("upstream_collection"),
            label="audio_source.upstream_collection",
        ),
        license=license_name,
        license_url=_text(source.get("license_url"), label="audio_source.license_url"),
        source_url=_text(source.get("source_url"), label="audio_source.source_url"),
        attribution=_text(source.get("attribution"), label="audio_source.attribution"),
    )


def _audio_source_override(
    value: Any, *, label: str
) -> StaticAudioSourceOverride | None:
    if value is None:
        return None
    source = _mapping(value, label=label)
    return StaticAudioSourceOverride(
        license=_text(source.get("license"), label=f"{label}.license"),
        license_url=_text(source.get("license_url"), label=f"{label}.license_url"),
        source_url=_text(source.get("source_url"), label=f"{label}.source_url"),
        attribution=_text(source.get("attribution"), label=f"{label}.attribution"),
        modification_notice=_text(
            source.get("modification_notice"),
            label=f"{label}.modification_notice",
        ),
    )


def _audio_filename(value: Any, *, slug: str, label: str) -> str:
    filename = _text(value, label=label)
    path = PurePosixPath(filename)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != filename
        or path.stem != slug
        or path.suffix.lower() not in _AUDIO_SUFFIXES
    ):
        raise ValueError(
            f"{label} must match the sample slug and use a supported audio suffix"
        )
    return filename


def _audio_sample(value: Any, *, index: int, collection: str) -> StaticAudioSample:
    label = f"{collection}[{index}]"
    source = _mapping(value, label=label)
    slug = _text(source.get("slug"), label=f"{label}.slug")
    if not _SLUG.fullmatch(slug):
        raise ValueError(f"{label}.slug is not URL-safe")
    utterance_id = _text(source.get("utterance_id"), label=f"{label}.utterance_id")
    if not _SOURCE_ITEM_ID.fullmatch(utterance_id):
        raise ValueError(f"{label}.utterance_id is invalid")
    filename = _audio_filename(
        source.get("filename", f"{slug}.flac"),
        slug=slug,
        label=f"{label}.filename",
    )
    sha256 = _text(source.get("sha256"), label=f"{label}.sha256")
    if not _HEX_64.fullmatch(sha256):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 hash")
    duration = source.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError(f"{label}.duration_seconds must be positive and finite")
    relationship = _text(
        source.get("lfm_fit_relationship"),
        label=f"{label}.lfm_fit_relationship",
    )
    if relationship not in _LFM_FIT_RELATIONSHIPS:
        raise ValueError(f"{label}.lfm_fit_relationship is unsupported")
    featured_value = source.get("featured_views", [])
    featured = tuple(
        _text(value, label=f"{label}.featured_views")
        for value in _list(featured_value, label=f"{label}.featured_views")
    )
    if len(set(featured)) != len(featured):
        raise ValueError(f"{label}.featured_views contains duplicates")
    return StaticAudioSample(
        slug=slug,
        title=_text(source.get("title"), label=f"{label}.title"),
        description=_text(source.get("description"), label=f"{label}.description"),
        utterance_id=utterance_id,
        filename=filename,
        reference_transcript=_text(
            source.get("reference_transcript"),
            label=f"{label}.reference_transcript",
        ),
        duration_seconds=float(duration),
        sha256=sha256,
        lfm_fit_relationship=relationship,
        source_override=_audio_source_override(
            source.get("source_override"), label=f"{label}.source_override"
        ),
        featured_views=featured,
    )


def _tts_example(value: Any, *, index: int) -> StaticTTSExample:
    label = f"tts_examples[{index}]"
    source = _mapping(value, label=label)
    example_id = _text(source.get("id"), label=f"{label}.id")
    if not _SLUG.fullmatch(example_id):
        raise ValueError(f"{label}.id is not URL-safe")
    curated_source = source.get("curated_source_id")
    if curated_source is not None:
        curated_source = _text(curated_source, label=f"{label}.curated_source_id")
    return StaticTTSExample(
        example_id=example_id,
        title=_text(source.get("title"), label=f"{label}.title"),
        prompt=_text(source.get("prompt"), label=f"{label}.prompt"),
        teaching_role=_text(
            source.get("teaching_role"), label=f"{label}.teaching_role"
        ),
        teaching_purpose=_text(
            source.get("teaching_purpose"), label=f"{label}.teaching_purpose"
        ),
        curated_source_id=curated_source,
    )


def _require_unique(values: list[str], *, label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {label}: {', '.join(duplicates)}")


def load_static_explorer_catalog(path: str | Path) -> StaticExplorerCatalog:
    """Load and fully validate an ordered static-explorer v2 catalog."""

    catalog_path = Path(path)
    try:
        source = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid catalog JSON: {catalog_path}") from error
    root = _mapping(source, label="catalog")
    if root.get("schema_id") != CATALOG_SCHEMA_ID:
        raise ValueError("unsupported static explorer catalog schema_id")
    if root.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported static explorer catalog schema_version")
    report_count = root.get("reports_per_family")
    if isinstance(report_count, bool) or not isinstance(report_count, int):
        raise ValueError("reports_per_family must be an integer")
    if report_count <= 0:
        raise ValueError("reports_per_family must be positive")

    raw_audio_samples = _list(root.get("audio_samples"), label="audio_samples")
    audio_samples = tuple(
        _audio_sample(value, index=index, collection="audio_samples")
        for index, value in enumerate(raw_audio_samples)
    )
    raw_asr_audio_samples = _list(
        root.get("asr_audio_samples", raw_audio_samples),
        label="asr_audio_samples",
    )
    asr_audio_samples = tuple(
        _audio_sample(value, index=index, collection="asr_audio_samples")
        for index, value in enumerate(raw_asr_audio_samples)
    )
    tts_examples = tuple(
        _tts_example(value, index=index)
        for index, value in enumerate(
            _list(root.get("tts_examples"), label="tts_examples")
        )
    )
    if len(audio_samples) != report_count:
        raise ValueError("audio_samples count does not match reports_per_family")
    if len(asr_audio_samples) != report_count:
        raise ValueError("asr_audio_samples count does not match reports_per_family")
    if len(tts_examples) != report_count:
        raise ValueError("tts_examples count does not match reports_per_family")
    _require_unique([item.slug for item in audio_samples], label="audio sample slug")
    _require_unique(
        [item.slug for item in asr_audio_samples], label="ASR audio sample slug"
    )
    _require_unique(
        [item.utterance_id for item in audio_samples],
        label="audio sample utterance_id",
    )
    _require_unique(
        [item.sha256 for item in audio_samples], label="audio sample SHA-256"
    )
    _require_unique(
        [item.utterance_id for item in asr_audio_samples],
        label="ASR audio sample utterance_id",
    )
    _require_unique(
        [item.sha256 for item in asr_audio_samples], label="ASR audio sample SHA-256"
    )
    _require_unique([item.example_id for item in tts_examples], label="TTS example id")
    for family, samples in (("speech", audio_samples), ("ASR", asr_audio_samples)):
        in_sample = [
            item
            for item in samples
            if item.lfm_fit_relationship == "in_sample_integration"
        ]
        if len(in_sample) != 1:
            raise ValueError(
                f"the {family} catalog must identify exactly one in-sample input"
            )

    return StaticExplorerCatalog(
        reports_per_family=report_count,
        curated_findings_policy=_text(
            root.get("curated_findings_policy"),
            label="curated_findings_policy",
        ),
        audio_source=_audio_source(root.get("audio_source")),
        audio_samples=audio_samples,
        tts_examples=tts_examples,
        asr_audio_samples=asr_audio_samples,
    )
