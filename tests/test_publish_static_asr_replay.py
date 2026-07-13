from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from jlens.static_explorer_catalog import (
    StaticAudioSample,
    StaticAudioSource,
    StaticAudioSourceOverride,
    StaticExplorerCatalog,
)
from scripts import publish_static_asr_replay as publisher
from scripts import validate_static_explorer_site as site_validator


def _candidate(token_id: int, text: str, *, score: float | None = None) -> dict:
    value = {
        "id": token_id,
        "text": text,
        "rank": 1,
        "rank_denominator": 51_864,
        "rank_space": "full_model_vocabulary",
        "rank_tie_policy": "1_plus_count_strictly_greater",
        "score_kind": "raw_readout_logit"
        if score is not None
        else "raw_head_probability",
        "vocabulary_filter": {
            "decoded_character_length": max(1, len(text.strip())),
            "display_lexical_eligible": True,
        },
    }
    if score is None:
        value.update({"probability": 0.9, "log_probability": -0.105})
    else:
        value["score"] = score
    return value


def _phone_metadata() -> dict:
    return {
        "available": True,
        "display_unit": "pooled_encoder_window",
        "effective_display_hop_seconds": 0.08,
        "effective_display_window_seconds": 0.1,
        "interpretation": "prototype cosine similarity, not probability",
        "method": "nearest_frozen_top_k_j_signature_phone_prototype",
        "phone_inventory": list(site_validator.PUBLIC_PHONE_INVENTORY),
        "phone_inventory_size": len(site_validator.PUBLIC_PHONE_INVENTORY),
        "score_kind": "phone_prototype_cosine_similarity",
        "signature_top_k": 100,
        "silence_or_unknown_class_available": False,
        "training_unit": "aligned_native_20_ms_phone_midpoint_state",
        "prototype_source": {
            "split": "train",
            "non_silence_rows": 3400,
            "development_or_test_opened": False,
            "lens_examples": 20,
        },
    }


def _raw_analysis(pieces: list[tuple[int, str]]) -> dict:
    tokens = []
    decoder_cells = []
    for index, (token_id, text) in enumerate(pieces):
        start = index * 0.1
        end = (index + 1) * 0.1
        head = _candidate(token_id, text)
        head.update(
            {
                "start_seconds": start,
                "end_seconds": end,
                "is_special": False,
                "top_tokens": [dict(head)],
            }
        )
        tokens.append(head)
        realized = _candidate(token_id, text, score=0.8)
        decoder_cells.append(
            {
                "position_index": index,
                "top_tokens": [dict(realized)],
                "top_tokens_by_length": {"10": [dict(realized)]},
                "realized_rank_by_max_length": {"10": 1},
                "realized_token": realized,
            }
        )

    first_id, first_text = pieces[0]
    encoder_realized = _candidate(first_id, first_text, score=0.7)
    phone_signatures = [
        {"phone": phone, "similarity": similarity, "rank": rank}
        for rank, (phone, similarity) in enumerate(
            (("AA", 0.8), ("B", 0.6), ("D", 0.4), ("EH", 0.3), ("F", 0.2)),
            start=1,
        )
    ]
    encoder_cell = {
        "position_index": 0,
        "time_window": {"start_seconds": 0.0, "end_seconds": 0.1},
        "top_tokens": [dict(encoder_realized)],
        "top_tokens_by_length": {"10": [dict(encoder_realized)]},
        "realized_rank_by_max_length": {"10": 1},
        "phone_signatures": phone_signatures,
        "realized_token_position": 0,
        "realized_token_alignment": {
            "match": "overlapping",
            "window_midpoint_seconds": 0.05,
            "token_start_seconds": 0.0,
            "token_end_seconds": 0.1,
            "overlap_seconds": 0.1,
            "overlap_fraction_of_window": 1.0,
        },
        "realized_token": encoder_realized,
    }
    return {
        "metadata": {
            "model_id": "openai/whisper-tiny.en",
            "display_vocabulary": {
                "maximum_decoded_character_length_counts": {"10": 100}
            },
            "phone_signature": _phone_metadata(),
        },
        "audio": {
            "duration_seconds": 0.7,
            "model_input_format": "mono 16 kHz",
            "waveform": [0.0, 0.1, -0.1],
        },
        "transcription": {
            "semantic_role": "generated_output",
            "text": "".join(text for _, text in pieces),
            "timing_quality": "model_derived",
            "timing_source": "whisper_cross_attention_dtw",
            "tokens": tokens,
        },
        "encoder": {
            "layers": [0, 1, 2, 3],
            "pooling": {
                "requested_window_seconds": 0.1,
                "requested_overlap_seconds": 0.02,
                "effective_window_seconds": 0.1,
                "effective_overlap_seconds": 0.02,
                "effective_hop_seconds": 0.08,
                "adaptive_for_max_bins": False,
                "max_time_bins": 100,
            },
            "realized_token_alignment": {
                "method": "maximum_token_interval_overlap",
                "tie_break": "closest_interval_midpoint_then_lower_token_position",
            },
            "cells": [[copy.deepcopy(encoder_cell)] for _ in range(4)],
        },
        "decoder": {"layers": [0], "cells": [decoder_cells]},
    }


def _capture(audio_hash: str) -> dict:
    generated = {
        "baseline": ("Lily!", [20037, 0], False),
        "yanny": ("Yanny!", [575, 7737, 0], True),
        "laurel": ("Laurel", [43442], True),
    }
    pieces = {
        "baseline": [(20037, " Lily"), (0, "!")],
        "yanny": [(575, " Y"), (7737, "anny"), (0, "!")],
        "laurel": [(43442, " Laurel")],
    }
    conditions = []
    schedules = {
        "baseline": [],
        "yanny": [
            {
                "phone": "Y",
                "start_seconds": 0.08,
                "end_seconds": 0.18,
                "start_position": 4,
                "end_position": 9,
            },
            {
                "phone": "AE",
                "start_seconds": 0.18,
                "end_seconds": 0.36,
                "start_position": 9,
                "end_position": 18,
            },
            {
                "phone": "N",
                "start_seconds": 0.36,
                "end_seconds": 0.48,
                "start_position": 18,
                "end_position": 24,
            },
            {
                "phone": "IY",
                "start_seconds": 0.48,
                "end_seconds": 0.68,
                "start_position": 24,
                "end_position": 34,
            },
        ],
        "laurel": [
            {
                "phone": "L",
                "start_seconds": 0.08,
                "end_seconds": 0.18,
                "start_position": 4,
                "end_position": 9,
            },
            {
                "phone": "AO",
                "start_seconds": 0.18,
                "end_seconds": 0.36,
                "start_position": 9,
                "end_position": 18,
            },
            {
                "phone": "R",
                "start_seconds": 0.36,
                "end_seconds": 0.44,
                "start_position": 18,
                "end_position": 22,
            },
            {
                "phone": "AH",
                "start_seconds": 0.44,
                "end_seconds": 0.56,
                "start_position": 22,
                "end_position": 28,
            },
            {
                "phone": "L",
                "start_seconds": 0.56,
                "end_seconds": 0.68,
                "start_position": 28,
                "end_position": 34,
            },
        ],
    }
    for condition_id in publisher.CONDITION_IDS:
        expected = publisher.EXPECTED_CONDITIONS[condition_id]
        text, token_ids, target_match = generated[condition_id]
        conditions.append(
            {
                "id": condition_id,
                "label": condition_id.title(),
                "recorded": True,
                "interpolated": False,
                "generated": {
                    "text": text,
                    "token_ids": token_ids,
                    "target_match": target_match,
                },
                "budget_fraction": expected["budget_fraction"],
                "coefficient_scale": expected["coefficient_scale"],
                "evidence": {
                    "tier": expected["evidence_tier"],
                    "badge": f"{condition_id} evidence",
                    "tone": "neutral",
                    "summary": "Recorded fixture result.",
                },
                "method": {
                    "kind": "phone_pullback" if condition_id != "baseline" else "none",
                    "label": "Recorded method",
                    "description": "A recorded forward pass.",
                    "coefficient_policy": "Frozen coefficients only.",
                },
                "layers": [] if condition_id == "baseline" else [0, 1, 2, 3],
                "schedule": schedules[condition_id],
                "recording": {
                    "coefficients": [1.0],
                    "pullback_audit": [{"private": "must be stripped"}],
                },
                "analysis": _raw_analysis(pieces[condition_id]),
            }
        )
    return {
        "schema_id": publisher.CAPTURE_SCHEMA_ID,
        "schema_version": 1,
        "default": "baseline",
        "source": {"audio_sha256": audio_hash},
        "provenance": {
            "recorded_only": True,
            "interpolated": False,
            "model_id": "openai/whisper-tiny.en",
            "model_revision": "revision",
            "model_fingerprint": "fingerprint",
            "artifact_sha256": {
                "encoder_lens": "1" * 64,
                "decoder_lens": "2" * 64,
                "display_phone_prototypes": "3" * 64,
                **publisher.PINNED_PRIVATE_ARTIFACT_SHA256,
            },
        },
        "conditions": conditions,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _entry(site_root: Path, slug: str) -> dict:
    report = site_root / "explorer" / "data" / "asr" / f"{slug}.json"
    sidecar = site_root / "explorer" / "data" / "asr" / f"{slug}.filters.json"
    _write_json(report, {"fixture": slug})
    _write_json(sidecar, {"fixture": slug})
    return {
        "id": f"asr-{slug}",
        "title": slug,
        "summary": slug,
        "reference_transcript": slug,
        "utterance_id": slug,
        "audio_url": f"/audio-jacobian-lens/audio/{slug}.flac",
        "report_url": f"/audio-jacobian-lens/explorer/data/asr/{slug}.json",
        "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "bytes": report.stat().st_size,
        "character_length_filter_cache": {
            "url": f"/audio-jacobian-lens/explorer/data/asr/{slug}.filters.json",
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "bytes": sidecar.stat().st_size,
        },
    }


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("generated", "secret"),
        ("evidence", "secret"),
        ("method", "secret"),
        ("schedule", "secret"),
    ),
)
def test_capture_rejects_nested_public_field_injection(
    container: str, field: str
) -> None:
    capture = _capture("a" * 64)
    condition = capture["conditions"][1]
    if container == "schedule":
        condition["schedule"][0][field] = "private material"
    else:
        condition[container][field] = "private material"

    with pytest.raises(ValueError, match="unexpected or missing fields"):
        publisher._validate_capture(capture)


def test_capture_rejects_changed_layer_and_schedule_shapes() -> None:
    shortened_layers = _capture("a" * 64)
    shortened_layers["conditions"][1]["layers"] = [0, 1, 2]
    with pytest.raises(ValueError, match="layers changed"):
        publisher._validate_capture(shortened_layers)

    shortened_schedule = _capture("a" * 64)
    shortened_schedule["conditions"][2]["schedule"].pop()
    with pytest.raises(ValueError, match="schedule must have 5 segments"):
        publisher._validate_capture(shortened_schedule)

    off_grid_schedule = _capture("a" * 64)
    off_grid_schedule["conditions"][1]["schedule"][0]["start_seconds"] = 0.081
    with pytest.raises(ValueError, match="native 20 ms positions"):
        publisher._validate_capture(off_grid_schedule)


def test_capture_provenance_binds_every_private_artifact_hash() -> None:
    capture = _capture("a" * 64)
    manifest = {
        "provenance": {
            "model": {
                "id": "openai/whisper-tiny.en",
                "revision": "revision",
                "model_fingerprint": "fingerprint",
            },
            "lens": {
                "encoder": {"sha256": "1" * 64},
                "decoder": {"sha256": "2" * 64},
                "phone_signature": {"sha256": "3" * 64},
            },
        }
    }
    publisher._validate_capture_provenance(
        capture, manifest, expected_audio_sha256="a" * 64
    )

    for field in publisher.PINNED_PRIVATE_ARTIFACT_SHA256:
        changed = copy.deepcopy(capture)
        changed["provenance"]["artifact_sha256"][field] = "0" * 64
        with pytest.raises(ValueError, match=field):
            publisher._validate_capture_provenance(
                changed, manifest, expected_audio_sha256="a" * 64
            )

    injected = copy.deepcopy(capture)
    injected["provenance"]["artifact_sha256"]["unreviewed_artifact"] = "9" * 64
    with pytest.raises(ValueError, match="artifact hash map changed"):
        publisher._validate_capture_provenance(
            injected, manifest, expected_audio_sha256="a" * 64
        )


def test_cross_directory_publication_rolls_back_every_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_root = (tmp_path / "site").resolve()
    original_files = {
        "audio/laurel-yanny.mp3": b"old audio",
        "explorer/data/asr/manifest.json": b"old asr manifest",
        "site-manifest.json": b"old site manifest",
        "explorer/data/asr/impossible.json": b"old impossible report",
        "explorer/data/asr/impossible.filters.json": b"old impossible filter",
    }
    for relative_path, content in original_files.items():
        path = site_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    stage = tmp_path / "stage"
    staged_files: dict[str, Path] = {}
    for relative_path in (
        "audio/laurel-yanny.mp3",
        "explorer/data/asr/laurel-yanny.json",
        "explorer/data/asr/laurel-yanny.filters.json",
        "explorer/data/asr/manifest.json",
        "site-manifest.json",
    ):
        path = stage / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"new {relative_path}".encode())
        staged_files[relative_path] = path
    deleted_paths = {
        "explorer/data/asr/impossible.json",
        "explorer/data/asr/impossible.filters.json",
    }

    def fail_after_promotion(**_kwargs: object) -> None:
        raise ValueError("post-promotion validation failed")

    monkeypatch.setattr(
        publisher, "_validate_committed_publication", fail_after_promotion
    )
    with pytest.raises(ValueError, match="post-promotion validation failed"):
        publisher._promote_publication_transaction(
            site_root=site_root,
            staged_files=staged_files,
            deleted_paths=deleted_paths,
        )

    for relative_path, content in original_files.items():
        assert (site_root / relative_path).read_bytes() == content
    assert not (site_root / "explorer/data/asr/laurel-yanny.json").exists()
    assert not (site_root / "explorer/data/asr/laurel-yanny.filters.json").exists()


def test_publisher_reduces_three_real_runs_and_replaces_only_asr_impossible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = b"rights-cleared replay fixture"
    audio_hash = hashlib.sha256(audio).hexdigest()
    source = StaticAudioSource(
        dataset_id="fixture/default",
        dataset_revision="a" * 40,
        parquet_path="fixture.parquet",
        upstream_collection="Fixture speech",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://www.openslr.org/12",
        attribution="Fixture default attribution.",
    )
    normal = StaticAudioSample(
        slug="question",
        title="Question",
        description="Question",
        utterance_id="1-2-3",
        filename="question.flac",
        reference_transcript="Question?",
        duration_seconds=0.2,
        sha256="4" * 64,
        lfm_fit_relationship="in_sample_integration",
    )
    impossible = StaticAudioSample(
        slug="impossible",
        title="Impossible",
        description="Impossible",
        utterance_id="1-2-4",
        filename="impossible.flac",
        reference_transcript="Impossible.",
        duration_seconds=0.2,
        sha256="5" * 64,
        lfm_fit_relationship="held_out_from_one_clip_fit",
    )
    laurel = StaticAudioSample(
        slug="laurel-yanny",
        title="Laurel or Yanny?",
        description="Recorded replay",
        utterance_id="bosker-audio-s7",
        filename="laurel-yanny.mp3",
        reference_transcript="Ambiguous source clip.",
        duration_seconds=0.3,
        sha256=audio_hash,
        lfm_fit_relationship="held_out_from_one_clip_fit",
        source_override=StaticAudioSourceOverride(
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            source_url="https://hrbosker.github.io/demos/laurel-yanny/",
            attribution=site_validator.ASR_REPLAY_ATTRIBUTION,
            modification_notice=site_validator.ASR_REPLAY_MODIFICATION_NOTICE,
        ),
    )
    catalog = StaticExplorerCatalog(
        reports_per_family=2,
        curated_findings_policy="fixture",
        audio_source=source,
        audio_samples=(normal, impossible),
        tts_examples=(),
        asr_audio_samples=(normal, laurel),
    )
    monkeypatch.setattr(
        publisher, "load_static_explorer_catalog", lambda _path: catalog
    )

    site_root = tmp_path / "site"
    asr_dir = site_root / "explorer" / "data" / "asr"
    old_entries = [_entry(site_root, "question"), _entry(site_root, "impossible")]
    manifest = {
        "schema_id": "audio-jacobian-lens.cached-explorer-manifest",
        "schema_version": 1,
        "family": "asr",
        "mode": "static_cached_explorer",
        "report_count": 2,
        "catalog": {},
        "provenance": {
            "model": {
                "id": "openai/whisper-tiny.en",
                "revision": "revision",
                "model_fingerprint": "fingerprint",
            },
            "lens": {
                "encoder": {"sha256": "1" * 64},
                "decoder": {"sha256": "2" * 64},
                "phone_signature": {"sha256": "3" * 64},
            },
            "rights": {
                "alignment_source_url": "https://zenodo.org/records/2619474",
                "alignment_license": "CC BY 4.0",
            },
            "rights_policy": {},
        },
        "reports": old_entries,
    }
    _write_json(asr_dir / "manifest.json", manifest)
    _write_json(
        site_root / "site-manifest.json",
        {
            "schema_version": 2,
            "sha256": {
                "explorer/data/asr/manifest.json": hashlib.sha256(
                    (asr_dir / "manifest.json").read_bytes()
                ).hexdigest()
            },
        },
    )
    speech_impossible = site_root / "explorer" / "data" / "speech" / "impossible.json"
    _write_json(speech_impossible, {"must": "remain"})
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "laurel-yanny.mp3").write_bytes(audio)
    capture_path = tmp_path / "capture.json"
    _write_json(capture_path, _capture(audio_hash))
    catalog_path = tmp_path / "catalog.json"
    _write_json(catalog_path, {"fixture": True})

    rebuilt = publisher.publish(
        capture_path=capture_path,
        site_root=site_root,
        catalog_path=catalog_path,
        samples_dir=samples_dir,
    )

    assert [entry["id"] for entry in rebuilt["reports"]] == [
        "asr-question",
        "asr-laurel-yanny",
    ]
    assert (
        rebuilt["catalog"]["sha256"]
        == hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    )
    assert not (asr_dir / "impossible.json").exists()
    assert not (asr_dir / "impossible.filters.json").exists()
    assert speech_impossible.is_file()
    report = json.loads((asr_dir / "laurel-yanny.json").read_text())
    replay = report["recorded_intervention_replay"]
    assert [condition["id"] for condition in replay["conditions"]] == [
        "baseline",
        "yanny",
        "laurel",
    ]
    assert "analysis" not in replay["conditions"][0]
    serialized = json.dumps(replay).lower()
    assert "recording" not in serialized
    assert "pullback_audit" not in serialized
    assert '"audio"' not in serialized
    assert (site_root / "audio" / "laurel-yanny.mp3").read_bytes() == audio
    site_validator._validate_recorded_intervention_replay(report)


def test_validator_rejects_residual_data_and_geometry_changes() -> None:
    capture = _capture("a" * 64)
    conditions = publisher._validate_capture(capture)
    baseline = publisher.exporter._reduce_payload(conditions["baseline"]["analysis"])
    report = {
        "family": "asr",
        "example_id": "asr-laurel-yanny",
        "source": {
            "audio_url": "/audio-jacobian-lens/audio/laurel-yanny.mp3",
            "sha256": "a" * 64,
            "source_url": "https://hrbosker.github.io/demos/laurel-yanny/",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": site_validator.ASR_REPLAY_ATTRIBUTION,
            "modification_notice": site_validator.ASR_REPLAY_MODIFICATION_NOTICE,
            "rights_status": site_validator.ASR_REPLAY_RIGHTS_STATUS,
        },
        "payload": baseline,
        "recorded_intervention_replay": {
            "schema_id": publisher.REPLAY_SCHEMA_ID,
            "schema_version": 1,
            "mode": publisher.REPLAY_MODE,
            "default_condition": "baseline",
            "conditions": [
                publisher._public_condition(
                    conditions[condition_id],
                    include_analysis=condition_id != "baseline",
                )
                for condition_id in publisher.CONDITION_IDS
            ],
        },
    }
    site_validator._validate_recorded_intervention_replay(report)

    unsafe = copy.deepcopy(report)
    unsafe["recorded_intervention_replay"]["conditions"][1]["recording"] = {
        "residuals": [1.0]
    }
    with pytest.raises(ValueError, match="unexpected or missing fields|forbidden"):
        site_validator._validate_recorded_intervention_replay(unsafe)

    shifted = copy.deepcopy(report)
    shifted["recorded_intervention_replay"]["conditions"][2]["analysis"]["encoder"][
        "pooling"
    ]["effective_hop_seconds"] = 0.09
    with pytest.raises(ValueError, match="geometry changed"):
        site_validator._validate_recorded_intervention_replay(shifted)
