from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from jlens.static_explorer_catalog import (
    StaticAudioSample,
    StaticAudioSource,
    StaticExplorerCatalog,
)
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
        "rank_tie_policy": "1_plus_count_strictly_greater",
    }


def _realized_candidate(
    token_id: int, text: str, score: float, rank: int
) -> dict[str, object]:
    return {
        "id": token_id,
        "text": text,
        "score": score,
        "score_kind": "raw_readout_logit",
        "rank": rank,
        "rank_denominator": 61_690,
        "rank_space": "lexical_display_vocabulary",
        "rank_tie_policy": "1_plus_count_strictly_greater",
        "full_vocabulary_rank": rank + 3,
        "full_vocabulary_denominator": 65_536,
        "vocabulary_filter": {
            "decoded_character_length": len(text.strip()),
            "display_lexical_filter_applied": True,
            "display_lexical_eligible": True,
        },
        "server_only_debug": "must not be published",
    }


def _cell(
    position: int,
    *,
    with_filter: bool,
    realized_id: int,
    realized_text: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "position_index": position,
        "time_window": {
            "start_seconds": position * 0.2,
            "end_seconds": (position + 1) * 0.2,
        },
        "top_tokens": [_candidate(100 + position, 1.0 - position)],
        "realized_token": _realized_candidate(
            realized_id, realized_text, 0.25 - position * 0.05, 15 + position
        ),
    }
    if with_filter:
        result["top_tokens_by_length"] = {
            "1": [_candidate(10 + position, 0.9 - position)],
            "2": [_candidate(20 + position, 0.8 - position)],
        }
        target_length = len(realized_text.strip())
        result["realized_rank_by_max_length"] = {
            str(limit): None if target_length > limit else 4 + position + limit
            for limit in range(1, 4)
        }
    return result


def _encoder_cell(
    position: int, *, realized_id: int, realized_text: str
) -> dict[str, object]:
    result = _cell(
        position,
        with_filter=True,
        realized_id=realized_id,
        realized_text=realized_text,
    )
    result["realized_token_position"] = position
    result["realized_token_alignment"] = {
        "match": "overlapping",
        "window_midpoint_seconds": position * 0.2 + 0.1,
        "token_start_seconds": position * 0.2,
        "token_end_seconds": (position + 1) * 0.2,
        "overlap_seconds": 0.2,
        "overlap_fraction_of_window": 1.0,
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
            "display_vocabulary": {
                "maximum_decoded_character_length_counts": {
                    "1": 10,
                    "2": 20,
                    "3": 30,
                }
            },
        },
        "audio": {
            "duration_seconds": 0.4,
            "model_input_format": "mono_16khz",
            "waveform": [float(index) for index in range(2048)],
        },
        "transcription": {
            "text": " a the",
            "tokens": [
                {
                    **_realized_candidate(42, " a", 0.5, 1),
                    "probability": 0.25,
                    "start_seconds": 0.0,
                    "end_seconds": 0.2,
                    "top_tokens": [_candidate(42, 0.5)],
                },
                {
                    **_realized_candidate(43, " the", 0.4, 2),
                    "probability": 0.2,
                    "start_seconds": 0.2,
                    "end_seconds": 0.4,
                    "top_tokens": [_candidate(43, 0.4)],
                },
            ],
        },
        "encoder": {
            "layers": [0, 1],
            "realized_token_alignment": {
                "method": "maximum_token_interval_overlap",
                "tie_break": "closest_interval_midpoint_then_lower_token_position",
            },
            "cells": [
                [
                    _encoder_cell(0, realized_id=42, realized_text=" a"),
                    _encoder_cell(1, realized_id=43, realized_text=" the"),
                ],
                [
                    _encoder_cell(0, realized_id=42, realized_text=" a"),
                    _encoder_cell(1, realized_id=43, realized_text=" the"),
                ],
            ],
        },
        "decoder": {
            "layers": [0, 1, 2],
            "cells": [
                [
                    _cell(0, with_filter=True, realized_id=42, realized_text=" a"),
                    _cell(1, with_filter=True, realized_id=43, realized_text=" the"),
                ],
                [
                    _cell(0, with_filter=True, realized_id=42, realized_text=" a"),
                    _cell(1, with_filter=True, realized_id=43, realized_text=" the"),
                ],
                [
                    _cell(0, with_filter=False, realized_id=42, realized_text=" a"),
                    _cell(1, with_filter=False, realized_id=43, realized_text=" the"),
                ],
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


def test_reduced_payload_preserves_allowlisted_realized_token_rank() -> None:
    raw = _raw_payload()
    raw["decoder"]["cells"][0][0]["realized_token"] = _realized_candidate(
        42, " the", 0.125, 15
    )

    reduced = exporter._reduce_payload(raw)
    realized = reduced["decoder"]["cells"][0][0]["realized_token"]

    assert realized["id"] == 42
    assert realized["text"] == " the"
    assert realized["rank"] == 15
    assert realized["rank_denominator"] == 61_690
    assert realized["full_vocabulary_rank"] == 18
    assert realized["rank_tie_policy"] == "1_plus_count_strictly_greater"
    assert "server_only_debug" not in realized


def _speech_report() -> dict[str, object]:
    token = {
        **_realized_candidate(42, " the", 0.25, 2),
        "probability": 0.2,
        "log_probability": -1.609,
        "top_tokens": [_candidate(99, 0.8)],
    }
    cell = {
        "position_index": 0,
        "selected_score": 0.8,
        "top_tokens": [_candidate(99, 0.8)],
        "realized_token": _realized_candidate(42, " the", 0.25, 15),
    }
    return {
        "family": "speech",
        "payload": {
            "transcription": {"tokens": [token]},
            "encoder": {"layers": [], "cells": []},
            "decoder": {"layers": [0, 1], "cells": [[cell], [dict(cell)]]},
        },
    }


def test_speech_matrix_requires_exact_realized_rank_for_layers_and_head() -> None:
    report = _speech_report()

    exporter._validate_matrix(report)

    del report["payload"]["decoder"]["cells"][1][0]["realized_token"]
    with pytest.raises(ValueError, match="no exact realized-token provenance"):
        exporter._validate_matrix(report)


def test_speech_matrix_rejects_realized_token_id_mismatch() -> None:
    report = _speech_report()
    report["payload"]["decoder"]["cells"][0][0]["realized_token"]["id"] = 7

    with pytest.raises(ValueError, match="does not match the output token"):
        exporter._validate_matrix(report)


def test_filter_cache_is_separate_and_aligned_with_base_report() -> None:
    raw = _raw_payload()
    report = {
        "family": "asr",
        "example_id": "asr-example",
        "payload": exporter._reduce_payload(raw),
    }
    cache = exporter._filter_cache(raw, example_id="asr-example")

    exporter._validate_matrix(report)
    exporter._validate_filter_cache(cache, report)

    assert cache["streams"]["encoder"]["layers"] == [0, 1]
    assert cache["streams"]["decoder"]["layers"] == [0, 1]
    assert (
        cache["streams"]["decoder"]["cells"][0][0]["top_tokens_by_length"]["2"][0]["id"]
        == 20
    )
    assert cache["streams"]["decoder"]["cells"][0][0][
        "realized_rank_by_max_length"
    ] == {"1": 5, "2": 6, "3": 7}
    assert cache["streams"]["decoder"]["cells"][0][1][
        "realized_rank_by_max_length"
    ] == {"1": None, "2": None, "3": 8}


def test_filter_cache_validation_rejects_shifted_coordinates() -> None:
    raw = _raw_payload()
    report = {
        "family": "asr",
        "example_id": "asr-example",
        "payload": exporter._reduce_payload(raw),
    }
    cache = exporter._filter_cache(raw, example_id="asr-example")
    cache["streams"]["encoder"]["cells"][0][1]["position_index"] = 99

    with pytest.raises(ValueError, match="coordinate mismatch"):
        exporter._validate_filter_cache(cache, report)


def test_filter_cache_allows_nonlexical_realized_token_to_remain_excluded() -> None:
    raw = _raw_payload()
    cell = raw["decoder"]["cells"][0][0]
    cell["realized_token"]["rank_space"] = "full_model_vocabulary"
    cell["realized_token"]["vocabulary_filter"] = {
        "decoded_character_length": 1,
        "display_lexical_filter_applied": False,
        "display_lexical_eligible": False,
    }
    cell["realized_rank_by_max_length"] = {"1": None, "2": None, "3": None}
    report = {
        "family": "asr",
        "example_id": "asr-example",
        "payload": exporter._reduce_payload(raw),
    }
    cache = exporter._filter_cache(raw, example_id="asr-example")

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


def _mini_catalog(audio_values: list[bytes]) -> StaticExplorerCatalog:
    samples = tuple(
        StaticAudioSample(
            slug=f"sample-{index}",
            title=f"Sample {index}",
            description=f"Description {index}",
            utterance_id=f"1-2-{index:04d}",
            reference_transcript=f"Reference {index}.",
            duration_seconds=0.4,
            sha256=sha256(value).hexdigest(),
            lfm_fit_relationship=(
                "in_sample_integration" if index == 0 else "held_out_from_one_clip_fit"
            ),
        )
        for index, value in enumerate(audio_values)
    )
    return StaticExplorerCatalog(
        reports_per_family=len(samples),
        curated_findings_policy="Fixture",
        audio_source=StaticAudioSource(
            dataset_id="example/dataset",
            dataset_revision="1" * 40,
            parquet_path="clean/example.parquet",
            upstream_collection="Example collection",
            license="CC BY 4.0",
            license_url="https://example.test/license",
            source_url="https://example.test/source",
            attribution="Example attribution",
        ),
        audio_samples=samples,
        tts_examples=(),
    )


def test_selective_export_preserves_existing_reports_and_rebuilds_ordered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_values = [b"first-audio", b"second-audio"]
    catalog = _mini_catalog(audio_values)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    for sample, value in zip(catalog.audio_samples, audio_values, strict=True):
        (samples_dir / sample.filename).write_bytes(value)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(exporter, "_analyze", lambda *_args, **_kwargs: _raw_payload())

    first_manifest = exporter.export_family(
        family="asr",
        endpoint="http://unused.test",
        samples_dir=samples_dir,
        output_dir=output_dir,
        provenance={},
        catalog=catalog,
        selected_slugs={sample.slug for sample in catalog.audio_samples},
        resume=False,
        catalog_sha256="0" * 64,
    )
    first_report = output_dir / "sample-0.json"
    first_bytes = first_report.read_bytes()
    (output_dir / "sample-1.json").unlink()
    (output_dir / "sample-1.filters.json").unlink()

    second_manifest = exporter.export_family(
        family="asr",
        endpoint="http://unused.test",
        samples_dir=samples_dir,
        output_dir=output_dir,
        provenance={},
        catalog=catalog,
        selected_slugs={"sample-1"},
        resume=False,
        catalog_sha256="0" * 64,
    )

    assert first_manifest["report_count"] == 2
    assert first_report.read_bytes() == first_bytes
    assert [entry["id"] for entry in second_manifest["reports"]] == [
        "asr-sample-0",
        "asr-sample-1",
    ]
    assert second_manifest["report_count"] == 2
    assert second_manifest["reports"][1]["summary"] == "Description 1"
    assert second_manifest["reports"][1]["reference_transcript"] == "Reference 1."


def test_selective_export_does_not_replace_manifest_until_catalog_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_values = [b"first-audio", b"second-audio"]
    catalog = _mini_catalog(audio_values)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    for sample, value in zip(catalog.audio_samples, audio_values, strict=True):
        (samples_dir / sample.filename).write_bytes(value)
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text('{"previous":true}\n', encoding="utf-8")
    monkeypatch.setattr(exporter, "_analyze", lambda *_args, **_kwargs: _raw_payload())

    with pytest.raises(ValueError, match="manifest has an invalid identity"):
        exporter.export_family(
            family="asr",
            endpoint="http://unused.test",
            samples_dir=samples_dir,
            output_dir=output_dir,
            provenance={},
            catalog=catalog,
            selected_slugs={"sample-0"},
            resume=False,
            catalog_sha256="0" * 64,
        )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"previous": True}


def test_resume_rejects_reports_bound_to_different_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_values = [b"first-audio", b"second-audio"]
    catalog = _mini_catalog(audio_values)
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    for sample, value in zip(catalog.audio_samples, audio_values, strict=True):
        (samples_dir / sample.filename).write_bytes(value)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(exporter, "_analyze", lambda *_args, **_kwargs: _raw_payload())
    exporter.export_family(
        family="asr",
        endpoint="http://unused.test",
        samples_dir=samples_dir,
        output_dir=output_dir,
        provenance={"model": {"id": "example/model"}},
        catalog=catalog,
        selected_slugs={sample.slug for sample in catalog.audio_samples},
        resume=False,
        catalog_sha256="0" * 64,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    monkeypatch.setattr(
        exporter,
        "_analyze",
        lambda *_args, **_kwargs: pytest.fail("resume must not call analysis"),
    )

    with pytest.raises(ValueError, match="different model, lens, or generation"):
        exporter.export_family(
            family="asr",
            endpoint="http://unused.test",
            samples_dir=samples_dir,
            output_dir=output_dir,
            provenance={"model": {"id": "different/model"}},
            catalog=catalog,
            selected_slugs={sample.slug for sample in catalog.audio_samples},
            resume=True,
            catalog_sha256="0" * 64,
        )

    assert manifest_path.read_bytes() == manifest_before


def test_staged_promotion_rolls_back_reports_when_manifest_replace_fails(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "live"
    stage_dir = tmp_path / "stage"
    output_dir.mkdir()
    stage_dir.mkdir()
    (output_dir / "sample.json").write_text("old report", encoding="utf-8")
    (output_dir / "manifest.json").write_text("old manifest", encoding="utf-8")
    (stage_dir / "sample.json").write_text("new report", encoding="utf-8")
    (stage_dir / "manifest.json").write_text("new manifest", encoding="utf-8")
    replacements = 0

    def fail_on_manifest(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("injected manifest promotion failure")
        exporter.os.replace(source, destination)

    with pytest.raises(OSError, match="injected manifest"):
        exporter._promote_staged_family(
            stage_dir=stage_dir,
            output_dir=output_dir,
            changed_filenames={"sample.json"},
            replace_file=fail_on_manifest,
        )

    assert (output_dir / "sample.json").read_text(encoding="utf-8") == "old report"
    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "old manifest"


def _speech_generation_report(max_new_tokens: int) -> dict[str, object]:
    policy = {
        "system_prompt": "Respond briefly.",
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "audio_temperature": 0.0,
        "audio_top_k": 1,
    }
    return {
        "payload": {
            "metadata": {
                "serving_generation": dict(policy),
                "generation": dict(policy),
                "generation_diagnostics": {"max_new_tokens": max_new_tokens},
            }
        }
    }


def test_speech_manifest_cap_is_derived_from_one_report_policy() -> None:
    provenance = {
        "evaluation_generation": {
            "mode": "interleaved",
            "max_new_tokens": 512,
            "text_temperature": 0.0,
            "text_top_k": 1,
            "audio_temperature": 0.0,
            "audio_top_k": 1,
        }
    }

    exporter._derive_speech_generation_provenance(
        provenance,
        [_speech_generation_report(1024), _speech_generation_report(1024)],
    )

    assert provenance["evaluation_generation"]["max_new_tokens"] == 1024
    assert provenance["evaluation_generation"]["system_prompt"] == "Respond briefly."


def test_speech_manifest_rejects_mixed_serving_caps() -> None:
    provenance = {
        "evaluation_generation": {
            "mode": "interleaved",
            "max_new_tokens": 512,
            "text_temperature": 0.0,
            "text_top_k": 1,
            "audio_temperature": 0.0,
            "audio_top_k": 1,
        }
    }

    with pytest.raises(ValueError, match="do not share one serving-generation"):
        exporter._derive_speech_generation_provenance(
            provenance,
            [_speech_generation_report(512), _speech_generation_report(1024)],
        )
