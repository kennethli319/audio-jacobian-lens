#!/usr/bin/env python3
"""Export deterministic, rights-safe Chatterbox TTS explorer reports.

The exporter talks to a ready local Chatterbox server sequentially.  Ephemeral
``analysis_id`` values are used only to request per-position traces and are
never copied into a report.  Generated audio, waveform arrays, and embedded
audio URIs are excluded by construction and checked again before each write.

The ten prompts and teaching metadata come from
``data/static_explorer_catalog_v2.json``. Immutable model/lens provenance and
the recorded bridge intervention remain sourced from the separate curated
``data/static_public_reports_v1.json`` snapshot. The bridge intervention is
metadata from the recorded, reviewed run; this script does not re-run either
the residual or direct-forced branch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jlens.static_explorer_catalog import load_static_explorer_catalog
except ModuleNotFoundError as error:
    if error.name != "jlens":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jlens.static_explorer_catalog import load_static_explorer_catalog

SCHEMA_ID = "audio-jacobian-lens.cached-explorer-report"
MANIFEST_SCHEMA_ID = "audio-jacobian-lens.cached-explorer-manifest"
SCHEMA_VERSION = 1

EXPECTED_GENERATION = {
    "max_speech_tokens": 96,
    "repetition_penalty": 1.2,
    "seed": 7,
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
}

MODEL_FIELDS = (
    "backend",
    "model_family",
    "model_id",
    "model_revision",
    "model_fingerprint",
    "weights_fingerprint",
    "model_config_fingerprint",
    "tokenizer_fingerprint",
    "voice_conditioning_fingerprint",
    "s3_tokenizer_id",
    "s3_tokenizer_revision",
    "runtime_versions",
    "quantization",
    "t3_layers",
    "t3_width",
    "attention_heads",
    "speech_vocab_size",
    "valid_speech_codes",
    "capture_convention",
    "target_head",
    "speech_code_rate_hz",
    "mel_frame_rate_hz",
    "generation",
)
INPUT_FIELDS = ("raw_text", "normalized_text")
TOKEN_FIELDS = ("index", "id", "text", "char_start", "char_end")
SPEECH_CODE_FIELDS = (
    "index",
    "id",
    "start_seconds",
    "end_seconds",
    "mel_start",
    "mel_end",
    "raw_probability",
    "raw_log_probability",
)
CANDIDATE_FIELDS = ("id", "probability", "special_token")
HEAD_FIELDS = (
    "schema_version",
    "top_k",
    "vocab_size",
    "target_ids",
    "target_probabilities",
    "target_log_probabilities",
    "target_ranks",
    "source_coordinate",
    "target_head",
    "normalization",
    "special_token_ids",
    "generation_processors_excluded",
    "warnings",
)
FITTED_FIELDS = (
    "schema_version",
    "layers",
    "target_ids",
    "target_probabilities",
    "target_log_probabilities",
    "target_ranks",
    "source_coordinate",
    "target_head",
    "normalization",
    "artifact",
    "warnings",
)
REPLAY_FIELDS = ("policy", "max_abs_logit_error")
TRACE_FIELDS = (
    "layers",
    "gradient_l2",
    "gradient_share",
    "gradient_text_mass",
    "attention_share",
    "attention_text_mass",
    "target_log_probability",
    "score_kind",
    "attention_kind",
    "warnings",
)
TRACE_SELECTION_FIELDS = (
    "speech_code_index",
    "speech_code_id",
    "start_seconds",
    "end_seconds",
    "mel_start",
    "mel_end",
    "raw_probability",
)

FORBIDDEN_KEYS = {
    "analysis_id",
    "parent_analysis_id",
    "branch_analysis_id",
    "audio_data_url",
    "waveform",
    "output_waveform",
    "generated_audio",
    "sample_rate",
    "duration_seconds",
    "nominal_content_duration_seconds",
    "trailing_audio_seconds",
}
ANALYSIS_HANDLE_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class TTSExample:
    """One pinned prompt and its reviewed teaching metadata."""

    example_id: str
    title: str
    teaching_role: str
    teaching_purpose: str
    prompt: str
    selected_position: Mapping[str, Any] | None
    intervention: Mapping[str, Any] | None


@dataclass(frozen=True)
class TTSCatalog:
    """Pinned public metadata for the complete TTS explorer family."""

    label: str
    description: str
    provenance: Mapping[str, Any]
    examples: tuple[TTSExample, ...]
    report_count: int


def _copy_fields(source: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """Deep-copy an explicit allowlist from a JSON object."""

    return {key: copy.deepcopy(source[key]) for key in fields if key in source}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def load_catalog(catalog_path: Path, curated_path: Path) -> TTSCatalog:
    """Combine the ten-prompt explorer catalog with curated provenance."""

    catalog = load_static_explorer_catalog(catalog_path)
    source = json.loads(curated_path.read_text(encoding="utf-8"))
    families = _mapping(source.get("families"), label="families")
    family = _mapping(families.get("tts"), label="families.tts")
    raw_examples = _list(family.get("examples"), label="families.tts.examples")
    curated_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_example in raw_examples:
        example = _mapping(raw_example, label="TTS example")
        example_id = str(example.get("id"))
        if example_id in curated_by_id:
            raise ValueError(f"duplicate curated TTS example id {example_id}")
        curated_by_id[example_id] = example

    examples: list[TTSExample] = []
    used_curated_ids: set[str] = set()
    for catalog_example in catalog.tts_examples:
        curated_source_id = catalog_example.curated_source_id
        curated = (
            curated_by_id.get(curated_source_id)
            if curated_source_id is not None
            else None
        )
        if curated_source_id is not None and curated is None:
            raise ValueError(
                f"{catalog_example.example_id} references missing curated "
                f"source {curated_source_id}"
            )
        if curated_source_id is not None:
            if curated_source_id != catalog_example.example_id:
                raise ValueError(
                    f"{catalog_example.example_id} cannot borrow curated "
                    f"metadata from {curated_source_id}"
                )
            if curated_source_id in used_curated_ids:
                raise ValueError(f"curated TTS source {curated_source_id} is reused")
            used_curated_ids.add(curated_source_id)
            assert curated is not None
            input_metadata = _mapping(
                curated.get("input"), label=f"{curated_source_id}.input"
            )
            if input_metadata.get("prompt") != catalog_example.prompt:
                raise ValueError(
                    f"{catalog_example.example_id} prompt disagrees with its "
                    "curated source"
                )

        selected_position = None
        intervention = None
        if curated is not None:
            if "selected_position" in curated:
                selected_position = copy.deepcopy(
                    _mapping(
                        curated["selected_position"],
                        label=(f"{catalog_example.example_id}.selected_position"),
                    )
                )
            if "intervention" in curated:
                intervention = copy.deepcopy(
                    _mapping(
                        curated["intervention"],
                        label=f"{catalog_example.example_id}.intervention",
                    )
                )
        if intervention is not None and catalog_example.example_id != "tts-bridge-s9":
            raise ValueError("only tts-bridge-s9 may carry an intervention")
        title = catalog_example.title
        teaching_role = catalog_example.teaching_role
        teaching_purpose = catalog_example.teaching_purpose
        if curated is not None:
            title = str(curated["title"])
            teaching_role = str(curated["teaching_role"])
            teaching_purpose = str(curated["teaching_purpose"])
        examples.append(
            TTSExample(
                example_id=catalog_example.example_id,
                title=title,
                teaching_role=teaching_role,
                teaching_purpose=teaching_purpose,
                prompt=catalog_example.prompt,
                selected_position=selected_position,
                intervention=intervention,
            )
        )

    if used_curated_ids != set(curated_by_id):
        missing = sorted(set(curated_by_id) - used_curated_ids)
        raise ValueError(
            "the detailed TTS catalog does not preserve every curated source: "
            + ", ".join(missing)
        )
    bridge = next(
        (example for example in examples if example.example_id == "tts-bridge-s9"),
        None,
    )
    if bridge is None or bridge.intervention is None:
        raise ValueError("tts-bridge-s9 must retain its curated intervention")

    provenance = _mapping(family.get("provenance"), label="families.tts.provenance")
    return TTSCatalog(
        label=str(family["label"]),
        description=(
            f"{catalog.reports_per_family} cached fitted-readout trajectories, "
            "including one recorded residual intervention; no generated WAV "
            "is shipped pending rights review."
        ),
        provenance=copy.deepcopy(provenance),
        examples=tuple(examples),
        report_count=catalog.reports_per_family,
    )


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    form: bool,
    timeout: float,
) -> dict[str, Any]:
    """POST one request and require an object-valued JSON response."""

    if form:
        body = urllib.parse.urlencode(payload).encode("utf-8")
        content_type = "application/x-www-form-urlencoded"
    else:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        content_type = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Accept": "application/json", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {error.code}: {detail}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return result


def _candidate(source: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist one candidate and add its additive-constant-free logit."""

    result = _copy_fields(source, CANDIDATE_FIELDS)
    if "id" not in result or "probability" not in result:
        raise ValueError("candidate is missing id or probability")
    probability = float(result["probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("candidate probability is invalid")
    result["log_probability"] = math.log(probability) if probability > 0.0 else None
    return result


def _speech_code(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy_fields(source, SPEECH_CODE_FIELDS)
    missing = {"index", "id", "raw_probability"} - result.keys()
    if missing:
        raise ValueError(f"speech code is missing {', '.join(sorted(missing))}")
    return result


def _token(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy_fields(source, TOKEN_FIELDS)
    if {"index", "id", "text"} - result.keys():
        raise ValueError("text token is missing index, id, or text")
    return result


def _head_readout(
    source: Mapping[str, Any], speech_codes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Whitelist HEAD matrices and add convenient per-position records."""

    result = _copy_fields(source, HEAD_FIELDS)
    top_codes = _list(source.get("top_codes"), label="HEAD top_codes")
    result["top_codes"] = [
        [_candidate(_mapping(item, label="HEAD candidate")) for item in row]
        for row in top_codes
    ]
    positions: list[dict[str, Any]] = []
    for index, code in enumerate(speech_codes):
        positions.append(
            {
                "position": index,
                "display_position": f"S{index + 1}",
                "realized_code_id": source["target_ids"][index],
                "realized_rank": source["target_ranks"][index],
                "realized_probability": source["target_probabilities"][index],
                "realized_log_probability": source["target_log_probabilities"][index],
                "candidates": copy.deepcopy(result["top_codes"][index]),
                "start_seconds": code.get("start_seconds"),
                "end_seconds": code.get("end_seconds"),
            }
        )
    result["positions"] = positions
    result["candidate_log_value_semantics"] = (
        "log_softmax_probability; raw logits are identifiable only up to an "
        "additive constant and are not emitted by the local API"
    )
    return result


def _fitted_readout(
    source: Mapping[str, Any], speech_codes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Whitelist fitted matrices and expose explicit layer/position rows."""

    result = _copy_fields(source, FITTED_FIELDS)
    source_top_codes = _list(source.get("top_codes"), label="fitted top_codes")
    result["top_codes"] = [
        [
            [
                _candidate(_mapping(item, label="fitted candidate"))
                for item in candidates
            ]
            for candidates in layer
        ]
        for layer in source_top_codes
    ]
    rows: list[dict[str, Any]] = []
    for row_index, layer in enumerate(source["layers"]):
        positions: list[dict[str, Any]] = []
        for position, code in enumerate(speech_codes):
            positions.append(
                {
                    "position": position,
                    "display_position": f"S{position + 1}",
                    "realized_code_id": source["target_ids"][position],
                    "realized_rank": source["target_ranks"][row_index][position],
                    "realized_probability": source["target_probabilities"][row_index][
                        position
                    ],
                    "realized_log_probability": source["target_log_probabilities"][
                        row_index
                    ][position],
                    "candidates": copy.deepcopy(
                        result["top_codes"][row_index][position]
                    ),
                    "start_seconds": code.get("start_seconds"),
                    "end_seconds": code.get("end_seconds"),
                }
            )
        rows.append({"layer": layer, "label": f"L{layer}", "positions": positions})
    result["rows"] = rows
    result["candidate_log_value_semantics"] = (
        "log_softmax_probability; raw logits are identifiable only up to an "
        "additive constant and are not emitted by the local API"
    )
    return result


def sanitize_generation(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build the public generation payload exclusively from allowlisted fields."""

    model = _mapping(source.get("model"), label="generation.model")
    input_payload = _mapping(source.get("input"), label="generation.input")
    output = _mapping(source.get("output"), label="generation.output")
    fitted = _mapping(
        source.get("fitted_speech_code_jlens"),
        label="generation.fitted_speech_code_jlens",
    )
    raw_codes = _list(output.get("speech_codes"), label="output.speech_codes")
    codes = [_speech_code(_mapping(code, label="speech code")) for code in raw_codes]
    tokens = [
        _token(_mapping(token, label="input token"))
        for token in _list(input_payload.get("tokens"), label="input.tokens")
    ]
    replay_source = _mapping(source.get("replay") or {}, label="generation.replay")
    return {
        "schema_version": source.get("schema_version"),
        "model": _copy_fields(model, MODEL_FIELDS),
        "input": {**_copy_fields(input_payload, INPUT_FIELDS), "tokens": tokens},
        "output": {
            "speech_codes": codes,
            "speech_head_candidates": _head_readout(
                _mapping(
                    output.get("speech_head_candidates"),
                    label="output.speech_head_candidates",
                ),
                codes,
            ),
        },
        "fitted_speech_code_jlens": _fitted_readout(fitted, codes),
        "replay": _copy_fields(replay_source, REPLAY_FIELDS),
        "warnings": copy.deepcopy(source.get("warnings") or []),
        "generated_audio_included": False,
    }


def reject_generation_cap(
    source: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    """Reject a sequence that filled an exposed generation safety cap.

    Chatterbox omits the stop token from ``speech_codes``. Reaching the exact
    configured maximum therefore cannot be distinguished from a naturally
    completed sequence of the same length, so the static exporter takes the
    conservative publication-safe choice and asks for a shorter prompt or a
    separately reviewed larger pinned cap.
    """

    model = source.get("model")
    if not isinstance(model, Mapping):
        return
    generation = model.get("generation")
    if not isinstance(generation, Mapping):
        return
    exposed_cap = generation.get("max_speech_tokens")
    if exposed_cap is None:
        return
    try:
        cap = int(exposed_cap)
    except (TypeError, ValueError) as error:
        raise ValueError("generation max_speech_tokens is invalid") from error
    if cap <= 0:
        raise ValueError("generation max_speech_tokens must be positive")
    codes = payload.get("output", {}).get("speech_codes")
    if not isinstance(codes, list):
        raise ValueError("sanitized generation has no speech-code list")
    if len(codes) >= cap:
        raise ValueError(
            f"generation produced {len(codes)} speech codes and reached the "
            f"exposed {cap}-code safety cap; do not publish a possibly "
            "truncated example"
        )


def sanitize_trace(source: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist one trace response while deliberately dropping analysis_id."""

    selection = _mapping(source.get("selection"), label="trace.selection")
    tokens = [
        _token(_mapping(token, label="trace text token"))
        for token in _list(source.get("text_tokens"), label="trace.text_tokens")
    ]
    return {
        "selection": _copy_fields(selection, TRACE_SELECTION_FIELDS),
        "text_tokens": tokens,
        **_copy_fields(source, TRACE_FIELDS),
    }


def _rounded_percent_matches(probability: float, expected_percent: float) -> bool:
    return math.isclose(
        round(probability * 100.0, 3),
        expected_percent,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def merge_curated_bridge_intervention(
    example: TTSExample, payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Attach the reviewed bridge branch summary after checking the live baseline."""

    if example.intervention is None:
        return None
    if example.selected_position is None:
        raise ValueError("an intervention example needs a selected position")

    selected = example.selected_position
    intervention = example.intervention
    position = int(selected["zero_based_code_index"])
    codes = payload["output"]["speech_codes"]
    head = payload["output"]["speech_head_candidates"]
    baseline = _mapping(intervention.get("baseline_winner"), label="baseline_winner")
    candidate = _mapping(
        intervention.get("baseline_runner_up"), label="baseline_runner_up"
    )
    steered = _mapping(intervention.get("steered_output"), label="steered_output")

    realized_code = int(codes[position]["id"])
    if realized_code != int(selected["realized_code_id"]):
        raise ValueError("live bridge baseline code does not match curated S9")
    if realized_code != int(baseline["code_id"]):
        raise ValueError("curated bridge baseline records disagree")
    realized_probability = float(head["target_probabilities"][position])
    realized_log_probability = float(head["target_log_probabilities"][position])
    realized_rank = int(head["target_ranks"][position])
    if realized_rank != int(baseline["rank"]):
        raise ValueError("live bridge baseline rank changed")
    if not _rounded_percent_matches(
        realized_probability, float(baseline["probability_percent"])
    ):
        raise ValueError("live bridge baseline probability changed")

    candidate_id = int(candidate["code_id"])
    top_candidates = head["top_codes"][position]
    candidate_index = next(
        (
            index
            for index, item in enumerate(top_candidates)
            if int(item["id"]) == candidate_id
        ),
        None,
    )
    if candidate_index is None or candidate_index + 1 != int(candidate["rank"]):
        raise ValueError("curated bridge candidate is not at its recorded rank")
    candidate_probability = float(top_candidates[candidate_index]["probability"])
    if not _rounded_percent_matches(
        candidate_probability, float(candidate["probability_percent"])
    ):
        raise ValueError("live bridge runner-up probability changed")
    candidate_log_probability = math.log(candidate_probability)
    logit_gap = realized_log_probability - candidate_log_probability
    if not math.isclose(
        logit_gap,
        float(candidate["logit_gap_from_winner"]),
        rel_tol=0.0,
        abs_tol=5e-5,
    ):
        raise ValueError("live bridge runner-up logit gap changed")

    edited_layers = [
        int(str(layer).removeprefix("L"))
        for layer in _list(
            intervention.get("edited_layers"), label="intervention.edited_layers"
        )
    ]
    branch_check = str(intervention["branch_check"])
    exact_match = "exactly matched" in branch_check
    chosen_norm = intervention["relative_residual_norm_per_edited_coordinate"]
    return {
        "kind": "replayed_residual_steering_with_direct_forced_code_check",
        "source": "recorded_reviewed_static_public_reports_v1",
        "request": {
            "speech_code_index": position,
            "display_position": str(selected["display_position"]),
            "target_code_id": candidate_id,
            "layers": edited_layers,
        },
        "baseline": {
            "realized_code_id": realized_code,
            "realized_rank": realized_rank,
            "realized_probability": realized_probability,
            "realized_log_probability": realized_log_probability,
            "curated_probability_percent": baseline["probability_percent"],
        },
        "candidate": {
            "code_id": candidate_id,
            "rank": candidate["rank"],
            "probability": candidate_probability,
            "log_probability": candidate_log_probability,
            "logit_gap_from_winner": candidate["logit_gap_from_winner"],
            "curated_probability_percent": candidate["probability_percent"],
        },
        "residual_steered": {
            "anchor_realized_code_id": steered["code_id"],
            "rank": steered["rank"],
            "probability": float(steered["probability_percent"]) / 100.0,
            "curated_probability_percent": steered["probability_percent"],
            "chosen_relative_residual_norm": chosen_norm,
            "intervention": {
                "kind": intervention["kind"],
                "speech_code_index": position,
                "target_code_id": candidate_id,
                "layers": edited_layers,
                "chosen_relative_residual_norm": chosen_norm,
                "target_became_raw_top1": int(steered["rank"]) == 1,
            },
        },
        "suffix_effect": _copy_fields(
            _mapping(intervention.get("suffix_effect"), label="suffix_effect"),
            (
                "same_index_codes_changed",
                "total_same_index_codes_compared",
                "downstream_codes_changed",
                "baseline_sequence_length",
                "steered_sequence_length",
            ),
        ),
        "direct_forced": {
            "anchor_realized_code_id": steered["code_id"],
            "sequence_exact_match": exact_match,
        },
        "direct_forced_sequence_exact_match": exact_match,
        "branch_check": branch_check,
        "generated_audio_included": False,
    }


def validate_safe(value: Any, *, path: str = "$") -> None:
    """Reject ephemeral handles and reconstructable generated-audio artifacts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = key.lower()
            if key in FORBIDDEN_KEYS or lowered.endswith("_analysis_id"):
                raise ValueError(f"forbidden field {path}.{key}")
            if "waveform" in lowered or "audio_data" in lowered:
                raise ValueError(f"forbidden generated-audio field {path}.{key}")
            if (
                lowered in {"audio_url", "audio_path", "audio_asset"}
                and item is not None
            ):
                raise ValueError(f"non-null generated-audio reference {path}.{key}")
            validate_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_safe(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.lower().startswith("data:audio/"):
            raise ValueError(f"embedded audio URI at {path}")
        if ANALYSIS_HANDLE_PATTERN.fullmatch(value):
            raise ValueError(f"ephemeral analysis handle at {path}")


def _validate_provenance(report: Mapping[str, Any]) -> None:
    provenance = report["provenance"]
    pinned_model = provenance["model"]
    live_model = report["payload"]["model"]
    for pinned_key, live_key in (
        ("id", "model_id"),
        ("revision", "model_revision"),
        ("model_fingerprint", "model_fingerprint"),
        ("model_config_fingerprint", "model_config_fingerprint"),
        ("s3_tokenizer_id", "s3_tokenizer_id"),
        ("s3_tokenizer_revision", "s3_tokenizer_revision"),
        ("speech_head_vocabulary_size", "speech_vocab_size"),
        ("ordinary_acoustic_code_count", "valid_speech_codes"),
        ("speech_code_rate_hz", "speech_code_rate_hz"),
    ):
        if pinned_model[pinned_key] != live_model[live_key]:
            raise ValueError(f"live model disagrees with pinned {pinned_key}")

    generation = live_model.get("generation")
    if generation != EXPECTED_GENERATION:
        raise ValueError(
            "live generation settings are not the pinned deterministic "
            f"settings: {EXPECTED_GENERATION}"
        )

    pinned_lens = provenance["lens"]
    fitted = report["payload"]["fitted_speech_code_jlens"]
    artifact = fitted["artifact"]
    if fitted["layers"] != pinned_lens["source_layers"]:
        raise ValueError("live fitted layers disagree with pinned lens")
    for key in ("target_layer", "examples_fingerprint", "capture_convention"):
        if artifact[key] != pinned_lens[key]:
            raise ValueError(f"live fitted artifact disagrees with pinned {key}")
    projection = artifact["projection"]
    if projection["method"] != pinned_lens["projection_method"]:
        raise ValueError("live projection method disagrees with pinned lens")
    if projection["rank"] != pinned_lens["projection_rank"]:
        raise ValueError("live projection rank disagrees with pinned lens")
    if projection["seed"] != pinned_lens["projection_seed"]:
        raise ValueError("live projection seed disagrees with pinned lens")


def validate_report(report: Mapping[str, Any]) -> None:
    """Check every matrix, trace coordinate, score, and pinned identity."""

    payload = report["payload"]
    if report["source"]["prompt"] != payload["input"]["raw_text"]:
        raise ValueError("report prompt does not match the generated input")
    codes = payload["output"]["speech_codes"]
    width = len(codes)
    if not width or [code["index"] for code in codes] != list(range(width)):
        raise ValueError("speech-code positions are empty or non-contiguous")

    head = payload["output"]["speech_head_candidates"]
    for key in (
        "target_ids",
        "target_probabilities",
        "target_log_probabilities",
        "target_ranks",
        "top_codes",
        "positions",
    ):
        if len(head[key]) != width:
            raise ValueError(f"HEAD {key} width mismatch")
    vocabulary_size = int(head["vocab_size"])
    for position, code in enumerate(codes):
        if int(head["target_ids"][position]) != int(code["id"]):
            raise ValueError(f"HEAD target/code mismatch at S{position + 1}")
        probability = float(head["target_probabilities"][position])
        log_probability = float(head["target_log_probabilities"][position])
        if probability <= 0.0 or not math.isclose(
            math.log(probability), log_probability, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"HEAD probability/log mismatch at S{position + 1}")
        rank = int(head["target_ranks"][position])
        if not 1 <= rank <= vocabulary_size:
            raise ValueError(f"HEAD rank out of range at S{position + 1}")
        if not head["top_codes"][position]:
            raise ValueError(f"HEAD candidate list is empty at S{position + 1}")
        explicit = head["positions"][position]
        if (
            int(explicit["position"]) != position
            or int(explicit["realized_code_id"]) != int(code["id"])
            or int(explicit["realized_rank"]) != rank
            or not math.isclose(
                float(explicit["realized_probability"]),
                probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(explicit["realized_log_probability"]),
                log_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"HEAD explicit row mismatch at S{position + 1}")

    fitted = payload["fitted_speech_code_jlens"]
    layers = fitted["layers"]
    if len(fitted["rows"]) != len(layers):
        raise ValueError("fitted explicit-row/layer mismatch")
    for key in (
        "target_probabilities",
        "target_log_probabilities",
        "target_ranks",
        "top_codes",
    ):
        matrix = fitted[key]
        if len(matrix) != len(layers) or any(len(row) != width for row in matrix):
            raise ValueError(f"fitted {key} matrix mismatch")
    for row_index, row in enumerate(fitted["rows"]):
        if row["layer"] != layers[row_index] or len(row["positions"]) != width:
            raise ValueError(f"fitted explicit row {row_index} mismatch")
        for position, explicit in enumerate(row["positions"]):
            probability = float(fitted["target_probabilities"][row_index][position])
            log_probability = float(
                fitted["target_log_probabilities"][row_index][position]
            )
            rank = int(fitted["target_ranks"][row_index][position])
            if probability <= 0.0 or not math.isclose(
                math.log(probability),
                log_probability,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"fitted probability/log mismatch at row {row_index}, "
                    f"S{position + 1}"
                )
            if not 1 <= rank <= vocabulary_size:
                raise ValueError(
                    f"fitted rank out of range at row {row_index}, S{position + 1}"
                )
            if not fitted["top_codes"][row_index][position]:
                raise ValueError(
                    f"fitted candidate list is empty at row {row_index}, "
                    f"S{position + 1}"
                )
            if (
                int(explicit["position"]) != position
                or int(explicit["realized_code_id"]) != int(codes[position]["id"])
                or int(explicit["realized_rank"]) != rank
                or not math.isclose(
                    float(explicit["realized_probability"]),
                    probability,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(explicit["realized_log_probability"]),
                    log_probability,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"fitted explicit value mismatch at row {row_index}, "
                    f"S{position + 1}"
                )

    traces = payload["traces_by_position"]
    if set(traces) != {str(position) for position in range(width)}:
        raise ValueError("trace position set does not match speech-code positions")
    input_tokens = payload["input"]["tokens"]
    for position in range(width):
        trace = traces[str(position)]
        selection = trace["selection"]
        if int(selection["speech_code_index"]) != position:
            raise ValueError(f"trace selection index mismatch at S{position + 1}")
        if int(selection["speech_code_id"]) != int(codes[position]["id"]):
            raise ValueError(f"trace selection code mismatch at S{position + 1}")
        if trace["text_tokens"] != input_tokens:
            raise ValueError(f"trace text-token mismatch at S{position + 1}")
        layer_count = len(trace["layers"])
        token_count = len(trace["text_tokens"])
        for key in (
            "gradient_l2",
            "gradient_share",
            "attention_share",
        ):
            matrix = trace[key]
            if len(matrix) != layer_count or any(
                len(row) != token_count for row in matrix
            ):
                raise ValueError(f"trace {key} matrix mismatch at S{position + 1}")
        for key in (
            "gradient_text_mass",
            "attention_text_mass",
            "target_log_probability",
        ):
            if len(trace[key]) != layer_count:
                raise ValueError(f"trace {key} layer mismatch at S{position + 1}")

    _validate_provenance(report)
    validate_safe(report)


def compact_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Render stable compact JSON used for both the file and its digest."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_report(path: Path, report: Mapping[str, Any]) -> tuple[str, int]:
    body = compact_json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest(), len(body)


def _report_digest(path: Path) -> tuple[str, int]:
    body = path.read_bytes()
    return hashlib.sha256(body).hexdigest(), len(body)


def _manifest_entry(
    example: TTSExample, *, digest: str, byte_count: int
) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "title": example.title,
        "teaching_role": example.teaching_role,
        "summary": example.teaching_purpose,
        "prompt": example.prompt,
        "audio_url": None,
        "report_url": (
            f"/audio-jacobian-lens/explorer/data/tts/{example.example_id}.json"
        ),
        "sha256": digest,
        "bytes": byte_count,
    }


def _validate_existing_report(
    path: Path,
    *,
    example: TTSExample,
    provenance: Mapping[str, Any],
) -> tuple[str, int]:
    """Validate a cached report before reusing it in a rebuilt manifest."""

    source = json.loads(path.read_text(encoding="utf-8"))
    report = _mapping(source, label=f"existing report {path}")
    expected_identity = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": "tts",
        "example_id": example.example_id,
        "title": example.title,
        "teaching_role": example.teaching_role,
        "teaching_purpose": example.teaching_purpose,
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected:
            raise ValueError(f"existing {example.example_id} report has stale {key}")
    if report.get("provenance") != provenance:
        raise ValueError(f"existing {example.example_id} report has stale provenance")
    source_metadata = _mapping(
        report.get("source"), label=f"existing {example.example_id}.source"
    )
    if source_metadata.get("prompt") != example.prompt:
        raise ValueError(f"existing {example.example_id} report has a stale prompt")
    payload = _mapping(
        report.get("payload"), label=f"existing {example.example_id}.payload"
    )
    has_intervention = "intervention_comparison" in payload
    if has_intervention != (example.intervention is not None):
        raise ValueError(
            f"existing {example.example_id} report has stale intervention data"
        )
    reject_generation_cap(payload, payload)
    validate_report(report)
    return _report_digest(path)


def _build_report(
    example: TTSExample,
    provenance: Mapping[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    intervention = merge_curated_bridge_intervention(example, payload)
    if intervention is not None:
        payload["intervention_comparison"] = intervention
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": "tts",
        "example_id": example.example_id,
        "title": example.title,
        "teaching_role": example.teaching_role,
        "teaching_purpose": example.teaching_purpose,
        "provenance": copy.deepcopy(provenance),
        "source": {
            "kind": "local_chatterbox_capture",
            "prompt": example.prompt,
            "rights_status": "project_authored_text",
            "generated_audio_included": False,
        },
        "payload": payload,
    }


def export_tts_family(
    *,
    base_url: str,
    output_dir: Path,
    catalog: TTSCatalog,
    timeout: float,
    example_ids: set[str] | None = None,
    resume_valid: bool = False,
) -> dict[str, Any]:
    """Export a complete ordered family, safely reusing validated reports.

    ``example_ids`` limits expensive generation, not the resulting manifest.
    Every unselected report must already exist and pass the same report,
    provenance, prompt, rights, and safety checks. With ``resume_valid``, a
    selected report is also reused when valid; invalid selected reports are
    regenerated.
    """

    base_url = base_url.rstrip("/")
    catalog_ids = {example.example_id for example in catalog.examples}
    if len(catalog_ids) != len(catalog.examples):
        raise ValueError("the TTS catalog contains duplicate example IDs")
    if len(catalog.examples) != catalog.report_count:
        raise ValueError("the TTS catalog report count does not match its examples")
    selected_ids = catalog_ids if example_ids is None else set(example_ids)
    unknown_ids = selected_ids - catalog_ids
    if unknown_ids:
        raise ValueError("unknown TTS example ID(s): " + ", ".join(sorted(unknown_ids)))

    manifest_entries: list[dict[str, Any]] = []
    for example in catalog.examples:
        report_path = output_dir / f"{example.example_id}.json"
        selected = example.example_id in selected_ids
        should_try_existing = not selected or resume_valid
        if should_try_existing and report_path.is_file():
            try:
                digest, byte_count = _validate_existing_report(
                    report_path,
                    example=example,
                    provenance=catalog.provenance,
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                if not selected:
                    raise ValueError(
                        f"unselected report {example.example_id} cannot be "
                        "reused; select it for regeneration"
                    ) from error
                print(
                    f"regenerating invalid {example.example_id}: {error}",
                    flush=True,
                )
            else:
                manifest_entries.append(
                    _manifest_entry(example, digest=digest, byte_count=byte_count)
                )
                print(
                    f"reused {report_path} ({byte_count:,} bytes)",
                    flush=True,
                )
                continue
        elif not selected:
            raise ValueError(
                f"unselected report {example.example_id} is missing; select "
                "it for generation"
            )

        print(f"generating {example.example_id}: {example.prompt}", flush=True)
        raw = _post_json(
            f"{base_url}/api/chatterbox/generate",
            {"text": example.prompt},
            form=True,
            timeout=timeout,
        )
        analysis_id = raw.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id:
            raise RuntimeError("generation response has no analysis_id for traces")
        payload = sanitize_generation(raw)
        reject_generation_cap(raw, payload)
        speech_codes = payload["output"]["speech_codes"]
        traces: dict[str, Any] = {}
        for position in range(len(speech_codes)):
            trace = _post_json(
                f"{base_url}/api/chatterbox/trace",
                {
                    "analysis_id": analysis_id,
                    "speech_code_index": position,
                },
                form=False,
                timeout=timeout,
            )
            traces[str(position)] = sanitize_trace(trace)
            if (
                position == 0
                or (position + 1) % 10 == 0
                or position + 1 == len(speech_codes)
            ):
                print(
                    f"  traces {position + 1}/{len(speech_codes)}",
                    flush=True,
                )
        payload["traces_by_position"] = traces
        report = _build_report(example, catalog.provenance, payload)
        validate_report(report)
        digest, byte_count = _write_report(report_path, report)
        manifest_entries.append(
            _manifest_entry(example, digest=digest, byte_count=byte_count)
        )
        print(f"wrote {report_path} ({byte_count:,} bytes)", flush=True)

    if len(manifest_entries) != catalog.report_count:
        raise ValueError("refusing to write an incomplete TTS manifest")
    manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "family": "tts",
        "mode": "static_cached_explorer",
        "label": catalog.label,
        "description": catalog.description,
        "provenance": copy.deepcopy(catalog.provenance),
        "report_count": len(manifest_entries),
        "reports": manifest_entries,
    }
    validate_safe(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        type=Path,
        required=True,
        help="The audio-jacobian-lens directory inside the static site checkout.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/static_explorer_catalog_v2.json"),
        help="Versioned catalog that defines the ordered ten TTS prompts.",
    )
    parser.add_argument(
        "--curated-source",
        type=Path,
        default=Path("data/static_public_reports_v1.json"),
        help="Curated provenance and bridge-intervention source.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--example-id",
        action="append",
        default=None,
        help=(
            "Generate only this catalog ID (repeatable); every other report "
            "must already exist and validate so the manifest remains complete."
        ),
    )
    parser.add_argument(
        "--resume-valid",
        action="store_true",
        help="Reuse valid selected reports and regenerate only missing/invalid ones.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    catalog = load_catalog(args.catalog, args.curated_source)
    export_tts_family(
        base_url=args.base_url,
        output_dir=args.site_root / "explorer" / "data" / "tts",
        catalog=catalog,
        timeout=args.timeout_seconds,
        example_ids=(set(args.example_id) if args.example_id else None),
        resume_valid=args.resume_valid,
    )


if __name__ == "__main__":
    main()
