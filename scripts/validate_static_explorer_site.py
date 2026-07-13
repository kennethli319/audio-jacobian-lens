#!/usr/bin/env python3
"""Validate the publishable, backend-free Audio Jacobian Lens site bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SITE_PREFIX = "/audio-jacobian-lens/"
PUBLIC_BASE = "https://kennethli319.github.io/audio-jacobian-lens/"
PUBLIC_CATALOG_URL = (
    "https://github.com/kennethli319/audio-jacobian-lens/blob/main/"
    "data/static_explorer_catalog_v2.json"
)
FAMILIES = ("asr", "speech")
EXPECTED_REPORT_COUNT = 10
EXPLORER_ASSET_VERSION = "20260713-22"
CANONICAL_DETAILED_ROUTES = {
    "asr": SITE_PREFIX,
    "speech": f"{SITE_PREFIX}speech/",
}
FINDINGS_ROUTES = {
    "asr": f"{SITE_PREFIX}findings/",
    "speech": f"{SITE_PREFIX}findings/speech/",
}
LEGACY_EXPLORER_ROUTES = {
    family: f"{SITE_PREFIX}explorer/{family}/" for family in FAMILIES
}
STEERING_ROUTE = f"{SITE_PREFIX}steering/"
RETIRED_PUBLIC_TTS_PATHS = (
    "tts",
    "findings/tts",
    "explorer/tts",
    "explorer/data/tts",
)
STEERING_ASSET_VERSION = "20260713-2"
FORBIDDEN_KEYS = {
    "analysis_id",
    "parent_analysis_id",
    "branch_analysis_id",
    "audio_data_url",
    "generated_audio",
    "output_waveform",
}
SPEECH_TERMINATION_SCRIPT_MARKERS = (
    "function renderSpeechTerminationStatus()",
    'data-speech-termination="budget-exhausted"',
    "response may be truncated",
)
SPEECH_TERMINATION_CSS_MARKERS = (".generation-status.capped",)
ASR_DECODER_HIERARCHY_SCRIPT_MARKERS = (
    'const asrDecoderCell = family === "asr" && kind === "decoder";',
    'data-value-role="top-candidate"',
    'realizedBadge = asrDecoderCell || (family === "asr" && kind === "head")',
    'const cellWidth = family === "asr" ? 92 : 82;',
    "renderSpeechRows(),",
    "Decoder boxes show each layer's top candidate",
)
ASR_DECODER_HIERARCHY_CSS_MARKERS = (
    '[data-family="asr"] .speech-matrix-grid .matrix-cell .matrix-cell-label',
    '[data-family="asr"] .speech-matrix-grid .matrix-cell .realized-rank-badge',
)
CROSS_FAMILY_SYNCHRONIZED_SCROLL_SCRIPT_MARKERS = (
    'const scrollableEncoder = family === "asr" && streamName === "encoder";',
    "const encoderCellWidth = phoneMode ? 28 : 72;",
    'const continuous = family === "asr" || family === "speech";',
    "const windowSize = continuous ? Math.max(tokens.length, 1) : 8;",
    'All ${count} ${family === "speech" ? "generated text positions" : "tokens"} · scroll horizontally',
    "const ttsCellWidth = 54;",
    'class="position-timeline scrollable',
    "function scrollTargetIntoHorizontalView(",
    "function revealSynchronizedSelection(",
    'workspace.querySelector(".position-timeline.scrollable")',
    'workspace.querySelectorAll(".scrollable-matrix-panel .layer-matrix")',
    'workspace.querySelectorAll(".speech-matrix-scroll")',
    'matrixScroller.querySelector(`.matrix-cell[data-kind="tts-layer"]',
    'syncSelectionDOM({ reveal: true, behavior: "auto" });',
    "target.focus({ preventScroll: true });",
)
CROSS_FAMILY_SYNCHRONIZED_SCROLL_CSS_MARKERS = (
    ".position-timeline.scrollable",
    ".scrollable-matrix-panel .layer-matrix",
    ".scrollable-matrix-panel .matrix-row",
    ".speech-matrix-scroll",
    "overflow-x: auto",
)
ASR_PHONE_SIGNATURE_SCRIPT_MARKERS = (
    'phoneSignatureEnabled: family === "asr"',
    '!["0", "false", "off"].includes(String(queryParams.get("phone")).toLowerCase())',
    "function validatePhoneSignatureReport(payload)",
    "function renderPhoneSignatureControl()",
    "On by default · turn off for normal token J-Lens readouts",
    'queryParams.set("phone", phoneQueryValue);',
    "const phoneCell = encoderPhoneMode(kind);",
    "label: phoneMode ? compactText(top?.phone)",
    "descriptor.candidates.slice(0, 5)",
    "Audio and alignment attribution",
    "exact 100/20/80 ms encoder pooling",
)
ASR_PHONE_SIGNATURE_CSS_MARKERS = (
    ".phone-signature-control",
    ".matrix-cell.phone-signature-cell .matrix-cell-label",
    ".phone-candidate-row",
    ".rights-block",
    ".explorer-tooltip",
)
ASR_ARCHITECTURE_SCRIPT_MARKERS = (
    'family === "asr" ? "Encoder: Across the audio representation"',
    'family === "asr" ? "Decoder: As each token resolves"',
    'class="matrix-architecture" aria-label="Encoder architecture"',
    'class="matrix-architecture" aria-label="Decoder architecture"',
    "Bidirectional audio-time states",
    "Final L${finalLayer} state → LM head · causal token time",
)
ASR_ARCHITECTURE_CSS_MARKERS = (".matrix-architecture",)
ASR_RECORDED_REPLAY_SCRIPT_MARKERS = (
    "recorded_intervention_replay",
    "audio-jacobian-lens.recorded-asr-intervention-replay",
    "function composeReplayReport(",
    "function activateReplayCondition(",
    "controls: `${renderReplayControl()}${renderPhoneSignatureControl()}`",
    'url.searchParams.set("condition", condition.id);',
    "Updates encoder · decoder · HEAD",
    'entry.id === "asr-laurel-yanny"',
    '<em class="sample-tag">steering exp</em>',
    "function effectiveProvenance()",
    "Original Laurel/Yanny post",
)
ASR_RECORDED_REPLAY_CSS_MARKERS = (
    ".recorded-replay",
    ".replay-condition-buttons",
    ".replay-active-summary",
    ".replay-attribution",
    ".sample-tag",
)
STEERING_SCRIPT_MARKERS = (
    'data.mode !== "static_recorded_checkpoints"',
    "checkpoint.recorded !== true || checkpoint.interpolated !== false",
    "baseline.decisions[targetKey]",
    "target.evidence?.tone",
    "target.method?.coefficient_policy",
    "checkpoint.coefficient_scale",
    "sequence_probability_product",
    "fetch(resultsUrl",
)
STEERING_CSS_MARKERS = (
    ".site-header { width: min(1180px, calc(100% - 48px));",
    ".site-nav { display: flex;",
    "@media (max-width: 740px)",
    ".site-nav { order: 3; width: 100%; }",
    ".coefficient-heatmap",
    ".heatmap-cell",
    ".decision-grid",
    ".candidate-row.target",
    ".evidence-badge.strong",
    ".evidence-badge.limited",
    "@media (max-width: 560px)",
    "@media (prefers-reduced-motion: reduce)",
)
PHONE_SIGNATURE_FIELDS = {
    "phone",
    "rank",
    "similarity",
}
PHONE_SIGNATURE_METADATA_FIELDS = {
    "available",
    "display_unit",
    "effective_display_hop_seconds",
    "effective_display_window_seconds",
    "interpretation",
    "method",
    "phone_inventory",
    "phone_inventory_size",
    "prototype_fit_opened_eval_splits",
    "prototype_fit_rows",
    "prototype_fit_split",
    "prototype_lens_examples",
    "schema_version",
    "score_kind",
    "signature_top_k",
    "silence_or_unknown_class_available",
    "training_unit",
}
PUBLIC_PHONE_INVENTORY = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UW",
    "V",
    "W",
    "Z",
)
ASR_REPLAY_SCHEMA_ID = "audio-jacobian-lens.recorded-asr-intervention-replay"
ASR_REPLAY_MODE = "static_recorded_analyses"
ASR_REPLAY_CONDITIONS = ("baseline", "yanny", "laurel")
ASR_REPLAY_ANALYSIS_FIELDS = {"metadata", "transcription", "encoder", "decoder"}
ASR_REPLAY_EXPECTED = {
    "baseline": {
        "text": "Lily!",
        "token_ids": [20037, 0],
        "target_match": False,
        "budget_fraction": 0.0,
        "coefficient_scale": 0.0,
        "evidence_tier": "observed_baseline",
        "layer_count": 0,
        "schedule_count": 0,
    },
    "yanny": {
        "text": "Yanny!",
        "token_ids": [575, 7737, 0],
        "target_match": True,
        "budget_fraction": 0.035,
        "coefficient_scale": 0.035,
        "evidence_tier": "open_loop_cross_fit_reproduced",
        "layer_count": 4,
        "schedule_count": 4,
    },
    "laurel": {
        "text": "Laurel",
        "token_ids": [43442],
        "target_match": True,
        "budget_fraction": 0.1452915875040831,
        "coefficient_scale": 0.7,
        "evidence_tier": "target_conditioned_clip_specific_existence",
        "layer_count": 4,
        "schedule_count": 5,
    },
}
ASR_REPLAY_ROOT_FIELDS = {
    "schema_id",
    "schema_version",
    "mode",
    "default_condition",
    "conditions",
}
ASR_REPLAY_CONDITION_FIELDS = {
    "id",
    "label",
    "recorded",
    "interpolated",
    "generated",
    "budget_fraction",
    "coefficient_scale",
    "evidence",
    "method",
    "layers",
    "schedule",
}
ASR_REPLAY_GENERATED_FIELDS = {"text", "token_ids", "target_match"}
ASR_REPLAY_EVIDENCE_FIELDS = {"tier", "badge", "tone", "summary"}
ASR_REPLAY_METHOD_FIELDS = {
    "kind",
    "label",
    "description",
    "coefficient_policy",
}
ASR_REPLAY_SCHEDULE_FIELDS = {
    "phone",
    "start_seconds",
    "end_seconds",
    "start_position",
    "end_position",
}
ASR_REPLAY_RIGHTS_STATUS = "attributed_under_source_page_license"
ASR_REPLAY_SOURCE_URL = "https://hrbosker.github.io/demos/laurel-yanny/"
ASR_REPLAY_LICENSE = "CC BY 4.0"
ASR_REPLAY_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ASR_REPLAY_ATTRIBUTION = (
    "‘Laurel/Yanny — original’ (Audio S7), reproduced from Hans Rutger "
    "Bosker's ‘Laurel or Yanny?’ demo. Bosker (2018) describes the viral "
    "clip as originating from Vocabulary.com's ‘laurel’ pronunciation."
)
ASR_REPLAY_MODIFICATION_NOTICE = (
    "Audio S7 MP3 is reproduced byte-for-byte unchanged; the cached model "
    "analysis and steering overlays are project-authored."
)
ASR_REPLAY_FORBIDDEN_KEYS = {
    "analysis_id",
    "audio",
    "audio_data_url",
    "audio_url",
    "branch_analysis_id",
    "coefficients",
    "delta",
    "deltas",
    "direction",
    "generated_audio",
    "model_input_wav",
    "output_waveform",
    "parent_analysis_id",
    "pullback_audit",
    "recording",
    "repository_path",
    "residual",
    "residuals",
    "waveform",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _site_path(site_root: Path, url: str) -> Path:
    if not url.startswith(SITE_PREFIX):
        raise ValueError(f"published URL is outside {SITE_PREFIX}: {url}")
    return site_root / url.removeprefix(SITE_PREFIX)


def _manifest_reports(
    manifest: Mapping[str, Any], *, family: str
) -> list[Mapping[str, Any]]:
    reports = manifest.get("reports")
    if manifest.get("report_count") != EXPECTED_REPORT_COUNT:
        raise ValueError(
            f"{family} manifest report_count must be {EXPECTED_REPORT_COUNT}"
        )
    if not isinstance(reports, list) or len(reports) != EXPECTED_REPORT_COUNT:
        raise ValueError(
            f"{family} must publish exactly {EXPECTED_REPORT_COUNT} reports"
        )
    if not all(isinstance(entry, Mapping) for entry in reports):
        raise ValueError(f"{family} manifest contains a non-object report entry")
    ids = [str(entry.get("id") or "") for entry in reports]
    report_urls = [str(entry.get("report_url") or "") for entry in reports]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{family} manifest report IDs are empty or duplicated")
    if any(not value for value in report_urls) or len(set(report_urls)) != len(
        report_urls
    ):
        raise ValueError(f"{family} manifest report URLs are empty or duplicated")
    if family == "tts":
        if any(entry.get("audio_url") is not None for entry in reports):
            raise ValueError("TTS manifest must not publish generated-audio URLs")
    else:
        audio_urls = [str(entry.get("audio_url") or "") for entry in reports]
        if any(not value for value in audio_urls) or len(set(audio_urls)) != len(
            audio_urls
        ):
            raise ValueError(f"{family} manifest audio URLs are empty or duplicated")
    if family == "asr":
        filter_urls: list[str] = []
        for entry in reports:
            reference = entry.get("character_length_filter_cache")
            if not isinstance(reference, Mapping) or not reference.get("url"):
                raise ValueError(
                    "ASR manifest entry is missing its character-filter URL"
                )
            filter_urls.append(str(reference["url"]))
            if "featured_views" in entry:
                raise ValueError("ASR manifest uses retired featured-view metadata")
        if len(set(filter_urls)) != len(filter_urls):
            raise ValueError("ASR manifest character-filter URLs are duplicated")
    return reports


def _validate_safe(
    value: Any, *, path: str = "$", reject_artifact_files: bool = False
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS or key.endswith("_analysis_id"):
                raise ValueError(f"forbidden cached field {path}.{key}")
            _validate_safe(
                item,
                path=f"{path}.{key}",
                reject_artifact_files=reject_artifact_files,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe(
                item,
                path=f"{path}[{index}]",
                reject_artifact_files=reject_artifact_files,
            )
    elif isinstance(value, str):
        lowered = value.strip().lower().replace("\\", "/")
        if lowered.startswith("data:audio/"):
            raise ValueError(f"embedded audio URI at {path}")
        if "/users/" in lowered or "artifacts/private/" in lowered:
            raise ValueError(f"private filesystem reference at {path}")
        if reject_artifact_files and lowered.endswith(
            (".pt", ".pth", ".npz", ".npy", ".textgrid")
        ):
            raise ValueError(f"private model or alignment artifact at {path}")


def _validate_asr_manifest_provenance(manifest: Mapping[str, Any]) -> None:
    provenance = manifest.get("provenance")
    lens = provenance.get("lens") if isinstance(provenance, Mapping) else None
    if not isinstance(lens, Mapping):
        raise ValueError("ASR manifest has no composite lens provenance")
    encoder = lens.get("encoder")
    decoder = lens.get("decoder")
    phones = lens.get("phone_signature")
    if not all(isinstance(value, Mapping) for value in (encoder, decoder, phones)):
        raise ValueError("ASR manifest lacks encoder, decoder, or phone provenance")
    for label, record in (
        ("encoder", encoder),
        ("decoder", decoder),
        ("phone", phones),
    ):
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"ASR {label} provenance has no pinned SHA-256")
    relationship = str(encoder.get("public_evaluation_relationship") or "")
    if "speaker-held-out" not in relationship or "1272" not in relationship:
        raise ValueError("ASR public examples are not declared speaker-held-out")
    if (
        phones.get("signature_top_k") != 100
        or phones.get("phone_inventory_size") != 34
        or phones.get("training_split") != "train"
        or phones.get("training_rows") != 3400
        or phones.get("development_or_test_opened_for_fit") is not False
    ):
        raise ValueError("ASR phone-prototype provenance is inconsistent")
    rights = provenance.get("rights")
    if not isinstance(rights, Mapping):
        raise ValueError("ASR manifest lacks complete public-source rights provenance")
    mixed_sources = rights.get("source_url") == PUBLIC_CATALOG_URL
    if (
        rights.get("license") != "CC BY 4.0"
        or rights.get("license_url") != "https://creativecommons.org/licenses/by/4.0/"
        or rights.get("source_url")
        not in {"https://www.openslr.org/12", PUBLIC_CATALOG_URL}
        or rights.get("alignment_source_url") != "https://zenodo.org/records/2619474"
        or rights.get("alignment_license") != "CC BY 4.0"
        or not str(rights.get("attribution") or "").strip()
        or (
            mixed_sources
            and "Per-report source metadata is authoritative"
            not in str(rights.get("attribution") or "")
        )
    ):
        raise ValueError("ASR manifest lacks complete public-source rights provenance")


def _validate_stream(
    stream: Mapping[str, Any], *, label: str, allow_empty: bool
) -> None:
    layers = stream.get("layers")
    cells = stream.get("cells")
    if not isinstance(layers, list) or not isinstance(cells, list):
        raise ValueError(f"{label} needs layer and cell arrays")
    if len(layers) != len(cells):
        raise ValueError(f"{label} layer/cell mismatch")
    if not cells:
        if allow_empty:
            return
        raise ValueError(f"{label} matrix is empty")
    widths = {len(row) for row in cells if isinstance(row, list)}
    if len(widths) != 1 or 0 in widths:
        raise ValueError(f"{label} matrix is empty or ragged")
    for row in cells:
        for cell in row:
            if not isinstance(cell.get("top_tokens"), list) or not cell["top_tokens"]:
                raise ValueError(f"{label} cell has no cached candidates")


def _validate_exact_realized_rank(
    candidate: Any,
    *,
    label: str,
    expected_id: Any,
    require_score: bool,
) -> None:
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{label} has no exact realized-token provenance")
    required = {
        "id",
        "text",
        "rank",
        "rank_denominator",
        "rank_space",
        "rank_tie_policy",
        "score_kind",
    }
    if require_score:
        required.add("score")
    missing = sorted(required - candidate.keys())
    if missing:
        raise ValueError(
            f"{label} realized-token provenance lacks {', '.join(missing)}"
        )
    if candidate["id"] != expected_id:
        raise ValueError(f"{label} realized-token ID does not match the output token")
    try:
        rank = int(candidate["rank"])
        denominator = int(candidate["rank_denominator"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} realized-token rank is invalid") from error
    if rank < 1 or denominator < rank:
        raise ValueError(f"{label} realized-token rank is outside its rank space")
    if not str(candidate["rank_space"]).strip():
        raise ValueError(f"{label} realized-token rank space is empty")
    if candidate["rank_tie_policy"] != "1_plus_count_strictly_greater":
        raise ValueError(f"{label} realized-token tie policy is unsupported")


def _validate_speech_generation_diagnostics(report: Mapping[str, Any]) -> None:
    diagnostics = (
        report.get("payload", {}).get("metadata", {}).get("generation_diagnostics")
    )
    if not isinstance(diagnostics, Mapping):
        raise ValueError("speech report has no generation-termination diagnostics")
    values: dict[str, int] = {}
    for field in (
        "generated_steps",
        "max_new_tokens",
        "text_tokens",
        "audio_frames",
    ):
        value = diagnostics.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"speech generation diagnostic {field} is invalid")
        values[field] = value
    generated_steps = values["generated_steps"]
    max_new_tokens = values["max_new_tokens"]
    if (
        generated_steps < 1
        or max_new_tokens < generated_steps
        or values["text_tokens"] < 0
        or values["audio_frames"] < 0
    ):
        raise ValueError("speech generation diagnostic counts are out of bounds")
    audio_eos_seen = diagnostics.get("audio_eos_seen")
    budget_exhausted = diagnostics.get("budget_exhausted")
    if not isinstance(audio_eos_seen, bool) or not isinstance(budget_exhausted, bool):
        raise ValueError("speech generation diagnostic flags are invalid")
    eos_step = 1 if audio_eos_seen else 0
    if generated_steps != values["text_tokens"] + values["audio_frames"] + eos_step:
        raise ValueError("speech generation diagnostic step accounting is invalid")
    termination_reason = diagnostics.get("termination_reason")
    natural = (
        termination_reason == "audio_eos" and audio_eos_seen and not budget_exhausted
    )
    capped = (
        termination_reason == "budget_exhausted"
        and not audio_eos_seen
        and budget_exhausted
        and generated_steps == max_new_tokens
    )
    if not natural and not capped:
        raise ValueError("speech generation termination state is inconsistent")


def _validate_asr_phone_signatures(report: Mapping[str, Any]) -> None:
    payload = report["payload"]
    metadata = payload.get("metadata", {}).get("phone_signature")
    if not isinstance(metadata, Mapping) or metadata.get("available") is not True:
        raise ValueError("ASR report has no public phone-signature metadata")
    pooling = payload.get("encoder", {}).get("pooling")
    expected_pooling = {
        "requested_window_seconds": 0.1,
        "requested_overlap_seconds": 0.02,
        "effective_window_seconds": 0.1,
        "effective_overlap_seconds": 0.02,
        "effective_hop_seconds": 0.08,
    }
    if not isinstance(pooling, Mapping):
        raise ValueError("ASR encoder pooling geometry is missing")
    try:
        pooling_matches = all(
            math.isclose(float(pooling.get(field)), expected, rel_tol=0, abs_tol=1e-9)
            for field, expected in expected_pooling.items()
        )
    except (TypeError, ValueError):
        pooling_matches = False
    if (
        not pooling_matches
        or pooling.get("adaptive_for_max_bins") is not False
        or pooling.get("max_time_bins") != 100
    ):
        raise ValueError("ASR encoder pooling must be exact 100/20/80 ms geometry")
    labels_value = metadata.get("phone_inventory")
    if not isinstance(labels_value, list) or any(
        not isinstance(label, str) or not label for label in labels_value
    ):
        raise ValueError("ASR report has an invalid phone inventory")
    labels = set(labels_value)
    denominator = metadata.get("phone_inventory_size")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or tuple(labels_value) != PUBLIC_PHONE_INVENTORY
        or denominator != len(PUBLIC_PHONE_INVENTORY)
        or len(labels) != denominator
    ):
        raise ValueError("ASR phone inventory size is inconsistent")
    if (
        set(metadata) != PHONE_SIGNATURE_METADATA_FIELDS
        or metadata.get("schema_version") != 1
        or metadata.get("score_kind") != "phone_prototype_cosine_similarity"
        or metadata.get("signature_top_k") != 100
        or metadata.get("display_unit") != "pooled_encoder_window"
        or metadata.get("method") != "nearest_frozen_top_k_j_signature_phone_prototype"
        or metadata.get("training_unit") != "aligned_native_20_ms_phone_midpoint_state"
        or not str(metadata.get("interpretation") or "").strip()
        or not math.isclose(
            float(metadata.get("effective_display_window_seconds") or 0),
            0.1,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(metadata.get("effective_display_hop_seconds") or 0),
            0.08,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or metadata.get("silence_or_unknown_class_available") is not False
        or metadata.get("prototype_fit_split") != "train"
        or metadata.get("prototype_fit_rows") != 3400
        or metadata.get("prototype_fit_opened_eval_splits") is not False
        or metadata.get("prototype_lens_examples") != 20
    ):
        raise ValueError("ASR phone-signature semantics are invalid")
    for row in payload.get("encoder", {}).get("cells", []):
        if len(row) > 100:
            raise ValueError("ASR encoder matrix exceeds the 100-bin release limit")
        for cell in row:
            candidates = cell.get("phone_signatures")
            if not isinstance(candidates, list) or len(candidates) != 5:
                raise ValueError("ASR encoder cell has no usable phone signature")
            previous: float | None = None
            seen: set[str] = set()
            for candidate in candidates:
                if (
                    not isinstance(candidate, Mapping)
                    or set(candidate) != PHONE_SIGNATURE_FIELDS
                ):
                    raise ValueError("ASR phone candidate has unapproved fields")
                phone = candidate.get("phone")
                rank = candidate.get("rank")
                similarity = candidate.get("similarity")
                if (
                    phone not in labels
                    or phone in seen
                    or isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or not 1 <= rank <= denominator
                    or isinstance(similarity, bool)
                    or not isinstance(similarity, (int, float))
                    or not math.isfinite(float(similarity))
                    or not -1 <= float(similarity) <= 1
                    or (previous is not None and float(similarity) > previous + 1e-8)
                ):
                    raise ValueError("ASR phone candidate is invalid")
                seen.add(phone)
                previous = float(similarity)
            if candidates[0]["rank"] != 1:
                raise ValueError("ASR top phone candidate does not rank first")
            for candidate in candidates:
                expected_rank = 1 + sum(
                    float(other["similarity"]) > float(candidate["similarity"])
                    for other in candidates
                )
                if candidate["rank"] != expected_rank:
                    raise ValueError("ASR phone candidate tie rank is inconsistent")


def _aligned_transcription_token(
    tokens: list[Mapping[str, Any]], time_window: Mapping[str, Any]
) -> tuple[int, Mapping[str, Any]]:
    """Match the backend's overlap-first encoder/token synchronization."""
    try:
        window_start = float(time_window["start_seconds"])
        window_end = float(time_window["end_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ASR encoder cell has no usable time window") from error
    if not math.isfinite(window_start) or not math.isfinite(window_end):
        raise ValueError("ASR encoder cell has a non-finite time window")
    if window_end < window_start:
        raise ValueError("ASR encoder cell has a reversed time window")
    midpoint = (window_start + window_end) / 2
    choices: list[tuple[tuple[float, float, int], int, Mapping[str, Any]]] = []
    for index, token in enumerate(tokens):
        start = token.get("start_seconds")
        end = token.get("end_seconds")
        if start is None or end is None:
            continue
        try:
            token_start = float(start)
            token_end = float(end)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(token_start)
            or not math.isfinite(token_end)
            or token_end < token_start
        ):
            continue
        overlap = max(
            0.0,
            min(window_end, token_end) - max(window_start, token_start),
        )
        midpoint_distance = abs(midpoint - (token_start + token_end) / 2)
        choices.append(((-overlap, midpoint_distance, index), index, token))
    if not choices:
        raise ValueError("ASR encoder cells cannot be aligned to untimed output tokens")
    _, best_index, best_token = min(choices, key=lambda choice: choice[0])
    return best_index, best_token


def _validate_encoder_alignment_provenance(
    alignment: Any,
    *,
    time_window: Mapping[str, Any],
    token: Mapping[str, Any],
) -> None:
    if not isinstance(alignment, Mapping):
        raise ValueError("ASR encoder cell has no realized-token alignment provenance")
    required = {
        "match",
        "window_midpoint_seconds",
        "token_start_seconds",
        "token_end_seconds",
        "overlap_seconds",
        "overlap_fraction_of_window",
    }
    if required - alignment.keys():
        raise ValueError("ASR encoder cell has incomplete alignment provenance")
    window_start = float(time_window["start_seconds"])
    window_end = float(time_window["end_seconds"])
    token_start = float(token["start_seconds"])
    token_end = float(token["end_seconds"])
    overlap = max(
        0.0,
        min(window_end, token_end) - max(window_start, token_start),
    )
    duration = max(0.0, window_end - window_start)
    expected = {
        "window_midpoint_seconds": (window_start + window_end) / 2,
        "token_start_seconds": token_start,
        "token_end_seconds": token_end,
        "overlap_seconds": overlap,
        "overlap_fraction_of_window": overlap / duration if duration > 0 else 0.0,
    }
    if alignment.get("match") != ("overlapping" if overlap > 0 else "nearest"):
        raise ValueError("ASR encoder cell has an invalid alignment match kind")
    for field, value in expected.items():
        try:
            recorded = float(alignment[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"ASR encoder alignment {field} is invalid") from error
        if not math.isclose(recorded, value, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"ASR encoder alignment {field} does not match timing")


def _matches_replay_number(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
        return math.isfinite(numeric) and math.isclose(
            numeric, expected, rel_tol=0.0, abs_tol=1e-8
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _is_replay_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_replay_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_recorded_replay_source(source: Any) -> None:
    required_source = {
        "audio_url",
        "sha256",
        "source_url",
        "license",
        "license_url",
        "attribution",
        "modification_notice",
        "rights_status",
    }
    if (
        not isinstance(source, Mapping)
        or required_source - source.keys()
        or source.get("rights_status") != ASR_REPLAY_RIGHTS_STATUS
        or source.get("source_url") != ASR_REPLAY_SOURCE_URL
        or source.get("license") != ASR_REPLAY_LICENSE
        or source.get("license_url") != ASR_REPLAY_LICENSE_URL
        or source.get("attribution") != ASR_REPLAY_ATTRIBUTION
        or source.get("modification_notice") != ASR_REPLAY_MODIFICATION_NOTICE
    ):
        raise ValueError(
            "Laurel/Yanny report lacks the exact report-local source, license, "
            "attribution, and change notice"
        )


def _validate_replay_condition_contract(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
    encoder_layers: list[Any],
    audio_duration_seconds: float,
) -> None:
    """Validate the exact public metadata schema for one recorded rerun."""

    expected = ASR_REPLAY_EXPECTED[condition_id]
    label = condition.get("label")
    generated = condition.get("generated")
    evidence = condition.get("evidence")
    method = condition.get("method")
    layers = condition.get("layers")
    schedule = condition.get("schedule")
    token_ids = generated.get("token_ids") if isinstance(generated, Mapping) else None

    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"recorded {condition_id} replay has an invalid label")
    if not isinstance(generated, Mapping) or set(generated) != (
        ASR_REPLAY_GENERATED_FIELDS
    ):
        raise ValueError(f"recorded {condition_id} generated schema changed")
    if (
        not isinstance(generated.get("text"), str)
        or generated.get("text") != expected["text"]
        or not isinstance(token_ids, list)
        or not all(_is_replay_integer(token_id) for token_id in token_ids)
        or token_ids != expected["token_ids"]
        or type(generated.get("target_match")) is not bool
        or generated.get("target_match") is not expected["target_match"]
    ):
        raise ValueError(f"recorded {condition_id} generated value changed")

    if not isinstance(evidence, Mapping) or set(evidence) != (
        ASR_REPLAY_EVIDENCE_FIELDS
    ):
        raise ValueError(f"recorded {condition_id} evidence schema changed")
    if evidence.get("tier") != expected["evidence_tier"] or not all(
        isinstance(evidence.get(field), str) and evidence[field].strip()
        for field in ASR_REPLAY_EVIDENCE_FIELDS
    ):
        raise ValueError(f"recorded {condition_id} evidence value changed")

    if not isinstance(method, Mapping) or set(method) != ASR_REPLAY_METHOD_FIELDS:
        raise ValueError(f"recorded {condition_id} method schema changed")
    if not all(
        isinstance(method.get(field), str) and method[field].strip()
        for field in ASR_REPLAY_METHOD_FIELDS
    ):
        raise ValueError(f"recorded {condition_id} method value changed")

    if (
        not isinstance(layers, list)
        or len(layers) != expected["layer_count"]
        or not all(_is_replay_integer(layer) and layer >= 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError(f"recorded {condition_id} layer list changed")
    expected_layers = [] if condition_id == "baseline" else encoder_layers
    if layers != expected_layers:
        raise ValueError(f"recorded {condition_id} layers do not match the encoder")

    if not isinstance(schedule, list) or len(schedule) != expected["schedule_count"]:
        raise ValueError(f"recorded {condition_id} schedule count changed")
    previous_end_seconds = 0.0
    previous_end_position = 0
    for index, item in enumerate(schedule):
        if not isinstance(item, Mapping) or set(item) != ASR_REPLAY_SCHEDULE_FIELDS:
            raise ValueError(
                f"recorded {condition_id} schedule item {index} schema changed"
            )
        phone = item.get("phone")
        start_seconds = item.get("start_seconds")
        end_seconds = item.get("end_seconds")
        start_position = item.get("start_position")
        end_position = item.get("end_position")
        if (
            not isinstance(phone, str)
            or not 1 <= len(phone) <= 3
            or phone != phone.strip().upper()
            or not phone.isalpha()
            or not _is_replay_number(start_seconds)
            or not _is_replay_number(end_seconds)
            or not _is_replay_integer(start_position)
            or not _is_replay_integer(end_position)
        ):
            raise ValueError(
                f"recorded {condition_id} schedule item {index} has invalid types"
            )
        start_seconds = float(start_seconds)
        end_seconds = float(end_seconds)
        if (
            start_seconds < 0.0
            or start_seconds >= end_seconds
            or end_seconds > audio_duration_seconds + 1e-8
            or start_position < 0
            or start_position >= end_position
            or start_seconds < previous_end_seconds - 1e-8
            or start_position < previous_end_position
        ):
            raise ValueError(
                f"recorded {condition_id} schedule item {index} has invalid bounds or order"
            )
        if not (
            math.isclose(
                start_seconds, start_position * 0.02, rel_tol=0.0, abs_tol=1e-8
            )
            and math.isclose(
                end_seconds, end_position * 0.02, rel_tol=0.0, abs_tol=1e-8
            )
        ):
            raise ValueError(
                f"recorded {condition_id} schedule item {index} time/position mismatch"
            )
        previous_end_seconds = end_seconds
        previous_end_position = end_position


def _validate_replay_safe(
    value: Any, *, path: str = "$.recorded_intervention_replay"
) -> None:
    """Reject recorder-only tensors, media, and private references."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in ASR_REPLAY_FORBIDDEN_KEYS or normalized.endswith(
                "_analysis_id"
            ):
                raise ValueError(f"forbidden recorded replay field {path}.{key}")
            _validate_replay_safe(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_replay_safe(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.strip().lower().replace("\\", "/")
        if (
            "/users/" in lowered
            or "artifacts/private/" in lowered
            or lowered.startswith("data:audio/")
            or lowered.endswith(
                (".pt", ".pth", ".npy", ".npz", ".wav", ".flac", ".mp3")
            )
        ):
            raise ValueError(f"private artifact or audio in recorded replay at {path}")


def _validate_replay_candidate_bounds(
    payload: Mapping[str, Any], *, condition_id: str
) -> None:
    tokens = payload.get("transcription", {}).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"recorded {condition_id} replay has no HEAD sequence")
    for token in tokens:
        candidates = token.get("top_tokens") if isinstance(token, Mapping) else None
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
            raise ValueError(f"recorded {condition_id} HEAD candidates are not bounded")
    for stream_name in ("encoder", "decoder"):
        cells = payload.get(stream_name, {}).get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError(
                f"recorded {condition_id} {stream_name} matrix is incomplete"
            )
        for row in cells:
            if not isinstance(row, list) or not row:
                raise ValueError(
                    f"recorded {condition_id} {stream_name} matrix is ragged"
                )
            for cell in row:
                candidates = (
                    cell.get("top_tokens") if isinstance(cell, Mapping) else None
                )
                if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
                    raise ValueError(
                        f"recorded {condition_id} {stream_name} candidates are not bounded"
                    )
                phones = cell.get("phone_signatures")
                if phones is not None and (
                    not isinstance(phones, list) or not 1 <= len(phones) <= 5
                ):
                    raise ValueError(
                        f"recorded {condition_id} phone candidates are not bounded"
                    )


def _replay_encoder_geometry(payload: Mapping[str, Any]) -> dict[str, Any]:
    encoder = payload.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("recorded replay has no encoder matrix")
    cells = encoder.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("recorded replay encoder matrix is empty")
    coordinates: list[list[dict[str, Any]]] = []
    widths: list[int] = []
    for row in cells:
        if not isinstance(row, list):
            raise ValueError("recorded replay encoder matrix is malformed")
        widths.append(len(row))
        coordinates.append(
            [
                {
                    "position_index": cell.get("position_index"),
                    "time_window": cell.get("time_window"),
                }
                for cell in row
            ]
        )
    return {
        "layers": encoder.get("layers"),
        "pooling": encoder.get("pooling"),
        "positions": encoder.get("positions"),
        "time_bins": encoder.get("time_bins"),
        "widths": widths,
        "coordinates": coordinates,
    }


def _compose_recorded_replay_report(
    report: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    analysis = condition.get("analysis")
    payload = dict(report["payload"])
    if isinstance(analysis, Mapping):
        payload.update(dict(analysis))
    return {**report, "payload": payload}


def _validate_recorded_intervention_replay(report: Mapping[str, Any]) -> None:
    replay = report.get("recorded_intervention_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("Laurel/Yanny ASR report has no recorded replay")
    if set(replay) != ASR_REPLAY_ROOT_FIELDS or (
        replay.get("schema_id") != ASR_REPLAY_SCHEMA_ID
        or replay.get("schema_version") != 1
        or replay.get("mode") != ASR_REPLAY_MODE
        or replay.get("default_condition") != "baseline"
    ):
        raise ValueError("recorded ASR replay has an invalid schema")
    _validate_recorded_replay_source(report.get("source"))

    conditions = replay.get("conditions")
    if not isinstance(conditions, list) or not all(
        isinstance(condition, Mapping) for condition in conditions
    ):
        raise ValueError("recorded ASR replay has no condition list")
    if [condition.get("id") for condition in conditions] != list(ASR_REPLAY_CONDITIONS):
        raise ValueError("recorded ASR replay must contain baseline, Yanny, and Laurel")
    _validate_replay_safe(replay)

    payload = report.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Laurel/Yanny ASR report has no baseline payload")
    encoder_layers = payload.get("encoder", {}).get("layers")
    audio_duration = payload.get("audio", {}).get("duration_seconds")
    if (
        not isinstance(encoder_layers, list)
        or not _is_replay_number(audio_duration)
        or float(audio_duration) <= 0.0
    ):
        raise ValueError("recorded ASR replay has invalid encoder or audio bounds")
    baseline_geometry = _replay_encoder_geometry(payload)
    for condition in conditions:
        condition_id = str(condition["id"])
        expected = ASR_REPLAY_EXPECTED[condition_id]
        expected_fields = set(ASR_REPLAY_CONDITION_FIELDS)
        if condition_id != "baseline":
            expected_fields.add("analysis")
        if set(condition) != expected_fields:
            raise ValueError(
                f"recorded {condition_id} replay has unexpected or missing fields"
            )
        if (
            condition.get("recorded") is not True
            or condition.get("interpolated") is not False
            or not _matches_replay_number(
                condition.get("budget_fraction"), expected["budget_fraction"]
            )
            or not _matches_replay_number(
                condition.get("coefficient_scale"), expected["coefficient_scale"]
            )
        ):
            raise ValueError(f"recorded {condition_id} replay changed")
        _validate_replay_condition_contract(
            condition,
            condition_id=condition_id,
            encoder_layers=encoder_layers,
            audio_duration_seconds=float(audio_duration),
        )
        if condition_id == "baseline" and "analysis" in condition:
            raise ValueError("recorded baseline must reuse the root report payload")
        if condition_id != "baseline" and set(condition.get("analysis", {})) != (
            ASR_REPLAY_ANALYSIS_FIELDS
        ):
            raise ValueError(f"recorded {condition_id} analysis is incomplete")

        composed = _compose_recorded_replay_report(report, condition)
        payload = composed["payload"]
        tokens = payload.get("transcription", {}).get("tokens")
        if not isinstance(tokens, list) or [
            token.get("id") for token in tokens
        ] != list(expected["token_ids"]):
            raise ValueError(f"recorded {condition_id} token sequence changed")
        if (
            str(payload.get("transcription", {}).get("text") or "").strip()
            != expected["text"]
        ):
            raise ValueError(f"recorded {condition_id} transcript changed")
        if _replay_encoder_geometry(payload) != baseline_geometry:
            raise ValueError(
                f"recorded {condition_id} encoder layers or geometry changed"
            )
        _validate_asr_or_speech(
            composed,
            family="asr",
            expected_rights_status=ASR_REPLAY_RIGHTS_STATUS,
        )
        _validate_replay_candidate_bounds(payload, condition_id=condition_id)


def _validate_asr_or_speech(
    report: Mapping[str, Any],
    *,
    family: str,
    expected_rights_status: str = "cleared_with_attribution",
) -> None:
    if family == "speech":
        _validate_speech_generation_diagnostics(report)
    payload = report["payload"]
    if family == "asr":
        _validate_asr_phone_signatures(report)
    tokens = payload.get("transcription", {}).get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{family} report has no output tokens")
    _validate_stream(payload["decoder"], label=f"{family} decoder", allow_empty=False)
    if len(payload["decoder"]["cells"][0]) != len(tokens):
        raise ValueError(f"{family} decoder/token width mismatch")
    if family in {"asr", "speech"}:
        for position, token in enumerate(tokens):
            _validate_exact_realized_rank(
                token,
                label=f"{family} HEAD position {position}",
                expected_id=token.get("id"),
                require_score=False,
            )
        for layer_index, row in enumerate(payload["decoder"]["cells"]):
            for position, cell in enumerate(row):
                _validate_exact_realized_rank(
                    cell.get("realized_token"),
                    label=(
                        f"{family} decoder layer {layer_index}, position {position}"
                    ),
                    expected_id=tokens[position].get("id"),
                    require_score=True,
                )
    _validate_stream(
        payload["encoder"], label=f"{family} encoder", allow_empty=family == "speech"
    )
    if family == "asr":
        alignment_metadata = payload["encoder"].get("realized_token_alignment")
        if (
            not isinstance(alignment_metadata, Mapping)
            or alignment_metadata.get("method") != "maximum_token_interval_overlap"
        ):
            raise ValueError("ASR encoder stream has no alignment-method provenance")
        for layer_index, row in enumerate(payload["encoder"]["cells"]):
            for position, cell in enumerate(row):
                token_index, token = _aligned_transcription_token(
                    tokens, cell.get("time_window") or {}
                )
                if cell.get("realized_token_position") != token_index:
                    raise ValueError(
                        "ASR encoder realized-token position does not match "
                        "overlap-first output-token synchronization"
                    )
                _validate_encoder_alignment_provenance(
                    cell.get("realized_token_alignment"),
                    time_window=cell.get("time_window") or {},
                    token=token,
                )
                _validate_exact_realized_rank(
                    cell.get("realized_token"),
                    label=(
                        f"asr encoder layer {layer_index}, position {position} "
                        f"(aligned output token {token_index})"
                    ),
                    expected_id=token.get("id"),
                    require_score=True,
                )
    preview = payload.get("audio", {}).get("waveform_preview", {})
    values = preview.get("values")
    if not isinstance(values, list) or not 0 < len(values) <= 1024:
        raise ValueError(f"{family} input waveform preview is invalid")
    if report.get("source", {}).get("rights_status") != expected_rights_status:
        raise ValueError(
            f"{family} input audio does not have the expected rights status"
        )


def _validate_tts(report: Mapping[str, Any]) -> None:
    payload = report["payload"]
    codes = payload.get("output", {}).get("speech_codes")
    head = payload.get("output", {}).get("speech_head_candidates", {})
    fitted = payload.get("fitted_speech_code_jlens", {})
    traces = payload.get("traces_by_position")
    if not isinstance(codes, list) or not codes:
        raise ValueError("TTS report has no speech-code positions")
    width = len(codes)
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("TTS report has no saved model metadata")
    generation = model.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("TTS report has no saved generation settings")
    generation_cap = generation.get("max_speech_tokens")
    if (
        isinstance(generation_cap, bool)
        or not isinstance(generation_cap, int)
        or generation_cap <= 0
    ):
        raise ValueError("TTS report has an invalid speech-code generation cap")
    if width >= generation_cap:
        raise ValueError(
            "TTS report reached its saved speech-code generation cap and may "
            "be truncated"
        )
    if len(head.get("positions", [])) != width:
        raise ValueError("TTS HEAD width mismatch")
    rows = fitted.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("TTS fitted rows are missing")
    if any(len(row.get("positions", [])) != width for row in rows):
        raise ValueError("TTS fitted matrix is ragged")
    if not isinstance(traces, Mapping) or set(traces) != {
        str(index) for index in range(width)
    }:
        raise ValueError("TTS trace cache does not cover every speech position")
    if payload.get("generated_audio_included") is not False:
        raise ValueError("TTS report must explicitly exclude generated audio")


def _validate_filter_cache(
    site_root: Path,
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    reference = entry.get("character_length_filter_cache")
    if not isinstance(reference, Mapping):
        raise ValueError("ASR report is missing its lazy character-filter cache")
    path = _site_path(site_root, str(reference["url"]))
    if not path.is_file() or _sha256(path) != reference["sha256"]:
        raise ValueError(f"ASR filter cache hash mismatch: {path}")
    if path.stat().st_size != int(reference.get("bytes", -1)):
        raise ValueError(f"ASR filter cache byte count mismatch: {path}")
    report_reference = report.get("cache_policy", {}).get(
        "character_length_filter_cache"
    )
    if report_reference != reference:
        raise ValueError("ASR report and manifest disagree on the filter cache")
    cache = _load(path)
    if cache.get("example_id") != report.get("example_id"):
        raise ValueError("ASR filter cache/report ID mismatch")
    denominators_by_length = (
        report.get("payload", {})
        .get("metadata", {})
        .get("display_vocabulary", {})
        .get("maximum_decoded_character_length_counts", {})
    )
    if not isinstance(denominators_by_length, Mapping) or not denominators_by_length:
        raise ValueError("ASR report has no character-filter denominators")
    for stream_name in ("encoder", "decoder"):
        filtered = cache["streams"][stream_name]
        base = report["payload"][stream_name]
        if any(layer not in base["layers"] for layer in filtered["layers"]):
            raise ValueError(f"ASR {stream_name} filter layer mismatch")
        width = len(base["cells"][0])
        if any(len(row) != width for row in filtered["cells"]):
            raise ValueError(f"ASR {stream_name} filter width mismatch")
        for layer, row in zip(filtered["layers"], filtered["cells"], strict=True):
            report_layer_index = base["layers"].index(layer)
            for position, cell in enumerate(row):
                if not cell.get("top_tokens_by_length"):
                    raise ValueError(f"ASR {stream_name} filter cell is empty")
                realized_rank_by_length = cell.get("realized_rank_by_max_length")
                if (
                    not isinstance(realized_rank_by_length, Mapping)
                    or not realized_rank_by_length
                ):
                    raise ValueError(
                        f"ASR {stream_name} filter cell has no exact realized ranks"
                    )
                base_cell = base["cells"][report_layer_index][position]
                target_filter = base_cell.get("realized_token", {}).get(
                    "vocabulary_filter", {}
                )
                target_length = target_filter.get("decoded_character_length")
                target_eligible = target_filter.get("display_lexical_eligible")
                if not isinstance(target_eligible, bool):
                    raise ValueError(
                        f"ASR {stream_name} realized token has no lexical eligibility"
                    )
                if set(realized_rank_by_length) != set(denominators_by_length):
                    raise ValueError(
                        f"ASR {stream_name} filtered ranks do not cover every limit"
                    )
                for limit, rank in realized_rank_by_length.items():
                    try:
                        numeric_limit = int(limit)
                        denominator = int(denominators_by_length[str(limit)])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"ASR {stream_name} filter limit is invalid"
                        ) from error
                    if rank is None:
                        if (
                            target_eligible
                            and target_length is not None
                            and int(target_length) <= numeric_limit
                        ):
                            raise ValueError(
                                f"ASR {stream_name} eligible realized rank is missing"
                            )
                        continue
                    try:
                        numeric_rank = int(rank)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"ASR {stream_name} filtered realized rank is invalid"
                        ) from error
                    if numeric_rank != rank or not 1 <= numeric_rank <= denominator:
                        raise ValueError(
                            f"ASR {stream_name} filtered realized rank is out of bounds"
                        )
                    if not target_eligible or (
                        target_length is not None and int(target_length) > numeric_limit
                    ):
                        raise ValueError(
                            f"ASR {stream_name} excluded realized token has a rank"
                        )
    _validate_safe(cache)


def _validate_audio_reference(
    site_root: Path,
    *,
    family: str,
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> Path:
    entry_url = str(entry.get("audio_url") or "")
    source = report.get("source")
    if not isinstance(source, Mapping) or source.get("audio_url") != entry_url:
        raise ValueError(f"{family} report and manifest audio URLs disagree")
    audio_path = _site_path(site_root, entry_url)
    if not audio_path.is_file():
        raise ValueError(f"missing cleared input audio: {audio_path}")
    expected_hash = source.get("sha256")
    if not isinstance(expected_hash, str) or _sha256(audio_path) != expected_hash:
        raise ValueError(f"{family} input audio hash mismatch: {audio_path}")
    return audio_path


def _validate_site_manifest_integrity(
    site_root: Path, *, counts: Mapping[str, int]
) -> None:
    manifest = _load(site_root / "site-manifest.json")
    _validate_safe(manifest)
    if manifest.get("report_counts") != dict(counts):
        raise ValueError("site manifest report counts do not match family manifests")
    recorded_hashes = manifest.get("sha256")
    if not isinstance(recorded_hashes, Mapping) or not recorded_hashes:
        raise ValueError("site manifest has no asset hashes")
    required = {
        "assets/explorer.js",
        "assets/explorer.css",
        "assets/steering.js",
        "assets/steering.css",
        "data/phone-steering-results.json",
        "steering/index.html",
        *(f"explorer/data/{family}/manifest.json" for family in FAMILIES),
    }
    if not required.issubset(recorded_hashes):
        raise ValueError("site manifest does not hash every explorer contract file")
    for relative_path, expected_hash in recorded_hashes.items():
        path = (site_root / str(relative_path)).resolve()
        if not path.is_relative_to(site_root) or not path.is_file():
            raise ValueError(f"site manifest hash path is missing or unsafe: {path}")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"site manifest hash mismatch: {path}")


def _validate_media_union(site_root: Path, referenced_media: set[Path]) -> None:
    """Require the union of family references, not identical family audio sets."""

    media = {
        path.resolve()
        for path in site_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}
    }
    if media != {path.resolve() for path in referenced_media}:
        raise ValueError(
            "the static site media set does not match the cleared manifest inputs"
        )


def _require_page(site_root: Path, relative_path: str, *, label: str) -> str:
    path = site_root / relative_path
    if not path.is_file():
        raise ValueError(f"missing {label} page: {relative_path}")
    html = path.read_text(encoding="utf-8")
    if 'name="robots" content="noindex,nofollow"' not in html:
        raise ValueError(f"{label} page is not marked noindex")
    return html


def _require_markers(html: str, markers: tuple[str, ...], *, label: str) -> None:
    for marker in markers:
        if marker not in html:
            raise ValueError(f"{label} page is missing {marker!r}")


def _validate_route_contract(site_root: Path) -> None:
    for relative_path in RETIRED_PUBLIC_TTS_PATHS:
        if (site_root / relative_path).exists():
            raise ValueError(f"retired public TTS path still exists: {relative_path}")
    detailed_pages = {
        "asr": (
            "index.html",
            "./assets/explorer.js",
            "./explorer/data/asr/manifest.json",
            (
                'href="./"',
                'href="./speech/"',
                'href="./steering/"',
            ),
        ),
        "speech": (
            "speech/index.html",
            "../assets/explorer.js",
            "../explorer/data/speech/manifest.json",
            (
                'href="../"',
                'href="./"',
                'href="../steering/"',
            ),
        ),
    }
    findings_pages = {
        "asr": (
            "findings/index.html",
            "../assets/app.js",
            "../data/reports.json",
        ),
        "speech": (
            "findings/speech/index.html",
            "../../assets/app.js",
            "../../data/reports.json",
        ),
    }
    alias_pages = {
        family: (
            f"explorer/{family}/index.html",
            "../../assets/explorer.js",
            f"../data/{family}/manifest.json",
        )
        for family in FAMILIES
    }

    for family, (path, script, manifest, nav_links) in detailed_pages.items():
        label = f"canonical {family} detailed explorer"
        html = _require_page(site_root, path, label=label)
        _require_markers(
            html,
            (
                'class="detailed-explorer"',
                f'data-family="{family}"',
                f'data-manifest-url="{manifest}"',
                f'src="{script}?v={EXPLORER_ASSET_VERSION}"',
                (
                    f'href="{script.removesuffix("explorer.js")}explorer.css'
                    f'?v={EXPLORER_ASSET_VERSION}"'
                ),
                f'<link rel="canonical" href="{PUBLIC_BASE}{CANONICAL_DETAILED_ROUTES[family].removeprefix(SITE_PREFIX)}">',
                'aria-label="Audio Jacobian Lens home"',
                'class="site-nav" aria-label="Model explorers"',
                '<span class="static-badge">PREVIEW · STATIC REPLAY</span>',
                *nav_links,
            ),
            label=label,
        )
        if "assets/app.js" in html:
            raise ValueError(f"{label} uses the findings renderer")
        if ">TTS</a>" in html:
            raise ValueError(f"{label} exposes the retired public TTS page")
        if 'class="summary-link"' in html or ">Experiment findings</a>" in html:
            raise ValueError(f"{label} retains the removed findings header link")

    for family, (path, script, data_url) in findings_pages.items():
        label = f"{family} findings"
        html = _require_page(site_root, path, label=label)
        _require_markers(
            html,
            (
                f'data-family="{family}"',
                f'data-data-url="{data_url}"',
                f'src="{script}"',
            ),
            label=label,
        )
        if 'class="detailed-explorer"' in html or "assets/explorer.js" in html:
            raise ValueError(f"{label} uses the detailed-explorer renderer")
        if ">TTS</a>" in html:
            raise ValueError(f"{label} exposes the retired public TTS page")

    canonical_alias_nav = (
        'href="../../"',
        'href="../../speech/"',
        'href="../../steering/"',
    )
    for family, (path, script, manifest) in alias_pages.items():
        label = f"legacy {family} explorer alias"
        html = _require_page(site_root, path, label=label)
        canonical_suffix = CANONICAL_DETAILED_ROUTES[family].removeprefix(SITE_PREFIX)
        _require_markers(
            html,
            (
                'class="detailed-explorer"',
                f'data-family="{family}"',
                f'data-manifest-url="{manifest}"',
                f'src="{script}?v={EXPLORER_ASSET_VERSION}"',
                (
                    f'href="{script.removesuffix("explorer.js")}explorer.css'
                    f'?v={EXPLORER_ASSET_VERSION}"'
                ),
                f'<link rel="canonical" href="{PUBLIC_BASE}{canonical_suffix}">',
                'aria-label="Audio Jacobian Lens home"',
                'class="site-nav" aria-label="Model explorers"',
                '<span class="static-badge">PREVIEW · STATIC REPLAY</span>',
                *canonical_alias_nav,
            ),
            label=label,
        )
        if "assets/app.js" in html:
            raise ValueError(f"{label} uses the findings renderer")
        if ">TTS</a>" in html:
            raise ValueError(f"{label} exposes the retired public TTS page")
        if 'class="summary-link"' in html or ">Experiment findings</a>" in html:
            raise ValueError(f"{label} retains the removed findings header link")

    steering_html = _require_page(
        site_root, "steering/index.html", label="recorded phone steering replay"
    )
    _require_markers(
        steering_html,
        (
            'data-results-url="../data/phone-steering-results.json"',
            f'src="../assets/steering.js?v={STEERING_ASSET_VERSION}"',
            f'href="../assets/steering.css?v={STEERING_ASSET_VERSION}"',
            f'<link rel="canonical" href="{PUBLIC_BASE}steering/"',
            'aria-label="Audio Jacobian Lens home"',
            'class="site-nav" aria-label="Model explorers"',
            '<span class="static-badge">PREVIEW · STATIC REPLAY</span>',
            'data-target="yanny"',
            'data-target="laurel"',
            'id="checkpoint-range" type="range"',
            'href="../"',
            'href="../speech/"',
        ),
        label="recorded phone steering replay",
    )
    if (
        "assets/app.js" in steering_html
        or "assets/explorer.js" in steering_html
        or "<audio" in steering_html
        or ">TTS</a>" in steering_html
    ):
        raise ValueError("recorded phone steering replay has an unsafe renderer")

    site_manifest = _load(site_root / "site-manifest.json")
    expected_routes = {
        "detailed_cached_explorers": list(CANONICAL_DETAILED_ROUTES.values()),
        "findings": list(FINDINGS_ROUTES.values()),
        "legacy_explorer_aliases": list(LEGACY_EXPLORER_ROUTES.values()),
        "recorded_interventions": [STEERING_ROUTE],
    }
    routes = site_manifest.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("site manifest has no route map")
    for name, expected in expected_routes.items():
        if routes.get(name) != expected:
            raise ValueError(f"site manifest has an invalid {name} route list")


def _validate_phone_steering_payload(payload: Mapping[str, Any]) -> None:
    def matches_number(value: Any, expected: float) -> bool:
        try:
            return math.isclose(float(value), expected)
        except (TypeError, ValueError, OverflowError):
            return False

    if (
        payload.get("schema_id") != "audio-jacobian-lens.phone-steering"
        or payload.get("schema_version") != 1
        or payload.get("mode") != "static_recorded_checkpoints"
    ):
        raise ValueError("phone steering payload has an invalid schema")
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("media_included") is not False:
        raise ValueError("phone steering payload does not exclude source media")
    baseline = payload.get("baseline")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("recorded") is not True
        or baseline.get("interpolated") is not False
        or baseline.get("generated")
        != {"text": "Lily!", "token_ids": [20037, 0], "target_match": False}
    ):
        raise ValueError("phone steering payload has an invalid baseline")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != {"yanny", "laurel"}:
        raise ValueError("phone steering payload has an invalid target set")
    baseline_decisions = baseline.get("decisions")
    if (
        not isinstance(baseline_decisions, Mapping)
        or set(baseline_decisions) != {"yanny", "laurel"}
        or any(
            not isinstance(decisions, list)
            or not decisions
            or any(not isinstance(decision, Mapping) for decision in decisions)
            for decisions in baseline_decisions.values()
        )
    ):
        raise ValueError("phone steering payload has invalid baseline decisions")
    for name, target in targets.items():
        if not isinstance(target, Mapping) or target.get("layers") != [0, 1, 2, 3]:
            raise ValueError(f"phone steering {name} has an invalid layer schedule")
        schedule = target.get("schedule")
        heatmap = target.get("coefficient_heatmap")
        checkpoints = target.get("checkpoints")
        if (
            not isinstance(schedule, list)
            or not schedule
            or any(
                not isinstance(segment, Mapping)
                or not isinstance(segment.get("phone"), str)
                or not segment["phone"]
                or not all(
                    isinstance(segment.get(field), (int, float))
                    and math.isfinite(float(segment[field]))
                    for field in (
                        "start_seconds",
                        "end_seconds",
                        "start_position",
                        "end_position",
                    )
                )
                or float(segment["start_seconds"]) >= float(segment["end_seconds"])
                or int(segment["start_position"]) != segment["start_position"]
                or int(segment["end_position"]) != segment["end_position"]
                or int(segment["start_position"]) >= int(segment["end_position"])
                for segment in schedule
            )
            or not isinstance(heatmap, list)
            or len(heatmap) != 4
            or any(not isinstance(row, Mapping) for row in heatmap)
            or [row.get("layer") for row in heatmap] != [0, 1, 2, 3]
            or any(
                not isinstance(row.get("values"), list)
                or len(row["values"]) != len(schedule)
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in row["values"]
                )
                for row in heatmap
            )
            or not isinstance(checkpoints, list)
            or any(not isinstance(item, Mapping) for item in checkpoints)
            or [item.get("id") for item in checkpoints]
            != ["last_failure", "first_success", "recommended"]
        ):
            raise ValueError(f"phone steering {name} has incomplete recorded data")
        if any(
            not isinstance(item, Mapping)
            or item.get("recorded") is not True
            or item.get("interpolated") is not False
            for item in checkpoints
        ):
            raise ValueError(f"phone steering {name} contains an interpolated run")
        if any(
            not isinstance(item.get("generated"), Mapping)
            or not isinstance(item.get("decisions"), list)
            or not item["decisions"]
            or any(not isinstance(decision, Mapping) for decision in item["decisions"])
            for item in checkpoints
        ):
            raise ValueError(f"phone steering {name} has an incomplete recorded run")

    yanny = targets["yanny"]
    yanny_recommended = yanny["checkpoints"][2]
    yanny_evidence = yanny.get("evidence")
    yanny_decisions = yanny_recommended.get("decisions")
    if (
        not isinstance(yanny_evidence, Mapping)
        or yanny_evidence.get("tier") != "open_loop_cross_fit_reproduced"
        or yanny_recommended.get("generated", {}).get("token_ids") != [575, 7737, 0]
        or not isinstance(yanny_decisions, list)
        or any(not isinstance(row, Mapping) for row in yanny_decisions)
        or [row.get("rank") for row in yanny_decisions] != [1, 1]
        or not matches_number(
            yanny_recommended.get("budget_fraction"),
            0.03499999945628625,
        )
        or not matches_number(
            yanny_recommended.get("sequence_probability_product"),
            0.03803320601582527,
        )
    ):
        raise ValueError("phone steering payload changed the verified Yanny result")

    laurel = targets["laurel"]
    laurel_recommended = laurel["checkpoints"][2]
    laurel_evidence = laurel.get("evidence")
    laurel_decisions = laurel_recommended.get("decisions")
    if (
        not isinstance(laurel_evidence, Mapping)
        or laurel_evidence.get("tier") != "target_conditioned_clip_specific_existence"
        or laurel_recommended.get("generated", {}).get("token_ids") != [43442]
        or not isinstance(laurel_decisions, list)
        or len(laurel_decisions) != 1
        or not isinstance(laurel_decisions[0], Mapping)
        or laurel_decisions[0].get("rank") != 1
        or not matches_number(
            laurel_recommended.get("budget_fraction"),
            0.1452915875040831,
        )
        or not matches_number(
            laurel_decisions[0].get("probability"),
            0.10604292899370193,
        )
    ):
        raise ValueError("phone steering payload changed the verified Laurel result")
    serialized = json.dumps(payload).lower().replace("\\", "/")
    for forbidden in (
        "/users/",
        "artifacts/private/",
        "data:audio",
        "audio_url",
        ".mp3",
        ".flac",
        ".wav",
    ):
        if forbidden in serialized:
            raise ValueError(
                f"phone steering payload exposes forbidden data: {forbidden}"
            )


def validate_site(site_root: Path) -> dict[str, int]:
    site_root = site_root.resolve()
    counts: dict[str, int] = {}
    referenced_media: set[Path] = set()
    replay_report_ids: set[str] = set()
    for asset in (
        "assets/explorer.js",
        "assets/explorer.css",
        "assets/app.js",
        "assets/styles.css",
        "assets/steering.js",
        "assets/steering.css",
        "data/phone-steering-results.json",
    ):
        if not (site_root / asset).is_file():
            raise ValueError(f"missing static-site asset: {asset}")
    for asset in ("assets/explorer.js", "assets/app.js", "assets/steering.js"):
        script = (site_root / asset).read_text(encoding="utf-8")
        if (
            "/api/" in script
            or 'method: "POST"' in script
            or "method: 'POST'" in script
        ):
            raise ValueError(f"published {asset} contains a live API call")
    steering_script = (site_root / "assets/steering.js").read_text(encoding="utf-8")
    steering_css = (site_root / "assets/steering.css").read_text(encoding="utf-8")
    for marker in STEERING_SCRIPT_MARKERS:
        if marker not in steering_script:
            raise ValueError(f"static steering renderer is missing {marker!r}")
    for marker in STEERING_CSS_MARKERS:
        if marker not in steering_css:
            raise ValueError(f"static steering styles are missing {marker!r}")
    steering_payload = _load(site_root / "data" / "phone-steering-results.json")
    _validate_safe(steering_payload, reject_artifact_files=True)
    _validate_phone_steering_payload(steering_payload)
    explorer_script = (site_root / "assets/explorer.js").read_text(encoding="utf-8")
    if 'URLSearchParams(window.location.search).get("sample")' not in explorer_script:
        raise ValueError("static explorer does not preserve ?sample selection")
    for marker in (
        "function renderSpeechRows()",
        'class="speech-matrix-window"',
        "cell?.realized_token",
        'class="realized-rank-badge"',
        "showASRRealizedRank",
        "expectedReportCount = 10",
        "payload.report_count !== expectedReportCount",
        'id="sample-search"',
        'class="sample-button-grid"',
        *SPEECH_TERMINATION_SCRIPT_MARKERS,
        *ASR_DECODER_HIERARCHY_SCRIPT_MARKERS,
        *CROSS_FAMILY_SYNCHRONIZED_SCROLL_SCRIPT_MARKERS,
        *ASR_PHONE_SIGNATURE_SCRIPT_MARKERS,
        *ASR_ARCHITECTURE_SCRIPT_MARKERS,
        *ASR_RECORDED_REPLAY_SCRIPT_MARKERS,
    ):
        if marker not in explorer_script:
            raise ValueError(
                "static explorer is missing the readable speech-band contract: "
                f"{marker}"
            )
    for retired_marker in (
        'id="static-filter-toggle"',
        "function renderFilterControl()",
        "function mergeLengthBuckets(",
        "async function toggleFilter(",
    ):
        if retired_marker in explorer_script:
            raise ValueError(
                "static explorer still exposes retired token-length filtering: "
                f"{retired_marker}"
            )
    explorer_css = (site_root / "assets/explorer.css").read_text(encoding="utf-8")
    if ".static-filter" in explorer_css:
        raise ValueError(
            "static explorer CSS still includes retired token-length filtering"
        )
    for marker in (
        ".position-timeline.speech-readable",
        ".speech-matrix-window",
        ".speech-matrix-grid",
        ".realized-rank-badge",
        '[data-family="asr"] .matrix-cell .realized-rank-badge',
        ".sample-picker-tools",
        ".sample-button-grid",
        "overflow-x: hidden",
        *SPEECH_TERMINATION_CSS_MARKERS,
        *ASR_DECODER_HIERARCHY_CSS_MARKERS,
        *CROSS_FAMILY_SYNCHRONIZED_SCROLL_CSS_MARKERS,
        *ASR_PHONE_SIGNATURE_CSS_MARKERS,
        *ASR_ARCHITECTURE_CSS_MARKERS,
        *ASR_RECORDED_REPLAY_CSS_MARKERS,
    ):
        if marker not in explorer_css:
            raise ValueError(
                "static explorer CSS is missing the readable speech-band "
                f"contract: {marker}"
            )

    _validate_route_contract(site_root)

    for family in FAMILIES:
        manifest_path = site_root / "explorer" / "data" / family / "manifest.json"
        manifest = _load(manifest_path)
        _validate_safe(manifest, reject_artifact_files=family == "asr")
        if (
            manifest.get("schema_id") != "audio-jacobian-lens.cached-explorer-manifest"
            or manifest.get("family") != family
            or manifest.get("mode") != "static_cached_explorer"
        ):
            raise ValueError(f"invalid {family} manifest envelope")
        reports = _manifest_reports(manifest, family=family)
        if not manifest.get("provenance", {}).get("lens"):
            raise ValueError(f"{family} manifest has no pinned lens provenance")
        lens_provenance = manifest["provenance"]["lens"]
        if family == "asr":
            _validate_asr_manifest_provenance(manifest)
            if "source_layers" in lens_provenance:
                raise ValueError(
                    "ASR lens provenance has an ambiguous source_layers field"
                )
            if not isinstance(lens_provenance.get("encoder_source_layers"), list):
                raise ValueError("ASR lens provenance has no encoder source layers")
            if not isinstance(lens_provenance.get("decoder_source_layers"), list):
                raise ValueError("ASR lens provenance has no decoder source layers")

        expected_family_files = {manifest_path.resolve()}
        for entry in reports:
            report_path = _site_path(site_root, str(entry["report_url"]))
            expected_parent = (site_root / "explorer" / "data" / family).resolve()
            if report_path.resolve().parent != expected_parent:
                raise ValueError(f"{family} report URL is outside its family directory")
            expected_family_files.add(report_path.resolve())
            if not report_path.is_file() or _sha256(report_path) != entry["sha256"]:
                raise ValueError(f"{family} report hash mismatch: {report_path}")
            if report_path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"{family} report byte count mismatch: {report_path}")
            report = _load(report_path)
            if (
                report.get("family") != family
                or report.get("example_id") != entry["id"]
            ):
                raise ValueError(f"{family} report/manifest identity mismatch")
            _validate_safe(report, reject_artifact_files=family == "asr")
            if family == "tts":
                _validate_tts(report)
            else:
                expected_rights_status = (
                    ASR_REPLAY_RIGHTS_STATUS
                    if family == "asr"
                    and report.get("example_id") == "asr-laurel-yanny"
                    else "cleared_with_attribution"
                )
                _validate_asr_or_speech(
                    report,
                    family=family,
                    expected_rights_status=expected_rights_status,
                )
                if family == "asr":
                    phone_metadata = report["payload"]["metadata"]["phone_signature"]
                    phone_provenance = lens_provenance["phone_signature"]
                    if (
                        phone_metadata["signature_top_k"]
                        != phone_provenance["signature_top_k"]
                        or phone_metadata["phone_inventory_size"]
                        != phone_provenance["phone_inventory_size"]
                        or phone_metadata["prototype_fit_split"]
                        != phone_provenance["training_split"]
                        or phone_metadata["prototype_fit_rows"]
                        != phone_provenance["training_rows"]
                        or phone_metadata["prototype_fit_opened_eval_splits"]
                        != phone_provenance["development_or_test_opened_for_fit"]
                    ):
                        raise ValueError(
                            "ASR report phone metadata disagrees with provenance"
                        )
                    if (
                        report["payload"]["encoder"]["layers"]
                        != lens_provenance["encoder_source_layers"]
                    ):
                        raise ValueError("ASR encoder layers disagree with provenance")
                    if (
                        report["payload"]["decoder"]["layers"]
                        != lens_provenance["decoder_source_layers"]
                    ):
                        raise ValueError("ASR decoder layers disagree with provenance")
                    if "recorded_intervention_replay" in report:
                        _validate_recorded_intervention_replay(report)
                        replay_report_ids.add(str(report.get("example_id") or ""))
                audio_path = _validate_audio_reference(
                    site_root,
                    family=family,
                    report=report,
                    entry=entry,
                )
                referenced_media.add(audio_path.resolve())
                if family == "asr":
                    _validate_filter_cache(site_root, report, entry)
                    filter_path = _site_path(
                        site_root,
                        str(entry["character_length_filter_cache"]["url"]),
                    )
                    if filter_path.resolve().parent != expected_parent:
                        raise ValueError(
                            "ASR filter URL is outside its family directory"
                        )
                    expected_family_files.add(filter_path.resolve())
        family_files = {
            path.resolve()
            for path in manifest_path.parent.glob("*.json")
            if path.is_file()
        }
        if family_files != expected_family_files:
            raise ValueError(
                f"{family} explorer data contains unreferenced or missing JSON files"
            )
        counts[family] = len(reports)

    if replay_report_ids != {"asr-laurel-yanny"}:
        raise ValueError(
            "the ASR explorer must publish exactly one Laurel/Yanny recorded replay"
        )
    _validate_media_union(site_root, referenced_media)
    _validate_site_manifest_integrity(site_root, counts=counts)
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "site_root",
        type=Path,
        help="Path to the published audio-jacobian-lens directory",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    counts = validate_site(args.site_root)
    print(
        "validated static explorer: "
        + ", ".join(f"{family}={count}" for family, count in counts.items())
    )


if __name__ == "__main__":
    main()
