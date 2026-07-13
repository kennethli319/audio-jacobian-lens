from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_static_explorer_site as validator


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _noindex(body: str) -> str:
    return (
        '<meta name="robots" content="noindex,nofollow">'
        f"<html><body {body}></body></html>"
    )


def _build_route_fixture(site_root: Path) -> None:
    canonical = {
        "asr": (
            "index.html",
            "./assets/explorer.js",
            "./explorer/data/asr/manifest.json",
            validator.PUBLIC_BASE,
            (
                'href="./"',
                'href="./speech/"',
            ),
        ),
        "speech": (
            "speech/index.html",
            "../assets/explorer.js",
            "../explorer/data/speech/manifest.json",
            f"{validator.PUBLIC_BASE}speech/",
            (
                'href="../"',
                'href="./"',
            ),
        ),
    }
    for family, (path, script, manifest, canonical_url, links) in canonical.items():
        _write(
            site_root / path,
            _noindex(
                f'class="detailed-explorer" data-family="{family}" '
                f'data-manifest-url="{manifest}"'
            )
            + f'<link rel="canonical" href="{canonical_url}">'
            + (
                f'<link rel="stylesheet" href="'
                f"{script.removesuffix('explorer.js')}explorer.css"
                f'?v={validator.EXPLORER_ASSET_VERSION}">'
            )
            + (f'<script src="{script}?v={validator.EXPLORER_ASSET_VERSION}"></script>')
            + 'aria-label="Audio Jacobian Lens home"'
            + 'class="site-nav" aria-label="Model explorers"'
            + (
                f'<a class="repo-link" href="{validator.SOURCE_REPOSITORY_URL}" '
                'target="_blank" rel="noreferrer" '
                'aria-label="Open Audio Jacobian Lens on GitHub to run or host your own copy">'
                "GitHub · self-host ↗</a>"
            )
            + '<span class="static-badge">PREVIEW · STATIC REPLAY</span>'
            + "".join(links),
        )

    findings = {
        "asr": ("findings/index.html", "../", "../"),
        "speech": ("findings/speech/index.html", "../../", "../../"),
    }
    for family, (path, asset_prefix, data_prefix) in findings.items():
        _write(
            site_root / path,
            _noindex(
                f'data-family="{family}" data-data-url="{data_prefix}data/reports.json"'
            )
            + f'<script src="{asset_prefix}assets/app.js"></script>',
        )

    for family in validator.FAMILIES:
        suffix = "" if family == "asr" else f"{family}/"
        _write(
            site_root / "explorer" / family / "index.html",
            _noindex(
                f'class="detailed-explorer" data-family="{family}" '
                f'data-manifest-url="../data/{family}/manifest.json"'
            )
            + (
                f'<link rel="canonical" href="{validator.PUBLIC_BASE}{suffix}">'
                '<link rel="stylesheet" href="../../assets/explorer.css'
                f'?v={validator.EXPLORER_ASSET_VERSION}">'
                '<script src="../../assets/explorer.js'
                f'?v={validator.EXPLORER_ASSET_VERSION}"></script>'
                'aria-label="Audio Jacobian Lens home" '
                'class="site-nav" aria-label="Model explorers" '
                f'<a class="repo-link" href="{validator.SOURCE_REPOSITORY_URL}" '
                'target="_blank" rel="noreferrer" '
                'aria-label="Open Audio Jacobian Lens on GitHub to run or host your own copy">'
                "GitHub · self-host ↗</a>"
                '<span class="static-badge">PREVIEW · STATIC REPLAY</span>'
                'href="../../" href="../../speech/"'
            ),
        )

    _write(
        site_root / "steering" / "index.html",
        _noindex(
            'get("target") `&condition=${target}` window.location.replace '
            'href="../?sample=asr-laurel-yanny"'
        )
        + (
            f'<link rel="canonical" href="{validator.PUBLIC_BASE}'
            '?sample=asr-laurel-yanny">'
        ),
    )

    routes = {
        "detailed_cached_explorers": list(validator.CANONICAL_DETAILED_ROUTES.values()),
        "findings": list(validator.FINDINGS_ROUTES.values()),
        "legacy_explorer_aliases": list(validator.LEGACY_EXPLORER_ROUTES.values()),
        "recorded_interventions": [validator.INTEGRATED_STEERING_ROUTE],
        "retired_redirects": [validator.STEERING_ROUTE],
    }
    _write(site_root / "site-manifest.json", json.dumps({"routes": routes}))


def test_route_contract_keeps_explorers_primary_and_findings_secondary(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)

    validator._validate_route_contract(tmp_path)


@pytest.mark.parametrize("relative_path", validator.RETIRED_PUBLIC_TTS_PATHS)
def test_route_contract_rejects_retired_public_tts_paths(
    tmp_path: Path, relative_path: str
) -> None:
    _build_route_fixture(tmp_path)
    retired = tmp_path / relative_path
    retired.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="retired public TTS path"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_tts_in_public_navigation(tmp_path: Path) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / "index.html"
    page.write_text(
        page.read_text(encoding="utf-8") + '<a href="./tts/">TTS</a>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="retired public TTS page"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_alias_navigation_to_legacy_routes(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)
    alias = tmp_path / "explorer" / "speech" / "index.html"
    html = alias.read_text(encoding="utf-8").replace(
        'href="../../speech/"', 'href="../speech/"'
    )
    alias.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="legacy speech explorer alias"):
        validator._validate_route_contract(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "index.html",
        "speech/index.html",
        "explorer/asr/index.html",
        "explorer/speech/index.html",
    ),
)
def test_route_contract_rejects_retired_steering_in_explorer_header(
    tmp_path: Path, relative_path: str
) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / relative_path
    html = page.read_text(encoding="utf-8").replace(
        "</body>", '<a href="./steering/">Steering</a></body>', 1
    )
    page.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="retired standalone steering"):
        validator._validate_route_contract(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "index.html",
        "speech/index.html",
        "explorer/asr/index.html",
        "explorer/speech/index.html",
    ),
)
def test_route_contract_requires_source_repository_link(
    tmp_path: Path, relative_path: str
) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / relative_path
    html = page.read_text(encoding="utf-8").replace(
        validator.SOURCE_REPOSITORY_URL,
        "https://example.invalid/missing-source",
        1,
    )
    page.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="explorer"):
        validator._validate_route_contract(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "index.html",
        "speech/index.html",
        "explorer/asr/index.html",
        "explorer/speech/index.html",
    ),
)
def test_route_contract_rejects_removed_findings_link_in_explorer_header(
    tmp_path: Path, relative_path: str
) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / relative_path
    html = page.read_text(encoding="utf-8").replace(
        "</body>",
        '<a class="summary-link" href="/findings/">Experiment findings</a></body>',
        1,
    )
    page.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="removed findings header link"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_findings_with_detailed_renderer(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)
    findings = tmp_path / "findings" / "index.html"
    html = findings.read_text(encoding="utf-8").replace(
        "../assets/app.js", "../assets/explorer.js"
    )
    findings.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="asr findings"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_stale_explorer_asset_version(
    tmp_path: Path,
) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / "speech" / "index.html"
    html = page.read_text(encoding="utf-8").replace(
        f"explorer.js?v={validator.EXPLORER_ASSET_VERSION}",
        "explorer.js?v=stale",
    )
    page.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="canonical speech"):
        validator._validate_route_contract(tmp_path)


def test_route_contract_rejects_stale_steering_redirect(tmp_path: Path) -> None:
    _build_route_fixture(tmp_path)
    page = tmp_path / "steering" / "index.html"
    html = page.read_text(encoding="utf-8").replace(
        "?sample=asr-laurel-yanny",
        "?sample=stale",
    )
    page.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="retired steering redirect"):
        validator._validate_route_contract(tmp_path)


def test_renderer_contract_requires_honest_speech_cap_warning() -> None:
    assert validator.SPEECH_TERMINATION_SCRIPT_MARKERS == (
        "function renderSpeechTerminationStatus()",
        'data-speech-termination="budget-exhausted"',
        "response may be truncated",
    )
    assert validator.SPEECH_TERMINATION_CSS_MARKERS == (".generation-status.capped",)


def test_renderer_contract_requires_readable_asr_decoder_hierarchy() -> None:
    assert validator.ASR_DECODER_HIERARCHY_SCRIPT_MARKERS == (
        'const asrDecoderCell = family === "asr" && kind === "decoder";',
        'data-value-role="top-candidate"',
        'realizedBadge = asrDecoderCell || (family === "asr" && kind === "head")',
        'const cellWidth = family === "asr" ? 92 : 82;',
        "renderSpeechRows(),",
        "Decoder boxes show each layer's top candidate",
    )
    assert validator.ASR_DECODER_HIERARCHY_CSS_MARKERS == (
        '[data-family="asr"] .speech-matrix-grid .matrix-cell .matrix-cell-label',
        '[data-family="asr"] .speech-matrix-grid .matrix-cell .realized-rank-badge',
    )


def test_renderer_contract_requires_synchronized_cross_family_scrolling() -> None:
    assert validator.CROSS_FAMILY_SYNCHRONIZED_SCROLL_SCRIPT_MARKERS == (
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
    assert validator.CROSS_FAMILY_SYNCHRONIZED_SCROLL_CSS_MARKERS == (
        ".position-timeline.scrollable",
        ".scrollable-matrix-panel .layer-matrix",
        ".scrollable-matrix-panel .matrix-row",
        ".speech-matrix-scroll",
        "overflow-x: auto",
    )


def test_renderer_contract_requires_asr_phone_signature_hybrid() -> None:
    assert validator.ASR_PHONE_SIGNATURE_SCRIPT_MARKERS == (
        'phoneSignatureEnabled: family === "asr"',
        '!["0", "false", "off"].includes(String(queryParams.get("phone")).toLowerCase())',
        "function validatePhoneSignatureReport(payload)",
        "function renderPhoneSignatureControl()",
        "On by default · turn off for normal token J-Lens readouts",
        'queryParams.set("phone", phoneQueryValue);',
        'workspace.querySelector("#static-phone-signature-toggle")?.focus({ preventScroll: true });',
        "const phoneCell = encoderPhoneMode(kind);",
        "label: phoneMode ? compactText(top?.phone)",
        "descriptor.candidates.slice(0, 5)",
        "Audio and alignment attribution",
        "exact 100/20/80 ms encoder pooling",
    )
    assert validator.ASR_PHONE_SIGNATURE_CSS_MARKERS == (
        ".phone-signature-control",
        ".matrix-cell.phone-signature-cell .matrix-cell-label",
        ".phone-candidate-row",
        ".rights-block",
        ".explorer-tooltip",
    )


def test_renderer_contract_restores_asr_architecture_context() -> None:
    assert validator.ASR_ARCHITECTURE_SCRIPT_MARKERS == (
        'family === "asr" ? "Encoder: Across the audio representation"',
        'family === "asr" ? "Decoder: As each token resolves"',
        'class="matrix-architecture" aria-label="Encoder architecture"',
        'class="matrix-architecture" aria-label="Decoder architecture"',
        "Bidirectional audio-time states",
        "Final L${finalLayer} state → LM head · causal token time",
    )
    assert validator.ASR_ARCHITECTURE_CSS_MARKERS == (".matrix-architecture",)


def test_renderer_contract_requires_recorded_asr_intervention_replay() -> None:
    assert validator.ASR_RECORDED_REPLAY_SCRIPT_MARKERS == (
        "recorded_intervention_replay",
        "audio-jacobian-lens.recorded-asr-intervention-replay",
        "function composeReplayReport(",
        "function activateReplayCondition(",
        "controls: `${renderReplayControl()}${renderPhoneSignatureControl()}`",
        'url.searchParams.set("condition", condition.id);',
        "Updates encoder · decoder · HEAD",
        'entry.id === "asr-laurel-yanny"',
        '<em class="sample-tag">steering experiment</em>',
        'workspace.querySelector(`[data-replay-condition="${condition.id}"]`)?.focus({ preventScroll: true });',
        "function effectiveProvenance()",
        "Original Laurel/Yanny post",
    )
    assert validator.ASR_RECORDED_REPLAY_CSS_MARKERS == (
        ".recorded-replay",
        ".replay-condition-buttons",
        ".replay-active-summary",
        ".replay-attribution",
        ".sample-tag",
        ".matrix-panel .replay-active-summary",
    )


def _recorded_replay_condition(condition_id: str) -> dict:
    expected = validator.ASR_REPLAY_EXPECTED[condition_id]
    schedules = {
        "baseline": [],
        "yanny": [
            ("Y", 4, 9),
            ("AE", 9, 18),
            ("N", 18, 24),
            ("IY", 24, 34),
        ],
        "laurel": [
            ("L", 4, 9),
            ("AO", 9, 18),
            ("R", 18, 22),
            ("AH", 22, 28),
            ("L", 28, 34),
        ],
    }
    return {
        "id": condition_id,
        "label": "No intervention"
        if condition_id == "baseline"
        else condition_id.title(),
        "recorded": True,
        "interpolated": False,
        "generated": {
            "text": expected["text"],
            "token_ids": list(expected["token_ids"]),
            "target_match": expected["target_match"],
        },
        "budget_fraction": expected["budget_fraction"],
        "coefficient_scale": expected["coefficient_scale"],
        "evidence": {
            "tier": expected["evidence_tier"],
            "badge": "Recorded condition",
            "tone": "neutral",
            "summary": "A nonempty evidence summary.",
        },
        "method": {
            "kind": "recorded_method",
            "label": "Recorded method",
            "description": "A nonempty method description.",
            "coefficient_policy": "A nonempty coefficient policy.",
        },
        "layers": [] if condition_id == "baseline" else [0, 1, 2, 3],
        "schedule": [
            {
                "phone": phone,
                "start_seconds": start * 0.02,
                "end_seconds": end * 0.02,
                "start_position": start,
                "end_position": end,
            }
            for phone, start, end in schedules[condition_id]
        ],
    }


@pytest.mark.parametrize("condition_id", validator.ASR_REPLAY_CONDITIONS)
def test_recorded_replay_condition_contract_accepts_exact_schema(
    condition_id: str,
) -> None:
    validator._validate_replay_condition_contract(
        _recorded_replay_condition(condition_id),
        condition_id=condition_id,
        encoder_layers=[0, 1, 2, 3],
        audio_duration_seconds=0.9255328798,
    )


@pytest.mark.parametrize("nested_field", ("generated", "evidence", "method"))
def test_recorded_replay_condition_rejects_unknown_nested_fields(
    nested_field: str,
) -> None:
    condition = _recorded_replay_condition("yanny")
    condition[nested_field]["injected"] = "must not publish"

    with pytest.raises(ValueError, match=f"{nested_field} schema changed"):
        validator._validate_replay_condition_contract(
            condition,
            condition_id="yanny",
            encoder_layers=[0, 1, 2, 3],
            audio_duration_seconds=0.9255328798,
        )


def test_recorded_replay_condition_rejects_unknown_schedule_fields() -> None:
    condition = _recorded_replay_condition("laurel")
    condition["schedule"][0]["injected"] = 1

    with pytest.raises(ValueError, match="schedule item 0 schema changed"):
        validator._validate_replay_condition_contract(
            condition,
            condition_id="laurel",
            encoder_layers=[0, 1, 2, 3],
            audio_duration_seconds=0.9255328798,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda condition: condition["generated"]["token_ids"].__setitem__(0, True),
        lambda condition: condition["generated"].__setitem__("target_match", 1),
        lambda condition: condition["evidence"].__setitem__("summary", 7),
        lambda condition: condition["method"].__setitem__("kind", []),
        lambda condition: condition["layers"].pop(),
        lambda condition: condition["schedule"].pop(),
        lambda condition: condition["schedule"][0].__setitem__("start_seconds", -0.02),
        lambda condition: condition["schedule"][1].__setitem__("start_position", 8),
        lambda condition: condition["schedule"][0].__setitem__("end_seconds", 1.0),
    ),
)
def test_recorded_replay_condition_rejects_wrong_types_counts_and_bounds(
    mutation,
) -> None:
    condition = _recorded_replay_condition("yanny")
    mutation(condition)

    with pytest.raises(ValueError, match="recorded yanny"):
        validator._validate_replay_condition_contract(
            condition,
            condition_id="yanny",
            encoder_layers=[0, 1, 2, 3],
            audio_duration_seconds=0.9255328798,
        )


def _recorded_replay_source() -> dict:
    return {
        "audio_url": "/audio-jacobian-lens/audio/laurel-yanny.mp3",
        "sha256": "1aac3c17d79ab00e7747f7fbaa2e4b1ba05538d9c2666c820d9a3a8444069a49",
        "source_url": validator.ASR_REPLAY_SOURCE_URL,
        "license": validator.ASR_REPLAY_LICENSE,
        "license_url": validator.ASR_REPLAY_LICENSE_URL,
        "attribution": validator.ASR_REPLAY_ATTRIBUTION,
        "modification_notice": validator.ASR_REPLAY_MODIFICATION_NOTICE,
        "rights_status": validator.ASR_REPLAY_RIGHTS_STATUS,
    }


def test_recorded_replay_source_requires_exact_attributed_rights_metadata() -> None:
    validator._validate_recorded_replay_source(_recorded_replay_source())

    for field, replacement in (
        ("rights_status", "cleared_with_attribution"),
        ("source_url", "https://example.invalid/"),
        ("license", "MIT"),
        ("license_url", "https://example.invalid/license"),
        ("attribution", "Generic attribution"),
        ("modification_notice", "Changed"),
    ):
        source = _recorded_replay_source()
        source[field] = replacement
        with pytest.raises(ValueError, match="exact report-local source"):
            validator._validate_recorded_replay_source(source)


def test_safe_cache_rejects_private_artifact_paths() -> None:
    with pytest.raises(ValueError, match="private filesystem"):
        validator._validate_safe({"source": "/Users/example/private/lens.pt"})
    with pytest.raises(ValueError, match="private model or alignment"):
        validator._validate_safe(
            {"source": "phone-prototypes.npz"}, reject_artifact_files=True
        )


def test_asr_manifest_provenance_requires_held_out_train_only_fit() -> None:
    manifest = {
        "provenance": {
            "lens": {
                "encoder": {
                    "sha256": "0" * 64,
                    "public_evaluation_relationship": (
                        "speaker-held-out; public clips use speaker 1272"
                    ),
                },
                "decoder": {"artifact": "decoder lens", "sha256": "1" * 64},
                "phone_signature": {
                    "sha256": "2" * 64,
                    "signature_top_k": 100,
                    "phone_inventory_size": 34,
                    "training_split": "train",
                    "training_rows": 3400,
                    "development_or_test_opened_for_fit": False,
                },
            },
            "rights": {
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "source_url": "https://www.openslr.org/12",
                "alignment_source_url": "https://zenodo.org/records/2619474",
                "alignment_license": "CC BY 4.0",
                "attribution": "LibriSpeech dev-clean",
            },
        }
    }
    validator._validate_asr_manifest_provenance(manifest)

    manifest["provenance"]["lens"]["phone_signature"][
        "development_or_test_opened_for_fit"
    ] = True
    with pytest.raises(ValueError, match="phone-prototype provenance"):
        validator._validate_asr_manifest_provenance(manifest)


def _ten_report_manifest(family: str) -> dict:
    reports = []
    for index in range(validator.EXPECTED_REPORT_COUNT):
        entry = {
            "id": f"{family}-sample-{index}",
            "report_url": (
                f"/audio-jacobian-lens/explorer/data/{family}/sample-{index}.json"
            ),
            "audio_url": (
                None
                if family == "tts"
                else f"/audio-jacobian-lens/audio/sample-{index}.flac"
            ),
        }
        if family == "asr":
            entry["character_length_filter_cache"] = {
                "url": (
                    "/audio-jacobian-lens/explorer/data/asr/"
                    f"sample-{index}.filters.json"
                )
            }
        reports.append(entry)
    return {
        "report_count": validator.EXPECTED_REPORT_COUNT,
        "reports": reports,
    }


@pytest.mark.parametrize("family", validator.FAMILIES)
def test_manifest_index_accepts_ten_unique_reports(family: str) -> None:
    manifest = _ten_report_manifest(family)

    reports = validator._manifest_reports(manifest, family=family)

    assert len(reports) == validator.EXPECTED_REPORT_COUNT


def test_manifest_index_rejects_retired_asr_feature_badge_metadata() -> None:
    manifest = _ten_report_manifest("asr")
    manifest["reports"][0]["featured_views"] = ["asr_phone_signature"]

    with pytest.raises(ValueError, match="retired featured-view"):
        validator._manifest_reports(manifest, family="asr")


def test_manifest_index_rejects_wrong_count_and_duplicate_urls() -> None:
    manifest = _ten_report_manifest("speech")
    manifest["report_count"] = 9
    with pytest.raises(ValueError, match="report_count"):
        validator._manifest_reports(manifest, family="speech")

    manifest = _ten_report_manifest("speech")
    manifest["reports"][1]["report_url"] = manifest["reports"][0]["report_url"]
    with pytest.raises(ValueError, match="URLs"):
        validator._manifest_reports(manifest, family="speech")


def test_manifest_index_rejects_tts_generated_audio_url() -> None:
    manifest = _ten_report_manifest("tts")
    manifest["reports"][0]["audio_url"] = "/generated.wav"

    with pytest.raises(ValueError, match="generated-audio"):
        validator._manifest_reports(manifest, family="tts")


def test_audio_reference_requires_matching_url_and_content_hash(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio" / "sample.flac"
    _write(audio, "cleared audio fixture")
    url = "/audio-jacobian-lens/audio/sample.flac"
    report = {"source": {"audio_url": url, "sha256": validator._sha256(audio)}}
    entry = {"audio_url": url}

    assert (
        validator._validate_audio_reference(
            tmp_path, family="asr", report=report, entry=entry
        )
        == audio
    )

    report["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audio hash mismatch"):
        validator._validate_audio_reference(
            tmp_path, family="asr", report=report, entry=entry
        )


def test_media_validation_accepts_distinct_family_inputs_as_exact_union(
    tmp_path: Path,
) -> None:
    asr_audio = tmp_path / "audio" / "asr-only.mp3"
    speech_audio = tmp_path / "audio" / "speech-only.flac"
    _write(asr_audio, "asr")
    _write(speech_audio, "speech")

    validator._validate_media_union(tmp_path, {asr_audio, speech_audio})

    _write(tmp_path / "audio" / "orphan.wav", "orphan")
    with pytest.raises(ValueError, match="media set"):
        validator._validate_media_union(tmp_path, {asr_audio, speech_audio})


def test_site_manifest_integrity_checks_counts_and_hashes(tmp_path: Path) -> None:
    relative_paths = {
        "assets/explorer.js": "script",
        "assets/explorer.css": "styles",
        "assets/steering.js": "steering script",
        "assets/steering.css": "steering styles",
        "data/phone-steering-results.json": "steering data",
        "steering/index.html": "steering page",
        "explorer/data/asr/manifest.json": "asr",
        "explorer/data/speech/manifest.json": "speech",
    }
    for relative_path, body in relative_paths.items():
        _write(tmp_path / relative_path, body)
    counts = {family: validator.EXPECTED_REPORT_COUNT for family in validator.FAMILIES}
    site_manifest = {
        "report_counts": dict(counts),
        "sha256": {
            relative_path: validator._sha256(tmp_path / relative_path)
            for relative_path in relative_paths
        },
    }
    _write(tmp_path / "site-manifest.json", json.dumps(site_manifest))

    validator._validate_site_manifest_integrity(tmp_path, counts=counts)

    site_manifest["report_counts"]["asr"] = 9
    _write(tmp_path / "site-manifest.json", json.dumps(site_manifest))
    with pytest.raises(ValueError, match="report counts"):
        validator._validate_site_manifest_integrity(tmp_path, counts=counts)


def test_phone_steering_payload_rejects_interpolation_and_evidence_collapse() -> None:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data"
            / "static_phone_steering_v1.json"
        ).read_text(encoding="utf-8")
    )
    validator._validate_phone_steering_payload(payload)

    payload["targets"]["yanny"]["checkpoints"][0]["interpolated"] = True
    with pytest.raises(ValueError, match="interpolated"):
        validator._validate_phone_steering_payload(payload)

    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data"
            / "static_phone_steering_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["targets"]["laurel"]["evidence"]["tier"] = "open_loop_cross_fit_reproduced"
    with pytest.raises(ValueError, match="Laurel"):
        validator._validate_phone_steering_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["targets"]["yanny"]["checkpoints"].__setitem__(0, None),
        lambda payload: payload["targets"]["yanny"].update(
            {"schedule": [{}], "coefficient_heatmap": [None] * 4}
        ),
        lambda payload: payload["targets"]["yanny"]["checkpoints"][2].update(
            {"decisions": "not-a-list"}
        ),
        lambda payload: payload["targets"]["yanny"]["checkpoints"][2].update(
            {"generated": "not-an-object", "budget_fraction": None}
        ),
        lambda payload: payload["baseline"].update({"decisions": {}}),
    ),
)
def test_phone_steering_payload_rejects_malformed_nested_data(mutation) -> None:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data"
            / "static_phone_steering_v1.json"
        ).read_text(encoding="utf-8")
    )
    mutation(payload)

    with pytest.raises(ValueError, match="phone steering payload|phone steering yanny"):
        validator._validate_phone_steering_payload(payload)


def _ranked_token(token_id: int, *, rank: int, score: float | None = None) -> dict:
    token = {
        "id": token_id,
        "text": " the",
        "rank": rank,
        "rank_denominator": 61_690,
        "rank_space": "lexical_display_vocabulary",
        "rank_tie_policy": "1_plus_count_strictly_greater",
        "score_kind": "raw_readout_logit"
        if score is not None
        else "raw_head_probability",
        "vocabulary_filter": {
            "decoded_character_length": 3,
            "display_lexical_eligible": True,
        },
    }
    if score is not None:
        token["score"] = score
    return token


def _speech_report() -> dict:
    head = {
        **_ranked_token(42, rank=2),
        "probability": 0.2,
        "log_probability": -1.609,
        "top_tokens": [{"id": 99, "text": " a", "probability": 0.4}],
    }
    cell = {
        "top_tokens": [{"id": 99, "text": " a", "score": 0.8}],
        "realized_token": _ranked_token(42, rank=15, score=0.25),
    }
    return {
        "source": {"rights_status": "cleared_with_attribution"},
        "payload": {
            "metadata": {
                "generation_diagnostics": {
                    "termination_reason": "audio_eos",
                    "audio_eos_seen": True,
                    "budget_exhausted": False,
                    "generated_steps": 10,
                    "max_new_tokens": 512,
                    "text_tokens": 3,
                    "audio_frames": 6,
                }
            },
            "audio": {"waveform_preview": {"values": [0.0, 0.1]}},
            "transcription": {"tokens": [head]},
            "encoder": {"layers": [], "cells": []},
            "decoder": {"layers": [0, 1], "cells": [[cell], [dict(cell)]]},
        },
    }


def test_speech_site_validation_accepts_exact_realized_rank_provenance() -> None:
    validator._validate_asr_or_speech(_speech_report(), family="speech")


def test_speech_site_validation_accepts_honest_safety_cap_provenance() -> None:
    report = _speech_report()
    report["payload"]["metadata"]["generation_diagnostics"] = {
        "termination_reason": "budget_exhausted",
        "audio_eos_seen": False,
        "budget_exhausted": True,
        "generated_steps": 512,
        "max_new_tokens": 512,
        "text_tokens": 17,
        "audio_frames": 495,
    }

    validator._validate_asr_or_speech(report, family="speech")


def test_speech_site_validation_rejects_missing_or_inconsistent_termination() -> None:
    report = _speech_report()
    del report["payload"]["metadata"]["generation_diagnostics"]
    with pytest.raises(ValueError, match="no generation-termination"):
        validator._validate_asr_or_speech(report, family="speech")

    report = _speech_report()
    diagnostics = report["payload"]["metadata"]["generation_diagnostics"]
    diagnostics["termination_reason"] = "budget_exhausted"
    diagnostics["budget_exhausted"] = True
    with pytest.raises(ValueError, match="termination state is inconsistent"):
        validator._validate_asr_or_speech(report, family="speech")


def test_speech_site_validation_rejects_missing_layer_realized_rank() -> None:
    report = _speech_report()
    del report["payload"]["decoder"]["cells"][1][0]["realized_token"]

    with pytest.raises(ValueError, match="no exact realized-token provenance"):
        validator._validate_asr_or_speech(report, family="speech")


def _asr_report() -> dict:
    head = {
        **_ranked_token(42, rank=2),
        "probability": 0.2,
        "log_probability": -1.609,
        "start_seconds": 0.0,
        "end_seconds": 0.2,
        "top_tokens": [{"id": 99, "text": " a", "probability": 0.4}],
    }
    decoder_cell = {
        "top_tokens": [{"id": 99, "text": " a", "score": 0.8}],
        "realized_token": _ranked_token(42, rank=15, score=0.25),
    }
    encoder_cell = {
        **decoder_cell,
        "phone_signatures": [
            {"phone": "AA", "similarity": 0.8, "rank": 1},
            {"phone": "B", "similarity": 0.6, "rank": 2},
            {"phone": "D", "similarity": 0.4, "rank": 3},
            {"phone": "EH", "similarity": 0.3, "rank": 4},
            {"phone": "F", "similarity": 0.2, "rank": 5},
        ],
        "time_window": {"start_seconds": 0.0, "end_seconds": 0.1},
        "realized_token_position": 0,
        "realized_token_alignment": {
            "match": "overlapping",
            "window_midpoint_seconds": 0.05,
            "token_start_seconds": 0.0,
            "token_end_seconds": 0.2,
            "overlap_seconds": 0.1,
            "overlap_fraction_of_window": 1.0,
        },
    }
    return {
        "example_id": "asr-example",
        "source": {"rights_status": "cleared_with_attribution"},
        "payload": {
            "metadata": {
                "display_vocabulary": {
                    "maximum_decoded_character_length_counts": {
                        "1": 10,
                        "2": 20,
                        "3": 30,
                    }
                },
                "phone_signature": {
                    "available": True,
                    "display_unit": "pooled_encoder_window",
                    "effective_display_hop_seconds": 0.08,
                    "effective_display_window_seconds": 0.1,
                    "interpretation": "prototype cosine, not probability",
                    "method": "nearest_frozen_top_k_j_signature_phone_prototype",
                    "phone_inventory": list(validator.PUBLIC_PHONE_INVENTORY),
                    "phone_inventory_size": len(validator.PUBLIC_PHONE_INVENTORY),
                    "prototype_fit_opened_eval_splits": False,
                    "prototype_fit_rows": 3400,
                    "prototype_fit_split": "train",
                    "prototype_lens_examples": 20,
                    "schema_version": 1,
                    "score_kind": "phone_prototype_cosine_similarity",
                    "signature_top_k": 100,
                    "silence_or_unknown_class_available": False,
                    "training_unit": "aligned_native_20_ms_phone_midpoint_state",
                },
            },
            "audio": {"waveform_preview": {"values": [0.0, 0.1]}},
            "transcription": {"tokens": [head]},
            "encoder": {
                "layers": [0],
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
                    "tie_break": (
                        "closest_interval_midpoint_then_lower_token_position"
                    ),
                },
                "cells": [[encoder_cell]],
            },
            "decoder": {"layers": [0], "cells": [[decoder_cell]]},
        },
    }


def test_asr_site_validation_accepts_exact_realized_rank_provenance() -> None:
    validator._validate_asr_or_speech(_asr_report(), family="asr")


def test_asr_site_validation_does_not_globally_weaken_rights_status() -> None:
    report = _asr_report()
    report["source"]["rights_status"] = validator.ASR_REPLAY_RIGHTS_STATUS

    with pytest.raises(ValueError, match="expected rights status"):
        validator._validate_asr_or_speech(report, family="asr")

    validator._validate_asr_or_speech(
        report,
        family="asr",
        expected_rights_status=validator.ASR_REPLAY_RIGHTS_STATUS,
    )


def test_speech_site_validation_rejects_laurel_yanny_rights_status() -> None:
    report = _speech_report()
    report["source"]["rights_status"] = validator.ASR_REPLAY_RIGHTS_STATUS

    with pytest.raises(ValueError, match="expected rights status"):
        validator._validate_asr_or_speech(report, family="speech")


def test_asr_site_validation_rejects_shifted_encoder_token_mapping() -> None:
    report = _asr_report()
    report["payload"]["encoder"]["cells"][0][0]["realized_token_position"] = 1

    with pytest.raises(ValueError, match="overlap-first"):
        validator._validate_asr_or_speech(report, family="asr")


def test_asr_site_validation_rejects_missing_phone_candidate() -> None:
    report = _asr_report()
    report["payload"]["encoder"]["cells"][0][0]["phone_signatures"].pop()

    with pytest.raises(ValueError, match="no usable phone signature"):
        validator._validate_asr_or_speech(report, family="asr")


def test_asr_site_validation_rejects_adaptive_or_stale_pooling() -> None:
    report = _asr_report()
    report["payload"]["encoder"]["pooling"]["effective_hop_seconds"] = 0.18

    with pytest.raises(ValueError, match="exact 100/20/80 ms"):
        validator._validate_asr_or_speech(report, family="asr")


def _tts_report(*, width: int, generation_cap: int) -> dict:
    return {
        "payload": {
            "model": {
                "generation": {"max_speech_tokens": generation_cap},
            },
            "output": {
                "speech_codes": [{"index": index} for index in range(width)],
                "speech_head_candidates": {
                    "positions": [{"position": index} for index in range(width)]
                },
            },
            "fitted_speech_code_jlens": {
                "rows": [
                    {
                        "layer": 0,
                        "positions": [{"position": index} for index in range(width)],
                    }
                ]
            },
            "traces_by_position": {
                str(index): {"selection": {"speech_code_index": index}}
                for index in range(width)
            },
            "generated_audio_included": False,
        }
    }


def test_tts_site_validation_accepts_sequence_below_saved_cap() -> None:
    validator._validate_tts(_tts_report(width=2, generation_cap=3))


def test_tts_site_validation_rejects_sequence_at_saved_cap() -> None:
    with pytest.raises(ValueError, match="may be truncated"):
        validator._validate_tts(_tts_report(width=3, generation_cap=3))


def test_asr_filter_cache_requires_exact_compact_realized_ranks(
    tmp_path: Path,
) -> None:
    report = _asr_report()
    cache = {
        "example_id": "asr-example",
        "streams": {
            "encoder": {
                "layers": [0],
                "cells": [
                    [
                        {
                            "top_tokens_by_length": {"1": [{"id": 99}]},
                            "realized_rank_by_max_length": {
                                "1": None,
                                "2": None,
                                "3": 7,
                            },
                        }
                    ]
                ],
            },
            "decoder": {
                "layers": [0],
                "cells": [
                    [
                        {
                            "top_tokens_by_length": {"1": [{"id": 99}]},
                            "realized_rank_by_max_length": {
                                "1": None,
                                "2": None,
                                "3": 8,
                            },
                        }
                    ]
                ],
            },
        },
    }
    cache_path = tmp_path / "filters.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    reference = {
        "url": "/audio-jacobian-lens/filters.json",
        "sha256": validator._sha256(cache_path),
        "bytes": cache_path.stat().st_size,
    }
    entry = {
        "character_length_filter_cache": reference,
    }
    report["cache_policy"] = {"character_length_filter_cache": dict(reference)}
    validator._validate_filter_cache(tmp_path, report, entry)

    del cache["streams"]["encoder"]["cells"][0][0]["realized_rank_by_max_length"]
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    entry["character_length_filter_cache"]["sha256"] = validator._sha256(cache_path)
    entry["character_length_filter_cache"]["bytes"] = cache_path.stat().st_size
    report["cache_policy"]["character_length_filter_cache"] = dict(
        entry["character_length_filter_cache"]
    )
    with pytest.raises(ValueError, match="no exact realized ranks"):
        validator._validate_filter_cache(tmp_path, report, entry)
