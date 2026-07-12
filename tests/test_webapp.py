from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from jlens.webapp import create_app


class _WorkspaceMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.body_workspace: str | None = None
        self.script_sources: list[str] = []
        self.workspace_links: dict[str, str] = {}
        self._active_workspace_link: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "body":
            self.body_workspace = attributes.get("data-workspace")
        elif tag == "script" and attributes.get("src"):
            self.script_sources.append(str(attributes["src"]))
        elif tag == "a" and attributes.get("data-workspace-link"):
            self._active_workspace_link = str(attributes["data-workspace-link"])
            self.workspace_links[self._active_workspace_link] = ""

    def handle_data(self, data: str) -> None:
        if self._active_workspace_link is not None:
            self.workspace_links[self._active_workspace_link] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._active_workspace_link = None


def _workspace_markup(path: Path) -> _WorkspaceMarkupParser:
    parser = _WorkspaceMarkupParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.workspace_links = {
        workspace: label.strip()
        for workspace, label in parser.workspace_links.items()
    }
    return parser


def short_wav() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(800, dtype=np.float32), 16_000, format="WAV")
    return stream.getvalue()


def sample_directory(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "hello.wav").write_bytes(short_wav())
    (directory / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "hello-world",
                        "file": "hello.wav",
                        "title": "Hello world",
                        "description": "A short, quiet test phrase.",
                        "transcript": "Hello world.",
                        "duration_seconds": 0.05,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_status_and_analyze_without_lens(tmp_path):
    client = TestClient(create_app(None, web_dir=tmp_path))
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["ready"] is False
    response = client.post(
        "/api/analyze", files={"audio": ("short.wav", short_wav(), "audio/wav")}
    )
    assert response.status_code == 503
    assert "No fitted audio lens" in response.json()["detail"]


def test_app_serves_showcase_and_legacy_causal_alias(tmp_path):
    (tmp_path / "index.html").write_text("index", encoding="utf-8")
    showcase = '<body data-workspace="showcase">Showcase replacement</body>'
    (tmp_path / "showcase.html").write_text(showcase, encoding="utf-8")
    (tmp_path / "causal.html").write_text("legacy causal archive", encoding="utf-8")
    client = TestClient(create_app(None, web_dir=tmp_path))

    canonical = client.get("/showcase")
    legacy = client.get("/causal")
    archive = client.get("/causal.html")

    assert canonical.status_code == 200
    assert legacy.status_code == 200
    assert archive.status_code == 200
    assert canonical.text == showcase
    assert legacy.text == showcase
    assert archive.text == "legacy causal archive"
    assert "legacy causal archive" not in legacy.text


def test_showcase_is_static_evidence_with_all_three_model_paths():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    response = TestClient(create_app(None, web_dir=web_dir)).get("/showcase")

    assert response.status_code == 200
    assert 'data-workspace="showcase"' in response.text
    assert "See what becomes readable—and what still fails." in response.text
    assert 'id="asr-showcase"' in response.text
    assert 'id="tts-showcase"' in response.text
    assert 'id="speech-showcase"' in response.text
    assert "this page makes no backend requests" in response.text

    script = (web_dir / "showcase.js").read_text(encoding="utf-8")
    assert "1272-135031-0003" in script
    assert 'asr: { label: "51,864-entry Whisper vocabulary", size: 51864 }' in script
    assert 'tts: { label: "6,561 ordinary acoustic codes", size: 6561 }' in script
    assert 'speech: { label: "61,690-token LFM vocabulary", size: 61690 }' in script
    assert "ID 4106" in script
    assert "0.001708984375 relative residual norm" in script
    assert 'id: "four"' in script
    assert "/api/" not in script


def test_private_phonetic_experiment_requires_explicit_local_mount(tmp_path):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("main explorer", encoding="utf-8")

    experiment_dir = tmp_path / "phonetic-site"
    experiment_dir.mkdir()
    experiment_markup = (
        '<meta name="robots" content="noindex,nofollow">'
        '<body data-experiment="phonetic-signatures">Private results</body>'
    )
    (experiment_dir / "index.html").write_text(
        experiment_markup, encoding="utf-8"
    )
    (experiment_dir / "data.json").write_text(
        '{"status":"private_development"}\n', encoding="utf-8"
    )

    hidden_client = TestClient(create_app(None, web_dir=web_dir))
    assert hidden_client.get("/experiments/phonetic-signatures/").status_code == 404

    mounted_client = TestClient(
        create_app(
            None,
            web_dir=web_dir,
            phonetic_experiment_dir=experiment_dir,
        )
    )
    redirect = mounted_client.get(
        "/experiments/phonetic-signatures",
        follow_redirects=False,
    )
    page = mounted_client.get("/experiments/phonetic-signatures/")
    payload = mounted_client.get("/experiments/phonetic-signatures/data.json")

    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/experiments/phonetic-signatures/"
    assert page.status_code == 200
    assert page.text == experiment_markup
    assert 'content="noindex,nofollow"' in page.text
    assert payload.status_code == 200
    assert payload.json() == {"status": "private_development"}


def test_private_phonetic_experiment_mount_validates_index(tmp_path):
    experiment_dir = tmp_path / "phonetic-site"
    experiment_dir.mkdir()

    with pytest.raises(ValueError, match="must contain index.html"):
        create_app(None, phonetic_experiment_dir=experiment_dir)


def test_app_serves_the_separate_chatterbox_trace_page():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    assert "speech-code predictions form through T3 and relate to their text context" in response.text
    assert 'id="chatterbox-reading-guide"' in response.text
    assert "adapted to text-to-speech" in response.text
    assert "It does not establish" in response.text
    assert "not the paper’s fitted, corpus-averaged lens" in response.text
    assert 'id="chatterbox-waveform"' in response.text
    assert 'id="chatterbox-text-tokens"' in response.text
    assert 'id="chatterbox-inspector-gradient-mass"' in response.text
    assert 'maxlength="240"' in response.text
    assert "per-run text cross-Jacobian" in response.text
    assert "Within-text attention share" in response.text
    assert 'aria-label="Workspace pages"' in response.text


def test_explorer_uses_shared_light_navigation_without_dead_inline_study():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))
    response = client.get("/")
    assert response.status_code == 200
    assert 'content="#f7f8fa"' in response.text
    assert 'aria-label="Workspace pages"' in response.text
    assert 'id="causal-study"' not in response.text
    assert 'id="generated-audio-player"' in response.text
    assert 'id="transcript-label"' in response.text
    assert 'id="model-badge"' in response.text
    assert "Trace audio through a model’s vocabulary space" in response.text
    assert 'id="mode-guide"' in response.text
    assert "How J-lens applies to ASR" in response.text
    assert "Encoder-to-decoder extension" in response.text
    assert "It does not establish" in response.text
    assert 'href="./chatterbox"' in response.text

    script = (web_dir / "app.js").read_text(encoding="utf-8")
    assert "How J-lens applies to speech-to-speech" in script
    assert "playback only; the current language J-lens does not explain" in script
    assert "The FastConformer, audio adapter, codebook heads" in script


def test_workspace_navigation_contract_is_shared_and_packaged():
    root = Path(__file__).resolve().parents[1]
    web_dir = root / "web"
    expected_links = {
        "asr": "ASR",
        "speech": "Speech",
        "tts": "TTS",
        "showcase": "Showcase",
    }
    pages = {
        "index.html": "asr",
        "chatterbox.html": "tts",
        "causal.html": "causal",
        "showcase.html": "showcase",
    }

    for filename, workspace in pages.items():
        markup = _workspace_markup(web_dir / filename)
        assert markup.body_workspace == workspace
        assert any(
            source.startswith("./workspace-nav.js")
            for source in markup.script_sources
        )
        assert markup.workspace_links == expected_links

    navigation_script = (web_dir / "workspace-nav.js").read_text(
        encoding="utf-8"
    )
    assert "[data-workspace-link]" in navigation_script
    assert "document.body.dataset.workspace" in navigation_script
    assert 'localUrl(8000, "/showcase")' in navigation_script
    assert 'showcase: "./showcase.html"' in navigation_script
    assert "configured.causal" in navigation_script

    packaging = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"web/workspace-nav.js"' in packaging
    assert '"web/showcase.css"' in packaging
    assert '"web/showcase.html"' in packaging
    assert '"web/showcase.js"' in packaging
    assert '"web/data/showcase-examples.json"' in packaging


def test_explorer_has_a_mode_specific_speech_demo_and_section_numbering():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    load_demo_start = script.index("function loadDemo()")
    speech_demo_start = script.index("function buildSpeechDemoData()")
    asr_demo_start = script.index("function buildDemoData()")
    load_demo = script[load_demo_start:speech_demo_start]
    speech_demo = script[speech_demo_start:asr_demo_start]

    assert 'state.serverMode === "speech"' in load_demo
    assert "buildSpeechDemoData()" in load_demo
    assert "buildDemoData()" in load_demo
    assert 'model_family: "lfm2_audio"' in speech_demo
    assert 'streams: ["decoder"]' in speech_demo
    assert 'timing_source: "unavailable"' in speech_demo
    assert (
        'elements.decoderSectionLabel.textContent = "04 · LANGUAGE J-LENS"'
        in script
    )
    assert (
        'elements.decoderSectionLabel.textContent = "05 · DECODER LENS"'
        in script
    )


def test_speech_generated_audio_reports_budget_or_eos_termination():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    response = TestClient(create_app(None, web_dir=web_dir)).get("/")
    assert response.status_code == 200
    assert 'id="generation-diagnostics"' in response.text
    assert 'id="generation-stop-status"' in response.text
    assert 'id="generation-stop-detail"' in response.text

    script = (web_dir / "app.js").read_text(encoding="utf-8")
    helper_start = script.index("function generationDiagnosticsFor(")
    helper_end = script.index("function renderGeneratedAudio(", helper_start)
    helper = script[helper_start:helper_end]
    assert "payload?.metadata?.generation_diagnostics" in helper
    assert "payload?.model?.generation_diagnostics" in helper
    assert "termination_reason" in helper
    assert "budget_exhausted" in helper
    assert "max_new_tokens" in helper
    assert "generated_steps" in helper
    assert "text_tokens" in helper
    assert "audio_frames" in helper
    assert "audio_eos_seen" in helper
    assert 'status = "Stopped at generation budget"' in helper
    assert 'status = "Completed at audio EOS"' in helper
    assert 'const completedAtAudioEos = reason === "audio_eos"' in helper

    generated_start = helper_end
    generated_end = script.index("function setRecordError(", generated_start)
    generated = script[generated_start:generated_end]
    assert "generationDiagnosticsFor(payload)" in generated
    assert 'classList.toggle("budget-exhausted"' in generated
    assert 'classList.toggle("completed-at-eos"' in generated

    metadata_start = script.index("function renderMetadata(")
    metadata_end = script.index("function renderTranscriptTokens(", metadata_start)
    metadata = script[metadata_start:metadata_end]
    assert "generationDiagnosticsFor(payload)" in metadata
    assert '"Generation stop"' in metadata


def test_explorer_exposes_full_vocabulary_decoder_length_filter_for_l0_l1():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="decoder-token-length-filter"' in response.text
    assert 'id="decoder-max-token-length"' in response.text
    assert "Rerank the complete lexical L0–L1 vocabulary" in response.text
    assert "Decoder L2 and the output head stay unfiltered" in response.text


def test_asr_explorer_exposes_compact_waveform_aligned_lens_timelines():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    response = TestClient(create_app(None, web_dir=web_dir)).get("/")

    assert response.status_code == 200
    assert re.search(
        r'<div\b(?=[^>]*\bid="encoder-layers")'
        r'(?=[^>]*\bclass="[^"]*\blens-timeline-chart\b[^"]*")'
        r'(?=[^>]*\brole="group")[^>]*>',
        response.text,
    )
    assert re.search(
        r'<div\b(?=[^>]*\bid="decoder-layers")'
        r'(?=[^>]*\bclass="[^"]*\blens-timeline-chart\b[^"]*")'
        r'(?=[^>]*\brole="group")[^>]*>',
        response.text,
    )
    assert re.search(
        r'id="encoder-navigator"[^>]*\bhidden\b', response.text
    )
    assert re.search(
        r'id="decoder-navigator"[^>]*\bhidden\b', response.text
    )
    assert "overlapping audio window" in response.text
    assert "target-mean-relative logit deltas" in response.text
    assert "raw J-lens readout logits" in response.text
    assert "Orange HEAD is actual output probability" in response.text
    assert "exact candidates and score semantics" in response.text


def test_asr_compact_timeline_preserves_overlapping_and_token_time_geometry():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    range_start = script.index("function timelineRangeForColumn(")
    range_end = script.index("function timelineCellGeometry(", range_start)
    range_helper = script[range_start:range_end]
    assert "const direct = timedRange(column)" in range_helper
    assert "if (direct) return" in range_helper
    assert 'kind === "decoder"' in range_helper
    assert "timedRange(state.result?.transcription?.tokens" in range_helper
    assert "logicalOnly: true" in range_helper

    geometry_start = range_end
    geometry_end = script.index("function timelineRankDetails(", geometry_start)
    geometry = script[geometry_start:geometry_end]
    assert "analysisDuration()" in geometry
    assert "timelineRangeForColumn(kind, column, columnIndex)" in geometry
    assert "range.start / duration" in geometry
    assert "range.end / duration" in geometry
    assert "left:" in geometry
    assert "width:" in geometry

    row_start = script.index("function renderTimelineRow(")
    row_end = script.index("function renderOutputHeadTimelineRow(", row_start)
    row = script[row_start:row_end]
    assert 'createElement("button", `lens-timeline-cell ${view.kind}-timeline-cell`)' in row
    assert "view.columns.forEach((column, columnIndex)" in row
    assert "timelineCellGeometry(view.kind, column, columnIndex)" in row
    assert "button.style.left = geometry.left" in row
    assert "button.style.width = geometry.width" in row
    assert 'button.setAttribute("aria-label", description)' in row
    assert "button.title" not in row
    assert 'button.addEventListener("pointerenter"' in row
    assert 'button.addEventListener("focus"' in row
    assert "view.cellButtons.push(button)" in row

    head_start = row_end
    head_end = script.index("function renderCompactTimeline(", head_start)
    head = script[head_start:head_end]
    assert 'createElement("button", "lens-timeline-cell head-timeline-cell")' in head
    assert "view.columns.forEach((column, columnIndex)" in head
    assert 'timelineCellGeometry("decoder", column, columnIndex)' in head
    assert "button.style.left = geometry.left" in head
    assert "button.style.width = geometry.width" in head
    assert 'button.setAttribute("aria-label", description)' in head
    assert "button.title" not in head
    assert 'button.addEventListener("pointerenter"' in head
    assert 'button.addEventListener("focus"' in head
    assert "view.headButtons.push(button)" in head

    compact_start = head_end
    compact_end = script.index("function renderStream(", compact_start)
    compact = script[compact_start:compact_end]
    assert "view.layers.forEach" in compact
    assert "renderTimelineRow(view, layerIndex)" in compact
    assert "renderOutputHeadTimelineRow(view)" in compact
    assert 'container.classList.toggle("logical-token-axis"' in compact


def test_asr_timeline_details_use_scoped_rank_and_score_provenance():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    tokens_start = script.index("function tokensForCell(")
    tokens_end = script.index("function scoreForCell(", tokens_start)
    tokens = script[tokens_start:tokens_end]
    assert "top_tokens_by_length" in tokens
    assert "maximum_decoded_character_length_counts" in tokens
    assert "exact_decoded_character_length_counts" in tokens
    assert "candidate.score" in tokens
    assert "token.score" in tokens
    assert "rank_denominator: denominator" in tokens
    assert 'rank_tie_policy: token.rank_tie_policy || "1_plus_count_strictly_greater"' in tokens

    rank_start = script.index("function timelineRankDetails(")
    rank_end = script.index("function timelineRankLabel(", rank_start)
    rank = script[rank_start:rank_end]
    assert "token?.full_vocabulary_rank" in rank
    assert "token?.full_vocabulary_denominator" in rank
    assert 'token?.rank_space === "full_model_vocabulary"' in rank
    assert 'scope: "full-model-vocabulary"' in rank
    assert "full-model-vocabulary rank #" in rank
    assert "token?.display_vocabulary_rank" in rank
    assert "token?.display_vocabulary_denominator" in rank
    assert 'token?.rank_space === "lexical_display_vocabulary"' in rank
    assert 'scope: "lexical-display-vocabulary"' in rank
    assert "lexical-display-vocabulary rank #" in rank
    assert "≤${filter.limit}-character lexical rank #" in rank
    assert 'scope: "reported returned-list"' in rank
    assert "token?.rank_tie_policy" in rank

    score_start = script.index("function timelineCandidateScoreLabel(")
    score_end = script.index("function renderTimelineRow(", score_start)
    score = script[score_start:score_end]
    assert "candidate?.probability" in score
    assert "candidate?.log_probability" in score
    assert "metric?.columnLabel" in score
    assert "candidate?.score" in score

    lens_start = script.index("function inspectTimelineCell(")
    lens_end = script.index("function inspectOutputHeadPosition(", lens_start)
    lens_detail = script[lens_start:lens_end]
    assert "tokensForCell(kind, cell, layer)" in lens_detail
    assert "topToken.id" in lens_detail
    assert "timelineRankDetails(topToken" in lens_detail
    assert "timelineFilterState(kind, layer)" in lens_detail
    assert "view.metric.description" in lens_detail
    assert "view.metric.columnLabel" in lens_detail
    assert "lensContext(kind, column, safeColumn)" in lens_detail

    head_start = lens_end
    head_end = script.index("function inspectLayerForView(", head_start)
    head_detail = script[head_start:head_end]
    assert "token.top_tokens" in head_detail
    assert "token.id" in head_detail
    assert "token.log_probability" in head_detail
    assert "timelineRankDetails(token, { isHead: true })" in head_detail
    assert "Raw full-softmax output probability" in head_detail
    assert "Full model vocabulary, explicitly unfiltered" in head_detail
    assert "Character-length filters never apply to HEAD" in head_detail
    assert "direct model output, not a J-lens readout" in head_detail

    list_start = script.index("function renderTopTokens(")
    list_end = script.index("function clearInspector(", list_start)
    candidate_list = script[list_start:list_end]
    assert "timelineRankDetails(" in candidate_list
    assert "rank.denominator" in candidate_list
    assert "visibleToken(token.text)" in candidate_list
    assert "token.id" in candidate_list
    assert "timelineCandidateScoreLabel(" in candidate_list
    assert 'item.setAttribute("aria-label"' in candidate_list


def test_asr_compact_timeline_shares_selection_and_native_keyboard_navigation():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    selection_start = script.index("function updateCompactTimelineSelection(")
    selection_end = script.index("function selectTimelineCoordinate(", selection_start)
    selection = script[selection_start:selection_end]
    assert "view.pinnedIndex" in selection
    assert "state.inspectorSelection" in selection
    assert "view.cellButtons.forEach" in selection
    assert "view.headButtons.forEach" in selection
    assert 'button.classList.toggle("selected-column"' in selection
    assert 'button.classList.toggle("selected-coordinate"' in selection
    assert 'button.setAttribute("aria-pressed"' in selection
    assert "button.tabIndex = selectedCoordinate ? 0 : -1" in selection

    coordinate_start = selection_end
    coordinate_end = script.index("function navigateTimelineCell(", coordinate_start)
    coordinate = script[coordinate_start:coordinate_end]
    assert 'kind === "encoder"' in coordinate
    assert "timelineRangeForColumn(" in coordinate
    assert "selectTimelineAtTime(" in coordinate
    assert "tokenIndexForDecoderPosition(columnIndex)" in coordinate
    assert "selectTimelineToken(tokenIndex" in coordinate
    assert "inspectTimelineCell(" in coordinate
    assert "inspectOutputHeadPosition(" in coordinate
    assert "updateCompactTimelineSelection(state.views.encoder)" in coordinate
    assert "updateCompactTimelineSelection(state.views.decoder)" in coordinate
    assert "focusCell" in coordinate

    keyboard_start = coordinate_end
    keyboard_end = script.index("function intensityWord(", keyboard_start)
    keyboard = script[keyboard_start:keyboard_end]
    for key in (
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "Home",
        "End",
    ):
        assert key in keyboard
    assert "event.preventDefault()" in keyboard
    assert "selectTimelineCoordinate(" in keyboard
    assert "focusCell: true" in keyboard

    stream_start = script.index("function showStreamPosition(")
    stream_end = script.index("function renderLayerComparison(", stream_start)
    stream = script[stream_start:stream_end]
    assert "view.pinnedIndex = safeIndex" in stream
    assert "updateCompactTimelineSelection(view)" in stream
    assert "inspectTimelineCell(" in stream
    assert "renderLayerComparison(view)" not in stream

    token_start = script.index("function selectTimelineToken(")
    token_end = script.index("function selectTimelineAtTime(", token_start)
    token_selection = script[token_start:token_end]
    assert "state.timelineSelection =" in token_selection
    assert 'showStreamPosition("encoder"' in token_selection
    assert 'showStreamPosition("decoder"' in token_selection
    assert "setSelectedOutputToken(" in token_selection
    assert "updateTimelineOverlays(" in token_selection

    refresh_start = script.index("function refreshCompactStream(")
    refresh_end = script.index("function pointerTimeInTimeline(", refresh_start)
    refresh = script[refresh_start:refresh_end]
    assert "preserveState: true" in refresh
    assert "refreshed.pinnedIndex" in refresh
    assert "inspectTimelineCell(" in refresh
    assert "inspectOutputHeadPosition(" in refresh

    encoder_filter_start = script.index("function updateEncoderTokenLengthFilter(")
    encoder_filter_end = script.index(
        "function updateDecoderTokenLengthFilter(", encoder_filter_start
    )
    assert 'refreshCompactStream("encoder")' in script[
        encoder_filter_start:encoder_filter_end
    ]
    decoder_filter_end = script.index("function bindEvents(", encoder_filter_end)
    assert 'refreshCompactStream("decoder")' in script[
        encoder_filter_end:decoder_filter_end
    ]


def test_asr_compact_timeline_fits_the_viewport_without_horizontal_scrolling():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    assert re.search(
        r'body\[data-workspace="asr"\]\s*\{[^}]*overflow-x:\s*(?:clip|hidden)',
        styles,
        re.DOTALL,
    )

    timeline_start = styles.index(".legacy-lens-navigator[hidden]")
    timeline_end = styles.index(".inspector {", timeline_start)
    timeline = styles[timeline_start:timeline_end]
    assert ".lens-timeline-chart" in timeline
    assert ".lens-timeline-axis" in timeline
    assert ".lens-timeline-row" in timeline
    assert ".lens-timeline-track" in timeline
    assert ".lens-timeline-cell" in timeline
    assert "grid-template-columns: 42px minmax(0, 1fr)" in timeline
    assert re.search(
        r"\.lens-timeline-track\s*\{[^}]*position:\s*relative",
        timeline,
        re.DOTALL,
    )
    assert re.search(
        r"\.lens-timeline-cell\s*\{[^}]*position:\s*absolute",
        timeline,
        re.DOTALL,
    )
    assert "overflow-x: auto" not in timeline
    assert ".lens-timeline-cell:hover" in timeline
    assert ".lens-timeline-cell:focus-visible" in timeline
    assert ".lens-timeline-cell.selected-column" in timeline
    assert ".lens-timeline-cell.selected-coordinate" in timeline

    mobile_start = styles.index("@media (max-width: 760px)")
    mobile_end = styles.index("@media (max-width: 470px)", mobile_start)
    mobile = styles[mobile_start:mobile_end]
    assert ".lens-timeline-row" in mobile
    assert "minmax(0, 1fr)" in mobile
    assert ".lens-timeline-track { height:" in mobile


def test_asr_compact_timeline_exposes_an_accessible_responsive_tooltip():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    response = TestClient(create_app(None, web_dir=web_dir)).get("/")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    assert response.status_code == 200
    assert re.search(
        r'<[^>]+(?=[^>]*\bid="lens-timeline-tooltip")'
        r'(?=[^>]*\bclass="[^"]*\blens-timeline-tooltip\b[^"]*")'
        r'(?=[^>]*\brole="tooltip")'
        r'(?=[^>]*\bdata-visible="false")'
        r'(?=[^>]*\bhidden\b)[^>]*>',
        response.text,
    )
    for class_name in (
        "lens-tooltip-eyebrow",
        "lens-tooltip-token",
        "lens-tooltip-coordinate",
        "lens-tooltip-metrics",
        "lens-tooltip-candidates",
    ):
        assert class_name in response.text

    tooltip_start = styles.index(".lens-timeline-tooltip")
    tooltip_end = styles.index(".inspector {", tooltip_start)
    tooltip_styles = styles[tooltip_start:tooltip_end]
    assert "position: absolute" in tooltip_styles
    assert "pointer-events: none" in tooltip_styles
    assert re.search(r"(?:width|max-width):[^;]*(?:vw|100%)", tooltip_styles)
    assert "overflow-wrap:" in tooltip_styles
    assert ".lens-tooltip-candidates" in tooltip_styles
    assert ".lens-tooltip-candidate" in tooltip_styles
    timeline_start = styles.index(".layer-comparison.lens-timeline-chart")
    timeline_end = styles.index(".lens-timeline-axis,", timeline_start)
    assert "position: relative" in styles[timeline_start:timeline_end]


def test_asr_timeline_tooltip_shows_three_candidates_with_exact_provenance():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    details_start = script.index("function compactRankValue(")
    details_end = script.index("function renderTimelineTooltip(", details_start)
    details = script[details_start:details_end]
    assert "tokensForCell(" in details
    assert "top_tokens" in details
    assert ".slice(0, 3)" in details
    assert "timelineRankDetails(" in details
    assert "rank.denominator" in details
    assert "rank.scope" in details
    assert "timelineCandidateScoreLabel(" in details
    assert "timelineFilterState(" in details
    assert "lensContext(" in details
    assert "visibleToken(" in details
    assert ".id" in details
    assert "isHead" in details

    render_start = details_end
    render_end = script.index("function positionTimelineTooltip(", render_start)
    render = script[render_start:render_end]
    for element_name in (
        "tooltipEyebrow",
        "tooltipToken",
        "tooltipTokenId",
        "tooltipCoordinate",
        "tooltipMetrics",
        "tooltipCandidates",
    ):
        assert f"elements.{element_name}" in render
    for class_name in (
        "lens-tooltip-candidate",
        "lens-tooltip-rank",
        "lens-tooltip-candidate-token",
        "lens-tooltip-candidate-id",
        "lens-tooltip-candidate-score",
    ):
        assert class_name in render
    assert "details.candidates" in render
    assert "renderTopTokens(" not in render


def test_asr_timeline_tooltip_pointer_and_keyboard_lifecycle_stays_unpinned():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    position_start = script.index("function positionTimelineTooltip(")
    position_end = script.index("function showTimelineTooltip(", position_start)
    position = script[position_start:position_end]
    assert re.search(
        r"(?:window\.innerHeight|document\.documentElement\.clientHeight)",
        position,
    )
    assert 'closest(".lens-timeline-chart")' in position
    assert position.count("getBoundingClientRect()") >= 3
    assert "Math.min(" in position
    assert "Math.max(" in position
    assert ".style.left" in position
    assert ".style.top" in position
    assert "clientX" not in position
    assert "clientY" not in position

    show_start = position_end
    show_end = script.index("function hideTimelineTooltip(", show_start)
    show = script[show_start:show_end]
    assert "state.timelineTooltip" in show
    assert "trigger" in show
    assert "renderTimelineTooltip(" in show
    assert "positionTimelineTooltip(" in show
    assert 'setAttribute("aria-describedby", "lens-timeline-tooltip")' in show
    assert "timelineTooltip.hidden = false" in show
    assert 'timelineTooltip.dataset.visible = "true"' in show
    assert re.search(r"chart\.(?:append|appendChild)\(elements\.timelineTooltip\)", show)

    hide_start = show_end
    hide_end = script.index("function renderTimelineRow(", hide_start)
    hide = script[hide_start:hide_end]
    assert "state.timelineTooltip" in hide
    assert 'removeAttribute("aria-describedby")' in hide
    assert 'timelineTooltip.dataset.visible = "false"' in hide
    assert "timelineTooltip.hidden = true" in hide
    assert re.search(r"trigger:\s*null", hide)

    tooltip_lifecycle = position + show + hide
    assert "state.inspectorSelection" not in tooltip_lifecycle
    assert "inspectTimelineCell(" not in tooltip_lifecycle
    assert "inspectOutputHeadPosition(" not in tooltip_lifecycle
    assert "selectTimelineCoordinate(" not in tooltip_lifecycle

    row_start = hide_end
    row_end = script.index("function renderOutputHeadTimelineRow(", row_start)
    row = script[row_start:row_end]
    head_start = row_end
    head_end = script.index("function renderCompactTimeline(", head_start)
    head = script[head_start:head_end]
    for handlers in (row, head):
        assert 'addEventListener("pointerenter"' in handlers
        assert 'addEventListener("pointerleave"' in handlers
        assert 'addEventListener("pointerdown"' in handlers
        assert 'addEventListener("focus"' in handlers
        assert 'addEventListener("blur"' in handlers
        assert "showTimelineTooltip(" in handlers
        assert "hideTimelineTooltip(" in handlers
        assert '"touch"' in handlers
        assert re.search(
            r'addEventListener\("click".*?showTimelineTooltip\(',
            handlers,
            re.DOTALL,
        )
        assert "button.title" not in handlers
    assert "inspectTimelineCell(" not in row
    assert "inspectOutputHeadPosition(" not in head
    assert "selectTimelineCoordinate(" in row
    assert "selectTimelineCoordinate(" in head

    bind_start = script.index("function bindEvents(")
    bind_end = script.index("\nbindEvents();", bind_start)
    bindings = script[bind_start:bind_end]
    assert "Escape" in bindings
    assert "hideTimelineTooltip(" in bindings
    assert 'addEventListener("scroll"' in bindings
    assert 'addEventListener("resize"' in bindings


def test_asr_demo_supplies_exact_candidate_and_coordinate_provenance():
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "app.js").read_text(encoding="utf-8")

    lens_start = script.index("function decorateDemoLensCandidates(")
    lens_end = script.index("function decorateDemoHeadCandidates(", lens_start)
    lens_candidates = script[lens_start:lens_end]
    for field in (
        "rank_denominator",
        "rank_space",
        "display_vocabulary_rank",
        "display_vocabulary_denominator",
        "full_vocabulary_rank",
        "full_vocabulary_denominator",
        "rank_tie_policy",
        "score_kind",
        "vocabulary_filter",
        "display_lexical_filter_applied",
        "character_length_filter_applied",
        "decoded_character_length",
        "character_length_constraint",
    ):
        assert field in lens_candidates
    assert '"lexical_display_vocabulary"' in lens_candidates
    assert '"exact_decoded_character_length_bucket"' in lens_candidates
    assert 'operator: "exact"' in lens_candidates

    head_start = lens_end
    head_end = script.index("function buildDemoData(", head_start)
    head_candidates = script[head_start:head_end]
    assert 'rank_space: "full_model_vocabulary"' in head_candidates
    assert 'score_kind: "raw_teacher_forced_probability"' in head_candidates
    assert "probability:" in head_candidates
    assert "log_probability:" in head_candidates
    assert "display_lexical_filter_applied: false" in head_candidates
    assert "character_length_filter_applied: false" in head_candidates
    assert "character_length_constraint: null" in head_candidates

    demo_start = head_end
    demo_end = script.index("function createDemoAudio(", demo_start)
    demo = script[demo_start:demo_end]
    for field in (
        "position_index",
        "time_window",
        "candidate_space",
        "primary_rank_space",
        "primary_rank_denominator",
        "character_length_filter_available",
        "character_length_filter_policy",
        "exact_decoded_character_length_counts",
        "maximum_decoded_character_length_counts",
        "candidate_rank_semantics",
    ):
        assert field in demo
    assert 'timing_source: "encoder_pooling_window"' in demo
    assert 'timing_source: "whisper_cross_attention_dtw"' in demo
    assert 'method: "1_plus_count_strictly_greater"' in demo
    assert 'lens_primary_space: "lexical_display_vocabulary"' in demo
    assert 'output_head_primary_space: "full_model_vocabulary"' in demo
    assert "merge disjoint exact-length buckets" in demo


class _FakeBackend:
    def __init__(self):
        self.time_bin_overlap_seconds = None

    def status(self):
        return {"ready": True, "model_id": "fake"}

    def analyze(self, payload: bytes, *, time_bin_overlap_seconds=None):
        self.time_bin_overlap_seconds = time_bin_overlap_seconds
        return {"received": len(payload)}


class _FakeChatterboxBackend:
    def __init__(self):
        self.generated_text = None
        self.trace_request = None

    def status(self):
        return {
            "ready": True,
            "backend": "fake-chatterbox",
            "model": {"model_id": "fake/chatterbox"},
        }

    def synthesize(self, text):
        self.generated_text = text
        return {"analysis_id": "run-1", "input": {"raw_text": text}}

    def trace(self, analysis_id, speech_code_index):
        self.trace_request = (analysis_id, speech_code_index)
        return {
            "analysis_id": analysis_id,
            "selection": {"speech_code_index": speech_code_index},
        }


def test_chatterbox_api_reports_demo_mode_without_backend(tmp_path):
    client = TestClient(create_app(None, web_dir=tmp_path))

    status = client.get("/api/chatterbox/status")
    generation = client.post(
        "/api/chatterbox/generate", data={"text": "Hello from Chatterbox."}
    )
    trace = client.post(
        "/api/chatterbox/trace",
        json={"analysis_id": "run-1", "speech_code_index": 0},
    )

    assert status.status_code == 200
    assert status.json()["ready"] is False
    assert "synthetic" in status.json()["message"]
    assert generation.status_code == 503
    assert trace.status_code == 503


def test_chatterbox_api_delegates_generation_and_trace(tmp_path):
    backend = _FakeChatterboxBackend()
    client = TestClient(
        create_app(None, web_dir=tmp_path, chatterbox_backend=backend)
    )

    status = client.get("/api/chatterbox/status")
    generation = client.post(
        "/api/chatterbox/generate", data={"text": "Trace this sentence."}
    )
    trace = client.post(
        "/api/chatterbox/trace",
        json={"analysis_id": "run-1", "speech_code_index": 7},
    )

    assert status.json()["backend"] == "fake-chatterbox"
    assert generation.status_code == 200
    assert generation.json()["analysis_id"] == "run-1"
    assert backend.generated_text == "Trace this sentence."
    assert trace.status_code == 200
    assert trace.json()["selection"]["speech_code_index"] == 7
    assert backend.trace_request == ("run-1", 7)


def test_chatterbox_api_rejects_cross_origin_generation(tmp_path):
    client = TestClient(
        create_app(
            None,
            web_dir=tmp_path,
            chatterbox_backend=_FakeChatterboxBackend(),
        )
    )

    response = client.post(
        "/api/chatterbox/generate",
        data={"text": "Do not accept this request."},
        headers={"origin": "https://example.invalid"},
    )

    assert response.status_code == 403


def test_app_delegates_upload_to_backend(tmp_path):
    backend = _FakeBackend()
    client = TestClient(create_app(backend, web_dir=tmp_path))
    payload = short_wav()
    response = client.post(
        "/api/analyze", files={"audio": ("short.wav", payload, "audio/wav")}
    )
    assert response.status_code == 200
    assert response.json() == {"received": len(payload)}
    assert backend.time_bin_overlap_seconds is None


def test_app_delegates_encoder_overlap_to_backend(tmp_path):
    backend = _FakeBackend()
    client = TestClient(create_app(backend, web_dir=tmp_path))
    response = client.post(
        "/api/analyze",
        files={"audio": ("short.wav", short_wav(), "audio/wav")},
        data={"time_bin_overlap_seconds": "0.02"},
    )
    assert response.status_code == 200
    assert backend.time_bin_overlap_seconds == pytest.approx(0.02)


def test_app_rejects_cross_origin_analysis(tmp_path):
    client = TestClient(create_app(_FakeBackend(), web_dir=tmp_path))
    response = client.post(
        "/api/analyze",
        files={"audio": ("short.wav", short_wav(), "audio/wav")},
        headers={"origin": "https://example.invalid"},
    )
    assert response.status_code == 403


def test_app_rejects_oversized_content_length_before_parsing(tmp_path):
    client = TestClient(create_app(_FakeBackend(), web_dir=tmp_path))
    response = client.post(
        "/api/analyze",
        content=b"not parsed",
        headers={"content-length": str(66 * 1024 * 1024)},
    )
    assert response.status_code == 413


def test_app_lists_and_serves_bundled_samples(tmp_path):
    directory = sample_directory(tmp_path)
    client = TestClient(
        create_app(None, web_dir=tmp_path / "web", samples_dir=directory)
    )

    response = client.get("/api/samples")
    assert response.status_code == 200
    assert response.json() == {
        "samples": [
            {
                "id": "hello-world",
                "title": "Hello world",
                "description": "A short, quiet test phrase.",
                "transcript": "Hello world.",
                "duration_seconds": 0.05,
                "filename": "hello.wav",
                "media_type": "audio/wav",
                "audio_url": "/api/samples/hello-world",
            }
        ]
    }

    audio = client.get(response.json()["samples"][0]["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.headers["cache-control"] == "public, max-age=3600"
    assert audio.content == (directory / "hello.wav").read_bytes()


def test_real_sample_catalog_recommends_the_phone_signature_example():
    root = Path(__file__).resolve().parents[1]
    client = TestClient(
        create_app(None, web_dir=root / "web", samples_dir=root / "samples")
    )

    samples = client.get("/api/samples").json()["samples"]
    assert samples[0]["id"] == "buzzer-whirr"
    assert samples[0]["badge"] == "PHONE SIGNATURE EXAMPLE"
    assert samples[0]["recommended_for"] == "phone-signature"
    assert "recommended for Phone signature view" in samples[0]["description"]


def test_app_returns_404_for_missing_and_traversal_sample_ids(tmp_path):
    directory = sample_directory(tmp_path)
    client = TestClient(
        create_app(None, web_dir=tmp_path / "web", samples_dir=directory)
    )

    assert client.get("/api/samples/missing").status_code == 404
    traversal = client.get("/api/samples/%2E%2E%2Foutside.wav")
    assert traversal.status_code == 404


@pytest.mark.parametrize("filename", ["../outside.wav", "missing.wav"])
def test_sample_manifest_rejects_unsafe_or_missing_files(tmp_path, filename):
    outside = tmp_path / "outside.wav"
    outside.write_bytes(short_wav())
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "samples.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "unsafe",
                        "file": filename,
                        "title": "Unsafe",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="basename|does not exist"):
        create_app(None, web_dir=tmp_path / "web", samples_dir=directory)
