from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jlens.webapp import AnalysisBusyError, create_app


class _ChatterboxBackend:
    def __init__(self) -> None:
        self.generated: list[str] = []
        self.traced: list[tuple[str, int]] = []
        self.branched: list[tuple[str, int, int]] = []
        self.residual_branched: list[
            tuple[str, int, int, tuple[int, ...], int, float]
        ] = []
        self.generate_error: Exception | None = None
        self.trace_error: Exception | None = None
        self.branch_error: Exception | None = None
        self.residual_branch_error: Exception | None = None

    @staticmethod
    def status() -> dict[str, object]:
        return {
            "ready": True,
            "backend": "fake-chatterbox",
            "capabilities": {
                "code_to_text_gradient": True,
                "residual_code_steering": True,
            },
        }

    def synthesize(self, text: str) -> dict[str, object]:
        if self.generate_error is not None:
            raise self.generate_error
        self.generated.append(text)
        return {
            "analysis_id": "run-123",
            "schema_version": 3,
            "input": {"raw_text": text},
        }

    def trace(self, analysis_id: str, speech_code_index: int) -> dict[str, object]:
        if self.trace_error is not None:
            raise self.trace_error
        self.traced.append((analysis_id, speech_code_index))
        return {
            "analysis_id": analysis_id,
            "selection": {"speech_code_index": speech_code_index},
        }

    def branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        replacement_code_id: int,
    ) -> dict[str, object]:
        if self.branch_error is not None:
            raise self.branch_error
        self.branched.append(
            (analysis_id, speech_code_index, replacement_code_id)
        )
        return {
            "analysis_id": "branch-456",
            "schema_version": 3,
            "intervention": {
                "parent_analysis_id": analysis_id,
                "speech_code_index": speech_code_index,
                "replacement_code_id": replacement_code_id,
            },
        }

    def residual_branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        target_code_id: int,
        layers: list[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> dict[str, object]:
        if self.residual_branch_error is not None:
            raise self.residual_branch_error
        self.residual_branched.append(
            (
                analysis_id,
                speech_code_index,
                target_code_id,
                tuple(layers),
                forward_span,
                max_relative_residual_norm,
            )
        )
        positions = list(
            range(speech_code_index, speech_code_index + forward_span)
        )
        fitted_layers = [0, 4, 8]
        before = [0.04 + index * 0.01 for index in range(forward_span)]
        after = [value + 0.08 for value in before]
        return {
            "analysis_id": "residual-789",
            "schema_version": 3,
            "intervention": {
                "schema_version": 1,
                "kind": "t3_post_block_residual_steering_branch",
                "method": "parent_path_local_margin_gradient_calibrated",
                "parent_analysis_id": analysis_id,
                "speech_code_index": speech_code_index,
                "target_code_id": target_code_id,
                "original_realized_code_id": 107,
                "layers": list(layers),
                "forward_span": forward_span,
                "requested_positions": positions,
                "applied_positions": positions,
                "coordinate": (
                    "post_t3_block_output_at_speech_prediction_position"
                ),
                "direction_objective": (
                    "target_raw_logit_minus_parent_strongest_non_target_raw_logit"
                ),
                "direction_source": "parent_teacher_forced_path",
                "future_direction_policy": (
                    "position_specific_parent_path_direction_applied_on_dynamic_branch_path"
                ),
                "suffix_policy": (
                    "argmax_after_repetition_penalty_and_temperature"
                ),
                "max_relative_residual_norm": max_relative_residual_norm,
                "chosen_relative_residual_norm": min(
                    0.25, max_relative_residual_norm
                ),
                "target_became_raw_top1": True,
                "processed_greedy_code_id_at_anchor": target_code_id,
                "processed_greedy_equals_target": True,
                "calibration_status": "succeeded",
                "calibration_attempts": [
                    {
                        "attempt_index": 0,
                        "relative_residual_norm": 0.125,
                        "target_probability": 0.12,
                        "target_log_probability": -2.1203,
                        "target_rank": 2,
                        "raw_top1_code_id": 52,
                        "target_logit_margin_to_strongest_other": -0.24,
                        "target_is_raw_top1": False,
                        "processed_greedy_code_id": 52,
                        "processed_greedy_equals_target": False,
                        "success": False,
                    },
                    {
                        "attempt_index": 1,
                        "relative_residual_norm": 0.25,
                        "target_probability": 0.31,
                        "target_log_probability": -1.1712,
                        "target_rank": 1,
                        "raw_top1_code_id": target_code_id,
                        "target_logit_margin_to_strongest_other": 0.08,
                        "target_is_raw_top1": True,
                        "processed_greedy_code_id": target_code_id,
                        "processed_greedy_equals_target": True,
                        "success": True,
                    },
                ],
                "coordinates": [
                    {
                        "layer": layer,
                        "speech_code_index": position,
                        "competitor_code_id": 52,
                        "gradient_l2_norm": 1.5,
                        "baseline_residual_l2_norm": 16.0,
                        "applied_delta_l2_norm": 4.0,
                        "applied_relative_residual_norm": 0.25,
                        "applied": True,
                    }
                    for layer in layers
                    for position in positions
                ],
                "parent_speech_code_count": 12,
                "branch_speech_code_count": 12,
                "branch_emitted_code_id_at_start": target_code_id,
                "first_suffix_divergence_index": speech_code_index,
                "limitations": [
                    "Target success is local to the selected raw output head."
                ],
                "target_diagnostics": {
                    "schema_version": 1,
                    "normalization": (
                        "full_speech_head_softmax_before_generation_processors"
                    ),
                    "positions": positions,
                    "fitted_layers": fitted_layers,
                    "before_probabilities": [before for _ in fitted_layers],
                    "after_probabilities": [after for _ in fitted_layers],
                    "before_ranks": [
                        [9 - index for index in range(forward_span)]
                        for _ in fitted_layers
                    ],
                    "after_ranks": [
                        [4 - min(index, 2) for index in range(forward_span)]
                        for _ in fitted_layers
                    ],
                    "head_before_probabilities": before,
                    "head_after_probabilities": after,
                    "head_before_ranks": [
                        8 - index for index in range(forward_span)
                    ],
                    "head_after_ranks": [
                        3 - min(index, 1) for index in range(forward_span)
                    ],
                    "parent_realized_ids": [100 + index for index in positions],
                    "branch_realized_ids": [
                        target_code_id
                        if index == speech_code_index
                        else 200 + index
                        for index in positions
                    ],
                    "edited_coordinates": [
                        {
                            "layer": layer,
                            "speech_code_index": position,
                        }
                        for layer in layers
                        for position in positions
                    ],
                    "first_suffix_divergence_index": speech_code_index,
                },
            },
        }


def _client(tmp_path, backend=None) -> TestClient:
    return TestClient(
        create_app(None, web_dir=tmp_path / "missing-web", chatterbox_backend=backend)
    )


def test_app_serves_dedicated_chatterbox_page() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    assert "Chatterbox Trace" in response.text
    assert "speech-code predictions form through T3 and relate to their text context" in response.text
    assert "adapted to text-to-speech" in response.text
    assert "Per-run code→text cross-Jacobian" in response.text
    assert "exact word-to-waveform alignment" in response.text
    assert "per-run text cross-Jacobian" in response.text
    assert 'id="chatterbox-waveform"' in response.text
    assert 'id="chatterbox-text-tokens"' in response.text


def test_chatterbox_exposes_separate_fitted_speech_code_lens_panel() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    for element_id in (
        "chatterbox-speech-lens",
        "chatterbox-speech-lens-title",
        "chatterbox-speech-lens-empty",
        "chatterbox-speech-lens-focus",
        "chatterbox-speech-lens-heatmap",
        "chatterbox-speech-lens-note",
    ):
        assert f'id="{element_id}"' in response.text
    assert "04 · FITTED SPEECH-CODE J-LENS" in response.text
    assert "05 · TEXT CONTEXT" in response.text
    assert "06 · LAYER × TOKEN MATRICES" in response.text
    assert "No compatible fitted speech-code J-lens was loaded" in response.text
    assert "Fitted J-lens probability" in response.text
    assert "Base output-head probability" in response.text
    assert response.text.index('id="chatterbox-output-title"') < response.text.index(
        'id="chatterbox-speech-lens-title"'
    ) < response.text.index('class="chatterbox-trace-workspace"')


def test_chatterbox_exposes_semantic_selected_position_candidate_inspector() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    for element_id in (
        "chatterbox-speech-candidate-inspector",
        "chatterbox-speech-candidate-title",
        "chatterbox-speech-candidate-context",
        "chatterbox-speech-candidate-rows",
    ):
        assert f'id="{element_id}"' in response.text
    assert re.search(
        r'<section\b[^>]*\bid="chatterbox-speech-candidate-inspector"'
        r'[^>]*\baria-labelledby="chatterbox-speech-candidate-title"[^>]*>',
        response.text,
    )
    assert (
        "Candidate acoustic codes at the selected speech position"
        in response.text
    )
    assert (
        "These IDs are learned acoustic codes, not words or phonemes. "
        "A code’s audible effect depends on neighboring codes and voice context."
        in response.text
    )
    assert re.search(
        r'id="chatterbox-speech-candidate-rows"[^>]*\brole="list"',
        response.text,
    )
    assert response.text.index('id="chatterbox-speech-lens-heatmap"') < (
        response.text.index('id="chatterbox-speech-candidate-inspector"')
    ) < response.text.index('id="chatterbox-speech-lens-note"')


def test_chatterbox_fitted_lens_strips_validate_and_preserve_code_order() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    for selector in (
        '#chatterbox-speech-lens',
        '#chatterbox-speech-lens-empty',
        '#chatterbox-speech-lens-focus',
        '#chatterbox-speech-lens-heatmap',
        '#chatterbox-speech-lens-note',
    ):
        assert f'$("{selector}")' in script
    validation = script[
        script.index("function validateFittedSpeechLens(") : script.index(
            "async function generateSpeech("
        )
    ]
    for payload_key in (
        "target_ids",
        "target_probabilities",
        "target_log_probabilities",
        "target_ranks",
        "top_codes",
        "source_coordinate",
        "target_head",
        "normalization",
        "artifact",
        "warnings",
    ):
        assert payload_key in validation

    rendering = script[
        script.index("function renderFittedSpeechLens(") : script.index(
            "function codeIndexAtTime("
        )
    ]
    assert "chatterbox-speech-lens-band" not in rendering
    assert "Speech positions ${bandStart" not in rendering
    assert "codes.slice(" not in rendering
    assert "for (let bandStart" not in rendering
    assert "const bandCodes = codes" in rendering
    assert "lens.layers.forEach((layer, layerIndex)" in rendering
    assert "bandCodes.forEach((code, bandIndex)" in rendering
    assert rendering.count("bandCodes.forEach((code, bandIndex)") == 2
    assert "const probability = clamp(probabilities[codeIndex])" in rendering
    assert "const rank = Number(ranks[codeIndex])" in rendering
    assert "const topCodes = lens.top_codes[layerIndex][codeIndex]" in rendering
    assert 'cell.dataset.lensLayerIndex = String(layerIndex)' in rendering
    assert 'cell.dataset.codeIndex = String(codeIndex)' in rendering
    assert "metric-fitted-probability" in rendering
    assert "chatterbox-speech-lens-base-row" in rendering
    assert "metric-output-probability" in rendering
    assert "state.selectedLensLayerIndex" in rendering
    assert "state.selectedCodeIndex" in rendering
    assert "selectSpeechCode(codeIndex" in rendering
    assert "target_ranks" in rendering
    assert "top_codes" in rendering
    assert "fitted_speech_code_jlens: createDemoFittedSpeechLens(speechCodes)" in script
    assert ".chatterbox-speech-lens-card" in styles
    assert ".chatterbox-speech-lens-base-row" in styles
    assert ".chatterbox-speech-lens-band" not in styles


def test_chatterbox_fitted_strips_reuse_the_waveform_time_geometry() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    geometry = script[
        script.index("function codeTimelineGeometry(") : script.index(
            "function renderCodeSlices("
        )
    ]
    assert "code?.start_seconds) / safeDuration" in geometry
    assert "code?.end_seconds, code?.start_seconds) / safeDuration" in geometry
    assert "left: `${(start * 100).toFixed(4)}%`" in geometry
    assert "width: `${Math.max(0.08, (end - start) * 100).toFixed(4)}%`" in geometry

    waveform = script[
        script.index("function renderCodeSlices(") : script.index(
            "function fittedLensCellSelector("
        )
    ]
    assert waveform.count("codeTimelineGeometry(code, duration)") == 2
    assert "slice.style.left = geometry.left" in waveform
    assert "slice.style.width = geometry.width" in waveform
    assert "elements.waveformSelection.style.left = geometry.left" in waveform
    assert "elements.waveformSelection.style.width = geometry.width" in waveform

    fitted = script[
        script.index("function renderFittedSpeechLens(") : script.index(
            "function renderFittedSpeechLensFocus("
        )
    ]
    assert "const duration = Math.max(outputDuration(), 0.001)" in fitted
    assert fitted.count("codeTimelineGeometry(code, duration)") == 2
    assert fitted.count("cell.style.left = geometry.left") == 2
    assert fitted.count("cell.style.width = geometry.width") == 2
    assert "--speech-content-end" in fitted
    assert "trailing decoded audio with no T3 speech code" in fitted

    strip_styles = styles[
        styles.index(".chatterbox-speech-lens-chart") : styles.index(
            ".chatterbox-trace-workspace"
        )
    ]
    assert "width: 100%" in strip_styles
    assert "min-width: 0" in strip_styles
    assert ".chatterbox-speech-lens-strip-track" in strip_styles
    assert "position: relative" in strip_styles
    assert "overflow: hidden" in strip_styles
    assert ".chatterbox-speech-lens-strip-cell" in strip_styles
    assert "position: absolute" in strip_styles
    assert "left: var(--speech-content-end)" in strip_styles
    assert "right: 0" in strip_styles
    assert "overflow-x: auto" not in strip_styles
    assert re.search(
        r"\.chatterbox-page\s*\{[^}]*overflow-x:\s*clip",
        styles,
    )


def test_chatterbox_waveform_and_fitted_strips_share_selection_and_focus() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    waveform_selection = script[
        script.index("function renderCodeSelection(") : script.index(
            "function fittedLensCellSelector("
        )
    ]
    assert "state.selectedCodeIndex = index" in waveform_selection
    assert "updateFittedSpeechLensSelection()" in waveform_selection

    fitted_selection = script[
        script.index("function updateFittedSpeechLensSelection(") : script.index(
            "function codeIndexAtTime("
        )
    ]
    assert "Number(cell.dataset.codeIndex) === state.selectedCodeIndex" in fitted_selection
    assert 'cell.setAttribute("aria-pressed", String(selected))' in fitted_selection
    assert "cell.tabIndex = selected ? 0 : -1" in fitted_selection
    assert "selectSpeechCode(codeIndex, { seek: true, announceChange: false })" in fitted_selection
    assert "cell?.focus()" in fitted_selection

    rendering = script[
        script.index("function renderFittedSpeechLens(") : script.index(
            "function codeIndexAtTime("
        )
    ]
    assert 'cell.setAttribute("aria-label", description)' in rendering
    assert "cell.title = description" in rendering
    assert "selectFittedSpeechLensCell(layerIndex, codeIndex, true, true)" in rendering
    assert "selectSpeechCode(codeIndex, { seek: true })" in rendering

    events = script[script.index("function bindEvents(") :]
    assert 'event.target.closest("button.chatterbox-speech-lens-cell")' in events
    assert "selectFittedSpeechLensCell(layerIndex, codeIndex, false, true)" in events
    assert "selectSpeechCode(codeIndexAtTime(time), { seek: true })" in events
    assert "selectSpeechCode(index, { seek: true })" in events


def test_chatterbox_candidate_inspector_renders_layers_head_and_top_codes() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    for selector in (
        '#chatterbox-speech-candidate-inspector',
        '#chatterbox-speech-candidate-context',
        '#chatterbox-speech-candidate-rows',
    ):
        assert f'$("{selector}")' in script

    assert "function renderSpeechCandidateInspector()" in script
    renderer_start = script.index("function renderSpeechCandidateInspector()")
    renderer_end = script.index("\nfunction ", renderer_start + 1)
    renderer = script[renderer_start:renderer_end]

    assert "state.selectedCodeIndex" in renderer
    assert "lens.layers.forEach((layer, layerIndex)" in renderer
    assert "lens.top_codes[layerIndex][codeIndex]" in renderer
    assert "lens.target_ranks[layerIndex][codeIndex]" in renderer
    assert "const head = speechHeadCandidates()" in renderer
    assert "head?.top_codes?.[codeIndex]" in renderer
    assert "head?.target_ranks?.[codeIndex]" in renderer
    assert 'kind: "fitted"' in renderer
    assert 'kind: "head"' in renderer
    assert "bars use an absolute 0–100% scale" in renderer

    helper_start = script.index("function candidateEntriesWithRealized(")
    helper_end = script.index("function renderSpeechCandidateInspector()")
    helper = script[helper_start:helper_end]
    assert "entries.find((entry) => entry.realized)" in helper
    assert "entries.push({" in helper
    assert "outsideTopK: true" in helper
    assert "candidateEntriesWithRealized(" in helper
    assert 'candidate.dataset.codeId = String(entry.id)' in helper
    assert 'candidate.setAttribute("aria-current", "true")' in helper
    assert "chatterbox-speech-candidate-realized" in helper
    assert "Realized rank #${Number(realizedRank)}" in helper
    assert '"--candidate-probability"' in helper
    assert 'isHead ? "HEAD" : `L${layer}`' in helper
    assert "metric-fitted-probability" in helper
    assert "metric-output-probability" in helper


def test_chatterbox_validates_position_aligned_output_head_candidates() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    generation_validation = script[
        script.index("function validateGeneration(") : script.index(
            "function validateFittedSpeechLens("
        )
    ]
    assert "validateSpeechHeadCandidates(" in generation_validation
    validation_start = generation_validation.index(
        "function validateSpeechHeadCandidates("
    )
    validation = generation_validation[validation_start:]
    for field in (
        "target_ids",
        "target_probabilities",
        "target_log_probabilities",
        "target_ranks",
        "top_codes",
    ):
        assert field in validation
    assert "head[key].length !== codeCount" in validation
    assert "Number(targetId) !== Number(codes[codeIndex]?.id)" in validation
    assert "Number(value) < 1" in validation
    assert "Number(entry.probability) < 0" in validation


def test_chatterbox_candidate_inspector_tracks_shared_speech_selection() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    selection = script[
        script.index("function renderCodeSelection(") : script.index(
            "function fittedLensCellSelector("
        )
    ]
    assert "state.selectedCodeIndex = index" in selection
    assert "renderSpeechCandidateInspector()" in selection

    candidate_renderer_start = script.index(
        "function renderSpeechCandidateInspector()"
    )
    candidate_renderer_end = script.index("\nfunction ", candidate_renderer_start + 1)
    candidate_renderer = script[candidate_renderer_start:candidate_renderer_end]
    assert "const code = codes[codeIndex]" in candidate_renderer
    assert "realized acoustic code ID ${code.id}" in candidate_renderer
    assert "speech position s${codeindex + 1}" in candidate_renderer.lower()


def test_chatterbox_candidate_inspector_is_compact_and_responsive() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    for class_name in (
        "chatterbox-speech-candidate-row",
        "chatterbox-speech-candidate-layer",
        "chatterbox-speech-candidate-list",
        "chatterbox-speech-candidate",
        "chatterbox-speech-candidate-bar",
        "chatterbox-speech-candidate-realized",
        "chatterbox-speech-candidate-rank",
    ):
        assert f".{class_name}" in styles

    candidate_start = styles.index(".chatterbox-speech-candidate-inspector")
    candidate_end = styles.index(".chatterbox-trace-workspace", candidate_start)
    candidate_styles = styles[candidate_start:candidate_end]
    assert "min-width: 0" in candidate_styles
    assert "minmax(0, 1fr)" in candidate_styles
    assert "width: var(--candidate-probability)" in candidate_styles
    assert "overflow-x: auto" not in candidate_styles
    assert re.search(
        r"\.chatterbox-page\s*\{[^}]*overflow-x:\s*clip",
        styles,
    )


def test_chatterbox_candidate_branch_ui_explains_output_decision_intervention() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    for element_id in (
        "chatterbox-branch-action",
        "chatterbox-branch-selection",
        "chatterbox-branch-button",
        "chatterbox-branch-label",
        "chatterbox-branch-comparison",
        "chatterbox-original-audio",
        "chatterbox-branch-summary",
    ):
        assert f'id="{element_id}"' in response.text
    branch_start = response.text.index('id="chatterbox-branch-action"')
    branch_end = response.text.index(
        'id="chatterbox-speech-lens-note"', branch_start
    )
    branch_markup = response.text[branch_start:branch_end].lower()
    assert "force" in branch_markup
    assert "actual output decision" in branch_markup
    assert "continu" in branch_markup
    assert "not residual steering" in branch_markup
    assert re.search(
        r'id="chatterbox-branch-button"[^>]*\bdisabled\b',
        response.text,
    )


def test_chatterbox_exposes_distinct_residual_steering_mode_and_controls() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    response = TestClient(create_app(None, web_dir=web_dir)).get(
        "/chatterbox"
    )

    assert response.status_code == 200
    for element_id in (
        "chatterbox-intervention-mode",
        "chatterbox-intervention-force",
        "chatterbox-intervention-residual",
        "chatterbox-residual-controls",
        "chatterbox-residual-selection",
        "chatterbox-residual-target",
        "chatterbox-residual-source-layers",
        "chatterbox-residual-start-position",
        "chatterbox-residual-forward-span",
        "chatterbox-residual-time-range",
        "chatterbox-residual-auto-budget",
        "chatterbox-residual-max-relative-norm",
        "chatterbox-residual-button",
        "chatterbox-residual-label",
        "chatterbox-residual-status",
        "chatterbox-residual-result",
        "chatterbox-residual-summary",
        "chatterbox-residual-focus",
        "chatterbox-residual-delta-matrix",
        "chatterbox-residual-original-audio",
    ):
        assert f'id="{element_id}"' in response.text

    assert re.search(
        r'id="chatterbox-intervention-force"[^>]*\bvalue="force"'
        r'[^>]*\bchecked\b',
        response.text,
    )
    assert re.search(
        r'id="chatterbox-intervention-residual"[^>]*'
        r'\bvalue="residual"',
        response.text,
    )
    assert "Force output code" in response.text
    assert "Steer residual" in response.text

    residual_start = response.text.index('id="chatterbox-residual-controls"')
    residual_end = response.text.index(
        'id="chatterbox-branch-comparison"', residual_start
    )
    residual_copy = response.text[residual_start:residual_end].lower()
    assert "nominates" in residual_copy
    assert "not directly forced" in residual_copy
    assert "may never become top-1" in residual_copy
    assert "added after the selected t3 block" in residual_copy
    assert "causal residual intervention" in residual_copy
    assert "not a fitted-probability swap" in residual_copy
    assert re.search(
        r'id="chatterbox-residual-button"[^>]*\bdisabled\b',
        response.text,
    )


def test_chatterbox_candidates_are_two_step_forced_code_choices() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    row_start = script.index("function createSpeechCandidateRow(")
    row_end = script.index("function renderBranchAction()")
    row_renderer = script[row_start:row_end]
    assert (
        'createElement("button", "chatterbox-speech-candidate '
        'chatterbox-speech-candidate-choice")'
    ) in row_renderer
    assert 'candidate.type = "button"' in row_renderer
    assert "candidate.disabled" in row_renderer
    assert "entry.realized" in row_renderer
    assert "entry.specialToken" in row_renderer
    assert 'const unavailableInDemo = state.mode === "demo"' in row_renderer
    assert "state.branchLoading" in row_renderer
    assert 'candidate.setAttribute("aria-pressed"' in row_renderer
    assert "selectBranchCandidate(" in row_renderer
    assert "branchSpeechFromCandidate()" not in row_renderer

    selection_start = script.index("function selectBranchCandidate(")
    selection_end = script.index("\nfunction ", selection_start + 1)
    selection = script[selection_start:selection_end]
    for shared_field in (
        "speechCodeIndex",
        "kind",
        "layerIndex",
        "codeId",
    ):
        assert shared_field in selection
    assert "state.selectedCodeIndex" in selection
    assert "state.selectedLensLayerIndex" in selection
    assert "state.branchSelection" in selection
    assert "renderSpeechCandidateInspector()" in selection
    assert "branchSpeechFromCandidate()" not in selection

    speech_selection_start = script.index("async function selectSpeechCode(")
    speech_selection_end = script.index(
        "function clearTraceViewForSelection(", speech_selection_start
    )
    speech_selection = script[speech_selection_start:speech_selection_end]
    assert "nextCodeIndex !== state.selectedCodeIndex" in speech_selection
    assert "state.branchSelection = null" in speech_selection
    assert "state.selectedCodeIndex = nextCodeIndex" in speech_selection

    events = script[script.index("function bindEvents(") :]
    assert (
        'elements.branchButton.addEventListener("click", '
        "branchSpeechFromCandidate)"
    ) in events


def test_chatterbox_residual_selection_shares_candidate_position_and_layers() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    for selector in (
        '#chatterbox-intervention-mode',
        '#chatterbox-intervention-force',
        '#chatterbox-intervention-residual',
        '#chatterbox-residual-controls',
        '#chatterbox-residual-source-layers',
        '#chatterbox-residual-start-position',
        '#chatterbox-residual-forward-span',
        '#chatterbox-residual-time-range',
        '#chatterbox-residual-max-relative-norm',
        '#chatterbox-residual-button',
        '#chatterbox-residual-result',
        '#chatterbox-residual-delta-matrix',
    ):
        assert f'$("{selector}")' in script
    for state_field in (
        'interventionMode: "force"',
        "residualController: null",
        "residualLoading: false",
        "residualSelection: null",
        "residualLayers: new Set()",
        "residualStartIndex: 0",
        "residualResult: null",
        'residualStatus: ""',
    ):
        assert state_field in script
    assert "state.capabilities.residual_code_steering" in script

    selection_start = script.index("function selectBranchCandidate(")
    selection_end = script.index(
        "function createSpeechCandidateRow(", selection_start
    )
    selection = script[selection_start:selection_end]
    assert "state.residualSelection = { ...selection }" in selection
    assert "state.residualStartIndex = state.selectedCodeIndex" in selection
    assert "state.residualLayers = new Set([Number(layer)])" in selection
    assert "state.selectedLensLayerIndex" in selection
    assert "runResidualSteering" not in selection

    choices_start = script.index("function renderResidualLayerChoices()")
    choices_end = script.index("function residualSteeringConfig()")
    choices = script[choices_start:choices_end]
    assert "const lens = fittedSpeechLens()" in choices
    assert "Array.isArray(lens?.layers)" in choices
    assert "state.residualLayers" in choices
    assert 'input.type = "checkbox"' in choices
    assert "input.dataset.residualLayer" in choices

    config_start = script.index("function residualSteeringConfig()")
    config_end = script.index("function renderResidualTimeRange(")
    config = script[config_start:config_end]
    assert "codes.length - start" in config
    assert "Math.min(8" in config
    assert "elements.residualForwardSpan.value" in config
    assert "elements.residualMaxRelativeNorm.value" in config
    assert "0.01, 2" in config
    assert "[...state.residualLayers].sort" in config

    time_start = script.index("function renderResidualTimeRange(")
    time_end = script.index("function renderResidualControls()")
    time_range = script[time_start:time_end]
    assert "startCode.start_seconds" in time_range
    assert "endCode.end_seconds" in time_range
    assert "config.forwardSpan" in time_range

    mode_start = script.index("function renderInterventionMode()")
    mode_end = script.index("function renderBranchAction()")
    mode = script[mode_start:mode_end]
    assert "forcedBranchAvailable()" in mode
    assert "residualSteeringAvailable()" in mode
    assert 'state.interventionMode = "residual"' in mode
    assert 'state.interventionMode = "force"' in mode
    assert "elements.branchAction.hidden" in mode
    assert "renderResidualControls()" in mode


def test_chatterbox_branch_posts_exact_forced_code_request_and_refocuses() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    assert 'branch: "/api/chatterbox/branch"' in script
    branch_start = script.index("async function branchSpeechFromCandidate()")
    branch_end = script.index(
        "function renderSpeechCandidateInspector()", branch_start + 1
    )
    branch = script[branch_start:branch_end]
    assert "state.mode === \"demo\"" in branch
    assert "state.branchSelection" in branch
    assert "const controller = new AbortController()" in branch
    assert "state.branchController = controller" in branch
    assert "fetch(CHATTERBOX_API.branch" in branch
    assert 'method: "POST"' in branch
    assert '"Content-Type": "application/json"' in branch
    for request_field in (
        "analysis_id: parent.analysis_id",
        "speech_code_index: selection.speechCodeIndex",
        "replacement_code_id: selection.codeId",
    ):
        assert request_field in branch
    assert "validateBranchGeneration(payload" in branch
    assert "const comparison = { parent, branch, source" in branch
    assert "branch.intervention" in branch
    assert 'renderGeneration(branch, "branch")' in branch
    assert "state.branchComparison = comparison" in branch
    assert "selectSpeechCode(request.speech_code_index" in branch

    validation_start = script.index("function validateBranchGeneration(")
    validation_end = script.index(
        "function validateSpeechHeadCandidates(", validation_start
    )
    validation = script[validation_start:validation_end]
    assert "validateGeneration(payload)" in validation
    assert '"forced_speech_code_autoregressive_branch"' in validation
    assert "intervention.parent_analysis_id" in validation
    assert "intervention.speech_code_index" in validation
    assert "intervention.replacement_code_id" in validation
    assert "branch.analysis_id" in validation


def test_chatterbox_residual_response_requires_exact_arbitrary_target_diagnostics() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    start = script.index("function validateResidualSteeringGeneration(")
    end = script.index("function validateSpeechHeadCandidates(", start)
    validation = script[start:end]
    assert "validateGeneration(payload)" in validation
    for exact_value in (
        "t3_post_block_residual_steering_branch",
        "parent_path_local_margin_gradient_calibrated",
        "post_t3_block_output_at_speech_prediction_position",
        "target_raw_logit_minus_parent_strongest_non_target_raw_logit",
        "parent_teacher_forced_path",
        "position_specific_parent_path_direction_applied_on_dynamic_branch_path",
        "argmax_after_repetition_penalty_and_temperature",
        "full_speech_head_softmax_before_generation_processors",
    ):
        assert exact_value in validation
    for provenance_field in (
        "parent_analysis_id",
        "speech_code_index",
        "target_code_id",
        "layers",
        "forward_span",
        "requested_positions",
        "applied_positions",
        "max_relative_residual_norm",
        "chosen_relative_residual_norm",
        "target_became_raw_top1",
        "processed_greedy_code_id_at_anchor",
        "processed_greedy_equals_target",
        "calibration_status",
        "calibration_attempts",
        "coordinates",
        "limitations",
    ):
        assert provenance_field in validation
    for attempt_field in (
        "attempt_index",
        "relative_residual_norm",
        "target_probability",
        "target_log_probability",
        "target_rank",
        "raw_top1_code_id",
        "target_logit_margin_to_strongest_other",
        "target_is_raw_top1",
        "processed_greedy_code_id",
        "processed_greedy_equals_target",
        "success",
    ):
        assert attempt_field in validation
    assert 'intervention.calibration_status === "succeeded"' in validation
    assert "intervention.target_became_raw_top1" in validation
    assert "intervention.processed_greedy_equals_target" in validation

    for diagnostic_field in (
        "positions",
        "fitted_layers",
        "before_probabilities",
        "after_probabilities",
        "before_ranks",
        "after_ranks",
        "head_before_probabilities",
        "head_after_probabilities",
        "head_before_ranks",
        "head_after_ranks",
        "parent_realized_ids",
        "branch_realized_ids",
        "edited_coordinates",
        "first_suffix_divergence_index",
    ):
        assert diagnostic_field in validation
    assert "request.forward_span" in validation
    assert "request.max_relative_residual_norm" in validation
    assert "branch.fitted_speech_code_jlens" in validation
    assert "validateMatrix" in validation
    assert 'validateMatrix("before_probabilities"' in validation
    assert 'validateMatrix("after_probabilities"' in validation
    assert 'validateMatrix("before_ranks"' in validation
    assert 'validateMatrix("after_ranks"' in validation
    assert "top_codes" not in validation
    assert "speechHeadCandidates" not in validation


def test_chatterbox_residual_branch_posts_exact_request_and_reports_outcome() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    assert 'residualBranch: "/api/chatterbox/residual-branch"' in script
    start = script.index("async function runResidualSteering()")
    end = script.index("async function branchSpeechFromCandidate()", start)
    run = script[start:end]
    assert 'state.mode === "demo"' in run
    assert "residualSteeringAvailable()" in run
    assert "state.residualSelection" in run
    assert "residualSteeringConfig()" in run
    assert "const request = {" in run
    for request_field in (
        "analysis_id: parent.analysis_id",
        "speech_code_index: config.start",
        "target_code_id: selection.codeId",
        "layers: config.layers",
        "forward_span: config.forwardSpan",
        "max_relative_residual_norm: config.maxRelativeResidualNorm",
    ):
        assert request_field in run
    assert "const controller = new AbortController()" in run
    assert "state.residualController = controller" in run
    assert "fetch(CHATTERBOX_API.residualBranch" in run
    assert "fetch(CHATTERBOX_API.branch" not in run
    assert "replacement_code_id" not in run
    assert 'method: "POST"' in run
    assert '"Content-Type": "application/json"' in run
    assert "body: JSON.stringify(request)" in run
    assert "validateResidualSteeringGeneration(payload, request)" in run
    assert 'renderGeneration(branch, "residual-branch")' in run
    assert 'state.interventionMode = "residual"' in run
    assert "state.residualResult = result" in run
    assert "state.residualStatus" in run
    assert 'calibration_status === "succeeded"' in run
    assert "selectSpeechCode(request.speech_code_index" in run
    assert "Residual steering failed" in run

    events = script[script.index("function bindEvents(") :]
    assert (
        'elements.residualButton.addEventListener("click", '
        "runResidualSteering)"
    ) in events
    assert "elements.interventionMode.addEventListener" in events
    assert "elements.residualSourceLayers.addEventListener" in events
    assert "elements.residualStartPosition.addEventListener" in events
    assert "elements.residualForwardSpan.addEventListener" in events
    assert "elements.residualMaxRelativeNorm.addEventListener" in events


def test_chatterbox_residual_result_uses_exact_layer_position_diagnostics() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    matrix_start = script.index("function renderResidualDeltaMatrix()")
    matrix_end = script.index("function renderResidualResult()", matrix_start)
    matrix = script[matrix_start:matrix_end]
    assert "result.intervention.target_diagnostics" in matrix
    assert "diagnostics.positions" in matrix
    assert "diagnostics.fitted_layers.length + 1" in matrix
    assert "residualDiagnosticRow(diagnostics, rowIndex)" in matrix
    assert "diagnostics.before_probabilities.flat()" in matrix
    assert "diagnostics.after_probabilities.flat()" in matrix
    assert "diagnostics.head_before_probabilities" in matrix
    assert "diagnostics.head_after_probabilities" in matrix
    assert "values.beforeRanks[positionIndex]" in matrix
    assert "values.afterRanks[positionIndex]" in matrix
    assert "diagnostics.edited_coordinates" in matrix
    assert "diagnostics.parent_realized_ids" in matrix
    assert "diagnostics.branch_realized_ids" in matrix
    assert "diagnostics.first_suffix_divergence_index" in matrix
    assert 'cell.dataset.residualRowIndex = String(rowIndex)' in matrix
    assert (
        'cell.dataset.residualPositionIndex = String(positionIndex)' in matrix
    )
    assert 'cell.setAttribute("aria-label", description)' in matrix
    assert "selectResidualDeltaCell(rowIndex, positionIndex)" in matrix
    assert "top_codes" not in matrix
    assert "speechHeadCandidates" not in matrix

    focus_start = script.index("function selectResidualDeltaCell(")
    focus_end = script.index("function renderResidualDeltaMatrix()", focus_start)
    focus = script[focus_start:focus_end]
    assert "state.residualFocusLayerIndex" in focus
    assert "state.residualFocusPosition" in focus
    assert "diagnostics.positions[state.residualFocusPosition]" in focus
    assert "selectSpeechCode(speechPosition" in focus

    result_start = script.index("function renderResidualResult()")
    result_end = script.index("async function runResidualSteering()", result_start)
    result = script[result_start:result_end]
    assert "intervention.chosen_relative_residual_norm" in result
    assert "intervention.max_relative_residual_norm" in result
    assert "intervention.applied_positions" in result
    assert "intervention.requested_positions" in result
    assert "intervention.calibration_attempts.length" in result
    assert "intervention.target_became_raw_top1" in result
    assert "intervention.processed_greedy_equals_target" in result
    assert "intervention.processed_greedy_code_id" in result
    assert "diagnostics.first_suffix_divergence_index" in result
    assert "Calibration succeeded after" in result
    assert "Budget exhausted after" in result
    assert "only nominated the target/layer" in result
    assert "elements.residualOriginalAudio.src" in result
    assert "parent.output.audio_data_url" in result
    assert "renderResidualDeltaMatrix()" in result


def test_chatterbox_branch_preserves_original_and_branch_provenance_and_audio() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    for selector in (
        '#chatterbox-branch-action',
        '#chatterbox-branch-selection',
        '#chatterbox-branch-button',
        '#chatterbox-branch-label',
        '#chatterbox-branch-comparison',
        '#chatterbox-original-audio',
        '#chatterbox-branch-summary',
    ):
        assert f'$("{selector}")' in script

    comparison_start = script.index("function renderBranchComparison(")
    comparison_end = script.index("\nfunction ", comparison_start + 1)
    comparison = script[comparison_start:comparison_end]
    assert "state.branchComparison" in comparison
    assert "const { parent, intervention, source }" in comparison
    assert "intervention" in comparison
    assert "replacement_code_id" in comparison
    assert "original_realized_code_id" in comparison
    assert "speech_code_index" in comparison
    assert "prefix_length" in comparison
    assert "regenerated_suffix_start_index" in comparison
    assert "replacement_global_rank" in comparison
    assert "replacement_raw_probability" in comparison
    assert "elements.originalAudio.src" in comparison
    assert "parent.output.audio_data_url" in comparison
    assert "elements.branchSummary" in comparison

    generation_start = script.index("function renderGeneration(")
    generation_end = script.index("function codeTimelineGeometry(")
    generation = script[generation_start:generation_end]
    assert "elements.audio.src = payload.output.audio_data_url" in generation
    assert 'mode === "branch" ? "Forced-code branch"' in generation


def test_chatterbox_branch_controls_are_demo_safe_and_responsive() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    action_start = script.index("function renderBranchAction()")
    action_end = script.index("\nfunction ", action_start + 1)
    action = script[action_start:action_end]
    assert "state.mode === \"demo\"" in action
    assert "elements.branchButton.disabled" in action
    assert "demo" in action.lower()
    assert "real" in action.lower()

    for class_name in (
        "chatterbox-branch-action",
        "chatterbox-branch-action-copy",
        "chatterbox-branch-button",
        "chatterbox-branch-status",
        "chatterbox-branch-comparison",
        "chatterbox-branch-audio-grid",
    ):
        assert f".{class_name}" in styles
    branch_styles_start = styles.index(".chatterbox-branch-action")
    branch_styles_end = styles.index(
        ".chatterbox-trace-workspace", branch_styles_start
    )
    branch_styles = styles[branch_styles_start:branch_styles_end]
    assert "min-width: 0" in branch_styles
    assert "grid-template-columns: minmax(0, 1fr)" in branch_styles
    assert "width: 100%" in branch_styles
    assert "overflow-x: auto" not in branch_styles
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-branch-action\s*\{[^}]*grid-template-columns:\s*1fr",
        styles,
    )
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-branch-audio-grid\s*\{[^}]*grid-template-columns:\s*1fr",
        styles,
    )
    assert re.search(
        r"\.chatterbox-page\s*\{[^}]*overflow-x:\s*clip",
        styles,
    )


def test_chatterbox_residual_controls_are_demo_safe_and_responsive() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    controls_start = script.index("function renderResidualControls()")
    controls_end = script.index("function renderInterventionMode()")
    controls = script[controls_start:controls_end]
    assert 'state.interventionMode !== "residual"' in controls
    assert 'state.mode === "demo"' in controls
    assert "residualSteeringAvailable()" in controls
    assert "Residual steering is unavailable in the synthetic demo" in controls
    assert "not directly forced" in controls
    assert "elements.residualButton.disabled" in controls
    assert "elements.residualSourceLayers.disabled" in controls
    assert "elements.residualStartPosition.disabled" in controls
    assert "elements.residualForwardSpan.disabled" in controls
    assert "elements.residualMaxRelativeNorm.disabled" in controls
    assert "elements.residualAutoBudget.disabled = true" in controls

    for class_name in (
        "chatterbox-intervention-mode",
        "chatterbox-residual-controls",
        "chatterbox-residual-control-grid",
        "chatterbox-residual-source-layers",
        "chatterbox-residual-run-row",
        "chatterbox-residual-button",
        "chatterbox-residual-result",
        "chatterbox-residual-focus",
        "chatterbox-residual-delta-matrix",
        "chatterbox-residual-delta-chart",
        "chatterbox-residual-delta-row",
        "chatterbox-residual-delta-cell",
    ):
        assert f".{class_name}" in styles
    residual_start = styles.index(".chatterbox-intervention-mode")
    residual_end = styles.index(".chatterbox-trace-workspace", residual_start)
    residual_styles = styles[residual_start:residual_end]
    assert "min-width: 0" in residual_styles
    assert "minmax(0, 1fr)" in residual_styles
    assert "overflow-x: auto" not in residual_styles
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-intervention-mode\s*\{[^}]*"
        r"grid-template-columns:\s*1fr",
        styles,
    )
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-residual-control-grid\s*\{[^}]*"
        r"grid-template-columns:\s*1fr",
        styles,
    )
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-residual-run-row\s*\{[^}]*"
        r"grid-template-columns:\s*1fr",
        styles,
    )
    assert re.search(
        r"@media \(max-width: 760px\)[\s\S]*?"
        r"\.chatterbox-residual-button\s*\{[^}]*width:\s*100%",
        styles,
    )
    assert re.search(
        r"\.chatterbox-page\s*\{[^}]*overflow-x:\s*clip",
        styles,
    )


def test_chatterbox_demo_supplies_full_candidate_fields_for_fitted_and_head_rows() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")

    fitted_start = script.index("function createDemoFittedSpeechLens(")
    head_start = script.index("function createDemoSpeechHeadCandidates(")
    generation_start = script.index("function createDemoGeneration(")
    fitted_demo = script[fitted_start:head_start]
    head_demo = script[head_start:generation_start]
    generation_demo = script[
        generation_start : script.index("function createDemoTrace(")
    ]
    assert "top_codes: topCodes" in fitted_demo
    assert "speech_head_candidates: createDemoSpeechHeadCandidates(speechCodes)" in (
        generation_demo
    )
    for field in (
        "target_ids",
        "target_probabilities",
        "target_log_probabilities",
        "target_ranks",
        "top_codes",
    ):
        assert field in head_demo


def test_chatterbox_results_expose_shared_provenance_and_metric_hooks() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    assert 'id="chatterbox-result-mode"' in response.text
    assert 'id="chatterbox-model-stamp"' in response.text
    assert 'id="chatterbox-metadata-list"' in response.text
    for metric_class in (
        "metric-probability",
        "metric-gradient",
        "metric-attention",
    ):
        assert re.search(
            rf'class="[^"]*\b{metric_class}\b[^"]*"', response.text
        )


def test_chatterbox_exposes_two_layer_by_token_heatmaps() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    client = TestClient(create_app(None, web_dir=web_dir))

    response = client.get("/chatterbox")

    assert response.status_code == 200
    assert "All T3 layers across every input token" in response.text
    assert "How the diagnostics change through T3" not in response.text
    for element_id in (
        "chatterbox-layer-matrices",
        "chatterbox-gradient-heatmap",
        "chatterbox-attention-heatmap",
        "chatterbox-matrix-focus",
    ):
        assert f'id="{element_id}"' in response.text
    assert (
        'aria-label="Within-text gradient share by T3 layer and input token"'
        in response.text
    )
    assert (
        'aria-label="Within-text self-attention share by T3 layer and input token"'
        in response.text
    )


def test_chatterbox_heatmaps_render_all_coordinates_and_sync_selection() -> None:
    web_dir = Path(__file__).resolve().parents[1] / "web"
    script = (web_dir / "chatterbox.js").read_text(encoding="utf-8")
    styles = (web_dir / "styles.css").read_text(encoding="utf-8")

    for selector in (
        '#chatterbox-layer-matrices',
        '#chatterbox-gradient-heatmap',
        '#chatterbox-attention-heatmap',
        '#chatterbox-matrix-focus',
    ):
        assert f'$("{selector}")' in script

    facet = script[
        script.index("function renderHeatmapFacet(") : script.index(
            "function renderMatrixFocus("
        )
    ]
    layer_loop = facet.index("layers.forEach((layer, layerIndex)")
    assert "bandTokens.forEach((token, bandIndex)" in facet[layer_loop:]
    assert "const tokenIndex = bandStart + bandIndex" in facet[layer_loop:]
    assert 'cell.dataset.heatmapMetric = config.key' in facet
    assert 'cell.dataset.layerIndex = String(layerIndex)' in facet
    assert 'cell.dataset.textTokenIndex = String(tokenIndex)' in facet
    assert "const value = clamp(values[tokenIndex])" in facet
    assert (
        'cell.style.setProperty("--heatmap-value", value.toFixed(6))'
        in facet
    )
    assert (
        "selectHeatmapCell(layerIndex, tokenIndex, config.key, true, true)"
        in facet
    )

    matrices = script[
        script.index("function renderLayerMatrices(") : script.index(
            "function selectHeatmapCell("
        )
    ]
    assert "renderHeatmapFacet(elements.gradientHeatmap" in matrices
    assert 'matrixKey: "gradient_share"' in matrices
    assert "renderHeatmapFacet(elements.attentionHeatmap" in matrices
    assert 'matrixKey: "attention_share"' in matrices

    selection = script[
        script.index("function selectHeatmapCell(") : script.index(
            "function selectLayer("
        )
    ]
    assert "state.selectedLayerIndex =" in selection
    assert "state.selectedTextIndex =" in selection
    assert "renderTrace()" in selection

    assert "button.chatterbox-heatmap-cell" in script
    assert "chatterbox-layer-summary" not in script
    assert "chatterbox-layer-summary" not in styles


def test_chatterbox_status_reports_demo_and_backend_modes(tmp_path) -> None:
    unavailable = _client(tmp_path).get("/api/chatterbox/status")
    assert unavailable.status_code == 200
    assert unavailable.json()["ready"] is False
    assert "synthetic" in unavailable.json()["message"]

    available = _client(tmp_path, _ChatterboxBackend()).get(
        "/api/chatterbox/status"
    )
    assert available.status_code == 200
    assert available.json()["backend"] == "fake-chatterbox"
    assert available.json()["capabilities"]["code_to_text_gradient"] is True
    assert available.json()["capabilities"]["residual_code_steering"] is True


def test_generate_and_trace_delegate_valid_same_origin_requests(tmp_path) -> None:
    backend = _ChatterboxBackend()
    client = _client(tmp_path, backend)

    generated = client.post(
        "/api/chatterbox/generate",
        data={"text": "The lighthouse glowed."},
        headers={"origin": "http://testserver"},
    )
    traced = client.post(
        "/api/chatterbox/trace",
        json={"analysis_id": "run-123", "speech_code_index": 7},
        headers={"origin": "http://testserver"},
    )

    assert generated.status_code == 200
    assert generated.json()["schema_version"] == 3
    assert backend.generated == ["The lighthouse glowed."]
    assert traced.status_code == 200
    assert traced.json()["selection"]["speech_code_index"] == 7
    assert backend.traced == [("run-123", 7)]


def test_forced_code_branch_delegates_a_strict_same_origin_request(tmp_path) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/branch",
        json={
            "analysis_id": "run-123",
            "speech_code_index": 7,
            "replacement_code_id": 4133,
        },
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 200
    assert response.json()["analysis_id"] == "branch-456"
    assert response.json()["intervention"] == {
        "parent_analysis_id": "run-123",
        "speech_code_index": 7,
        "replacement_code_id": 4133,
    }
    assert backend.branched == [("run-123", 7, 4133)]


def test_residual_branch_delegates_exact_same_origin_request_and_diagnostics(
    tmp_path,
) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/residual-branch",
        json={
            "analysis_id": "run-123",
            "speech_code_index": 7,
            "target_code_id": 4133,
            "layers": [4, 8],
            "forward_span": 3,
            "max_relative_residual_norm": 0.5,
        },
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_id"] == "residual-789"
    intervention = payload["intervention"]
    assert intervention["kind"] == "t3_post_block_residual_steering_branch"
    assert intervention["method"] == (
        "parent_path_local_margin_gradient_calibrated"
    )
    assert intervention["parent_analysis_id"] == "run-123"
    assert intervention["target_code_id"] == 4133
    assert intervention["layers"] == [4, 8]
    assert intervention["forward_span"] == 3
    assert intervention["max_relative_residual_norm"] == 0.5
    assert intervention["chosen_relative_residual_norm"] == 0.25
    assert intervention["requested_positions"] == [7, 8, 9]
    assert intervention["applied_positions"] == [7, 8, 9]
    assert intervention["target_became_raw_top1"] is True
    assert intervention["processed_greedy_equals_target"] is True
    assert len(intervention["calibration_attempts"]) == 2
    assert len(intervention["coordinates"]) == 6

    diagnostics = intervention["target_diagnostics"]
    assert diagnostics["positions"] == [7, 8, 9]
    assert diagnostics["fitted_layers"] == [0, 4, 8]
    assert len(diagnostics["before_probabilities"]) == 3
    assert all(len(row) == 3 for row in diagnostics["before_probabilities"])
    assert len(diagnostics["after_probabilities"]) == 3
    assert diagnostics["head_before_probabilities"] == [0.04, 0.05, 0.06]
    assert diagnostics["head_after_probabilities"] == [0.12, 0.13, 0.14]
    assert diagnostics["parent_realized_ids"] == [107, 108, 109]
    assert diagnostics["branch_realized_ids"] == [4133, 208, 209]
    assert {
        coordinate["layer"]
        for coordinate in diagnostics["edited_coordinates"]
    } == {4, 8}
    assert backend.residual_branched == [
        ("run-123", 7, 4133, (4, 8), 3, 0.5)
    ]


@pytest.mark.parametrize(
    ("path", "request_kwargs"),
    [
        ("/api/chatterbox/generate", {"data": {"text": "hello"}}),
        (
            "/api/chatterbox/trace",
            {"json": {"analysis_id": "run", "speech_code_index": 0}},
        ),
        (
            "/api/chatterbox/branch",
            {
                "json": {
                    "analysis_id": "run",
                    "speech_code_index": 0,
                    "replacement_code_id": 1,
                }
            },
        ),
        (
            "/api/chatterbox/residual-branch",
            {
                "json": {
                    "analysis_id": "run",
                    "speech_code_index": 0,
                    "target_code_id": 1,
                    "layers": [4],
                    "forward_span": 2,
                    "max_relative_residual_norm": 0.5,
                }
            },
        ),
    ],
)
def test_chatterbox_mutations_reject_cross_origin_requests(
    tmp_path, path, request_kwargs
) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        path,
        headers={"origin": "https://example.invalid"},
        **request_kwargs,
    )

    assert response.status_code == 403
    assert "cross-origin" in response.json()["detail"]
    assert backend.generated == []
    assert backend.traced == []
    assert backend.branched == []
    assert backend.residual_branched == []


def test_chatterbox_endpoints_require_a_backend(tmp_path) -> None:
    client = _client(tmp_path)
    generated = client.post(
        "/api/chatterbox/generate", data={"text": "hello"}
    )
    traced = client.post(
        "/api/chatterbox/trace",
        json={"analysis_id": "run", "speech_code_index": 0},
    )
    branched = client.post(
        "/api/chatterbox/branch",
        json={
            "analysis_id": "run",
            "speech_code_index": 0,
            "replacement_code_id": 1,
        },
    )
    residual_branched = client.post(
        "/api/chatterbox/residual-branch",
        json={
            "analysis_id": "run",
            "speech_code_index": 0,
            "target_code_id": 1,
            "layers": [4],
            "forward_span": 2,
            "max_relative_residual_norm": 0.5,
        },
    )

    assert generated.status_code == 503
    assert "audio-jlens-chatterbox" in generated.json()["detail"]
    assert traced.status_code == 503
    assert branched.status_code == 503
    assert residual_branched.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"analysis_id": "", "speech_code_index": 0},
        {"analysis_id": 3, "speech_code_index": 0},
        {"analysis_id": "run", "speech_code_index": True},
        {"analysis_id": "run", "speech_code_index": 1.5},
        {"analysis_id": "run"},
    ],
)
def test_trace_rejects_invalid_identifiers_and_indices(tmp_path, payload) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/trace", json=payload
    )

    assert response.status_code == 400
    assert backend.traced == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "analysis_id": "",
            "speech_code_index": 0,
            "replacement_code_id": 1,
        },
        {
            "analysis_id": "   ",
            "speech_code_index": 0,
            "replacement_code_id": 1,
        },
        {
            "analysis_id": 3,
            "speech_code_index": 0,
            "replacement_code_id": 1,
        },
        {
            "analysis_id": "run",
            "speech_code_index": True,
            "replacement_code_id": 1,
        },
        {
            "analysis_id": "run",
            "speech_code_index": 1.5,
            "replacement_code_id": 1,
        },
        {
            "analysis_id": "run",
            "speech_code_index": 0,
            "replacement_code_id": False,
        },
        {
            "analysis_id": "run",
            "speech_code_index": 0,
            "replacement_code_id": 1.5,
        },
        {"analysis_id": "run", "speech_code_index": 0},
        {
            "analysis_id": "run",
            "speech_code_index": 0,
            "replacement_code_id": 1,
            "unexpected": "field",
        },
    ],
)
def test_branch_rejects_invalid_or_unknown_request_fields(tmp_path, payload) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/branch", json=payload
    )

    assert response.status_code == 400
    assert backend.branched == []


_VALID_RESIDUAL_BRANCH_REQUEST = {
    "analysis_id": "run",
    "speech_code_index": 0,
    "target_code_id": 1,
    "layers": [4, 8],
    "forward_span": 2,
    "max_relative_residual_norm": 0.5,
}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "analysis_id": ""},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "analysis_id": "   "},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "analysis_id": 3},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "speech_code_index": True},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "speech_code_index": 1.5},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "target_code_id": False},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "target_code_id": 1.5},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "layers": "4,8"},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "layers": []},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "layers": [4, True]},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "layers": [4, 8.5]},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "layers": [4, 4]},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "forward_span": True},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "forward_span": 1.5},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "forward_span": 0},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "forward_span": 9},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "max_relative_residual_norm": True},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "max_relative_residual_norm": "0.5"},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "max_relative_residual_norm": 0},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "max_relative_residual_norm": 2.01},
        {**_VALID_RESIDUAL_BRANCH_REQUEST, "unexpected": "field"},
    ],
)
def test_residual_branch_rejects_invalid_or_unknown_request_fields(
    tmp_path, payload
) -> None:
    backend = _ChatterboxBackend()
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/residual-branch", json=payload
    )

    assert response.status_code == 400
    assert backend.residual_branched == []


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ValueError("text is too long"), 400, "text is too long"),
        (AnalysisBusyError("retry shortly"), 429, "retry shortly"),
        (RuntimeError("generation exploded"), 500, "generation exploded"),
    ],
)
def test_generate_maps_backend_errors(tmp_path, error, status_code, detail) -> None:
    backend = _ChatterboxBackend()
    backend.generate_error = error
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/generate", data={"text": "hello"}
    )

    assert response.status_code == status_code
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (KeyError("expired"), 404, "expired"),
        (ValueError("code index is outside the sequence"), 400, "outside"),
        (AnalysisBusyError("retry shortly"), 429, "retry shortly"),
        (RuntimeError("trace exploded"), 500, "trace exploded"),
    ],
)
def test_trace_maps_backend_errors(tmp_path, error, status_code, detail) -> None:
    backend = _ChatterboxBackend()
    backend.trace_error = error
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/trace",
        json={"analysis_id": "run", "speech_code_index": 0},
    )

    assert response.status_code == status_code
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (KeyError("expired"), 404, "expired"),
        (ValueError("replacement is a stop control token"), 400, "stop"),
        (AnalysisBusyError("retry shortly"), 429, "retry shortly"),
        (RuntimeError("branch exploded"), 500, "branch exploded"),
    ],
)
def test_branch_maps_backend_errors(tmp_path, error, status_code, detail) -> None:
    backend = _ChatterboxBackend()
    backend.branch_error = error
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/branch",
        json={
            "analysis_id": "run",
            "speech_code_index": 0,
            "replacement_code_id": 1,
        },
    )

    assert response.status_code == status_code
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (KeyError("expired"), 404, "expired"),
        (ValueError("layer 7 has no fitted lens"), 400, "layer 7"),
        (AnalysisBusyError("retry shortly"), 429, "retry shortly"),
        (RuntimeError("residual replay exploded"), 500, "exploded"),
    ],
)
def test_residual_branch_maps_backend_errors(
    tmp_path, error, status_code, detail
) -> None:
    backend = _ChatterboxBackend()
    backend.residual_branch_error = error
    response = _client(tmp_path, backend).post(
        "/api/chatterbox/residual-branch",
        json=_VALID_RESIDUAL_BRANCH_REQUEST,
    )

    assert response.status_code == status_code
    assert detail in response.json()["detail"]
