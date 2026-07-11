"use strict";

const CHATTERBOX_API = Object.freeze({
  status: "/api/chatterbox/status",
  generate: "/api/chatterbox/generate",
  trace: "/api/chatterbox/trace",
  branch: "/api/chatterbox/branch",
  residualBranch: "/api/chatterbox/residual-branch",
});

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  statusButton: $("#chatterbox-status-button"),
  statusDot: $("#chatterbox-status-dot"),
  statusText: $("#chatterbox-status-text"),
  demoButton: $("#chatterbox-demo-button"),
  text: $("#chatterbox-text"),
  characterCount: $("#chatterbox-character-count"),
  presetButtons: $$('[data-chatterbox-preset]'),
  generateButton: $("#chatterbox-generate-button"),
  generateLabel: $("#chatterbox-generate-label"),
  generateLoader: $("#chatterbox-generate-loader"),
  progress: $("#chatterbox-progress"),
  progressLabel: $("#chatterbox-progress-label"),
  progressTime: $("#chatterbox-progress-time"),
  error: $("#chatterbox-error"),
  liveRegion: $("#chatterbox-live-region"),
  inputPanel: $("#chatterbox-input-panel"),
  results: $("#chatterbox-results"),
  resultMode: $("#chatterbox-result-mode"),
  modelStamp: $("#chatterbox-model-stamp"),
  metadataList: $("#chatterbox-metadata-list"),
  resetButton: $("#chatterbox-reset-button"),
  audio: $("#chatterbox-audio"),
  currentTime: $("#chatterbox-current-time"),
  duration: $("#chatterbox-duration"),
  waveform: $("#chatterbox-waveform"),
  waveformCanvas: $("#chatterbox-waveform-canvas"),
  codeSlices: $("#chatterbox-code-slices"),
  waveformSelection: $("#chatterbox-waveform-selection"),
  waveformPlayhead: $("#chatterbox-waveform-playhead"),
  waveformHover: $("#chatterbox-waveform-hover"),
  codeCount: $("#chatterbox-code-count"),
  contentDuration: $("#chatterbox-content-duration"),
  trailingAudio: $("#chatterbox-trailing-audio"),
  selectedCode: $("#chatterbox-selected-code"),
  selectedRange: $("#chatterbox-selected-range"),
  selectedProbability: $("#chatterbox-selected-probability"),
  selectedLogProbability: $("#chatterbox-selected-log-probability"),
  speechLensSection: $("#chatterbox-speech-lens"),
  speechLensEmpty: $("#chatterbox-speech-lens-empty"),
  speechLensFocus: $("#chatterbox-speech-lens-focus"),
  speechLensHeatmap: $("#chatterbox-speech-lens-heatmap"),
  speechLensNote: $("#chatterbox-speech-lens-note"),
  speechCandidateInspector: $("#chatterbox-speech-candidate-inspector"),
  speechCandidateContext: $("#chatterbox-speech-candidate-context"),
  speechCandidateRows: $("#chatterbox-speech-candidate-rows"),
  interventionMode: $("#chatterbox-intervention-mode"),
  interventionForce: $("#chatterbox-intervention-force"),
  interventionResidual: $("#chatterbox-intervention-residual"),
  branchAction: $("#chatterbox-branch-action"),
  branchSelection: $("#chatterbox-branch-selection"),
  branchSource: $("#chatterbox-branch-source"),
  branchButton: $("#chatterbox-branch-button"),
  branchLabel: $("#chatterbox-branch-label"),
  branchLoader: $("#chatterbox-branch-loader"),
  branchStatus: $("#chatterbox-branch-status"),
  branchComparison: $("#chatterbox-branch-comparison"),
  branchIdChange: $("#chatterbox-branch-id-change"),
  branchSummary: $("#chatterbox-branch-summary"),
  originalAudio: $("#chatterbox-original-audio"),
  residualControls: $("#chatterbox-residual-controls"),
  residualSelection: $("#chatterbox-residual-selection"),
  residualTarget: $("#chatterbox-residual-target"),
  residualSourceLayers: $("#chatterbox-residual-source-layers"),
  residualStartPosition: $("#chatterbox-residual-start-position"),
  residualForwardSpan: $("#chatterbox-residual-forward-span"),
  residualTimeRange: $("#chatterbox-residual-time-range"),
  residualAutoBudget: $("#chatterbox-residual-auto-budget"),
  residualMaxRelativeNorm: $("#chatterbox-residual-max-relative-norm"),
  residualButton: $("#chatterbox-residual-button"),
  residualLabel: $("#chatterbox-residual-label"),
  residualLoader: $("#chatterbox-residual-loader"),
  residualStatus: $("#chatterbox-residual-status"),
  residualResult: $("#chatterbox-residual-result"),
  residualResultTarget: $("#chatterbox-residual-result-target"),
  residualSummary: $("#chatterbox-residual-summary"),
  residualFocus: $("#chatterbox-residual-focus"),
  residualDeltaMatrix: $("#chatterbox-residual-delta-matrix"),
  residualOriginalAudio: $("#chatterbox-residual-original-audio"),
  traceLoading: $("#chatterbox-trace-loading"),
  traceLoadingLabel: $("#chatterbox-trace-loading-label"),
  layerSelect: $("#chatterbox-layer-select"),
  textTokens: $("#chatterbox-text-tokens"),
  layerMatrices: $("#chatterbox-layer-matrices"),
  gradientHeatmap: $("#chatterbox-gradient-heatmap"),
  attentionHeatmap: $("#chatterbox-attention-heatmap"),
  matrixFocus: $("#chatterbox-matrix-focus"),
  scoreKind: $("#chatterbox-score-kind"),
  inspectorTitle: $("#chatterbox-inspector-title"),
  inspectorContext: $("#chatterbox-inspector-context"),
  inspectorLayer: $("#chatterbox-inspector-layer"),
  inspectorToken: $("#chatterbox-inspector-token"),
  inspectorGradient: $("#chatterbox-inspector-gradient"),
  inspectorGradientShare: $("#chatterbox-inspector-gradient-share"),
  inspectorGradientMass: $("#chatterbox-inspector-gradient-mass"),
  inspectorAttention: $("#chatterbox-inspector-attention"),
  inspectorTextMass: $("#chatterbox-inspector-text-mass"),
  warnings: $("#chatterbox-warnings"),
  warningList: $("#chatterbox-warning-list"),
};

const state = {
  backendReady: false,
  capabilities: {},
  loading: false,
  mode: null,
  analysis: null,
  trace: null,
  traceCache: new Map(),
  selectedCodeIndex: 0,
  selectedLensLayerIndex: 0,
  selectedLayerIndex: 0,
  selectedTextIndex: null,
  generationController: null,
  branchController: null,
  branchLoading: false,
  branchSelection: null,
  branchComparison: null,
  branchStatus: "",
  interventionMode: "force",
  residualController: null,
  residualLoading: false,
  residualSelection: null,
  residualLayers: new Set(),
  residualStartIndex: 0,
  residualResult: null,
  residualStatus: "",
  residualFocusLayerIndex: 0,
  residualFocusPosition: 0,
  traceController: null,
  traceDebounceTimer: null,
  progressStartedAt: 0,
  progressTimer: null,
  demoAudioUrl: "",
  resizeObserver: null,
  matrixResizeObserver: null,
  matrixBandSize: null,
};

function clamp(value, minimum = 0, maximum = 1) {
  const number = Number(value);
  return Math.min(maximum, Math.max(minimum, Number.isFinite(number) ? number : 0));
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatTime(seconds) {
  const safe = Math.max(0, finiteNumber(seconds));
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${(safe - minutes * 60).toFixed(2).padStart(5, "0")}`;
}

function formatProbability(value, digits = 2) {
  const probability = clamp(value);
  if (probability > 0 && probability < 0.0001) return "<0.01%";
  return `${(probability * 100).toFixed(digits)}%`;
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 1) return number.toFixed(3);
  if (Math.abs(number) >= 0.001) return number.toFixed(5);
  return number.toExponential(3);
}

function visibleToken(value) {
  const text = String(value ?? "");
  if (!text) return "<empty>";
  return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function announce(message) {
  elements.liveRegion.textContent = "";
  window.requestAnimationFrame(() => { elements.liveRegion.textContent = message; });
}

function modelLabel(model) {
  if (typeof model === "string" && model) return model;
  if (model && typeof model === "object") {
    return model.model_id || model.id || model.name || "Chatterbox · MLX";
  }
  return "Chatterbox · MLX";
}

function humanizeKey(value) {
  return String(value).replace(/_/g, " ");
}

function compactMetadataValue(value) {
  if (value === null || value === undefined || value === "") return "Not provided";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value !== "object") return String(value);
  if (Array.isArray(value)) return value.map(compactMetadataValue).join(" · ");
  const entries = Object.entries(value).filter(([, entry]) => entry !== null && entry !== undefined && entry !== "");
  if (!entries.length) return "Not provided";
  return entries.map(([key, entry]) => `${humanizeKey(key)} ${compactMetadataValue(entry)}`).join(" · ");
}

function speechCodes() {
  return Array.isArray(state.analysis?.output?.speech_codes)
    ? state.analysis.output.speech_codes
    : [];
}

function fittedSpeechLens() {
  const lens = state.analysis?.fitted_speech_code_jlens;
  return lens && typeof lens === "object" ? lens : null;
}

function speechHeadCandidates() {
  const candidates = state.analysis?.output?.speech_head_candidates;
  return candidates && typeof candidates === "object" ? candidates : null;
}

function outputDuration() {
  return Math.max(0, finiteNumber(state.analysis?.output?.duration_seconds));
}

function traceLayers() {
  return Array.isArray(state.trace?.layers) ? state.trace.layers : [];
}

function traceTextTokens() {
  if (Array.isArray(state.trace?.text_tokens)) return state.trace.text_tokens;
  return Array.isArray(state.analysis?.input?.tokens) ? state.analysis.input.tokens : [];
}

function matrixRow(matrix, layer, layerIndex, layers = traceLayers()) {
  if (!matrix) return [];
  if (Array.isArray(matrix)) {
    if (matrix.length === layers.length && Array.isArray(matrix[layerIndex])) return matrix[layerIndex];
    if (Array.isArray(matrix[Number(layer)])) return matrix[Number(layer)];
    return [];
  }
  const row = matrix[String(layer)] ?? matrix[layer];
  return Array.isArray(row) ? row : [];
}

function layerScalar(values, layer, layerIndex) {
  if (Array.isArray(values)) {
    if (values.length === traceLayers().length) return finiteNumber(values[layerIndex]);
    return finiteNumber(values[Number(layer)]);
  }
  if (values && typeof values === "object") return finiteNumber(values[String(layer)] ?? values[layer]);
  return 0;
}

function indexOfMaximum(values) {
  if (!values.length) return 0;
  let index = 0;
  let maximum = -Infinity;
  values.forEach((value, candidate) => {
    const number = finiteNumber(value, -Infinity);
    if (number > maximum) {
      maximum = number;
      index = candidate;
    }
  });
  return index;
}

function clearError() {
  elements.error.hidden = true;
  elements.error.textContent = "";
}

function showError(message) {
  elements.error.textContent = String(message);
  elements.error.hidden = false;
  announce(String(message));
}

function updateCharacterCount() {
  const maximum = Number(elements.text.maxLength) || 800;
  elements.characterCount.textContent = `${elements.text.value.length} / ${maximum} characters`;
}

function updateGenerateAvailability() {
  const hasText = Boolean(elements.text.value.trim());
  const busy = state.loading || state.branchLoading || state.residualLoading;
  elements.generateButton.disabled = busy || !state.backendReady || !hasText;
  elements.demoButton.disabled = busy;
  elements.text.disabled = busy;
  elements.presetButtons.forEach((button) => { button.disabled = busy; });
}

function setLoading(loading) {
  state.loading = loading;
  elements.inputPanel.setAttribute("aria-busy", String(loading));
  elements.generateLabel.textContent = loading ? "Generating and tracing" : "Generate and trace speech";
  elements.generateLoader.classList.toggle("visible", loading);
  updateGenerateAvailability();
}

function startProgress(label = "Generating speech locally…") {
  window.clearInterval(state.progressTimer);
  state.progressStartedAt = Date.now();
  elements.progressLabel.textContent = label;
  elements.progressTime.textContent = "00:00";
  elements.progress.hidden = false;
  state.progressTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - state.progressStartedAt) / 1000);
    elements.progressTime.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }, 250);
}

function stopProgress() {
  window.clearInterval(state.progressTimer);
  state.progressTimer = null;
  elements.progress.hidden = true;
}

async function checkStatus() {
  elements.statusText.textContent = "Checking backend…";
  elements.statusDot.classList.remove("online", "offline");
  try {
    const response = await fetch(CHATTERBOX_API.status, { headers: { Accept: "application/json" } });
    const payload = await response.json();
    state.backendReady = response.ok && Boolean(payload.ready);
    state.capabilities = payload.capabilities && typeof payload.capabilities === "object" ? payload.capabilities : {};
    elements.statusDot.classList.add(state.backendReady ? "online" : "offline");
    elements.statusText.textContent = state.backendReady ? "Chatterbox ready" : "Demo only";
    elements.statusButton.setAttribute(
      "aria-label",
      `Chatterbox backend status: ${payload.message || (state.backendReady ? "ready" : "unavailable")}. Refresh status.`,
    );
  } catch (error) {
    state.backendReady = false;
    state.capabilities = {};
    elements.statusDot.classList.add("offline");
    elements.statusText.textContent = "Demo only";
    elements.statusButton.setAttribute("aria-label", `Chatterbox backend unavailable: ${error.message}. Refresh status.`);
  }
  updateGenerateAvailability();
  renderInterventionMode();
}

function validateGeneration(payload) {
  if (!payload || typeof payload !== "object") throw new Error("The Chatterbox backend returned an empty generation.");
  if (!payload.analysis_id) throw new Error("The generation is missing an analysis ID.");
  if (!payload.input || !Array.isArray(payload.input.tokens)) throw new Error("The generation is missing input text tokens.");
  if (!payload.output || !Array.isArray(payload.output.waveform) || !Array.isArray(payload.output.speech_codes)) {
    throw new Error("The generation is missing its waveform or speech-code timeline.");
  }
  if (!payload.output.speech_codes.length) throw new Error("Chatterbox generated no inspectable speech codes.");
  if (typeof payload.output.audio_data_url !== "string") throw new Error("The generation is missing playable output audio.");
  if (payload.output.speech_head_candidates !== undefined && payload.output.speech_head_candidates !== null) {
    validateSpeechHeadCandidates(payload.output.speech_head_candidates, payload.output.speech_codes);
  }
  if (payload.fitted_speech_code_jlens !== undefined && payload.fitted_speech_code_jlens !== null) {
    validateFittedSpeechLens(payload.fitted_speech_code_jlens, payload.output.speech_codes);
  }
  return payload;
}

function validateBranchGeneration(payload, parentAnalysisId, speechCodeIndex, replacementCodeId) {
  const branch = validateGeneration(payload);
  const intervention = branch.intervention;
  if (!intervention || typeof intervention !== "object") {
    throw new Error("The forced-code branch is missing intervention provenance.");
  }
  if (intervention.kind !== "forced_speech_code_autoregressive_branch") {
    throw new Error("The branch returned an unexpected intervention kind.");
  }
  if (String(intervention.parent_analysis_id) !== String(parentAnalysisId)) {
    throw new Error("The branch belongs to a different parent analysis.");
  }
  if (Number(intervention.speech_code_index) !== Number(speechCodeIndex)) {
    throw new Error("The branch returned a different forced speech position.");
  }
  if (Number(intervention.replacement_code_id) !== Number(replacementCodeId)) {
    throw new Error("The branch returned a different replacement code ID.");
  }
  if (String(branch.analysis_id) === String(parentAnalysisId)) {
    throw new Error("The forced-code branch did not create a new analysis.");
  }
  return branch;
}

function validateResidualSteeringGeneration(payload, request) {
  const branch = validateGeneration(payload);
  const intervention = branch.intervention;
  if (!intervention || typeof intervention !== "object") {
    throw new Error("The residual-steering branch is missing intervention provenance.");
  }
  if (intervention.kind !== "t3_post_block_residual_steering_branch") {
    throw new Error("The residual branch returned an unexpected intervention kind.");
  }
  if (intervention.method !== "parent_path_local_margin_gradient_calibrated") {
    throw new Error("The residual branch returned an unexpected steering method.");
  }
  if (!["succeeded", "budget_exhausted"].includes(intervention.calibration_status)
      || !Array.isArray(intervention.calibration_attempts)
      || !intervention.calibration_attempts.length) {
    throw new Error("The residual branch is missing its automatic budget-calibration result.");
  }
  if (String(intervention.parent_analysis_id) !== String(request.analysis_id)
      || Number(intervention.speech_code_index) !== Number(request.speech_code_index)
      || Number(intervention.target_code_id) !== Number(request.target_code_id)) {
    throw new Error("The residual branch does not match the requested parent, position, or target code.");
  }
  const returnedLayers = Array.isArray(intervention.layers) ? intervention.layers.map(Number) : [];
  if (returnedLayers.length !== request.layers.length
      || returnedLayers.some((layer, index) => layer !== request.layers[index])) {
    throw new Error("The residual branch returned different source layers.");
  }
  if (Number(intervention.forward_span) !== request.forward_span
      || Math.abs(Number(intervention.max_relative_residual_norm) - request.max_relative_residual_norm) > 1e-9) {
    throw new Error("The residual branch returned a different steering window or norm cap.");
  }
  const chosenNorm = Number(intervention.chosen_relative_residual_norm);
  if (!Number.isFinite(chosenNorm) || chosenNorm <= 0 || chosenNorm > request.max_relative_residual_norm + 1e-9) {
    throw new Error("The residual branch returned an invalid chosen relative residual norm.");
  }
  const provenance = {
    coordinate: "post_t3_block_output_at_speech_prediction_position",
    direction_objective: "target_raw_logit_minus_parent_strongest_non_target_raw_logit",
    direction_source: "parent_teacher_forced_path",
    future_direction_policy: "position_specific_parent_path_direction_applied_on_dynamic_branch_path",
    suffix_policy: "argmax_after_repetition_penalty_and_temperature",
  };
  Object.entries(provenance).forEach(([key, value]) => {
    if (intervention[key] !== value) throw new Error(`The residual branch returned unexpected ${key} provenance.`);
  });
  if (typeof intervention.target_became_raw_top1 !== "boolean"
      || typeof intervention.processed_greedy_equals_target !== "boolean"
      || (intervention.processed_greedy_code_id_at_anchor !== null && !Number.isInteger(Number(intervention.processed_greedy_code_id_at_anchor)))) {
    throw new Error("The residual branch is missing its raw-head or processed-greedy outcome.");
  }
  const overallSuccess = intervention.target_became_raw_top1 && intervention.processed_greedy_equals_target;
  if ((intervention.calibration_status === "succeeded") !== overallSuccess) {
    throw new Error("Residual calibration status disagrees with its raw-head and processed-greedy outcomes.");
  }
  intervention.calibration_attempts.forEach((attempt, index) => {
    const invalid = !attempt
      || Number(attempt.attempt_index) !== index
      || !Number.isFinite(Number(attempt.relative_residual_norm))
      || !Number.isFinite(Number(attempt.target_probability))
      || !Number.isFinite(Number(attempt.target_log_probability))
      || !Number.isInteger(Number(attempt.target_rank))
      || !Number.isInteger(Number(attempt.raw_top1_code_id))
      || !Number.isFinite(Number(attempt.target_logit_margin_to_strongest_other))
      || typeof attempt.target_is_raw_top1 !== "boolean"
      || (attempt.processed_greedy_code_id !== null && !Number.isInteger(Number(attempt.processed_greedy_code_id)))
      || typeof attempt.processed_greedy_equals_target !== "boolean"
      || typeof attempt.success !== "boolean"
      || attempt.success !== (attempt.target_is_raw_top1 && attempt.processed_greedy_equals_target);
    if (invalid) throw new Error(`Residual calibration attempt ${index + 1} is invalid.`);
  });
  const chosenAttempt = intervention.calibration_attempts.find((attempt) => (
    Math.abs(Number(attempt.relative_residual_norm) - chosenNorm) <= 1e-9
  ));
  const chosenProcessedId = chosenAttempt?.processed_greedy_code_id;
  const anchorProcessedId = intervention.processed_greedy_code_id_at_anchor;
  const processedIdsMatch = (chosenProcessedId === null) === (anchorProcessedId === null)
    && (chosenProcessedId === null || Number(chosenProcessedId) === Number(anchorProcessedId));
  if (!chosenAttempt
      || chosenAttempt.target_is_raw_top1 !== intervention.target_became_raw_top1
      || chosenAttempt.processed_greedy_equals_target !== intervention.processed_greedy_equals_target
      || !processedIdsMatch) {
    throw new Error("Residual calibration outcomes do not match the chosen automatic-budget attempt.");
  }
  if (!Array.isArray(intervention.coordinates) || !intervention.coordinates.length
      || intervention.coordinates.some((coordinate) => (
        !coordinate || !Number.isInteger(Number(coordinate.layer))
        || !Number.isInteger(Number(coordinate.speech_code_index))
        || !Number.isInteger(Number(coordinate.competitor_code_id))
        || !Number.isFinite(Number(coordinate.gradient_l2_norm))
        || !Number.isFinite(Number(coordinate.baseline_residual_l2_norm))
        || !Number.isFinite(Number(coordinate.applied_delta_l2_norm))
        || !Number.isFinite(Number(coordinate.applied_relative_residual_norm))
        || typeof coordinate.applied !== "boolean"
      ))) {
    throw new Error("The residual branch contains invalid edited-coordinate provenance.");
  }
  if (!Array.isArray(intervention.limitations) || intervention.limitations.some((value) => typeof value !== "string")) {
    throw new Error("The residual branch is missing its limitations.");
  }
  if (String(branch.analysis_id) === String(request.analysis_id)) {
    throw new Error("The residual intervention did not create a new analysis.");
  }

  const diagnostics = intervention.target_diagnostics;
  if (!diagnostics || typeof diagnostics !== "object") {
    throw new Error("The residual branch is missing exact target diagnostics.");
  }
  if (diagnostics.normalization !== "full_speech_head_softmax_before_generation_processors") {
    throw new Error("The residual target diagnostics use an unexpected normalization.");
  }
  if (Number(diagnostics.schema_version) !== 1) {
    throw new Error("The residual target diagnostics use an unsupported schema version.");
  }
  const positions = Array.isArray(diagnostics.positions) ? diagnostics.positions.map(Number) : [];
  if (positions.length !== request.forward_span
      || positions.some((position, index) => !Number.isInteger(position) || position !== request.speech_code_index + index)) {
    throw new Error("The residual target diagnostics have an invalid speech-position axis.");
  }
  const requestedPositions = Array.isArray(intervention.requested_positions) ? intervention.requested_positions.map(Number) : [];
  const appliedPositions = Array.isArray(intervention.applied_positions) ? intervention.applied_positions.map(Number) : [];
  if (requestedPositions.length !== positions.length
      || requestedPositions.some((position, index) => position !== positions[index])
      || appliedPositions.some((position) => !positions.includes(position))) {
    throw new Error("The residual branch returned invalid requested or applied positions.");
  }
  const positionCount = positions.length;
  const fittedLayers = Array.isArray(diagnostics.fitted_layers) ? diagnostics.fitted_layers.map(Number) : [];
  if (fittedLayers.some((layer) => !Number.isInteger(layer))) {
    throw new Error("The residual target diagnostics have invalid fitted layers.");
  }
  const expectedFittedLayers = Array.isArray(branch.fitted_speech_code_jlens?.layers)
    ? branch.fitted_speech_code_jlens.layers.map(Number)
    : [];
  if (fittedLayers.length !== expectedFittedLayers.length
      || fittedLayers.some((layer, index) => layer !== expectedFittedLayers[index])) {
    throw new Error("The residual target diagnostics do not cover every loaded fitted J-lens layer.");
  }
  const nullableProbability = (value) => value === null || (Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1);
  const nullableRank = (value) => value === null || (Number.isInteger(Number(value)) && Number(value) >= 1);
  const validateMatrix = (key, validator) => {
    const matrix = diagnostics[key];
    if (!Array.isArray(matrix) || matrix.length !== fittedLayers.length
        || matrix.some((row) => !Array.isArray(row) || row.length !== positionCount || row.some((value) => !validator(value)))) {
      throw new Error(`The residual target diagnostics have an invalid ${key} matrix.`);
    }
  };
  validateMatrix("before_probabilities", nullableProbability);
  validateMatrix("after_probabilities", nullableProbability);
  validateMatrix("before_ranks", nullableRank);
  validateMatrix("after_ranks", nullableRank);
  ["head_before_probabilities", "head_after_probabilities", "head_before_ranks", "head_after_ranks"].forEach((key) => {
    const values = diagnostics[key];
    const validator = key.includes("probabilities") ? nullableProbability : nullableRank;
    if (!Array.isArray(values) || values.length !== positionCount || values.some((value) => !validator(value))) {
      throw new Error(`The residual target diagnostics have an invalid ${key} row.`);
    }
  });
  if (!Array.isArray(diagnostics.parent_realized_ids)
      || diagnostics.parent_realized_ids.length !== positionCount
      || diagnostics.parent_realized_ids.some((value) => !Number.isInteger(Number(value)))) {
    throw new Error("The residual target diagnostics have invalid parent_realized_ids.");
  }
  if (!Array.isArray(diagnostics.branch_realized_ids)
      || diagnostics.branch_realized_ids.length !== positionCount
      || diagnostics.branch_realized_ids.some((value) => value !== null && !Number.isInteger(Number(value)))) {
    throw new Error("The residual target diagnostics have invalid branch_realized_ids.");
  }
  if (!Array.isArray(diagnostics.edited_coordinates)
      || diagnostics.edited_coordinates.some((coordinate) => (
        !coordinate || !Number.isInteger(Number(coordinate.layer))
        || !Number.isInteger(Number(coordinate.speech_code_index))
      ))) {
    throw new Error("The residual target diagnostics have invalid edited coordinates.");
  }
  const appliedCoordinates = new Set(intervention.coordinates
    .filter((coordinate) => coordinate.applied)
    .map((coordinate) => `${Number(coordinate.layer)}:${Number(coordinate.speech_code_index)}`));
  const diagnosticCoordinates = new Set(diagnostics.edited_coordinates
    .map((coordinate) => `${Number(coordinate.layer)}:${Number(coordinate.speech_code_index)}`));
  if (appliedCoordinates.size !== diagnosticCoordinates.size
      || [...appliedCoordinates].some((coordinate) => !diagnosticCoordinates.has(coordinate))) {
    throw new Error("The residual diagnostic edit mask differs from the applied residual coordinates.");
  }
  if (diagnostics.first_suffix_divergence_index !== null
      && !Number.isInteger(Number(diagnostics.first_suffix_divergence_index))) {
    throw new Error("The residual target diagnostics have an invalid suffix divergence index.");
  }
  return branch;
}

function validateSpeechHeadCandidates(head, codes) {
  if (!head || typeof head !== "object") throw new Error("The speech-head candidate payload is invalid.");
  const codeCount = codes.length;
  const vocabSize = Number(head.vocab_size);
  const topK = Number(head.top_k);
  if (!Number.isInteger(vocabSize) || vocabSize < 1) throw new Error("The speech-head candidate vocabulary size is invalid.");
  if (!Number.isInteger(topK) || topK < 1 || topK > vocabSize) throw new Error("The speech-head candidate top-k value is invalid.");
  const vectorKeys = ["target_ids", "target_probabilities", "target_log_probabilities", "target_ranks", "top_codes"];
  vectorKeys.forEach((key) => {
    if (!Array.isArray(head[key]) || head[key].length !== codeCount) {
      throw new Error(`The speech-head candidates have an invalid ${key} position dimension.`);
    }
  });
  head.target_ids.forEach((targetId, codeIndex) => {
    if (Number(targetId) !== Number(codes[codeIndex]?.id)) {
      throw new Error(`The speech-head candidate target at position ${codeIndex + 1} differs from the generated code.`);
    }
  });
  head.target_probabilities.forEach((value, codeIndex) => {
    const probability = Number(value);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error(`The speech-head target probability at position ${codeIndex + 1} is invalid.`);
    }
  });
  head.target_log_probabilities.forEach((value, codeIndex) => {
    if (!Number.isFinite(Number(value))) {
      throw new Error(`The speech-head target log probability at position ${codeIndex + 1} is invalid.`);
    }
  });
  head.target_ranks.forEach((value, codeIndex) => {
    if (!Number.isInteger(Number(value)) || Number(value) < 1 || Number(value) > vocabSize) {
      throw new Error(`The speech-head target rank at position ${codeIndex + 1} is invalid.`);
    }
  });
  head.top_codes.forEach((entries, codeIndex) => {
    const ids = Array.isArray(entries) ? entries.map((entry) => Number(entry?.id)) : [];
    const expectedCount = Math.min(topK, vocabSize);
    const invalidEntry = !Array.isArray(entries) || entries.length !== expectedCount || entries.some((entry) => (
      !entry || !Number.isInteger(Number(entry.id)) || Number(entry.id) < 0 || Number(entry.id) >= vocabSize
      || !Number.isFinite(Number(entry.probability)) || Number(entry.probability) < 0 || Number(entry.probability) > 1
      || (entry.special_token !== undefined && typeof entry.special_token !== "string")
    ));
    const unsorted = Array.isArray(entries) && entries.some((entry, index) => (
      index > 0 && Number(entry.probability) > Number(entries[index - 1].probability) + 1e-12
    ));
    const topMass = Array.isArray(entries)
      ? entries.reduce((sum, entry) => sum + Number(entry.probability), 0)
      : Infinity;
    const realizedOutside = !ids.includes(Number(head.target_ids[codeIndex]));
    const displayedMass = topMass + (realizedOutside ? Number(head.target_probabilities[codeIndex]) : 0);
    if (invalidEntry || new Set(ids).size !== ids.length || unsorted || displayedMass > 1.000001) {
      throw new Error(`The speech-head top codes at position ${codeIndex + 1} are invalid.`);
    }
  });
  ["source_coordinate", "target_head", "normalization"].forEach((key) => {
    if (typeof head[key] !== "string" || !head[key]) throw new Error(`The speech-head candidates are missing ${key}.`);
  });
  if (!head.special_token_ids || typeof head.special_token_ids !== "object") {
    throw new Error("The speech-head candidates are missing special-token IDs.");
  }
  Object.values(head.special_token_ids).forEach((value) => {
    if (!Number.isInteger(Number(value)) || Number(value) < 0 || Number(value) >= vocabSize) {
      throw new Error("The speech-head candidates contain an invalid special-token ID.");
    }
  });
  if (!Array.isArray(head.generation_processors_excluded)) {
    throw new Error("The speech-head candidates are missing generation-processor provenance.");
  }
  if (!Array.isArray(head.warnings)) throw new Error("The speech-head candidate warnings must be an array.");
  return head;
}

function validateFittedSpeechLens(lens, codes) {
  if (!lens || typeof lens !== "object") throw new Error("The fitted speech-code J-lens payload is invalid.");
  if (!Array.isArray(lens.layers) || !lens.layers.length) throw new Error("The fitted speech-code J-lens contains no source layers.");
  const codeCount = codes.length;
  if (!Array.isArray(lens.target_ids) || lens.target_ids.length !== codeCount) {
    throw new Error("The fitted speech-code J-lens target IDs do not match the generated sequence.");
  }
  lens.target_ids.forEach((targetId, codeIndex) => {
    if (Number(targetId) !== Number(codes[codeIndex]?.id)) {
      throw new Error(`The fitted speech-code J-lens target at position ${codeIndex + 1} differs from the generated code.`);
    }
  });
  const matrixKeys = ["target_probabilities", "target_log_probabilities", "target_ranks", "top_codes"];
  matrixKeys.forEach((key) => {
    if (!Array.isArray(lens[key]) || lens[key].length !== lens.layers.length) {
      throw new Error(`The fitted speech-code J-lens has an invalid ${key} layer dimension.`);
    }
    lens[key].forEach((row, layerIndex) => {
      if (!Array.isArray(row) || row.length !== codeCount) {
        throw new Error(`The fitted speech-code J-lens has an invalid ${key} row for layer ${lens.layers[layerIndex]}.`);
      }
    });
  });
  lens.target_probabilities.forEach((row, layerIndex) => row.forEach((value, codeIndex) => {
    const probability = Number(value);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error(`The fitted probability at layer ${lens.layers[layerIndex]}, position ${codeIndex + 1} is invalid.`);
    }
  }));
  lens.target_log_probabilities.forEach((row, layerIndex) => row.forEach((value, codeIndex) => {
    if (!Number.isFinite(Number(value))) {
      throw new Error(`The fitted log probability at layer ${lens.layers[layerIndex]}, position ${codeIndex + 1} is invalid.`);
    }
  }));
  lens.target_ranks.forEach((row, layerIndex) => row.forEach((value, codeIndex) => {
    if (!Number.isInteger(Number(value)) || Number(value) < 1) {
      throw new Error(`The fitted rank at layer ${lens.layers[layerIndex]}, position ${codeIndex + 1} is invalid.`);
    }
  }));
  lens.top_codes.forEach((row, layerIndex) => row.forEach((entries, codeIndex) => {
    const ids = Array.isArray(entries) ? entries.map((entry) => Number(entry?.id)) : [];
    const invalidEntry = !Array.isArray(entries) || entries.some((entry) => (
      !entry || !Number.isInteger(Number(entry.id)) || !Number.isFinite(Number(entry.probability))
      || Number(entry.probability) < 0 || Number(entry.probability) > 1
    ));
    const unsorted = Array.isArray(entries) && entries.some((entry, index) => (
      index > 0 && Number(entry.probability) > Number(entries[index - 1].probability) + 1e-12
    ));
    const topMass = Array.isArray(entries)
      ? entries.reduce((sum, entry) => sum + Number(entry.probability), 0)
      : Infinity;
    const realizedOutside = !ids.includes(Number(lens.target_ids[codeIndex]));
    const displayedMass = topMass + (realizedOutside ? Number(lens.target_probabilities[layerIndex][codeIndex]) : 0);
    if (invalidEntry || new Set(ids).size !== ids.length || unsorted || displayedMass > 1.000001) {
      throw new Error(`The fitted top codes at layer ${lens.layers[layerIndex]}, position ${codeIndex + 1} are invalid.`);
    }
  }));
  ["source_coordinate", "target_head", "normalization"].forEach((key) => {
    if (typeof lens[key] !== "string" || !lens[key]) throw new Error(`The fitted speech-code J-lens is missing ${key}.`);
  });
  if (!lens.artifact || typeof lens.artifact !== "object" || Array.isArray(lens.artifact)) {
    throw new Error("The fitted speech-code J-lens is missing artifact provenance.");
  }
  if (!Array.isArray(lens.warnings)) throw new Error("The fitted speech-code J-lens warnings must be an array.");
  return lens;
}

function validateTrace(payload) {
  if (!payload || typeof payload !== "object") throw new Error("The trace endpoint returned an empty result.");
  if (!Array.isArray(payload.layers) || !payload.layers.length) throw new Error("The trace contains no T3 layers.");
  if (!Array.isArray(payload.text_tokens) || !payload.text_tokens.length) throw new Error("The trace contains no input-text tokens.");
  if (!payload.gradient_l2 || !payload.gradient_share || !payload.attention_share) {
    throw new Error("The trace is missing gradient or self-attention diagnostics.");
  }
  const tokenCount = payload.text_tokens.length;
  ["gradient_l2", "gradient_share", "attention_share"].forEach((key) => {
    payload.layers.forEach((layer, layerIndex) => {
      const row = matrixRow(payload[key], layer, layerIndex, payload.layers);
      if (row.length !== tokenCount || row.some((value) => !Number.isFinite(Number(value)))) {
        throw new Error(`The trace has an invalid ${key} row for layer ${layer}.`);
      }
    });
  });
  return payload;
}

async function generateSpeech() {
  const text = elements.text.value.trim();
  if (!text || state.loading || state.branchLoading || state.residualLoading || !state.backendReady) return;
  state.branchController?.abort();
  state.residualController?.abort();
  clearError();
  setLoading(true);
  startProgress();
  state.generationController?.abort();
  const controller = new AbortController();
  state.generationController = controller;
  const form = new FormData();
  form.append("text", text);
  try {
    const response = await fetch(CHATTERBOX_API.generate, {
      method: "POST",
      body: form,
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    let payload;
    try { payload = await response.json(); } catch (error) { throw new Error(`The generation endpoint returned ${response.status} without JSON.`); }
    if (!response.ok) throw new Error(payload.detail || payload.message || `Generation failed with status ${response.status}.`);
    renderGeneration(validateGeneration(payload), "live");
    await selectSpeechCode(0, { seek: true, announceChange: false });
    announce("Chatterbox speech is ready. Speech code one is selected.");
  } catch (error) {
    if (error.name !== "AbortError") showError(`${error.message} The synthetic UI demo remains available.`);
  } finally {
    if (state.generationController === controller) state.generationController = null;
    stopProgress();
    setLoading(false);
  }
}

function releaseDemoAudio() {
  if (state.demoAudioUrl) URL.revokeObjectURL(state.demoAudioUrl);
  state.demoAudioUrl = "";
}

function renderChatterboxMetadata(payload, mode) {
  const model = payload.model && typeof payload.model === "object" ? payload.model : {};
  const replay = payload.replay && typeof payload.replay === "object" ? payload.replay : {};
  const output = payload.output && typeof payload.output === "object" ? payload.output : {};
  const speechLens = payload.fitted_speech_code_jlens && typeof payload.fitted_speech_code_jlens === "object"
    ? payload.fitted_speech_code_jlens
    : null;
  const rates = [
    `${finiteNumber(model.speech_code_rate_hz, 25)} Hz speech codes`,
    `${finiteNumber(model.mel_frame_rate_hz, 50)} Hz mel frames`,
    `${finiteNumber(output.sample_rate || model.sample_rate, 24000).toLocaleString()} Hz waveform`,
  ].join(" · ");
  const architecture = model.t3_layers
    ? `${model.t3_layers} layers · width ${model.t3_width ?? "?"} · ${model.attention_heads ?? "?"} heads`
    : "T3 architecture not provided";
  const outputHead = model.speech_vocab_size
    ? `${model.speech_vocab_size} logits · ${model.valid_speech_codes ?? "?"} valid speech codes`
    : "T3 speech-code head";
  const headCandidateSummary = speechHeadCandidates()
    ? `Top ${speechHeadCandidates().top_k ?? "?"} of ${speechHeadCandidates().vocab_size ?? model.speech_vocab_size ?? "?"} raw speech-head entries · full-softmax rank retained`
    : "Not provided";
  const tokenizer = model.s3_tokenizer_id
    ? `${model.s3_tokenizer_id}${model.s3_tokenizer_revision ? ` · revision ${model.s3_tokenizer_revision}` : ""}`
    : "Not provided";
  const replaySummary = replay.policy
    ? `${humanizeKey(replay.policy)}${Number.isFinite(Number(replay.max_abs_logit_error)) ? ` · max |Δ logit| ${formatMetric(replay.max_abs_logit_error)}` : ""}`
    : mode === "demo" ? "Synthetic fixture · no model replay" : "Not provided";
  const lensSummary = speechLens
    ? `${mode === "demo" ? "Synthetic / fabricated" : "Loaded"} · layers ${speechLens.layers.join(", ")} · ${compactMetadataValue(speechLens.artifact)}`
    : "Not loaded · fitted speech-position timeline unavailable";
  const lensCoordinate = speechLens
    ? `${humanizeKey(speechLens.source_coordinate)} → ${humanizeKey(speechLens.target_head)} · ${humanizeKey(speechLens.normalization)}`
    : "Not available";
  const values = [
    ["Analysis type", speechLens ? "Fitted speech-code J-lens + per-run code→text cross-Jacobian sensitivity" : "Per-run code→text cross-Jacobian sensitivity · fitted speech-code J-lens not loaded", true],
    ["Run", mode === "demo" ? "synthetic-ui-demo" : String(payload.analysis_id || "Not provided")],
    ["Model", modelLabel(payload.model), true],
    ["Model revision", model.model_revision || "Not provided"],
    ["Backend", mode === "demo" ? "Synthetic browser fixture" : `${model.backend || "mlx"} · ${model.model_family || "chatterbox"}`],
    ["Quantization", compactMetadataValue(model.quantization)],
    ["S3 tokenizer", tokenizer, true],
    ["T3 architecture", architecture, true],
    ["Output head", outputHead, true],
    ["Candidate inspector", headCandidateSummary, true],
    ["Position coordinate", rates, true],
    ["Generation policy", compactMetadataValue(model.generation), true],
    ["Replay validation", replaySummary, true],
    ["Fitted speech-code J-lens", lensSummary, true],
    ["Fitted lens coordinate", lensCoordinate, true],
    ["Runtime", compactMetadataValue(model.runtime_versions), true],
    ["Schema", payload.schema_version ?? "Not provided"],
  ];
  elements.metadataList.replaceChildren();
  values.forEach(([label, value, wide]) => {
    const wrapper = createElement("div");
    if (wide) wrapper.classList.add("metadata-wide");
    const description = createElement("dd", "", String(value));
    description.title = String(value);
    wrapper.append(createElement("dt", "", label), description);
    elements.metadataList.append(wrapper);
  });
}

function renderGeneration(payload, mode) {
  window.clearTimeout(state.traceDebounceTimer);
  state.traceDebounceTimer = null;
  state.traceController?.abort();
  state.traceController = null;
  state.mode = mode;
  state.analysis = payload;
  state.trace = null;
  state.traceCache = new Map();
  state.selectedCodeIndex = 0;
  state.selectedLensLayerIndex = 0;
  state.selectedLayerIndex = 0;
  state.selectedTextIndex = null;
  state.branchSelection = null;
  state.branchComparison = null;
  state.branchStatus = "";
  state.residualSelection = null;
  state.residualLayers = new Set();
  state.residualStartIndex = 0;
  state.residualResult = null;
  state.residualStatus = "";
  state.residualFocusLayerIndex = 0;
  state.residualFocusPosition = 0;

  if (mode !== "demo") releaseDemoAudio();
  elements.resultMode.textContent = mode === "demo" ? "Synthetic demo" : mode === "branch" ? "Forced-code branch" : mode === "residual-branch" ? "Residual-steered branch" : "Live analysis";
  elements.modelStamp.textContent = modelLabel(payload.model);
  elements.modelStamp.title = elements.modelStamp.textContent;
  renderChatterboxMetadata(payload, mode);
  elements.audio.src = payload.output.audio_data_url;
  elements.audio.load();
  elements.duration.textContent = formatTime(payload.output.duration_seconds);
  elements.currentTime.textContent = formatTime(0);
  elements.codeCount.textContent = `${payload.output.speech_codes.length} speech-code steps · 25 Hz`;
  elements.contentDuration.textContent = `${finiteNumber(payload.output.nominal_content_duration_seconds).toFixed(2)} s nominal code content`;
  elements.trailingAudio.textContent = `${finiteNumber(payload.output.trailing_audio_seconds).toFixed(2)} s trailing decoder audio`;

  renderCodeSlices();
  renderCodeSelection();
  renderFittedSpeechLens();
  renderBranchComparison();
  renderResidualResult();
  renderWarnings([
    ...(payload.warnings || []),
    ...((payload.output?.speech_head_candidates?.warnings) || []),
    ...((payload.fitted_speech_code_jlens?.warnings) || []),
    ...((payload.intervention?.limitations) || []),
  ]);
  elements.traceLoading.hidden = true;
  elements.textTokens.removeAttribute("aria-busy");
  elements.layerSelect.disabled = true;
  elements.layerSelect.replaceChildren(new Option("Waiting for trace"));
  elements.textTokens.replaceChildren();
  elements.gradientHeatmap.replaceChildren();
  elements.attentionHeatmap.replaceChildren();
  elements.matrixFocus.hidden = true;
  elements.results.hidden = false;
  drawWaveform();
  window.requestAnimationFrame(() => {
    drawWaveform();
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function codeTimelineGeometry(code, duration = Math.max(outputDuration(), 0.001)) {
  const safeDuration = Math.max(finiteNumber(duration), 0.001);
  const start = clamp(finiteNumber(code?.start_seconds) / safeDuration);
  const end = clamp(finiteNumber(code?.end_seconds, code?.start_seconds) / safeDuration);
  return {
    start,
    end,
    left: `${(start * 100).toFixed(4)}%`,
    width: `${Math.max(0.08, (end - start) * 100).toFixed(4)}%`,
  };
}

function renderCodeSlices() {
  const duration = Math.max(outputDuration(), 0.001);
  elements.codeSlices.replaceChildren();
  speechCodes().forEach((code, index) => {
    const geometry = codeTimelineGeometry(code, duration);
    const slice = createElement("span", "chatterbox-code-slice");
    slice.dataset.codeIndex = String(index);
    slice.style.left = geometry.left;
    slice.style.width = geometry.width;
    slice.style.setProperty("--code-alpha", (0.025 + clamp(code.raw_probability) * 0.13).toFixed(3));
    elements.codeSlices.append(slice);
  });
}

function renderCodeSelection() {
  const codes = speechCodes();
  if (!codes.length) return;
  const index = Math.max(0, Math.min(codes.length - 1, state.selectedCodeIndex));
  state.selectedCodeIndex = index;
  const code = codes[index];
  const duration = Math.max(outputDuration(), 0.001);
  const geometry = codeTimelineGeometry(code, duration);
  elements.waveformSelection.style.left = geometry.left;
  elements.waveformSelection.style.width = geometry.width;
  elements.codeSlices.querySelectorAll(".chatterbox-code-slice").forEach((slice, candidate) => {
    slice.classList.toggle("selected", candidate === index);
  });
  elements.selectedCode.textContent = `#${index + 1} · ID ${code.id}`;
  const melRange = Number.isInteger(Number(code.mel_start)) && Number.isInteger(Number(code.mel_end))
    ? ` · mel ${Number(code.mel_start)}–${Number(code.mel_end)}`
    : "";
  elements.selectedRange.textContent = `${finiteNumber(code.start_seconds).toFixed(2)}–${finiteNumber(code.end_seconds).toFixed(2)} s${melRange}`;
  elements.selectedProbability.textContent = formatProbability(code.raw_probability, 3);
  elements.selectedLogProbability.textContent = formatMetric(code.raw_log_probability);
  elements.waveform.setAttribute("aria-valuemin", "1");
  elements.waveform.setAttribute("aria-valuemax", String(codes.length));
  elements.waveform.setAttribute("aria-valuenow", String(index + 1));
  elements.waveform.setAttribute(
    "aria-valuetext",
    `Speech code ${index + 1} of ${codes.length}, ID ${code.id}, nominal range ${finiteNumber(code.start_seconds).toFixed(2)} to ${finiteNumber(code.end_seconds).toFixed(2)} seconds`,
  );
  updateFittedSpeechLensSelection();
  renderSpeechCandidateInspector();
}

function fittedLensCellSelector(layerIndex, codeIndex) {
  return `.chatterbox-speech-lens-cell[data-lens-layer-index="${layerIndex}"][data-code-index="${codeIndex}"]`;
}

function fittedLensColorMaximum(lens) {
  const fitted = lens.target_probabilities.flat().map((value) => clamp(value));
  const head = speechHeadCandidates();
  const actual = Array.isArray(head?.target_probabilities)
    ? head.target_probabilities.map((value) => clamp(value))
    : speechCodes().map((code) => clamp(code.raw_probability));
  return Math.max(0.000001, ...fitted, ...actual);
}

function topCodeSummary(entries) {
  if (!Array.isArray(entries) || !entries.length) return "No top-code list supplied";
  return entries.map((entry) => {
    const special = typeof entry.special_token === "string" ? entry.special_token : speechCodeSpecialLabel(entry.id);
    return `ID ${entry.id}${special ? ` (${special})` : ""} ${formatProbability(entry.probability, 2)}`;
  }).join(" · ");
}

function speechCodeSpecialLabel(codeId) {
  const specialIds = speechHeadCandidates()?.special_token_ids;
  if (!specialIds || typeof specialIds !== "object") return "";
  const match = Object.entries(specialIds).find(([, value]) => Number(value) === Number(codeId));
  return match ? String(match[0]) : "";
}

function candidateEntriesWithRealized(topCodes, realizedId, realizedProbability, realizedRank) {
  const targetId = Number(realizedId);
  const targetProbability = clamp(realizedProbability);
  const targetRank = Number(realizedRank);
  const entries = (Array.isArray(topCodes) ? topCodes : []).map((entry, index) => ({
    id: Number(entry.id),
    probability: clamp(entry.probability),
    specialToken: typeof entry.special_token === "string" ? entry.special_token : speechCodeSpecialLabel(entry.id),
    rank: index + 1,
    realized: Number(entry.id) === targetId,
    outsideTopK: false,
  }));
  const realized = entries.find((entry) => entry.realized);
  if (realized) {
    realized.probability = targetProbability;
    realized.rank = Number.isInteger(targetRank) && targetRank > 0 ? targetRank : realized.rank;
    return entries;
  }
  entries.push({
    id: targetId,
    probability: targetProbability,
    specialToken: "",
    rank: Number.isInteger(targetRank) && targetRank > 0 ? targetRank : null,
    realized: true,
    outsideTopK: true,
  });
  return entries;
}

function branchSourceLabel(kind, layer) {
  return kind === "head" ? "HEAD · parent raw output logits" : `fitted J-lens L${layer}`;
}

function interventionSelection() {
  return state.interventionMode === "residual" ? state.residualSelection : state.branchSelection;
}

function forcedBranchAvailable() {
  return Boolean(state.backendReady && state.capabilities.forced_code_branching);
}

function residualSteeringAvailable() {
  return Boolean(state.backendReady && state.capabilities.residual_code_steering);
}

function branchSelectionMatches(kind, layerIndex, codeId) {
  const selection = interventionSelection();
  return Boolean(selection
    && selection.speechCodeIndex === state.selectedCodeIndex
    && selection.kind === kind
    && selection.layerIndex === layerIndex
    && selection.codeId === Number(codeId));
}

function selectBranchCandidate({ codeId, probability, rank, kind, layer, layerIndex }) {
  if (state.mode === "demo" || state.branchLoading || state.residualLoading) return;
  if (kind === "fitted" && Number.isInteger(layerIndex)) {
    state.selectedLensLayerIndex = Math.max(0, Math.min(fittedSpeechLens().layers.length - 1, layerIndex));
    updateFittedSpeechLensSelection();
  }
  const selection = {
    codeId: Number(codeId),
    probability: clamp(probability),
    rank: Number(rank),
    kind,
    layer: kind === "fitted" && Number.isInteger(Number(layer)) ? Number(layer) : null,
    layerIndex: kind === "fitted" && Number.isInteger(layerIndex) ? layerIndex : null,
    speechCodeIndex: state.selectedCodeIndex,
  };
  state.branchSelection = { ...selection };
  state.residualSelection = { ...selection };
  state.residualStartIndex = state.selectedCodeIndex;
  if (kind === "fitted" && Number.isInteger(layer)) state.residualLayers = new Set([Number(layer)]);
  state.branchStatus = "";
  state.residualStatus = "";
  renderSpeechCandidateInspector();
  window.requestAnimationFrame(() => {
    const layerSelector = kind === "fitted" ? `[data-lens-layer-index="${layerIndex}"]` : ":not([data-lens-layer-index])";
    elements.speechCandidateRows.querySelector(`button.chatterbox-speech-candidate-choice[data-candidate-kind="${kind}"][data-code-id="${codeId}"]${layerSelector}`)?.focus();
  });
  announce(`Acoustic code ID ${codeId} selected from ${branchSourceLabel(kind, layer)}. It has not been generated. Review the separate ${state.interventionMode === "residual" ? "residual-steering" : "forced-code"} action before running the intervention.`);
}

function createSpeechCandidateRow({ layer, layerIndex, kind, topCodes, realizedId, realizedProbability, realizedRank, focused = false }) {
  const isHead = kind === "head";
  const row = createElement(
    "div",
    `chatterbox-speech-candidate-row ${isHead ? "metric-output-probability" : "metric-fitted-probability"}`,
  );
  row.setAttribute("role", "listitem");
  row.dataset.candidateKind = kind;
  if (Number.isInteger(layerIndex)) row.dataset.lensLayerIndex = String(layerIndex);
  row.classList.toggle("focused-layer", focused);
  if (focused) row.setAttribute("aria-current", "step");

  const label = createElement("div", "chatterbox-speech-candidate-layer");
  label.append(
    createElement("strong", "", isHead ? "HEAD" : `L${layer}`),
    createElement("span", "", isHead ? "Actual logits" : "Fitted readout"),
    ...(focused ? [createElement("span", "chatterbox-speech-candidate-focused", "Focused layer")] : []),
    createElement(
      "span",
      "chatterbox-speech-candidate-rank",
      Number.isInteger(Number(realizedRank)) && Number(realizedRank) > 0
        ? `Realized rank #${Number(realizedRank)}`
        : "Realized rank not supplied",
    ),
  );

  const list = createElement("ul", "chatterbox-speech-candidate-list");
  list.setAttribute(
    "aria-label",
    `${isHead ? "Actual output head" : `Fitted layer ${layer}`} top acoustic-code candidates`,
  );
  candidateEntriesWithRealized(topCodes, realizedId, realizedProbability, realizedRank).forEach((entry) => {
    const item = createElement("li", "chatterbox-speech-candidate-item");
    const candidate = createElement("button", "chatterbox-speech-candidate chatterbox-speech-candidate-choice");
    candidate.type = "button";
    candidate.dataset.codeId = String(entry.id);
    candidate.dataset.candidateKind = kind;
    if (Number.isInteger(layerIndex)) candidate.dataset.lensLayerIndex = String(layerIndex);
    candidate.style.setProperty("--candidate-probability", `${(entry.probability * 100).toFixed(6)}%`);
    candidate.classList.toggle("chatterbox-speech-candidate-realized", entry.realized);
    candidate.classList.toggle("outside-top-k", entry.outsideTopK);
    const selected = branchSelectionMatches(kind, Number.isInteger(layerIndex) ? layerIndex : null, entry.id);
    candidate.classList.toggle("selected-for-branch", selected);
    const unavailableInDemo = state.mode === "demo";
    const interventionUnavailable = state.interventionMode === "residual"
      ? !residualSteeringAvailable()
      : !forcedBranchAvailable();
    const disabledReason = entry.realized
      ? "This is already the realized code."
      : entry.specialToken
        ? "Special speech-head entries cannot be intervention targets."
        : unavailableInDemo
          ? "Interventions require a live Chatterbox generation."
          : interventionUnavailable
            ? `${state.interventionMode === "residual" ? "Residual steering" : "Forced-code branching"} is unavailable in this backend.`
          : "";
    candidate.disabled = Boolean(disabledReason) || state.branchLoading || state.residualLoading;
    if (!candidate.disabled) candidate.setAttribute("aria-pressed", String(selected));
    if (entry.realized) candidate.setAttribute("aria-current", "true");
    const candidateLabel = createElement("span", "chatterbox-speech-candidate-label");
    candidateLabel.append(
      createElement("strong", "", `ID ${entry.id}`),
      createElement("span", "", formatProbability(entry.probability, 3)),
    );
    if (entry.specialToken) candidateLabel.append(createElement("em", "", entry.specialToken));
    const rankLabel = entry.realized
      ? `${entry.outsideTopK ? "Realized · outside top list" : "Realized"}${entry.rank ? ` · rank #${entry.rank}` : ""}`
      : `Candidate rank #${entry.rank}`;
    candidate.append(
      candidateLabel,
      createElement("small", "", rankLabel),
      createElement("i", "chatterbox-speech-candidate-bar"),
    );
    candidate.setAttribute(
      "aria-label",
      `${isHead ? "Actual output head" : `Fitted layer ${layer}`}, ${rankLabel.toLowerCase()}, acoustic code ID ${entry.id}${entry.specialToken ? `, special token ${entry.specialToken}` : ""}, softmax probability ${formatProbability(entry.probability, 3)}.${disabledReason ? ` ${disabledReason}` : ` Select this candidate as the ${state.interventionMode === "residual" ? "residual-steering target" : "forced output code"}; selection alone does not run generation.`}`,
    );
    candidate.title = disabledReason || `Select ID ${entry.id} from ${branchSourceLabel(kind, layer)} for a forced output-decision branch.`;
    if (!candidate.disabled) {
      candidate.addEventListener("click", () => selectBranchCandidate({
        codeId: entry.id,
        probability: entry.probability,
        rank: entry.rank,
        kind,
        layer,
        layerIndex: Number.isInteger(layerIndex) ? layerIndex : null,
      }));
    }
    item.append(candidate);
    list.append(item);
  });
  row.append(label, list);
  return row;
}

function renderResidualLayerChoices() {
  const lens = fittedSpeechLens();
  const layers = Array.isArray(lens?.layers) ? lens.layers.map(Number).filter(Number.isInteger) : [];
  const validLayers = new Set(layers);
  state.residualLayers = new Set([...state.residualLayers].filter((layer) => validLayers.has(layer)));
  if (!state.residualLayers.size && layers.length) {
    const focused = layers[Math.max(0, Math.min(layers.length - 1, state.selectedLensLayerIndex))];
    state.residualLayers.add(focused);
  }
  const signature = layers.join(",");
  if (elements.residualSourceLayers.dataset.layerSignature !== signature) {
    elements.residualSourceLayers.dataset.layerSignature = signature;
    const legend = createElement("legend", "", "Source T3 layer(s)");
    if (!layers.length) {
      elements.residualSourceLayers.replaceChildren(legend, createElement("p", "", "A compatible fitted lens is required to choose visible source layers."));
    } else {
      const choices = createElement("div", "chatterbox-residual-layer-choices");
      layers.forEach((layer) => {
        const label = createElement("label");
        const input = createElement("input");
        input.type = "checkbox";
        input.value = String(layer);
        input.checked = state.residualLayers.has(layer);
        input.dataset.residualLayer = String(layer);
        label.append(input, createElement("span", "", `L${layer}`));
        choices.append(label);
      });
      elements.residualSourceLayers.replaceChildren(legend, choices);
    }
  }
  elements.residualSourceLayers.querySelectorAll("input[data-residual-layer]").forEach((input) => {
    input.checked = state.residualLayers.has(Number(input.value));
  });
}

function residualSteeringConfig() {
  const codes = speechCodes();
  const start = Math.max(0, Math.min(codes.length - 1, Number(elements.residualStartPosition.value) - 1 || 0));
  const maximumSpan = Math.max(1, Math.min(8, codes.length - start));
  const forwardSpan = Math.max(1, Math.min(maximumSpan, Number(elements.residualForwardSpan.value) || 1));
  const maxRelativeResidualNorm = clamp(elements.residualMaxRelativeNorm.value, 0.01, 2);
  return {
    start,
    forwardSpan,
    maxRelativeResidualNorm,
    layers: [...state.residualLayers].sort((left, right) => left - right),
  };
}

function renderResidualTimeRange(config = residualSteeringConfig()) {
  const codes = speechCodes();
  if (!codes.length) {
    elements.residualTimeRange.textContent = "Waiting for a generated speech timeline";
    return;
  }
  const startCode = codes[config.start];
  const endIndex = Math.min(codes.length - 1, config.start + config.forwardSpan - 1);
  const endCode = codes[endIndex];
  elements.residualTimeRange.textContent = `S${config.start + 1}–S${endIndex + 1} · ${finiteNumber(startCode.start_seconds).toFixed(2)}–${finiteNumber(endCode.end_seconds).toFixed(2)} s nominal speech-code time · ${config.forwardSpan} position${config.forwardSpan === 1 ? "" : "s"}`;
}

function renderResidualControls() {
  elements.residualControls.hidden = state.interventionMode !== "residual";
  if (elements.residualControls.hidden) return;
  renderResidualLayerChoices();
  const codes = speechCodes();
  state.residualStartIndex = Math.max(0, Math.min(codes.length - 1, state.residualStartIndex));
  elements.residualStartPosition.min = "1";
  elements.residualStartPosition.max = String(Math.max(1, codes.length));
  elements.residualStartPosition.value = String(state.residualStartIndex + 1);
  const maximumSpan = Math.max(1, Math.min(8, codes.length - state.residualStartIndex));
  elements.residualForwardSpan.max = String(maximumSpan);
  elements.residualForwardSpan.value = String(Math.max(1, Math.min(maximumSpan, Number(elements.residualForwardSpan.value) || 1)));
  const config = residualSteeringConfig();
  const selection = state.residualSelection;
  const available = residualSteeringAvailable() && state.mode !== "demo";
  const validSelection = Boolean(selection && Number.isInteger(selection.codeId));
  const source = validSelection ? branchSourceLabel(selection.kind, selection.layer) : "";
  elements.residualControls.setAttribute("aria-busy", String(state.residualLoading));
  elements.residualLoader.classList.toggle("visible", state.residualLoading);
  elements.residualTarget.textContent = validSelection ? `Target ID ${selection.codeId}` : "No target selected";
  if (state.mode === "demo") {
    elements.residualSelection.textContent = "Residual steering is unavailable in the synthetic demo because its candidates and residuals are fabricated.";
  } else if (!residualSteeringAvailable()) {
    elements.residualSelection.textContent = "This backend does not expose context-specific T3 residual steering.";
  } else if (!validSelection) {
    elements.residualSelection.textContent = "Choose a non-realized acoustic code above. Its row only nominates a target ID and source layer; the steering direction is computed from the parent teacher-forced raw-head margin gradient. The target is not directly forced and may never become top-1.";
  } else {
    elements.residualSelection.textContent = `ID ${selection.codeId} was nominated from ${source} at S${selection.speechCodeIndex + 1}. The intervention computes a context-specific gradient of target raw-head logit minus the strongest non-target raw-head logit on the parent teacher-forced path, then adds normalized edits after the checked T3 block(s). The target is not directly forced and may fail to become top-1.`;
  }
  renderResidualTimeRange(config);
  const disableControls = !available || state.residualLoading;
  elements.residualSourceLayers.disabled = disableControls;
  elements.residualStartPosition.disabled = disableControls;
  elements.residualForwardSpan.disabled = disableControls;
  elements.residualMaxRelativeNorm.disabled = disableControls;
  elements.residualAutoBudget.disabled = true;
  elements.residualButton.disabled = disableControls || !validSelection || !config.layers.length;
  elements.residualLabel.textContent = state.residualLoading
    ? `Steering ID ${selection?.codeId ?? "—"} from S${config.start + 1}…`
    : validSelection
      ? `Steer toward ID ${selection.codeId} from S${config.start + 1}`
      : "Choose a target code to steer";
  elements.residualStatus.textContent = state.residualLoading
    ? `Calibrating a margin-gradient edit up to ${config.maxRelativeResidualNorm.toFixed(2)}× relative residual norm, then regenerating…`
    : state.residualStatus;
}

function renderInterventionMode() {
  const demo = state.mode === "demo";
  const interventionBusy = state.branchLoading || state.residualLoading;
  elements.interventionForce.disabled = interventionBusy || (!demo && !forcedBranchAvailable());
  elements.interventionResidual.disabled = interventionBusy || (!demo && !residualSteeringAvailable());
  if (state.interventionMode === "force" && elements.interventionForce.disabled && !elements.interventionResidual.disabled) {
    state.interventionMode = "residual";
  } else if (state.interventionMode === "residual" && elements.interventionResidual.disabled && !elements.interventionForce.disabled) {
    state.interventionMode = "force";
  }
  elements.interventionForce.checked = state.interventionMode === "force";
  elements.interventionResidual.checked = state.interventionMode === "residual";
  elements.branchAction.hidden = state.interventionMode !== "force";
  renderBranchAction();
  renderResidualControls();
}

function renderBranchAction() {
  if (state.interventionMode !== "force") {
    elements.branchAction.hidden = true;
    return;
  }
  elements.branchAction.hidden = false;
  const selection = state.branchSelection;
  const codes = speechCodes();
  const codeIndex = Math.max(0, Math.min(codes.length - 1, state.selectedCodeIndex));
  const source = selection ? branchSourceLabel(selection.kind, selection.layer) : "";
  const validSelection = Boolean(selection
    && selection.speechCodeIndex === codeIndex
    && Number.isInteger(selection.codeId));
  elements.branchAction.setAttribute("aria-busy", String(state.branchLoading));
  elements.branchLoader.classList.toggle("visible", state.branchLoading);
  elements.branchStatus.textContent = state.branchLoading
    ? `Forcing acoustic code ID ${selection?.codeId ?? "—"} at S${codeIndex + 1} and regenerating the suffix…`
    : state.branchStatus;

  if (state.mode === "demo") {
    elements.branchSelection.textContent = "Forced-code branching is unavailable in the synthetic UI demo";
    elements.branchSource.textContent = "Generate live Chatterbox audio first. Demo candidate values are fabricated and cannot be replayed through the model.";
    elements.branchLabel.textContent = "Live generation required";
    elements.branchButton.disabled = true;
    return;
  }
  if (!forcedBranchAvailable()) {
    elements.branchSelection.textContent = "Forced-code branching is unavailable in this backend";
    elements.branchSource.textContent = "The candidate distribution remains inspectable, but this server cannot replay a forced output decision.";
    elements.branchLabel.textContent = "Forced branching unavailable";
    elements.branchButton.disabled = true;
    return;
  }
  if (!validSelection) {
    elements.branchSelection.textContent = "Choose a non-realized acoustic code above";
    elements.branchSource.textContent = "The selected fitted layer or HEAD row identifies a candidate. The intervention itself forces that ID at the actual T3 output decision; it does not steer a fitted residual layer. This is not residual steering.";
    elements.branchLabel.textContent = "Choose a candidate to branch";
    elements.branchButton.disabled = true;
    return;
  }

  elements.branchSelection.textContent = `ID ${selection.codeId} · ${source} · S${codeIndex + 1}`;
  elements.branchSource.textContent = `Selected at candidate rank #${selection.rank} with ${formatProbability(selection.probability, 3)} in ${source}. This source row is evidence for choosing the code; generation will force ID ${selection.codeId} at the actual T3 output decision, preserve the earlier code prefix, and autoregressively regenerate everything after it.`;
  elements.branchLabel.textContent = state.branchLoading
    ? `Regenerating from S${codeIndex + 1}…`
    : `Force ID ${selection.codeId} and regenerate from S${codeIndex + 1}`;
  elements.branchButton.disabled = state.branchLoading;
}

function renderBranchComparison() {
  const comparison = state.branchComparison;
  elements.branchComparison.hidden = !comparison;
  if (!comparison) {
    elements.originalAudio.pause();
    elements.originalAudio.removeAttribute("src");
    elements.branchIdChange.textContent = "—";
    elements.branchSummary.textContent = "—";
    return;
  }
  const { parent, intervention, source } = comparison;
  const position = Number(intervention.speech_code_index);
  const displayPosition = position + 1;
  const prefixLength = Math.max(0, Number(intervention.prefix_length) || 0);
  const suffixStart = Math.max(displayPosition + 1, Number(intervention.regenerated_suffix_start_index) + 1 || displayPosition + 1);
  const prefixLabel = prefixLength > 0 ? `S1–S${prefixLength} stayed code-for-code unchanged` : "There was no earlier speech-code prefix";
  const sourceLabel = branchSourceLabel(source.kind, source.layer);
  const rawRank = Number(intervention.replacement_global_rank);
  const rawProbability = clamp(intervention.replacement_raw_probability);
  const rawTop1Probability = clamp(intervention.raw_top1_probability);
  const minimumBias = finiteNumber(intervention.minimum_additive_bias_to_be_unique_raw_top1);
  elements.branchIdChange.textContent = `S${displayPosition} · ID ${intervention.original_realized_code_id} → ID ${intervention.replacement_code_id}`;
  elements.branchSummary.textContent = `${prefixLabel}. ID ${intervention.replacement_code_id} was forced only at S${displayPosition}; S${suffixStart} onward was not copied and was autoregressively regenerated with ${humanizeKey(intervention.suffix_policy)}. The candidate was chosen from ${sourceLabel}, but the intervention occurred at the actual output decision. This is not residual steering. At the parent raw full-softmax HEAD before generation processors, top-1 was ID ${intervention.raw_top1_code_id} at ${formatProbability(rawTop1Probability, 3)}; the replacement was rank #${rawRank} at ${formatProbability(rawProbability, 3)} and needed a minimum +${formatMetric(minimumBias)} logit bias to become the unique raw top-1 code. Parent ${intervention.parent_analysis_id} → branch ${comparison.branch.analysis_id}.`;
  elements.originalAudio.src = parent.output.audio_data_url;
  elements.originalAudio.load();
}

function residualDiagnosticRow(diagnostics, rowIndex) {
  const fittedCount = diagnostics.fitted_layers.length;
  if (rowIndex < fittedCount) {
    return {
      label: `L${diagnostics.fitted_layers[rowIndex]}`,
      layer: diagnostics.fitted_layers[rowIndex],
      beforeProbabilities: diagnostics.before_probabilities[rowIndex],
      afterProbabilities: diagnostics.after_probabilities[rowIndex],
      beforeRanks: diagnostics.before_ranks[rowIndex],
      afterRanks: diagnostics.after_ranks[rowIndex],
    };
  }
  return {
    label: "HEAD",
    layer: null,
    beforeProbabilities: diagnostics.head_before_probabilities,
    afterProbabilities: diagnostics.head_after_probabilities,
    beforeRanks: diagnostics.head_before_ranks,
    afterRanks: diagnostics.head_after_ranks,
  };
}

function residualDiagnosticValue(value, formatter) {
  return value === null || value === undefined ? "not emitted" : formatter(value);
}

function renderResidualFocus() {
  const result = state.residualResult;
  if (!result) {
    elements.residualFocus.replaceChildren();
    return;
  }
  const diagnostics = result.intervention.target_diagnostics;
  const rowCount = diagnostics.fitted_layers.length + 1;
  state.residualFocusLayerIndex = Math.max(0, Math.min(rowCount - 1, Number(state.residualFocusLayerIndex) || 0));
  state.residualFocusPosition = Math.max(0, Math.min(diagnostics.positions.length - 1, Number(state.residualFocusPosition) || 0));
  const row = residualDiagnosticRow(diagnostics, state.residualFocusLayerIndex);
  const index = state.residualFocusPosition;
  const position = diagnostics.positions[index];
  const beforeProbability = row.beforeProbabilities[index];
  const afterProbability = row.afterProbabilities[index];
  const beforeRank = row.beforeRanks[index];
  const afterRank = row.afterRanks[index];
  const probabilityDelta = beforeProbability === null || afterProbability === null
    ? "Δ unavailable"
    : `${(Number(afterProbability) - Number(beforeProbability) >= 0 ? "+" : "")}${((Number(afterProbability) - Number(beforeProbability)) * 100).toFixed(3)} pp`;
  const rankDelta = beforeRank === null || afterRank === null
    ? "rank Δ unavailable"
    : `${Number(beforeRank) - Number(afterRank) >= 0 ? "+" : ""}${Number(beforeRank) - Number(afterRank)} rank places`;
  const edited = diagnostics.edited_coordinates.some((coordinate) => (
    Number(coordinate.layer) === Number(row.layer) && Number(coordinate.speech_code_index) === position
  ));
  const parentId = diagnostics.parent_realized_ids[index];
  const branchId = diagnostics.branch_realized_ids[index];
  const divergence = diagnostics.first_suffix_divergence_index;
  elements.residualFocus.replaceChildren(
    createElement("span", "", "Focused target diagnostic"),
    createElement("strong", "", `${row.label} × S${position + 1} · target ID ${result.intervention.target_code_id}`),
    createElement("span", "", `Before ${residualDiagnosticValue(beforeProbability, (value) => formatProbability(value, 3))} · rank ${residualDiagnosticValue(beforeRank, (value) => `#${value}`)}`),
    createElement("span", "", `After ${residualDiagnosticValue(afterProbability, (value) => formatProbability(value, 3))} · rank ${residualDiagnosticValue(afterRank, (value) => `#${value}`)}`),
    createElement("span", "", `${probabilityDelta} · ${rankDelta}`),
    createElement("small", "", `${edited ? "Edited coordinate" : "Not directly edited"} · realized ${parentId === null ? "—" : `ID ${parentId}`} → ${branchId === null ? "not emitted" : `ID ${branchId}`}${divergence !== null && position >= Number(divergence) ? ` · divergent suffix from S${Number(divergence) + 1}` : ""}`),
  );
}

function updateResidualDeltaSelection(focusCell = false) {
  elements.residualDeltaMatrix.querySelectorAll("button.chatterbox-residual-delta-cell").forEach((cell) => {
    const selected = Number(cell.dataset.residualRowIndex) === state.residualFocusLayerIndex
      && Number(cell.dataset.residualPositionIndex) === state.residualFocusPosition;
    cell.classList.toggle("selected", selected);
    cell.setAttribute("aria-pressed", String(selected));
    cell.tabIndex = selected ? 0 : -1;
  });
  renderResidualFocus();
  if (focusCell) {
    window.requestAnimationFrame(() => {
      elements.residualDeltaMatrix.querySelector(
        `button[data-residual-row-index="${state.residualFocusLayerIndex}"][data-residual-position-index="${state.residualFocusPosition}"]`,
      )?.focus();
    });
  }
}

function selectResidualDeltaCell(rowIndex, positionIndex, { focusCell = false, syncTimeline = true } = {}) {
  const result = state.residualResult;
  if (!result) return;
  const diagnostics = result.intervention.target_diagnostics;
  state.residualFocusLayerIndex = Math.max(0, Math.min(diagnostics.fitted_layers.length, Number(rowIndex) || 0));
  state.residualFocusPosition = Math.max(0, Math.min(diagnostics.positions.length - 1, Number(positionIndex) || 0));
  updateResidualDeltaSelection(focusCell);
  const speechPosition = diagnostics.positions[state.residualFocusPosition];
  if (syncTimeline && speechPosition < speechCodes().length) {
    selectSpeechCode(speechPosition, { seek: true, announceChange: false });
  }
}

function renderResidualDeltaMatrix() {
  const result = state.residualResult;
  elements.residualDeltaMatrix.replaceChildren();
  if (!result) return;
  const diagnostics = result.intervention.target_diagnostics;
  const positions = diagnostics.positions;
  const edited = new Set(diagnostics.edited_coordinates.map((coordinate) => `${Number(coordinate.layer)}:${Number(coordinate.speech_code_index)}`));
  const probabilityValues = [
    ...diagnostics.before_probabilities.flat(),
    ...diagnostics.after_probabilities.flat(),
    ...diagnostics.head_before_probabilities,
    ...diagnostics.head_after_probabilities,
  ].filter((value) => value !== null).map(Number);
  const probabilityMaximum = Math.max(0.000001, ...probabilityValues);
  const chart = createElement("div", "chatterbox-residual-delta-chart");
  chart.style.setProperty("--residual-position-count", String(positions.length));
  const axis = createElement("div", "chatterbox-residual-delta-axis");
  axis.append(createElement("span", "", "Layer"));
  const axisPositions = createElement("div");
  axisPositions.style.gridTemplateColumns = `repeat(${positions.length}, minmax(0, 1fr))`;
  positions.forEach((position) => axisPositions.append(createElement("span", "", `S${position + 1}`)));
  axis.append(axisPositions);
  chart.append(axis);

  const rowCount = diagnostics.fitted_layers.length + 1;
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const values = residualDiagnosticRow(diagnostics, rowIndex);
    const row = createElement("div", `chatterbox-residual-delta-row${values.layer === null ? " head" : ""}`);
    row.append(createElement("strong", "", values.label));
    const cells = createElement("div", "chatterbox-residual-delta-cells");
    cells.style.gridTemplateColumns = `repeat(${positions.length}, minmax(0, 1fr))`;
    positions.forEach((position, positionIndex) => {
      const beforeProbability = values.beforeProbabilities[positionIndex];
      const afterProbability = values.afterProbabilities[positionIndex];
      const beforeRank = values.beforeRanks[positionIndex];
      const afterRank = values.afterRanks[positionIndex];
      const parentId = diagnostics.parent_realized_ids[positionIndex];
      const branchId = diagnostics.branch_realized_ids[positionIndex];
      const isEdited = values.layer !== null && edited.has(`${values.layer}:${position}`);
      const changed = branchId !== null && Number(parentId) !== Number(branchId);
      const divergent = diagnostics.first_suffix_divergence_index !== null
        && position >= Number(diagnostics.first_suffix_divergence_index);
      const cell = createElement("button", "chatterbox-residual-delta-cell");
      cell.type = "button";
      cell.dataset.residualRowIndex = String(rowIndex);
      cell.dataset.residualPositionIndex = String(positionIndex);
      cell.style.setProperty("--before-intensity", beforeProbability === null ? "0" : (clamp(beforeProbability) / probabilityMaximum).toFixed(6));
      cell.style.setProperty("--after-intensity", afterProbability === null ? "0" : (clamp(afterProbability) / probabilityMaximum).toFixed(6));
      cell.classList.toggle("edited", isEdited);
      cell.classList.toggle("realized-changed", changed);
      cell.classList.toggle("diverged-suffix", divergent);
      cell.classList.toggle("missing-after", afterProbability === null);
      if (beforeRank !== null && afterRank !== null) {
        cell.classList.toggle("rank-improved", Number(afterRank) < Number(beforeRank));
        cell.classList.toggle("rank-worse", Number(afterRank) > Number(beforeRank));
      }
      const description = `${values.label}, speech position S${position + 1}, target acoustic code ID ${result.intervention.target_code_id}. Before probability ${residualDiagnosticValue(beforeProbability, (value) => formatProbability(value, 3))}, rank ${residualDiagnosticValue(beforeRank, (value) => `#${value}`)}. After probability ${residualDiagnosticValue(afterProbability, (value) => formatProbability(value, 3))}, rank ${residualDiagnosticValue(afterRank, (value) => `#${value}`)}. ${isEdited ? "This residual coordinate was edited." : "This coordinate was not directly edited."} Realized code ${parentId === null ? "not available" : `ID ${parentId}`} to ${branchId === null ? "not emitted" : `ID ${branchId}`}.${divergent ? " This position is in the divergent suffix." : ""}`;
      cell.setAttribute("aria-label", description);
      cell.title = description;
      cell.addEventListener("click", () => selectResidualDeltaCell(rowIndex, positionIndex));
      cells.append(cell);
    });
    row.append(cells);
    chart.append(row);
  }

  const codeRow = createElement("div", "chatterbox-residual-code-delta-row");
  codeRow.append(createElement("strong", "", "Code Δ"));
  const codeCells = createElement("div");
  codeCells.style.gridTemplateColumns = `repeat(${positions.length}, minmax(0, 1fr))`;
  positions.forEach((position, index) => {
    const parentId = diagnostics.parent_realized_ids[index];
    const branchId = diagnostics.branch_realized_ids[index];
    const mark = createElement("span");
    mark.classList.toggle("changed", branchId !== null && Number(parentId) !== Number(branchId));
    mark.classList.toggle("missing", branchId === null);
    mark.title = `S${position + 1}: ID ${parentId} → ${branchId === null ? "not emitted" : `ID ${branchId}`}`;
    mark.setAttribute("aria-label", mark.title);
    codeCells.append(mark);
  });
  codeRow.append(codeCells);
  chart.append(codeRow);
  elements.residualDeltaMatrix.append(chart);
  updateResidualDeltaSelection();
}

function renderResidualResult() {
  const result = state.residualResult;
  elements.residualResult.hidden = !result;
  if (!result) {
    elements.residualOriginalAudio.pause();
    elements.residualOriginalAudio.removeAttribute("src");
    elements.residualDeltaMatrix.replaceChildren();
    elements.residualFocus.replaceChildren();
    return;
  }
  const { intervention, source, parent, branch } = result;
  const diagnostics = intervention.target_diagnostics;
  const chosenBudget = finiteNumber(intervention.chosen_relative_residual_norm);
  const maxBudget = finiteNumber(intervention.max_relative_residual_norm);
  const appliedPositions = Array.isArray(intervention.applied_positions) ? intervention.applied_positions : [];
  const requestedPositions = Array.isArray(intervention.requested_positions) ? intervention.requested_positions : diagnostics.positions;
  const divergence = diagnostics.first_suffix_divergence_index;
  const calibration = intervention.calibration_status === "succeeded"
    ? `Calibration succeeded after ${intervention.calibration_attempts.length} attempt${intervention.calibration_attempts.length === 1 ? "" : "s"}`
    : `Budget exhausted after ${intervention.calibration_attempts.length} attempts; the target did not reach the calibrated margin objective`;
  const rawOutcome = intervention.target_became_raw_top1 ? "became raw HEAD top-1" : "did not become raw HEAD top-1";
  const emittedOutcome = intervention.processed_greedy_equals_target
    ? "and was the processed greedy code at the anchor"
    : `and the processed greedy anchor code was ${intervention.processed_greedy_code_id_at_anchor === null ? "not emitted" : `ID ${intervention.processed_greedy_code_id_at_anchor}`}`;
  const budgetWarning = chosenBudget > 0.5
    ? " The chosen edit exceeds 0.5× relative residual norm and may be off-manifold; treat the audio and downstream diagnostics cautiously."
    : "";
  const limitationSummary = intervention.limitations.length
    ? ` Limitations: ${intervention.limitations.join(" ")}`
    : "";
  elements.residualResultTarget.textContent = `Target ID ${intervention.target_code_id}`;
  elements.residualSummary.textContent = `${calibration}. The selected target ${rawOutcome} ${emittedOutcome}. Applied ${appliedPositions.length} of ${requestedPositions.length} requested position edits across L${intervention.layers.join(", L")} at ${chosenBudget.toFixed(3)}× relative residual norm (cap ${maxBudget.toFixed(3)}×). ${divergence === null ? "No realized-code divergence was observed inside the requested diagnostic window." : `The realized suffix first diverged at S${Number(divergence) + 1}.`} The candidate came from ${branchSourceLabel(source.kind, source.layer)}, which only nominated the target/layer; directions were parent-path raw-logit-margin gradients applied open-loop at future dynamic-branch positions. Parent ${intervention.parent_analysis_id} → branch ${branch.analysis_id}.${budgetWarning}${limitationSummary}`;
  elements.residualOriginalAudio.src = parent.output.audio_data_url;
  elements.residualOriginalAudio.load();
  renderResidualDeltaMatrix();
}

async function runResidualSteering() {
  const selection = state.residualSelection;
  const parent = state.analysis;
  if (!selection || !parent || state.residualLoading || state.mode === "demo" || !residualSteeringAvailable()) return;
  const config = residualSteeringConfig();
  if (!config.layers.length) {
    state.residualStatus = "Select at least one fitted T3 source layer.";
    renderResidualControls();
    return;
  }
  const request = {
    analysis_id: parent.analysis_id,
    speech_code_index: config.start,
    target_code_id: selection.codeId,
    layers: config.layers,
    forward_span: config.forwardSpan,
    max_relative_residual_norm: config.maxRelativeResidualNorm,
  };
  const source = { ...selection };
  state.residualController?.abort();
  const controller = new AbortController();
  state.residualController = controller;
  state.residualLoading = true;
  state.residualStatus = "";
  state.residualResult = null;
  state.branchComparison = null;
  renderResidualResult();
  renderBranchComparison();
  clearError();
  updateGenerateAvailability();
  renderSpeechCandidateInspector();
  try {
    const response = await fetch(CHATTERBOX_API.residualBranch, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    let payload;
    try { payload = await response.json(); } catch (error) { throw new Error(`The residual branch endpoint returned ${response.status} without JSON.`); }
    if (!response.ok) throw new Error(payload.detail || payload.message || `Residual steering failed with status ${response.status}.`);
    const branch = validateResidualSteeringGeneration(payload, request);
    const result = { parent, branch, source, intervention: branch.intervention };
    renderGeneration(branch, "residual-branch");
    state.interventionMode = "residual";
    state.residualLayers = new Set(request.layers);
    state.residualStartIndex = request.speech_code_index;
    elements.residualForwardSpan.value = String(request.forward_span);
    elements.residualMaxRelativeNorm.value = request.max_relative_residual_norm.toFixed(2);
    state.residualResult = result;
    state.residualFocusPosition = 0;
    const sourceLayerIndex = branch.intervention.target_diagnostics.fitted_layers.indexOf(source.layer);
    state.residualFocusLayerIndex = sourceLayerIndex >= 0 ? sourceLayerIndex : branch.intervention.target_diagnostics.fitted_layers.length;
    state.residualStatus = branch.intervention.calibration_status === "succeeded"
      ? `Residual branch ready at ${finiteNumber(branch.intervention.chosen_relative_residual_norm).toFixed(3)}× relative norm.`
      : `Residual branch ready, but the automatic search exhausted the ${finiteNumber(branch.intervention.max_relative_residual_norm).toFixed(3)}× budget before reaching its margin objective.`;
    renderResidualResult();
    await selectSpeechCode(request.speech_code_index, { seek: true, announceChange: false });
    announce(`Residual-steering branch ready for target acoustic code ID ${request.target_code_id}. ${state.residualStatus}`);
  } catch (error) {
    if (error.name !== "AbortError") {
      state.residualStatus = `Residual steering failed: ${error.message}`;
      showError(`Could not create the residual-steering branch: ${error.message}`);
    }
  } finally {
    if (state.residualController === controller) {
      state.residualController = null;
      state.residualLoading = false;
      updateGenerateAvailability();
      renderSpeechCandidateInspector();
      renderResidualResult();
    }
  }
}

async function branchSpeechFromCandidate() {
  const selection = state.branchSelection;
  const parent = state.analysis;
  if (!selection || !parent || state.branchLoading || state.residualLoading || state.mode === "demo") return;
  if (selection.speechCodeIndex !== state.selectedCodeIndex) {
    state.branchSelection = null;
    state.branchStatus = "The speech position changed. Choose a candidate at the current position before branching.";
    renderSpeechCandidateInspector();
    return;
  }

  const request = {
    analysis_id: parent.analysis_id,
    speech_code_index: selection.speechCodeIndex,
    replacement_code_id: selection.codeId,
  };
  const source = { ...selection };
  state.branchController?.abort();
  const controller = new AbortController();
  state.branchController = controller;
  state.branchLoading = true;
  state.branchStatus = "";
  state.branchComparison = null;
  state.residualResult = null;
  renderBranchComparison();
  renderResidualResult();
  clearError();
  updateGenerateAvailability();
  renderSpeechCandidateInspector();
  try {
    const response = await fetch(CHATTERBOX_API.branch, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    let payload;
    try { payload = await response.json(); } catch (error) { throw new Error(`The branch endpoint returned ${response.status} without JSON.`); }
    if (!response.ok) throw new Error(payload.detail || payload.message || `Forced-code branching failed with status ${response.status}.`);
    const branch = validateBranchGeneration(payload, request.analysis_id, request.speech_code_index, request.replacement_code_id);
    const comparison = { parent, branch, source, intervention: branch.intervention };
    renderGeneration(branch, "branch");
    state.branchComparison = comparison;
    state.branchStatus = `Branch ready. S${request.speech_code_index + 1} is focused in the regenerated sequence.`;
    renderBranchComparison();
    await selectSpeechCode(request.speech_code_index, { seek: true, announceChange: false });
    announce(`Forced-code branch ready. Speech position ${request.speech_code_index + 1} changed from ID ${branch.intervention.original_realized_code_id} to ID ${branch.intervention.replacement_code_id}; the suffix was regenerated.`);
  } catch (error) {
    if (error.name !== "AbortError") {
      state.branchStatus = `Branch failed: ${error.message}`;
      showError(`Could not create the forced-code branch: ${error.message}`);
    }
  } finally {
    if (state.branchController === controller) {
      state.branchController = null;
      state.branchLoading = false;
      updateGenerateAvailability();
      renderSpeechCandidateInspector();
    }
  }
}

function renderSpeechCandidateInspector() {
  const lens = fittedSpeechLens();
  const head = speechHeadCandidates();
  const codes = speechCodes();
  elements.speechCandidateRows.replaceChildren();
  elements.speechCandidateInspector.hidden = !lens || !codes.length;
  if (!lens || !codes.length) {
    state.branchSelection = null;
    state.residualSelection = null;
    renderInterventionMode();
    return;
  }

  const codeIndex = Math.max(0, Math.min(codes.length - 1, state.selectedCodeIndex));
  const code = codes[codeIndex];
  const fittedTopK = Math.max(0, ...lens.top_codes.map((row) => row[codeIndex]?.length || 0));
  const topK = Math.max(fittedTopK, Number(head?.top_k) || 0);
  const vocabSize = Number(head?.vocab_size) || Number(state.analysis?.model?.speech_vocab_size) || 0;
  const vocabularyLabel = vocabSize > 0 ? `${vocabSize.toLocaleString()}-entry` : "full";
  elements.speechCandidateContext.textContent = `Speech position S${codeIndex + 1} · ${finiteNumber(code.start_seconds).toFixed(2)}–${finiteNumber(code.end_seconds).toFixed(2)} s · realized acoustic code ID ${code.id}. Each percentage is normalized over the ${vocabularyLabel} speech head. Rows show the top ${topK || "available"} candidates plus the realized code when it falls outside that list; bars use an absolute 0–100% scale.`;

  lens.layers.forEach((layer, layerIndex) => {
    elements.speechCandidateRows.append(createSpeechCandidateRow({
      layer,
      layerIndex,
      kind: "fitted",
      topCodes: lens.top_codes[layerIndex][codeIndex],
      realizedId: code.id,
      realizedProbability: lens.target_probabilities[layerIndex][codeIndex],
      realizedRank: lens.target_ranks[layerIndex][codeIndex],
      focused: layerIndex === state.selectedLensLayerIndex,
    }));
  });

  elements.speechCandidateRows.append(createSpeechCandidateRow({
    layer: null,
    layerIndex: null,
    kind: "head",
    topCodes: head?.top_codes?.[codeIndex] || [],
    realizedId: code.id,
    realizedProbability: head?.target_probabilities?.[codeIndex] ?? code.raw_probability,
    realizedRank: head?.target_ranks?.[codeIndex],
  }));
  renderInterventionMode();
}

function renderFittedSpeechLens() {
  const lens = fittedSpeechLens();
  const head = speechHeadCandidates();
  elements.speechLensHeatmap.replaceChildren();
  elements.speechLensEmpty.hidden = Boolean(lens);
  elements.speechLensHeatmap.hidden = !lens;
  elements.speechLensFocus.hidden = true;
  elements.speechLensNote.hidden = !lens;
  if (!lens) return;

  state.selectedLensLayerIndex = Math.max(0, Math.min(lens.layers.length - 1, state.selectedLensLayerIndex));
  const codes = speechCodes();
  const maximum = fittedLensColorMaximum(lens);
  const duration = Math.max(outputDuration(), 0.001);
  const lastGeometry = codeTimelineGeometry(codes[codes.length - 1], duration);

  const heading = createElement("div", "chatterbox-heatmap-heading");
  const copy = createElement("div");
  copy.append(
    createElement("strong", "", "Probability assigned to each realized speech code"),
    createElement("span", "", "Blue rows are fitted intermediate readouts; the orange BASE HEAD row is the actual model output."),
  );
  const scale = createElement("div", "chatterbox-heatmap-scale chatterbox-speech-lens-scale");
  scale.append(
    createElement("span", "", "0%"),
    createElement("i", "metric-fitted-probability"),
    createElement("span", "", formatProbability(maximum, 1)),
  );
  scale.setAttribute("aria-label", `Shared color scale for this run, zero to ${formatProbability(maximum, 2)}.`);
  heading.append(copy, scale);
  elements.speechLensHeatmap.append(heading);

  const chart = createElement("div", "chatterbox-speech-lens-chart");
  chart.style.setProperty("--speech-content-end", `${(lastGeometry.end * 100).toFixed(4)}%`);
  const axis = createElement("div", "chatterbox-speech-lens-axis");
  axis.append(createElement("span", "chatterbox-speech-lens-strip-label", "TIME"));
  const axisTrack = createElement("div", "chatterbox-speech-lens-axis-track");
  axisTrack.append(
    createElement("span", "", "0.00 s"),
    createElement("span", "", `${(duration / 2).toFixed(2)} s`),
    createElement("span", "", `${duration.toFixed(2)} s`),
  );
  const contentMarker = createElement("i", "chatterbox-speech-lens-content-marker");
  contentMarker.style.left = `${(lastGeometry.end * 100).toFixed(4)}%`;
  contentMarker.title = `Generated speech codes end at ${finiteNumber(codes[codes.length - 1].end_seconds).toFixed(2)} seconds; the hatched remainder is trailing decoded audio.`;
  axisTrack.append(contentMarker);
  axis.append(axisTrack);
  chart.append(axis);

  const bandStart = 0;
  const bandCodes = codes;
  lens.layers.forEach((layer, layerIndex) => {
    const probabilities = lens.target_probabilities[layerIndex];
    const logProbabilities = lens.target_log_probabilities[layerIndex];
    const ranks = lens.target_ranks[layerIndex];
    const row = createElement("div", "chatterbox-speech-lens-strip-row chatterbox-speech-lens-layer-row");
    const layerLabel = createElement("span", "chatterbox-speech-lens-strip-label", `L${layer}`);
    layerLabel.title = `Fitted J-lens source · ${humanizeKey(lens.source_coordinate)} L${layer}`;
    const track = createElement("div", "chatterbox-speech-lens-strip-track");
    bandCodes.forEach((code, bandIndex) => {
      const codeIndex = bandStart + bandIndex;
      const geometry = codeTimelineGeometry(code, duration);
      const probability = clamp(probabilities[codeIndex]);
      const intensity = clamp(probability / maximum);
      const rank = Number(ranks[codeIndex]);
      const topCodes = lens.top_codes[layerIndex][codeIndex];
      const selected = layerIndex === state.selectedLensLayerIndex && codeIndex === state.selectedCodeIndex;
      const cell = createElement("button", "chatterbox-speech-lens-strip-cell chatterbox-speech-lens-cell metric-fitted-probability");
      cell.type = "button";
      cell.dataset.lensLayerIndex = String(layerIndex);
      cell.dataset.codeIndex = String(codeIndex);
      cell.style.left = geometry.left;
      cell.style.width = geometry.width;
      cell.style.setProperty("--heatmap-value", intensity.toFixed(6));
      cell.classList.toggle("selected", selected);
      cell.classList.toggle("selected-column", codeIndex === state.selectedCodeIndex);
      cell.setAttribute("aria-pressed", String(selected));
      cell.tabIndex = selected ? 0 : -1;
      const description = `Fitted J-lens, post-block layer ${layer}, speech position ${codeIndex + 1}, realized code ID ${code.id}, nominal ${finiteNumber(code.start_seconds).toFixed(2)} to ${finiteNumber(code.end_seconds).toFixed(2)} seconds: probability ${formatProbability(probability, 3)}, log probability ${formatMetric(logProbabilities[codeIndex])}, rank ${rank}. Top fitted codes: ${topCodeSummary(topCodes)}.`;
      cell.setAttribute("aria-label", description);
      cell.title = description;
      cell.addEventListener("click", () => selectFittedSpeechLensCell(layerIndex, codeIndex, true, true));
      track.append(cell);
    });
    row.append(track, layerLabel);
    chart.append(row);
  });

  const baseRow = createElement("div", "chatterbox-speech-lens-strip-row chatterbox-speech-lens-base-row");
  const baseLabel = createElement("span", "chatterbox-speech-lens-strip-label", "HEAD");
  baseLabel.title = "Base model output head · actual final T3 logits · not fitted";
  const baseTrack = createElement("div", "chatterbox-speech-lens-strip-track");
  bandCodes.forEach((code, bandIndex) => {
    const codeIndex = bandStart + bandIndex;
    const geometry = codeTimelineGeometry(code, duration);
    const probability = clamp(head?.target_probabilities?.[codeIndex] ?? code.raw_probability);
    const intensity = clamp(probability / maximum);
    const cell = createElement("button", "chatterbox-speech-lens-strip-cell chatterbox-speech-lens-reference-cell metric-output-probability");
    cell.type = "button";
    cell.dataset.codeIndex = String(codeIndex);
    cell.style.left = geometry.left;
    cell.style.width = geometry.width;
    cell.style.setProperty("--heatmap-value", intensity.toFixed(6));
    cell.classList.toggle("selected-column", codeIndex === state.selectedCodeIndex);
    cell.tabIndex = -1;
    const rank = Number(head?.target_ranks?.[codeIndex]);
    const rankSummary = Number.isInteger(rank) && rank > 0 ? `, rank ${rank}` : "";
    const topSummary = head?.top_codes?.[codeIndex]
      ? ` Top actual codes: ${topCodeSummary(head.top_codes[codeIndex])}.`
      : "";
    const description = `Base model output-head probability, speech position ${codeIndex + 1}, realized code ID ${code.id}, nominal ${finiteNumber(code.start_seconds).toFixed(2)} to ${finiteNumber(code.end_seconds).toFixed(2)} seconds: ${formatProbability(probability, 3)}${rankSummary} from the actual final logits, before generation processors.${topSummary}`;
    cell.setAttribute("aria-label", description);
    cell.title = description;
    cell.addEventListener("click", () => selectSpeechCode(codeIndex, { seek: true }));
    baseTrack.append(cell);
  });
  baseRow.append(baseTrack, baseLabel);
  chart.append(baseRow);
  elements.speechLensHeatmap.append(chart);

  const artifact = lens.artifact || {};
  const artifactLabel = artifact.fingerprint || artifact.id || artifact.name || artifact.format || "see Run details";
  elements.speechLensNote.textContent = `Source ${humanizeKey(lens.source_coordinate)} → ${humanizeKey(lens.target_head)}. Normalization: ${humanizeKey(lens.normalization)}. Artifact: ${artifactLabel}. Blue and orange share a 0–${formatProbability(maximum, 1)} color scale for this run; exact values are available on focus, and the colors are not comparable across runs. Every strip uses the waveform’s full ${duration.toFixed(2)} s geometry, so the hatched region after ${(lastGeometry.end * 100).toFixed(1)}% is trailing decoded audio with no T3 speech code. A fitted probability is not the base model’s actual intermediate confidence.`;
  updateFittedSpeechLensSelection();
}

function renderFittedSpeechLensFocus() {
  const lens = fittedSpeechLens();
  const head = speechHeadCandidates();
  const codes = speechCodes();
  if (!lens || !codes.length) {
    elements.speechLensFocus.hidden = true;
    return;
  }
  const layerIndex = Math.max(0, Math.min(lens.layers.length - 1, state.selectedLensLayerIndex));
  const codeIndex = Math.max(0, Math.min(codes.length - 1, state.selectedCodeIndex));
  const layer = lens.layers[layerIndex];
  const code = codes[codeIndex];
  const fitted = clamp(lens.target_probabilities[layerIndex][codeIndex]);
  const fittedLog = finiteNumber(lens.target_log_probabilities[layerIndex][codeIndex]);
  const actual = clamp(head?.target_probabilities?.[codeIndex] ?? code.raw_probability);
  const rank = Number(lens.target_ranks[layerIndex][codeIndex]);
  const deltaPoints = (fitted - actual) * 100;
  const delta = `${deltaPoints >= 0 ? "+" : ""}${deltaPoints.toFixed(2)} pp`;
  elements.speechLensFocus.replaceChildren(
    createElement("span", "", "Selected fitted coordinate"),
    createElement("strong", "", `L${layer} × S${codeIndex + 1} · ID ${code.id}`),
    createElement("span", "metric-fitted-probability", `Fitted ${formatProbability(fitted, 3)}`),
    createElement("span", "metric-output-probability", `Base head ${formatProbability(actual, 3)}`),
    createElement("span", "chatterbox-speech-lens-rank", `Rank #${rank} · Δ ${delta}`),
    createElement("small", "", `Fitted ln p ${formatMetric(fittedLog)} · top fitted codes: ${topCodeSummary(lens.top_codes[layerIndex][codeIndex])}`),
  );
  elements.speechLensFocus.hidden = false;
}

function updateFittedSpeechLensSelection() {
  const lens = fittedSpeechLens();
  if (!lens || elements.speechLensHeatmap.hidden) return;
  elements.speechLensHeatmap.querySelectorAll("button.chatterbox-speech-lens-cell").forEach((cell) => {
    const selected = Number(cell.dataset.lensLayerIndex) === state.selectedLensLayerIndex
      && Number(cell.dataset.codeIndex) === state.selectedCodeIndex;
    cell.classList.toggle("selected", selected);
    cell.classList.toggle("selected-column", Number(cell.dataset.codeIndex) === state.selectedCodeIndex);
    cell.setAttribute("aria-pressed", String(selected));
    cell.tabIndex = selected ? 0 : -1;
  });
  elements.speechLensHeatmap.querySelectorAll("button.chatterbox-speech-lens-reference-cell").forEach((cell) => {
    cell.classList.toggle("selected-column", Number(cell.dataset.codeIndex) === state.selectedCodeIndex);
  });
  renderFittedSpeechLensFocus();
}

function selectFittedSpeechLensCell(layerIndex, codeIndex, announceChange = false, focusCell = false) {
  const lens = fittedSpeechLens();
  if (!lens) return;
  state.selectedLensLayerIndex = Math.max(0, Math.min(lens.layers.length - 1, layerIndex));
  selectSpeechCode(codeIndex, { seek: true, announceChange: false });
  updateFittedSpeechLensSelection();
  if (announceChange) {
    announce(`Fitted J-lens layer ${lens.layers[state.selectedLensLayerIndex]}, speech position ${state.selectedCodeIndex + 1} selected.`);
  }
  if (focusCell) {
    window.requestAnimationFrame(() => {
      const cell = elements.speechLensHeatmap.querySelector(fittedLensCellSelector(state.selectedLensLayerIndex, state.selectedCodeIndex));
      cell?.focus();
      cell?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  }
}

function codeIndexAtTime(seconds) {
  const codes = speechCodes();
  if (!codes.length) return 0;
  const time = Math.max(0, finiteNumber(seconds));
  const containing = codes.findIndex((code) => time >= finiteNumber(code.start_seconds) && time < finiteNumber(code.end_seconds));
  if (containing >= 0) return containing;
  let nearest = 0;
  let distance = Infinity;
  codes.forEach((code, index) => {
    const midpoint = (finiteNumber(code.start_seconds) + finiteNumber(code.end_seconds)) / 2;
    if (Math.abs(midpoint - time) < distance) {
      distance = Math.abs(midpoint - time);
      nearest = index;
    }
  });
  return nearest;
}

async function selectSpeechCode(index, { seek = false, announceChange = true } = {}) {
  const codes = speechCodes();
  if (!codes.length) return;
  window.clearTimeout(state.traceDebounceTimer);
  state.traceDebounceTimer = null;
  state.traceController?.abort();
  const nextCodeIndex = Math.max(0, Math.min(codes.length - 1, Number(index) || 0));
  if (nextCodeIndex !== state.selectedCodeIndex && !state.branchLoading && !state.residualLoading) {
    state.branchSelection = null;
    state.residualSelection = null;
    state.branchStatus = "";
    state.residualStatus = "";
    state.residualStartIndex = nextCodeIndex;
  }
  state.selectedCodeIndex = nextCodeIndex;
  state.selectedTextIndex = null;
  renderCodeSelection();
  if (seek) {
    const time = finiteNumber(codes[state.selectedCodeIndex].start_seconds);
    try { elements.audio.currentTime = Math.min(time, finiteNumber(elements.audio.duration, time)); } catch (error) { /* Metadata may not be loaded yet. */ }
    updatePlayback();
  }
  if (announceChange) announce(`Speech code ${state.selectedCodeIndex + 1} selected. Loading its text-context trace.`);

  if (state.traceCache.has(state.selectedCodeIndex)) {
    applyTrace(state.traceCache.get(state.selectedCodeIndex));
    return;
  }
  if (state.mode === "demo") {
    const trace = createDemoTrace(state.analysis, state.selectedCodeIndex);
    state.traceCache.set(state.selectedCodeIndex, trace);
    applyTrace(trace);
    return;
  }
  clearTraceViewForSelection();
  const codeIndex = state.selectedCodeIndex;
  state.traceDebounceTimer = window.setTimeout(() => {
    state.traceDebounceTimer = null;
    requestTrace(codeIndex);
  }, 180);
}

function clearTraceViewForSelection() {
  state.trace = null;
  state.selectedTextIndex = null;
  elements.layerSelect.disabled = true;
  elements.layerSelect.replaceChildren(new Option(`Tracing code ${state.selectedCodeIndex + 1}…`));
  elements.traceLoadingLabel.textContent = `Preparing speech code ${state.selectedCodeIndex + 1}…`;
  elements.traceLoading.hidden = false;
  elements.textTokens.setAttribute("aria-busy", "true");
  elements.textTokens.replaceChildren();
  elements.gradientHeatmap.replaceChildren();
  elements.attentionHeatmap.replaceChildren();
  elements.matrixFocus.hidden = true;
  elements.inspectorTitle.textContent = "Trace in progress";
  elements.inspectorContext.textContent = `Waiting for speech code ${state.selectedCodeIndex + 1}; values from the previous code have been cleared.`;
  [
    elements.inspectorLayer,
    elements.inspectorToken,
    elements.inspectorGradient,
    elements.inspectorGradientShare,
    elements.inspectorGradientMass,
    elements.inspectorAttention,
    elements.inspectorTextMass,
  ].forEach((element) => { element.textContent = "—"; });
  renderWarnings([
    ...(state.analysis?.warnings || []),
    ...(state.analysis?.output?.speech_head_candidates?.warnings || []),
    ...(state.analysis?.fitted_speech_code_jlens?.warnings || []),
    ...(state.analysis?.intervention?.limitations || []),
  ]);
}

async function requestTrace(codeIndex) {
  state.traceController?.abort();
  const controller = new AbortController();
  state.traceController = controller;
  elements.traceLoadingLabel.textContent = `Tracing speech code ${codeIndex + 1} through T3…`;
  elements.traceLoading.hidden = false;
  elements.textTokens.setAttribute("aria-busy", "true");
  try {
    const response = await fetch(CHATTERBOX_API.trace, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysis.analysis_id,
        speech_code_index: codeIndex,
      }),
      signal: controller.signal,
    });
    let payload;
    try { payload = await response.json(); } catch (error) { throw new Error(`The trace endpoint returned ${response.status} without JSON.`); }
    if (!response.ok) throw new Error(payload.detail || payload.message || `Trace failed with status ${response.status}.`);
    const trace = validateTrace(payload);
    if (trace.analysis_id && trace.analysis_id !== state.analysis.analysis_id) {
      throw new Error("The trace belongs to a different Chatterbox analysis.");
    }
    if (Number(trace.selection?.speech_code_index) !== codeIndex) {
      throw new Error("The trace returned a different speech-code selection.");
    }
    state.traceCache.set(codeIndex, trace);
    if (state.selectedCodeIndex === codeIndex) applyTrace(trace);
  } catch (error) {
    if (error.name !== "AbortError") {
      showError(`Could not trace speech code ${codeIndex + 1}: ${error.message}`);
      if (state.selectedCodeIndex === codeIndex) {
        elements.inspectorTitle.textContent = "Trace unavailable";
        elements.inspectorContext.textContent = `Speech code ${codeIndex + 1} could not be traced. Select it again to retry.`;
      }
    }
  } finally {
    if (state.traceController === controller) {
      state.traceController = null;
      elements.traceLoading.hidden = true;
      elements.textTokens.removeAttribute("aria-busy");
    }
  }
}

function applyTrace(trace) {
  state.trace = trace;
  const layers = traceLayers();
  const previousLayer = Number(elements.layerSelect.value);
  const retainedIndex = layers.findIndex((layer) => Number(layer) === previousLayer);
  state.selectedLayerIndex = retainedIndex >= 0 ? retainedIndex : Math.max(0, layers.length - 1);
  elements.layerSelect.replaceChildren(...layers.map((layer) => new Option(`T3 L${layer} · gradient post-block + attention block`, String(layer))));
  elements.layerSelect.value = String(layers[state.selectedLayerIndex]);
  elements.layerSelect.disabled = false;
  elements.traceLoading.hidden = true;
  elements.textTokens.removeAttribute("aria-busy");
  const attentionKind = trace.attention_kind
    ? ` Self-attention diagnostic: ${String(trace.attention_kind).replace(/_/g, " ")}.`
    : "";
  elements.scoreKind.textContent = `${humanizeScoreKind(trace.score_kind)}${attentionKind} Raw gradient norms remain unnormalized; blue token fills and matrix cells show within-text normalized shares. Nominal code timing does not isolate an exact waveform cause.`;
  renderTrace();
  renderWarnings([
    ...(state.analysis.warnings || []),
    ...(state.analysis.output?.speech_head_candidates?.warnings || []),
    ...(state.analysis.fitted_speech_code_jlens?.warnings || []),
    ...(state.analysis.intervention?.limitations || []),
    ...(trace.warnings || []),
  ]);
  announce(`Trace ready for speech code ${state.selectedCodeIndex + 1}. T3 post-block layer ${layers[state.selectedLayerIndex]} is selected.`);
}

function humanizeScoreKind(scoreKind) {
  const raw = String(scoreKind || "selected speech-code raw log-probability gradient");
  if (raw.includes("raw_log_probability_gradient_l2")) {
    return `${raw.startsWith("synthetic_") ? "Synthetic " : ""}raw local sensitivity ‖∇H log p(selected speech code)‖₂.`;
  }
  return `${raw.replace(/_/g, " ").replace(/^./, (character) => character.toUpperCase())}.`;
}

function renderTrace() {
  if (!state.trace) return;
  const layers = traceLayers();
  if (!layers.length) return;
  state.selectedLayerIndex = Math.max(0, Math.min(layers.length - 1, state.selectedLayerIndex));
  const layer = layers[state.selectedLayerIndex];
  elements.layerSelect.value = String(layer);
  const gradientRow = matrixRow(state.trace.gradient_l2, layer, state.selectedLayerIndex);
  if (state.selectedTextIndex === null || state.selectedTextIndex >= traceTextTokens().length) {
    state.selectedTextIndex = indexOfMaximum(gradientRow);
  }
  renderTextTokens();
  renderLayerMatrices();
  renderInspector();
}

function renderTextTokens() {
  const layers = traceLayers();
  const layer = layers[state.selectedLayerIndex];
  const tokens = traceTextTokens();
  const gradientRaw = matrixRow(state.trace.gradient_l2, layer, state.selectedLayerIndex);
  const gradientShare = matrixRow(state.trace.gradient_share, layer, state.selectedLayerIndex);
  const attentionShare = matrixRow(state.trace.attention_share, layer, state.selectedLayerIndex);
  elements.textTokens.replaceChildren();
  tokens.forEach((token, index) => {
    const gradient = Math.max(0, finiteNumber(gradientShare[index]));
    const attention = Math.max(0, finiteNumber(attentionShare[index]));
    const button = createElement("button", "chatterbox-text-token");
    button.type = "button";
    button.dataset.textTokenIndex = String(index);
    button.classList.toggle("selected", index === state.selectedTextIndex);
    button.setAttribute("aria-pressed", String(index === state.selectedTextIndex));
    button.style.setProperty("--gradient-width", `${(clamp(gradient) * 100).toFixed(2)}%`);
    button.style.setProperty("--attention-width", `${(clamp(attention) * 100).toFixed(2)}%`);
    const label = createElement("span", "chatterbox-token-text", visibleToken(token.text));
    const metric = createElement("small", "", `g ${formatProbability(gradient, 1)} · a ${formatProbability(attention, 1)}`);
    button.append(label, metric);
    button.setAttribute(
      "aria-label",
      `Input token ${index + 1}, ${visibleToken(token.text)}. Within-text gradient share ${formatProbability(gradient)}. Within-text self-attention share ${formatProbability(attention)}. Raw local gradient norm ${formatMetric(gradientRaw[index])}.`,
    );
    button.addEventListener("click", () => {
      state.selectedTextIndex = index;
      renderTrace();
      window.requestAnimationFrame(() => elements.textTokens.querySelector(`[data-text-token-index="${index}"]`)?.focus());
      announce(`Text token ${visibleToken(token.text)} selected at layer ${layer}.`);
    });
    elements.textTokens.append(button);
  });
}

function heatmapBandSize() {
  const width = elements.layerMatrices.getBoundingClientRect().width || window.innerWidth;
  if (width < 480) return 4;
  if (width < 720) return 6;
  return 8;
}

function heatmapGridColumns(tokenCount) {
  return `var(--heatmap-layer-column, 46px) repeat(${tokenCount}, minmax(0, 1fr)) var(--heatmap-mass-column, 70px)`;
}

function heatmapCellSelector(metricKey, layerIndex, tokenIndex) {
  return `.chatterbox-heatmap-cell[data-heatmap-metric="${metricKey}"][data-layer-index="${layerIndex}"][data-text-token-index="${tokenIndex}"]`;
}

function renderHeatmapFacet(container, config, tokens, layers, bandSize) {
  container.replaceChildren();
  const heading = createElement("div", "chatterbox-heatmap-heading");
  const copy = createElement("div");
  copy.append(
    createElement("strong", "", config.title),
    createElement("span", "", config.subtitle),
  );
  const scale = createElement("div", "chatterbox-heatmap-scale");
  scale.append(
    createElement("span", "", "0%"),
    createElement("i", config.className),
    createElement("span", "", "100%"),
  );
  heading.append(copy, scale);
  container.append(heading);

  for (let bandStart = 0; bandStart < tokens.length; bandStart += bandSize) {
    const bandTokens = tokens.slice(bandStart, bandStart + bandSize);
    const band = createElement("div", "chatterbox-heatmap-band");
    if (tokens.length > bandSize) {
      band.append(createElement("p", "chatterbox-heatmap-band-label", `Tokens ${bandStart + 1}–${bandStart + bandTokens.length} of ${tokens.length}`));
    }
    const header = createElement("div", "chatterbox-heatmap-row chatterbox-heatmap-header");
    header.style.gridTemplateColumns = heatmapGridColumns(bandTokens.length);
    header.append(createElement("span", "chatterbox-heatmap-corner", "Layer"));
    bandTokens.forEach((token, bandIndex) => {
      const tokenIndex = bandStart + bandIndex;
      const label = createElement("span", "chatterbox-heatmap-token", visibleToken(token.text));
      label.setAttribute("aria-label", `Input token ${tokenIndex + 1}, ${visibleToken(token.text)}`);
      label.title = `T${tokenIndex + 1} · ${visibleToken(token.text)}`;
      header.append(label);
    });
    const massHeader = createElement("span", "chatterbox-heatmap-mass-header", "Text / prefix");
    massHeader.title = config.massTitle;
    header.append(massHeader);
    band.append(header);

    layers.forEach((layer, layerIndex) => {
      const values = matrixRow(state.trace[config.matrixKey], layer, layerIndex);
      const rawValues = config.rawKey ? matrixRow(state.trace[config.rawKey], layer, layerIndex) : [];
      const mass = clamp(layerScalar(state.trace[config.massKey], layer, layerIndex));
      const row = createElement("div", "chatterbox-heatmap-row");
      row.style.gridTemplateColumns = heatmapGridColumns(bandTokens.length);
      const layerLabel = createElement("span", "chatterbox-heatmap-layer", `L${layer}`);
      layerLabel.title = config.layerTitle(layer);
      row.append(layerLabel);
      bandTokens.forEach((token, bandIndex) => {
        const tokenIndex = bandStart + bandIndex;
        const value = clamp(values[tokenIndex]);
        const selected = layerIndex === state.selectedLayerIndex && tokenIndex === state.selectedTextIndex;
        const cell = createElement("button", `chatterbox-heatmap-cell ${config.className}`, formatProbability(value, 1));
        cell.type = "button";
        cell.dataset.heatmapMetric = config.key;
        cell.dataset.layerIndex = String(layerIndex);
        cell.dataset.textTokenIndex = String(tokenIndex);
        cell.style.setProperty("--heatmap-value", value.toFixed(6));
        cell.classList.toggle("selected", selected);
        cell.classList.toggle("strong-value", value >= 0.68);
        cell.setAttribute("aria-pressed", String(selected));
        cell.tabIndex = selected ? 0 : -1;
        const rawDetail = config.rawKey ? ` Raw local sensitivity ${formatMetric(rawValues[tokenIndex])}.` : "";
        const label = `${config.cellPrefix(layer)}, input token ${tokenIndex + 1}, ${visibleToken(token.text)}: ${formatProbability(value, 3)}.${rawDetail} ${config.massAria(mass)}`;
        cell.setAttribute("aria-label", label);
        cell.title = label;
        cell.addEventListener("click", () => selectHeatmapCell(layerIndex, tokenIndex, config.key, true, true));
        row.append(cell);
      });
      const massCell = createElement("span", `chatterbox-heatmap-mass ${config.className}`, formatProbability(mass, 1));
      massCell.style.setProperty("--heatmap-value", mass.toFixed(6));
      massCell.classList.toggle("strong-value", mass >= 0.68);
      massCell.setAttribute("aria-label", config.massAria(mass));
      massCell.title = config.massAria(mass);
      row.append(massCell);
      band.append(row);
    });
    container.append(band);
  }
}

function renderMatrixFocus() {
  const tokens = traceTextTokens();
  const layers = traceLayers();
  if (!tokens.length || !layers.length || state.selectedTextIndex === null) {
    elements.matrixFocus.hidden = true;
    return;
  }
  const layer = layers[state.selectedLayerIndex];
  const token = tokens[state.selectedTextIndex];
  const gradient = matrixRow(state.trace.gradient_share, layer, state.selectedLayerIndex)[state.selectedTextIndex];
  const attention = matrixRow(state.trace.attention_share, layer, state.selectedLayerIndex)[state.selectedTextIndex];
  const gradientMass = layerScalar(state.trace.gradient_text_mass, layer, state.selectedLayerIndex);
  const attentionMass = layerScalar(state.trace.attention_text_mass, layer, state.selectedLayerIndex);
  elements.matrixFocus.replaceChildren(
    createElement("span", "", "Selected coordinate"),
    createElement("strong", "", `L${layer} × T${state.selectedTextIndex + 1} · ${visibleToken(token.text)}`),
    createElement("span", "metric-gradient", `Gradient ${formatProbability(gradient, 3)}`),
    createElement("span", "metric-attention", `Attention ${formatProbability(attention, 3)}`),
    createElement("small", "", `Text / prefix mass · gradient ${formatProbability(gradientMass)} · attention ${formatProbability(attentionMass)}`),
  );
  elements.matrixFocus.hidden = false;
}

function renderLayerMatrices() {
  const tokens = traceTextTokens();
  const layers = traceLayers();
  if (!tokens.length || !layers.length) return;
  const bandSize = heatmapBandSize();
  state.matrixBandSize = bandSize;
  renderHeatmapFacet(elements.gradientHeatmap, {
    key: "gradient",
    title: "Post-block gradient sensitivity",
    subtitle: "Within-text share by tokenizer piece · each row sums to 100%",
    className: "metric-gradient",
    matrixKey: "gradient_share",
    rawKey: "gradient_l2",
    massKey: "gradient_text_mass",
    massTitle: "Text share of summed positionwise gradient norms over the full causal prefix",
    layerTitle: (layer) => `T3 gradient source · post-block L${layer}`,
    cellPrefix: (layer) => `T3 post-block layer ${layer} gradient share`,
    massAria: (mass) => `Text share of summed positionwise gradient norms is ${formatProbability(mass, 3)}.`,
  }, tokens, layers, bandSize);
  renderHeatmapFacet(elements.attentionHeatmap, {
    key: "attention",
    title: "Block self-attention",
    subtitle: "Within-text share by tokenizer piece · each row sums to 100%",
    className: "metric-attention",
    matrixKey: "attention_share",
    rawKey: null,
    massKey: "attention_text_mass",
    massTitle: "Self-attention mass reaching all input-text positions",
    layerTitle: (layer) => `T3 self-attention · block L${layer}`,
    cellPrefix: (layer) => `T3 block layer ${layer} within-text self-attention share`,
    massAria: (mass) => `Self-attention mass reaching text is ${formatProbability(mass, 3)}.`,
  }, tokens, layers, bandSize);
  renderMatrixFocus();
}

function selectHeatmapCell(layerIndex, tokenIndex, metricKey, announceChange = false, focusCell = false) {
  state.selectedLayerIndex = Math.max(0, Math.min(traceLayers().length - 1, layerIndex));
  state.selectedTextIndex = Math.max(0, Math.min(traceTextTokens().length - 1, tokenIndex));
  renderTrace();
  if (announceChange) {
    const layer = traceLayers()[state.selectedLayerIndex];
    const token = traceTextTokens()[state.selectedTextIndex];
    announce(`T3 layer ${layer}, input token ${visibleToken(token.text)} selected in the ${metricKey} matrix.`);
  }
  if (focusCell) {
    window.requestAnimationFrame(() => {
      const cell = elements.layerMatrices.querySelector(heatmapCellSelector(metricKey, state.selectedLayerIndex, state.selectedTextIndex));
      cell?.focus();
      cell?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  }
}

function selectLayer(layerIndex, announceChange = false) {
  state.selectedLayerIndex = Math.max(0, Math.min(traceLayers().length - 1, layerIndex));
  renderTrace();
  if (announceChange) announce(`T3 layer ${traceLayers()[state.selectedLayerIndex]} selected.`);
}

function renderInspector() {
  const tokens = traceTextTokens();
  const layers = traceLayers();
  if (!tokens.length || !layers.length) return;
  const tokenIndex = Math.max(0, Math.min(tokens.length - 1, state.selectedTextIndex ?? 0));
  const token = tokens[tokenIndex];
  const layer = layers[state.selectedLayerIndex];
  const gradientRaw = matrixRow(state.trace.gradient_l2, layer, state.selectedLayerIndex);
  const gradientShare = matrixRow(state.trace.gradient_share, layer, state.selectedLayerIndex);
  const attentionShare = matrixRow(state.trace.attention_share, layer, state.selectedLayerIndex);
  const gradientMass = layerScalar(state.trace.gradient_text_mass, layer, state.selectedLayerIndex);
  const textMass = layerScalar(state.trace.attention_text_mass, layer, state.selectedLayerIndex);
  const start = Number.isFinite(Number(token.char_start)) ? Number(token.char_start) : null;
  const end = Number.isFinite(Number(token.char_end)) ? Number(token.char_end) : null;
  elements.inspectorTitle.textContent = visibleToken(token.text);
  elements.inspectorContext.textContent = start !== null && end !== null
    ? `Input token ${tokenIndex + 1} · normalized-text characters ${start}–${end}`
    : `Input token ${tokenIndex + 1} · model token ID ${token.id ?? "unavailable"}`;
  elements.inspectorLayer.textContent = `T3 L${layer} · gradient post-block + attention block`;
  elements.inspectorToken.textContent = `#${tokenIndex + 1} · ID ${token.id ?? "—"}`;
  elements.inspectorGradient.textContent = formatMetric(gradientRaw[tokenIndex]);
  elements.inspectorGradientShare.textContent = formatProbability(gradientShare[tokenIndex], 3);
  elements.inspectorGradientMass.textContent = formatProbability(gradientMass, 2);
  elements.inspectorAttention.textContent = formatProbability(attentionShare[tokenIndex], 3);
  elements.inspectorTextMass.textContent = formatProbability(textMass, 2);
}

function renderWarnings(warnings) {
  const unique = [...new Set((Array.isArray(warnings) ? warnings : []).filter(Boolean).map(String))];
  elements.warningList.replaceChildren(...unique.map((warning) => createElement("li", "", warning)));
  elements.warnings.hidden = !unique.length;
  elements.warnings.closest(".metadata-card")?.classList.toggle("no-warnings", !unique.length);
}

function drawWaveform() {
  const canvas = elements.waveformCanvas;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  const waveform = Array.isArray(state.analysis?.output?.waveform) ? state.analysis.output.waveform : [];
  if (!waveform.length) return;
  const style = getComputedStyle(document.documentElement);
  const lens = style.getPropertyValue("--lens-rgb").trim() || "23, 111, 152";
  const middle = rect.height / 2;
  const maximum = Math.max(0.000001, ...waveform.map((value) => Math.abs(finiteNumber(value))));
  const width = Math.max(1, rect.width / waveform.length * 0.64);
  context.fillStyle = `rgba(${lens}, 0.55)`;
  waveform.forEach((value, index) => {
    const x = (index + 0.5) / waveform.length * rect.width;
    const height = Math.max(1, Math.abs(finiteNumber(value)) / maximum * (rect.height - 20));
    context.fillRect(x - width / 2, middle - height / 2, width, height);
  });
  context.strokeStyle = `rgba(${lens}, 0.13)`;
  context.beginPath();
  context.moveTo(0, middle + 0.5);
  context.lineTo(rect.width, middle + 0.5);
  context.stroke();
}

function updatePlayback() {
  const duration = Math.max(outputDuration(), finiteNumber(elements.audio.duration));
  const current = Math.max(0, finiteNumber(elements.audio.currentTime));
  elements.currentTime.textContent = formatTime(current);
  elements.waveformPlayhead.style.left = `${(duration ? clamp(current / duration) : 0) * 100}%`;
}

function resetPage() {
  state.generationController?.abort();
  state.branchController?.abort();
  state.residualController?.abort();
  window.clearTimeout(state.traceDebounceTimer);
  state.traceDebounceTimer = null;
  state.traceController?.abort();
  state.analysis = null;
  state.trace = null;
  state.traceCache.clear();
  state.mode = null;
  state.branchLoading = false;
  state.branchSelection = null;
  state.branchComparison = null;
  state.branchStatus = "";
  state.residualLoading = false;
  state.residualSelection = null;
  state.residualLayers = new Set();
  state.residualStartIndex = 0;
  state.residualResult = null;
  state.residualStatus = "";
  state.residualFocusLayerIndex = 0;
  state.residualFocusPosition = 0;
  state.selectedLensLayerIndex = 0;
  state.selectedTextIndex = null;
  elements.audio.pause();
  elements.audio.removeAttribute("src");
  elements.audio.load();
  releaseDemoAudio();
  renderBranchComparison();
  renderResidualResult();
  elements.results.hidden = true;
  clearError();
  updateGenerateAvailability();
  elements.text.focus();
  announce("Chatterbox workspace reset. Enter a new text prompt.");
}

function syntheticTokens(text) {
  const tokens = [];
  const pattern = / ?[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    tokens.push({
      index: tokens.length,
      id: 310 + tokens.length * 17,
      text: match[0],
      char_start: match.index,
      char_end: match.index + match[0].length,
    });
  }
  return tokens.length ? tokens : [{ index: 0, id: 310, text, char_start: 0, char_end: text.length }];
}

function createDemoWav(duration, textTokenCount) {
  const sampleRate = 24_000;
  const sampleCount = Math.floor(duration * sampleRate);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const write = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  write(0, "RIFF"); view.setUint32(4, 36 + sampleCount * 2, true); write(8, "WAVE"); write(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); write(36, "data"); view.setUint32(40, sampleCount * 2, true);
  for (let index = 0; index < sampleCount; index += 1) {
    const time = index / sampleRate;
    const tokenPhase = time / duration * Math.max(1, textTokenCount);
    const local = tokenPhase % 1;
    const frequency = 150 + (Math.floor(tokenPhase) % 7) * 24;
    const envelope = Math.pow(Math.sin(Math.PI * local), 1.4) * Math.pow(Math.sin(Math.PI * time / duration), 0.35) * 0.18;
    const carrier = Math.sin(2 * Math.PI * frequency * time) + 0.3 * Math.sin(2 * Math.PI * frequency * 2.07 * time);
    view.setInt16(44 + index * 2, clamp(carrier * envelope, -1, 1) * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function createDemoCandidateDistribution(targetId, targetProbability, targetRank, topK, salt) {
  const probability = clamp(targetProbability, 0.0001, 0.96);
  const rank = Math.max(1, Number(targetRank) || 1);
  let probabilities;
  if (rank <= topK) {
    const higherCount = rank - 1;
    const triangle = Math.max(1, higherCount * (higherCount + 1) / 2);
    const step = clamp((0.92 - rank * probability) / triangle, 0.001, 0.012);
    const higher = Array.from({ length: higherCount }, (_, index) => (
      probability + step * (higherCount - index)
    ));
    const lowerCount = topK - rank;
    const lowerRaw = Array.from({ length: lowerCount }, (_, index) => probability * Math.pow(0.68, index + 1));
    const fixedMass = higher.reduce((sum, value) => sum + value, 0) + probability;
    const lowerMass = lowerRaw.reduce((sum, value) => sum + value, 0);
    const lowerScale = lowerMass > 0 ? Math.min(1, Math.max(0, 0.96 - fixedMass) / lowerMass) : 1;
    probabilities = [...higher, probability, ...lowerRaw.map((value) => value * lowerScale)];
  } else {
    const triangle = topK * (topK + 1) / 2;
    const step = clamp((0.96 - (topK + 1) * probability) / Math.max(1, triangle), 0.001, 0.02);
    probabilities = Array.from({ length: topK }, (_, index) => probability + step * (topK - index));
  }
  return probabilities.map((candidateProbability, index) => {
    if (rank <= topK && index === rank - 1) return { id: Number(targetId), probability };
    let candidateId = (Number(targetId) + 491 * (index + 1) + salt * 37) % 6561;
    if (candidateId === Number(targetId)) candidateId = (candidateId + 1) % 6561;
    return { id: candidateId, probability: clamp(candidateProbability, 0.0001, 0.96) };
  });
}

function createDemoFittedSpeechLens(speechCodes) {
  const layers = [0, 4, 8, 12, 16, 20, 22];
  const targetProbabilities = [];
  const targetLogProbabilities = [];
  const targetRanks = [];
  const topCodes = [];
  layers.forEach((layer, layerIndex) => {
    const progress = layerIndex / Math.max(1, layers.length - 1);
    const probabilities = speechCodes.map((code, codeIndex) => clamp(
      finiteNumber(code.raw_probability) * (0.2 + progress * 0.77)
        + (1 - progress) * (0.006 + 0.012 * (0.5 + 0.5 * Math.sin(codeIndex * 0.61 + layerIndex))),
      0.0001,
      0.98,
    ));
    const ranks = probabilities.map((probability, codeIndex) => {
      if (probability >= 0.45) return 1;
      if (probability >= 0.28) return 2;
      if (probability >= 0.14) return 3;
      return Math.max(4, Math.round(11 - progress * 7 + 2 * (0.5 + 0.5 * Math.sin(codeIndex * 0.47))));
    });
    const layerTopCodes = probabilities.map((probability, codeIndex) => {
      const target = speechCodes[codeIndex];
      const rank = ranks[codeIndex];
      return createDemoCandidateDistribution(target.id, probability, rank, 5, layerIndex + codeIndex);
    });
    targetProbabilities.push(probabilities);
    targetLogProbabilities.push(probabilities.map((probability) => Math.log(probability)));
    targetRanks.push(ranks);
    topCodes.push(layerTopCodes);
  });
  return {
    layers,
    target_ids: speechCodes.map((code) => Number(code.id)),
    target_probabilities: targetProbabilities,
    target_log_probabilities: targetLogProbabilities,
    target_ranks: targetRanks,
    top_codes: topCodes,
    source_coordinate: "post_block_residual",
    target_head: "final_norm_then_t3_speech_head",
    normalization: "full_speech_head_softmax_before_generation_processors",
    artifact: {
      format: "synthetic_fitted_speech_code_jlens_fixture",
      fingerprint: "fabricated-ui-data",
      fit_examples: 0,
      estimator: "fabricated browser demo",
    },
    warnings: [
      "Synthetic fitted J-lens demo: every blue fitted probability, rank, and top-code list is fabricated for interface testing.",
    ],
  };
}

function createDemoSpeechHeadCandidates(speechCodes) {
  const topK = 5;
  const targetProbabilities = speechCodes.map((code) => clamp(code.raw_probability));
  const targetRanks = targetProbabilities.map((probability, codeIndex) => {
    if (probability >= 0.5) return 1;
    if (probability >= 0.28) return 2;
    if (probability >= 0.15) return 3;
    if (probability >= 0.1) return 4;
    return 7 + (codeIndex % 3);
  });
  const topCodes = speechCodes.map((code, codeIndex) => {
    const targetProbability = targetProbabilities[codeIndex];
    const targetRank = targetRanks[codeIndex];
    return createDemoCandidateDistribution(code.id, targetProbability, targetRank, topK, codeIndex + 113);
  });
  return {
    schema_version: 1,
    top_k: topK,
    vocab_size: 6563,
    target_ids: speechCodes.map((code) => Number(code.id)),
    target_probabilities: targetProbabilities,
    target_log_probabilities: targetProbabilities.map((probability) => Math.log(probability)),
    target_ranks: targetRanks,
    top_codes: topCodes,
    source_coordinate: "final_t3_speech_prediction_position",
    target_head: "t3_speech_head_after_final_norm",
    normalization: "full_speech_head_softmax_before_generation_processors",
    special_token_ids: { start: 6561, stop: 6562 },
    generation_processors_excluded: ["repetition_penalty", "temperature", "top_k", "top_p"],
    warnings: [
      "Synthetic output-head candidate lists are fabricated for interface testing.",
      "Speech-code IDs are learned acoustic symbols, not words or published phoneme labels.",
    ],
  };
}

function createDemoGeneration() {
  const rawText = elements.text.value.trim() || "The lighthouse glowed softly through the rain.";
  const normalizedText = rawText.replace(/\s+/g, " ").trim();
  const tokens = syntheticTokens(normalizedText);
  const codeCount = Math.min(120, Math.max(42, tokens.length * 9));
  const nominalDuration = codeCount * 0.04;
  const trailingAudio = 0.12;
  const duration = nominalDuration + trailingAudio;
  const waveform = Array.from({ length: 360 }, (_, index) => {
    const time = index / 360 * duration;
    if (time > nominalDuration) return 0.03 * (1 - (time - nominalDuration) / trailingAudio);
    const syllable = Math.pow(Math.max(0, Math.sin(time * Math.PI * 4.1)), 0.65);
    return 0.035 + syllable * (0.48 + 0.22 * Math.sin(index * 1.73)) * Math.pow(Math.sin(Math.PI * time / nominalDuration), 0.3);
  });
  const speechCodes = Array.from({ length: codeCount }, (_, index) => {
    const probability = clamp(0.08 + 0.68 * Math.pow(0.5 + 0.5 * Math.sin(index * 0.81 + 0.4), 1.4), 0.01, 0.94);
    return {
      index,
      id: (1847 + index * 317) % 6561,
      start_seconds: index * 0.04,
      end_seconds: (index + 1) * 0.04,
      mel_start: index * 2,
      mel_end: index * 2 + 2,
      raw_probability: probability,
      raw_log_probability: Math.log(probability),
    };
  });
  releaseDemoAudio();
  state.demoAudioUrl = URL.createObjectURL(createDemoWav(duration, tokens.length));
  return {
    analysis_id: "synthetic-ui-demo",
    schema_version: 3,
    model: {
      backend: "synthetic",
      model_family: "chatterbox_turbo",
      model_id: "Synthetic Chatterbox trace fixture",
      model_revision: "fabricated-ui-data",
      s3_tokenizer_id: "synthetic S3 tokenizer",
      s3_tokenizer_revision: "fabricated-ui-data",
      quantization: "not applicable",
      t3_layers: 23,
      t3_width: 1024,
      attention_heads: 16,
      speech_vocab_size: 6563,
      valid_speech_codes: 6561,
      speech_code_rate_hz: 25,
      mel_frame_rate_hz: 50,
      sample_rate: 24_000,
      generation: { policy: "fabricated deterministic browser demo" },
    },
    input: { raw_text: rawText, normalized_text: normalizedText, tokens },
    output: {
      audio_data_url: state.demoAudioUrl,
      sample_rate: 24_000,
      duration_seconds: duration,
      waveform,
      speech_codes: speechCodes,
      speech_head_candidates: createDemoSpeechHeadCandidates(speechCodes),
      nominal_content_duration_seconds: nominalDuration,
      trailing_audio_seconds: trailingAudio,
    },
    replay: { policy: "synthetic_fixture", max_abs_logit_error: 0 },
    fitted_speech_code_jlens: createDemoFittedSpeechLens(speechCodes),
    warnings: [
      "Synthetic demo: speech codes, waveform, output probabilities, fitted-lens readouts, gradients, and attention values are fabricated for interface testing.",
      "A nominal 40 ms code boundary is not an exact acoustic boundary after flow decoding and vocoding.",
    ],
  };
}

function createDemoTrace(analysis, codeIndex) {
  const layers = [0, 4, 8, 12, 16, 20, 22];
  const tokens = analysis.input.tokens;
  const codeCount = analysis.output.speech_codes.length;
  const target = codeCount <= 1 ? 0 : codeIndex / (codeCount - 1) * Math.max(0, tokens.length - 1);
  const gradientL2 = [];
  const gradientShare = [];
  const gradientTextMass = [];
  const attentionShare = [];
  const attentionTextMass = [];
  layers.forEach((layer, layerIndex) => {
    const progress = layerIndex / Math.max(1, layers.length - 1);
    const width = 2.8 - progress * 1.9;
    const gradient = tokens.map((token, tokenIndex) => {
      const local = Math.exp(-Math.pow(tokenIndex - target, 2) / (2 * width * width));
      return 0.002 + local * (0.12 + progress * 0.26) * (0.82 + 0.18 * Math.sin((tokenIndex + 1) * (layerIndex + 2)));
    });
    const gradientTotal = gradient.reduce((sum, value) => sum + value, 0) || 1;
    const attention = tokens.map((token, tokenIndex) => {
      const shiftedTarget = target + Math.sin(layerIndex * 0.8) * 0.45;
      return 0.005 + Math.exp(-Math.pow(tokenIndex - shiftedTarget, 2) / (2 * Math.pow(width * 0.82, 2)));
    });
    const attentionTotal = attention.reduce((sum, value) => sum + value, 0) || 1;
    const textMass = 0.31 + progress * 0.34 + 0.04 * Math.sin(layerIndex);
    gradientL2.push(gradient);
    gradientShare.push(gradient.map((value) => value / gradientTotal));
    gradientTextMass.push(0.18 + progress * 0.42 + 0.03 * Math.cos(layerIndex));
    attentionShare.push(attention.map((value) => value / attentionTotal));
    attentionTextMass.push(textMass);
  });
  return {
    selection: {
      speech_code_index: codeIndex,
      speech_code_id: analysis.output.speech_codes[codeIndex].id,
    },
    layers,
    text_tokens: tokens,
    gradient_l2: gradientL2,
    gradient_share: gradientShare,
    gradient_text_mass: gradientTextMass,
    attention_share: attentionShare,
    attention_text_mass: attentionTextMass,
    attention_kind: "synthetic_text_prefix_causal_self_attention",
    score_kind: "synthetic_selected_raw_log_probability_gradient_l2",
    warnings: [
      "Demo gradient and attention values are synthetic and do not come from Chatterbox.",
      "Attention is causal self-attention over a concatenated context; it is not a word-to-audio alignment.",
    ],
  };
}

function loadDemo() {
  if (state.loading || state.branchLoading || state.residualLoading) return;
  clearError();
  const analysis = createDemoGeneration();
  renderGeneration(analysis, "demo");
  selectSpeechCode(Math.min(8, analysis.output.speech_codes.length - 1), { seek: true, announceChange: false });
  announce("Synthetic Chatterbox interface demo loaded. All displayed values are fabricated.");
}

function bindEvents() {
  elements.statusButton.addEventListener("click", checkStatus);
  elements.text.addEventListener("input", () => { updateCharacterCount(); updateGenerateAvailability(); });
  elements.presetButtons.forEach((button) => button.addEventListener("click", () => {
    elements.text.value = button.dataset.chatterboxPreset;
    updateCharacterCount();
    updateGenerateAvailability();
    elements.text.focus();
  }));
  elements.generateButton.addEventListener("click", generateSpeech);
  elements.demoButton.addEventListener("click", loadDemo);
  elements.branchButton.addEventListener("click", branchSpeechFromCandidate);
  elements.residualButton.addEventListener("click", runResidualSteering);
  elements.interventionMode.addEventListener("change", (event) => {
    const mode = event.target.closest('input[name="chatterbox-intervention-mode"]')?.value;
    if (!mode || !["force", "residual"].includes(mode)) return;
    state.interventionMode = mode;
    state.branchStatus = "";
    state.residualStatus = "";
    renderSpeechCandidateInspector();
    announce(`${mode === "residual" ? "Residual steering" : "Forced output code"} intervention mode selected. Candidate selection alone will not run the intervention.`);
  });
  elements.residualSourceLayers.addEventListener("change", (event) => {
    const input = event.target.closest("input[data-residual-layer]");
    if (!input) return;
    const layer = Number(input.value);
    if (input.checked) state.residualLayers.add(layer);
    else state.residualLayers.delete(layer);
    renderResidualControls();
  });
  elements.residualStartPosition.addEventListener("change", () => {
    const codes = speechCodes();
    if (!codes.length) return;
    const next = Math.max(0, Math.min(codes.length - 1, Number(elements.residualStartPosition.value) - 1 || 0));
    state.residualStartIndex = next;
    if (next !== state.selectedCodeIndex) {
      selectSpeechCode(next, { seek: true, announceChange: false });
      announce(`Residual start moved to speech position ${next + 1}. The stale target nomination was cleared; choose a candidate at this position.`);
    } else {
      renderResidualControls();
    }
  });
  elements.residualForwardSpan.addEventListener("input", renderResidualControls);
  elements.residualMaxRelativeNorm.addEventListener("change", () => {
    elements.residualMaxRelativeNorm.value = residualSteeringConfig().maxRelativeResidualNorm.toFixed(2);
    renderResidualControls();
  });
  elements.resetButton.addEventListener("click", resetPage);
  elements.layerSelect.addEventListener("change", () => {
    const index = traceLayers().findIndex((layer) => String(layer) === elements.layerSelect.value);
    if (index >= 0) selectLayer(index, true);
  });
  elements.speechLensHeatmap.addEventListener("keydown", (event) => {
    const current = event.target.closest("button.chatterbox-speech-lens-cell");
    const lens = fittedSpeechLens();
    if (!current || !lens) return;
    let layerIndex = Number(current.dataset.lensLayerIndex);
    let codeIndex = Number(current.dataset.codeIndex);
    if (!Number.isInteger(layerIndex) || !Number.isInteger(codeIndex)) return;
    if ((event.ctrlKey || event.metaKey) && event.key === "Home") {
      layerIndex = 0;
      codeIndex = 0;
    } else if ((event.ctrlKey || event.metaKey) && event.key === "End") {
      layerIndex = lens.layers.length - 1;
      codeIndex = speechCodes().length - 1;
    } else if (event.key === "ArrowLeft") codeIndex -= 1;
    else if (event.key === "ArrowRight") codeIndex += 1;
    else if (event.key === "ArrowUp") layerIndex -= 1;
    else if (event.key === "ArrowDown") layerIndex += 1;
    else if (event.key === "Home") codeIndex = 0;
    else if (event.key === "End") codeIndex = speechCodes().length - 1;
    else return;
    event.preventDefault();
    selectFittedSpeechLensCell(layerIndex, codeIndex, false, true);
  });
  elements.textTokens.addEventListener("keydown", (event) => {
    const current = event.target.closest("button[data-text-token-index]");
    if (!current) return;
    const buttons = [...elements.textTokens.querySelectorAll("button[data-text-token-index]")];
    let index = buttons.indexOf(current);
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") index -= 1;
    else if (event.key === "ArrowRight" || event.key === "ArrowDown") index += 1;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = buttons.length - 1;
    else return;
    event.preventDefault();
    const next = buttons[Math.max(0, Math.min(buttons.length - 1, index))];
    const tokenIndex = Number(next?.dataset.textTokenIndex);
    if (Number.isInteger(tokenIndex)) {
      state.selectedTextIndex = tokenIndex;
      renderTrace();
      window.requestAnimationFrame(() => elements.textTokens.querySelector(`[data-text-token-index="${tokenIndex}"]`)?.focus());
    }
  });
  elements.layerMatrices.addEventListener("keydown", (event) => {
    const current = event.target.closest("button.chatterbox-heatmap-cell");
    if (!current) return;
    const metricKey = current.dataset.heatmapMetric;
    let layerIndex = Number(current.dataset.layerIndex);
    let tokenIndex = Number(current.dataset.textTokenIndex);
    if (!Number.isInteger(layerIndex) || !Number.isInteger(tokenIndex)) return;
    if ((event.ctrlKey || event.metaKey) && event.key === "Home") {
      layerIndex = 0;
      tokenIndex = 0;
    } else if ((event.ctrlKey || event.metaKey) && event.key === "End") {
      layerIndex = traceLayers().length - 1;
      tokenIndex = traceTextTokens().length - 1;
    } else if (event.key === "ArrowLeft") tokenIndex -= 1;
    else if (event.key === "ArrowRight") tokenIndex += 1;
    else if (event.key === "ArrowUp") layerIndex -= 1;
    else if (event.key === "ArrowDown") layerIndex += 1;
    else if (event.key === "Home") tokenIndex = 0;
    else if (event.key === "End") tokenIndex = traceTextTokens().length - 1;
    else return;
    event.preventDefault();
    selectHeatmapCell(layerIndex, tokenIndex, metricKey, false, true);
  });
  elements.residualDeltaMatrix.addEventListener("keydown", (event) => {
    const current = event.target.closest("button.chatterbox-residual-delta-cell");
    const result = state.residualResult;
    if (!current || !result) return;
    let rowIndex = Number(current.dataset.residualRowIndex);
    let positionIndex = Number(current.dataset.residualPositionIndex);
    const diagnostics = result.intervention.target_diagnostics;
    if ((event.ctrlKey || event.metaKey) && event.key === "Home") {
      rowIndex = 0;
      positionIndex = 0;
    } else if ((event.ctrlKey || event.metaKey) && event.key === "End") {
      rowIndex = diagnostics.fitted_layers.length;
      positionIndex = diagnostics.positions.length - 1;
    } else if (event.key === "ArrowLeft") positionIndex -= 1;
    else if (event.key === "ArrowRight") positionIndex += 1;
    else if (event.key === "ArrowUp") rowIndex -= 1;
    else if (event.key === "ArrowDown") rowIndex += 1;
    else if (event.key === "Home") positionIndex = 0;
    else if (event.key === "End") positionIndex = diagnostics.positions.length - 1;
    else return;
    event.preventDefault();
    selectResidualDeltaCell(rowIndex, positionIndex, { focusCell: true });
  });

  ["timeupdate", "play", "pause", "loadedmetadata"].forEach((type) => elements.audio.addEventListener(type, updatePlayback));
  elements.waveform.addEventListener("pointermove", (event) => {
    const rect = elements.waveform.getBoundingClientRect();
    elements.waveformHover.hidden = false;
    elements.waveformHover.style.left = `${clamp((event.clientX - rect.left) / Math.max(1, rect.width)) * 100}%`;
  });
  elements.waveform.addEventListener("pointerleave", () => { elements.waveformHover.hidden = true; });
  elements.waveform.addEventListener("click", (event) => {
    const rect = elements.waveform.getBoundingClientRect();
    const time = clamp((event.clientX - rect.left) / Math.max(1, rect.width)) * outputDuration();
    selectSpeechCode(codeIndexAtTime(time), { seek: true });
  });
  elements.waveform.addEventListener("keydown", (event) => {
    const codes = speechCodes();
    if (!codes.length) return;
    let index = state.selectedCodeIndex;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") index -= 1;
    else if (event.key === "ArrowRight" || event.key === "ArrowUp") index += 1;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = codes.length - 1;
    else return;
    event.preventDefault();
    selectSpeechCode(index, { seek: true });
  });

  if ("ResizeObserver" in window) {
    state.resizeObserver = new ResizeObserver(drawWaveform);
    state.resizeObserver.observe(elements.waveform);
    state.matrixResizeObserver = new ResizeObserver(() => {
      const bandSize = heatmapBandSize();
      if (state.trace && bandSize !== state.matrixBandSize) renderLayerMatrices();
    });
    state.matrixResizeObserver.observe(elements.layerMatrices);
  } else {
    window.addEventListener("resize", () => {
      drawWaveform();
      if (state.trace) renderLayerMatrices();
    });
  }
  window.addEventListener("beforeunload", () => {
    state.generationController?.abort();
    state.branchController?.abort();
    state.residualController?.abort();
    window.clearTimeout(state.traceDebounceTimer);
    state.traceController?.abort();
    state.resizeObserver?.disconnect();
    state.matrixResizeObserver?.disconnect();
    releaseDemoAudio();
  });
}

bindEvents();
updateCharacterCount();
updateGenerateAvailability();
checkStatus();
