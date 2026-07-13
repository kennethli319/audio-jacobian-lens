"use strict";

const API = Object.freeze({
  status: "/api/status",
  samples: "/api/samples",
  analyze: "/api/analyze",
  analysisJobs: "/api/analysis/jobs",
});

const ANALYSIS_JOB_POLL_INTERVAL_MS = 750;

const MAX_RECORD_SECONDS = 30;
const RECORD_AUTO_STOP_SECONDS = 29;
const ENCODER_WINDOW_SECONDS = 0.1;
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  brandSubtitle: $("#brand-subtitle"),
  modelBadge: $("#model-badge"),
  heroTitle: $("#hero-title"),
  heroCopy: $("#hero-copy"),
  probabilityPrimerTitle: $("#probability-primer-title"),
  probabilityPrimerCopy: $("#probability-primer-copy"),
  lensPrimerTitle: $("#lens-primer-title"),
  lensPrimerCopy: $("#lens-primer-copy"),
  modeGuideLabel: $("#mode-guide-label"),
  modeGuideTitle: $("#mode-guide-title"),
  modeGuideSummary: $("#mode-guide-summary"),
  modeGuidePrimaryTitle: $("#mode-guide-primary-title"),
  modeGuidePrimary: $("#mode-guide-primary"),
  modeGuideSecondaryRow: $("#mode-guide-secondary-row"),
  modeGuideSecondaryTitle: $("#mode-guide-secondary-title"),
  modeGuideSecondary: $("#mode-guide-secondary"),
  modeGuideImplication: $("#mode-guide-implication"),
  modeGuideBoundary: $("#mode-guide-boundary"),
  modeGuideSteps: $("#mode-guide-steps"),
  modeGuideScale: $("#mode-guide-scale"),
  encoderAnalysisSettings: $("#encoder-analysis-settings"),
  statusButton: $("#status-button"),
  statusDot: $("#status-dot"),
  statusText: $("#status-text"),
  hostedNotice: $("#hosted-notice"),
  inputPanel: $("#input-panel"),
  sourceTabs: $$("[data-source-tab]"),
  sourcePanels: $$(".source-panel"),
  sampleList: $("#sample-list"),
  samplesStatus: $("#samples-status"),
  audioFile: $("#audio-file"),
  dropZone: $("#drop-zone"),
  browseButton: $("#browse-button"),
  fileLabel: $("#file-label"),
  fileDetail: $("#file-detail"),
  analyzeButton: $("#analyze-button"),
  analyzeLabel: $("#analyze-label"),
  buttonLoader: $("#button-loader"),
  encoderOverlapSeconds: $("#encoder-overlap-seconds"),
  encoderHopSummary: $("#encoder-hop-summary"),
  selectedSource: $("#selected-source"),
  selectedTitle: $("#selected-title"),
  selectedDetail: $("#selected-detail"),
  selectedAnalyze: $("#selected-analyze"),
  recordStart: $("#record-start"),
  recordStop: $("#record-stop"),
  recordDiscard: $("#record-discard"),
  recordAnalyze: $("#record-analyze"),
  recordState: $("#record-state"),
  recordDetail: $("#record-detail"),
  recordTimer: $("#record-timer"),
  recordVisual: $("#record-visual"),
  recordReview: $("#record-review"),
  recordPlayer: $("#record-player"),
  recordError: $("#record-error"),
  demoButton: $("#demo-button"),
  resetButton: $("#reset-button"),
  progressPanel: $("#progress-panel"),
  progressLabel: $("#progress-label"),
  progressTime: $("#progress-time"),
  progressTrack: $("#progress-track"),
  progressDescription: $("#progress-description"),
  errorMessage: $("#error-message"),
  liveRegion: $("#live-region"),
  results: $("#results"),
  resultsTitle: $("#results-title"),
  resultMode: $("#result-mode"),
  analysisModel: $("#analysis-model"),
  metadataList: $("#metadata-list"),
  warnings: $("#warnings"),
  warningList: $("#warning-list"),
  audioPlayer: $("#audio-player"),
  playbackSource: $("#playback-source"),
  generatedAudio: $("#generated-audio"),
  generatedAudioPlayer: $("#generated-audio-player"),
  generatedAudioDetail: $("#generated-audio-detail"),
  generationDiagnostics: $("#generation-diagnostics"),
  generationStopStatus: $("#generation-stop-status"),
  generationStopDetail: $("#generation-stop-detail"),
  playbackTitle: $("#playback-title"),
  waveform: $("#waveform"),
  timelineKey: $("#timeline-key"),
  waveformCanvas: $("#waveform-canvas"),
  waveformSelection: $("#waveform-selection"),
  waveformTokenSelection: $("#waveform-token-selection"),
  waveformPlayhead: $("#waveform-playhead"),
  waveformHover: $("#waveform-hover"),
  currentTime: $("#current-time"),
  durationTime: $("#duration-time"),
  transcriptText: $("#transcript-text"),
  transcriptLabel: $("#transcript-label"),
  timingNote: $("#timing-note"),
  tokenCards: $("#token-cards"),
  lensLayout: $("#lens-layout"),
  encoderSection: $("#encoder-section"),
  encoderDescription: $("#encoder-description"),
  decoderSection: $("#decoder-section"),
  decoderSectionLabel: $("#decoder-section-label"),
  decoderTitle: $("#decoder-title"),
  decoderDescription: $("#decoder-description"),
  encoderScoreLabel: $("#encoder-score-label"),
  encoderPhoneMode: $("#encoder-phone-mode"),
  encoderPhoneSignatureToggle: $("#encoder-phone-signature-toggle"),
  encoderPhoneSignatureStatus: $("#encoder-phone-signature-status"),
  encoderTokenLengthFilter: $("#encoder-token-length-filter"),
  encoderMaxTokenLength: $("#encoder-max-token-length"),
  encoderLengthSummary: $("#encoder-length-summary"),
  decoderTokenLengthFilter: $("#decoder-token-length-filter"),
  decoderMaxTokenLength: $("#decoder-max-token-length"),
  decoderLengthSummary: $("#decoder-length-summary"),
  decoderFilterNote: $("#decoder-filter-note"),
  decoderScoreLabel: $("#decoder-score-label"),
  encoderNavigator: $("#encoder-navigator"),
  encoderWaveformCanvas: $("#encoder-waveform-canvas"),
  encoderWaveformSlices: $("#encoder-waveform-slices"),
  decoderNavigator: $("#decoder-navigator"),
  encoderFocusLabel: $("#encoder-focus-label"),
  decoderFocusLabel: $("#decoder-focus-label"),
  encoderLayers: $("#encoder-layers"),
  decoderLayers: $("#decoder-layers"),
  inspector: $("#inspector"),
  inspectorEmpty: $("#inspector-empty"),
  inspectorEmptyCopy: $("#inspector-empty-copy"),
  inspectorContent: $("#inspector-content"),
  inspectorKind: $("#inspector-kind"),
  inspectorCellTitle: $("#inspector-cell-title"),
  inspectorContext: $("#inspector-context"),
  scoreKicker: $("#score-kicker"),
  scoreDescription: $("#score-description"),
  selectedScore: $("#selected-score"),
  rankKeyLabel: $("#rank-key-label"),
  topkMetricLabel: $("#topk-metric-label"),
  topkList: $("#topk-list"),
  closeInspector: $("#close-inspector"),
  timelineTooltip: $("#lens-timeline-tooltip"),
  tooltipEyebrow: $("#lens-timeline-tooltip .lens-tooltip-eyebrow"),
  tooltipToken: $("#lens-timeline-tooltip .lens-tooltip-token"),
  tooltipTokenId: $("#lens-timeline-tooltip .lens-tooltip-token-id"),
  tooltipCoordinate: $("#lens-timeline-tooltip .lens-tooltip-coordinate"),
  tooltipMetrics: $("#lens-timeline-tooltip .lens-tooltip-metrics"),
  tooltipCandidatesLabel: $("#lens-timeline-tooltip .lens-tooltip-candidates-label"),
  tooltipCandidates: $("#lens-timeline-tooltip .lens-tooltip-candidates"),
};

const MODE_GUIDES = Object.freeze({
  asr: {
    label: "ASR · J-LENS GUIDE",
    title: "How J-lens applies to ASR",
    summary: "Whisper turns audio into text through a bidirectional audio encoder and a causal text decoder. Its two lens views answer different questions.",
    primaryTitle: "Decoder J-lens",
    primary: "A fitted, corpus-averaged Jacobian carries an intermediate decoder residual toward a later decoder state, which is then read through Whisper’s text head.",
    secondaryTitle: "Encoder-to-decoder extension",
    secondary: "A separately fitted cross-modal Jacobian maps a pooled 100 ms audio window into the final decoder’s vocabulary space.",
    implication: "A token becoming readable in later decoder layers can show when its lexical direction becomes accessible. An encoder peak can associate an audio location with a downstream lexical direction.",
    boundary: "Emission probability, a phoneme probability, a streaming belief, conscious consideration, or causal use. Whisper’s bidirectional encoder can use surrounding and later audio.",
    steps: [
      "Choose a prepared sample, record, or upload audio; select the encoder overlap and analyze.",
      "Select an orange output token or a waveform position to synchronize the output head, decoder, and nearest encoder window.",
      "Compare early and late timeline rows, then hover, focus, or choose a slice to inspect token IDs, scoped ranks, and exact metric values.",
      "Use token-length filters only as vocabulary reranking aids; they do not alter generation.",
      "Check run provenance and interpretation warnings before comparing results.",
    ],
    scale: "Orange values are raw teacher-forced output probabilities. Blue J-lens scores are uncalibrated lexical readouts shown by display rank—not percentages, confidence, or causal effect sizes.",
  },
  speech: {
    label: "SPEECH-TO-SPEECH · J-LENS GUIDE",
    title: "How J-lens applies to speech-to-speech",
    summary: "LFM2.5 consumes speech and generates interleaved text and audio. This page currently lenses only its causal language backbone.",
    primaryTitle: "Projected language J-lens",
    primary: "At each generated-text position, a fitted low-rank approximation carries an intermediate language residual toward the late language state, then reads it through LFM’s tied text head.",
    secondaryTitle: null,
    secondary: null,
    implication: "A highly ranked token is a lexical direction made readable by the approximate average transport. Compare when response-token directions become more or less prominent through the language backbone.",
    boundary: "Input-audio attribution, token timing, causal contribution, calibrated confidence, or an explanation of the generated audio. The FastConformer, audio adapter, codebook heads, voice, prosody, and waveform decoder are outside this lens.",
    steps: [
      "Choose a prepared clip, record, or upload spoken input, then analyze it.",
      "Use the input player and waveform only to hear or seek the model input; they do not select a language token.",
      "Play the generated speech as output, remembering that its acoustic path is not traced by this lens.",
      "Select a generated response token, compare the fitted language-layer cards with the tied-text-head probability, and open a card for its ranked candidates.",
      "Check projection rank, fit examples, checkpoint, and warnings in run provenance before interpreting fine token order.",
    ],
    scale: "The output-head card shows a raw teacher-forced token probability. Layer cards show rank-limited projected lexical scores—not probabilities, causal effects, or explanations of the waveform.",
  },
});

const state = {
  selectedFile: null,
  selectedSourceKind: "upload",
  result: null,
  waveform: [],
  duration: 0,
  audioUrl: "",
  recordUrl: "",
  requestController: null,
  analysisQueue: null,
  activeAnalysisJob: null,
  progressTimer: null,
  progressHideTimer: null,
  progressStartedAt: 0,
  resizeObserver: null,
  loading: false,
  serverMode: "asr",
  encoderPhoneSignatureAvailable: false,
  encoderPhoneSignatureEnabled: false,
  encoderTokenLengthFilterEnabled: false,
  encoderMaxTokenLength: 2,
  decoderTokenLengthFilterEnabled: false,
  decoderMaxTokenLength: 2,
  timing: { showRanges: false },
  views: {},
  selectedCellButton: null,
  mediaRecorder: null,
  mediaStream: null,
  recordChunks: [],
  recordBlob: null,
  recordFile: null,
  recordDuration: 0,
  recordStartedAt: 0,
  recordTimerHandle: null,
  recordAutoStopHandle: null,
  recordingPending: false,
  timelineSelection: {
    timeSeconds: null,
    encoderIndex: null,
    decoderIndex: null,
    tokenIndex: null,
    tokenMatch: "unavailable",
    origin: null,
  },
  inspectorSelection: { kind: null, layerIndex: null },
  timelineTooltip: { trigger: null, mode: null, chart: null },
  restoringTimelineFocus: false,
};

function clamp(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, Number(value) || 0));
}

function asFiniteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatTime(seconds, precise = true) {
  const safe = Math.max(0, asFiniteNumber(seconds));
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${minutes}:${precise ? remainder.toFixed(1).padStart(4, "0") : Math.floor(remainder).toString().padStart(2, "0")}`;
}

function formatScore(score) {
  const value = asFiniteNumber(score, NaN);
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function formatProbability(probability) {
  return `${Math.round(clamp(probability) * 100)}%`;
}

function formatApproximateWait(seconds) {
  const value = finiteNumberOrNull(seconds);
  if (value === null) return null;
  const safe = Math.max(0, value);
  if (safe < 2) return "a few seconds";
  if (safe < 60) return `about ${Math.ceil(safe)} seconds`;
  const minutes = Math.ceil(safe / 60);
  return `about ${minutes} minute${minutes === 1 ? "" : "s"}`;
}

function formatProbabilityPrecise(probability, digits = 3) {
  const value = finiteNumberOrNull(probability);
  return value === null ? "—" : `${(clamp(value) * 100).toFixed(digits)}%`;
}

function visibleToken(text) {
  const value = String(text ?? "");
  if (!value) return "<empty>";
  return value.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function timedRange(item) {
  const start = finiteNumberOrNull(item?.start_seconds);
  const end = finiteNumberOrNull(item?.end_seconds);
  if (start === null || end === null || end < start) return null;
  return { start, end };
}

function analysisDuration() {
  if (state.duration > 0) return state.duration;
  const ranges = [
    ...(state.views.encoder?.columns || []),
    ...(state.result?.transcription?.tokens || []),
  ].map(timedRange).filter(Boolean);
  return ranges.length ? Math.max(...ranges.map((range) => range.end)) : 0;
}

function rangeIndexAtTime(columns, seconds, { nearest = false, preferredIndex = null } = {}) {
  const time = Math.max(0, asFiniteNumber(seconds));
  const candidates = (Array.isArray(columns) ? columns : [])
    .map((column, index) => ({ index, range: timedRange(column) }))
    .filter((candidate) => candidate.range);
  if (!candidates.length) return null;
  const finalCandidate = candidates[candidates.length - 1];
  const matches = candidates.filter((candidate) => {
    const { start, end } = candidate.range;
    if (start === end) return Math.abs(time - start) < 1e-7;
    return time >= start && (time < end || (candidate === finalCandidate && time <= end));
  });
  if (matches.length) {
    const preferred = matches.find((candidate) => candidate.index === preferredIndex);
    if (preferred) return preferred.index;
    matches.sort((left, right) => (left.range.end - left.range.start) - (right.range.end - right.range.start) || left.index - right.index);
    return matches[0].index;
  }
  if (!nearest) return null;
  candidates.sort((left, right) => {
    const distance = (candidate) => time < candidate.range.start ? candidate.range.start - time : time > candidate.range.end ? time - candidate.range.end : 0;
    return distance(left) - distance(right) || left.index - right.index;
  });
  return candidates[0].index;
}

function tokenIndexForDecoderPosition(positionIndex) {
  const view = state.views.decoder;
  const reportedIndex = finiteNumberOrNull(view?.columns?.[positionIndex]?.index);
  return reportedIndex === null ? positionIndex : Math.max(0, Math.trunc(reportedIndex));
}

function decoderPositionForTokenIndex(tokenIndex) {
  const view = state.views.decoder;
  if (!view) return null;
  const hasExplicitIndices = view.columns.some((column) => finiteNumberOrNull(column?.index) !== null);
  const exact = view.columns.findIndex((column) => {
    const reportedIndex = finiteNumberOrNull(column?.index);
    return reportedIndex !== null && Math.trunc(reportedIndex) === tokenIndex;
  });
  if (exact >= 0) return exact;
  if (hasExplicitIndices) return null;
  return tokenIndex >= 0 && tokenIndex < view.columns.length ? tokenIndex : null;
}

function setSelectedOutputToken(tokenIndex) {
  const selected = Number.isInteger(tokenIndex) ? tokenIndex : null;
  elements.tokenCards.querySelectorAll(".token-card").forEach((card) => {
    const active = selected !== null && Number(card.dataset.tokenIndex) === selected;
    card.classList.toggle("selected", active);
    card.classList.toggle("nearest", active && state.timelineSelection.tokenMatch === "nearest");
    card.setAttribute("aria-pressed", String(active));
  });
}

function updateEncoderSliderAria(index = state.timelineSelection.encoderIndex) {
  const view = state.views.encoder;
  if (!view || index === null || !view.columns[index]) return;
  elements.encoderNavigator.setAttribute("aria-valuenow", String(index + 1));
  let alignment = "no output-token timing alignment";
  const token = state.result?.transcription?.tokens?.[state.timelineSelection.tokenIndex];
  if (token && state.timelineSelection.tokenMatch === "covering") alignment = `aligned output token ${visibleToken(token.text)}`;
  else if (token && state.timelineSelection.tokenMatch === "nearest") alignment = `nearest approximate output token ${visibleToken(token.text)}`;
  elements.encoderNavigator.setAttribute("aria-valuetext", `${lensContext("encoder", view.columns[index], index)}; ${alignment}`);
}

function updateTimelineAria() {
  const duration = analysisDuration();
  const time = finiteNumberOrNull(state.timelineSelection.timeSeconds);
  const progress = time === null || !duration ? 0 : clamp(time / duration);
  elements.waveform.setAttribute("aria-valuenow", String(Math.round(progress * 100)));
  const token = state.result?.transcription?.tokens?.[state.timelineSelection.tokenIndex];
  if (time === null) {
    elements.waveform.setAttribute("aria-valuetext", token ? `Selected output token ${visibleToken(token.text)}; token timing unavailable` : "No synchronized timeline position selected");
    return;
  }
  let tokenText = "no timed output token";
  if (token && state.timelineSelection.tokenMatch === "covering") tokenText = `aligned output token ${visibleToken(token.text)}`;
  else if (token && state.timelineSelection.tokenMatch === "nearest") tokenText = `nearest approximate output token ${visibleToken(token.text)}`;
  elements.waveform.setAttribute("aria-valuetext", `Selected ${formatTime(time)} of ${formatTime(duration)}; ${tokenText}`);
}

function positionWaveformRange(element, range) {
  const duration = analysisDuration();
  if (!element || !range || !duration) {
    if (element) element.hidden = true;
    return;
  }
  const start = clamp(range.start / duration);
  const end = clamp(range.end / duration);
  element.style.left = `${start * 100}%`;
  element.style.width = `${Math.max(0.35, (end - start) * 100)}%`;
  element.hidden = false;
}

function updateTimelineOverlays(encoderRange, tokenRange) {
  positionWaveformRange(elements.waveformSelection, encoderRange);
  positionWaveformRange(elements.waveformTokenSelection, tokenRange);
}

function showDecoderTimeGap(seconds) {
  const view = state.views.decoder;
  if (!view) return;
  view.timelineUnmatched = true;
  [...view.cellButtons, ...view.headButtons].forEach((button) => {
    button.classList.remove("selected-column", "selected-coordinate");
    button.setAttribute("aria-pressed", "false");
    button.tabIndex = -1;
  });
  elements.decoderFocusLabel.textContent = `No safely matched decoder position at ${asFiniteNumber(seconds).toFixed(2)}s`;
}

function selectTimelineToken(tokenIndex, { origin = "output", seek = false, play = false, announceChange = true } = {}) {
  const tokens = state.result?.transcription?.tokens || [];
  if (!tokens.length) return;
  const safeTokenIndex = Math.max(0, Math.min(tokens.length - 1, Math.trunc(asFiniteNumber(tokenIndex))));
  const token = tokens[safeTokenIndex];
  const tokenRange = state.timing.showRanges ? timedRange(token) : null;
  const decoderIndex = decoderPositionForTokenIndex(safeTokenIndex);
  let encoderIndex = state.views.encoder?.pinnedIndex ?? null;
  let encoderRange = null;
  let timeSeconds = null;

  if (tokenRange) {
    timeSeconds = (tokenRange.start + tokenRange.end) / 2;
    encoderIndex = rangeIndexAtTime(state.views.encoder?.columns, timeSeconds, { nearest: true, preferredIndex: encoderIndex });
    encoderRange = encoderIndex === null ? null : timedRange(state.views.encoder.columns[encoderIndex]);
  }
  state.timelineSelection = { timeSeconds, encoderIndex, decoderIndex, tokenIndex: safeTokenIndex, tokenMatch: tokenRange ? "covering" : "unavailable", origin };
  if (tokenRange && encoderIndex !== null) showStreamPosition("encoder", encoderIndex, { pin: true, announceChange: false, updateInspector: false });
  if (decoderIndex !== null) showStreamPosition("decoder", decoderIndex, { pin: true, announceChange: false, updateInspector: false });
  setSelectedOutputToken(safeTokenIndex);
  updateTimelineOverlays(encoderRange, tokenRange);
  updateEncoderSliderAria(encoderIndex);
  updateTimelineAria();
  if (decoderIndex !== null) inspectLayerForView("decoder", state.views.decoder.pinnedLayerIndex, { pin: true, announceChange: false });
  else if (encoderIndex !== null) inspectLayerForView("encoder", state.views.encoder.pinnedLayerIndex, { pin: true, announceChange: false });
  if (seek && tokenRange) seekAudio(timeSeconds, play);
  if (announceChange) {
    const isSpeechToSpeech = isMlxLfmInfo(state.result?.metadata || {});
    let mapping = isSpeechToSpeech
      ? "Language J-lens layers and tied text head synchronized; response-token timing is unavailable, so input-audio playback is unchanged."
      : "Output head and decoder layers synchronized; token timing is unavailable, so the encoder slice is unchanged.";
    if (tokenRange && encoderIndex !== null) mapping = "Output head, decoder layers, and representative encoder slice synchronized.";
    else if (tokenRange) mapping = "Output head and decoder layers synchronized; no encoder stream is available.";
    announce(`${visibleToken(token.text)} selected. ${mapping}`);
  }
}

function selectTimelineAtTime(seconds, { origin = "waveform", seek = true, announceChange = true } = {}) {
  const duration = analysisDuration();
  if (!duration) return;
  const timeSeconds = clamp(asFiniteNumber(seconds) / duration) * duration;
  const encoderIndex = rangeIndexAtTime(state.views.encoder?.columns, timeSeconds, {
    nearest: true,
    preferredIndex: state.timelineSelection.encoderIndex,
  });
  const encoderRange = encoderIndex === null ? null : timedRange(state.views.encoder.columns[encoderIndex]);

  if (!state.timing.showRanges) {
    const tokenIndex = state.timelineSelection.tokenIndex;
    const decoderIndex = state.timelineSelection.decoderIndex;
    state.timelineSelection = { timeSeconds, encoderIndex, decoderIndex, tokenIndex, tokenMatch: "unavailable", origin };
    if (encoderIndex !== null) showStreamPosition("encoder", encoderIndex, { pin: true, announceChange: false, updateInspector: false });
    setSelectedOutputToken(tokenIndex);
    updateTimelineOverlays(encoderRange, null);
    updateEncoderSliderAria(encoderIndex);
    updateTimelineAria();
    if (encoderIndex !== null) inspectLayerForView("encoder", state.views.encoder.pinnedLayerIndex, { pin: true, announceChange: false });
    if (seek) seekAudio(timeSeconds);
    if (announceChange) {
      const message = encoderIndex !== null
        ? `Audio position ${timeSeconds.toFixed(2)} seconds selected. Encoder slice updated; token timing is unavailable, so the pinned output-head and decoder token are unchanged.`
        : `Input audio position ${timeSeconds.toFixed(2)} seconds selected for playback. Response tokens have no audio timestamps, so the pinned language J-lens and text-head token are unchanged.`;
      announce(message);
    }
    return;
  }

  const coveringTokenIndex = rangeIndexAtTime(state.result?.transcription?.tokens, timeSeconds, { preferredIndex: state.timelineSelection.tokenIndex });
  const tokenIndex = coveringTokenIndex ?? rangeIndexAtTime(state.result?.transcription?.tokens, timeSeconds, { nearest: true, preferredIndex: state.timelineSelection.tokenIndex });
  const tokenMatch = coveringTokenIndex === null && tokenIndex !== null ? "nearest" : tokenIndex === null ? "unavailable" : "covering";
  const decoderIndex = tokenIndex === null ? null : decoderPositionForTokenIndex(tokenIndex);
  const tokenRange = tokenIndex === null ? null : timedRange(state.result.transcription.tokens[tokenIndex]);

  state.timelineSelection = { timeSeconds, encoderIndex, decoderIndex, tokenIndex, tokenMatch, origin };
  if (encoderIndex !== null) showStreamPosition("encoder", encoderIndex, { pin: true, announceChange: false, updateInspector: false });
  if (decoderIndex !== null) showStreamPosition("decoder", decoderIndex, { pin: true, announceChange: false, updateInspector: false });
  else showDecoderTimeGap(timeSeconds);
  if (tokenMatch === "nearest" && decoderIndex !== null) elements.decoderFocusLabel.textContent += " · nearest approximate interval";
  setSelectedOutputToken(tokenIndex);
  updateTimelineOverlays(encoderRange, tokenRange);
  updateEncoderSliderAria(encoderIndex);
  updateTimelineAria();
  if (encoderIndex !== null) inspectLayerForView("encoder", state.views.encoder.pinnedLayerIndex, { pin: true, announceChange: false });
  else if (decoderIndex !== null) inspectLayerForView("decoder", state.views.decoder.pinnedLayerIndex, { pin: true, announceChange: false });
  if (seek) seekAudio(timeSeconds);
  if (announceChange) {
    let tokenMessage = "No approximate output-token interval is available.";
    if (tokenIndex !== null && tokenMatch === "covering") tokenMessage = `${visibleToken(state.result.transcription.tokens[tokenIndex].text)} is the aligned output token.`;
    else if (tokenIndex !== null) tokenMessage = `${visibleToken(state.result.transcription.tokens[tokenIndex].text)} is the nearest approximate output token.`;
    const encoderMessage = encoderIndex === null ? "No encoder stream is available." : "Encoder slice updated.";
    announce(`Audio position ${timeSeconds.toFixed(2)} seconds selected. ${encoderMessage} ${tokenMessage}`);
  }
}

function initializeTimelineSelection() {
  const tokens = state.result?.transcription?.tokens || [];
  if (tokens.length) selectTimelineToken(0, { origin: "initial", announceChange: false });
  else if (state.views.encoder?.columns?.length) {
    const range = timedRange(state.views.encoder.columns[0]);
    if (range) selectTimelineAtTime((range.start + range.end) / 2, { origin: "initial", seek: false, announceChange: false });
  }
}

function availableStreams(payload) {
  const reported = payload?.metadata?.streams;
  if (Array.isArray(reported)) return new Set(reported.filter((value) => value === "encoder" || value === "decoder"));
  const inferred = new Set();
  ["encoder", "decoder"].forEach((kind) => {
    const data = payload?.[kind];
    if (Array.isArray(data?.layers) && data.layers.length && Array.isArray(data?.cells) && data.cells.length) inferred.add(kind);
  });
  return inferred;
}

function timingDetails(transcription, mode) {
  const source = String(transcription?.timing_source || "");
  const quality = String(transcription?.timing_quality || "");
  const tokens = Array.isArray(transcription?.tokens) ? transcription.tokens : [];
  const hasRanges = tokens.some((token) => timedRange(token));
  if (mode === "demo" || source === "synthetic_demo") {
    return hasRanges
      ? { showRanges: true, label: "Token ranges are synthetic and exist only to exercise the interface.", metadataLabel: "Synthetic demo timing" }
      : { showRanges: false, label: "Synthetic response positions have no audio timing; input-waveform playback is not linked to generated text tokens.", metadataLabel: "Synthetic demo · timing unavailable" };
  }
  if (source === "unavailable" || quality === "unavailable") return { showRanges: false, label: "Token timing is unavailable; exact ranges and seek actions are omitted.", metadataLabel: "Unavailable" };
  if (source === "whisper_cross_attention_dtw" && hasRanges) return { showRanges: true, label: "Token ranges are approximate Whisper cross-attention/DTW alignments, not exact word boundaries.", metadataLabel: "Whisper cross-attention/DTW (approximate)" };
  if (!source && hasRanges) return { showRanges: true, label: "This legacy response did not report a timing source; ranges should be treated as approximate.", metadataLabel: "Legacy ranges; source not reported" };
  if (quality === "model_derived" && hasRanges) return { showRanges: true, label: "Token ranges are model-derived estimates, not exact word boundaries.", metadataLabel: source || "Model-derived (approximate)" };
  return { showRanges: false, label: "No supported token timing provenance was reported; exact ranges and seek actions are omitted.", metadataLabel: source ? `${source} (not displayed)` : "Not reported" };
}

function displayVocabularySummary(metadata) {
  const value = metadata?.display_vocabulary;
  if (typeof value === "string" && value) return value;
  if (!value || typeof value !== "object") return "Lexical display filter; policy not reported";
  const policyNames = { alphanumeric_lexical_tokens: "Alphanumeric lexical tokens", synthetic_demo_lexical_tokens: "Synthetic demo lexical tokens" };
  const policy = policyNames[value.policy] || String(value.policy || "Lexical filter").replaceAll("_", " ");
  const shown = finiteNumberOrNull(value.display_vocabulary_size);
  const full = finiteNumberOrNull(value.full_vocabulary_size);
  return shown !== null && full !== null ? `${policy} · ${shown.toLocaleString()} of ${full.toLocaleString()} tokens` : policy;
}

function scoreKindDetails(kind, data) {
  const scoreKind = String(data?.score_kind || "");
  if (scoreKind === "target_mean_relative_logit_delta") return {
    shortLabel: "target-mean-relative readout-logit delta",
    metadataLabel: "Target-mean-relative readout-logit delta",
    columnLabel: "relative logit delta",
    description: "The fitted corpus target-mean logit is subtracted before lexical ranking. Its sign is neither causal polarity nor model probability.",
  };
  if (scoreKind === "raw_readout_logit") return {
    shortLabel: "raw J-lens readout logit",
    metadataLabel: "Raw J-lens readout logit",
    columnLabel: "raw logit",
    description: "This raw logit ranks tokens in the display-filtered vocabulary. It is not the model’s output probability or a signed causal effect.",
  };
  const fallback = scoreKind ? `Reported score: ${scoreKind.replaceAll("_", " ")}` : `Legacy ${kind} score; definition not reported`;
  return { shortLabel: fallback.toLowerCase(), metadataLabel: fallback, columnLabel: "reported score", description: "The response did not provide a recognized score definition. Do not interpret this value as probability or causal polarity." };
}

function announce(message) {
  elements.liveRegion.textContent = "";
  window.setTimeout(() => { elements.liveRegion.textContent = message; }, 20);
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = false;
  announce(message);
}

function clearError() {
  elements.errorMessage.hidden = true;
  elements.errorMessage.textContent = "";
}

function isMlxLfmInfo(info = {}) {
  const backend = String(info.backend || "");
  const modelFamily = String(info.model_family || "");
  const modelId = String(info.model_id || info.model || "");
  const capabilities = info.capabilities || {};
  const languageOnlySpeech = Boolean(
    capabilities.language_jlens
    && capabilities.generated_text
    && !capabilities.audio_encoder_jlens
    && !capabilities.audio_codebook_jlens
  );
  return backend === "mlx-lfm"
    || (backend === "mlx" && modelFamily === "lfm2_audio")
    || languageOnlySpeech
    || modelId.includes("LFM2.5-Audio");
}

function applyModeGuide(guide) {
  elements.modeGuideLabel.textContent = guide.label;
  elements.modeGuideTitle.textContent = guide.title;
  elements.modeGuideSummary.textContent = guide.summary;
  elements.modeGuidePrimaryTitle.textContent = guide.primaryTitle;
  elements.modeGuidePrimary.textContent = guide.primary;
  elements.modeGuideSecondaryRow.hidden = !guide.secondary;
  if (guide.secondary) {
    elements.modeGuideSecondaryTitle.textContent = guide.secondaryTitle;
    elements.modeGuideSecondary.textContent = guide.secondary;
  }
  elements.modeGuideImplication.textContent = guide.implication;
  elements.modeGuideBoundary.textContent = guide.boundary;
  elements.modeGuideSteps.replaceChildren(
    ...guide.steps.map((step) => createElement("li", "", step)),
  );
  elements.modeGuideScale.replaceChildren(
    createElement("strong", "", "Read the scales separately."),
    document.createTextNode(` ${guide.scale}`),
  );
}

function applyBackendBranding(info = {}) {
  const backend = String(info.backend || "");
  const modelId = String(info.model_id || info.model || "");
  const isMlxLfm = isMlxLfmInfo(info);
  const isWhisper = backend === "whisper-hf" || modelId.toLowerCase().includes("whisper");
  document.body.dataset.workspace = isMlxLfm ? "speech" : "asr";
  if (isMlxLfm) {
    applyModeGuide(MODE_GUIDES.speech);
    elements.brandSubtitle.textContent = "Local MLX speech-to-speech workspace";
    elements.modelBadge.textContent = "LFM2.5 · MLX";
    elements.heroTitle.textContent = "Inspect how LFM2.5’s language backbone forms a response.";
    elements.heroCopy.textContent = "Give the model spoken input, then compare final text-token probabilities with projected lexical readouts across language layers. Generated speech is playable, but its acoustic-code and waveform path is not traced.";
    elements.probabilityPrimerTitle.textContent = "Output-head probability";
    elements.probabilityPrimerCopy.textContent = "Tied text head · raw teacher-forced probability · 0–100%";
    elements.lensPrimerTitle.textContent = "Projected language J-lens";
    elements.lensPrimerCopy.textContent = "Rank-limited lexical score · not probability or causal effect";
    elements.progressDescription.textContent = "LFM2.5 is generating speech and computing projected internal vocabulary readouts.";
    elements.decoderSectionLabel.textContent = "04 · LANGUAGE J-LENS";
    elements.decoderTitle.textContent = "Through the language backbone";
    elements.decoderDescription.textContent = "Choose a generated text token. Each layer shows which lexical candidates the projected average Jacobian maps toward the final text head.";
    elements.playbackTitle.textContent = "Input audio and generated response";
    elements.transcriptLabel.textContent = "Generated response text";
    elements.tokenCards.setAttribute("aria-label", "Generated response token probabilities");
    elements.waveform.setAttribute("aria-label", "Input audio playback timeline; seeking does not select or align a generated response token");
    elements.timelineKey.hidden = true;
    elements.inspectorEmptyCopy.textContent = "Choose a response token, then hover or select a language layer to inspect its ranked lexical candidates.";
    elements.decoderFilterNote.hidden = true;
    elements.encoderAnalysisSettings.hidden = true;
    return;
  }
  applyModeGuide(MODE_GUIDES.asr);
  elements.encoderAnalysisSettings.hidden = false;
  elements.heroCopy.textContent = "Compare the model’s emitted-token probability with layer-by-layer Jacobian Lens readouts. Focus one sound or token at a time—without navigating a sprawling matrix.";
  elements.probabilityPrimerTitle.textContent = "Output-head probability";
  elements.probabilityPrimerCopy.textContent = "Whisper LM head · raw teacher-forced probability · 0–100%";
  elements.lensPrimerTitle.textContent = "J-lens intensity";
  elements.lensPrimerCopy.textContent = "Relative rank within the current view · not a percent";
  elements.waveform.setAttribute("aria-label", "Audio waveform timeline; selecting a position synchronizes output, encoder, and decoder views");
  elements.timelineKey.hidden = false;
  elements.inspectorEmptyCopy.textContent = "Select a time or token, then hover or choose a layer to inspect its ranked lexical candidates.";
  if (!state.result) elements.decoderFilterNote.hidden = false;
  elements.encoderDescription.textContent = "Select a pooled audio window to see which downstream text-token directions become readable across encoder layers. This cross-modal view is experimental and baseline-relative.";
  if (isWhisper) {
    elements.brandSubtitle.textContent = "Whisper interpretability workspace";
    elements.modelBadge.textContent = "WHISPER · TINY";
    elements.heroTitle.textContent = "Trace audio through Whisper’s vocabulary space.";
    elements.progressDescription.textContent = "Whisper is transcribing and computing internal vocabulary readouts.";
  } else {
    elements.brandSubtitle.textContent = "Speech-model interpretability workspace";
    elements.modelBadge.textContent = "AUDIO MODEL";
    elements.heroTitle.textContent = "Trace audio through a model’s vocabulary space.";
    elements.progressDescription.textContent = "The model is generating and computing internal vocabulary readouts.";
  }
  elements.decoderSectionLabel.textContent = "05 · DECODER LENS";
  elements.decoderTitle.textContent = "As each token resolves";
  elements.decoderDescription.textContent = "Select an emitted token to see when its text direction—and competing directions—become readable across causal decoder layers.";
  elements.playbackTitle.textContent = "Audio and transcription";
  elements.transcriptLabel.textContent = "Actual model output";
  elements.tokenCards.setAttribute("aria-label", "Transcription token probabilities");
}

function streamDisplayName(kind) {
  const configured = state.result?.metadata?.stream_labels?.[kind];
  if (typeof configured === "string" && configured.trim()) return configured.trim();
  return kind === "encoder" ? "Encoder" : "Decoder";
}

function generationDiagnosticsFor(payload) {
  const candidates = [
    payload?.metadata?.generation_diagnostics,
    payload?.model?.generation_diagnostics,
    payload?.generation_diagnostics,
  ];
  const raw = candidates.find((value) => value && typeof value === "object" && !Array.isArray(value));
  if (!raw) return null;

  const count = (value) => {
    const number = finiteNumberOrNull(value);
    return number === null || number < 0 ? null : Math.trunc(number);
  };
  const reason = String(raw.termination_reason || "").trim().toLowerCase();
  const generatedSteps = count(raw.generated_steps);
  const maxNewTokens = count(raw.max_new_tokens);
  const textTokens = count(raw.text_tokens);
  const audioFrames = count(raw.audio_frames);
  const audioEosSeen = typeof raw.audio_eos_seen === "boolean" ? raw.audio_eos_seen : null;
  const budgetExhausted = raw.budget_exhausted === true
    || reason.includes("budget")
    || reason.includes("max_new");
  const completedAtAudioEos = reason === "audio_eos";

  let status = "Generation stop reported";
  if (budgetExhausted) status = "Stopped at generation budget";
  else if (completedAtAudioEos) status = "Completed at audio EOS";
  else if (reason) status = `Stopped: ${reason.replaceAll("_", " ")}`;

  const details = [];
  if (generatedSteps !== null && maxNewTokens !== null) details.push(`${generatedSteps} / ${maxNewTokens} interleaved steps`);
  else if (generatedSteps !== null) details.push(`${generatedSteps} interleaved steps`);
  else if (maxNewTokens !== null) details.push(`budget ${maxNewTokens} interleaved steps`);
  if (textTokens !== null) details.push(`${textTokens} text tokens`);
  if (audioFrames !== null) details.push(`${audioFrames} audio frames`);
  if (audioEosSeen !== null) details.push(audioEosSeen ? "audio EOS observed" : "audio EOS not observed");

  return {
    status,
    detail: details.join(" · ") || "Step counts were not reported.",
    budgetExhausted,
    completedAtAudioEos,
  };
}

function renderGeneratedAudio(payload, mode) {
  elements.generatedAudioPlayer.pause();
  elements.generatedAudioPlayer.removeAttribute("src");
  elements.generationDiagnostics.hidden = true;
  elements.generationDiagnostics.classList.remove("budget-exhausted", "completed-at-eos");
  elements.generationStopStatus.textContent = "";
  elements.generationStopDetail.textContent = "";
  const output = mode === "demo" ? null : payload.audio?.model_output_wav;
  const available = typeof output === "string" && output.startsWith("data:audio/");
  elements.generatedAudio.hidden = !available;
  if (!available) {
    elements.generatedAudioPlayer.load();
    return;
  }
  elements.generatedAudioPlayer.src = output;
  elements.generatedAudioPlayer.load();
  const duration = finiteNumberOrNull(payload.audio?.model_output_duration_seconds);
  const format = payload.audio?.model_output_format || "generated PCM audio";
  const prefix = `${format}${duration === null ? "" : ` · ${duration.toFixed(2)} s`}`;
  elements.generatedAudioDetail.textContent = isMlxLfmInfo(payload.metadata || {})
    ? `${prefix} · playback only; the current language J-lens does not explain audio codes, timing, voice, prosody, or waveform samples`
    : `${prefix} · produced by the acoustic-token path`;
  const diagnostics = generationDiagnosticsFor(payload);
  if (diagnostics) {
    elements.generationStopStatus.textContent = diagnostics.status;
    elements.generationStopDetail.textContent = diagnostics.detail;
    elements.generationDiagnostics.classList.toggle("budget-exhausted", diagnostics.budgetExhausted);
    elements.generationDiagnostics.classList.toggle("completed-at-eos", diagnostics.completedAtAudioEos && !diagnostics.budgetExhausted);
    elements.generationDiagnostics.hidden = false;
  }
}

function setRecordError(message = "") {
  elements.recordError.textContent = message;
  elements.recordError.hidden = !message;
  if (message) announce(message);
}

function recordingInProgress() {
  return state.recordingPending || state.mediaRecorder?.state === "recording";
}

function analysisQueueCapability(value) {
  if (!value || typeof value !== "object" || value.enabled !== true) return null;
  return {
    enabled: true,
    capacity: Math.max(0, Math.trunc(asFiniteNumber(value.capacity))),
    queued: Math.max(0, Math.trunc(asFiniteNumber(value.queued))),
    running: Math.max(0, Math.trunc(asFiniteNumber(value.running))),
    averageSeconds: finiteNumberOrNull(value.average_seconds ?? value.estimated_job_seconds),
  };
}

async function checkBackendStatus() {
  elements.statusDot.className = "status-dot";
  elements.statusText.textContent = "Checking backend…";
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(API.status, { headers: { Accept: "application/json" }, signal: controller.signal });
    if (!response.ok) throw new Error(`Status endpoint returned ${response.status}`);
    const payload = await response.json();
    document.body.dataset.asrOnly = payload.asr_only === true ? "true" : "false";
    elements.hostedNotice.hidden = payload.asr_only !== true;
    state.serverMode = isMlxLfmInfo(payload) ? "speech" : "asr";
    state.analysisQueue = analysisQueueCapability(payload.analysis_queue);
    applyBackendBranding(payload);
    const ready = payload.ready ?? payload.ok ?? true;
    const model = payload.model_id || payload.model || "Audio backend";
    elements.statusDot.className = `status-dot ${ready ? "online" : "offline"}`;
    elements.statusText.textContent = ready ? "Backend ready" : "Backend not ready";
    elements.statusButton.setAttribute("aria-label", `Backend status: ${ready ? "ready" : "not ready"}. Refresh status.`);
    const queueDetail = state.analysisQueue
      ? ` Queue active${state.analysisQueue.capacity ? ` with ${state.analysisQueue.capacity} waiting slots` : ""}.`
      : "";
    elements.statusButton.title = `${model}: ${payload.message || (ready ? "ready" : "not ready")}.${queueDetail} Click to refresh.`;
  } catch (error) {
    state.analysisQueue = null;
    elements.statusDot.className = "status-dot offline";
    elements.statusText.textContent = "Demo available";
    elements.statusButton.setAttribute("aria-label", "Backend unavailable. Synthetic demo is available. Refresh status.");
    elements.statusButton.title = "The backend is unavailable. Click to retry.";
  } finally {
    window.clearTimeout(timeout);
  }
}

function activateSourceTab(name, { focus = false } = {}) {
  if (recordingInProgress() && name !== "record") return;
  elements.sourceTabs.forEach((tab) => {
    const active = tab.dataset.sourceTab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  elements.sourcePanels.forEach((panel) => { panel.hidden = panel.id !== `${name}-panel`; });
}

async function loadSamples() {
  try {
    const response = await fetch(API.samples, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Prepared samples endpoint returned ${response.status}`);
    const payload = await response.json();
    const samples = Array.isArray(payload.samples) ? payload.samples : [];
    elements.sampleList.replaceChildren();
    if (!samples.length) {
      const message = createElement("p", "", "No prepared samples are installed yet. Record, upload, or load the synthetic UI demo.");
      elements.sampleList.append(message);
      return;
    }
    samples.forEach((sample) => {
      const button = createElement("button", "sample-card");
      button.type = "button";
      button.dataset.sampleId = sample.id || "";
      const phoneExample = sample.recommended_for === "phone-signature";
      button.classList.toggle("recommended-sample", phoneExample);
      if (sample.badge) button.append(createElement("span", "sample-badge", sample.badge));
      const top = createElement("span", "sample-card-top");
      const duration = finiteNumberOrNull(sample.duration_seconds);
      top.append(
        createElement("strong", "", sample.title || sample.id || "Audio sample"),
        createElement("span", "sample-duration", duration === null ? "Duration unknown" : formatTime(duration)),
      );
      button.append(top, createElement("span", "sample-description", sample.description || "Prepared speech sample"));
      if (sample.transcript) button.append(createElement("span", "sample-transcript", `“${sample.transcript}”`));
      button.append(createElement("span", "sample-action", phoneExample ? "Analyze phone example →" : "Analyze sample →"));
      button.setAttribute("aria-label", `Analyze prepared sample ${sample.title || sample.id || "audio"}`);
      button.addEventListener("click", () => loadSample(sample, button));
      elements.sampleList.append(button);
    });
    setLoading(state.loading);
  } catch (error) {
    elements.sampleList.replaceChildren(createElement("p", "", "Prepared samples are unavailable. You can still record, upload, or use the synthetic UI demo."));
  }
}

async function loadSample(sample, button) {
  if (state.loading || recordingInProgress()) return;
  const audioUrl = sample.audio_url || sample.url;
  if (!audioUrl) {
    showError("This prepared sample does not provide an audio URL.");
    return;
  }
  clearError();
  button.classList.add("loading");
  button.querySelector(".sample-action").textContent = "Loading audio…";
  try {
    const response = await fetch(audioUrl);
    if (!response.ok) throw new Error(`Sample audio returned ${response.status}`);
    const blob = await response.blob();
    const filename = sample.filename || `${sample.id || "sample"}.${blob.type.includes("mpeg") ? "mp3" : "wav"}`;
    const file = new File([blob], filename, { type: blob.type || sample.media_type || "audio/wav" });
    selectFile(file, { kind: "sample", title: sample.title || filename, detail: sample.description || sample.transcript || "Prepared sample" });
    await analyzeSelectedFile();
  } catch (error) {
    showError(`Could not load this prepared sample: ${error.message}`);
  } finally {
    button.classList.remove("loading");
    const action = button.querySelector(".sample-action");
    if (action) action.textContent = sample.recommended_for === "phone-signature" ? "Analyze phone example →" : "Analyze sample →";
  }
}

function selectFile(file, { kind = "upload", title = file?.name, detail = "" } = {}) {
  if (!file || state.loading || recordingInProgress()) return;
  if (file.type && !file.type.startsWith("audio/")) {
    showError("That file does not appear to be audio. Choose a supported audio file.");
    return;
  }
  state.selectedFile = file;
  state.selectedSourceKind = kind;
  const sizeMb = file.size / 1024 / 1024;
  const size = `${sizeMb < 0.1 ? "< 0.1" : sizeMb.toFixed(1)} MB`;
  elements.fileLabel.textContent = file.name;
  elements.fileDetail.textContent = `${size} · ready to analyze`;
  elements.analyzeButton.disabled = false;
  elements.analyzeLabel.textContent = "Analyze selected audio";
  elements.selectedTitle.textContent = title || file.name;
  elements.selectedDetail.textContent = detail ? `${detail} · ${size}` : `${size} · ready to analyze`;
  elements.selectedSource.hidden = kind === "upload" || kind === "recording";
  clearError();
  announce(`${title || file.name} selected.`);
}

function defaultAnalysisProgressDescription() {
  if (state.serverMode === "speech") return "LFM2.5 is generating speech and computing projected internal vocabulary readouts.";
  return "Whisper is transcribing and computing internal vocabulary readouts.";
}

function startProgress() {
  window.clearInterval(state.progressTimer);
  window.clearTimeout(state.progressHideTimer);
  state.progressStartedAt = Date.now();
  elements.progressPanel.hidden = false;
  elements.progressTrack.classList.remove("complete");
  elements.progressTrack.setAttribute("aria-label", "Audio analysis is in progress");
  elements.progressLabel.textContent = "Analyzing audio…";
  elements.progressTime.textContent = "00:00";
  elements.progressDescription.textContent = defaultAnalysisProgressDescription();
  state.progressTimer = window.setInterval(() => {
    elements.progressTime.textContent = formatTime(Math.floor((Date.now() - state.progressStartedAt) / 1000), false);
  }, 1000);
}

function finishProgress(success) {
  window.clearInterval(state.progressTimer);
  window.clearTimeout(state.progressHideTimer);
  state.progressTimer = null;
  if (success) {
    elements.progressTrack.classList.add("complete");
    elements.progressTrack.setAttribute("aria-label", "Audio analysis is complete");
    elements.progressLabel.textContent = "Analysis ready";
    elements.progressPanel.hidden = true;
  } else {
    elements.progressPanel.hidden = true;
    elements.progressTrack.classList.remove("complete");
  }
}

function setLoading(loading) {
  state.loading = loading;
  const recording = recordingInProgress();
  const inputsLocked = loading || recording;
  elements.inputPanel.setAttribute("aria-busy", String(loading));
  elements.analyzeButton.disabled = inputsLocked || !state.selectedFile;
  elements.selectedAnalyze.disabled = inputsLocked || !state.selectedFile;
  elements.recordAnalyze.disabled = inputsLocked || !state.recordFile;
  elements.demoButton.disabled = inputsLocked;
  elements.browseButton.disabled = inputsLocked;
  elements.audioFile.disabled = inputsLocked;
  elements.encoderOverlapSeconds.disabled = inputsLocked;
  elements.recordStart.disabled = inputsLocked || !recordingSupported();
  elements.recordStop.disabled = loading || state.mediaRecorder?.state !== "recording";
  elements.sourceTabs.forEach((tab) => { tab.disabled = inputsLocked; });
  $$(".sample-card").forEach((button) => { button.disabled = inputsLocked; });
  elements.buttonLoader.classList.toggle("visible", loading);
  elements.analyzeLabel.textContent = loading ? "Analyzing audio" : "Analyze selected audio";
}

function validateAnalysis(payload) {
  if (!payload || typeof payload !== "object") throw new Error("The backend returned an empty analysis.");
  if (!payload.transcription || !Array.isArray(payload.transcription.tokens)) throw new Error("The analysis response is missing transcription tokens.");
  const streams = availableStreams(payload);
  if (!streams.size) throw new Error("The analysis response contains no lens stream.");
  streams.forEach((kind) => {
    if (!Array.isArray(payload[kind]?.layers) || !Array.isArray(payload[kind]?.cells)) throw new Error(`The analysis response is missing the ${kind} lens grid.`);
  });
  return payload;
}

function abortError() {
  const error = new Error("Analysis polling was cancelled.");
  error.name = "AbortError";
  return error;
}

function delayWithSignal(milliseconds, signal) {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, milliseconds);
    const cancel = () => {
      window.clearTimeout(timeout);
      reject(abortError());
    };
    signal.addEventListener("abort", cancel, { once: true });
  });
}

function responseErrorMessage(payload, response) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (detail && typeof detail === "object") {
    const nested = detail.message || detail.error || detail.detail;
    if (typeof nested === "string" && nested.trim()) return nested.trim();
  }
  for (const value of [payload?.error, payload?.message]) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return `Request failed with status ${response.status}.`;
}

async function responseJson(response, endpointName) {
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`${endpointName} returned ${response.status} without JSON.`);
  }
  if (response.ok) return payload;
  if (response.status === 429) {
    const detail = payload?.detail && typeof payload.detail === "object" ? payload.detail : payload;
    const retryHeader = finiteNumberOrNull(response.headers.get("Retry-After"));
    const retrySeconds = finiteNumberOrNull(detail?.retry_after_seconds ?? payload?.retry_after_seconds) ?? retryHeader;
    const wait = formatApproximateWait(retrySeconds);
    throw new Error(`The analysis queue is full${wait ? `; it should have room in ${wait}` : ""}. Please try again shortly.`);
  }
  throw new Error(responseErrorMessage(payload, response));
}

function updateQueuedAnalysisProgress(job) {
  const jobState = String(job?.state || "queued").toLowerCase();
  if (jobState === "queued") {
    const position = Math.trunc(asFiniteNumber(job.queue_position));
    elements.progressLabel.textContent = position > 0 ? `Waiting in queue · #${position}` : "Waiting in queue…";
    elements.progressTrack.setAttribute("aria-label", position > 0
      ? `Audio analysis is waiting at queue position ${position}`
      : "Audio analysis is waiting in the queue");
    const wait = formatApproximateWait(job.estimated_wait_seconds);
    elements.progressDescription.textContent = wait
      ? `Expected to start in ${wait}. Queue position and timing are estimates based on recent analyses.`
      : "Your audio is safely queued. Queue position and timing update while you wait.";
    return;
  }
  if (jobState === "running") {
    elements.progressLabel.textContent = "Analyzing audio…";
    elements.progressTrack.setAttribute("aria-label", "Audio analysis is running");
    const elapsed = Math.max(0, asFiniteNumber(job.elapsed_seconds));
    const serverEstimate = finiteNumberOrNull(job.estimated_wait_seconds);
    const averageEstimate = finiteNumberOrNull(state.analysisQueue?.averageSeconds);
    const remaining = serverEstimate ?? (averageEstimate === null ? null : Math.max(0, averageEstimate - elapsed));
    const wait = formatApproximateWait(remaining);
    elements.progressDescription.textContent = wait
      ? `${defaultAnalysisProgressDescription()} Approximately ${wait} remaining.`
      : defaultAnalysisProgressDescription();
  }
}

async function analyzeWithLegacyEndpoint(formData, signal) {
  const response = await fetch(API.analyze, {
    method: "POST",
    body: formData,
    headers: { Accept: "application/json" },
    signal,
  });
  return responseJson(response, "The analysis endpoint");
}

async function analyzeWithQueue(formData, signal) {
  const submissionResponse = await fetch(API.analysisJobs, {
    method: "POST",
    body: formData,
    headers: { Accept: "application/json" },
    signal,
  });
  let job = await responseJson(submissionResponse, "The analysis queue");
  const jobId = String(job.id || "");
  const statusUrl = job.status_url || (jobId ? `${API.analysisJobs}/${encodeURIComponent(jobId)}` : null);
  let resultUrl = job.result_url || (jobId ? `${API.analysisJobs}/${encodeURIComponent(jobId)}/result` : null);
  if (!jobId || !statusUrl || !resultUrl) throw new Error("The analysis queue returned an incomplete job reference.");
  state.activeAnalysisJob = { id: jobId, statusUrl, resultUrl };

  while (true) {
    if (signal.aborted) throw abortError();
    const jobState = String(job.state || "").toLowerCase();
    if (jobState === "queued" || jobState === "running") {
      updateQueuedAnalysisProgress(job);
      await delayWithSignal(ANALYSIS_JOB_POLL_INTERVAL_MS, signal);
      const statusResponse = await fetch(statusUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal,
      });
      job = await responseJson(statusResponse, "The analysis job status endpoint");
      resultUrl = job.result_url || resultUrl;
      state.activeAnalysisJob.resultUrl = resultUrl;
      continue;
    }
    if (jobState === "succeeded" || jobState === "completed") {
      const resultResponse = await fetch(resultUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal,
      });
      const payload = await responseJson(resultResponse, "The analysis job result endpoint");
      return payload.result || payload.analysis || payload;
    }
    if (jobState === "failed") throw new Error(responseErrorMessage(job, { status: 500 }));
    if (jobState === "cancelled" || jobState === "canceled") throw new Error("The queued analysis was cancelled before it completed.");
    throw new Error(`The analysis queue returned an unknown job state: ${jobState || "missing"}.`);
  }
}

function cancelActiveAnalysisJob({ keepalive = false } = {}) {
  const active = state.activeAnalysisJob;
  if (!active?.statusUrl) return;
  state.activeAnalysisJob = null;
  fetch(active.statusUrl, {
    method: "DELETE",
    headers: { Accept: "application/json" },
    cache: "no-store",
    keepalive,
  }).catch(() => {});
}

async function analyzeSelectedFile() {
  if (!state.selectedFile || state.loading) return;
  clearError();
  setLoading(true);
  startProgress();
  cancelActiveAnalysisJob();
  state.requestController?.abort();
  state.requestController = new AbortController();
  const formData = new FormData();
  formData.append("audio", state.selectedFile, state.selectedFile.name);
  formData.append("time_bin_overlap_seconds", elements.encoderOverlapSeconds.value);
  try {
    const payload = state.analysisQueue?.enabled
      ? await analyzeWithQueue(formData, state.requestController.signal)
      : await analyzeWithLegacyEndpoint(formData, state.requestController.signal);
    renderResult(validateAnalysis(payload), "live");
    finishProgress(true);
    announce("Audio-model analysis is ready.");
  } catch (error) {
    if (error.name === "AbortError") {
      cancelActiveAnalysisJob();
    } else {
      // A lost polling/result request should not leave an abandoned upload
      // waiting for model time when the browser can still cancel it.
      cancelActiveAnalysisJob();
      finishProgress(false);
      showError(`${error.message} The synthetic interface demo remains available.`);
    }
  } finally {
    state.activeAnalysisJob = null;
    state.requestController = null;
    setLoading(false);
  }
}

function setAudioSource(source) {
  elements.audioPlayer.pause();
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = "";
  if (typeof source === "string") elements.audioPlayer.src = source;
  else if (source) {
    state.audioUrl = URL.createObjectURL(source);
    elements.audioPlayer.src = state.audioUrl;
  } else elements.audioPlayer.removeAttribute("src");
  elements.audioPlayer.load();
}

function renderResult(payload, mode) {
  const streams = availableStreams(payload);
  const timing = timingDetails(payload.transcription, mode);
  applyBackendBranding(payload.metadata || {});
  state.result = payload;
  state.waveform = Array.isArray(payload.audio?.waveform) ? payload.audio.waveform.map((value) => asFiniteNumber(value)) : [];
  state.duration = Math.max(0, asFiniteNumber(payload.audio?.duration_seconds));
  state.timing = timing;
  state.views = {};
  state.selectedCellButton = null;
  state.timelineSelection = { timeSeconds: null, encoderIndex: null, decoderIndex: null, tokenIndex: null, tokenMatch: "unavailable", origin: null };
  state.inspectorSelection = { kind: null, layerIndex: null };

  if (mode === "demo") {
    setAudioSource(createDemoAudio(state.duration || 5.6));
    elements.playbackSource.textContent = "Playback source: synthetic demo audio.";
  } else {
    const modelInput = payload.audio?.model_input_wav;
    const hasModelInput = typeof modelInput === "string" && modelInput.startsWith("data:audio/");
    setAudioSource(hasModelInput ? modelInput : state.selectedFile);
    elements.playbackSource.textContent = hasModelInput
      ? `Playback source: exact model input (${payload.audio?.model_input_format || "mono 16 kHz WAV"}).`
      : "Playback source: original source audio; processed model input was not included in this response.";
  }
  renderGeneratedAudio(payload, mode);

  elements.resultMode.textContent = mode === "demo" ? "Synthetic interface demo" : "Live analysis";
  elements.analysisModel.textContent = payload.metadata?.model_id || "Audio model";
  elements.durationTime.textContent = formatTime(state.duration);
  elements.currentTime.textContent = formatTime(0);
  elements.transcriptText.textContent = payload.transcription.text || payload.transcription.tokens.map((token) => token.text || "").join("");
  elements.timingNote.textContent = timing.label;
  elements.timingNote.classList.toggle("unavailable", !timing.showRanges);
  renderMetadata(payload, streams, timing);
  renderTranscriptTokens(payload.transcription.tokens, timing);

  elements.encoderSection.hidden = !streams.has("encoder");
  elements.decoderSection.hidden = !streams.has("decoder");
  elements.lensLayout.hidden = !streams.size;
  elements.inspector.hidden = !streams.size;
  updateEncoderPhoneSignatureMode({ rerender: false });
  updateEncoderMetricLabel();
  elements.decoderScoreLabel.textContent = state.decoderTokenLengthFilterEnabled
    ? `L0–L1 vocabulary limited to ≤ ${state.decoderMaxTokenLength} characters · L2 and output head unchanged`
    : "Blue strip intensity: within-layer percentile of raw readout logit · orange HEAD: actual probability";
  if (streams.has("encoder")) renderStream("encoder", payload.encoder);
  else elements.encoderWaveformSlices.replaceChildren();
  if (streams.has("decoder")) renderStream("decoder", payload.decoder);
  else elements.decoderNavigator.replaceChildren();
  updateEncoderTokenLengthFilter({ normalizeInput: true, rerender: false });
  updateDecoderTokenLengthFilter({ normalizeInput: true, rerender: false });
  if (!streams.size) clearInspector();
  else initializeTimelineSelection();

  drawWaveform();
  updatePlaybackUi();
  elements.results.hidden = false;
  elements.resetButton.hidden = false;
  window.requestAnimationFrame(() => {
    drawWaveform();
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function renderMetadata(payload, streams, timing) {
  const metadata = payload.metadata || {};
  const isSpeechToSpeech = isMlxLfmInfo(metadata);
  const generationDiagnostics = generationDiagnosticsFor(payload);
  const pooling = payload.encoder?.pooling;
  const poolingSummary = pooling
    ? `${Math.round(asFiniteNumber(pooling.effective_window_seconds) * 1000)} ms window · ${Math.round(asFiniteNumber(pooling.effective_overlap_seconds) * 1000)} ms overlap · ${Math.round(asFiniteNumber(pooling.effective_hop_seconds) * 1000)} ms hop${pooling.adaptive_for_max_bins ? " · widened for display limit" : ""}`
    : "Not provided";
  const values = [
    ["Analysis type", isSpeechToSpeech
      ? "Projected corpus-averaged language J-lens"
      : streams.has("encoder")
        ? "Corpus-averaged decoder J-lens + experimental encoder→decoder J-lens"
        : "Corpus-averaged decoder J-lens"],
    ["Model", metadata.model_id || "Not provided"],
    ["Backend", metadata.backend || "Not provided"],
    ["Output head", isSpeechToSpeech ? "Tied text head" : "Whisper LM head"],
    ["Lens streams", [...streams].join(" + ") || "None"],
    ...(streams.has("encoder") ? [["Encoder score", scoreKindDetails("encoder", payload.encoder).metadataLabel]] : []),
    ...(metadata.phone_signature?.available !== false && metadata.phone_signature ? [["Encoder phone view", `${String(metadata.phone_signature.method || metadata.phone_signature.estimator || "Fitted phone prototypes").replaceAll("_", " ")} · cosine similarity · not probability`]] : []),
    ...(streams.has("encoder") ? [["Encoder pooling", poolingSummary]] : []),
    ...(streams.has("decoder") ? [["Decoder score", scoreKindDetails("decoder", payload.decoder).metadataLabel]] : []),
    ["Estimator", metadata.estimator || "Not provided"],
    ["Lens examples", metadata.lens_examples ?? "Not provided"],
    ...(metadata.projection ? [["Projection", `${metadata.projection.method || "projected Jacobian"} · rank ${metadata.projection.rank}/${metadata.projection.target_dim}`]] : []),
    ["Display vocabulary", displayVocabularySummary(metadata)],
    ["Token timing", timing.metadataLabel],
    ["Audio duration", `${asFiniteNumber(payload.audio?.duration_seconds).toFixed(2)} s`],
    ["Output tokens", payload.transcription?.tokens?.length || 0],
    ...(payload.audio?.model_output_wav ? [["Generated speech", `${asFiniteNumber(payload.audio?.model_output_duration_seconds).toFixed(2)} s · ${payload.audio?.model_output_format || "audio"}`]] : []),
    ...(isSpeechToSpeech && generationDiagnostics ? [["Generation stop", `${generationDiagnostics.status} · ${generationDiagnostics.detail}`]] : []),
  ];
  elements.metadataList.replaceChildren();
  values.forEach(([label, value]) => {
    const wrapper = createElement("div");
    if (["Analysis type", "Encoder score", "Encoder phone view", "Encoder pooling", "Decoder score", "Estimator", "Projection", "Display vocabulary", "Token timing", "Generated speech", "Generation stop"].includes(label)) wrapper.classList.add("metadata-wide");
    const description = createElement("dd", "", String(value));
    description.title = String(value);
    wrapper.append(createElement("dt", "", label), description);
    elements.metadataList.append(wrapper);
  });
  const warnings = Array.isArray(metadata.warnings) ? metadata.warnings.filter(Boolean) : [];
  elements.warningList.replaceChildren(...warnings.map((warning) => createElement("li", "", String(warning))));
  elements.warnings.hidden = !warnings.length;
  elements.warnings.closest(".metadata-card").classList.toggle("no-warnings", !warnings.length);
}

function renderTranscriptTokens(tokens, timing) {
  elements.tokenCards.replaceChildren();
  const isSpeechToSpeech = isMlxLfmInfo(state.result?.metadata || {});
  tokens.forEach((token, index) => {
    const probability = clamp(token.probability);
    const range = timing.showRanges ? timedRange(token) : null;
    const card = createElement("button", `token-card${range ? "" : " token-card-static"}`);
    card.type = "button";
    card.style.setProperty("--prob-fill", `${(probability * 100).toFixed(1)}%`);
    card.style.setProperty("--prob-border-alpha", (0.1 + probability * 0.5).toFixed(3));
    card.style.setProperty("--prob-bg-alpha", (0.035 + probability * 0.13).toFixed(3));
    if (range) { card.dataset.start = range.start; card.dataset.end = range.end; }
    card.dataset.tokenIndex = index;
    const top = createElement("span", "token-card-top");
    top.append(createElement("span", "token-card-token", visibleToken(token.text)), createElement("span", "token-card-probability", formatProbability(probability)));
    const meta = createElement("span", "token-card-meta");
    const headRank = timelineRankDetails(token, { isHead: true });
    meta.append(createElement("span", range ? "" : "timing-unavailable", range ? `${range.start.toFixed(2)}–${range.end.toFixed(2)}s` : "timing unavailable"), createElement("span", "", `ID ${token.id ?? "?"} · ${headRank.rank === null ? "rank —" : `#${headRank.rank}`}`));
    card.append(top, meta);
    const alternatives = Array.isArray(token.top_tokens) ? token.top_tokens.slice(0, 4) : [];
    const alternate = alternatives.find((candidate) => String(candidate.id) !== String(token.id) || candidate.text !== token.text);
    if (alternate) {
      const detail = createElement("span", "token-card-alt");
      detail.append("top alternative · ", createElement("strong", "", visibleToken(alternate.text)), ` ${formatProbability(alternate.probability)}`);
      card.append(detail);
    }
    const synchronization = isSpeechToSpeech
      ? " Activate to synchronize the tied text head and language J-lens layers. Response-token timing is unavailable, so input-audio playback is unchanged."
      : ` Activate to synchronize the output head and lens views.${range ? " Audio and encoder timing will also be aligned." : " Token timing is unavailable, so the encoder slice will remain unchanged."}`;
    card.setAttribute("aria-label", `Output token ${visibleToken(token.text)}, token ID ${token.id ?? "unknown"}, output-head probability ${formatProbabilityPrecise(probability)}, ${headRank.label}.${synchronization}`);
    card.addEventListener("click", () => {
      selectTimelineToken(index, { origin: "output", seek: Boolean(range), play: false, announceChange: true });
      const decoderIndex = decoderPositionForTokenIndex(index);
      if (decoderIndex !== null) inspectOutputHeadPosition(decoderIndex, { pin: true, announceChange: false });
    });
    elements.tokenCards.append(card);
  });
}

function decodedTokenLength(token) {
  const text = String(token?.text ?? "").trim();
  return text.includes("<|") ? 0 : [...text].length;
}

function phoneSignaturesForCell(cell) {
  return (Array.isArray(cell?.phone_signatures) ? cell.phone_signatures : [])
    .map((candidate, index) => ({
      phone: String(candidate?.phone ?? "").trim(),
      similarity: finiteNumberOrNull(candidate?.similarity),
      rank: Math.max(1, Math.trunc(asFiniteNumber(candidate?.rank, index + 1))),
      rankDenominator: finiteNumberOrNull(candidate?.rank_denominator),
    }))
    .filter((candidate) => candidate.phone && candidate.similarity !== null)
    .sort((left, right) => left.rank - right.rank || right.similarity - left.similarity);
}

function phoneSignatureMetadata() {
  const metadata = state.result?.metadata?.phone_signature;
  return metadata && typeof metadata === "object" && !Array.isArray(metadata) ? metadata : null;
}

function phoneSignatureAvailable() {
  const metadata = phoneSignatureMetadata();
  const cells = state.result?.encoder?.cells;
  if (!metadata || metadata.available === false || !Array.isArray(cells) || !cells.length) return false;
  const flattened = cells.flatMap((layerCells) => Array.isArray(layerCells) ? layerCells : []);
  return flattened.length > 0 && flattened.every((cell) => phoneSignaturesForCell(cell).length > 0);
}

function phoneSignatureCandidateCount(cell) {
  const metadata = phoneSignatureMetadata();
  const candidateReported = finiteNumberOrNull(phoneSignaturesForCell(cell)[0]?.rankDenominator);
  const reported = candidateReported ?? finiteNumberOrNull(
    metadata?.phone_inventory_size ?? metadata?.phone_count ?? metadata?.prototype_count ?? metadata?.label_count,
  );
  return reported === null ? phoneSignaturesForCell(cell).length : Math.max(1, Math.trunc(reported));
}

function encoderUsesPhoneSignatures(kind) {
  return kind === "encoder" && state.encoderPhoneSignatureEnabled && state.encoderPhoneSignatureAvailable;
}

function decoderLengthFilterLayers() {
  const layers = Array.isArray(state.result?.decoder?.layers) ? state.result.decoder.layers : [];
  const cells = Array.isArray(state.result?.decoder?.cells) ? state.result.decoder.cells : [];
  const bucketLayers = layers.filter((layer, layerIndex) => {
    const layerCells = cells[layerIndex];
    return Array.isArray(layerCells)
      && layerCells.length > 0
      && layerCells.every((cell) => cell?.top_tokens_by_length && typeof cell.top_tokens_by_length === "object");
  }).map(Number);
  const configured = state.result?.metadata?.decoder_token_length_filter?.eligible_source_layers;
  if (Array.isArray(configured)) {
    return configured.map(Number).filter((layer) => bucketLayers.includes(layer));
  }
  return bucketLayers.filter((layer) => layer === 0 || layer === 1);
}

function tokenLengthFilterSettings(kind, layer) {
  if (kind === "encoder" && state.encoderTokenLengthFilterEnabled && !encoderUsesPhoneSignatures(kind)) {
    return { active: true, limit: state.encoderMaxTokenLength };
  }
  if (
    kind === "decoder"
    && state.decoderTokenLengthFilterEnabled
    && decoderLengthFilterLayers().includes(Number(layer))
  ) {
    return { active: true, limit: state.decoderMaxTokenLength };
  }
  return { active: false, limit: null };
}

function tokensForCell(kind, cell, layer = null) {
  const tokens = Array.isArray(cell?.top_tokens) ? cell.top_tokens : [];
  const filter = tokenLengthFilterSettings(kind, layer);
  if (!filter.active) return tokens;
  const buckets = cell?.top_tokens_by_length;
  if (buckets && typeof buckets === "object") {
    const eligibleBuckets = Object.entries(buckets)
      .filter(([length]) => Number(length) <= filter.limit);
    const bucketDenominator = eligibleBuckets.reduce((sum, [, candidates]) => {
      const reported = finiteNumberOrNull(Array.isArray(candidates) ? candidates[0]?.rank_denominator : null);
      return sum + (reported === null ? 0 : Math.max(0, Math.trunc(reported)));
    }, 0);
    const maximumCounts = state.result?.metadata?.display_vocabulary?.maximum_decoded_character_length_counts;
    const exactCounts = state.result?.metadata?.display_vocabulary?.exact_decoded_character_length_counts;
    const metadataDenominator = finiteNumberOrNull(maximumCounts?.[String(filter.limit)]);
    const exactCountSum = exactCounts && typeof exactCounts === "object"
      ? Object.entries(exactCounts).filter(([length]) => Number(length) <= filter.limit).reduce((sum, [, count]) => sum + Math.max(0, Math.trunc(asFiniteNumber(count))), 0)
      : 0;
    const denominator = metadataDenominator ?? (exactCountSum || bucketDenominator);
    const deduplicated = new Map();
    eligibleBuckets
      .flatMap(([, candidates]) => Array.isArray(candidates) ? candidates : [])
      .forEach((token) => {
        const key = String(token.id ?? `${token.text}:${token.score}`);
        const previous = deduplicated.get(key);
        if (!previous || asFiniteNumber(token.score, -Infinity) > asFiniteNumber(previous.score, -Infinity)) deduplicated.set(key, token);
      });
    const eligibleCandidates = [...deduplicated.values()];
    return eligibleCandidates
      .sort((left, right) => asFiniteNumber(right.score) - asFiniteNumber(left.score))
      .slice(0, Math.max(1, tokens.length || 5))
      .map((token) => ({
        ...token,
        rank: 1 + eligibleCandidates.filter((candidate) => asFiniteNumber(candidate.score, -Infinity) > asFiniteNumber(token.score, -Infinity)).length,
        rank_denominator: denominator || null,
        rank_space: "maximum_decoded_character_length_vocabulary",
        rank_tie_policy: token.rank_tie_policy || "1_plus_count_strictly_greater",
        vocabulary_filter: { kind: "maximum_decoded_character_length", maximum_characters: filter.limit },
      }));
  }
  if (kind === "decoder") return tokens;
  if (kind === "encoder" && filter.limit === 2 && Array.isArray(cell?.short_top_tokens)) return cell.short_top_tokens;
  return tokens.filter((token) => {
    const length = decodedTokenLength(token);
    return length > 0 && length <= filter.limit;
  });
}

function scoreForCell(kind, cell, layer = null) {
  if (encoderUsesPhoneSignatures(kind)) return finiteNumberOrNull(phoneSignaturesForCell(cell)[0]?.similarity);
  const tokens = tokensForCell(kind, cell, layer);
  if (tokenLengthFilterSettings(kind, layer).active) return finiteNumberOrNull(tokens[0]?.score);
  return finiteNumberOrNull(cell?.selected_score ?? tokens[0]?.score);
}

function buildLayerStrengthFunctions(cells, kind, layers) {
  return cells.map((layerCells, layerIndex) => {
    const layer = layers[layerIndex];
    const scores = (Array.isArray(layerCells) ? layerCells : [])
      .map((cell) => scoreForCell(kind, cell, layer))
      .filter((score) => score !== null)
      .sort((a, b) => a - b);
    return (score) => {
      if (scores.length <= 1) return 0.6;
      let count = 0;
      scores.forEach((candidate) => { if (candidate <= score) count += 1; });
      return clamp((count - 1) / (scores.length - 1));
    };
  });
}

function timelineRangeForColumn(kind, column, columnIndex) {
  const direct = timedRange(column);
  if (direct) return { ...direct, approximate: kind === "decoder" };
  if (kind === "decoder") {
    const tokenIndex = tokenIndexForDecoderPosition(columnIndex);
    const tokenRange = state.timing.showRanges ? timedRange(state.result?.transcription?.tokens?.[tokenIndex]) : null;
    if (tokenRange) return { ...tokenRange, approximate: true };
  }
  const count = Math.max(1, state.views[kind]?.columns?.length || (kind === "decoder" ? state.result?.decoder?.positions?.length : state.result?.encoder?.time_bins?.length) || 1);
  const duration = Math.max(analysisDuration(), 0.001);
  return {
    start: columnIndex / count * duration,
    end: (columnIndex + 1) / count * duration,
    approximate: false,
    logicalOnly: true,
  };
}

function timelineCellGeometry(kind, column, columnIndex) {
  const duration = Math.max(analysisDuration(), 0.001);
  const range = timelineRangeForColumn(kind, column, columnIndex);
  const start = clamp(range.start / duration);
  const end = clamp(range.end / duration);
  const widthPercent = Math.max(0, end - start) * 100;
  return {
    range,
    left: `${(start * 100).toFixed(5)}%`,
    width: `${(widthPercent > 0 ? widthPercent : 0.001).toFixed(5)}%`,
  };
}

function timelineRankDetails(token, { kind, layer = null, isHead = false } = {}) {
  if (isHead) {
    const rank = finiteNumberOrNull(token?.full_vocabulary_rank ?? (token?.rank_space === "full_model_vocabulary" ? token?.rank : null));
    const denominator = finiteNumberOrNull(token?.full_vocabulary_denominator ?? token?.rank_denominator);
    if (rank !== null) {
      return {
        rank: Math.trunc(rank),
        denominator: denominator === null ? null : Math.trunc(denominator),
        scope: "full-model-vocabulary",
        label: `full-model-vocabulary rank #${Math.trunc(rank)}${denominator === null ? "" : ` of ${Math.trunc(denominator)}`}`,
        tiePolicy: token?.rank_tie_policy || "not reported",
      };
    }
    const fallback = finiteNumberOrNull(token?.rank);
    return {
      rank: fallback === null ? null : Math.trunc(fallback),
      denominator: null,
      scope: "reported output-head top-k",
      label: fallback === null ? "full-model-vocabulary rank unavailable" : `reported output-head top-k rank #${Math.trunc(fallback)}`,
      tiePolicy: token?.rank_tie_policy || "not reported",
    };
  }

  const filter = tokenLengthFilterSettings(kind, layer);
  if (filter.active) {
    const rank = finiteNumberOrNull(token?.rank);
    const denominator = finiteNumberOrNull(token?.rank_denominator);
    return {
      rank: rank === null ? null : Math.trunc(rank),
      denominator: denominator === null ? null : Math.trunc(denominator),
      scope: `≤${filter.limit}-character lexical`,
      label: rank === null
        ? `≤${filter.limit}-character lexical rank unavailable`
        : `≤${filter.limit}-character lexical rank #${Math.trunc(rank)}${denominator === null ? "" : ` of ${Math.trunc(denominator)}`}`,
      tiePolicy: token?.rank_tie_policy || "1_plus_count_strictly_greater",
    };
  }

  const displayRank = finiteNumberOrNull(token?.display_vocabulary_rank ?? (token?.rank_space === "lexical_display_vocabulary" ? token?.rank : null));
  const displayDenominator = finiteNumberOrNull(token?.display_vocabulary_denominator ?? token?.rank_denominator);
  if (displayRank !== null) {
    return {
      rank: Math.trunc(displayRank),
      denominator: displayDenominator === null ? null : Math.trunc(displayDenominator),
      scope: "lexical-display-vocabulary",
      label: `lexical-display-vocabulary rank #${Math.trunc(displayRank)}${displayDenominator === null ? "" : ` of ${Math.trunc(displayDenominator)}`}`,
      tiePolicy: token?.rank_tie_policy || "not reported",
    };
  }
  const fallback = finiteNumberOrNull(token?.rank);
  return {
    rank: fallback === null ? null : Math.trunc(fallback),
    denominator: null,
    scope: "reported returned-list",
    label: fallback === null ? "lexical-display-vocabulary rank unavailable" : `reported returned-list rank #${Math.trunc(fallback)}`,
    tiePolicy: token?.rank_tie_policy || "not reported",
  };
}

function timelineRankLabel(token, options = {}) {
  return timelineRankDetails(token, options).label;
}

function timelineFilterState(kind, layer = null) {
  const filter = tokenLengthFilterSettings(kind, layer);
  if (!filter.active) return kind === "encoder" || kind === "decoder" ? "Character-length filter off" : "Unfiltered full model vocabulary";
  return `Character-length filter on · maximum ${filter.limit} decoded characters`;
}

function timelineCandidateScoreLabel(candidate, { isHead = false, metric = null } = {}) {
  if (isHead) {
    const probability = finiteNumberOrNull(candidate?.probability);
    const logProbability = finiteNumberOrNull(candidate?.log_probability);
    return `${probability === null ? "probability unavailable" : `${formatProbabilityPrecise(probability)} probability`}${logProbability === null ? "" : ` · ln p ${formatScore(logProbability)}`}`;
  }
  return `${metric?.columnLabel || "raw score"} ${formatScore(candidate?.score)}`;
}

function compactRankValue(rank) {
  if (rank.rank === null) return "rank unavailable";
  return `#${rank.rank}${rank.denominator === null ? "" : ` / ${rank.denominator}`}`;
}

function timelineTooltipCandidate(candidate, options) {
  const rank = timelineRankDetails(candidate, options);
  return {
    token: visibleToken(candidate?.text),
    tokenId: candidate?.id ?? "unknown",
    rank: compactRankValue(rank),
    rankLabel: rank.label,
    score: timelineCandidateScoreLabel(candidate, options),
  };
}

function outputHeadCandidates(token) {
  return (Array.isArray(token?.top_tokens) ? token.top_tokens : [])
    .map((candidate, index) => ({ ...candidate, rank: candidate.rank ?? index + 1 }))
    .sort((left, right) => asFiniteNumber(right.probability, -Infinity) - asFiniteNumber(left.probability, -Infinity));
}

function realizedRankForCell(kind, cell, layer) {
  const realized = cell?.realized_token;
  if (!realized || typeof realized !== "object") return null;
  const filter = tokenLengthFilterSettings(kind, layer);
  if (filter.active) {
    const filteredRank = finiteNumberOrNull(cell?.realized_rank_by_max_length?.[String(filter.limit)]);
    if (filteredRank === null) return null;
    const denominator = finiteNumberOrNull(
      state.result?.metadata?.display_vocabulary?.maximum_decoded_character_length_counts?.[String(filter.limit)],
    );
    return {
      rank: Math.trunc(filteredRank),
      denominator: denominator === null ? null : Math.trunc(denominator),
    };
  }
  const rank = timelineRankDetails(realized, { kind, layer });
  return rank.rank === null ? null : { rank: rank.rank, denominator: rank.denominator };
}

function appendTimelineCellLabel(button, label, { realizedRank = null } = {}) {
  button.classList.add("candidate-label-cell");
  button.append(createElement("span", "lens-timeline-cell-label", label));
  if (realizedRank?.rank !== null && realizedRank?.rank !== undefined) {
    const badge = createElement("span", "lens-timeline-cell-rank", `R#${realizedRank.rank}`);
    const denominator = realizedRank.denominator;
    badge.title = `Realized-token rank #${realizedRank.rank}${denominator === null || denominator === undefined ? "" : ` of ${denominator}`}`;
    button.append(badge);
  }
}

function phoneTooltipCandidate(candidate, denominator) {
  return {
    token: candidate.phone,
    tokenId: "frozen prototype",
    tokenIdPrefix: "",
    rank: `#${candidate.rank}${denominator ? ` / ${denominator}` : ""}`,
    rankLabel: `phone-prototype rank #${candidate.rank}${denominator ? ` of ${denominator}` : ""}`,
    score: `cosine ${formatScore(candidate.similarity)}`,
  };
}

function timelineTooltipDetails(kind, layerIndex, columnIndex) {
  const isHead = kind === "head";
  const view = isHead ? state.views.decoder : state.views[kind];
  if (!view?.columns?.[columnIndex]) return null;
  const column = view.columns[columnIndex];

  if (isHead) {
    const tokenIndex = tokenIndexForDecoderPosition(columnIndex);
    const token = state.result?.transcription?.tokens?.[tokenIndex];
    if (!token) return null;
    const rank = timelineRankDetails(token, { isHead: true });
    const reported = outputHeadCandidates(token);
    const candidates = (reported.length ? reported : [token])
      .slice(0, 5)
      .map((candidate) => timelineTooltipCandidate(candidate, { isHead: true }));
    const probability = finiteNumberOrNull(token.probability);
    const logProbability = finiteNumberOrNull(token.log_probability);
    return {
      kind: "head",
      eyebrow: "LM HEAD · REALIZED TOKEN",
      token: visibleToken(token.text),
      tokenId: token.id ?? "unknown",
      coordinate: lensContext("decoder", column, columnIndex),
      metrics: [
        { label: "Probability", value: probability === null ? "unavailable" : formatProbabilityPrecise(probability) },
        { label: "Log probability", value: logProbability === null ? "unavailable" : `ln p ${formatScore(logProbability)}` },
        { label: `${rank.scope} rank`, value: compactRankValue(rank) },
        { label: "Token filter", value: "Off · full model vocabulary" },
      ],
      candidates,
    };
  }

  if (!view.layers.length) return null;
  const safeLayerIndex = Math.max(0, Math.min(view.layers.length - 1, Number(layerIndex) || 0));
  const layer = view.layers[safeLayerIndex];
  const cell = view.cells[safeLayerIndex]?.[columnIndex];
  if (encoderUsesPhoneSignatures(kind)) {
    const signatures = phoneSignaturesForCell(cell);
    if (!signatures.length) return null;
    const topSignature = signatures[0];
    const denominator = phoneSignatureCandidateCount(cell);
    return {
      kind,
      eyebrow: `ENCODER PHONE SIGNATURE · L${layer}`,
      token: topSignature.phone,
      tokenId: "frozen prototype",
      tokenIdPrefix: "",
      coordinate: lensContext(kind, column, columnIndex),
      metrics: [
        { label: "Prototype similarity", value: `cosine ${formatScore(topSignature.similarity)}` },
        { label: "Prototype rank", value: `#${topSignature.rank}${denominator ? ` / ${denominator}` : ""}` },
        { label: "Character-length filter", value: "Paused in phone view" },
      ],
      candidatesLabel: "Nearest phone prototypes",
      candidates: signatures.slice(0, 5).map((candidate) => phoneTooltipCandidate(candidate, denominator)),
    };
  }
  const tokens = tokensForCell(kind, cell, layer);
  if (!cell || !tokens.length) return null;
  const topToken = tokens[0];
  const rank = timelineRankDetails(topToken, { kind, layer });
  const filterState = timelineFilterState(kind, layer);
  const lengthFilter = tokenLengthFilterSettings(kind, layer);
  const filterSummary = lengthFilter.active
    ? `On · ≤${lengthFilter.limit} decoded characters`
    : filterState.replace("Character-length filter ", "");
  return {
    kind,
    eyebrow: `${streamDisplayName(kind).toUpperCase()} · L${layer}`,
    token: visibleToken(topToken.text),
    tokenId: topToken.id ?? "unknown",
    coordinate: lensContext(kind, column, columnIndex),
    metrics: [
      { label: view.metric.columnLabel, value: formatScore(topToken.score ?? scoreForCell(kind, cell, layer)) },
      { label: `${rank.scope} rank`, value: compactRankValue(rank) },
      { label: "Token filter", value: filterSummary },
    ],
    candidates: tokens
      .slice(0, 5)
      .map((candidate) => timelineTooltipCandidate(candidate, { kind, layer, metric: view.metric })),
  };
}

function renderTimelineTooltip(details) {
  if (!details) return false;
  elements.timelineTooltip.dataset.kind = details.kind;
  elements.tooltipEyebrow.textContent = details.eyebrow;
  elements.tooltipToken.textContent = details.token;
  elements.tooltipTokenId.textContent = `${details.tokenIdPrefix ?? "ID "}${details.tokenId}`;
  elements.tooltipCoordinate.textContent = details.coordinate;
  elements.tooltipMetrics.replaceChildren();
  details.metrics.forEach(({ label, value }) => {
    const row = createElement("div", "lens-tooltip-metric");
    row.append(createElement("dt", "", label), createElement("dd", "", value));
    elements.tooltipMetrics.append(row);
  });
  elements.tooltipCandidates.replaceChildren();
  details.candidates.forEach((candidate) => {
    const row = createElement("li", "lens-tooltip-candidate");
    const candidateCopy = createElement("span", "lens-tooltip-candidate-copy");
    candidateCopy.append(
      createElement("strong", "lens-tooltip-candidate-token", candidate.token),
      createElement("small", "lens-tooltip-candidate-id", candidate.tokenIdPrefix === "" ? String(candidate.tokenId) : `ID ${candidate.tokenId}`),
    );
    row.append(
      createElement("span", "lens-tooltip-rank", candidate.rank),
      candidateCopy,
      createElement("span", "lens-tooltip-candidate-score", candidate.score),
    );
    row.setAttribute("aria-label", `${candidate.rankLabel}; ${candidate.token}; ${candidate.tokenIdPrefix === "" ? candidate.tokenId : `token ID ${candidate.tokenId}`}; ${candidate.score}.`);
    elements.tooltipCandidates.append(row);
  });
  elements.tooltipCandidatesLabel.textContent = details.candidatesLabel || "Top candidates";
  elements.tooltipCandidatesLabel.hidden = !details.candidates.length;
  elements.tooltipCandidates.hidden = !details.candidates.length;
  return true;
}

function positionTimelineTooltip(trigger) {
  if (!trigger?.isConnected || elements.timelineTooltip.hidden) return;
  const chart = trigger.closest(".lens-timeline-chart");
  if (!chart) return;
  const margin = 6;
  const gap = 10;
  const viewportHeight = document.documentElement.clientHeight;
  const chartRect = chart.getBoundingClientRect();
  const triggerRect = trigger.getBoundingClientRect();
  const tooltipRect = elements.timelineTooltip.getBoundingClientRect();
  const centeredLeft = triggerRect.left - chartRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
  const maximumLeft = Math.max(margin, chartRect.width - tooltipRect.width - margin);
  const left = Math.max(margin, Math.min(maximumLeft, centeredLeft));
  const aboveViewportTop = triggerRect.top - tooltipRect.height - gap;
  const belowViewportTop = triggerRect.bottom + gap;
  let placement = "top";
  let preferredViewportTop = aboveViewportTop;
  if (aboveViewportTop < margin && belowViewportTop + tooltipRect.height <= viewportHeight - margin) {
    placement = "bottom";
    preferredViewportTop = belowViewportTop;
  } else if (aboveViewportTop < margin) {
    const spaceAbove = triggerRect.top - margin;
    const spaceBelow = viewportHeight - margin - triggerRect.bottom;
    placement = spaceBelow > spaceAbove ? "bottom" : "top";
    preferredViewportTop = placement === "bottom" ? belowViewportTop : aboveViewportTop;
  }
  const maximumViewportTop = Math.max(margin, viewportHeight - tooltipRect.height - margin);
  const viewportTop = Math.max(margin, Math.min(maximumViewportTop, preferredViewportTop));
  if (Math.abs(viewportTop - preferredViewportTop) > 0.5) placement = "center";
  const top = viewportTop - chartRect.top;
  elements.timelineTooltip.dataset.placement = placement;
  elements.timelineTooltip.style.left = `${Math.round(left)}px`;
  elements.timelineTooltip.style.top = `${Math.round(top)}px`;
}

function showTimelineTooltip(trigger, details, { mode = "pointer" } = {}) {
  if (!trigger || !renderTimelineTooltip(details)) return;
  const chart = trigger.closest(".lens-timeline-chart");
  if (!chart) return;
  if (state.timelineTooltip.trigger && state.timelineTooltip.trigger !== trigger) {
    state.timelineTooltip.trigger.removeAttribute("aria-describedby");
  }
  if (elements.timelineTooltip.parentElement !== chart) chart.append(elements.timelineTooltip);
  state.timelineTooltip = { trigger, mode, chart };
  trigger.setAttribute("aria-describedby", "lens-timeline-tooltip");
  elements.timelineTooltip.hidden = false;
  elements.timelineTooltip.dataset.visible = "false";
  positionTimelineTooltip(trigger);
  elements.timelineTooltip.dataset.visible = "true";
}

function hideTimelineTooltip() {
  state.timelineTooltip.trigger?.removeAttribute("aria-describedby");
  state.timelineTooltip = { trigger: null, mode: null, chart: null };
  elements.timelineTooltip.dataset.visible = "false";
  elements.timelineTooltip.hidden = true;
}

function renderTimelineRow(view, layerIndex) {
  const layer = view.layers[layerIndex];
  const row = createElement("div", "lens-timeline-row");
  row.dataset.timelineKind = view.kind;
  row.dataset.layerIndex = String(layerIndex);
  const label = createElement("strong", "lens-timeline-row-label", `L${layer}`);
  const track = createElement("div", "lens-timeline-track");
  track.setAttribute("role", "group");
  track.setAttribute("aria-label", `${streamDisplayName(view.kind)} layer ${layer} across ${view.kind === "encoder" ? "overlapping audio windows" : "approximate output-token spans"}`);
  view.columns.forEach((column, columnIndex) => {
    const cell = view.cells[layerIndex]?.[columnIndex];
    const phoneMode = encoderUsesPhoneSignatures(view.kind);
    const phoneSignatures = phoneMode ? phoneSignaturesForCell(cell) : [];
    const tokens = tokensForCell(view.kind, cell, layer);
    if (!cell || (phoneMode ? !phoneSignatures.length : !tokens.length)) return;
    const topToken = tokens[0];
    const topSignature = phoneSignatures[0];
    const score = scoreForCell(view.kind, cell, layer);
    const strength = view.layerStrength[layerIndex](score ?? 0);
    const geometry = timelineCellGeometry(view.kind, column, columnIndex);
    const button = createElement("button", `lens-timeline-cell ${view.kind}-timeline-cell`);
    button.type = "button";
    button.dataset.timelineKind = view.kind;
    button.dataset.layerIndex = String(layerIndex);
    button.dataset.columnIndex = String(columnIndex);
    button.style.left = geometry.left;
    button.style.width = geometry.width;
    button.style.setProperty("--timeline-intensity", strength.toFixed(6));
    button.setAttribute("aria-pressed", "false");
    button.tabIndex = -1;
    let description;
    if (phoneMode) {
      const denominator = phoneSignatureCandidateCount(cell);
      button.classList.add("phone-signature-cell");
      appendTimelineCellLabel(button, topSignature.phone);
      description = `Encoder layer ${layer}, ${lensContext(view.kind, column, columnIndex)}. Nearest frozen phone prototype ${topSignature.phone}, cosine similarity ${formatScore(topSignature.similarity)}, rank #${topSignature.rank}${denominator ? ` of ${denominator}` : ""}. Exploratory fitted readout, not probability or confidence.`;
    } else {
      const rankLabel = timelineRankLabel(topToken, { kind: view.kind, layer });
      const filterState = timelineFilterState(view.kind, layer);
      const realizedRank = realizedRankForCell(view.kind, cell, layer);
      appendTimelineCellLabel(button, visibleToken(topToken.text), {
        realizedRank,
      });
      const realizedRankDescription = realizedRank
        ? ` Realized-token rank #${realizedRank.rank}${realizedRank.denominator === null ? "" : ` of ${realizedRank.denominator}`}.`
        : "";
      description = `${streamDisplayName(view.kind)} layer ${layer}, ${lensContext(view.kind, column, columnIndex)}. Top candidate ${visibleToken(topToken.text)}, token ID ${topToken.id ?? "unknown"}, ${timelineCandidateScoreLabel(topToken, { metric: view.metric })}, ${rankLabel}.${realizedRankDescription} ${filterState}.`;
    }
    button.setAttribute("aria-label", description);
    button.addEventListener("pointerenter", (event) => {
      if (event.pointerType !== "touch") showTimelineTooltip(button, timelineTooltipDetails(view.kind, layerIndex, columnIndex), { mode: "pointer" });
    });
    button.addEventListener("pointerleave", (event) => {
      if (event.pointerType !== "touch" && state.timelineTooltip.trigger === button) hideTimelineTooltip();
    });
    button.addEventListener("pointerdown", (event) => { button.dataset.lastPointerType = event.pointerType || "mouse"; });
    button.addEventListener("focus", () => {
      if (!state.restoringTimelineFocus) {
        selectTimelineCoordinate(view.kind, layerIndex, columnIndex, { focusCell: false, announceChange: false });
        showTimelineTooltip(button, timelineTooltipDetails(view.kind, layerIndex, columnIndex), { mode: "focus" });
      }
    });
    button.addEventListener("blur", () => {
      if (state.timelineTooltip.trigger === button && state.timelineTooltip.mode === "focus") hideTimelineTooltip();
    });
    button.addEventListener("click", (event) => {
      selectTimelineCoordinate(view.kind, layerIndex, columnIndex, { focusCell: false, announceChange: true });
      const mode = event.detail === 0 ? "focus" : button.dataset.lastPointerType === "touch" ? "touch" : "pointer";
      showTimelineTooltip(button, timelineTooltipDetails(view.kind, layerIndex, columnIndex), { mode });
    });
    button.addEventListener("keydown", (event) => navigateTimelineCell(event, view.kind, layerIndex, columnIndex));
    view.cellButtons.push(button);
    track.append(button);
  });
  row.append(label, track);
  return row;
}

function renderOutputHeadTimelineRow(view) {
  const row = createElement("div", "lens-timeline-row head-timeline-row");
  const label = createElement("strong", "lens-timeline-row-label", "HEAD");
  const track = createElement("div", "lens-timeline-track");
  track.setAttribute("role", "group");
  track.setAttribute("aria-label", "Whisper LM-head actual teacher-forced token probabilities across approximate output-token spans");
  view.columns.forEach((column, columnIndex) => {
    const tokenIndex = tokenIndexForDecoderPosition(columnIndex);
    const token = state.result?.transcription?.tokens?.[tokenIndex];
    if (!token) return;
    const geometry = timelineCellGeometry("decoder", column, columnIndex);
    const probability = clamp(token.probability);
    const button = createElement("button", "lens-timeline-cell head-timeline-cell");
    button.type = "button";
    button.dataset.timelineKind = "head";
    button.dataset.columnIndex = String(columnIndex);
    button.dataset.tokenIndex = String(tokenIndex);
    button.style.left = geometry.left;
    button.style.width = geometry.width;
    button.style.setProperty("--timeline-intensity", probability.toFixed(6));
    button.setAttribute("aria-pressed", "false");
    button.tabIndex = -1;
    const headCandidates = outputHeadCandidates(token);
    if (headCandidates.length) {
      appendTimelineCellLabel(button, visibleToken(headCandidates[0].text), {
        realizedRank: timelineRankDetails(token, { isHead: true }),
      });
    }
    const rankLabel = timelineRankLabel(token, { isHead: true });
    const headCandidateDescription = headCandidates.length
      ? ` Top candidate ${visibleToken(headCandidates[0].text)}, token ID ${headCandidates[0].id ?? "unknown"}.`
      : "";
    const description = `Whisper LM head, ${lensContext("decoder", column, columnIndex)}.${headCandidateDescription} Realized token ${visibleToken(token.text)}, token ID ${token.id ?? "unknown"}, ${timelineCandidateScoreLabel(token, { isHead: true })}, ${rankLabel}. Unfiltered full model vocabulary.`;
    button.setAttribute("aria-label", description);
    button.addEventListener("pointerenter", (event) => {
      if (event.pointerType !== "touch") showTimelineTooltip(button, timelineTooltipDetails("head", null, columnIndex), { mode: "pointer" });
    });
    button.addEventListener("pointerleave", (event) => {
      if (event.pointerType !== "touch" && state.timelineTooltip.trigger === button) hideTimelineTooltip();
    });
    button.addEventListener("pointerdown", (event) => { button.dataset.lastPointerType = event.pointerType || "mouse"; });
    button.addEventListener("focus", () => {
      if (!state.restoringTimelineFocus) {
        selectTimelineCoordinate("head", null, columnIndex, { focusCell: false, announceChange: false });
        showTimelineTooltip(button, timelineTooltipDetails("head", null, columnIndex), { mode: "focus" });
      }
    });
    button.addEventListener("blur", () => {
      if (state.timelineTooltip.trigger === button && state.timelineTooltip.mode === "focus") hideTimelineTooltip();
    });
    button.addEventListener("click", (event) => {
      selectTimelineCoordinate("head", null, columnIndex, { focusCell: false, announceChange: true });
      const mode = event.detail === 0 ? "focus" : button.dataset.lastPointerType === "touch" ? "touch" : "pointer";
      showTimelineTooltip(button, timelineTooltipDetails("head", null, columnIndex), { mode });
    });
    button.addEventListener("keydown", (event) => navigateTimelineCell(event, "head", view.layers.length, columnIndex));
    view.headButtons.push(button);
    track.append(button);
  });
  row.append(label, track);
  return row;
}

function renderCompactTimeline(kind, view) {
  const container = kind === "encoder" ? elements.encoderLayers : elements.decoderLayers;
  hideTimelineTooltip();
  container.replaceChildren();
  container.classList.toggle("logical-token-axis", kind === "decoder" && !state.timing.showRanges);
  const minimumCellWidth = kind === "encoder" ? 28 : 92;
  container.style.setProperty("--timeline-track-min-width", `${Math.max(0, view.columns.length * minimumCellWidth)}px`);
  const scroller = createElement("div", "lens-timeline-scroll");
  scroller.addEventListener("scroll", () => {
    if (state.timelineTooltip.chart === container) hideTimelineTooltip();
  }, { passive: true });
  const axis = createElement("div", "lens-timeline-axis");
  axis.append(createElement("span", "lens-timeline-row-label", kind === "decoder" && !state.timing.showRanges ? "Order" : "Time"));
  const ticks = createElement("div", "lens-timeline-axis-track");
  if (kind === "decoder" && !state.timing.showRanges) {
    ticks.append(createElement("span", "", "first token"), createElement("span", "", "token order · timing unavailable"), createElement("span", "", "last token"));
  } else {
    const duration = analysisDuration();
    ticks.append(createElement("span", "", "0.00 s"), createElement("span", "", `${(duration / 2).toFixed(2)} s`), createElement("span", "", `${duration.toFixed(2)} s`));
  }
  axis.append(ticks);
  scroller.append(axis);
  view.layers.forEach((_, layerIndex) => scroller.append(renderTimelineRow(view, layerIndex)));
  if (kind === "decoder") scroller.append(renderOutputHeadTimelineRow(view));
  container.append(scroller);
  container.onpointerleave = (event) => {
    if (event.pointerType !== "touch" && state.timelineTooltip.mode === "pointer") hideTimelineTooltip();
  };
}

function renderStream(kind, data, { preserveState = false } = {}) {
  const layers = Array.isArray(data.layers) ? data.layers : [];
  const columns = kind === "encoder" ? (Array.isArray(data.time_bins) ? data.time_bins : []) : (Array.isArray(data.positions) ? data.positions : []);
  const cells = Array.isArray(data.cells) ? data.cells : [];
  const comparison = kind === "encoder" ? elements.encoderLayers : elements.decoderLayers;
  const previous = preserveState ? state.views[kind] : null;
  if (kind === "encoder") elements.encoderWaveformSlices.replaceChildren();
  else elements.decoderNavigator.replaceChildren();
  comparison.replaceChildren();
  if (!layers.length || !columns.length) {
    comparison.append(createElement("p", "empty-stream", `The ${kind} stream is available but returned no displayable positions.`));
    return;
  }
  const view = {
    kind,
    data,
    layers,
    columns,
    cells,
    metric: scoreKindDetails(kind, data),
    layerStrength: buildLayerStrengthFunctions(cells, kind, layers),
    pinnedIndex: Math.max(0, Math.min(columns.length - 1, previous?.pinnedIndex ?? 0)),
    focusIndex: Math.max(0, Math.min(columns.length - 1, previous?.focusIndex ?? previous?.pinnedIndex ?? 0)),
    pinnedLayerIndex: Math.max(0, Math.min(layers.length - 1, previous?.pinnedLayerIndex ?? layers.length - 1)),
    navButtons: [],
    cellButtons: [],
    headButtons: [],
    timelineUnmatched: false,
  };
  state.views[kind] = view;
  renderCompactTimeline(kind, view);
  view.navButtons = columns.map((_, columnIndex) => (
    view.cellButtons.find((button) => Number(button.dataset.columnIndex) === columnIndex && Number(button.dataset.layerIndex) === view.pinnedLayerIndex)
    || view.headButtons.find((button) => Number(button.dataset.columnIndex) === columnIndex)
  )).filter(Boolean);
  showStreamPosition(kind, view.pinnedIndex, { pin: true, announceChange: false, updateInspector: false });
}

function refreshCompactStream(kind) {
  const view = state.views[kind];
  if (!view) return;
  renderStream(kind, view.data, { preserveState: true });
  const refreshed = state.views[kind];
  showStreamPosition(kind, refreshed.pinnedIndex, { pin: true, announceChange: false, updateInspector: false });
  if (state.inspectorSelection.kind === kind) inspectTimelineCell(kind, refreshed.pinnedLayerIndex, refreshed.pinnedIndex, { pin: true });
  else if (state.inspectorSelection.kind === "head" && kind === "decoder") inspectOutputHeadPosition(refreshed.pinnedIndex, { pin: true });
}

function pointerTimeInTimeline(event, element) {
  const rect = element.getBoundingClientRect();
  if (!rect.width) return 0;
  return clamp((event.clientX - rect.left) / rect.width) * analysisDuration();
}

function navigateEncoderWaveform(event, view) {
  let target = view.pinnedIndex;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") target -= 1;
  else if (event.key === "ArrowRight" || event.key === "ArrowUp") target += 1;
  else if (event.key === "Home") target = 0;
  else if (event.key === "End") target = view.columns.length - 1;
  else return;
  event.preventDefault();
  target = Math.max(0, Math.min(view.columns.length - 1, target));
  const range = timedRange(view.columns[target]);
  if (range) selectTimelineAtTime((range.start + range.end) / 2, { origin: "encoder", seek: true, announceChange: true });
}

function navigatePositionButtons(event, view, index) {
  let target = index;
  if (event.key === "ArrowLeft" || event.key === "ArrowUp") target -= 1;
  else if (event.key === "ArrowRight" || event.key === "ArrowDown") target += 1;
  else if (event.key === "Home") target = 0;
  else if (event.key === "End") target = view.navButtons.length - 1;
  else return;
  event.preventDefault();
  target = Math.max(0, Math.min(view.navButtons.length - 1, target));
  view.navButtons[target].focus();
}

function updateCompactTimelineSelection(view) {
  if (!view) return;
  const pinnedColumn = view.timelineUnmatched ? null : view.pinnedIndex;
  const fallbackColumn = view.pinnedIndex;
  const selectedKind = state.inspectorSelection.kind;
  const selectedLayerIndex = state.inspectorSelection.layerIndex;
  let tabbableAssigned = false;
  view.cellButtons.forEach((button) => {
    const columnIndex = Number(button.dataset.columnIndex);
    const layerIndex = Number(button.dataset.layerIndex);
    const selectedColumn = columnIndex === pinnedColumn;
    const selectedCoordinate = selectedColumn && selectedKind === view.kind && layerIndex === selectedLayerIndex;
    button.classList.toggle("selected-column", selectedColumn);
    button.classList.toggle("selected-coordinate", selectedCoordinate);
    button.setAttribute("aria-pressed", String(selectedCoordinate));
    button.tabIndex = selectedCoordinate ? 0 : -1;
    if (selectedCoordinate) tabbableAssigned = true;
  });
  view.headButtons.forEach((button) => {
    const selectedColumn = Number(button.dataset.columnIndex) === pinnedColumn;
    const selectedCoordinate = selectedColumn && selectedKind === "head";
    button.classList.toggle("selected-column", selectedColumn);
    button.classList.toggle("selected-coordinate", selectedCoordinate);
    button.setAttribute("aria-pressed", String(selectedCoordinate));
    button.tabIndex = selectedCoordinate ? 0 : -1;
    if (selectedCoordinate) tabbableAssigned = true;
  });
  if (!tabbableAssigned) {
    const fallback = view.cellButtons.find((button) => (
      Number(button.dataset.columnIndex) === fallbackColumn
      && Number(button.dataset.layerIndex) === view.pinnedLayerIndex
    )) || view.headButtons.find((button) => Number(button.dataset.columnIndex) === fallbackColumn);
    if (fallback) fallback.tabIndex = 0;
  }
}

function revealTimelineColumn(view, columnIndex, { behavior = "auto" } = {}) {
  if (!view || !Number.isInteger(columnIndex)) return;
  const target = view.cellButtons.find((button) => (
    Number(button.dataset.columnIndex) === columnIndex
    && Number(button.dataset.layerIndex) === view.pinnedLayerIndex
  )) || view.headButtons.find((button) => Number(button.dataset.columnIndex) === columnIndex);
  if (!target) return;
  const reveal = () => {
    const scroller = target.closest(".lens-timeline-scroll");
    if (!scroller?.isConnected) return;
    const track = target.parentElement;
    const targetStart = (track?.offsetLeft || 0) + target.offsetLeft;
    const targetEnd = targetStart + target.offsetWidth;
    const visibleStart = scroller.scrollLeft + 56;
    const visibleEnd = scroller.scrollLeft + scroller.clientWidth - 16;
    let destination = scroller.scrollLeft;
    if (targetStart < visibleStart) destination = targetStart - 64;
    else if (targetEnd > visibleEnd) destination = targetEnd - scroller.clientWidth + 24;
    destination = Math.max(0, Math.min(scroller.scrollWidth - scroller.clientWidth, destination));
    if (Math.abs(destination - scroller.scrollLeft) > 0.5) {
      if (behavior === "auto") scroller.scrollLeft = destination;
      else scroller.scrollTo({ left: destination, behavior });
    }
  };
  window.requestAnimationFrame(() => {
    reveal();
    // Synchronized encoder/decoder updates can change the page geometry in the
    // same frame. A post-event pass also wins over native focus auto-scrolling.
    window.setTimeout(reveal, 32);
  });
}

function selectTimelineCoordinate(kind, layerIndex, columnIndex, { focusCell = false, announceChange = false } = {}) {
  const view = kind === "head" ? state.views.decoder : state.views[kind];
  if (!view || !view.columns[columnIndex]) return;
  if (kind !== "head") view.pinnedLayerIndex = Math.max(0, Math.min(view.layers.length - 1, Number(layerIndex) || 0));
  if (kind === "encoder") {
    const range = timelineRangeForColumn("encoder", view.columns[columnIndex], columnIndex);
    selectTimelineAtTime((range.start + range.end) / 2, { origin: "encoder-cell", seek: true, announceChange: false });
    inspectTimelineCell("encoder", view.pinnedLayerIndex, columnIndex, { pin: true });
  } else {
    const tokenIndex = tokenIndexForDecoderPosition(columnIndex);
    selectTimelineToken(tokenIndex, { origin: kind === "head" ? "head-cell" : "decoder-cell", seek: true, announceChange: false });
    if (kind === "head") inspectOutputHeadPosition(columnIndex, { pin: true });
    else inspectTimelineCell("decoder", view.pinnedLayerIndex, columnIndex, { pin: true });
  }
  updateCompactTimelineSelection(state.views.encoder);
  updateCompactTimelineSelection(state.views.decoder);
  if (focusCell) {
    const candidates = kind === "head" ? view.headButtons : view.cellButtons;
    window.requestAnimationFrame(() => candidates.find((button) => (
      Number(button.dataset.columnIndex) === columnIndex
      && (kind === "head" || Number(button.dataset.layerIndex) === view.pinnedLayerIndex)
    ))?.focus());
  }
  if (announceChange) announce(`${kind === "head" ? "LM head" : `${streamDisplayName(kind)} layer ${view.layers[view.pinnedLayerIndex]}`} pinned at ${lensContext(kind === "head" ? "decoder" : kind, view.columns[columnIndex], columnIndex)}.`);
}

function navigateTimelineCell(event, kind, layerIndex, columnIndex) {
  const view = kind === "head" ? state.views.decoder : state.views[kind];
  if (!view) return;
  const rowCount = view.layers.length + (view.kind === "decoder" ? 1 : 0);
  let targetLayer = kind === "head" ? view.layers.length : Number(layerIndex);
  let targetColumn = Number(columnIndex);
  if ((event.ctrlKey || event.metaKey) && event.key === "Home") {
    targetLayer = 0;
    targetColumn = 0;
  } else if ((event.ctrlKey || event.metaKey) && event.key === "End") {
    targetLayer = rowCount - 1;
    targetColumn = view.columns.length - 1;
  } else if (event.key === "ArrowLeft") targetColumn -= 1;
  else if (event.key === "ArrowRight") targetColumn += 1;
  else if (event.key === "ArrowUp") targetLayer -= 1;
  else if (event.key === "ArrowDown") targetLayer += 1;
  else if (event.key === "Home") targetColumn = 0;
  else if (event.key === "End") targetColumn = view.columns.length - 1;
  else return;
  event.preventDefault();
  targetLayer = Math.max(0, Math.min(rowCount - 1, targetLayer));
  targetColumn = Math.max(0, Math.min(view.columns.length - 1, targetColumn));
  const targetKind = view.kind === "decoder" && targetLayer === view.layers.length ? "head" : view.kind;
  selectTimelineCoordinate(targetKind, targetLayer, targetColumn, { focusCell: true, announceChange: false });
}

function intensityWord(strength) {
  if (strength >= 0.82) return "high within layer";
  if (strength >= 0.55) return "upper-middle within layer";
  if (strength >= 0.28) return "lower-middle within layer";
  return "low within layer";
}

function showStreamPosition(kind, index, { pin = false, announceChange = false, updateInspector = true } = {}) {
  const view = state.views[kind];
  if (!view) return;
  const safeIndex = Math.max(0, Math.min(view.columns.length - 1, index));
  view.focusIndex = safeIndex;
  if (pin) {
    view.pinnedIndex = safeIndex;
    view.timelineUnmatched = false;
  }
  const column = view.columns[safeIndex];
  const label = kind === "encoder" ? elements.encoderFocusLabel : elements.decoderFocusLabel;
  label.textContent = lensContext(kind, column, safeIndex);
  if (kind === "encoder" && pin) updateEncoderSliderAria(safeIndex);
  updateCompactTimelineSelection(view);
  revealTimelineColumn(view, safeIndex);
  if (updateInspector) inspectTimelineCell(kind, view.pinnedLayerIndex, safeIndex, { pin: false });
  if (announceChange) announce(`${kind === "encoder" ? "Audio slice" : "Token position"} selected. Layer comparison updated.`);
}

function renderLayerComparison(view) {
  updateCompactTimelineSelection(view);
}

function buildOutputHeadCard(positionIndex) {
  const token = state.result?.transcription?.tokens?.[positionIndex];
  if (!token) return createElement("article", "output-head-card layer-card", "Output head unavailable");
  const nearest = state.timelineSelection.tokenIndex === positionIndex && state.timelineSelection.tokenMatch === "nearest";
  const probability = clamp(token.probability);
  const card = createElement("article", "layer-card output-head-card");
  card.classList.toggle("nearest", nearest);
  card.style.setProperty("--prob-border-alpha", (0.16 + probability * 0.45).toFixed(3));
  card.style.setProperty("--prob-bg-alpha", (0.02 + probability * 0.11).toFixed(3));
  const metadata = state.result?.metadata || {};
  const outputLabel = metadata.stream_labels?.output || "LM / OUTPUT HEAD";
  const targetLayer = finiteNumberOrNull(state.result?.decoder?.target_layer);
  const isMlxLfm = metadata.backend === "mlx" || metadata.model_family === "lfm2_audio";
  const stageDescription = isMlxLfm
    ? `LFM post-block L${targetLayer === null ? "final" : targetLayer} → final RMS norm → tied text head · teacher-forced on the realized interleaved path`
    : "Whisper post-block decoder L3 → final norm → LM head · before suppression and timestamp rules";
  const heading = createElement("span", "layer-card-head");
  heading.append(createElement("span", "output-index", String(outputLabel).toUpperCase()), createElement("span", "output-stage", "0–100% probability"));
  card.append(heading, createElement("strong", "layer-token", visibleToken(token.text)), createElement("span", "output-probability", `${formatProbability(probability)} output-head probability`));
  const candidates = createElement("span", "mini-candidates");
  (Array.isArray(token.top_tokens) ? token.top_tokens : [])
    .filter((candidate) => String(candidate.id) !== String(token.id) || candidate.text !== token.text)
    .slice(0, 3)
    .forEach((candidate, index) => {
    const row = createElement("span", "mini-candidate");
    row.append(createElement("span", "", `TOP ALTERNATIVE ${index + 1} · ${visibleToken(candidate.text)}`), createElement("span", "", formatProbability(candidate.probability)));
    candidates.append(row);
  });
  const mappingNote = nearest ? "Nearest approximate token interval to the selected audio point · " : "";
  card.append(candidates, createElement("span", "output-head-note", `${mappingNote}${stageDescription} · direct model output, not J-lens`));
  card.setAttribute("aria-label", `${nearest ? "Nearest approximate output-token interval. " : ""}${stageDescription}. Emitted token ${visibleToken(token.text)}, output-head probability ${formatProbability(probability)}. This is direct model output, not J-lens.`);
  return card;
}

function lensContext(kind, column, columnIndex) {
  if (kind === "encoder") return `slice ${columnIndex + 1} · ${asFiniteNumber(column.start_seconds).toFixed(2)}–${asFiniteNumber(column.end_seconds).toFixed(2)}s`;
  const range = state.timing.showRanges ? timedRange(column) : null;
  return `token ${columnIndex + 1} · ${visibleToken(column.text)}${range ? ` · ≈${range.start.toFixed(2)}–${range.end.toFixed(2)}s` : ""}`;
}

function inspectTimelineCell(kind, layerIndex, columnIndex, { pin = false, announceChange = false } = {}) {
  const view = state.views[kind];
  if (!view || !view.layers.length) return;
  const safeLayer = Math.max(0, Math.min(view.layers.length - 1, layerIndex));
  const safeColumn = Math.max(0, Math.min(view.columns.length - 1, columnIndex));
  if (pin) {
    view.pinnedLayerIndex = safeLayer;
    state.inspectorSelection = { kind, layerIndex: safeLayer };
  }
  const layer = view.layers[safeLayer];
  const cell = view.cells[safeLayer]?.[safeColumn];
  const phoneMode = encoderUsesPhoneSignatures(kind);
  const phoneSignatures = phoneMode ? phoneSignaturesForCell(cell) : [];
  const tokens = tokensForCell(kind, cell, layer);
  if (!cell || (phoneMode ? !phoneSignatures.length : !tokens.length)) return;
  const column = view.columns[safeColumn];
  const focusedButton = view.cellButtons.find((button) => (
    Number(button.dataset.layerIndex) === safeLayer && Number(button.dataset.columnIndex) === safeColumn
  ));
  if (pin) state.selectedCellButton = focusedButton || null;
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  if (phoneMode) {
    const topSignature = phoneSignatures[0];
    const denominator = phoneSignatureCandidateCount(cell);
    elements.inspectorKind.textContent = "Encoder · fitted phone-prototype readout";
    elements.inspectorCellTitle.textContent = `Encoder L${layer} · phone ${topSignature.phone}`;
    elements.inspectorContext.textContent = `${lensContext(kind, column, safeColumn)} · frozen prototype rank #${topSignature.rank}${denominator ? ` of ${denominator}` : ""}`;
    elements.scoreKicker.textContent = "Cosine similarity to frozen phone prototype";
    elements.selectedScore.textContent = formatScore(topSignature.similarity);
    elements.rankKeyLabel.textContent = "Rank among fitted phone prototypes";
    elements.scoreDescription.textContent = `The nearest fitted phone label is ${topSignature.phone}. Similarity is computed against frozen training prototypes from the matching encoder J-lens. It is not a model probability, phoneme confidence, framewise vote, boundary, or causal effect; this pooled 100 ms window may still contain more than one phone.`;
    elements.topkMetricLabel.textContent = "Phone-prototype rank · ARPAbet label · cosine similarity. The character-length token filter is paused in this view. Blue intensity is a within-layer display percentile, not probability.";
    renderPhoneSignatures(phoneSignatures, denominator);
    if (pin) {
      updateCompactTimelineSelection(view);
      updateCompactTimelineSelection(state.views.decoder);
    }
    if (announceChange) announce(`Encoder layer ${layer} phone signature ${topSignature.phone} pinned in the focused readout.`);
    return;
  }
  const topToken = tokens[0];
  const score = finiteNumberOrNull(topToken.score ?? scoreForCell(kind, cell, layer));
  const streamName = streamDisplayName(kind);
  elements.inspectorKind.textContent = `${streamName} · ${view.metric.shortLabel}`;
  elements.inspectorCellTitle.textContent = `${streamName} L${layer} · ${visibleToken(topToken.text)}`;
  elements.inspectorContext.textContent = `${lensContext(kind, column, safeColumn)} · token ID ${topToken.id ?? "unknown"}`;
  elements.scoreKicker.textContent = view.metric.metadataLabel;
  elements.selectedScore.textContent = formatScore(score);
  elements.rankKeyLabel.textContent = "Exact rank in the labeled vocabulary scope";
  const lengthFilter = tokenLengthFilterSettings(kind, layer);
  const filterState = timelineFilterState(kind, layer);
  const rank = timelineRankDetails(topToken, { kind, layer });
  const lengthNote = lengthFilter.active
    ? ` Ranking is restricted to lexical tokens with at most ${lengthFilter.limit} decoded characters before top-k selection.${kind === "encoder" ? " This is an Audio J-Lens exploration aid, not a phoneme classifier." : " This changes the L0/L1 readout only, not Whisper generation or output probabilities."}`
    : "";
  elements.scoreDescription.textContent = `${view.metric.description} Top candidate ${visibleToken(topToken.text)} · ID ${topToken.id ?? "unknown"} · ${rank.label}. ${filterState}. Tie policy: ${String(rank.tiePolicy).replaceAll("_", " ")}.${lengthNote}`;
  elements.topkMetricLabel.textContent = `${rank.scope} rank · token string and ID · ${view.metric.columnLabel}. ${filterState}. Blue cell intensity is a within-layer display percentile, not probability.`;
  renderTopTokens(tokens, { kind, layer, metric: view.metric });
  if (pin) {
    updateCompactTimelineSelection(view);
    updateCompactTimelineSelection(kind === "encoder" ? state.views.decoder : state.views.encoder);
  }
  if (announceChange) announce(`${kind} layer ${layer} pinned in the focused readout.`);
}

function inspectOutputHeadPosition(columnIndex, { pin = false, announceChange = false } = {}) {
  const view = state.views.decoder;
  if (!view || !view.columns[columnIndex]) return;
  const tokenIndex = tokenIndexForDecoderPosition(columnIndex);
  const token = state.result?.transcription?.tokens?.[tokenIndex];
  if (!token) return;
  if (pin) state.inspectorSelection = { kind: "head", layerIndex: null };
  const reported = Array.isArray(token.top_tokens) ? token.top_tokens.map((candidate, index) => ({ ...candidate, rank: candidate.rank ?? index + 1 })) : [];
  if (!reported.some((candidate) => String(candidate.id) === String(token.id))) reported.push({ ...token, realized: true });
  const candidates = reported.sort((left, right) => asFiniteNumber(right.probability, -Infinity) - asFiniteNumber(left.probability, -Infinity));
  const rank = timelineRankDetails(token, { isHead: true });
  const column = view.columns[columnIndex];
  const focusedButton = view.headButtons.find((button) => Number(button.dataset.columnIndex) === columnIndex);
  if (pin) state.selectedCellButton = focusedButton || null;
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  elements.inspectorKind.textContent = "Whisper LM head · actual teacher-forced probability";
  elements.inspectorCellTitle.textContent = `HEAD · ${visibleToken(token.text)}`;
  elements.inspectorContext.textContent = `${lensContext("decoder", column, columnIndex)} · token ID ${token.id ?? "unknown"}`;
  elements.scoreKicker.textContent = "Raw full-softmax output probability";
  elements.selectedScore.textContent = formatProbabilityPrecise(token.probability);
  elements.rankKeyLabel.textContent = "Exact rank in the labeled vocabulary scope";
  const logProbability = finiteNumberOrNull(token.log_probability);
  elements.scoreDescription.textContent = `Actual Whisper LM-head probability${logProbability === null ? "" : ` · ln p ${formatScore(logProbability)}`} · ${rank.label}. Full model vocabulary, explicitly unfiltered. Tie policy: ${String(rank.tiePolicy).replaceAll("_", " ")}. This is direct model output, not a J-lens readout.`;
  elements.topkMetricLabel.textContent = "Full-model-vocabulary rank · token string and ID · raw probability and log probability. Character-length filters never apply to HEAD.";
  renderTopTokens(candidates, { isHead: true });
  if (pin) {
    updateCompactTimelineSelection(view);
    updateCompactTimelineSelection(state.views.encoder);
  }
  if (announceChange) announce(`LM head pinned at output token ${visibleToken(token.text)}.`);
}

function inspectLayerForView(kind, layerIndex, { pin = false, announceChange = false } = {}) {
  const view = state.views[kind];
  if (!view) return;
  inspectTimelineCell(kind, layerIndex, view.focusIndex, { pin, announceChange });
}

function restorePinnedInspector() {
  const { kind, layerIndex } = state.inspectorSelection;
  if (kind === "head") inspectOutputHeadPosition(state.views.decoder?.pinnedIndex ?? 0, { pin: false, announceChange: false });
  else if (kind && Number.isInteger(layerIndex)) inspectLayerForView(kind, layerIndex, { pin: false, announceChange: false });
  else clearInspector();
}

function renderTopTokens(tokens, { kind = null, layer = null, metric = null, isHead = false } = {}) {
  elements.topkList.replaceChildren();
  tokens.forEach((token, index) => {
    const item = createElement("li", "topk-item");
    const rank = timelineRankDetails({ ...token, rank: token.rank ?? index + 1 }, { kind, layer, isHead });
    const strength = tokens.length <= 1 ? 1 : 1 - index / (tokens.length - 1);
    item.style.setProperty("--rank-border-alpha", (0.08 + strength * 0.25).toFixed(3));
    item.style.setProperty("--rank-bg-alpha", (0.02 + strength * 0.09).toFixed(3));
    item.style.setProperty("--rank-fill", `${(strength * 100).toFixed(1)}%`);
    const rankElement = createElement("span", "topk-rank", rank.rank === null ? "—" : `#${rank.rank}`);
    if (rank.denominator !== null) rankElement.append(createElement("small", "", `/ ${rank.denominator}`));
    const tokenElement = createElement("span", "topk-token");
    tokenElement.append(createElement("strong", "", visibleToken(token.text)), createElement("small", "", `ID ${token.id ?? "unknown"}`));
    const scoreElement = createElement("span", "topk-score", timelineCandidateScoreLabel(token, { isHead, metric }));
    item.append(rankElement, tokenElement, scoreElement);
    item.setAttribute("aria-label", `${rank.label}; token ${visibleToken(token.text)}; token ID ${token.id ?? "unknown"}; ${timelineCandidateScoreLabel(token, { isHead, metric })}; tie policy ${String(rank.tiePolicy).replaceAll("_", " ")}.`);
    item.title = `${rank.label}; token ID ${token.id ?? "unknown"}; ${timelineCandidateScoreLabel(token, { isHead, metric })}`;
    elements.topkList.append(item);
  });
}

function renderPhoneSignatures(signatures, denominator) {
  elements.topkList.replaceChildren();
  signatures.forEach((candidate, index) => {
    const item = createElement("li", "topk-item");
    const strength = signatures.length <= 1 ? 1 : 1 - index / (signatures.length - 1);
    item.style.setProperty("--rank-border-alpha", (0.08 + strength * 0.25).toFixed(3));
    item.style.setProperty("--rank-bg-alpha", (0.02 + strength * 0.09).toFixed(3));
    item.style.setProperty("--rank-fill", `${(strength * 100).toFixed(1)}%`);
    const rankElement = createElement("span", "topk-rank", `#${candidate.rank}`);
    if (denominator) rankElement.append(createElement("small", "", `/ ${denominator}`));
    const phoneElement = createElement("span", "topk-token");
    phoneElement.append(createElement("strong", "", candidate.phone), createElement("small", "", "frozen phone prototype"));
    const scoreElement = createElement("span", "topk-score", `cosine ${formatScore(candidate.similarity)}`);
    item.append(rankElement, phoneElement, scoreElement);
    item.setAttribute("aria-label", `Phone-prototype rank #${candidate.rank}${denominator ? ` of ${denominator}` : ""}; phone ${candidate.phone}; cosine similarity ${formatScore(candidate.similarity)}; exploratory fitted readout, not probability or confidence.`);
    item.title = `Phone prototype ${candidate.phone} · cosine ${formatScore(candidate.similarity)}`;
    elements.topkList.append(item);
  });
}

function clearInspector({ restoreFocus = false } = {}) {
  const previousButton = state.selectedCellButton;
  const previousKind = previousButton?.dataset.timelineKind === "head" ? "decoder" : previousButton?.dataset.timelineKind || null;
  state.inspectorSelection = { kind: null, layerIndex: null };
  previousButton?.classList.remove("selected");
  previousButton?.removeAttribute("aria-pressed");
  state.selectedCellButton = null;
  updateCompactTimelineSelection(state.views.encoder);
  updateCompactTimelineSelection(state.views.decoder);
  elements.inspectorEmpty.hidden = false;
  elements.inspectorContent.hidden = true;
  elements.rankKeyLabel.textContent = "Exact rank in the labeled vocabulary scope";
  elements.topkList.replaceChildren();
  if (restoreFocus && previousKind) {
    state.restoringTimelineFocus = true;
    previousButton?.focus({ preventScroll: true });
    window.requestAnimationFrame(() => { state.restoringTimelineFocus = false; });
  }
}

function drawWaveform() {
  const lensRgb = getComputedStyle(document.documentElement)
    .getPropertyValue("--lens-rgb")
    .trim();
  drawWaveformCanvas(elements.waveformCanvas, `rgba(${lensRgb},.58)`);
  drawWaveformCanvas(elements.encoderWaveformCanvas, `rgba(${lensRgb},.38)`);
}

function drawWaveformCanvas(canvas, fillStyle) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!state.waveform.length) {
    context.strokeStyle = "rgba(111,123,134,.42)";
    context.setLineDash([4, 5]);
    context.beginPath();
    context.moveTo(0, rect.height / 2);
    context.lineTo(rect.width, rect.height / 2);
    context.stroke();
    return;
  }
  const maximum = Math.max(0.000001, ...state.waveform.map((value) => Math.abs(value)));
  const center = rect.height / 2;
  const amplitude = rect.height * 0.38;
  const barWidth = Math.max(1, rect.width / state.waveform.length * 0.62);
  context.fillStyle = fillStyle;
  state.waveform.forEach((value, index) => {
    const x = (index + 0.5) / state.waveform.length * rect.width;
    const height = Math.max(1.5, Math.abs(value) / maximum * amplitude);
    context.fillRect(x - barWidth / 2, center - height, barWidth, height * 2);
  });
  context.fillStyle = "rgba(24,33,42,.16)";
  context.fillRect(0, center, rect.width, 1);
}

function effectiveDuration() {
  return Number.isFinite(elements.audioPlayer.duration) && elements.audioPlayer.duration > 0 ? elements.audioPlayer.duration : state.duration;
}

function seekAudio(seconds, play = false) {
  if (!elements.audioPlayer.src) return;
  const duration = Math.max(effectiveDuration(), 0.0001);
  elements.audioPlayer.currentTime = clamp(seconds / duration) * duration;
  updatePlaybackUi();
  if (play) elements.audioPlayer.play().catch(() => {});
}

function updatePlaybackUi() {
  const duration = Math.max(0, effectiveDuration());
  const current = Math.max(0, asFiniteNumber(elements.audioPlayer.currentTime));
  const progress = duration ? clamp(current / duration) : 0;
  elements.waveformPlayhead.style.left = `${progress * 100}%`;
  elements.currentTime.textContent = formatTime(current);
  elements.durationTime.textContent = formatTime(duration || state.duration);
  elements.tokenCards.querySelectorAll(".token-card[data-start]").forEach((card) => {
    const start = asFiniteNumber(card.dataset.start);
    const end = asFiniteNumber(card.dataset.end, start);
    card.classList.toggle("playing", current >= start && current < end && !elements.audioPlayer.paused);
  });
}

function resetAnalysis() {
  hideTimelineTooltip();
  cancelActiveAnalysisJob();
  state.requestController?.abort();
  window.clearInterval(state.progressTimer);
  window.clearTimeout(state.progressHideTimer);
  discardRecording();
  state.requestController = null;
  state.result = null;
  state.waveform = [];
  state.duration = 0;
  state.views = {};
  state.timelineSelection = { timeSeconds: null, encoderIndex: null, decoderIndex: null, tokenIndex: null, tokenMatch: "unavailable", origin: null };
  state.inspectorSelection = { kind: null, layerIndex: null };
  state.selectedFile = null;
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = "";
  elements.audioPlayer.pause();
  elements.audioPlayer.removeAttribute("src");
  elements.audioPlayer.load();
  elements.generatedAudioPlayer.pause();
  elements.generatedAudioPlayer.removeAttribute("src");
  elements.generatedAudioPlayer.load();
  elements.generatedAudio.hidden = true;
  elements.audioFile.value = "";
  elements.fileLabel.textContent = "Drop an audio file here";
  elements.fileDetail.textContent = "WAV, MP3, M4A, FLAC, OGG, or WebM";
  elements.analyzeButton.disabled = true;
  elements.selectedSource.hidden = true;
  elements.results.hidden = true;
  elements.progressPanel.hidden = true;
  elements.waveformSelection.hidden = true;
  elements.waveformTokenSelection.hidden = true;
  elements.tokenCards.replaceChildren();
  elements.encoderWaveformSlices.replaceChildren();
  elements.encoderLayers.replaceChildren();
  elements.decoderNavigator.replaceChildren();
  elements.decoderLayers.replaceChildren();
  elements.encoderFocusLabel.textContent = "Select a slice";
  elements.decoderFocusLabel.textContent = "Select a token";
  clearError();
  announce("Analysis cleared.");
  document.querySelector(".source-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function loadDemo() {
  if (recordingInProgress()) return;
  clearError();
  cancelActiveAnalysisJob();
  state.requestController?.abort();
  finishProgress(false);
  setLoading(false);
  const payload = state.serverMode === "speech"
    ? buildSpeechDemoData()
    : buildDemoData();
  renderResult(payload, "demo");
  announce("Synthetic interface demo loaded. Its analysis values are not model output.");
}

function recordingSupported() {
  return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
}

function preferredRecordingType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return types.find((type) => MediaRecorder.isTypeSupported?.(type)) || "";
}

async function startRecording() {
  setRecordError();
  if (state.loading || recordingInProgress()) return;
  if (!recordingSupported()) {
    setRecordError("In-browser recording is not supported here. Use a prepared sample or upload an audio file.");
    return;
  }
  discardRecording({ keepMessage: true });
  state.recordingPending = true;
  setLoading(state.loading);
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    const mimeType = preferredRecordingType();
    state.mediaRecorder = mimeType ? new MediaRecorder(state.mediaStream, { mimeType }) : new MediaRecorder(state.mediaStream);
    state.recordChunks = [];
    state.mediaRecorder.addEventListener("dataavailable", (event) => { if (event.data?.size) state.recordChunks.push(event.data); });
    state.mediaRecorder.addEventListener("stop", finalizeRecording, { once: true });
    state.mediaRecorder.start(250);
    state.recordStartedAt = Date.now();
    state.recordingPending = false;
    elements.recordState.textContent = "Recording";
    elements.recordDetail.textContent = `Speak naturally. Recording stops automatically just before the ${MAX_RECORD_SECONDS}-second upload limit.`;
    elements.recordVisual.classList.add("active");
    elements.recordStart.hidden = true;
    elements.recordStop.hidden = false;
    elements.recordDiscard.hidden = true;
    elements.recordReview.hidden = true;
    updateRecordTimer();
    state.recordTimerHandle = window.setInterval(updateRecordTimer, 200);
    state.recordAutoStopHandle = window.setTimeout(stopRecording, RECORD_AUTO_STOP_SECONDS * 1000);
    setLoading(state.loading);
    elements.recordStop.focus();
    announce("Recording started.");
  } catch (error) {
    state.recordingPending = false;
    stopMediaTracks();
    setLoading(state.loading);
    const denied = error.name === "NotAllowedError" || error.name === "PermissionDeniedError";
    setRecordError(denied ? "Microphone permission was denied. Allow access in browser settings, or use a sample or upload." : `Could not start the microphone: ${error.message}`);
  }
}

function updateRecordTimer() {
  const elapsed = Math.max(0, (Date.now() - state.recordStartedAt) / 1000);
  elements.recordTimer.textContent = formatTime(elapsed, false);
  if (elapsed >= RECORD_AUTO_STOP_SECONDS) stopRecording();
}

function stopRecording() {
  if (state.mediaRecorder?.state === "recording") state.mediaRecorder.stop();
  window.clearInterval(state.recordTimerHandle);
  window.clearTimeout(state.recordAutoStopHandle);
  state.recordTimerHandle = null;
  state.recordAutoStopHandle = null;
  stopMediaTracks();
}

function stopMediaTracks() {
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
}

function finalizeRecording() {
  state.recordingPending = false;
  window.clearInterval(state.recordTimerHandle);
  window.clearTimeout(state.recordAutoStopHandle);
  state.recordTimerHandle = null;
  state.recordAutoStopHandle = null;
  stopMediaTracks();
  const type = state.mediaRecorder?.mimeType || state.recordChunks[0]?.type || "audio/webm";
  state.recordBlob = new Blob(state.recordChunks, { type });
  if (!state.recordBlob.size) {
    setRecordError("The recording was empty. Try again or use another audio source.");
    discardRecording({ keepMessage: true });
    return;
  }
  if (state.recordUrl) URL.revokeObjectURL(state.recordUrl);
  state.recordUrl = URL.createObjectURL(state.recordBlob);
  elements.recordPlayer.src = state.recordUrl;
  elements.recordPlayer.load();
  const extension = type.includes("mp4") ? "m4a" : "webm";
  state.recordDuration = Math.max(0, (Date.now() - state.recordStartedAt) / 1000);
  state.recordFile = new File([state.recordBlob], `microphone-recording.${extension}`, { type });
  selectFile(state.recordFile, { kind: "recording", title: "Microphone recording", detail: `${formatTime(state.recordDuration)} recording` });
  elements.recordState.textContent = "Recording ready";
  elements.recordDetail.textContent = "Review the captured audio, then analyze it with the active speech model.";
  elements.recordTimer.textContent = formatTime(state.recordDuration, false);
  elements.recordVisual.classList.remove("active");
  elements.recordStart.hidden = false;
  elements.recordStart.textContent = "Record again";
  elements.recordStop.hidden = true;
  elements.recordDiscard.hidden = false;
  elements.recordReview.hidden = false;
  setLoading(state.loading);
  elements.recordAnalyze.focus();
  announce("Recording stopped and ready for review.");
}

function discardRecording({ keepMessage = false } = {}) {
  state.recordingPending = false;
  if (state.mediaRecorder?.state === "recording") {
    state.mediaRecorder.removeEventListener("stop", finalizeRecording);
    state.mediaRecorder.stop();
  }
  window.clearInterval(state.recordTimerHandle);
  window.clearTimeout(state.recordAutoStopHandle);
  state.recordTimerHandle = null;
  state.recordAutoStopHandle = null;
  stopMediaTracks();
  state.mediaRecorder = null;
  state.recordChunks = [];
  state.recordBlob = null;
  state.recordFile = null;
  state.recordDuration = 0;
  if (state.recordUrl) URL.revokeObjectURL(state.recordUrl);
  state.recordUrl = "";
  elements.recordPlayer.removeAttribute("src");
  elements.recordReview.hidden = true;
  elements.recordVisual.classList.remove("active");
  elements.recordStart.hidden = false;
  elements.recordStart.textContent = "Start recording";
  elements.recordStop.hidden = true;
  elements.recordDiscard.hidden = true;
  elements.recordState.textContent = "Microphone ready";
  elements.recordDetail.textContent = "Record a short phrase, then review it before analysis.";
  elements.recordTimer.textContent = "00:00";
  if (state.selectedSourceKind === "recording") state.selectedFile = null;
  setLoading(state.loading);
  if (!keepMessage) setRecordError();
}

async function analyzeRecording() {
  if (!state.recordFile || state.loading || recordingInProgress()) return;
  selectFile(state.recordFile, {
    kind: "recording",
    title: "Microphone recording",
    detail: `${formatTime(state.recordDuration)} recording`,
  });
  await analyzeSelectedFile();
}

function buildSpeechDemoData() {
  const duration = 3.2;
  const layers = [0, 4, 8, 12, 14];
  const tokenSpecs = [
    [40, " I", 0.48, 2.39, " My"],
    [2846, " can", 0.62, 1.34, " will"],
    [17704, " hear", 0.57, 1.45, " understand"],
    [498, " you", 0.81, 0.71, " that"],
    [11065, " clearly", 0.74, 0.83, " now"],
    [13, ".", 0.95, 0.28, "!"],
  ];
  const tokens = tokenSpecs.map(([id, text, probability, entropy, alternative]) => ({
    id,
    text,
    probability,
    entropy,
    top_tokens: [
      { id, text, probability },
      { id: id + 100, text: alternative, probability: Math.max(0.01, (1 - probability) * 0.55) },
      { id: id + 200, text: " maybe", probability: Math.max(0.005, (1 - probability) * 0.2) },
    ],
  }));
  const positions = tokens.map((token, index) => ({
    index,
    token_id: token.id,
    text: token.text,
  }));
  const earlyCandidates = [" Let", " Could", " There", " Maybe", " I", " We"];
  const cells = layers.map((layer, layerIndex) => positions.map((position, positionIndex) => {
    const base = 7.8 - layerIndex * 0.16 + Math.sin((positionIndex + 1) * (layerIndex + 2)) * 0.24;
    const primary = layerIndex < 2
      ? earlyCandidates[(positionIndex + layerIndex) % earlyCandidates.length]
      : position.text;
    return {
      selected_score: base,
      top_tokens: [primary, position.text, earlyCandidates[(positionIndex + 2) % earlyCandidates.length], " response", " audio"]
        .map((text, rank) => ({
          id: 20_000 + layerIndex * 100 + positionIndex * 10 + rank,
          text,
          score: base - rank * (0.24 + layerIndex * 0.03),
          rank: rank + 1,
        })),
    };
  }));
  const waveform = Array.from({ length: 260 }, (_, index) => {
    const time = index / 260 * duration;
    const syllable = Math.pow(Math.max(0, Math.sin(time * Math.PI * 2.4)), 0.62);
    return Math.max(0.015, syllable * (0.38 + 0.52 * Math.sin(Math.PI * time / duration)) * (0.7 + 0.18 * Math.sin(index * 1.31)));
  });
  return {
    audio: { duration_seconds: duration, waveform },
    transcription: {
      text: "I can hear you clearly.",
      tokens,
      timing_source: "unavailable",
      timing_quality: "unavailable",
      semantic_role: "generated_response_text",
    },
    decoder: {
      score_kind: "raw_readout_logit",
      stream_kind: "causal_language_backbone",
      target_layer: 15,
      layers,
      positions,
      cells,
    },
    metadata: {
      backend: "mlx",
      model_family: "lfm2_audio",
      model_id: "mlx-community/LFM2.5-Audio-1.5B-8bit · synthetic demo",
      streams: ["decoder"],
      stream_labels: {
        decoder: "LFM language backbone",
        output: "Tied text head",
      },
      capabilities: {
        input_audio: true,
        generated_text: true,
        generated_audio: false,
        language_jlens: true,
        audio_encoder_jlens: false,
        audio_codebook_jlens: false,
      },
      lens_examples: 6,
      estimator: "Synthetic projected language J-lens fixture",
      projection: {
        method: "synthetic_subsampled_hadamard_output_probe_vjp",
        rank: 64,
        seed: 7,
      },
      display_vocabulary: {
        policy: "synthetic_demo_lexical_tokens",
        display_vocabulary_size: DEMO_DISPLAY_VOCABULARY_SIZE,
        full_vocabulary_size: DEMO_FULL_VOCABULARY_SIZE,
        exact_decoded_character_length_counts: { ...DEMO_LENGTH_COUNTS },
        maximum_decoded_character_length_counts: Object.fromEntries(
          Object.keys(DEMO_LENGTH_COUNTS).map((limit) => [
            limit,
            Object.entries(DEMO_LENGTH_COUNTS).filter(([length]) => Number(length) <= Number(limit)).reduce((sum, [, count]) => sum + count, 0),
          ]),
        ),
      },
      candidate_rank_semantics: {
        method: "1_plus_count_strictly_greater",
        ties: "equal scores share the same competition rank",
        lens_primary_space: "lexical_display_vocabulary",
        output_head_primary_space: "full_model_vocabulary",
        character_filter_merge: "merge disjoint exact-length buckets, sort by score, and rank by strictly greater scores",
      },
      decoder_token_length_filter: { eligible_source_layers: [] },
      warnings: [
        "Demo response text, probabilities, waveform, and projected readouts are synthetic.",
        "The synthetic fixture covers only the LFM language backbone and tied text head.",
        "Generated speech playback is omitted because this demo does not exercise an acoustic-code or waveform model.",
        "Projected lexical readouts are not probabilities, causal effects, or input-audio attribution.",
      ],
    },
  };
}

const DEMO_FULL_VOCABULARY_SIZE = 51_865;
const DEMO_DISPLAY_VOCABULARY_SIZE = 32;
const DEMO_LENGTH_COUNTS = Object.freeze({ 1: 5, 2: 5, 3: 5, 4: 5, 5: 6, 6: 6 });

function decorateDemoLensCandidates(candidates, { scoreKind, exactLength = null } = {}) {
  return [...candidates]
    .sort((left, right) => asFiniteNumber(right.score) - asFiniteNumber(left.score))
    .map((candidate, index) => {
      const displayRank = Math.min(DEMO_DISPLAY_VOCABULARY_SIZE, index + 1 + (exactLength === null ? 0 : Number(exactLength) - 1));
      const fullRank = Math.min(DEMO_FULL_VOCABULARY_SIZE, displayRank + 2);
      return {
        ...candidate,
        rank: index + 1,
        rank_denominator: exactLength === null ? DEMO_DISPLAY_VOCABULARY_SIZE : DEMO_LENGTH_COUNTS[exactLength],
        rank_space: exactLength === null ? "lexical_display_vocabulary" : "exact_decoded_character_length_bucket",
        display_vocabulary_rank: displayRank,
        display_vocabulary_denominator: DEMO_DISPLAY_VOCABULARY_SIZE,
        full_vocabulary_rank: fullRank,
        full_vocabulary_denominator: DEMO_FULL_VOCABULARY_SIZE,
        rank_tie_policy: "1_plus_count_strictly_greater",
        score_kind: scoreKind,
        vocabulary_filter: {
          display_lexical_filter_applied: true,
          character_length_filter_applied: exactLength !== null,
          decoded_character_length: String(candidate.text || "").trim().length,
          character_length_constraint: exactLength === null ? null : { operator: "exact", value: Number(exactLength) },
        },
      };
    });
}

function decorateDemoHeadCandidates(candidates) {
  return [...candidates]
    .sort((left, right) => asFiniteNumber(right.probability) - asFiniteNumber(left.probability))
    .map((candidate, index) => ({
      ...candidate,
      probability: clamp(candidate.probability),
      log_probability: Math.log(Math.max(1e-12, clamp(candidate.probability))),
      rank: index + 1,
      rank_denominator: DEMO_FULL_VOCABULARY_SIZE,
      rank_space: "full_model_vocabulary",
      full_vocabulary_rank: index + 1,
      full_vocabulary_denominator: DEMO_FULL_VOCABULARY_SIZE,
      rank_tie_policy: "1_plus_count_strictly_greater",
      score_kind: "raw_teacher_forced_probability",
      vocabulary_filter: {
        display_lexical_filter_applied: false,
        character_length_filter_applied: false,
        decoded_character_length: String(candidate.text || "").trim().length,
        character_length_constraint: null,
      },
    }));
}

function buildDemoData() {
  const duration = 5.6;
  const tokenSpecs = [
    [50364, " The", 0.18, 0.78, 0.94, 0.21], [2068, " quick", 0.78, 1.42, 0.88, 0.34],
    [7586, " brown", 1.42, 2.12, 0.82, 0.47], [21831, " fox", 2.12, 2.75, 0.91, 0.27],
    [11645, " jumps", 2.75, 3.58, 0.76, 0.61], [626, " over", 3.58, 4.24, 0.86, 0.39],
    [257, " it", 4.24, 4.74, 0.69, 0.72], [13, ".", 4.74, 5.08, 0.97, 0.09],
  ];
  const alternativePool = [[464, " A"], [2067, " slow"], [16562, " black"], [3290, " dog"], [18045, " runs"], [572, " under"], [340, " this"], [30, "!"]];
  const tokens = tokenSpecs.map(([id, text, start, end, probability, entropy], index) => {
    const topTokens = decorateDemoHeadCandidates([
      { id, text, probability },
      { id: alternativePool[index][0], text: alternativePool[index][1], probability: Math.max(0.01, (1 - probability) * 0.58) },
      { id: 50257, text: "<eot>", probability: Math.max(0.005, (1 - probability) * 0.19) },
    ]);
    const realized = topTokens.find((candidate) => candidate.id === id);
    return {
      id,
      text,
      start_seconds: start,
      end_seconds: end,
      probability,
      log_probability: Math.log(Math.max(1e-12, probability)),
      entropy,
      rank: realized.rank,
      rank_denominator: realized.rank_denominator,
      rank_space: realized.rank_space,
      full_vocabulary_rank: realized.full_vocabulary_rank,
      full_vocabulary_denominator: realized.full_vocabulary_denominator,
      rank_tie_policy: realized.rank_tie_policy,
      score_kind: realized.score_kind,
      vocabulary_filter: realized.vocabulary_filter,
      candidate_space: {
        primary_rank_space: "full_model_vocabulary",
        primary_rank_denominator: DEMO_FULL_VOCABULARY_SIZE,
        display_lexical_filter_applied: false,
        character_length_filter_available: false,
        character_length_filter_policy: null,
      },
      top_tokens: topTokens,
    };
  });
  const overlapSeconds = asFiniteNumber(elements.encoderOverlapSeconds?.value, 0.02);
  const windowSeconds = ENCODER_WINDOW_SECONDS;
  const hopSeconds = windowSeconds - overlapSeconds;
  const timeBins = [];
  for (let start = 0; start < duration; start += hopSeconds) {
    const end = Math.min(duration, start + windowSeconds);
    if (timeBins.length && end <= timeBins[timeBins.length - 1].end_seconds) break;
    timeBins.push({ start_seconds: start, end_seconds: end });
  }
  const encoderLayers = [0, 1, 2, 3];
  const acousticTokens = ["speech", "th", "kw", "br", "f", "j", "ow", "ih", "period", "end", "silence", "end"];
  const lexicalTokens = [" The", " quick", " quick", " brown", " fox", " jumps", " jumps", " over", " it", " period", " end", " end"];
  const demoPhones = ["DH", "AH", "K", "W", "IH", "B", "R", "AW", "N", "F", "AA", "K", "S", "JH", "AH", "M", "P", "S", "OW", "V", "ER", "IH", "T", "SIL"];
  const phoneAlternatives = ["AH", "IH", "EH", "N", "S", "T", "R", "K"];
  const tokensByLength = {
    1: ["a", "i", "s", "f", "o"], 2: ["th", "sh", "ng", "ch", "ow"],
    3: ["fox", "the", "end", "run", "mod"], 4: ["over", "slow", "card", "word", "jump"],
    5: ["quick", "brown", "jumps", "where", "sound"], 6: ["speech", "period", "tokens", "sample", "listen"],
  };
  const encoderCells = encoderLayers.map((layer, layerIndex) => timeBins.map((bin, columnIndex) => {
    const tokenIndex = Math.min(
      acousticTokens.length - 1,
      Math.floor(columnIndex / Math.max(1, timeBins.length) * acousticTokens.length),
    );
    const target = layerIndex < 2 ? acousticTokens[tokenIndex] : lexicalTokens[tokenIndex];
    const next = lexicalTokens[Math.min(lexicalTokens.length - 1, tokenIndex + 1)];
    const phoneIndex = Math.min(demoPhones.length - 1, Math.floor(columnIndex / Math.max(1, timeBins.length) * demoPhones.length));
    const primaryPhone = demoPhones[phoneIndex];
    const base = 0.09 + layerIndex * 0.085 + Math.sin(columnIndex * 1.4 + layerIndex) * 0.055;
    return {
      position_index: columnIndex,
      time_window: { start_seconds: bin.start_seconds, end_seconds: bin.end_seconds, timing_source: "encoder_pooling_window" },
      candidate_space: {
        primary_rank_space: "lexical_display_vocabulary",
        primary_rank_denominator: DEMO_DISPLAY_VOCABULARY_SIZE,
        display_lexical_filter_applied: true,
        character_length_filter_available: true,
        character_length_filter_policy: "exact_decoded_character_length_buckets",
      },
      selected_score: base,
      phone_signatures: [primaryPhone, ...phoneAlternatives.filter((phone) => phone !== primaryPhone)]
        .slice(0, 5)
        .map((phone, rank) => ({ phone, rank: rank + 1, similarity: 0.78 - rank * 0.075 + layerIndex * 0.012 + Math.sin(columnIndex + rank) * 0.018 })),
      top_tokens: decorateDemoLensCandidates(
        [target, next, " speech", " the", " end"].map((text, rank) => ({ id: 1000 + columnIndex * 20 + layerIndex * 5 + rank, text, score: base - rank * (0.026 + layerIndex * 0.006) })),
        { scoreKind: "target_mean_relative_logit_delta" },
      ),
      top_tokens_by_length: Object.fromEntries(Object.entries(tokensByLength).map(([length, candidates]) => [
        length,
        decorateDemoLensCandidates(
          candidates.map((text, rank) => ({ id: 3000 + Number(length) * 100 + columnIndex * 20 + layerIndex * 5 + rank, text, score: base - Number(length) * 0.032 - rank * (0.021 + layerIndex * 0.004) })),
          { scoreKind: "target_mean_relative_logit_delta", exactLength: Number(length) },
        ),
      ])),
    };
  }));
  const decoderLayers = [0, 1, 2];
  const positions = tokens.map((token, index) => ({ index, token_id: token.id, text: token.text, start_seconds: token.start_seconds, end_seconds: token.end_seconds }));
  const generic = [" the", " and", " a", " to"];
  const decoderCells = decoderLayers.map((layer, layerIndex) => positions.map((position, columnIndex) => {
    const actual = position.text;
    const first = layerIndex === 0 ? generic[columnIndex % generic.length] : actual;
    const base = 0.16 + layerIndex * 0.15 + Math.sin(columnIndex * 1.9 + layerIndex) * 0.065;
    const cell = {
      position_index: columnIndex,
      time_window: { start_seconds: position.start_seconds, end_seconds: position.end_seconds, timing_source: "whisper_cross_attention_dtw" },
      candidate_space: {
        primary_rank_space: "lexical_display_vocabulary",
        primary_rank_denominator: DEMO_DISPLAY_VOCABULARY_SIZE,
        display_lexical_filter_applied: true,
        character_length_filter_available: layer === 0 || layer === 1,
        character_length_filter_policy: layer === 0 || layer === 1 ? "exact_decoded_character_length_buckets" : null,
      },
      selected_score: base,
      top_tokens: decorateDemoLensCandidates(
        [first, layerIndex === 0 ? actual : alternativePool[columnIndex][1], " end", " the", " speech"].map((text, rank) => ({ id: 5000 + columnIndex * 20 + layerIndex * 5 + rank, text, score: base - rank * (0.037 + layerIndex * 0.012) })),
        { scoreKind: "raw_readout_logit" },
      ),
    };
    if (layer === 0 || layer === 1) {
      cell.top_tokens_by_length = Object.fromEntries(Object.entries(tokensByLength).map(([length, candidates]) => [
        length,
        decorateDemoLensCandidates(
          candidates.map((text, rank) => ({ id: 7000 + Number(length) * 100 + columnIndex * 20 + layerIndex * 5 + rank, text, score: base - Number(length) * 0.029 - rank * (0.024 + layerIndex * 0.005) })),
          { scoreKind: "raw_readout_logit", exactLength: Number(length) },
        ),
      ]));
    }
    return cell;
  }));
  const waveform = Array.from({ length: 260 }, (_, index) => {
    const time = index / 260 * duration;
    const syllable = Math.pow(Math.max(0, Math.sin(time * Math.PI * 2.15)), 0.55);
    return Math.max(0.018, syllable * (0.36 + 0.64 * Math.pow(Math.sin(time / duration * Math.PI), 0.7)) * (0.56 + 0.24 * Math.sin(index * 1.83) + 0.12 * Math.sin(index * 0.47)));
  });
  return {
    audio: { duration_seconds: duration, waveform },
    transcription: { text: "The quick brown fox jumps over it.", tokens, timing_source: "synthetic_demo", timing_quality: "synthetic" },
    encoder: {
      score_kind: "target_mean_relative_logit_delta", layers: encoderLayers, time_bins: timeBins, cells: encoderCells,
      pooling: {
        requested_window_seconds: windowSeconds, requested_overlap_seconds: overlapSeconds,
        effective_window_seconds: windowSeconds, effective_overlap_seconds: overlapSeconds,
        effective_hop_seconds: hopSeconds, adaptive_for_max_bins: false, max_time_bins: 100,
      },
    },
    decoder: { score_kind: "raw_readout_logit", layers: decoderLayers, positions, cells: decoderCells },
    metadata: {
      model_id: "openai/whisper-tiny · demo", streams: ["encoder", "decoder"], lens_examples: 32, estimator: "Synthetic UI fixture",
      display_vocabulary: {
        policy: "synthetic_demo_lexical_tokens",
        display_vocabulary_size: DEMO_DISPLAY_VOCABULARY_SIZE,
        full_vocabulary_size: DEMO_FULL_VOCABULARY_SIZE,
        exact_decoded_character_length_counts: { ...DEMO_LENGTH_COUNTS },
        maximum_decoded_character_length_counts: Object.fromEntries(
          Object.keys(DEMO_LENGTH_COUNTS).map((limit) => [
            limit,
            Object.entries(DEMO_LENGTH_COUNTS).filter(([length]) => Number(length) <= Number(limit)).reduce((sum, [, count]) => sum + count, 0),
          ]),
        ),
      },
      candidate_rank_semantics: {
        method: "1_plus_count_strictly_greater",
        ties: "equal scores share the same competition rank",
        lens_primary_space: "lexical_display_vocabulary",
        output_head_primary_space: "full_model_vocabulary",
        character_filter_merge: "merge disjoint exact-length buckets, sort by score, and rank by strictly greater scores",
      },
      encoder_token_length_filter: { policy: "exact_decoded_character_length_buckets", maximum_available_length: 6, character_count_ignores_surrounding_whitespace: true },
      phone_signature: {
        available: true,
        method: "synthetic frozen phone-prototype cosine",
        phone_count: 34,
        score_kind: "cosine_similarity",
        demo_only: true,
      },
      decoder_token_length_filter: { policy: "exact_decoded_character_length_buckets", eligible_source_layers: [0, 1], maximum_available_length: 6, character_count_ignores_surrounding_whitespace: true },
      warnings: [
        "Demo transcript, timing, probabilities, and readout scores are synthetic.",
        "J-lens colors are layer-normalized rank/intensity and cannot be compared as percentages.",
        "Encoder and decoder readouts are not signed causal effects or calibrated model confidence.",
        "Demo phone labels and prototype similarities are synthetic and only exercise the interface.",
        "Decoder token-length filtering reranks the complete synthetic L0/L1 vocabulary buckets; L2 and the output head remain unchanged.",
      ],
    },
  };
}

function createDemoAudio(duration) {
  const sampleRate = 16000;
  const sampleCount = Math.floor(duration * sampleRate);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeString = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  writeString(0, "RIFF"); view.setUint32(4, 36 + sampleCount * 2, true); writeString(8, "WAVE"); writeString(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); writeString(36, "data"); view.setUint32(40, sampleCount * 2, true);
  const notes = [185, 232, 208, 262, 247, 294, 220, 330];
  for (let index = 0; index < sampleCount; index += 1) {
    const time = index / sampleRate;
    const noteIndex = Math.min(notes.length - 1, Math.floor(time / duration * notes.length));
    const local = (time / duration * notes.length) % 1;
    const envelope = Math.pow(Math.sin(Math.PI * local), 1.8) * 0.22;
    const carrier = Math.sin(2 * Math.PI * notes[noteIndex] * time) + 0.28 * Math.sin(2 * Math.PI * notes[noteIndex] * 2.01 * time);
    view.setInt16(44 + index * 2, clamp(carrier * envelope, -1, 1) * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function updateEncoderMetricLabel() {
  elements.encoderScoreLabel.textContent = state.encoderPhoneSignatureEnabled
    ? "Phone labels: cosine similarity to frozen prototypes · blue intensity is within-layer percentile · never probability"
    : state.encoderTokenLengthFilterEnabled
      ? `Vocabulary limited to ≤ ${state.encoderMaxTokenLength} characters · encoder layers reranked · not probability`
      : "Blue strip intensity: within-layer percentile of target-mean-relative logit delta · not probability";
}

function updateEncoderPhoneSignatureMode({ enabled = null, announceChange = false, rerender = true } = {}) {
  const available = phoneSignatureAvailable();
  state.encoderPhoneSignatureAvailable = available;
  if (enabled !== null) state.encoderPhoneSignatureEnabled = Boolean(enabled) && available;
  if (!available) state.encoderPhoneSignatureEnabled = false;
  const metadata = phoneSignatureMetadata();
  const method = String(metadata?.method || metadata?.estimator || "fitted phone prototypes").replaceAll("_", " ");
  elements.encoderPhoneSignatureToggle.disabled = !available;
  elements.encoderPhoneSignatureToggle.setAttribute("aria-pressed", String(state.encoderPhoneSignatureEnabled));
  elements.encoderPhoneSignatureToggle.querySelector("i").textContent = state.encoderPhoneSignatureEnabled ? "On" : "Off";
  elements.encoderPhoneMode.classList.toggle("available", available);
  elements.encoderPhoneSignatureStatus.textContent = available
    ? state.encoderPhoneSignatureEnabled
      ? `${method} enabled · token-length reranking is paused`
      : `${method} available · token readout remains the default`
    : state.result
      ? "Unavailable for this result: no matching fitted phone signatures were returned."
      : "Run an analysis with matching fitted phone prototypes to enable this view.";
  elements.encoderTokenLengthFilter.disabled = state.encoderPhoneSignatureEnabled;
  elements.encoderMaxTokenLength.disabled = state.encoderPhoneSignatureEnabled || !state.encoderTokenLengthFilterEnabled;
  elements.encoderLengthSummary.textContent = state.encoderPhoneSignatureEnabled
    ? `Paused in phone view · lexical limit remains ≤ ${state.encoderMaxTokenLength}`
    : `${state.encoderTokenLengthFilterEnabled ? "Keeping" : "Enable to keep"} tokens with ≤ ${state.encoderMaxTokenLength} characters`;
  updateEncoderMetricLabel();
  if (rerender && state.views.encoder) refreshCompactStream("encoder");
  if (announceChange) announce(state.encoderPhoneSignatureEnabled
    ? "Phone signature view enabled. Encoder cells now show cosine similarity to frozen phone prototypes; the token-length filter is paused."
    : available
      ? "Phone signature view disabled. Encoder lexical token readouts restored."
      : "Phone signature view is unavailable for this result because matching fitted prototypes were not returned.");
}

function updateEncoderTokenLengthFilter({ announceChange = false, normalizeInput = false, rerender = true } = {}) {
  const parsedLength = Math.floor(Number(elements.encoderMaxTokenLength.value));
  if (Number.isFinite(parsedLength) && parsedLength >= 1) state.encoderMaxTokenLength = parsedLength;
  else if (normalizeInput) elements.encoderMaxTokenLength.value = String(state.encoderMaxTokenLength);
  state.encoderTokenLengthFilterEnabled = elements.encoderTokenLengthFilter.checked;
  elements.encoderTokenLengthFilter.disabled = state.encoderPhoneSignatureEnabled;
  elements.encoderMaxTokenLength.disabled = state.encoderPhoneSignatureEnabled || !state.encoderTokenLengthFilterEnabled;
  elements.encoderLengthSummary.textContent = state.encoderPhoneSignatureEnabled
    ? `Paused in phone view · lexical limit remains ≤ ${state.encoderMaxTokenLength}`
    : `${state.encoderTokenLengthFilterEnabled ? "Keeping" : "Enable to keep"} tokens with ≤ ${state.encoderMaxTokenLength} characters`;
  const view = state.views.encoder;
  updateEncoderMetricLabel();
  if (rerender && view) {
    refreshCompactStream("encoder");
  }
  if (announceChange) announce(state.encoderTokenLengthFilterEnabled
    ? `Encoder token-length filter enabled at ${state.encoderMaxTokenLength} characters.`
    : "Encoder token-length filter disabled. Full lexical display rankings restored.");
}

function updateDecoderTokenLengthFilter({ announceChange = false, normalizeInput = false, rerender = true } = {}) {
  const parsedLength = Math.floor(Number(elements.decoderMaxTokenLength.value));
  if (Number.isFinite(parsedLength) && parsedLength >= 1) state.decoderMaxTokenLength = parsedLength;
  else if (normalizeInput) elements.decoderMaxTokenLength.value = String(state.decoderMaxTokenLength);
  const available = !state.result || decoderLengthFilterLayers().length > 0;
  elements.decoderFilterNote.hidden = Boolean(state.result) && !available;
  elements.decoderTokenLengthFilter.disabled = !available;
  if (!available) elements.decoderTokenLengthFilter.checked = false;
  state.decoderTokenLengthFilterEnabled = available && elements.decoderTokenLengthFilter.checked;
  elements.decoderMaxTokenLength.disabled = !state.decoderTokenLengthFilterEnabled;
  elements.decoderLengthSummary.textContent = available
    ? `${state.decoderTokenLengthFilterEnabled ? "Keeping" : "Enable to keep"} L0–L1 tokens with ≤ ${state.decoderMaxTokenLength} characters`
    : "Unavailable: this result has no full-vocabulary L0–L1 length buckets";
  const view = state.views.decoder;
  elements.decoderScoreLabel.textContent = state.decoderTokenLengthFilterEnabled
    ? `L0–L1 vocabulary limited to ≤ ${state.decoderMaxTokenLength} characters · L2 and output head unchanged`
    : "Blue strip intensity: within-layer percentile of raw readout logit · orange HEAD: actual probability";
  if (rerender && view) {
    refreshCompactStream("decoder");
  }
  if (announceChange) announce(!available
    ? "Decoder token-length filtering is unavailable because this result has no full-vocabulary L0 and L1 length buckets."
    : state.decoderTokenLengthFilterEnabled
      ? `Decoder token-length filter enabled at ${state.decoderMaxTokenLength} characters for L0 and L1. L2 and the output head remain unchanged.`
      : "Decoder token-length filter disabled. Full lexical display rankings restored for L0 and L1.");
}

function bindEvents() {
  const updateEncoderWindowSummary = ({ announceChange = false } = {}) => {
    const overlapMilliseconds = Math.round(asFiniteNumber(elements.encoderOverlapSeconds.value) * 1000);
    const windowMilliseconds = Math.round(ENCODER_WINDOW_SECONDS * 1000);
    elements.encoderHopSummary.textContent = `${windowMilliseconds - overlapMilliseconds} ms hop`;
    if (announceChange) announce(`${overlapMilliseconds} millisecond encoder overlap selected. It will apply to the next analysis.`);
  };
  elements.encoderOverlapSeconds.addEventListener("change", () => updateEncoderWindowSummary({ announceChange: true }));
  updateEncoderWindowSummary();
  elements.statusButton.addEventListener("click", checkBackendStatus);
  elements.sourceTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateSourceTab(tab.dataset.sourceTab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let target = index;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") target += 1;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") target -= 1;
      else if (event.key === "Home") target = 0;
      else target = elements.sourceTabs.length - 1;
      target = (target + elements.sourceTabs.length) % elements.sourceTabs.length;
      activateSourceTab(elements.sourceTabs[target].dataset.sourceTab, { focus: true });
    });
  });
  elements.browseButton.addEventListener("click", () => elements.audioFile.click());
  elements.audioFile.addEventListener("change", (event) => selectFile(event.target.files?.[0], { kind: "upload" }));
  elements.analyzeButton.addEventListener("click", analyzeSelectedFile);
  elements.selectedAnalyze.addEventListener("click", analyzeSelectedFile);
  elements.demoButton.addEventListener("click", loadDemo);
  elements.resetButton.addEventListener("click", resetAnalysis);
  elements.closeInspector.addEventListener("click", () => clearInspector({ restoreFocus: true }));
  elements.recordStart.addEventListener("click", startRecording);
  elements.recordStop.addEventListener("click", stopRecording);
  elements.recordDiscard.addEventListener("click", discardRecording);
  elements.recordAnalyze.addEventListener("click", analyzeRecording);

  ["dragenter", "dragover"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.add("drag-active"); }));
  ["dragleave", "drop"].forEach((type) => elements.dropZone.addEventListener(type, (event) => { event.preventDefault(); elements.dropZone.classList.remove("drag-active"); }));
  elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer?.files?.[0], { kind: "upload" }));
  elements.dropZone.addEventListener("click", (event) => { if (event.target === elements.dropZone || event.target.closest(".drop-copy, .drop-icon")) elements.audioFile.click(); });

  ["timeupdate", "play", "pause"].forEach((type) => elements.audioPlayer.addEventListener(type, updatePlaybackUi));
  elements.audioPlayer.addEventListener("loadedmetadata", () => { if (!state.duration && Number.isFinite(elements.audioPlayer.duration)) state.duration = elements.audioPlayer.duration; updatePlaybackUi(); });
  elements.waveform.addEventListener("pointermove", (event) => {
    const rect = elements.waveform.getBoundingClientRect();
    elements.waveformHover.hidden = false;
    elements.waveformHover.style.left = `${clamp((event.clientX - rect.left) / rect.width) * 100}%`;
  });
  elements.waveform.addEventListener("pointerleave", () => { elements.waveformHover.hidden = true; });
  elements.waveform.addEventListener("click", (event) => {
    const rect = elements.waveform.getBoundingClientRect();
    selectTimelineAtTime(clamp((event.clientX - rect.left) / rect.width) * analysisDuration(), { origin: "waveform", seek: true, announceChange: true });
  });
  elements.waveform.addEventListener("keydown", (event) => {
    const duration = effectiveDuration();
    if (!duration) return;
    let target = finiteNumberOrNull(state.timelineSelection.timeSeconds) ?? elements.audioPlayer.currentTime;
    if (event.key === "ArrowLeft") target -= event.shiftKey ? 5 : 0.5;
    else if (event.key === "ArrowRight") target += event.shiftKey ? 5 : 0.5;
    else if (event.key === "Home") target = 0;
    else if (event.key === "End") target = duration;
    else return;
    event.preventDefault();
    selectTimelineAtTime(clamp(target / duration) * analysisDuration(), { origin: "waveform", seek: true, announceChange: true });
  });

  if ("ResizeObserver" in window) {
    state.resizeObserver = new ResizeObserver(drawWaveform);
    state.resizeObserver.observe(elements.waveform);
    state.resizeObserver.observe(elements.encoderLayers);
  } else window.addEventListener("resize", drawWaveform);
  elements.encoderPhoneSignatureToggle.addEventListener("click", () => updateEncoderPhoneSignatureMode({
    enabled: !state.encoderPhoneSignatureEnabled,
    announceChange: true,
  }));
  elements.encoderTokenLengthFilter.addEventListener("change", () => updateEncoderTokenLengthFilter({ announceChange: true }));
  elements.encoderMaxTokenLength.addEventListener("input", () => updateEncoderTokenLengthFilter({ announceChange: false }));
  elements.encoderMaxTokenLength.addEventListener("change", () => updateEncoderTokenLengthFilter({ announceChange: true, normalizeInput: true }));
  elements.decoderTokenLengthFilter.addEventListener("change", () => updateDecoderTokenLengthFilter({ announceChange: true }));
  elements.decoderMaxTokenLength.addEventListener("input", () => updateDecoderTokenLengthFilter({ announceChange: false }));
  elements.decoderMaxTokenLength.addEventListener("change", () => updateDecoderTokenLengthFilter({ announceChange: true, normalizeInput: true }));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.timelineTooltip.hidden) hideTimelineTooltip();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!elements.timelineTooltip.hidden && !event.target.closest(".lens-timeline-cell")) hideTimelineTooltip();
  });
  window.addEventListener("scroll", hideTimelineTooltip, { capture: true, passive: true });
  window.addEventListener("resize", hideTimelineTooltip, { passive: true });
  window.addEventListener("beforeunload", () => {
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    if (state.recordUrl) URL.revokeObjectURL(state.recordUrl);
    cancelActiveAnalysisJob({ keepalive: true });
    state.requestController?.abort();
    stopMediaTracks();
  });
}

bindEvents();
checkBackendStatus();
loadSamples();
if (!recordingSupported()) {
  elements.recordDetail.textContent = "Recording is not supported in this browser. Use a prepared sample or upload.";
  elements.recordStart.disabled = true;
}
