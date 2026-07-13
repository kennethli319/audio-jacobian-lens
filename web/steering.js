"use strict";

(() => {
  const body = document.body;
  const resultsUrl = body.dataset.resultsUrl || "./data/phone-steering-results.json";
  const targetKeys = ["yanny", "laurel"];
  const state = { data: null, targetKey: "yanny", checkpointIndex: 0 };

  const elements = {
    status: document.querySelector("#load-status"),
    app: document.querySelector("#steering-app"),
    model: document.querySelector("#model-label"),
    baseline: document.querySelector("#baseline-label"),
    source: document.querySelector("#source-link"),
    targetButtons: [...document.querySelectorAll("[data-target]")],
    evidenceBadge: document.querySelector("#evidence-badge"),
    evidenceSummary: document.querySelector("#evidence-summary"),
    methodLabel: document.querySelector("#method-label"),
    methodDescription: document.querySelector("#method-description"),
    timelineRuler: document.querySelector("#timeline-ruler"),
    phoneRibbon: document.querySelector("#phone-ribbon"),
    heatmap: document.querySelector("#coefficient-heatmap"),
    coefficientNote: document.querySelector("#coefficient-note"),
    range: document.querySelector("#checkpoint-range"),
    checkpointLabel: document.querySelector("#checkpoint-label"),
    checkpointMarks: document.querySelector("#checkpoint-marks"),
    checkpointNote: document.querySelector("#checkpoint-note"),
    budget: document.querySelector("#budget-value"),
    baselineTranscript: document.querySelector("#baseline-transcript"),
    baselineTokenIds: document.querySelector("#baseline-token-ids"),
    selectedCard: document.querySelector("#selected-outcome-card"),
    selectedTranscript: document.querySelector("#selected-transcript"),
    selectedTokenIds: document.querySelector("#selected-token-ids"),
    outcomeRole: document.querySelector("#outcome-role"),
    decisionTitle: document.querySelector("#decision-title"),
    rankSemantics: document.querySelector("#rank-semantics"),
    decisionGrid: document.querySelector("#decision-grid"),
    pathSummary: document.querySelector("#path-summary"),
    controls: document.querySelector("#control-grid"),
    caveats: document.querySelector("#caveat-list"),
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    })[character]);
  }

  function asFinite(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function assertResults(data) {
    if (!data || typeof data !== "object") throw new Error("The steering result is not an object.");
    if (data.schema_version !== 1) throw new Error("This page does not support the steering-result schema version.");
    if (data.mode !== "static_recorded_checkpoints") throw new Error("The steering result is not marked as a static recorded-checkpoint replay.");
    if (!data.baseline?.generated || !data.baseline?.decisions) throw new Error("The steering result has no complete baseline.");
    targetKeys.forEach((key) => {
      const target = data.targets?.[key];
      if (!target || !Array.isArray(target.schedule) || !target.schedule.length) throw new Error(`The ${key} phone schedule is missing.`);
      if (!Array.isArray(target.layers) || !target.layers.length) throw new Error(`The ${key} layer schedule is missing.`);
      if (!Array.isArray(target.checkpoints) || !target.checkpoints.length) throw new Error(`The ${key} recorded checkpoints are missing.`);
      if (!Array.isArray(data.baseline.decisions[key])) throw new Error(`The ${key} baseline head decisions are missing.`);
      target.checkpoints.forEach((checkpoint) => {
        if (checkpoint.recorded !== true || checkpoint.interpolated !== false) throw new Error(`The ${key} replay contains a non-recorded checkpoint.`);
        if (!checkpoint.generated || !Array.isArray(checkpoint.decisions)) throw new Error(`The ${key} checkpoint is incomplete.`);
      });
    });
    return data;
  }

  function generatedText(generated) {
    return String(generated?.text ?? generated?.transcript ?? "—").trim() || "—";
  }

  function tokenIds(generated) {
    const ids = generated?.token_ids ?? generated?.ordinary_token_ids ?? [];
    return Array.isArray(ids) ? `[${ids.join(", ")}]` : "—";
  }

  function percent(value) {
    const numeric = asFinite(value, NaN);
    if (!Number.isFinite(numeric)) return "—";
    const scaled = Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
    if (Math.abs(scaled) < 0.001) return `${scaled.toFixed(6)}%`;
    if (Math.abs(scaled) < 0.1) return `${scaled.toFixed(4)}%`;
    if (Math.abs(scaled) < 10) return `${scaled.toFixed(3)}%`;
    return `${scaled.toFixed(2)}%`;
  }

  function budgetLabel(checkpoint) {
    return percent(asFinite(checkpoint.budget_fraction));
  }

  function checkpointLabel(checkpoint) {
    return String(checkpoint.label || checkpoint.role || checkpoint.id || "Recorded point");
  }

  function baselineCheckpoint(targetKey) {
    const baseline = state.data.baseline;
    return {
      id: "baseline",
      label: baseline.label || "No edit",
      role: "baseline",
      recorded: true,
      interpolated: false,
      budget_fraction: asFinite(baseline.budget_fraction),
      coefficient_scale: 0,
      generated: baseline.generated,
      decisions: baseline.decisions[targetKey],
      sequence_probability_product: baseline.sequence_probability_products?.[targetKey],
      note: "Original Whisper generation with no residual intervention.",
    };
  }

  function checkpoints(targetKey = state.targetKey) {
    const target = state.data.targets[targetKey];
    const recorded = [...target.checkpoints];
    return recorded.some((checkpoint) => checkpoint.role === "baseline" || asFinite(checkpoint.budget_fraction) === 0)
      ? recorded
      : [baselineCheckpoint(targetKey), ...recorded];
  }

  function targetData() {
    return state.data.targets[state.targetKey];
  }

  function selectedCheckpoint() {
    return checkpoints()[state.checkpointIndex] || checkpoints()[0];
  }

  function defaultCheckpointIndex(targetKey) {
    const rows = checkpoints(targetKey);
    const index = rows.findIndex((checkpoint) => checkpoint.role === "recommended" || checkpoint.recommended === true);
    return index >= 0 ? index : rows.length - 1;
  }

  function evidenceTone(target) {
    const tone = String(target.evidence?.tone || target.evidence?.tier || "caution").toLowerCase();
    if (/strong|open.loop|cross.fit/.test(tone)) return "strong";
    if (/limit|weak|overfit/.test(tone)) return "limited";
    return "caution";
  }

  function renderTargetHeader() {
    const target = targetData();
    elements.targetButtons.forEach((button) => {
      const selected = button.dataset.target === state.targetKey;
      button.setAttribute("aria-pressed", String(selected));
    });
    const evidence = target.evidence || {};
    elements.evidenceBadge.className = `evidence-badge ${evidenceTone(target)}`;
    elements.evidenceBadge.textContent = evidence.badge || evidence.tier || "Exploratory evidence";
    elements.evidenceSummary.textContent = evidence.summary || "Review the recorded controls with the target outcome.";
    elements.methodLabel.textContent = target.method?.label || target.method?.kind || "Recorded encoder residual schedule";
    elements.methodDescription.textContent = target.method?.description || "Fitted phone-prototype directions are pulled back to encoder residual space before downstream recomputation.";
  }

  function timelineDuration(target) {
    const scheduledEnd = Math.max(...target.schedule.map((segment) => asFinite(segment.end_seconds)));
    return Math.max(scheduledEnd, asFinite(state.data.source?.duration_seconds, scheduledEnd));
  }

  function renderTimeline() {
    const target = targetData();
    const duration = timelineDuration(target);
    const tickCount = 5;
    elements.timelineRuler.replaceChildren(...Array.from({ length: tickCount }, (_, index) => {
      const tick = document.createElement("i");
      const time = duration * index / (tickCount - 1);
      tick.className = "timeline-tick";
      tick.style.left = `${100 * index / (tickCount - 1)}%`;
      tick.innerHTML = `<span>${time.toFixed(2)} s</span>`;
      return tick;
    }));
    elements.phoneRibbon.replaceChildren(...target.schedule.map((segment) => {
      const start = asFinite(segment.start_seconds);
      const end = asFinite(segment.end_seconds);
      const item = document.createElement("div");
      item.className = "phone-segment";
      item.style.left = `${100 * start / duration}%`;
      item.style.width = `${100 * Math.max(0, end - start) / duration}%`;
      item.innerHTML = `<strong>${escapeHtml(segment.phone)}</strong><small>${start.toFixed(2)}–${end.toFixed(2)}</small>`;
      item.setAttribute("aria-label", `${segment.phone}, ${start.toFixed(2)} to ${end.toFixed(2)} seconds`);
      return item;
    }));
  }

  function coefficientRows(target, checkpoint) {
    const rows = Array.isArray(checkpoint.coefficient_heatmap)
      ? checkpoint.coefficient_heatmap
      : target.coefficient_heatmap;
    const byLayer = new Map((rows || []).map((row) => [String(row.layer), row.values || row.coefficients || []]));
    const scale = asFinite(checkpoint.coefficient_scale, 1);
    return target.layers.map((layer) => ({
      layer,
      values: (byLayer.get(String(layer)) || target.schedule.map(() => 0)).map((value) => asFinite(value) * scale),
    }));
  }

  function maximumCoefficient(target) {
    return Math.max(1e-12, ...checkpoints().flatMap((checkpoint) => coefficientRows(target, checkpoint).flatMap((row) => row.values.map(Math.abs))));
  }

  function coefficientText(value) {
    const absolute = Math.abs(value);
    if (absolute === 0) return "0";
    if (absolute < 0.001) return value.toExponential(2);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function renderHeatmap() {
    const target = targetData();
    const checkpoint = selectedCheckpoint();
    const rows = coefficientRows(target, checkpoint);
    const maxCoefficient = maximumCoefficient(target);
    const columns = target.schedule.map((segment) => `${Math.max(0.01, asFinite(segment.end_seconds) - asFinite(segment.start_seconds))}fr`).join(" ");
    elements.heatmap.replaceChildren(...rows.map((row) => {
      const layerRow = document.createElement("div");
      layerRow.className = "heatmap-row";
      layerRow.setAttribute("role", "row");
      const cells = row.values.map((value, index) => {
        const segment = target.schedule[index] || {};
        const heat = Math.min(1, Math.abs(value) / maxCoefficient);
        return `<div class="heatmap-cell${value === 0 ? " zero" : ""}" style="--heat:${heat.toFixed(4)}" role="cell" aria-label="Encoder L${escapeHtml(row.layer)}, ${escapeHtml(segment.phone)}, coefficient ${escapeHtml(coefficientText(value))}"><strong>${escapeHtml(segment.phone || "—")}</strong><span>${escapeHtml(coefficientText(value))}</span></div>`;
      }).join("");
      layerRow.innerHTML = `<strong class="heatmap-layer" role="rowheader">L${escapeHtml(row.layer)}</strong><div class="heatmap-cells" style="grid-template-columns:${columns}">${cells}</div>`;
      return layerRow;
    }));
    const policy = target.method?.coefficient_policy || "Cell values are recorded residual-norm coefficients.";
    const scaleText = checkpoint.role === "baseline" ? "All coefficients are zero." : `Selected coefficient scale: ${coefficientText(asFinite(checkpoint.coefficient_scale, 1))}.`;
    elements.coefficientNote.textContent = `${policy} ${scaleText}`;
  }

  function renderCheckpointControls() {
    const rows = checkpoints();
    elements.range.disabled = false;
    elements.range.min = "0";
    elements.range.max = String(rows.length - 1);
    elements.range.value = String(state.checkpointIndex);
    elements.range.setAttribute("aria-valuetext", checkpointLabel(selectedCheckpoint()));
    elements.checkpointMarks.replaceChildren(...rows.map((checkpoint, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.checkpointIndex = String(index);
      button.textContent = checkpoint.short_label || (checkpoint.role === "baseline" ? "0%" : budgetLabel(checkpoint));
      button.setAttribute("aria-label", `Show ${checkpointLabel(checkpoint)}, ${budgetLabel(checkpoint)} aggregate budget`);
      button.setAttribute("aria-current", String(index === state.checkpointIndex));
      button.addEventListener("click", () => selectCheckpoint(index, true));
      return button;
    }));
  }

  function roleLabel(checkpoint) {
    const role = String(checkpoint.role || "recorded").replaceAll("_", " ");
    if (checkpoint.generated?.target_match === true) return role === "recorded" ? "Target generated" : role;
    return role;
  }

  function renderOutcome() {
    const target = targetData();
    const checkpoint = selectedCheckpoint();
    const baseline = state.data.baseline.generated;
    const generated = checkpoint.generated;
    const targetMatch = generated?.target_match === true;
    elements.baselineTranscript.textContent = generatedText(baseline);
    elements.baselineTokenIds.textContent = `ordinary IDs ${tokenIds(baseline)}`;
    elements.selectedTranscript.textContent = generatedText(generated);
    elements.selectedTokenIds.textContent = `ordinary IDs ${tokenIds(generated)}`;
    elements.selectedCard.className = targetMatch ? "changed" : (checkpoint.role === "baseline" ? "" : "miss");
    elements.outcomeRole.className = `outcome-role ${targetMatch ? "success" : (String(checkpoint.role).includes("bound") || String(checkpoint.role).includes("failure") ? "boundary" : "")}`;
    elements.outcomeRole.textContent = roleLabel(checkpoint);
    elements.budget.textContent = budgetLabel(checkpoint);
    elements.checkpointLabel.textContent = checkpointLabel(checkpoint);
    const ray = checkpoint.ray_scale === undefined ? "" : ` · optimized-ray scale ${asFinite(checkpoint.ray_scale).toFixed(3)}`;
    elements.checkpointNote.textContent = `${checkpoint.note || "Saved free-generation and teacher-forced head measurements."}${ray} Recorded point; no browser interpolation.`;
    elements.decisionTitle.textContent = state.targetKey === "yanny" ? "Two tokenizer decisions—not one Yanny rank" : "One tokenizer decision for Laurel";
    elements.rankSemantics.textContent = state.data.rank_semantics?.description || `Rank among ${Number(state.data.model?.vocabulary_size || 51864).toLocaleString()} output-vocabulary entries`;

    const decisions = checkpoint.decisions || [];
    elements.decisionGrid.className = `decision-grid${decisions.length === 1 ? " single" : ""}`;
    elements.decisionGrid.innerHTML = decisions.map((decision, index) => decisionMarkup(decision, index)).join("");
    renderPathSummary(checkpoint, target);
  }

  function candidateMarkup(candidate, decision) {
    const target = Number(candidate.token_id) === Number(decision.token_id);
    return `<div class="candidate-row${target ? " target" : ""}" role="row"><span role="cell">#${Number(candidate.rank).toLocaleString()}</span><div class="candidate-token" role="cell"><strong>${escapeHtml(candidate.text || "[blank]")}</strong><small>ID ${escapeHtml(candidate.token_id)}</small></div><span role="cell">${escapeHtml(percent(candidate.probability))}</span></div>`;
  }

  function decisionMarkup(decision, index) {
    const rank = Number(decision.rank).toLocaleString();
    const denominator = Number(decision.rank_denominator || state.data.model?.vocabulary_size || 51864).toLocaleString();
    const context = decision.conditioning || decision.context || (index === 0 ? "conditioned on audio + decoder prefix" : "teacher-forced on the earlier target piece");
    const candidates = (decision.top_candidates || []).slice(0, 5);
    return `<article class="decision-card"><header class="decision-card-header"><div><span>${escapeHtml(decision.position || `decision ${index + 1}`)}</span><h4>${escapeHtml(decision.token || decision.text || `token ${decision.token_id}`)}</h4><p>${escapeHtml(context)}</p></div><div class="target-metric"><strong>${escapeHtml(percent(decision.probability))}</strong><span>#${escapeHtml(rank)} / ${escapeHtml(denominator)}</span></div></header><div class="candidate-table" role="table" aria-label="Top five full-head candidates for ${escapeHtml(decision.token || decision.text || decision.token_id)}">${candidates.map((candidate) => candidateMarkup(candidate, decision)).join("")}</div></article>`;
  }

  function renderPathSummary(checkpoint, target) {
    if (checkpoint.sequence_probability_product === undefined || checkpoint.decisions?.length < 2) {
      elements.pathSummary.hidden = true;
      elements.pathSummary.replaceChildren();
      return;
    }
    const baseline = asFinite(state.data.baseline.sequence_probability_products?.[state.targetKey], NaN);
    const current = asFinite(checkpoint.sequence_probability_product, NaN);
    const change = Number.isFinite(baseline) && baseline > 0 && Number.isFinite(current) ? current / baseline : null;
    elements.pathSummary.hidden = false;
    elements.pathSummary.innerHTML = `<div><span>Complete teacher-forced tokenizer path</span><strong>${escapeHtml(target.label)} joins the two conditional decisions above; EOS is excluded.</strong></div><strong>${escapeHtml(percent(current))}${change === null ? "" : ` · ${change.toFixed(change >= 10 ? 1 : 2)}×`}</strong>`;
  }

  function renderControls() {
    const target = targetData();
    elements.controls.innerHTML = (target.controls || []).map((control) => {
      const status = String(control.status || "recorded").toLowerCase();
      const statusTone = /fail|limit|does_not_transfer/.test(status)
        ? "limitation"
        : (/pass|support|reproduc/.test(status) ? "passed" : "recorded");
      const metrics = Array.isArray(control.metrics)
        ? control.metrics
        : Object.entries(control.metrics || {}).map(([label, value]) => ({ label, value }));
      return `<article class="control-card"><header><h3>${escapeHtml(control.label)}</h3><span class="control-status ${statusTone}">${escapeHtml(status.replaceAll("_", " "))}</span></header><p>${escapeHtml(control.summary)}</p>${metrics.length ? `<div class="control-metrics">${metrics.map((metric) => `<div><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong></div>`).join("")}</div>` : ""}</article>`;
    }).join("");
  }

  function renderCaveats() {
    const caveats = [...(targetData().caveats || []), ...(state.data.caveats || [])];
    elements.caveats.innerHTML = [...new Set(caveats.map(String))].map((caveat) => `<p>${escapeHtml(caveat)}</p>`).join("");
  }

  function updateUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set("target", state.targetKey);
    url.searchParams.set("checkpoint", selectedCheckpoint().id || String(state.checkpointIndex));
    history.replaceState(null, "", url);
  }

  function selectCheckpoint(index, updateHistory = false) {
    const last = checkpoints().length - 1;
    state.checkpointIndex = Math.max(0, Math.min(last, Number(index) || 0));
    elements.range.value = String(state.checkpointIndex);
    elements.range.setAttribute("aria-valuetext", checkpointLabel(selectedCheckpoint()));
    elements.checkpointMarks.querySelectorAll("button").forEach((button) => button.setAttribute("aria-current", String(Number(button.dataset.checkpointIndex) === state.checkpointIndex)));
    renderHeatmap();
    renderOutcome();
    if (updateHistory) updateUrl();
  }

  function selectTarget(targetKey, requestedCheckpoint = null) {
    state.targetKey = targetKeys.includes(targetKey) ? targetKey : "yanny";
    const rows = checkpoints();
    const requestedIndex = requestedCheckpoint === null ? -1 : rows.findIndex((checkpoint) => String(checkpoint.id) === String(requestedCheckpoint));
    state.checkpointIndex = requestedIndex >= 0 ? requestedIndex : defaultCheckpointIndex(state.targetKey);
    renderTargetHeader();
    renderTimeline();
    renderCheckpointControls();
    renderHeatmap();
    renderOutcome();
    renderControls();
    renderCaveats();
    updateUrl();
  }

  function initialize(data) {
    state.data = assertResults(data);
    const model = data.model || {};
    elements.model.textContent = model.label || model.id || model.model_id || "Whisper Tiny.en";
    elements.baseline.textContent = generatedText(data.baseline.generated);
    const sourceUrl = data.source?.page_url || data.source?.source_page_url;
    if (sourceUrl) elements.source.href = sourceUrl;
    const parameters = new URLSearchParams(window.location.search);
    const requestedTarget = parameters.get("target");
    const requestedCheckpoint = parameters.get("checkpoint");
    selectTarget(requestedTarget, requestedCheckpoint);
    elements.status.hidden = true;
    elements.app.hidden = false;
  }

  function showError(error) {
    elements.status.className = "load-status error";
    elements.status.textContent = `The cached steering replay could not be loaded: ${error.message}`;
    elements.app.hidden = true;
  }

  elements.targetButtons.forEach((button) => button.addEventListener("click", () => selectTarget(button.dataset.target)));
  elements.range.addEventListener("input", () => selectCheckpoint(Number(elements.range.value), true));
  elements.range.addEventListener("keydown", (event) => {
    if (event.key === "Home") { event.preventDefault(); selectCheckpoint(0, true); }
    if (event.key === "End") { event.preventDefault(); selectCheckpoint(checkpoints().length - 1, true); }
  });

  fetch(resultsUrl, { headers: { Accept: "application/json" } })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(initialize)
    .catch(showError);
})();
