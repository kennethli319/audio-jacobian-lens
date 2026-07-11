"use strict";

(() => {
  const vocabularies = {
    asr: { label: "51,864-entry Whisper vocabulary", size: 51864 },
    tts: { label: "6,561 ordinary acoustic codes", size: 6561 },
    speech: { label: "61,690-token LFM vocabulary", size: 61690 },
  };

  const examples = {
    asr: [
      {
        id: "door",
        eyebrow: "EARLY READABILITY · NATURAL SPEECH",
        title: "The final word is already readable at decoder L0.",
        status: "CC BY 4.0 source",
        statusTone: "ready",
        source: "LibriSpeech 1272-135031-0003",
        transcript: "The little girl had been asleep, but she heard the raps and opened the door.",
        generated: "Whisper writes “wraps” instead of “raps,” then correctly emits “door.”",
        purpose: "For the realized piece “door,” the fitted decoder readout ranks the final token fourth at L0 and first from L1 onward. The homophone error remains visible: early readability is not transcription correctness.",
        layers: ["L0", "L1", "L2", "HEAD"],
        tracks: [{ token: "door", ranks: [4, 1, 1, 1], note: "realized token" }],
        boundary: "Decoder ranks use the complete Whisper vocabulary. HEAD is the actual teacher-forced output distribution; L0–L2 are fitted readouts.",
      },
      {
        id: "late",
        eyebrow: "LATE EMERGENCE · NATURAL SPEECH",
        title: "Not every word is explicit in an early residual.",
        status: "CC BY 4.0 source",
        statusTone: "ready",
        source: "LibriSpeech 1272-135031-0012",
        transcript: "Where is my brother now?",
        generated: "Whisper realizes all three tracked pieces on this path.",
        purpose: "“now” is still below six thousand candidates at L0 and becomes rank 3 only at L2. The comparison prevents a simplistic claim that all future words are present from the first decoder layer.",
        layers: ["L0", "L1", "L2", "HEAD"],
        tracks: [
          { token: "Where", ranks: [523, 547, 15, 1] },
          { token: "brother", ranks: [627, 454, 26, 1] },
          { token: "now", ranks: [6319, 7237, 3, 1], note: "selected contrast" },
        ],
        boundary: "Rank can worsen before it improves. These are realized-token ranks on a teacher-forced generated path, not probabilities that the word was consciously considered.",
      },
      {
        id: "subwords",
        eyebrow: "BPE PIECES · NATURAL SPEECH",
        title: "A word can resolve differently across tokenizer pieces.",
        status: "CC BY 4.0 source",
        statusTone: "ready",
        source: "LibriSpeech 1272-141231-0004",
        transcript: "One minute, a voice said, and the time buzzer sounded.",
        generated: "The generated word is split into the pieces “buzz” + “er.”",
        purpose: "The stem stays poorly ranked through L2 while the suffix is already near the top. A lens follows tokenizer positions; it does not assign one hidden state to one human word.",
        layers: ["L0", "L1", "L2", "HEAD"],
        tracks: [
          { token: "buzz", ranks: [3966, 651, 981, 1], note: "stem piece" },
          { token: "er", ranks: [23, 2, 1, 1], note: "suffix piece" },
        ],
        boundary: "The two rows are different decoder positions. Comparing them is useful, but their ranks are not additive word confidence.",
      },
      {
        id: "silence",
        eyebrow: "NULL CONTROL · PROCEDURAL AUDIO",
        title: "Silence can still produce deceptively strong structure.",
        status: "Failure control",
        statusTone: "blocked",
        source: "Project-generated 3 s digital silence",
        transcript: "Reference: no speech.",
        generated: "Whisper hallucinates “you.”",
        purpose: "The hallucinated token becomes rank 1 at the output head, while the experimental encoder readout also looks unexpectedly strong. This is why the encoder lens is not the showcase’s positive claim.",
        layers: ["L0", "L1", "L2", "HEAD"],
        tracks: [
          { token: "you · decoder", ranks: [92, 17, 6, 1] },
          { token: "you · encoder", ranks: [69, 134, 14, 16], note: "negative pilot; no raw head" },
        ],
        boundary: "Whisper’s encoder is bidirectional and its timing here is Whisper-derived. A strong encoder rank on silence is a warning, not evidence of phoneme recognition.",
      },
    ],
    tts: [
      {
        id: "bridge",
        eyebrow: "CANONICAL TTS CASE · FITTED + CAUSAL",
        title: "One acoustic code becomes readable, then can be causally changed.",
        status: "Derived-audio review",
        statusTone: "review",
        source: "Project-authored prompt · Chatterbox T3",
        transcript: "A bright red train crossed the narrow bridge.",
        generated: "Selected position S9 · nominal 0.32–0.36 s · realized acoustic code ID 4106",
        purpose: "The realized code rises from rank 3,183 at L0 to rank 1 at L22. Local text diagnostics highlight “bright” and “red”; a very small residual edit then promotes runner-up ID 4358 and changes the autoregressive suffix.",
        layers: ["L0", "L4", "L8", "L12", "L16", "L20", "L22", "HEAD"],
        tracks: [{
          token: "ID 4106 · S9",
          ranks: [3183, 3137, 1892, 946, 396, 11, 1, 1],
          probabilities: ["0.0016%", "0.0016%", "0.0053%", "0.0183%", "0.0453%", "1.915%", "5.514%", "13.959%"],
          note: "realized acoustic code",
        }],
        trace: [
          { kind: "Gradient sensitivity", layer: "L0", token: "bright", value: "28.5%", text: "largest local gradient share" },
          { kind: "Gradient sensitivity", layer: "L4", token: "bright", value: "25.4%", text: "largest local gradient share" },
          { kind: "Within-text attention", layer: "L4", token: "bright", value: "46.8%", text: "largest attention share" },
          { kind: "Within-text attention", layer: "L8", token: "red", value: "51.0%", text: "largest attention share" },
        ],
        intervention: {
          baseline: { id: "4106", rank: "#1", probability: "13.959%" },
          candidate: { id: "4358", rank: "#2", probability: "12.639%" },
          steered: { id: "4358", rank: "#1", probability: "13.170%" },
          budget: "0.001708984375 relative residual norm at L20 + L22",
          propagation: "44 of 61 same-index codes changed; 43 changes are downstream. The residual branch exactly matched direct forcing in this run.",
        },
        boundary: "ID 4106 and ID 4358 are acoustic-code IDs—not words or phonemes. Gradient and attention are diagnostics; only the replayed residual branch is a causal intervention.",
      },
      {
        id: "turtles",
        eyebrow: "SECONDARY TTS CASE · NEARLY MONOTONIC",
        title: "Some realized codes sharpen almost layer by layer.",
        status: "Diagnostics only",
        statusTone: "review",
        source: "Project-authored prompt · held-out screen",
        transcript: "Tiny turtles travel together toward the tide.",
        generated: "A representative interior acoustic code is tracked through fitted T3 readouts.",
        purpose: "This clean trajectory is useful for learning the display, but it is paired with a non-monotonic example so the interface does not imply that every code must improve at every layer.",
        layers: ["L0", "L4", "L8", "L12", "L16", "L20", "L22"],
        tracks: [{ token: "realized code", ranks: [4149, 3551, 1658, 176, 18, 2, 1] }],
        population: "Across 378 screened interior positions, median rank moved from 1,785.5 at L0 to 12 at L22; only 6.6% were rank 1 at L22.",
        boundary: "The selected row is a strong illustration, not a representative frequency claim.",
      },
      {
        id: "music",
        eyebrow: "SECONDARY TTS CASE · NON-MONOTONIC",
        title: "Readability can become worse before the final layers resolve it.",
        status: "Diagnostics only",
        statusTone: "review",
        source: "Project-authored prompt · held-out screen",
        transcript: "Music fades as the evening grows quiet.",
        generated: "A representative interior acoustic code is tracked through fitted T3 readouts.",
        purpose: "The code drops back at L12 before rapidly sharpening at L20 and L22. Intermediate readouts are measurements, not a guaranteed refinement pipeline.",
        layers: ["L0", "L4", "L8", "L12", "L16", "L20", "L22"],
        tracks: [{ token: "realized code", ranks: [5197, 5262, 3809, 4392, 1443, 6, 1] }],
        population: "Across 378 screened interior positions, median rank moved from 1,785.5 at L0 to 12 at L22; only 6.6% were rank 1 at L22.",
        boundary: "A lower rank number is more readable; the temporary increase at L12 is real and intentionally preserved.",
      },
    ],
    speech: [
      {
        id: "hello",
        eyebrow: "PROVISIONAL SCREEN · MID-LAYER EMERGENCE",
        title: "A lexical direction can appear abruptly in later LFM layers.",
        status: "One-clip lens",
        statusTone: "blocked",
        source: "Spoken: “Say hello in one word.”",
        transcript: "Requested response: one word.",
        generated: "Hello! How can I help you today?",
        purpose: "Several realized pieces become readable by L12, while “Hello” remains rank 5 there. The overlong response also exposes an instruction-following failure rather than presenting fluency alone.",
        layers: ["L0", "L4", "L8", "L12", "L14"],
        tracks: [
          { token: "!", ranks: [21679, 2803, 4, 1, 1] },
          { token: " How", ranks: [6025, 6216, 34, 1, 1] },
          { token: "Hello", ranks: [28728, 38641, 13739, 5, 1], note: "requested word" },
        ],
        boundary: "This artifact was fit on one clip. Treat the run as integration evidence until a multi-clip lens is evaluated on disjoint held-out speech.",
      },
      {
        id: "four",
        eyebrow: "PROVISIONAL SCREEN · FITTED-LENS FAILURE",
        title: "The head can be clear while the fitted lens remains wrong.",
        status: "Negative result",
        statusTone: "blocked",
        source: "Spoken: “What is two plus two? Answer with one word.”",
        transcript: "Requested response: Four.",
        generated: "Four",
        purpose: "The actual head emits “Four” at rank 1 with 66.6%, but the latest fitted source layer still ranks it 141. The failure distinguishes model capability from lens quality.",
        layers: ["L0", "L4", "L8", "L12", "L14", "HEAD"],
        tracks: [{ token: "Four", ranks: [23420, 41719, 31297, 2197, 141, 1], probabilities: [null, null, null, null, null, "66.6%"] }],
        boundary: "A poor fitted rank does not mean the model lacked the answer. Here it means the current projected lens did not recover the final direction early enough.",
      },
    ],
  };

  const destinations = {
    asr: [document.querySelector("#asr-example-buttons"), document.querySelector("#asr-example-detail")],
    tts: [document.querySelector("#tts-example-buttons"), document.querySelector("#tts-example-detail")],
    speech: [document.querySelector("#speech-example-buttons"), document.querySelector("#speech-example-detail")],
  };

  const escapeHtml = (value) => String(value).replace(/[&<>"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
  })[character]);

  function evidenceStrength(rank, denominator) {
    const numeric = Math.max(1, Number(rank) || 1);
    return Math.max(0.04, Math.min(1, 1 - Math.log10(numeric) / Math.log10(denominator)));
  }

  function rankCell(layer, rank, probability, isHead, denominator) {
    const detail = `${layer} · global rank #${Number(rank).toLocaleString()}${probability ? ` · ${probability}` : ""}${isHead ? " · actual output head" : " · fitted readout"}`;
    return `<button class="rank-cell ${isHead ? "head" : "fitted"}" style="--strength:${evidenceStrength(rank, denominator).toFixed(3)}" data-tooltip="${escapeHtml(detail)}" aria-label="${escapeHtml(detail)}"><span>${escapeHtml(layer)}</span><strong>#${Number(rank).toLocaleString()}</strong>${probability ? `<small>${escapeHtml(probability)}</small>` : ""}</button>`;
  }

  function rankTracks(example, family) {
    const vocabulary = vocabularies[family];
    const tracks = example.tracks.map((track) => {
      const cells = track.ranks.map((rank, index) => rankCell(
        example.layers[index], rank, track.probabilities?.[index], example.layers[index] === "HEAD", vocabulary.size,
      )).join("");
      return `<section class="rank-track"><div class="rank-track-meta"><strong>${escapeHtml(track.token)}</strong>${track.note ? `<span>${escapeHtml(track.note)}</span>` : ""}</div><div class="rank-cell-grid" style="--columns:${track.ranks.length}">${cells}</div></section>`;
    }).join("");
    return `<div class="showcase-ranks"><div class="rank-heading"><div><p class="overline">REALIZED-CANDIDATE RANK</p><h3>How the selected output resolves through depth</h3></div><span>Hover or focus any cell · ${escapeHtml(vocabulary.label)}</span></div>${tracks}</div>`;
  }

  function tracePanel(trace) {
    if (!trace) return "";
    return `<section class="trace-panel"><div><p class="overline">LOCAL TEXT TRACE</p><h3>Different diagnostics highlight different context.</h3></div><div class="trace-grid">${trace.map((item) => `<article><span>${escapeHtml(item.kind)} · ${escapeHtml(item.layer)}</span><strong>${escapeHtml(item.token)} <i>${escapeHtml(item.value)}</i></strong><p>${escapeHtml(item.text)}</p></article>`).join("")}</div><p class="panel-boundary">Gradient share is local sensitivity; attention share is within-text routing. Neither number is percent causation.</p></section>`;
  }

  function interventionPanel(intervention) {
    if (!intervention) return "";
    return `<section class="intervention-panel"><div class="intervention-heading"><div><p class="overline">REPLAYED RESIDUAL INTERVENTION</p><h3>A runner-up becomes the emitted code.</h3></div><span>${escapeHtml(intervention.budget)}</span></div><div class="intervention-flow"><article><span>Baseline winner</span><strong>ID ${escapeHtml(intervention.baseline.id)}</strong><p>${escapeHtml(intervention.baseline.rank)} · ${escapeHtml(intervention.baseline.probability)}</p></article><i aria-hidden="true">→</i><article><span>Baseline runner-up</span><strong>ID ${escapeHtml(intervention.candidate.id)}</strong><p>${escapeHtml(intervention.candidate.rank)} · ${escapeHtml(intervention.candidate.probability)}</p></article><i aria-hidden="true">→</i><article class="steered"><span>After L20 + L22 edit</span><strong>ID ${escapeHtml(intervention.steered.id)}</strong><p>${escapeHtml(intervention.steered.rank)} · ${escapeHtml(intervention.steered.probability)}</p></article></div><p class="propagation-note">${escapeHtml(intervention.propagation)}</p></section>`;
  }

  function exampleMarkup(example, family) {
    return `<header class="example-detail-heading"><div><p class="overline">${escapeHtml(example.eyebrow)}</p><h3>${escapeHtml(example.title)}</h3></div><span class="rights-status ${escapeHtml(example.statusTone)}">${escapeHtml(example.status)}</span></header><div class="example-context"><div><span>SOURCE / PROMPT</span><strong>${escapeHtml(example.source)}</strong><p>${escapeHtml(example.transcript)}</p></div><div><span>MODEL PATH</span><strong>${escapeHtml(example.generated)}</strong></div></div><p class="example-purpose">${escapeHtml(example.purpose)}</p>${rankTracks(example, family)}${tracePanel(example.trace)}${interventionPanel(example.intervention)}${example.population ? `<p class="population-note"><strong>Pooled context</strong>${escapeHtml(example.population)}</p>` : ""}<aside class="evidence-boundary"><strong>What this does not prove</strong><p>${escapeHtml(example.boundary)}</p></aside>`;
  }

  function renderFamily(family, selectedId) {
    const [buttons, detail] = destinations[family];
    const selected = examples[family].find((item) => item.id === selectedId) || examples[family][0];
    buttons.innerHTML = examples[family].map((item, index) => `<button type="button" class="showcase-example-button ${item.id === selected.id ? "active" : ""}" data-example-id="${escapeHtml(item.id)}" aria-pressed="${item.id === selected.id}"><span>${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(item.title)}</strong></button>`).join("");
    detail.innerHTML = exampleMarkup(selected, family);
    buttons.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => renderFamily(family, button.dataset.exampleId)));
  }

  Object.keys(destinations).forEach((family) => renderFamily(family));

  const tooltip = document.createElement("div");
  tooltip.className = "showcase-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  document.body.append(tooltip);

  function placeTooltip(target, clientX, clientY) {
    tooltip.textContent = target.dataset.tooltip;
    tooltip.hidden = false;
    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    tooltip.style.left = `${Math.max(10, Math.min(innerWidth - width - 10, clientX + 14))}px`;
    tooltip.style.top = `${Math.max(10, Math.min(innerHeight - height - 10, clientY + 14))}px`;
  }

  document.addEventListener("pointermove", (event) => {
    const target = event.target.closest?.("[data-tooltip]");
    if (target) placeTooltip(target, event.clientX, event.clientY);
    else tooltip.hidden = true;
  });
  document.addEventListener("focusin", (event) => {
    const target = event.target.closest?.("[data-tooltip]");
    if (!target) return;
    const rect = target.getBoundingClientRect();
    placeTooltip(target, rect.left + rect.width / 2, rect.top + rect.height / 2);
  });
  document.addEventListener("focusout", () => { tooltip.hidden = true; });
})();
