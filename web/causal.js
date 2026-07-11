"use strict";

const targetData = {
  yanny: {
    label: "Yanny",
    status: "Using total candidate-token likelihood, Yanny remains below 1% of the restricted four-string set at every budget. Stronger edits change free generation from Lily! to Yay!—not to Yanny.",
    sweep: [
      { budget: "20%", generated: "Lily!", target: "0.38%", targetAbsolute: "0.052113%", lead: "Lily! 95.72%", values: [95.7156297443, 3.9042029784, 0.3772584138, 0.0029088634] },
      { budget: "40%", generated: "Yay!", target: "0.75%", targetAbsolute: "0.040902%", lead: "Lily! 86.90%", values: [86.9049687686, 12.3462937898, 0.7459735602, 0.0027638814] },
      { budget: "80%", generated: "Yay!", target: "0.95%", targetAbsolute: "0.045453%", lead: "Lily! 82.17%", values: [82.1725054060, 16.8739013053, 0.9511097449, 0.0024835438] },
      { budget: "120%", generated: "Yay!", target: "0.90%", targetAbsolute: "0.043770%", lead: "Lily! 82.56%", values: [82.5605315887, 16.5346494933, 0.9024072488, 0.0024116693] },
    ],
    propagation: [
      ["Encoder L1 · edited", "0.359", "1.652"], ["Encoder L2 · edited", "1.393", "3.961"], ["Encoder L3 · edited", "4.349", "11.008"],
      ["Decoder L0", "0.254", "0.443"], ["Decoder L1", "0.337", "0.456"], ["Decoder L2", "0.497", "0.722"], ["Decoder L3", "1.254", "2.809"], ["Output head", "63.920", "105.821"],
    ],
  },
  laurel: {
    label: "Laurel",
    status: "Laurel reaches 99% of the restricted total-path comparison only after the other three strings collapse. Its absolute token-path likelihood stays below 0.001%, and free generation remains repetitive or unrelated—not Laurel.",
    sweep: [
      { budget: "20%", generated: "L-E-L-E…", target: "0.60%", targetAbsolute: "0.000564%", lead: "Yay! 51.06%", values: [47.9564843007, 51.0578410094, 0.3864616473, 0.5992130425] },
      { budget: "40%", generated: "…", target: "9.11%", targetAbsolute: "0.000182%", lead: "Yay! 90.61%", values: [0.0631148948, 90.6053965314, 0.2232619318, 9.1082266420] },
      { budget: "80%", generated: "The", target: "99.30%", targetAbsolute: "0.000027%", lead: "Laurel 99.30%", values: [0.0565558967, 0.0206759359, 0.6207521163, 99.3020160511] },
      { budget: "120%", generated: "the", target: "99.98%", targetAbsolute: "0.000524%", lead: "Laurel 99.98%", values: [0.0190652256, 0.0000183890, 0.0018725119, 99.9790438735] },
    ],
    propagation: [
      ["Encoder L1 · edited", "0.359", "1.652"], ["Encoder L2 · edited", "2.366", "5.099"], ["Encoder L3 · edited", "10.385", "18.995"],
      ["Decoder L0", "1.531", "4.286"], ["Decoder L1", "2.305", "4.262"], ["Decoder L2", "3.260", "4.267"], ["Decoder L3", "5.980", "7.458"], ["Output head", "907.169", "1679.613"],
    ],
  },
};

const baselineCandidates = [
  { label: "Lily!", value: 73.6753523350, logp: -0.8661649227 },
  { label: "Yay!", value: 23.1519073248, logp: -2.0237560272 },
  { label: "Yanny", value: 3.1708255410, logp: -4.0118412971 },
  { label: "Laurel", value: 0.0019092069, logp: -11.4269008636 },
];

const candidateTokenMetadata = {
  "Lily!": { tokenIds: [20037, 0], tokenCount: 2 },
  "Yay!": { tokenIds: [575, 323, 0], tokenCount: 3 },
  Yanny: { tokenIds: [575, 7737], tokenCount: 2 },
  Laurel: { tokenIds: [43442], tokenCount: 1 },
};

function totalCandidateMetrics(candidates) {
  const totals = candidates.map((candidate) => {
    const metadata = candidateTokenMetadata[candidate.label];
    return candidate.logp * metadata.tokenCount;
  });
  const maximum = Math.max(...totals);
  const denominator = totals.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  return candidates.map((candidate, index) => {
    const metadata = candidateTokenMetadata[candidate.label];
    return {
      ...candidate,
      ...metadata,
      meanLogP: candidate.logp,
      restrictedMeanShare: candidate.value,
      totalLogP: totals[index],
      sequencePercent: 100 * Math.exp(totals[index]),
      value: 100 * Math.exp(totals[index] - maximum) / denominator,
    };
  });
}

const pieceStudyData = {
  yanny: {
    label: "Yanny",
    targetCandidate: "Yanny",
    defaultCondition: "both",
    pieces: [
      { key: "y", display: "\" Y\"", tokenId: 575, encoder: "0.00–0.46 s", decoder: "candidate prediction 1", familyPrefix: "Y", familyCount: 114, familyExamples: "Y · You · York · Yes · Your" },
      { key: "anny", display: "\"anny\"", tokenId: 7737, encoder: "0.46–0.92 s", decoder: "candidate prediction 2", familyPrefix: "anny", familyCount: 1, familyExamples: "anny" },
    ],
    conditions: [
      { key: "baseline", label: "Baseline" },
      { key: "y", label: "\" Y\" only", familyLabel: "Y* family only" },
      { key: "anny", label: "\"anny\" only", familyLabel: "anny* family only" },
      { key: "both", label: "Both pieces", familyLabel: "Both prefix families" },
    ],
    baseline: {
      label: "Baseline",
      generated: "Lily!",
      activePieces: [],
      probabilities: [12.3330518603, 0.2656368778],
      joint: 0.0327611339,
      candidates: baselineCandidates,
      propagation: null,
      random: null,
      note: "No residual edit. Whisper Tiny generates Lily!, and Yanny holds 0.182% of the restricted total candidate-token likelihood comparison.",
    },
    streams: {
      encoder: {
        y: {
          label: "Encoder · \" Y\" only", generated: "Yay!", activePieces: ["y"],
          probabilities: [26.8659621477, 0.8785165736], joint: 0.2360219301,
          candidates: [
            { label: "Lily!", value: 21.8139082193, logp: -2.2439305782 }, { label: "Yay!", value: 68.1907415390, logp: -1.1041695277 },
            { label: "Yanny", value: 9.9939301610, logp: -3.0245003700 }, { label: "Laurel", value: 0.0014212311, logp: -11.8827104568 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 0.8476743828], ["Enc L2", 3.1581677095], ["Enc L3", 9.8648678531], ["Dec L3", 3.5897256434]], head: 318.4231338501 },
          random: { generated: "Lily!", targetShare: 0.2505565400 },
          note: "Encoder steering over the first audio half raises actual p(\" Y\") to 26.87% and Yanny's restricted total-path share to 4.72%. Generation becomes Yay!, not Yanny.",
        },
        anny: {
          label: "Encoder · \"anny\" only", generated: "Lily!", activePieces: ["anny"],
          probabilities: [11.7194503546, 0.2478731785], joint: 0.0290493741,
          candidates: [
            { label: "Lily!", value: 72.8406071663, logp: -1.0163532495 }, { label: "Yay!", value: 23.7269461155, logp: -2.1380155881 },
            { label: "Yanny", value: 3.4303504974, logp: -4.0719642639 }, { label: "Laurel", value: 0.0020910360, logp: -11.4747228622 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 0.8013291955], ["Enc L2", 2.5608946700], ["Enc L3", 7.4653721892], ["Dec L3", 1.5106213242]], head: 130.5547790527 },
          random: { generated: "Lily!", targetShare: 0.1318769150 },
          note: "Encoder steering over the second audio half has little effect here: Yanny's restricted total-path share moves from 0.182% to 0.219%, and generation remains Lily!.",
        },
        both: {
          label: "Encoder · both pieces", generated: "Yay!", activePieces: ["y", "anny"],
          probabilities: [11.5704044700, 0.8067724384], joint: 0.0933468343,
          candidates: [
            { label: "Lily!", value: 31.4656734467, logp: -2.0267386436 }, { label: "Yay!", value: 61.2362921238, logp: -1.3608957926 },
            { label: "Yanny", value: 7.2960555553, logp: -3.4883017540 }, { label: "Laurel", value: 0.0019821957, logp: -11.6991863251 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 1.1660217332], ["Enc L2", 3.5414434309], ["Enc L3", 11.3948713178], ["Dec L3", 3.6796724200]], head: 348.0869483948 },
          random: { generated: "Lily!", targetShare: 0.2339638240 },
          note: "The tokenizer-faithful encoder default splits the 20% budget across both audio halves. Yanny reaches 2.65% of the restricted total-path set and generation becomes Yay!, not Yanny.",
        },
      },
      encoder_prefix: {
        y: {
          label: "Encoder prefix family · Y* only", generated: "Yay!", activePieces: ["y"],
          probabilities: [52.9032507289, 0.1296286021], joint: 0.0685777444,
          candidates: [
            { label: "Lily!", value: 19.8683246970, logp: -2.4902770519 }, { label: "Yay!", value: 73.8535225391, logp: -1.1773200830 },
            { label: "Yanny", value: 6.2772080302, logp: -3.6424787045 }, { label: "Laurel", value: 0.0009402386, logp: -12.4487810135 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 0.8476745434], ["Enc L2", 3.3212051625], ["Enc L3", 9.2748222299], ["Dec L3", 3.1528040171]], head: 338.6899261475 },
          random: { generated: "Lily!", targetShare: 0.2505565110 },
          note: "Averaging 114 Y* tokens versus 131 La* tokens strongly raises p(\" Y\") to 52.90%, but conditional p(\"anny\") falls. Yanny's restricted total-path share reaches 1.86% and generation becomes Yay!, not Yanny.",
        },
        anny: {
          label: "Encoder prefix family · anny* only", generated: "Lily!", activePieces: ["anny"],
          probabilities: [10.7945010177, 0.2834895189], joint: 0.0306012790,
          candidates: [
            { label: "Lily!", value: 75.9366095066, logp: -0.8826580644 }, { label: "Yay!", value: 20.8495453000, logp: -2.1752248406 },
            { label: "Yanny", value: 3.2111048698, logp: -4.0459418297 }, { label: "Laurel", value: 0.0027366077, logp: -11.1135931015 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 0.8013291722], ["Enc L2", 2.5388805827], ["Enc L3", 6.5915105110], ["Dec L3", 1.3745074272]], head: 146.2621250153 },
          random: { generated: "Lily!", targetShare: 0.1318769070 },
          note: "The decoded anny* family contains only token 7737, so this positive side is not broader than exact-token steering. With a La* negative family, Yanny's restricted total-path share is 0.177% and generation remains Lily!.",
        },
        both: {
          label: "Encoder prefix families · Y* then anny*", generated: "Yay!", activePieces: ["y", "anny"],
          probabilities: [35.2716334102, 0.2605932967], joint: 0.0919155123,
          candidates: [
            { label: "Lily!", value: 25.6450206041, logp: -2.2111276984 }, { label: "Yay!", value: 67.2578930855, logp: -1.2469428678 },
            { label: "Yanny", value: 7.0954196155, logp: -3.4960278273 }, { label: "Laurel", value: 0.0016685004, logp: -11.8513069153 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 1.1660217591], ["Enc L2", 3.5613426644], ["Enc L3", 10.0297747073], ["Dec L3", 3.0699167550]], head: 331.1395111084 },
          random: { generated: "Lily!", targetShare: 0.2339637130 },
          note: "The prefix-family default uses Y* (114 tokens) on the first half and anny* (1 token) on the second, contrasted with La* (131 tokens). Yanny's restricted total-path share reaches 2.51% and generation becomes Yay!.",
        },
      },
      decoder: {
        y: {
          label: "Decoder · \" Y\" only", generated: "Lily!", activePieces: ["y"],
          probabilities: [14.5128949119, 0.2791324855], joint: 0.0405102043,
          candidates: [
            { label: "Lily!", value: 67.3024713993, logp: -1.1587542892 }, { label: "Yay!", value: 28.3798038960, logp: -2.0222736597 },
            { label: "Yanny", value: 4.3157253414, logp: -3.9056857824 }, { label: "Laurel", value: 0.0019938750, logp: -11.5856266022 },
          ],
          propagation: { label: "Mean ΔL2 on positive path", nodes: [["Dec L0", 0.1339534024], ["Dec L1", 0.3468060618], ["Dec L2", 0.6395840446], ["Dec L3", 1.1691059669]], head: 61.9731521606 },
          random: { generated: "Lily!", targetShare: 0.3269293170 },
          note: "Decoder L0→L2 steering at the first prediction position raises p(\" Y\") to 14.51%, but Yanny's restricted total-path share reaches only 0.400%. Generation remains Lily!.",
        },
        anny: {
          label: "Decoder · \"anny\" only", generated: "Lily!", activePieces: ["anny"],
          probabilities: [12.3330516485, 5.2070470290], joint: 0.6421877994,
          candidates: [
            { label: "Lily!", value: 66.8661534786, logp: -0.7975385785 }, { label: "Yay!", value: 21.2361410260, logp: -1.9445270499 },
            { label: "Yanny", value: 11.8960820138, logp: -2.5240223408 }, { label: "Laurel", value: 0.0016178305, logp: -11.4269008636 },
          ],
          propagation: { label: "Mean ΔL2 on positive path", nodes: [["Dec L0", 0.1194967330], ["Dec L1", 0.3217051824], ["Dec L2", 0.5964187781], ["Dec L3", 0.7793002923]], head: 28.2811101278 },
          random: { generated: "Lily!", targetShare: 0.1876013410 },
          note: "Decoder steering at the second prediction position raises conditional p(\"anny\" | \" Y\") from 0.266% to 5.207%. Yanny's restricted total-path share reaches 3.03%, but free generation remains Lily!.",
        },
        both: {
          label: "Decoder · both pieces", generated: "Lily!", activePieces: ["y", "anny"],
          probabilities: [14.5427485543, 2.4305144538], joint: 0.3534636056,
          candidates: [
            { label: "Lily!", value: 64.9073958397, logp: -0.9767974317 }, { label: "Yay!", value: 24.8419523239, logp: -1.9372252027 },
            { label: "Yanny", value: 10.2490656078, logp: -2.8225724697 }, { label: "Laurel", value: 0.0015777730, logp: -11.6014995575 },
          ],
          propagation: { label: "Mean ΔL2 on positive path", nodes: [["Dec L0", 0.1792162955], ["Dec L1", 0.4542989333], ["Dec L2", 0.8334958553], ["Dec L3", 1.2479429245]], head: 36.8446667989 },
          random: { generated: "Lily!", targetShare: 0.2847025450 },
          note: "The decoder default edits prediction positions 1 and 2 across L0→L2. Yanny reaches 2.38% of the restricted total-path set, while free generation remains Lily!.",
        },
      },
    },
  },
  laurel: {
    label: "Laurel",
    targetCandidate: "Laurel",
    defaultCondition: "laurel",
    pieces: [
      { key: "laurel", display: "\" Laurel\"", tokenId: 43442, encoder: "0.00–0.92 s", decoder: "candidate prediction 1", familyPrefix: "La", familyCount: 131, familyExamples: "Lab · Law · Last · Lake · Land · Laura" },
    ],
    conditions: [
      { key: "baseline", label: "Baseline" },
      { key: "laurel", label: "\" Laurel\" · only actual piece", familyLabel: "La* vocabulary family" },
    ],
    baseline: {
      label: "Baseline", generated: "Lily!", activePieces: [],
      probabilities: [0.0010898331], joint: 0.0010898331, candidates: baselineCandidates,
      propagation: null, random: null,
      note: "Whisper tokenizes \" Laurel\" as one BPE token (43442), so its actual token probability and complete token-path likelihood are the same; it holds 0.00607% of the restricted total-path set at baseline.",
    },
    streams: {
      encoder: {
        laurel: {
          label: "Encoder · \" Laurel\"", generated: "Yeah, I think it's a good idea.", activePieces: ["laurel"],
          probabilities: [0.0003671631], joint: 0.0003671631,
          candidates: [
            { label: "Lily!", value: 2.1736992523, logp: -7.1281023026 }, { label: "Yay!", value: 97.1907436848, logp: -3.3278573354 },
            { label: "Yanny", value: 0.6256091874, logp: -8.3735618591 }, { label: "Laurel", value: 0.0099484154, logp: -12.5148744583 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 1.6490037441], ["Enc L2", 6.1054234712], ["Enc L3", 19.2047438207], ["Dec L3", 7.9897027016]], head: 1690.1980361938 },
          random: { generated: "Lily!", targetShare: 0.0115915730 },
          note: "Encoder steering raises Laurel's restricted total-path share from 0.00607% to 7.27%, but actual p(\" Laurel\") falls to 0.000367%. The larger share comes from greater damage to the other fixed candidates, and generation becomes unrelated text—not Laurel.",
        },
      },
      encoder_prefix: {
        laurel: {
          label: "Encoder prefix family · La*", generated: "Let's see.", activePieces: ["laurel"],
          probabilities: [0.0001989302], joint: 0.0001989302,
          candidates: [
            { label: "Lily!", value: 7.8642324437, logp: -7.1876611709 }, { label: "Yay!", value: 91.2862181664, logp: -4.7359862328 },
            { label: "Yanny", value: 0.8288595239, logp: -9.4376912117 }, { label: "Laurel", value: 0.0206975295, logp: -13.1277265549 },
          ],
          propagation: { label: "Recorded mean ΔL2", nodes: [["Enc L1", 1.6490037674], ["Enc L2", 6.1473196745], ["Enc L3", 19.1659655571], ["Dec L3", 8.4383240938]], head: 1467.9702148438 },
          random: { generated: "Lily!", targetShare: 0.0115915720 },
          note: "Averaging 131 La* tokens versus 114 Y* tokens gives Laurel 61.35% of the restricted total-path comparison, but its absolute probability falls from 0.001090% to 0.000199%. The share rises because all four paths are damaged; free generation becomes unrelated text—not Laurel.",
        },
      },
      decoder: {
        laurel: {
          label: "Decoder · \" Laurel\"", generated: "Lily!", activePieces: ["laurel"],
          probabilities: [0.0028506654], joint: 0.0028506654,
          candidates: [
            { label: "Lily!", value: 76.3826906681, logp: -0.7622250915 }, { label: "Yay!", value: 21.1726665497, logp: -2.0452702244 },
            { label: "Yanny", value: 2.4399826303, logp: -4.2059904337 }, { label: "Laurel", value: 0.0046662859, logp: -10.4653730392 },
          ],
          propagation: { label: "Mean ΔL2 on positive path", nodes: [["Dec L0", 0.2009301037], ["Dec L1", 0.4818045795], ["Dec L2", 0.8253049254], ["Dec L3", 1.2342591286]], head: 177.7509765625 },
          random: { generated: "Lily!", targetShare: 0.0076855270 },
          note: "Decoder L0→L2 steering raises actual p(\" Laurel\") from 0.001090% to 0.002851% and its restricted total-path share to 0.01295%. The effect is measurable but tiny; generation remains Lily!.",
        },
      },
    },
  },
};

const targetElements = {
  yanny: document.querySelector("#target-yanny"),
  laurel: document.querySelector("#target-laurel"),
  status: document.querySelector("#target-steering-status"),
  sweep: document.querySelector("#target-sweep-grid"),
  candidates: document.querySelector("#target-candidate-table"),
  propagation: document.querySelector("#target-propagation-table"),
  contrast: document.querySelector("#target-contrast"),
};

const pieceElements = {
  overline: document.querySelector("#piece-impact-overline"),
  title: document.querySelector("#piecewise-title"),
  copy: document.querySelector("#piece-impact-copy"),
  conditionButtons: document.querySelector("#piece-condition-buttons"),
  streamButtons: [...document.querySelectorAll("[data-piece-stream]")],
  map: document.querySelector("#piece-impact-map"),
  status: document.querySelector("#piece-impact-status"),
  tokenMetrics: document.querySelector("#piece-token-metrics"),
  pathBadge: document.querySelector("#piece-path-badge"),
  propagation: document.querySelector("#piece-propagation"),
  generated: document.querySelector("#piece-generated"),
  candidateBars: document.querySelector("#piece-candidate-bars"),
  random: document.querySelector("#piece-random-control"),
  rawTable: document.querySelector("#piece-raw-table"),
  methodNote: document.querySelector("#piece-method-note"),
};

const pieceState = { target: "yanny", stream: "encoder", condition: "both" };

function candidateTable(data) {
  const targetIndex = data.label === "Laurel" ? 3 : 2;
  const header = "<div class=\"target-candidate-row target-candidate-header\"><span>Budget</span><span>Lily!</span><span>Yay!</span><span>Yanny</span><span>Laurel</span></div>";
  const rows = data.sweep.map((item) => `<div class="target-candidate-row"><strong>${item.budget}</strong>${item.values.map((value, index) => `<span class="${index === targetIndex ? "target-value" : ""}">${value.toFixed(2)}%</span>`).join("")}</div>`).join("");
  return header + rows;
}

function propagationTable(data) {
  const header = "<div class=\"target-propagation-row target-propagation-header\"><span>State</span><span>Mean ΔL2</span><span>Peak ΔL2</span></div>";
  const rows = data.propagation.map((row, index) => `<div class="target-propagation-row ${index < 3 ? "encoder" : ""} ${index === data.propagation.length - 1 ? "head" : ""}"><span>${row[0]}</span><strong>${row[1]}</strong><strong>${row[2]}</strong></div>`).join("");
  return header + rows;
}

function renderTarget(target) {
  const data = targetData[target];
  targetElements.status.textContent = data.status;
  targetElements.contrast.textContent = `${data.label} − ${target === "yanny" ? "Laurel" : "Yanny"}`;
  targetElements.sweep.replaceChildren(...data.sweep.map((item) => {
    const card = document.createElement("article");
    card.className = "target-sweep-card";
    card.innerHTML = `<span>Total budget</span><strong>${item.budget}</strong><p>Free generation <b>${item.generated}</b></p><small>${data.label} · restricted total path: <em>${item.target}</em></small><small>Absolute target path: ${item.targetAbsolute}</small><small>Leading restricted candidate: ${item.lead}</small>`;
    return card;
  }));
  targetElements.candidates.innerHTML = candidateTable(data);
  targetElements.propagation.innerHTML = propagationTable(data);
  ["yanny", "laurel"].forEach((name) => {
    const selected = name === target;
    targetElements[name].classList.toggle("active", selected);
    targetElements[name].setAttribute("aria-pressed", String(selected));
  });
  pieceState.target = target;
  pieceState.condition = pieceStudyData[target].defaultCondition;
  renderPieceStudy();
}

function percent(value, digits = 2) {
  return `${value.toFixed(digits)}%`;
}

function sequencePercent(value) {
  if (value < 0.000001) return `${value.toExponential(2)}%`;
  if (value < 0.001) return `${value.toFixed(6)}%`;
  if (value < 1) return `${value.toFixed(4)}%`;
  return `${value.toFixed(2)}%`;
}

function probabilityDigits(value) {
  if (value < 0.01) return 6;
  if (value < 1) return 3;
  return 2;
}

function percentagePointDelta(value, baseline, digits = 2) {
  const delta = value - baseline;
  if (Math.abs(delta) < 10 ** (-digits)) return "no visible change";
  return `${delta > 0 ? "+" : ""}${delta.toFixed(digits)} pp`;
}

function selectedPieceData() {
  const target = pieceStudyData[pieceState.target];
  return pieceState.condition === "baseline"
    ? target.baseline
    : target.streams[pieceState.stream][pieceState.condition];
}

function renderPieceConditions(target) {
  pieceElements.conditionButtons.replaceChildren(...target.conditions.map((condition) => {
    const button = document.createElement("button");
    const selected = condition.key === pieceState.condition;
    button.className = `button secondary${selected ? " active" : ""}`;
    button.type = "button";
    button.dataset.pieceCondition = condition.key;
    button.setAttribute("aria-pressed", String(selected));
    button.textContent = pieceState.stream === "encoder_prefix" && condition.familyLabel
      ? condition.familyLabel
      : condition.label;
    button.addEventListener("click", () => {
      pieceState.condition = condition.key;
      renderPieceStudy();
      pieceElements.conditionButtons
        .querySelector(`[data-piece-condition="${condition.key}"]`)
        ?.focus();
    });
    return button;
  }));
}

function renderPieceMap(target, data) {
  const selectedLane = pieceState.condition === "baseline"
    ? null
    : (pieceState.stream === "encoder_prefix" ? "encoder" : pieceState.stream);
  const lanes = [
    { key: "encoder", label: "Encoder · audio-time slices", field: "encoder" },
    { key: "decoder", label: "Decoder · autoregressive positions", field: "decoder" },
  ];
  pieceElements.map.innerHTML = lanes.map((lane) => `<div class="piece-map-lane ${lane.key === selectedLane ? "selected" : ""}"><strong>${lane.label}</strong><div class="piece-impact-track piece-count-${target.pieces.length}">${target.pieces.map((piece) => {
    const prefixFamily = pieceState.stream === "encoder_prefix" && lane.key === "encoder";
    const active = lane.key === selectedLane && data.activePieces.includes(piece.key);
    const inactive = lane.key === selectedLane && data.activePieces.length > 0 && !active;
    const tokenLabel = prefixFamily ? `${piece.familyPrefix}*` : piece.display;
    const detail = prefixFamily
      ? `${piece.familyCount} vocabulary token${piece.familyCount === 1 ? "" : "s"} · ${piece[lane.field]} · e.g. ${piece.familyExamples}`
      : `token ${piece.tokenId} · ${piece[lane.field]}`;
    return `<div class="piece-impact-slice ${piece.key} ${active ? "active" : ""} ${inactive ? "inactive" : ""}"><code>${tokenLabel}</code><span>${detail}</span></div>`;
  }).join("")}</div></div>`).join("");
}

function renderPieceTokenPath(target, data) {
  const metrics = target.pieces.map((piece, index) => ({
    label: target.pieces.length === 1 ? "Only tokenizer piece" : `Piece ${index + 1}`,
    token: piece.display,
    context: index === 0 ? `p(${piece.display} | audio)` : `p(${piece.display} | earlier target pieces, audio)`,
    value: data.probabilities[index],
    baseline: target.baseline.probabilities[index],
  }));
  if (target.pieces.length > 1) {
    metrics.push({
      label: "Complete path",
      token: target.pieces.map((piece) => piece.display).join(" → "),
      context: "joint teacher-forced path",
      value: data.joint,
      baseline: target.baseline.joint,
    });
  }
  pieceElements.tokenMetrics.className = `piece-token-metrics metrics-${metrics.length}`;
  pieceElements.tokenMetrics.replaceChildren(...metrics.map((metric) => {
    const item = document.createElement("article");
    const digits = probabilityDigits(metric.value);
    item.className = "piece-token-metric";
    item.innerHTML = `<span>${metric.label}</span><code>${metric.token}</code><strong>${percent(metric.value, digits)}</strong><small>${pieceState.condition === "baseline" ? "baseline" : percentagePointDelta(metric.value, metric.baseline, digits)}</small><p>${metric.context}</p>`;
    return item;
  }));
}

function renderPieceCandidates(target, data) {
  const candidates = totalCandidateMetrics(data.candidates);
  const baselineCandidates = totalCandidateMetrics(target.baseline.candidates);
  pieceElements.candidateBars.replaceChildren(...candidates.map((candidate, index) => {
    const baseline = baselineCandidates[index];
    const row = document.createElement("div");
    const digits = candidate.value < 0.01 ? 6 : 2;
    const baselineDigits = baseline.value < 0.01 ? 6 : (baseline.value < 0.1 ? 4 : 2);
    row.className = `piece-candidate-row${candidate.label === target.targetCandidate ? " target" : ""}`;
    const baselineText = percent(baseline.value, baselineDigits);
    const deltaText = percentagePointDelta(candidate.value, baseline.value, digits);
    row.setAttribute("aria-label", `${candidate.label}, ${candidate.tokenCount} tokens: restricted total-path share ${percent(candidate.value, digits)}; absolute candidate-token likelihood ${sequencePercent(candidate.sequencePercent)}; baseline share ${baselineText}; ${deltaText}`);
    row.innerHTML = `<div class="piece-candidate-label"><strong>${candidate.label} <i>${candidate.tokenCount} ${candidate.tokenCount === 1 ? "token" : "tokens"}</i></strong><span>${percent(candidate.value, digits)}</span><small>path p ${sequencePercent(candidate.sequencePercent)}</small></div><div class="piece-candidate-track"><i class="piece-candidate-fill"></i><b class="piece-baseline-marker" title="Baseline restricted share ${baselineText}"></b></div><small>${pieceState.condition === "baseline" ? "baseline" : `<b>${deltaText}</b><span>base ${baselineText}</span>`}</small>`;
    row.querySelector(".piece-candidate-fill").style.width = `${Math.min(100, candidate.value)}%`;
    row.querySelector(".piece-baseline-marker").style.left = `${Math.min(100, baseline.value)}%`;
    return row;
  }));
}

function renderPiecePropagation(data) {
  if (!data.propagation) {
    pieceElements.propagation.innerHTML = "<span>Downstream state change</span><strong>Baseline · no residual edit</strong>";
    pieceElements.propagation.classList.add("baseline");
    return;
  }
  pieceElements.propagation.classList.remove("baseline");
  pieceElements.propagation.innerHTML = `<span>${data.propagation.label}</span><div class="piece-propagation-flow">${data.propagation.nodes.map(([label, value]) => `<div><small>${label}</small><strong>${value.toFixed(3)}</strong></div>`).join("")}<div class="output"><small>Output-logit ΔL2</small><strong>${data.propagation.head.toFixed(3)}</strong></div></div>`;
}

function renderPieceRawTable(target, data) {
  const candidates = totalCandidateMetrics(data.candidates);
  const baseline = totalCandidateMetrics(target.baseline.candidates);
  pieceElements.rawTable.setAttribute("role", "table");
  pieceElements.rawTable.setAttribute("aria-label", "Candidate total token-path log probabilities before and after intervention");
  pieceElements.rawTable.innerHTML = `<div class="piece-raw-row header" role="row"><span role="columnheader">Candidate</span><span role="columnheader">Baseline total log p</span><span role="columnheader">${data.label} total log p</span><span role="columnheader">Change</span></div>${candidates.map((candidate, index) => {
    const original = baseline[index];
    const delta = candidate.totalLogP - original.totalLogP;
    return `<div class="piece-raw-row" role="row"><strong role="rowheader">${candidate.label}</strong><span role="cell">${original.totalLogP.toFixed(3)}</span><span role="cell">${candidate.totalLogP.toFixed(3)}</span><span role="cell">${delta > 0 ? "+" : ""}${delta.toFixed(3)}</span></div>`;
  }).join("")}`;
}

function renderPieceStudy() {
  const target = pieceStudyData[pieceState.target];
  const data = selectedPieceData();
  pieceElements.overline.textContent = pieceState.condition === "baseline"
    ? "Shared no-edit baseline"
    : (pieceState.stream === "encoder_prefix" ? "Vocabulary-prefix family impact" : "Tokenizer-faithful piece impact");
  pieceElements.title.textContent = `How do encoder and decoder edits change ${target.label}'s real tokenizer path?`;
  if (pieceState.condition === "baseline") {
    pieceElements.copy.textContent = `Baseline applies no residual edit. It supplies the common reference for every exact-token, prefix-family, and decoder condition below.`;
  } else if (pieceState.stream === "encoder_prefix") {
    const families = target.pieces
      .filter((piece) => data.activePieces.includes(piece.key))
      .map((piece) => `${piece.familyPrefix}* (${piece.familyCount})`)
      .join(" then ");
    pieceElements.copy.textContent = `This exploratory encoder mode replaces each exact positive token with the mean J-lens direction over every ordinary vocabulary token in the matching decoded-token family: ${families}. The actual decoder probabilities below still measure ${target.label}'s exact tokenizer path.`;
  } else {
    pieceElements.copy.textContent = target.pieces.length === 1
      ? `Whisper represents \" ${target.label}\" as one BPE token, so the tokenizer-faithful view uses one complete piece. Encoder and decoder schedules each use a 20% within-stream relative budget; those budgets are not equal causal doses across streams.`
      : "Whisper represents Yanny as two BPE pieces. Encoder edits map them to audio slices; decoder edits map them to the positions that predict each piece. Both schedules use a 20% within-stream relative budget, not a cross-stream matched dose.";
  }
  renderPieceConditions(target);
  pieceElements.streamButtons.forEach((button) => {
    const selected = button.dataset.pieceStream === pieceState.stream;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderPieceMap(target, data);
  pieceElements.status.textContent = data.note;
  const streamLabel = pieceState.stream === "encoder_prefix" ? "encoder prefix family" : pieceState.stream;
  pieceElements.pathBadge.textContent = pieceState.condition === "baseline"
    ? "baseline · no edit"
    : `${streamLabel} · teacher forced`;
  renderPieceTokenPath(target, data);
  renderPieceCandidates(target, data);
  renderPiecePropagation(data);
  renderPieceRawTable(target, data);
  pieceElements.generated.innerHTML = pieceState.condition === "baseline"
    ? "<span>Baseline generation <strong>Lily!</strong></span>"
    : `<span>Baseline <strong>Lily!</strong></span><i aria-hidden="true">→</i><span>${streamLabel} <strong>${data.generated}</strong></span>`;
  const targetCandidate = totalCandidateMetrics(data.candidates).find((candidate) => candidate.label === target.targetCandidate);
  pieceElements.random.textContent = data.random
    ? `One matched random seed generated ${data.random.generated} and gave ${target.label} ${percent(data.random.targetShare, data.random.targetShare < 0.01 ? 6 : (data.random.targetShare < 0.1 ? 4 : 2))} of the restricted total-path set, versus ${percent(targetCandidate.value, targetCandidate.value < 0.01 ? 6 : 2)} for this J-lens edit. One seed is not a control distribution.`
    : "No intervention or random-direction control is applied in the baseline view.";
  if (pieceState.condition === "baseline") {
    pieceElements.methodNote.textContent = "No residual direction, layer, audio slice, or decoder position is edited in this view. Change the allocation to inspect an intervention against this shared baseline.";
  } else if (pieceState.stream === "decoder") {
    pieceElements.methodNote.textContent = "Decoder source edits use L0→L2; L3 is the downstream J-lens target. Free generation is open-loop by absolute prediction position, so a later-piece edit still runs if an earlier generated piece differs.";
  } else if (pieceState.stream === "encoder_prefix") {
    const negative = pieceState.target === "yanny" ? "La* (131 tokens)" : "Y* (114 tokens)";
    pieceElements.methodNote.textContent = `Prefix matching is case-sensitive after stripping leading whitespace only; it groups decoded vocabulary tokens, not phonemes. Each family direction is an equal-weight mean and is contrasted with ${negative}, so family size and composition affect the edit. Encoder ranges remain locations in a bidirectional encoder, not real-time belief states.`;
  } else {
    pieceElements.methodNote.textContent = "Encoder ranges are audio locations in a bidirectional encoder, not real-time belief states. The complete decoder and output head are rerun after every edit.";
  }
}

targetElements.yanny.addEventListener("click", () => renderTarget("yanny"));
targetElements.laurel.addEventListener("click", () => renderTarget("laurel"));
pieceElements.streamButtons.forEach((button) => {
  button.addEventListener("click", () => {
    pieceState.stream = button.dataset.pieceStream;
    renderPieceStudy();
  });
});
renderTarget("yanny");
