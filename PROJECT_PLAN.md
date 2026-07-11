# Audio Jacobian Lens: Project Plan

Last updated: 2026-07-11

This is the continuity document for the project. Read it before making changes,
update it when a milestone or design decision changes, and append a short entry
to the work log before ending a work session.

## Goal

Build a reproducible, local-first Jacobian-lens explorer for speech models. The
first complete path targets Whisper Tiny; later paths extend the same honest UI
contract to local MLX speech-to-speech and text-to-speech models. A user should
be able to upload, record, or choose audio and inspect:

1. the model's actual generated text and output-distribution diagnostics;
2. token readouts from intermediate language-model or decoder residual states;
3. audio-time readouts from encoder states when that stream has a separately
   fitted and validated cross-stream lens; and
4. generated speech as model output, clearly separated from any future lens
   over acoustic-codebook or waveform-generation states; and
5. for Chatterbox, inspect a corpus-fitted same-stream T3 speech-code readout
   across layers and output positions, then select a generated code slice and
   trace that run's separate local sensitivity back to input-text positions.

The interface should feel similar to a language-model J-lens explorer, but it
must preserve each architecture's distinct streams and must not present lens
scores as model confidence or as a complete account of "what the model is
thinking."

## Scientific contract

- **Raw output diagnostics** means statistics computed from the base model's
  teacher-forced output distribution at a generation step: probability,
  log-probability, entropy, margin, and alternatives. These values are not
  calibrated and precede generation-time token processors. For MLX LFM they
  apply only to ordinary text positions on the model's interleaved path; for
  Chatterbox they apply to the T3 speech-code head rather than a text head.
- **J-lens readout score** means a ranked token readout after a corpus-averaged
  Jacobian transport. It is not a calibrated probability of emission.
- The decoder lens is the closest replication of the paper's causal-language
  setup.
- The encoder-to-decoder lens is a new cross-modal extension. It needs its own
  validation and must not be described as a result established by the paper.
- Whisper's encoder is bidirectional. An encoder cell aligned to 1.0 seconds can
  contain information from later audio, so it is an audio-location view, not a
  real-time belief trace.
- Any claim that a lens token is functionally used should be backed by a causal
  intervention or ablation, not just a visually compelling top-k readout.
- The first MLX lens is a rank-limited projection over the language backbone
  and tied text head. Playing the model's speech response does not extend that
  claim to the separate acoustic-token or waveform path.
- The Chatterbox page contains two separate calculations: a projected,
  corpus-averaged J-lens from intermediate T3 speech-prediction positions to
  the final T3 speech-code head, and a per-run cross-Jacobian that
  differentiates one realized code's raw log-probability with respect to
  input-text residual positions. Neither explains S3Gen or waveform samples.
- Chatterbox's waveform-to-code selection is a nominal 25 Hz coordinate: one
  speech code is displayed as 40 ms/two 50 Hz mel frames. S3Gen and the vocoder
  mix context, so the highlighted waveform slice is not an exact causal
  boundary or word alignment. Speech-code IDs are not phoneme labels.

The detailed derivation and interpretation boundaries live in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Milestones

### M0 — Reproduce and understand the reference implementation

Status: **complete**

- [x] Clone `kennethli319/audio-jacobian-lens`.
- [x] Verify the fork against `anthropics/jacobian-lens`.
- [x] Read the paper's method, ablations, formalization, and limitations.
- [x] Trace the reference estimator, Hugging Face adapter, readout, and
  visualization code.
- [x] Run the unmodified test suite and linter.

Acceptance evidence:

- Fork and upstream both point at commit `581d398` at project start.
- Baseline: 32 tests pass and Ruff reports no issues.

### M1 — Specify the audio adaptation

Status: **complete**

- [x] Define separate decoder and encoder-to-decoder Jacobians.
- [x] Define honest UI terminology for confidence versus lens salience.
- [x] Freeze the training-example schema: audio, decoder prefix/transcript,
  valid audio-frame mask, and valid decoder-position mask.
- [x] Freeze lens artifact metadata and compatibility checks.
- [x] Specify a small evaluation protocol with positive, contrastive, silence,
  noise, homophone, and temporal-order cases; execution continues in M3.

Acceptance criteria:

- The equations identify source stream, target stream, source/target position
  reductions, prefix policy, and target decoder layer.
- Saved lenses contain enough metadata to reject an incompatible model,
  tokenizer, task, or estimator configuration.

### M2 — Implement and test the Whisper lens core

Status: **complete**

- [x] Add a Hugging Face `WhisperForConditionalGeneration` adapter.
- [x] Implement the decoder Jacobian estimator.
- [x] Implement global and DTW-aligned rectangular encoder-to-decoder estimators.
- [x] Implement checkpoint/resume and weighted lens merging.
- [x] Implement memory-bounded application over pooled audio windows.
- [x] Unit-test masks, causal direction, shapes, serialization, merging, and
  exact behavior on a tiny synthetic encoder-decoder.

Acceptance criteria:

- Existing language-model tests remain green.
- New unit tests compare the batched estimator to an explicit Jacobian on a
  deterministic tiny model.
- A saved artifact can be loaded and applied without retaining a backward graph.

### M3 — Run a real Whisper Tiny experiment

Status: **in progress**

- [x] Download and configuration-fingerprint `openai/whisper-tiny.en` at Hub
  revision `87c7102498dcde7456f24cfd30239ca606ed9063` for the pilot.
- [x] Add a reproducible macOS TTS paired-corpus generator for plumbing tests.
- [x] Fit global and DTW-aligned pilot lenses on 10 independent clips.
- [ ] Scale toward 100 natural-speech clips with independent alignment.
- [ ] Record runtime, peak memory, convergence, and artifact size on the local
  Apple M2 Pro (16 GB). Runtime and size are recorded; memory/convergence remain.
- [x] Compare decoder J-lens, encoder variants, direct-unembedding baselines,
  and the actual output distribution.
- [ ] Run silence/noise and transcript-prefix controls.
- [ ] Add clip-level aggregation, bootstrap uncertainty, and external or known
  alignment boundaries.

Acceptance criteria:

- A reproducible command produces a non-empty lens artifact and analysis JSON.
- The held-out analysis does not fit on the clip it visualizes.
- Results include failure cases and do not rely only on cherry-picked examples.

### M4 — Localhost explorer

Status: **in progress**

- [x] Add a local API for status and upload/analysis; result caching remains.
- [x] Add audio playback, waveform, and synchronized playhead.
- [x] Show raw teacher-forced transcript-token probability/entropy and
  alternatives without calling them calibrated confidence.
- [x] Show encoder layer × audio-time J-lens cells.
- [x] Show decoder layer × transcript-position J-lens cells.
- [x] Support top-k inspection, token pinning/rank tracking, and clear legends.
- [x] Surface model/lens provenance and interpretation warnings in the UI.
- [x] Add a built-in demo so frontend behavior is testable without a model
  download.
- [x] Add one-click, attributed sample audio and in-browser microphone
  recording.
- [x] Replace horizontally scrolling token grids with focus navigators,
  hover previews, keyboard/click/touch pinning, and a fixed all-layers inspector.
- [x] Replace encoder position cards with proportional waveform slices and use
  one pinned timeline selection to synchronize LM/output-head, encoder, and
  decoder readouts.
- [x] Encode output probability on an absolute 0–100% orange scale and J-lens
  magnitude on a separately labeled, within-layer blue intensity scale.
- [x] Add an encoder-only token-length input that reranks readouts over lexical
  tokens up to a session-selected decoded character count and labels this as a
  phoneme-oriented Audio J-Lens adaptation, not original J-Lens or probability.
- [x] Add an independent decoder token-length input for L0 and L1 that reranks
  vocabulary-wide exact-length buckets while keeping L2 and the output head as
  unfiltered controls.
- [x] Replace selected-position ASR layer cards with compact, full-duration
  layer strips: exact overlapping encoder windows, approximate timed decoder
  tokens, and an aligned orange LM-head row. Keep one synchronized pinned
  coordinate and expose token text/ID, raw metric, exact scoped rank, rank
  denominator, timing, and filter provenance on hover, focus, and selection.
- [x] Add a compact cell-local tooltip to the ASR layer strips. On hover or
  keyboard focus it shows the coordinate, filter state, and top three candidate
  token IDs with exact scoped ranks/denominators and raw lens scores; HEAD uses
  actual probability and log probability. Keep the full right inspector pinned
  until the user explicitly selects a different coordinate.
- [x] Add real encoder pooling overlap: 200 ms windows with a default 20 ms
  overlap, a zero-overlap control, exact range metadata, and adaptive widening
  for the 80-bin display limit.
- [x] Add visible model-path-specific guides explaining how the lens is applied,
  what its evidence can and cannot imply, and how to use the actual controls for
  Whisper ASR, MLX speech-to-speech, and Chatterbox TTS.
- [x] Standardize ASR, speech-to-speech, and TTS around one numbered
  input/analysis/provenance/output/layer hierarchy, one workspace navigation,
  and one semantic color contract without conflating their distinct metrics.
- [ ] Add analysis-result caching.

Acceptance criteria:

- One documented command starts the site on localhost.
- A user can load a sample, record, or upload audio; inspect both lens streams
  without horizontal page scrolling; pin a position or layer; and keep the
  audio playhead synchronized with the selected interval.
- The site remains usable on a laptop-sized screen and never labels a J-lens
  score as "confidence."

### M5 — Causal validation and release-quality hardening

Status: **in progress**

- [x] Implement a reusable encoder-residual intervention runner that injects a
  matched-norm vector at an exact raw 20 ms position span and recaptures every
  downstream encoder and decoder layer.
- [x] Add a reproducible Laurel/Yanny causal-trace CLI: contrast full candidate
  sequences, scan layer/window/strength, and serialize baseline-versus-steered
  outputs. The experiment must accept local audio first; do not bundle the
  source demo clip without verified reuse permission.
- [ ] Validate the first trace on the original clip plus documented spectral
  variants, with random-direction and matched-norm controls.
- [x] Add a static browser Causal Trace view for a recorded experiment before
  exposing interactive steering controls.
- [x] Support a matched-total-norm ordered multi-layer encoder schedule
  (for example L1+L2+L3) and record it alongside single-layer and random-control
  experiments on the separate causal-study page.
- [x] Add direction-selectable Yanny/Laurel strength-sweep visualizations that
  retain failed target flips, off-manifold generations, candidate-set values,
  and per-layer propagation rather than reporting only favorable edits.
- [x] Support repeated time-localized target-piece segments at the same encoder
  layers, with one tokenizer-faithful and one explicitly phoneme-proxy Yanny
  experiment recorded on the causal-study page.
- [x] Make tokenizer-faithful target-piece segmentation the causal CLI default:
  without explicit segments, divide the chosen audio span across the positive
  candidate's actual Whisper BPE pieces. Keep `--segment` as an explicit
  timing/proxy override.
- [x] Add an interactive fixed-budget piece-impact comparison for baseline,
  ` Y` only, `anny` only, and the tokenizer-faithful two-piece default. Show
  actual per-step token softmax, joint path probability, complete-candidate
  comparison, free generation, downstream propagation, and one random control.
- [x] Implement encoder-frame and decoder-position steering controls. Decoder
  schedules track absolute autoregressive positions through the KV cache, edit
  J-lens source L0–L2, and capture downstream target L3 and output logits.
- [x] Add an optional encoder vocabulary-prefix-family intervention. Resolve
  every ordinary decoded token with the requested case-sensitive prefix,
  average its source directions, and expose family counts, true target-token
  probabilities, restricted candidate shares, and negative results separately.
- [x] Make total candidate-token log probability the primary causal candidate
  comparison while retaining mean token log probability as a secondary
  length-normalized diagnostic. Show both restricted total-path share and
  absolute path likelihood, and state that EOS is excluded.
- [ ] Implement matched encoder-frame and decoder-position ablations.
- [ ] Test whether surfaced token directions causally affect relevant transcript
  tokens more than matched random directions.
- [x] Add corpus/example/version hashes and interruption-safe deterministic
  fitting checkpoints.
- [ ] Add user-facing fit progress reporting and explicit seed controls.
- [ ] Add performance budgets and graceful CPU/MPS fallbacks.
- [ ] Finish user documentation, limitations, screenshots, and troubleshooting.

Acceptance criteria:

- At least one preregistered acoustic or lexical hypothesis has a matched-norm
  causal control.
- All tests and lint checks pass from a clean environment.
- The README describes setup, fitting, serving, interpretation, and known
  limitations end to end.

### M6 — Hugging Face Docker Space hosting

Status: **in progress**

- [x] Add valid Docker Space card metadata for port 7860 and document the
  live deployment.
- [x] Add a least-privilege Docker image, smoke-test its entrypoint locally,
  build it remotely, bind `0.0.0.0:7860`, and include browser audio support.
- [x] Select a compatible lens for private hosting; record its provenance,
  checksum, immutable commit, and private delivery path.
- [ ] Complete an explicit redistribution review before making the fitted lens
  artifact public.
- [x] Create a Pro Protected Space, upload the deployment, and verify the build,
  model-backed `/api/status`, both pages, and a real sample analysis.
- [ ] Benchmark cold start, memory, and per-clip latency on the selected Space
  hardware; configure a paid-hardware sleep time and spending guardrail if an
  upgrade is used.
- [x] Publish a cloud-specific privacy/retention notice.
- [ ] Decide on authentication, rate limiting, and abuse/cost controls before
  inviting unrestricted public uploads.
- [ ] Configure the Pro custom domain (`CNAME` to `hf.space`), wait for Ready,
  and verify HTTPS microphone recording plus the website iframe/link.

Current private deployment artifact:

- path: `artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt`;
- size: 2,075,933 bytes;
- SHA-256: `489582ce50736e8e7b78b7cb9d0140b963d99c5d7784635ed93e71710ab293a2`;
- model: `openai/whisper-tiny.en` at
  `87c7102498dcde7456f24cfd30239ca606ed9063`;
- fit: ten synthetic macOS TTS clips, with the negative encoder-pilot result;
- delivery: private `kennethli319/audio-jacobian-lens-artifacts` repository at
  commit `baf116d4f1dfb695b7c5e6ba18e8ed6b338ce696`, plus an `HF_TOKEN` Space
  secret; and
- publication status: available to the Protected Space but blocked from public
  artifact distribution pending explicit provenance/redistribution review.

Acceptance criteria:

- A clean checkout builds locally and the same commit reaches a healthy Docker
  Space on port 7860.
- Model-backed `/api/status` reports the pinned Whisper revision and compatible
  lens provenance; a bundled sample completes within the recorded resource
  budget.
- Visibility, artifact licensing, upload handling, public-access limits, custom
  domain, and embedding behavior are documented and tested rather than assumed.

### M7 — Local MLX speech-to-speech vertical slice

Status: **in progress**

- [x] Pin an Apple-silicon runtime and
  `mlx-community/LFM2.5-Audio-1.5B-8bit` checkpoint at immutable revision
  `a569a7805a8e3eae954c244e54ba811d479a12c2`.
- [x] Add an optional adapter that generates deterministic interleaved text and
  audio, replays that path, and captures all 16 post-block language residuals.
- [x] Read the final-normalized residual through LFM's tied text head and expose
  raw text-token diagnostics separately from J-lens salience.
- [x] Implement native MLX output-probe VJPs with a seeded, subsampled Hadamard
  basis and a factorized, fingerprinted projected-lens artifact.
- [x] Add a fitting CLI and local backend that reuse the explorer for language
  readouts and generated-speech playback.
- [x] Verify full-rank reconstruction against an explicit Jacobian on a tiny
  deterministic MLX model and cover the backend payload/capability boundary.
- [x] Record the poor one-clip rank-64 smoke audit and set rank 512 as the
  minimum pilot default.
- [x] Fit and serve a one-clip rank-512 Hadamard smoke artifact to verify the
  complete fit, API, generated-speech playback, and synchronized browser path.
- [x] Separate the MLX serving-time interleaved generation budget from the
  artifact-fingerprinted fitting policy; use 512 total interleaved positions as
  an emergency ceiling while treating final audio EOS as normal completion.
- [ ] Fit and retain a rank-512 multi-clip pilot with disjoint held-out clips,
  rank/seed sensitivity, failure cases, and runtime/peak-memory measurements.
- [ ] Add interruption-safe fit checkpointing and an MLX-specific held-out
  evaluation command before scaling the corpus.
- [ ] Extend capture and fitting to the FastConformer/audio adapter with honest
  audio-time alignment.
- [ ] Define and validate separate Depthformer/audio-codebook targets before
  making any claim about the generated waveform.
- [ ] Complete model-license and derived-artifact review before distributing an
  LFM checkpoint or fitted MLX lens.

Acceptance criteria:

- On Apple silicon, the pinned optional environment can fit a rank-512 artifact
  and serve a short input through the local language-lens and speech-playback
  path.
- A multi-clip report compares projection ranks and seeds on disjoint held-out
  audio and includes negative examples; a one-clip smoke artifact is not enough.
- The UI and documentation never imply that a tied-text-head lens explains the
  FastConformer, audio codebooks, codec decoder, or played waveform.
- Conformer and audio-head work use separately named streams, estimators, and
  validation rather than being folded into the implemented language lens.

### M8 — Local MLX Chatterbox fitted speech-code lens and text trace

Status: **in progress**

- [x] Verify Chatterbox's actual causal sequence and distinguish ordinary T3
  self-attention from encoder-decoder cross-attention.
- [x] Pin `mlx-community/chatterbox-turbo-8bit` and `S3TokenizerV2` to immutable
  revisions in the existing Apple-silicon MLX runtime.
- [x] Implement deterministic greedy speech-code generation, waveform decode,
  full teacher-forced T3 replay, and replay-logit agreement checks.
- [x] Compute exact per-run VJPs of the realized speech code's raw
  log-probability to input-text residual positions at L0/L4/L8/L12/L16/L20/L22.
- [x] Reconstruct the selected speech position's causal self-attention to text,
  retaining both within-text shares and total text-prefix mass.
- [x] Add a separate `/chatterbox` page, bounded in-memory run cache, on-demand
  trace API, generated-audio playback, keyboard-accessible 40 ms navigator,
  layer/text synchronization, synthetic UI fixture, and interpretation notes.
- [x] Exercise a real pinned-model generation and trace through HTTP and the
  browser with no console errors or horizontal page overflow.
- [x] Bring the Chatterbox report onto the shared explorer hierarchy, add full
  run provenance near the result header, distinguish speech-head probability,
  gradient sensitivity, and attention by color, and encode actual within-text
  shares rather than layer-max-rescaled visual values.
- [x] Replace per-layer summary cards with synchronized blue gradient and violet
  attention layer-by-token matrices that expose every sampled layer × tokenizer
  piece value, retain numeric percentages, and band long prompts without page
  overflow.
- [x] Implement a paper-style same-stream T3 speech-position J-lens using
  signed joint multi-layer MLX VJPs, seeded Hadamard projection, strict artifact
  compatibility checks, and the model's real final normalization and speech
  head.
- [x] Fit the first retained ten-prompt rank-128/1024 pilot over 480 speech
  positions and evaluate all four disjoint held-out manifest prompts.
- [x] Add compact waveform-aligned fitted layer strips with an orange
  actual-head reference row, exact fitted softmax values, global ranks, top
  codes, artifact provenance, synchronized selection, and keyboard control.
- [x] Add a selected-position candidate inspector that compares bounded top-k
  acoustic code IDs and full-vocabulary probabilities at every fitted layer and
  the actual raw speech head, always retaining the realized code and global
  rank without inventing word or phoneme labels.
- [x] Add a forced-code counterfactual branch from the selected speech position:
  retain the parent prefix, force one ordinary candidate at the actual T3 output
  decision, greedily regenerate the suffix, decode a new waveform, and preserve
  the parent raw-head rank/probability/logit-gap provenance separately from the
  fitted layer that nominated the candidate.
- [x] Add a separate gradient-proposed residual-steering branch: choose one or
  more sampled post-block T3 layers and a bounded forward speech-position span,
  search a matched-relative-norm edit budget until the target is both unique raw
  top-1 and the processed greedy choice when feasible, regenerate the suffix,
  and expose exact target rank/probability movement across every fitted layer,
  the real HEAD, and every requested position.
- [ ] Compare rank 128 against at least one second rank and seed, a direct
  untransported logit-lens baseline, a shuffled control, and the exact
  full-rank map on a small real subset before treating fine ordering as stable.
- [ ] Add a deterministic S3 encoder frame-to-code VJP and clearly labeled
  two-stage frame-to-code-to-text sensitivity composition.
- [ ] Validate real-model gradients with finite differences, text substitutions,
  matched controls, and an unquantized/rank-sensitive checkpoint comparison.
- [ ] Add optional reference-voice input and record all voice/noise provenance.
- [ ] Complete conversion-card, model-license, and derived-output review before
  any hosted or bundled-weight deployment.

Acceptance criteria:

- A pinned local command opens `/chatterbox`, generates playable speech, and a
  selected output slice updates its speech-code, text-token, and T3-layer views.
- Replay logits reproduce generation logits within a recorded tolerance; all
  returned gradients are finite and raw values remain available beside display
  normalization.
- A compatible fitted artifact loads before serving, and an unseen generated
  sequence exposes fitted realized-code probability/rank across every fitted
  layer without changing the separate per-run text trace.
- Selecting a non-realized ordinary candidate and explicitly running a branch
  creates a new analysis whose codes before the selected position match the
  parent, whose selected code is the requested ID, and whose later codes and
  waveform are recomputed from that changed autoregressive path.
- Residual steering never directly substitutes the target code. It records the
  selected post-block layer/position coordinates, parent residual and gradient
  norms, applied delta norms, all calibration attempts, raw-top-1 and processed
  greedy outcomes separately, and a budget-exhausted result when the target does
  not win within the requested cap.
- The page never calls acoustic-code IDs phonemes, attention an explanation, or
  a nominal 40 ms coordinate an exact waveform attribution.
- End-to-end acoustic attribution remains explicitly incomplete until the S3
  frame-to-code stage and causal/finite-difference controls pass.

### M9 — Static, evidence-led showcase

Status: **in progress**

- [x] Screen candidate examples by exact realized-token/code rank trajectories,
  not visual impression alone.
- [x] Select a canonical Chatterbox example that combines fitted emergence,
  local text sensitivity, and a verified residual intervention.
- [x] Select rights-safe natural ASR examples for early readability, late
  emergence, BPE behavior, and a genuine model failure.
- [x] Specify silence/noise and pooled-distribution context so highlighted
  examples are not presented without controls.
- [x] Record source and derived-audio rights gates before copying screening
  assets into a public static bundle.
- [x] Replace the public Laurel/Yanny route with a backend-free curated
  Showcase across ASR, speech-to-speech, and TTS while retaining the original
  causal files as a reproducible archive.
- [x] Add selectable success, non-monotonic, failure, and null-control stories
  with exact rank trajectories, fitted-versus-head styling, local hover/focus
  details, intervention provenance, and publication-status boundaries.
- [x] Add a checksum-validated, rights-scoped LFM evidence artifact without
  generated audio, ephemeral server IDs, or unresolved-rights media.
- [x] Freeze a versioned nine-report public-review bundle with exactly three
  ASR, three speech-to-speech, and three TTS examples; retain full rank
  denominators, provenance, input hashes, caveats, and cached intervention
  semantics without live inference identifiers.
- [x] Publish an unlinked, `noindex` GitHub Pages review at
  `kennethli319.github.io/audio-jacobian-lens/` without changing the personal
  homepage; make the complete cached explorers canonical at the root,
  `/speech/`, and `/tts/`, move the curated interpretations under `/findings/`,
  retain functional `/explorer/{family}/` aliases, and include only the three
  rights-cleared LibriSpeech input FLACs.
- [x] Implement deterministic allowlist-based exporters and schema validation
  for ASR, LFM, Chatterbox generation, all-position trace, and the recorded
  bridge-intervention response.
- [x] Load precomputed manifests/payloads through one shared detailed-static
  renderer across ASR, speech-to-speech, and TTS. Keep this renderer separate
  from the live pages so upload, generation, and mutation controls cannot leak
  into the public build.
- [x] Keep every cached speech-to-speech layer × output-position top-token cell
  while dividing long response timelines into aligned, readable position bands
  with full token text available locally.
- [ ] Package the Chatterbox bridge baseline/steered pair after completing the
  conversion/S3/derived-output review.
- [x] Package the three attributed LibriSpeech inputs with immutable hashes and
  CC BY 4.0 attribution.
- [ ] Add procedural silence and obtain rights-cleared owner recordings for the
  controlled homophone pair.
- [ ] Fit and evaluate a multi-clip LFM lens on disjoint held-out examples before
  choosing a positive public speech-to-speech hero.
- [x] Add a static-host integrity gate and regeneration/deployment
  documentation covering hashes, matrices, trace coverage, media scope,
  no-index pages, and absence of live API calls.
- [ ] Add compressed-transfer size budgets and responsive/browser QA before
  promoting the review out of its unlinked `noindex` state.

Acceptance criteria:

- Every public example has a stable ID, explicit teaching purpose, immutable
  model/lens provenance, exact rank semantics, source/audio hashes, and a
  complete license/attribution record.
- Visitors can inspect all layer/position cells and neighboring counterexamples,
  not only a preselected hero coordinate.
- The detailed explorer is the entry point for each model family; curated
  findings are clearly linked secondary interpretation pages rather than a
  substitute for the underlying matrices.
- Long speech-to-speech responses remain readable without dropping, merging, or
  silently aggregating any saved top-token cell.
- The static pages preserve the live site's distinctions among fitted readout,
  raw head, gradient, attention, forced output, and residual steering.
- At least one failure/null control and pooled frequency context accompany the
  positive examples.
- No one-clip, in-sample, or unresolved-rights artifact is presented as held-out
  scientific evidence.

## Current implementation choices

These are defaults, not immutable facts. Change them only with a note in the
decision log.

- Model for the first end-to-end run: `openai/whisper-tiny.en`.
- Reference implementation is preserved; audio code is additive.
- Decoder fitting uses reference-transcript teacher forcing. The default
  aligned encoder fit and held-out evaluator use generated targets with
  Whisper-derived DTW timing. The interactive view uses generated prefixes.
- Decoder target is configurable; the pilot targets final block 3. Omitting one
  block would discard 25% of Tiny's decoder, so the paper's penultimate-layer
  preference is retained as an ablation rather than transferred by default.
- The default encoder estimator samples one generated token per clip and its
  cross-attention/DTW-aligned audio window. The global all-output estimator is
  retained as a comparison.
- Encoder readouts default to 200 ms windows with 20 ms overlap (180 ms hop)
  before unembedding. The site offers zero overlap as an analysis-time control;
  long clips widen the window as needed to remain within 80 display bins.
- The encoder grid ranks target-mean-relative logit changes; its absolute affine
  readout is retained for evaluation but is dominated by the corpus language
  prior in the 10-clip pilot.
- The explorer uses a focus-and-context layout: the encoder uses a proportional
  waveform-slice slider and the decoder uses a wrapping token navigator. One
  pinned timeline selection updates output token, encoder, decoder, waveform
  overlays, layer comparisons, and inspector; hover remains a local preview.
- Synchronization uses reported encoder ranges and approximate Whisper-derived
  token timing. A gap between reported token intervals uses the nearest token
  and labels it approximate; wholly unavailable timing remains unmatched.
- Output probabilities use an absolute percentage gradient. J-lens logits keep
  their raw numeric values and use only a within-layer visual intensity, never
  a percent or confidence label.
- The optional token-length views store vocabulary-wide top-k buckets for every
  exact decoded length (ignoring surrounding whitespace). The browser merges
  lengths `1…N` for independent, session-selected encoder and decoder limits,
  changing only the displayed rankings and rank-based tint. Decoder filtering
  is deliberately limited to L0 and L1; L2 and LM-head probabilities remain
  unfiltered controls. Neither view is treated as a true phoneme inventory.
- Padded audio frames, forced decoder-prefix positions, and positions without a
  next-token target are excluded from estimator averages.
- The public detailed replay uses three immutable family manifests and one
  shared static renderer. Base reports retain the full saved layer × position
  matrices and bounded candidates. ASR exact-length buckets live in separate,
  compact, hash-pinned sidecars and load only when the visitor enables the
  character filter.
- Static speech-to-speech pages distribute the cleared input waveform and
  generated text but not generated response audio. Long generated-text
  timelines wrap into aligned position bands: each band repeats the layer
  labels and preserves one individually selectable top-token cell per original
  layer/position coordinate. Static TTS pages distribute speech-code/readout/
  trace values and the recorded bridge intervention, but no generated waveform,
  audio URI, or ephemeral server analysis handle.
- Pilot lens: at least 10 clips for a plumbing/quality gate. A one-clip lens may
  be used only as an explicitly labeled smoke test.
- The first hosted target is a Hugging Face Docker Space on port 7860. It is
  model-backed by a private, pinned pilot lens while public redistribution
  remains pending. Pro Protected visibility hides repository contents but does
  not make the running app private.
- The first local MLX target is
  `mlx-community/LFM2.5-Audio-1.5B-8bit` at
  `a569a7805a8e3eae954c244e54ba811d479a12c2`, with MLX 0.32.0,
  MLX-Audio 0.4.5, MLX-LM 0.31.3, and Transformers 5.12.1 in a separate
  Apple-silicon environment. The 8-bit checkpoint is a memory-driven pilot
  choice, and its Jacobians are checkpoint-specific.
- The implemented MLX lens is language-to-language only: post-block,
  pre-final-normalization sources default to L0/L4/L8/L12/L14, target L15, and
  decoding uses the model's final normalization plus tied text embedding.
- MLX fitting uses `subsampled_hadamard_output_probe_vjp`. Rank 512 is the
  minimum pilot default for the 2,048-wide stream after rank 64 produced poor
  smoke readouts; rank 512 remains approximate and requires rank/seed and
  held-out checks. Rank 2,048 is the complete-basis dense reconstruction.
- Generated speech is returned for playback as model output. The 17-layer
  FastConformer, 512-to-2,048 adapter, Depthformer, eight audio-codebook heads,
  and waveform attribution remain explicitly outside the first slice.
- The first Chatterbox target is
  `mlx-community/chatterbox-turbo-8bit` at
  `2f2e21a03863f86a1274d1060dcc188e7cde77e1`, with pinned
  `mlx-community/S3TokenizerV2` at
  `e0c9886f0e1c35ae85b1f27277416fb19fc72bec` in the same MLX runtime.
- Chatterbox uses a deterministic greedy generated path and a full-sequence
  replay. Its default source layers are L0/L4/L8/L12/L16/L20/L22 of 24 T3
  blocks; L23 text-position gradients are excluded because the remaining final
  normalization and speech head are positionwise.
- The fitted Chatterbox lens uses the same sampled source layers at
  speech-prediction positions and targets the post-block L23 residual. Its
  estimator injects each target probe at all selected speech positions, takes
  the mean signed response over source positions, and averages examples. The
  first retained artifact is the uncentered rank-128 Hadamard projection at
  seed 29 fitted on ten prompts/480 positions; rank 1,024 is the exact complete
  basis.
- Fitted Chatterbox states are read through T3's real final normalization and
  6,563-way speech head. The displayed blue value is the full-head softmax for
  the realized acoustic code under the fitted readout; it is not the orange
  base-head probability, calibrated confidence, a phoneme distribution, or an
  account of the downstream waveform.
- The implemented Chatterbox score is the L2 norm of the realized code's raw
  `log_softmax` gradient at each text residual position. Within-text shares are
  display normalization; raw norms, gradient text-versus-prefix mass, ordinary
  self-attention share, and total attention-to-text mass remain separate.

## Evaluation questions

1. Does the decoder lens surface the current/future transcript word before it
   becomes the model's imminent next-token prediction?
2. Does the encoder lens localize phonetic or lexical hypotheses to the audio
   region that supports them, despite bidirectional context?
3. Do ambiguous phonemes show competing token concepts that resolve after more
   context, and is that resolution visible by layer?
4. Are lens scores stable across speakers, volume, background noise, and small
   time shifts?
5. Do ablations along surfaced directions affect the corresponding output more
   than random or unrelated token directions of matched norm?
6. Is Whisper Tiny deep and capable enough to exhibit a useful intermediate
   regime, or does it move directly from acoustics to output-like states?
7. Do Chatterbox T3 gradient hotspots move across input-text pieces as later
   output speech-code slices are selected?
8. Are those hotspots stable under small text edits, voice changes, checkpoint
   precision, and matched finite-difference interventions?
9. Once S3 sensitivity is added, does its nonlocal code receptive field differ
   materially from the nominal one-code-per-40-ms coordinate?
10. Does the fitted Chatterbox speech-code rank improve consistently through
    T3 on held-out prompts, and does that trend survive projection-rank, seed,
    corpus-size, direct-logit-lens, and shuffled controls?

## Known risks

- A four-layer encoder and four-layer decoder may be too shallow to show the
  layer regimes reported for much larger language models.
- ASR may be an "automatic" computation that does not use a sparse,
  verbalizable workspace in the paper's sense.
- Averaging over all decoder targets blurs time-local encoder effects. The pilot
  now compares this global map with a DTW-aligned estimator, but larger
  split-half experiments are needed to select between them.
- Whisper-derived DTW is used for both aligned fitting and the current held-out
  location metric. Independent forced alignment or known word boundaries are
  required before claiming localization.
- Token-pooled metrics overweight longer clips and currently lack bootstrap
  uncertainty; report clip-level aggregates and confidence intervals at scale.
- Teacher forcing can expose information not present during free generation;
  generated-prefix and gold-prefix views must be compared.
- Whisper tokenization fragments many words, and a single-token lens cannot
  express arbitrary phrases.
- Full-vocabulary unembedding at every 20 ms frame is expensive; pooling and
  chunking are required for an interactive site.
- MPS support for repeated retained-graph backward passes may be incomplete or
  slower than CPU for some operations.
- The first Chatterbox checkpoint is an 8-bit community MLX conversion;
  quantization can materially change gradients even when output audio is useful.
- Chatterbox T3 gradients are local sensitivities on one teacher-forced sampled
  path, not causal effects, probabilities of text use, or guarantees that a
  highlighted text piece controls the selected sound.
- The first fitted Chatterbox artifact has only ten prompts and rank 128/1024.
  Its encouraging held-out rank progression may reflect late-layer proximity,
  the projection seed, repeated code statistics, or the fixed built-in voice;
  it is a working pilot rather than validated mechanistic evidence.
- Nominal waveform-to-code timing omits S3 encoder, flow, and vocoder context.
  It must not be upgraded to end-to-end acoustic attribution without the
  separate S3 stage and validation controls.

## Decision log

### 2026-07-10 — Split output confidence from lens salience

The paper reports that the Jacobian lens is worse than a tuned lens, and often
worse than a logit lens, at predicting the actual next-token distribution. The
site will therefore show actual Whisper probabilities separately and will use
rank/salience language for J-lens readouts.

### 2026-07-10 — Build two lenses, not one

The decoder lens preserves the paper's same-stream causal structure. The
encoder-to-decoder lens answers the audio-segment question but is a new
cross-modal estimator with no `t' >= t` relation between audio and text axes.
They will be trained, stored, evaluated, and labeled separately.

### 2026-07-10 — Keep the original implementation intact

At project start the fork matched Anthropic's public commit exactly. Audio
support remains additive so the reference tests and behavior stay available
for direct comparison.

### 2026-07-10 — Center cross-modal transport at fitted activation means

Encoder and decoder residual streams do not share a meaningful zero point.
The reusable encoder map therefore uses the corpus-average affine
linearization `J(h - mean_encoder) + mean_decoder_target`; same-stream decoder
transport retains the paper-style linear readout.

### 2026-07-10 — Target Whisper Tiny's final decoder block first

The paper found a penultimate target slightly cleaner on a much deeper model.
For Whisper Tiny, dropping one decoder block removes a quarter of the stack, so
the pilot uses final block 3 and leaves the penultimate target as an ablation.

### 2026-07-10 — Compare aligned and global encoder maps

The global encoder map is the simplest cross-modal extension but mixes every
output token with every audio frame. The default fit samples one generated
token per clip and uses its cross-attention/DTW audio window. Both artifacts are
retained, but the centered 10-example pilot validates neither: aligned and
remote absolute ranks are nearly identical and baseline-relative recovery is
poor.

### 2026-07-10 — Use a focus-and-context explorer with distinct color scales

The original J-lens viewer keeps positions and layer details in stable regions
instead of making users chase a wide matrix. The Whisper explorer therefore
uses wrapping position navigators plus fixed layer comparison and inspection
areas. Orange encodes absolute model probability from 0–100%; blue encodes only
within-layer J-lens display intensity, with raw logits retained in text.

### 2026-07-10 — Use one honest shared audio-time selection

Encoder bins already have exact display-window boundaries, while output tokens
have approximate Whisper cross-attention/DTW ranges. The UI maps token clicks
through their midpoint to a representative encoder bin. Waveform clicks use a
covering token interval when possible and the explicitly labeled nearest
interval in a gap. A blue overlay marks the encoder slice and an orange band
marks the output-token interval; wholly missing timing remains unmatched.

### 2026-07-10 — Treat short encoder tokens as an exploration filter

Encoder cells often surface fragments rather than word-like tokens. The UI may
therefore switch to a separately computed ranking up to a user-selected token
length. This is only a phoneme-oriented proxy: Whisper uses text tokens,
not a phoneme vocabulary, and the source J-Lens paper did not define this
cross-modal filter. The interface must preserve the raw score and never label
the filtered ranking as probability or phoneme classification.

### 2026-07-10 — Keep decoder length filtering early and display-only

The decoder explorer may apply the same exact decoded-character masking to L0
and L1 before top-k selection, using a separate session limit from the encoder.
The backend stores vocabulary-wide exact-length buckets so low-scoring eligible
tokens can enter the reranked list; it never filters an already truncated top-k.
L2 and the direct LM/output head stay unfiltered as late-layer controls, and the
filter does not alter Whisper generation or output probabilities.

### 2026-07-10 — Default to modest overlap between encoder display windows

Whisper encoder positions are spaced 20 ms apart, so a 200 ms window contains
ten positions. The explorer defaults to one-position (20 ms) overlap and a
180 ms hop, with zero overlap available as an analysis-time comparison. This
smooths boundary transitions without pretending to increase temporal
resolution. Adjacent overlapping readouts are correlated, and long clips widen
the window while retaining the requested overlap to stay within 80 bins.

### 2026-07-10 — Treat causal steering as an intervention, not an encoder token distribution

Encoder token readouts are J-lens diagnostics; Whisper's encoder itself has no
vocabulary softmax. A causal experiment must add a matched-norm vector to an
actual encoder residual state over the raw positions behind a selected waveform
slice, then rerun all downstream encoder blocks and the complete decoder. The
primary score will be the normalized teacher-forced contrast between complete
candidate transcripts (for example, `Yanny` versus `Laurel`), with free
generation as a secondary outcome. A J-lens-derived contrast direction is a
testable proposed intervention, not proof that a displayed token was used.

### 2026-07-10 — Default causal target timing to the tokenizer path

When a causal run supplies an audio span but no manual segments, it now splits
that span evenly among the actual Whisper BPE pieces of the positive candidate.
This keeps the usual experiment tied to the model's text interface (for
example, ` Y` then `anny`) instead of silently imposing a phoneme segmentation.
Manual `--segment` entries remain available for deliberately nonuniform timing,
overlap, or phoneme-inspired proxy experiments and must be labelled as overrides.
The causal page compares piece allocations at a fixed 20% total budget. Because
single-piece and two-piece runs distribute that budget differently, the view is
an intervention comparison rather than an additive attribution of the word.

### 2026-07-10 — Keep decoder piece generation explicitly open-loop

Teacher-forced decoder edits use the positive candidate path to calibrate
residual norms and token positions. During free generation, the same absolute
position schedule is applied even if an earlier generated token differs from
the target prefix. This open-loop policy is reproducible and cache-correct, but
it is not the same claim as conditional `p(anny | Y, audio)`; the page labels
the distinction. Encoder and decoder percentage budgets remain within-stream
relative norms, not cross-stream matched doses.

### 2026-07-10 — Keep prefix-family steering separate from exact-token steering

Encoder prefix-family experiments remove leading whitespace and then perform a
case-sensitive match over decoded ordinary vocabulary tokens. They average each
matching token's fitted source direction with equal weight and contrast it with
a separately declared negative-prefix family. This is a vocabulary-group
intervention, not a phoneme distribution: `Y*` includes words such as `You` and
`York`, while `anny*` contains only the exact `anny` token in the pinned model.
The causal page retains exact-BPE and prefix-family modes side by side and always
judges both against the exact target's full-vocabulary token probabilities.

### 2026-07-10 — Prefer total token-path likelihood in causal comparisons

The causal runner and page use the sum of teacher-forced candidate-token log
probabilities as the primary candidate score. Restricted shares normalize those
totals only across the declared comparison strings; the page displays the
absolute path likelihood beside them. EOS is not scored, and longer tokenizer
paths multiply more conditional probabilities, so neither value is a calibrated
completed-transcript probability. Mean token log probability remains available
as an explicitly secondary length-normalized diagnostic.

### 2026-07-10 — Separate Space preparation from public launch

The repository may carry Docker Space metadata and a reproducible container
before a hosted instance is advertised. A live analysis deployment additionally
needs a deliberately published, model-compatible lens, hardware measurements,
and cloud-specific privacy and traffic controls. Pro Protected visibility is a
source-visibility choice: the app, custom domain, and embed remain public. A
Private Space restricts the app but cannot use the public embed or custom-domain
path. The image may prefetch the pinned public Whisper snapshot, but the fitted
lens stays separate until its training-data rights and provenance are reviewed;
a private Hub artifact repository is accessed through a read-only Space secret.

### 2026-07-10 — Pin the first MLX slice to the 8-bit LFM checkpoint

The first Apple-local speech-to-speech target is
`mlx-community/LFM2.5-Audio-1.5B-8bit` at immutable revision
`a569a7805a8e3eae954c244e54ba811d479a12c2`. Its smaller memory footprint makes
the complete generation/capture/VJP path practical on the 16 GB M2 Pro, but
quantization can change the Jacobian. Artifacts and claims are therefore tied
to this exact checkpoint and runtime fingerprint rather than generalized to
LiquidAI's BF16 model.

### 2026-07-10 — Use a Hadamard-projected MLX estimator with a rank-512 floor

The 2,048-wide language stream makes a direct reverse-mode dense fit require
2,048 probes per source layer. The MLX pilot stores target and source factors
from a seeded subset of an orthogonal Hadamard basis, recording method, rank,
seed, reduction, runtime, model, tokenizer, and example fingerprints. The
earlier one-clip rank-64 Rademacher smoke artifact had poor lexical readouts and
is not evidence. Rank 512 is the minimum pilot default, while rank 2,048 is the
complete-basis reconstruction; rank/seed sensitivity remains required.

### 2026-07-10 — Keep generated speech playback outside the text-head claim

The first MLX lens captures only the 16-layer LFM language backbone and decodes
through its tied text head. The application may play the model's generated
24 kHz response so the vertical slice remains genuinely speech-to-speech, but
that audio travels through separate Depthformer, codebook, and codec paths that
the lens does not yet target. Conformer, adapter, and audio-codebook transports
will be implemented and evaluated as separately named extensions.

### 2026-07-10 — Treat Chatterbox frame-to-text as a per-run trace

Chatterbox T3 is one causal transformer over voice conditioning, input text,
and prior speech codes; it has no encoder-decoder cross-attention. The first
page therefore differentiates the selected realized speech code's raw
log-probability with respect to post-block text residuals and reconstructs
ordinary causal self-attention only as a second association diagnostic. This
context-specific cross-Jacobian is not the paper's corpus-averaged J-lens,
attention is not causal attribution, and normalized display shares never
replace raw gradient norms or text-versus-prefix mass.

### 2026-07-10 — Pin Turbo 8-bit and keep waveform timing nominal

The first local Chatterbox slice uses
`mlx-community/chatterbox-turbo-8bit@2f2e21a03863f86a1274d1060dcc188e7cde77e1`
and the independently pinned S3TokenizerV2 revision. It fits a 16 GB M2 Pro and
supports fast full replay plus on-demand VJPs, but all gradient claims remain
checkpoint-specific until compared with an unquantized model. The UI maps one
25 Hz T3 code to a nominal 40 ms/two-mel-frame waveform band. Because S3Gen and
the vocoder mix context, the band is a navigation coordinate—not an exact
acoustic cause, alignment, phoneme, or word boundary.

### 2026-07-10 — Give each speech task its own J-lens explanation

A single generic definition hides important architecture and estimator
differences. The ASR page now distinguishes the paper-style decoder transport
from the experimental encoder-to-decoder map; the MLX speech-to-speech page
limits its claim to the projected causal language backbone and treats generated
audio as playback only; and the Chatterbox TTS page distinguishes its fitted
same-stream speech-code lens from its per-example code-to-text cross-Jacobian.
Each page shows the application, evidence boundary, separate scales, and
concrete usage steps before the user runs an analysis.

### 2026-07-10 — Unify presentation without pretending the metrics are equivalent

ASR, speech-to-speech, and TTS now use the same numbered information flow,
workspace navigation, result header, expandable provenance, output summary,
layer comparison, and focused diagnostic patterns. The shared color contract is
orange for direct output-head probability, blue for fitted J-lens or local
gradient sensitivity, violet for attention association, and gray for timing or
provenance. Presentation consistency does not erase the scientific distinction:
Whisper uses fitted decoder and experimental encoder-to-decoder transports,
LFM uses a rank-limited projected fitted language transport, while Chatterbox
uses both a projected fitted speech-position transport and a separate per-run
code-to-text cross-Jacobian. Chatterbox's token fills and text matrices directly
encode the reported normalized shares; they are no longer rescaled to each
layer's largest token.

### 2026-07-10 — Prefer paired layer-by-token matrices to layer summary cards

Chatterbox now renders two aligned small-multiple heatmaps: blue post-block
gradient share and violet block self-attention share. Rows are sampled T3
layers, columns are tokenizer pieces, and cells show both an absolute 0–100%
within-text color scale and the numeric percentage. A separate rightmost value
retains each layer's text-versus-prefix mass. Long token sequences are divided
into repeated column bands instead of creating horizontal page overflow. Cell
selection synchronizes the focused layer, token strip, matrix outline, and
inspector; arrow keys navigate the two-dimensional coordinates. The matrices
make cross-layer patterns visible without implying that gradient and attention
are the same metric or comparable as causal effects.

### 2026-07-10 — Fit the Chatterbox lens on the causal speech-code stream

The first fitted Chatterbox artifact maps post-block T3 residuals at
speech-prediction positions to the final post-block T3 speech-prediction
residual. This is the closest same-stream analogue of the published J-lens and
can be decoded through the real final normalization and speech head. It is kept
separate from the existing text-position gradient matrices: averaging unsigned
word-level gradient norms would discard the signed hidden-to-hidden transport
and would not create a fitted lens. A future fitted text-to-speech cross-lens
requires an explicit alignment or relative-position policy and will be named as
a separate estimator.

The retained integration artifact uses an uncentered rank-128/1024 seeded
Hadamard projection fitted over ten fixed-voice prompts and 480 selected speech
positions. Four disjoint held-out prompts show a median realized-code rank
progression from 1,593 at L0 to 12 at L22, but no alternate rank/seed,
direct-logit baseline, shuffled control, or unquantized comparison has passed.
The page therefore labels the artifact as a pilot and does not convert that
trend into a mechanistic claim.

### 2026-07-10 — Align compact fitted strips to the nominal waveform timeline

The fitted Chatterbox readout uses one unbroken time axis with one compact strip
per fitted T3 layer plus the actual speech-head strip. Every probability slice
reuses the corresponding speech code's start and end divided by the full
decoded-audio duration, exactly matching the waveform slice geometry. Decoded
audio after the last nominal code remains visibly hatched and unassigned rather
than stretching or inventing code coverage. Blue fitted values and the orange
actual-head values share one run-relative color domain; exact values remain in
the synchronized focus readout and accessible slice labels. The compact view
does not aggregate, drop, or rename speech codes as phonemes.

### 2026-07-10 — Expose acoustic candidates without inventing text labels

The selected-position inspector shows top-k speech-code IDs from every fitted
layer and the actual T3 speech head on their respective full 6,563-entry softmax
domains. The realized code and its global rank remain visible even outside the
bounded top-k response. Actual-head candidates are computed before repetition
penalty, temperature, top-k, and top-p. Because these IDs are learned acoustic
symbols with context-dependent realizations, the interface does not assign them
words or phonemes. Candidate audition or human-readable alignment remains a
separately labeled future estimator rather than being folded into J-lens.

### 2026-07-10 — Treat candidate forcing as an output-decision counterfactual

A selected fitted-layer or HEAD candidate may nominate an ordinary acoustic
code for a new causal branch. The branch preserves parent codes strictly before
position `t`, forces the selected ID at the actual T3 output decision at `t`,
feeds that ID back through the speech embedding and cache, greedily regenerates
the suffix after repetition penalty and temperature, teacher-forced replays the
new path, and S3Gen-decodes a new waveform. The fitted source row is evidence
for choosing the intervention target; no intermediate residual is edited, and
the implementation does not claim fitted-layer causal steering or literally
replace the stored parent logits. Rank, probability, top-1 gap, and minimum
unique-winner bias are reported from the parent's raw full speech head before
generation processors. Control IDs and the already-realized ID are not valid
replacement targets.

### 2026-07-11 — Separate residual steering from fitted readout and code forcing

The residual experiment uses a candidate row only to nominate an ordinary
target code and an initial displayed layer. For target `c`, parent-path speech
position `j`, and selected post-block layer `l`, its proposed direction is the
context-specific gradient of the actual raw-head margin between `c` and the
strongest non-`c` competitor. The edit is normalized and scaled against that
coordinate's parent residual norm. A common relative strength is searched up
to an explicit cap; success requires both a unique raw-head top-1 target and the
target as the post-processor greedy choice at the anchor. No output ID is
substituted in this mode.

Multiple selected layers and up to eight consecutive speech positions form an
open-loop schedule. Each future-position direction comes from the parent
teacher-forced path, then is applied to the dynamic branch path after earlier
edits may already have changed it. The complete edited path is replayed with the
same schedule, and only then is the regenerated code sequence decoded. Fitted
J-lens rows are used to measure before/after readable target distributions, not
to produce the steering direction or to claim the fitted transport is causal.
Large edits above half a residual norm are explicitly flagged as off-manifold;
the experiment remains incomplete without matched random directions,
finite-difference checks, seed/rank replication, and an unquantized model.

### 2026-07-11 — Align the Whisper explorer to one compact audio timeline

The ASR explorer uses the same compact timeline grammar as the fitted
Chatterbox view without equating their metrics. Encoder rows preserve the exact
overlapping pooling-window geometry; decoder and LM-head rows use Whisper's
approximate token timing when available and an explicitly logical token axis
otherwise. Blue still means a within-layer display percentile of a raw J-lens
score, while orange means actual teacher-forced output probability.

Hover and keyboard focus expose the candidate text, token ID, coordinate,
metric value, filter state, and exact competition rank. Default lens ranks are
scoped to the lexical display vocabulary, session-length-filtered ranks are
scoped to the eligible `≤ N`-character lexical vocabulary, and HEAD ranks are
scoped to the full model vocabulary. Ties use one plus the count of strictly
greater scores. These rank scopes are never presented as calibrated confidence.

### 2026-07-11 — Keep transient ASR detail beside the active timeline cell

Dense strip cells need local detail without forcing a repeated eye movement to
the sticky inspector. Pointer hover therefore opens a transient tooltip beside
the cell but does not change the pinned inspector or shared timeline selection.
Keyboard focus and click retain the same local detail while preserving the
existing explicit-selection behavior; touch detail persists until an outside
press, Escape, resize, or scroll.

The tooltip contains only the coordinate, active token filter, and the top three
candidates with token ID, exact scoped rank/denominator, and raw metric. HEAD
rows use actual teacher-forced probability and log probability. Placement is
derived from the cell rectangle, clamped to the chart horizontally, and flipped
or clamped within the viewport vertically. Native `title` tooltips are removed
to avoid duplicate or delayed information.

### 2026-07-11 — Make the cached explorers the public entry points

The full already-inferred layer × position views are the primary public
artifact, rather than a detail screen hidden behind a curated story. The site
root is therefore canonical ASR, with canonical speech-to-speech and TTS pages
at `/speech/` and `/tts/`. Each page links to its shorter experimental findings
under `/findings/`. Functional copies remain under `/explorer/{family}/` so
existing shared URLs and `?sample=...` selections continue to resolve without
redirect or query-string loss.

The static validator treats this information hierarchy as a release contract:
all canonical pages and aliases must use the detailed cached renderer, all
findings pages must use the curated-report renderer, and every route remains
`noindex` while the review is absent from the personal-site homepage.

### 2026-07-11 — Wrap long speech responses without discarding cells

The LFM output axis can be much longer than the ASR examples. Compressing every
position into one viewport makes even the visible top-token text unreadable.
The static speech-to-speech explorer therefore partitions a long output into
aligned position bands. Within each band the position header and all layer rows
share the same columns, layer labels repeat, and every original layer × position
cell remains individually selectable with its saved top candidate, rank, score,
and alternatives. This is a layout transformation only; it does not pool,
truncate, or recompute the cached evidence.

## Work log

### 2026-07-10

- Cloned the fork, created `codex/whisper-jacobian-lens`, and added the
  Anthropic repository as `upstream`.
- Verified no fork delta at the starting commit.
- Read and cross-checked the paper's core method, ablations, J-space
  formalization, limitations, and multi-token extension.
- Audited the released fitting, adapter, lens, and visualization code.
- Created a Python 3.12 virtual environment and established a green baseline:
  32 tests and Ruff.
- Implemented rectangular transports, Whisper preparation/capture, decoder and
  encoder estimators, artifact compatibility metadata, checkpoint/resume,
  merging, pooled application, and a held-out rank evaluator.
- Expanded the test suite to 66 passing tests, including explicit VJP equality
  and a real Hugging Face Whisper architecture instantiated offline.
- Fit one-example smoke lenses, then global and DTW-aligned 10-example pilot
  lenses on MPS. Recorded runtime, artifact size, six held-out failure-inclusive
  evaluations, and estimator comparisons in `docs/PILOT_RESULTS.md`.
- Built and browser-tested the localhost API and responsive frontend with an
  upload flow, waveform, actual output diagnostics, both lens grids, inspector,
  provenance warnings, and a no-backend synthetic demo.
- Added affine cross-stream centering, target-mean-relative encoder display,
  band-limited audio resampling, exact weight/tokenizer/preprocessor and example
  fingerprints, overlap rejection, normalized WER/CER, and safe v1/v2 artifact
  migration. Refit the pinned pilot and recorded the now-negative encoder
  result rather than retaining the legacy uncentered claim.
- Rebuilt the localhost UI as a compact research workspace with wrapping
  position navigators, fixed hover/pin layer comparisons, a direct output-head
  card, accessible keyboard/touch controls, and no token-axis page scrolling.
- Added three attributed LibriSpeech samples, safe sample-serving endpoints,
  in-browser recording/review, and honest dual color scales for raw output
  probabilities versus relative J-lens intensity.
- Hardened recording as an exclusive input state with a dedicated retained
  recording file, sub-30-second auto-stop margin, reset cleanup, and explicit
  keyboard focus handoffs between recording controls.
- Replaced numbered encoder-position cards with a proportional waveform-slice
  navigator and centralized token/waveform activation so LM/output-head,
  encoder, decoder, overlays, and audio seek state update together.
- Added an encoder-only short-token toggle backed by a separately computed
  one- or two-character lexical top-k, with synchronized cards, layer-relative
  intensity, inspector ranking, and explicit adaptation caveats.
- Verified the short-token path masks the complete vocabulary before top-k,
  added regression coverage for low-scoring eligible tokens, and brought the
  synthetic demo onto the same dual-ranking response shape as real analysis.
- Replaced the hard-coded two-character encoder filter with a session-editable
  maximum. Exact-length vocabulary buckets make every `≤ N` reranking correct
  and instantaneous without another Whisper forward pass.
- Added an independent decoder character-length filter for L0 and L1. It uses
  the same full-vocabulary exact-length masking before top-k, supports any
  session-selected `≤ N` limit, and deliberately leaves L2 and the direct
  output head unchanged for comparison.
- Verified the decoder filter in the browser at multiple character limits and
  across enable/disable transitions. The full project baseline is now 103
  passing tests plus clean Ruff, JavaScript syntax, lockfile, shell syntax, and
  whitespace checks.
- Added analysis-time encoder overlap selection. Real residual pooling now uses
  200 ms windows with a default 20 ms overlap/180 ms hop, reports exact and
  effective geometry, retains a zero-overlap control, and adapts safely for long
  clips.
- Increased the default encoder display window from 100 ms to 200 ms after
  interactive review. The 20 ms overlap remains, producing a 180 ms hop; the
  zero-overlap comparison now uses a 200 ms hop.
- Began M5 with a reusable causal-trace core and a Laurel/Yanny experiment
  plan. The source audio will remain a local upload until its redistribution
  terms are confirmed; the first result will include matched-norm and
  random-direction controls.
- Implemented and tested the causal runner, J-lens contrast direction,
  teacher-forced whole-candidate scoring, and reproducible matched-norm random
  control. A temporary local run of the original clip produced Tiny's baseline
  transcript `Lily!`; at encoder L1 / 0.18–0.38 s / strength 0.20, the
  Yanny-minus-Laurel normalized score contrast rose from 7.415 to 8.254, while
  one matched random control fell to 7.284. This is one exploratory trace, not
  validation; the free transcript remained `Lily!` and spectral variants plus
  a control distribution remain required.
- Added a clearly labeled static browser report for that fixed trace. It shows
  the candidate contrast, matched random control, unchanged free generation,
  and propagation through every remaining encoder/decoder layer and the output
  head without bundling the source audio.
- Extended the causal runner from one edit to ordered L1/L2/L3 schedules. With
  a total 20% budget at 0.18–0.38 s, the exploratory multi-layer run moved the
  contrast 7.415→8.645; its one random schedule also moved it to 8.129, so the
  separate `/causal` page records this as ambiguous rather than specific.
- Added recorded before/after candidate visualizations for `Lily!`, `Yay!`,
  `Yanny`, and `Laurel`: a restricted comparison over mean sequence scores and
  the actual first-BPE-token softmax. The page labels the distinction and
  preserves the shared first-piece caveat for Yay/Yanny.
- Added two target buttons to `/causal` with fixed 20/40/80/120% L1+L2+L3
  sweeps. The Yanny direction reached `Yay!`, never `Yanny`; the Laurel
  direction generated repetition/punctuation/unrelated text while Laurel
  remained weak. The page now exposes these negative results and the large
  Laurel-direction downstream changes.
- Added multi-slice schedules. An initial manually timed, overlapping
  ` Y` → `anny` run raised the restricted Yanny share from 3.17% to 7.48%
  versus 2.33% for one random control. The ` Y` → `an` → `ny` phoneme proxy
  reached 5.37% versus 2.14% random. A later audit found those multi-segment
  free-generation texts had applied each aggregated layer delta more than once;
  their candidate scores remain usable, but their generated text is excluded.
- Made tokenizer-faithful target-piece segmentation the causal CLI default.
  A standard selected span is now divided over the positive candidate's real
  Whisper BPE pieces; manual `--segment` is explicitly reserved for timing or
  phoneme-proxy overrides. The causal page labels the recorded BPE experiment
  as the default.
- Fixed multi-segment free generation so one aggregated delta is registered per
  encoder layer, matching candidate scoring and propagation. Reran canonical
  0.00–0.92 s, fixed-20%-budget component experiments: ` Y` only moves the
  restricted Yanny share 3.17%→9.99% and generates `Yay!`; `anny` only reaches
  3.43% and remains `Lily!`; both pieces reach 7.30% and generate `Yay!`.
  Added an interactive page view of raw `p(" Y")`, conditional `p("anny"|" Y")`,
  joint path probability, final candidates, propagation, and matched controls.
- Added decoder-residual causal schedules with cache-aware absolute-position
  hooks, per-token candidate scores, L0→L2 source edits, downstream L3 capture,
  open-loop free generation, component-piece selection, and matched random
  controls. Full and cached Hugging Face execution match under the same edits.
- Rebuilt the piece-impact view around the shared Yanny/Laurel selector and an
  Encoder/Decoder stream switch. Yanny renders its real ` Y`/`anny` path;
  Laurel correctly renders one full-width ` Laurel` token (43442). The page now
  exposes that encoder Laurel steering decreases actual token probability while
  increasing a restricted share by damaging alternatives, whereas decoder
  steering raises the token probability slightly but still generates `Lily!`.
- Added encoder vocabulary-prefix-family schedules and a separate browser mode.
  In the pinned tokenizer, `Y*` contains 114 ordinary tokens, `La*` 131, and
  `anny*` one. A 20% `Y*` edit raises actual `p(" Y")` to 52.90% and generates
  `Yay!`, while `Y*` then `anny*` still does not generate Yanny. The `La*` edit
  lowers actual Laurel probability to 0.000199% despite a larger restricted
  share, so the page records the result as broad prefix attraction/collateral
  damage rather than a successful target flip.
- Recomputed every causal candidate view from total candidate-token log
  probability and retained the former mean-token metric as secondary. The
  `La*` prefix-family run now correctly shows Laurel at 61.35% of the restricted
  total-path set alongside its much smaller 0.000199% absolute path likelihood
  and unrelated free generation. Added an explicit total-primary trace schema,
  regenerated the 20/40/80/120% sweeps, and updated random controls.
- Redesigned both localhost pages around a shared light, minimalist workbench:
  unified navigation, lighter surfaces, flatter diagnostic groups, readable
  waveforms, a primary token-path comparison, collapsible provenance and legacy
  traces, and fewer nested cards without changing explorer interactions.
- Opened M6 for Hugging Face hosting, added Docker Space card metadata, and
  documented local container checks, private-artifact delivery, hardware and
  sleep-time choices, Pro Protected visibility, custom-domain/iframe setup, and
  cloud upload/privacy boundaries.
- Added a non-root Docker runtime with locked dependencies, FFmpeg/browser-audio
  support, port 7860, a build-cached pinned Whisper snapshot, explicit
  demo/model-backed startup, and private Hub lens download support. Added three
  entrypoint regressions, bringing the suite to 101 passing tests. A real local
  CPU entrypoint run reported both lens streams ready and analyzed the bundled
  `question.flac` sample through HTTP in 1.25 seconds (472,799-byte response).
  A local Docker build was unavailable because no daemon is installed.
- Created the Protected `kennethli319/audio-jacobian-lens` Space on CPU Basic
  and the private `audio-jacobian-lens-artifacts` repository. Pinned the Space
  to artifact commit `baf116d4f1dfb695b7c5e6ba18e8ed6b338ce696`, completed
  the remote Linux build, and verified both web pages plus model-backed status.
  A same-origin external upload of `question.flac` completed in 4.44 seconds,
  transcribed “Where is my brother now?”, and returned four encoder layers,
  three decoder layers, and a 472,738-byte response. Custom-domain and
  microphone-in-embed verification remain pending.
- Deployed the independent decoder L0/L1 character-length filter to Space
  revision `8e5ac09c4fe41cedc25cba358ef47af0c9f2ae98`. A fresh model-backed
  `question.flac` request confirmed exact-length buckets on every L0/L1 cell,
  none on L2 or output tokens, and completed in 4.22 seconds with a 558,169-byte
  response. The hosted browser check confirmed `≤ 3` reranking changes L0/L1
  while leaving L2 and the LM/output-head card unchanged.
- Added the optional Apple-silicon MLX stack pinned to MLX 0.32.0, MLX-Audio
  0.4.5, MLX-LM 0.31.3, Transformers 5.12.1, and the immutable 8-bit LFM2.5
  Audio checkpoint revision.
- Implemented deterministic interleaved speech generation, cache-free replay,
  capture of all LFM language blocks, the tied text head, a native projected-VJP
  fitter, safe factorized artifacts, a fit CLI, and a serialized local backend
  that returns language readouts plus generated-speech playback.
- Replaced the exploratory rank-64 Rademacher smoke setup after its poor lexical
  audit with a seeded subsampled-Hadamard estimator and rank-512 minimum pilot
  default. A multi-clip rank/seed evaluation, FastConformer transport, and audio
  codebook/codec attribution remain open M7 work.
- Documented the MLX install, fit, and serve workflow plus its architecture,
  projection, quantization, license, and interpretation boundaries in
  `docs/MLX_LFM.md`.
- Fitted the one-clip rank-512 Hadamard integration artifact at seed 29 across
  L0/L4/L8/L12/L14 in 147.90 seconds. The local artifact is 12,588,293 bytes
  and remains ignored rather than distributed as scientific evidence.
- Verified the fresh rank-512 localhost backend with a same-origin
  `question.flac` request: HTTP 200 in 1.58 seconds, 174,214 bytes, generated
  text "I'm sorry, but I," and 0.96 seconds of mono 24 kHz speech. Browser QA
  confirmed synchronized output-token/language-layer selection, the five
  projected layers, hidden unsupported encoder/filter controls, explicit
  projection and text-head warnings, a playable data URL, and no console
  errors.
- Final regression gate after the MLX slice: 115 tests passed, one optional
  test skipped in the base environment, 13 MLX/projected focused tests passed
  in the pinned MLX environment, Ruff and JavaScript syntax checks passed, and
  the dependency lock remained current.
- Opened M8 for a separate MLX Chatterbox frame-to-text explorer. Pinned the
  350M Turbo 8-bit conversion and S3TokenizerV2, implemented deterministic
  greedy code generation, decoded 24 kHz audio, captured every T3 block, and
  required full teacher-forced replay before making a run inspectable.
- Added exact on-demand MLX VJPs for selected speech-code raw log-probability at
  L0/L4/L8/L12/L16/L20/L22, plus reconstructed causal self-attention. The API
  returns raw gradient norms, within-text shares, text-versus-prefix gradient
  mass, within-text attention share, and total attention-to-text mass as
  deliberately separate quantities.
- Added `/chatterbox` with generated waveform playback, proportional 40 ms code
  slices, mel-coordinate display, synchronized input-text and layer views,
  accessible keyboard controls with settled-slice trace debouncing, bounded
  client/server caches, strict response validation, a clearly fabricated
  synthetic demo, and method/provenance warnings. The dedicated local server
  remains loopback-only by default.
- Real pinned-model HTTP smoke on the M2 Pro generated 25 codes/1.1175 seconds
  of audio for “Hello world.” in 0.844 seconds after warmup, reproduced cached
  logits within `8.06e-05`, and traced seven layers in 1.049 seconds with finite
  gradients. A second browser run over 57 codes verified slice-to-code,
  mel-frame, text-token, and layer synchronization, no horizontal overflow,
  and no console errors.
- Expanded the regression suite to 147 passing tests plus one optional skip,
  including Chatterbox schema/configuration, stable payloads, backend locking
  and LRU eviction, API status/generate/trace behavior, same-origin protection,
  error mapping, page semantics, and navigation. Ruff, JavaScript syntax,
  lockfile, and whitespace checks remain clean.
- Documented exact installation/runtime pins, method equations, security,
  provenance/licenses, lack of an MLX PerTh watermark claim, and the current
  scientific boundary in `docs/CHATTERBOX.md`. Deterministic S3 frame-to-code
  VJPs and two-stage acoustic composition remain the next M8 implementation.
- Added visible three-part “apply / imply / use” guides for Whisper ASR,
  Chatterbox TTS, and MLX LFM speech-to-speech. The shared Explorer guide now
  switches from decoder-plus-cross-modal ASR copy to projected-language-only
  speech copy using backend family and capability metadata.
- Tightened the speech-to-speech UI boundary: input-waveform seeking no longer
  claims to update an absent encoder, generated speech is labeled playback only,
  Whisper-only filter/timeline controls hide before analysis, and response-token
  selection names the language backbone and tied text head.
- Reworded Chatterbox from “trace sound back to text” to local speech-code
  sensitivity, added a control-specific four-step guide, and removed a stray
  synthetic `CFG` score label even though Turbo has no T3 CFG path.
- Restored the real MLX LFM server on port 8001 alongside Whisper on 8000 and
  Chatterbox on 8002. Browser QA exercised a real prepared sample in both ASR
  and speech-to-speech plus the TTS demo, verified backend-specific controls and
  synchronization labels, and found no console errors or horizontal overflow.
- Final guide regression gate: 147 tests passed with one optional skip; Ruff,
  all three JavaScript syntax checks, dependency lock, and whitespace checks
  passed.
- Completed a cross-page consistency cycle for Whisper ASR, LFM
  speech-to-speech, and Chatterbox TTS. Added explicit localhost workspace
  navigation, a dedicated untimed LFM synthetic fixture, shared section
  numbering and output-head terminology, and cross-page analysis-type metadata.
- Reorganized Chatterbox into input → analysis/provenance → output → text
  context → layer comparison; added model/tokenizer/runtime/replay provenance,
  orange speech-head probabilities, blue direct-mapped gradient shares, violet
  direct-mapped attention shares, precise residual/attention coordinates,
  explicit metric labels, selection state, keyboard navigation, and responsive
  status visibility.
- Added the cross-page metric glossary and Chatterbox denominator/coordinate
  caveats to the methodology documents, plus static regression coverage for
  workspace navigation, mode-specific demos, numbering, provenance, and
  semantic metric hooks. Browser QA confirmed all three demo reports preserve
  their architecture-specific modes and the TTS page has no horizontal
  overflow at the 1280 px test viewport.
- Final consistency-cycle gate: 150 tests passed with one optional skip; Ruff,
  all three JavaScript syntax checks, and whitespace validation passed.
- Replaced Chatterbox's T3 layer summary cards with paired layer × tokenizer
  matrices. Every gradient-share and attention-share coordinate is now visible
  with its numeric percentage and direct 0–100% blue/violet fill; the rightmost
  column preserves text-versus-prefix mass. Matrix-cell selection and 2D arrow
  navigation synchronize the layer dropdown, token strip, both matrix outlines,
  and inspector. Long prompts are split into adaptive token bands, retaining all
  coordinates with no page-level horizontal overflow.
- Final matrix-cycle gate: 152 tests passed with one optional skip; Ruff, all
  JavaScript syntax checks, and whitespace validation passed.
- Added the genuine fitted Chatterbox T3 speech-position lens alongside the
  existing local text trace. The fitter uses one joint MLX VJP per Hadamard
  probe for all seven source layers, preserves signed responses until corpus
  averaging, fingerprints the checkpoint/tokenizer/built-in voice/corpus, and
  decodes transported residuals through the real final norm and speech head.
- Retained `chatterbox_turbo_10prompt_rank128_seed29.pt`: ten fit prompts, 480
  selected speech positions, rank 128/1024, seed 29, 391.86 projection seconds,
  2,104,061 bytes, model fingerprint `47f1c6108840fae0`, artifact fingerprint
  `67da0e6fef27e310`, and examples fingerprint
  `9de6342be062cdf4efdf53ae01570cf47f96d0790cb6aa1093e253e4e03bedf7`.
- On four disjoint held-out prompts/251 generated codes, median realized-code
  rank progressed L0→L22 as 1,593, 1,376, 952, 466, 187, 23, and 12; top-10
  recovery progressed 0.4%, 0.4%, 2.0%, 6.4%, 9.6%, 32.3%, and 45.8%.
  This is a positive integration pilot, not validation: rank/seed, corpus-size,
  direct-logit, shuffled, and unquantized controls remain open.
- Added the full fitted layer × speech-position matrix with exact probabilities,
  global ranks, top codes, a synchronized orange actual-head row, provenance,
  responsive position bands, and keyboard/click/waveform selection. A real
  held-out browser run rendered 483 fitted cells plus both 63-cell local text
  matrices, synchronized L22/S11 with waveform code 11, had zero page overflow,
  and emitted no browser console warnings or errors.
- Final fitted-Chatterbox gate: 160 standard tests passed with three optional
  skips, all 17 Chatterbox core/backend tests passed in the pinned real MLX
  environment, Ruff and all JavaScript syntax checks passed, the dependency
  lock resolved unchanged, whitespace checks passed, and the restarted port
  8002 backend reported the rank-128 ten-example artifact ready.
- Condensed the fitted Chatterbox matrix into seven uninterrupted blue layer
  strips plus one orange actual-head strip. All 72 synthetic-demo positions
  reused the waveform's exact normalized left edge and width over the full
  three-second decoded duration; the final 4% stayed hatched as trailing audio.
  Cell, waveform, head-row, focus-readout, click, and arrow-key selection remain
  synchronized, with exact probabilities/ranks in the focus and accessible
  labels rather than printed inside every narrow slice.
- Final compact-timeline gate: browser QA found 504 fitted cells plus 72 head
  cells, a 208 px seven-layer chart, exact geometry agreement on sampled slices,
  zero page overflow at 1,280 px and a 360 px mobile viewport, and no console
  warnings or errors. The full suite passed 162 tests with three optional skips;
  Ruff, all JavaScript syntax checks, the dependency lock, and whitespace checks
  passed, and port 8002 still reported the retained fitted artifact ready.
- Added position-aligned actual speech-head candidates to every Chatterbox run:
  full 6,563-entry raw-logit softmax and global realized-code rank are computed
  before generation processors, while only deterministic top-k IDs/probabilities
  are serialized. Start/stop candidates retain checkpoint-derived control labels,
  and the backend advertises this capability independently of the fitted lens.
- Added the synchronized selected-position candidate inspector below the compact
  timeline. It renders all seven fitted layers plus actual HEAD, uses absolute
  probability fills, keeps the realized code visible outside top-k, highlights
  the focused layer, and explicitly identifies every entry as an acoustic code
  rather than a word or phoneme. The synthetic fixture now uses internally
  consistent bounded candidate distributions.
- Final candidate-inspector gate: a live “Hello world.” run generated 25 codes
  and rendered the expected seven fitted rows plus HEAD; the first actual row
  matched the server's top-five IDs/probabilities and realized rank. Waveform,
  strip, candidate, and keyboard layer selection stayed synchronized. Desktop
  and 360 px mobile checks had zero horizontal overflow and no console warnings
  or errors. The full suite passed 171 tests with three optional skips; Ruff,
  JavaScript syntax, lockfile, and whitespace checks passed, and port 8002 was
  restarted with the retained fitted artifact and top-k five candidates ready.
- Added a two-step candidate counterfactual workflow to the Chatterbox page.
  Clicking a non-realized ordinary acoustic-code candidate only selects it;
  the separate branch action then preserves the earlier parent codes, forces
  that ID at the actual T3 output decision, regenerates every later code, and
  decodes a new waveform. The comparison retains both audio paths, refocuses
  the divergence position, and reports parent raw-head top-1, replacement
  probability/global rank, required unique-winner logit bias, source row, and
  parent/branch analysis IDs without calling the operation residual steering.
- Verified the branch through the pinned real MLX model and local API. A
  `Hello world.` parent branch at S6 changed ID 4726 to the raw-head rank-2 ID
  5006, preserved S1–S5, diverged first at S6, changed 13 of the 25 overlapping
  positions, and produced different decoded audio; the recorded minimum
  unique-winner bias was `+0.158247` logits. Browser QA separately selected an
  L0 fitted candidate, created a new branch, retained both playable audio paths,
  refocused the changed position, and showed the parent raw-head provenance.
  Desktop and 360 px checks had no page overflow or browser warnings/errors.
  The final gate passed 197 tests with three optional skips, Ruff, all four
  JavaScript syntax checks, lock validation, and whitespace validation.

### 2026-07-11

- Added a separate Chatterbox residual-steering API and runtime. For every
  requested layer×speech-position coordinate it computes the parent-path
  gradient of the target-versus-strongest-other raw-head margin, scales that
  direction against the coordinate's parent residual norm, automatically
  searches a bounded shared strength, and regenerates without directly forcing
  any speech code. Full replay reapplies the exact open-loop schedule and stores
  post-block states after their edits; fitted and HEAD arbitrary-target
  diagnostics therefore describe the actual steered branch.
- Added distinct `Force output code` and `Steer residual` modes to the local TTS
  page. Residual mode exposes sampled multi-layer checkboxes, a synchronized
  start position, a one-to-eight-position forward window with nominal time, and
  a `0.5×` default / `2.0×` hard norm cap. Its compact matrix shows exact
  target probability and global rank before/after across all seven fitted
  layers plus HEAD, marks directly edited coordinates and changed realized
  codes, retains original and steered audio, and reports calibration success or
  budget exhaustion without conflating the operation with fitted-lens causality.
- Verified a real pinned-model run for `Residual steering test.`. At S4, fitted
  L22 nominated non-realized ID 4124 while the raw HEAD ranked it second at
  `10.225%`. Applying the parent-path directions at L20 and L22 over S4–S6
  succeeded after six attempts at only `0.002197×` relative norm: HEAD rank
  moved to first at `10.871%`, processed greedy generation emitted ID 4124, and
  the suffix first diverged at S4. A separate rank-1/repetition-penalty stress
  case required `1.6875×`, validating the distinct processed-token criterion
  and the visible off-manifold warning for large budgets.
- Final residual-steering gate: 238 tests passed with three optional skips;
  Ruff, all four JavaScript syntax checks, dependency-lock validation, and
  whitespace checks passed. Desktop browser QA rendered all 24 requested
  layer×position cells (seven fitted rows plus HEAD over three positions),
  marked six edited L20/L22 coordinates, retained different original/steered
  audio, and emitted no browser warnings/errors. The 360 px check had no page
  overflow, and port 8002 remains ready with residual steering enabled.
- Replaced the Whisper selected-position layer cards with full-duration compact
  strips: four overlapping encoder rows, three timed decoder rows, and one
  orange actual LM-head row. Every slice is a native button; click and two-axis
  keyboard navigation pin one shared audio/token coordinate, while hover is a
  non-destructive preview in the fixed candidate inspector.
- Extended ASR analysis responses with exact competition-rank provenance for
  the active bucket, lexical display vocabulary, and full model vocabulary;
  denominators, tie policy, score kind, token ID, filter state, position, and
  timing source are serialized with each bounded candidate. HEAD candidates
  additionally expose exact teacher-forced probability and log probability.
  Grouped full/display ranks are computed once per bounded position chunk, then
  gathered for every exact-length bucket rather than repeating full-vocabulary
  comparisons for every group.
- Browser-verified the live `Where is my brother now?` Whisper run: 48 encoder
  cells, 18 decoder cells, and six HEAD cells used the 2.04-second waveform
  geometry; the 200 ms windows and 180 ms hop rendered at 9.80392% width and
  8.82353% offset. The active `≤ 2` encoder filter reported exact ranks out of
  2,812 eligible lexical tokens, while HEAD reported ranks out of the unfiltered
  51,864-token model vocabulary. Waveform, transcript, encoder, decoder, HEAD,
  pointer selection, and arrow-key movement stayed synchronized. Desktop and
  360 px responsive checks had zero horizontal overflow and no browser console
  warnings or errors.
- Final ASR compact-timeline gate: 248 tests passed with three optional skips;
  Ruff, all four JavaScript syntax checks, and whitespace checks passed. Port
  8000 was restarted with the pinned Whisper Tiny English model and both fitted
  streams ready.
- Added a cell-local ASR timeline tooltip so hover detail no longer requires an
  eye movement to the sticky inspector. It shows the active layer/coordinate,
  token filter, and top three candidates with token ID, exact scoped
  rank/denominator, and raw delta/logit; the orange HEAD variant shows actual
  probability and log probability. Hover is transient and leaves the pinned
  inspector untouched, while focus/click retains synchronized selection. Touch,
  Escape, outside press, scroll, and resize have explicit dismissal behavior.
- Browser QA confirmed the synthetic encoder tooltip at 318 px on desktop and
  267 px at a 360 px viewport, plus a readable two-row candidate layout at the
  320 px minimum. The narrowest tooltip measured 227×387 px, remained inside
  both its chart and viewport, and the page had zero horizontal overflow.
  Encoder and HEAD variants each rendered three exact-ranked candidates;
  keyboard movement updated both local tooltip and pinned inspector, Escape
  dismissed only the tooltip, and the console remained clean.
- Final local-tooltip gate: 251 tests passed with three optional skips; Ruff,
  all four JavaScript syntax checks, and whitespace checks passed. Asset cache
  version `0.7.6-local-timeline-tooltip` is live on port 8000.
- Diagnosed the 0.96-second LFM response as exhaustion of the smoke artifact's
  18-position interleaved budget: the completed short text consumed six
  positions and only 12 acoustic frames remained. Added a serving-only
  `--lfm-max-new-tokens` control that defaults to 512, while loading and
  validating the lens with its original fitted generation policy before the
  override. `/api/status` now reports both budgets and whether an override is
  active; no lens fingerprint or fitting metadata is rewritten.
- Added regressions for validation-before-override, status provenance, the
  serving-cap default, custom positive budgets, and invalid zero budgets.
  The complete gate passed 257 tests with three dependency warnings, and Ruff
  passed for the changed Python files. The running MLX server was intentionally
  not restarted during this implementation.
- Added per-run MLX interleaved-generation stop diagnostics at the generation
  boundary: yielded step count, text-token count, usable acoustic-frame count,
  effective maximum, whether any audio EOS appeared, and an exact natural-EOS,
  budget-exhausted, or other-model-stop reason. Analysis payloads expose this
  as `metadata.generation_diagnostics` for the generated-speech UI and label
  the artifact and serving generation policies separately. Focused diagnostics,
  analysis-schema, and backend tests passed (13 total), with Ruff clean; the
  running MLX server remained untouched.
- Added a compact generated-speech stop notice and matching run-detail row. It
  distinguishes natural audio EOS from budget exhaustion and shows the shared
  interleaved step budget, text-token count, usable acoustic-frame count, and
  EOS observation rather than making a short WAV look like a browser failure.
  Updated the README and MLX guide to document the serving-only control and its
  separation from fitted-artifact provenance.
- Restarted the MLX service on port 8001 with the interim 90-position preview.
  A real `universe.flac` request generated 30 text tokens and 60 acoustic frames,
  returning exactly 4.80 seconds of mono 24 kHz audio and an explicit
  budget-exhausted status. Browser QA showed the same duration and `90 / 90`
  stop notice with no console warnings or errors. The combined final gate passed
  261 tests with three dependency warnings; Ruff and JavaScript syntax checks
  passed.
- Replaced the interim five-second preview with an EOS-first serving policy.
  The 512-position shared counter is now only an emergency ceiling; normal
  completion requires a final audio EOS after text has completed. An audio EOS
  ending an earlier segment no longer causes the UI or diagnostics to label a
  later capped/text-stopped run as naturally complete.
- Replayed the exact `question.flac` refusal case under the unchanged greedy
  artifact sampling policy. It ended naturally at step 155 with 22 text steps,
  132 usable acoustic frames, one final EOS, and 10.56 seconds of audio. A
  separate Whisper pass heard both displayed sentences exactly. The waveform
  contains a roughly four-second low-level pause before the second sentence, so
  silence-based early stopping would recreate the bug and is intentionally not
  used. `universe.flac` required 375 steps/26.32 seconds before EOS, confirming
  that 512 is a safety ceiling rather than a target duration.
- Final EOS-first gate: 262 tests passed with three dependency warnings; Ruff
  and JavaScript syntax checks passed. README, the MLX guide, CLI help, status
  provenance, and generated-speech stop messaging all describe the new policy.
- Opened M9 with an evidence-led static-example screen across all three model
  paths. Recorded the selection rules, exact trajectories, rights gates,
  controls, static bundle contract, and implementation backlog in
  `docs/STATIC_SHOWCASE_CURATION.md`, plus a machine-readable candidate manifest
  at `data/static_showcase_candidates.json`.
- Selected the Chatterbox bridge run as the canonical hero. Browser QA reproduced
  code ID 4106 at S9 moving from global rank 3,183 at L0 to rank 11 at L20 and
  rank 1 at L22, with the actual head also rank 1. The paired L20+L22 residual
  edit to ID 4358 uses only `0.001708984375` relative norm, flips the raw-head
  winner, and propagates through the suffix. The live page remained free of
  console warnings/errors.
- Screened natural, synthetic, homophone, silence, and failure ASR cases using
  exact full-vocabulary ranks. Kept the encoder-to-decoder pilot explicitly
  negative: real-speech median ranks remain in the thousands, while silence can
  appear spuriously strong. Selected rights-safe LibriSpeech decoder stories
  for early readability (`door`: 4→1→1→1), late emergence (`now`:
  6319→7237→3→1), BPE behavior, and model error; owner-recorded homophones remain
  pending.
- Marked the current one-clip LFM lens provisional. Screening found a useful
  `Hello` mid-layer trajectory and a strong `Four` failure, but neither should
  become the public positive claim before a multi-clip held-out fit. During the
  screen, fixed LFM text control IDs being counted as ordinary targets; live
  `Four` now exposes one analysis token while retaining two generated text steps
  and unchanged audio EOS behavior.
- Final curation gate: 263 tests passed with three dependency warnings; Ruff,
  JSON validation, JavaScript syntax, and whitespace checks passed. Port 8001
  was restarted with the corrected target selection.
- Replaced the Laurel/Yanny study as the public-facing fourth workspace with an
  evidence-led static Showcase. `/showcase` is canonical and `/causal` aliases
  the replacement, while direct `causal.html`, `causal.js`, and the causal trace
  documentation remain available as the archived experiment.
- Added selectable ASR early/late/subword/null-control examples, the canonical
  Chatterbox bridge readout and residual intervention, monotonic and
  non-monotonic TTS comparisons, and provisional LFM success/failure screens.
  Rank cells expose exact full-vocabulary values on pointer hover and keyboard
  focus; fitted readouts, the raw head, gradient sensitivity, attention, and
  causal replay remain visually and semantically separate.
- Added `web/data/showcase-examples.json` plus checksum, rights, generation,
  rank, and serialization validation for three CC BY 4.0 LFM input cases. The
  artifact deliberately excludes generated speech, embedded audio, macOS TTS,
  and ephemeral analysis IDs, and is packaged under the installed web data
  directory for later manifest-loader work.
- Showcase validation gate: 262 tests passed and 3 platform-dependent tests
  skipped; Ruff, JavaScript syntax, JSON parsing, and whitespace checks passed.
  Desktop interaction QA, a 390 px responsive pass, and browser console checks
  passed without horizontal overflow, warnings, or errors. Port 8000 was
  restarted and the completed Showcase was left open at `/showcase`.
- Reworked the repository README as the project front door. It now summarizes
  the Whisper ASR, LFM speech-to-speech, Chatterbox TTS, and static Showcase
  tracks; states the positive, negative, and provisional evidence boundaries;
  links the M0–M9 plan, methodology, pilot, architecture, intervention,
  curation, and licensing documents; corrects the fitted-probability metric
  language; records the shared local ports; and replaces the stale fixed test
  count with the evergreen release commands.
- Generated and published the first static personal-site review bundle. The
  ASR, speech-to-speech, and TTS pages each expose exactly three cached reports
  through one schema-versioned payload; all rank cells retain their exact
  per-layer denominators and score semantics, the TTS bridge retains the
  recorded residual intervention, and only CC BY 4.0 LibriSpeech input audio is
  present. The public route is deliberately absent from the personal homepage
  and marked `noindex,nofollow` until owner review.
- Added `data/static_public_reports_v1.json` as the source-of-truth snapshot and
  a focused integrity test covering the 3×3 contract, rank bounds, sample
  hashes, rights scope, intervention type, and absence of ephemeral IDs or
  embedded/generated audio. The GitHub Pages build for personal-site commit
  `f871bbf` completed successfully, and all three public routes, the JSON
  payload, and cleared source audio returned HTTP 200.
- Extended that review with three detailed backend-free routes for ASR,
  speech-to-speech, and TTS. Each route selects among three already-inferred
  examples and exposes the complete saved layer × position matrix, local
  candidate ranks/scores, a pinned inspector, keyboard/pointer tooltips, and
  immutable model/lens provenance through one shared renderer.
- Preserved the live ASR character-length experiment exactly enough for static
  replay: all encoder layers and decoder L0–L1 have exact-length top-k buckets,
  stored as compact 3.2–7.1 MB sidecars that load only when the filter is
  enabled. The default reports remain 0.4–0.8 MB and L2/HEAD remain unfiltered
  controls. Speech-to-speech visibly records that this control is unavailable.
- Cached every Chatterbox speech-code position and its local text trace for the
  bridge (63), turtles (69), and music (59) prompts. The TTS page compares all
  seven fitted layers with the actual HEAD, renders gradient and attention
  shares across prompt tokens, and retains the recorded S9 residual/direct-force
  comparison without exposing a live branch control.
- Added deterministic ASR/LFM and TTS exporters, 15 focused exporter tests, and
  a static-site integrity gate. The published bundle contains only the three
  cleared LibriSpeech FLAC inputs; generated LFM/Chatterbox audio, embedded
  audio URIs, waveform outputs, and ephemeral analysis IDs are absent. Local
  route/cache replay and shared-renderer data-shape smokes passed for all three
  families; the personal homepage remains unchanged pending owner review.
- Promoted the complete cached explorers to the canonical public family routes:
  ASR at the review root, speech-to-speech at `/speech/`, and TTS at `/tts/`.
  Moved the shorter experiment narratives under `/findings/`, kept the earlier
  `/explorer/{family}/` pages as functional renderer aliases that preserve
  query-selected samples, and left the personal homepage unlinked.
- Reworked the long LFM response matrix into aligned position bands. Each band
  shows readable token text above repeated layer rows while preserving every
  original top-token cell and its click, focus, rank, score, and alternatives.
  Extended the static validation contract and focused tests to enforce the
  canonical explorers, secondary findings pages, functional aliases, renderer
  separation, `noindex` policy, and query-selection support.
