# Static showcase example curation

Last updated: 2026-07-11

This document freezes the first evidence-based example screen for a future
static Audio Jacobian Lens site. The goal is not to collect outputs that merely
look fluent. Each example must teach one specific fact about an intermediate
readout, an actual output head, a limitation, or a causal intervention.

## Selection rules

1. State the teaching purpose before packaging the example.
2. Preserve exact global ranks and their vocabulary denominators. Do not infer
   an unreported rank from a bounded top-k list.
3. Keep fitted J-lens readouts, raw output-head probabilities, local gradients,
   attention, and interventions visually and semantically separate.
4. Include at least one failure or null control beside positive examples.
5. Prefer held-out examples. Label any fit-corpus or near-fit example as
   in-sample integration evidence rather than scientific validation.
6. Freeze model, artifact, tokenizer, runtime, source-audio, and payload hashes.
7. Do not ship audio until its redistribution terms and attribution record are
   complete.

## Curated findings versus detailed explorers

The three examples per family selected below remain the compact findings set:
they are deliberately chosen to teach complementary positive, negative, and
intervention stories. The detailed cached explorers use the separate
`data/static_explorer_catalog_v2.json` contract and expose ten reports per
family. Expanding that browsing corpus does not turn all ten cases into curated
findings or change the claims attached to the original three.

The ten ASR and speech-to-speech reports share the same ordered, attributed
LibriSpeech inputs. The ten TTS reports use project-authored text prompts and
ship no generated audio. Every manifest remains static and hash-pinned; adding
examples does not add an inference, upload, generation, or steering endpoint.

## Recommended public sequence

### 1. ASR: a token is readable early

Candidate natural clip: LibriSpeech `1272-135031-0003`.

Reference: “The little girl had been asleep, but she heard the raps and opened
the door.” Whisper generates “wraps” rather than “raps,” but correctly emits
`door`. For the realized `door` token, exact full-vocabulary decoder rank moves

```text
L0  #4  →  L1  #1  →  L2  #1  →  raw head  #1
```

This is the clearest rights-safe ASR illustration that the final lexical
direction is already readable from an early decoder residual. The incorrect
`raps`/`wraps` homophone should remain visible so the example does not imply
perfect transcription.

### 2. ASR: late emergence is also real

Candidate natural clip: bundled LibriSpeech `1272-135031-0012`, “Where is my
brother now?” Exact realized-token decoder trajectories include:

```text
Where     523 → 547 → 15 → 1
brother   627 → 454 → 26 → 1
now      6319 → 7237 →  3 → 1
```

This counters the simplistic claim that every emitted word is already explicit
at the first layer. `now` becomes sharply readable only at L2.

### 3. ASR: subwords and failure controls

Use LibriSpeech `1272-141231-0004`, “One minute, a voice said, and the time
buzzer sounded,” to explain BPE pieces. The generated `buzz` piece follows
`3966 → 651 → 981 → 1`, while suffix `er` follows `23 → 2 → 1 → 1`.

Retain the bundled buzzer/whirr clip as a model-error example. The lens follows
the model's wrong generated path; it is not a reference-transcript oracle.
Also add project-generated three-second digital silence. In screening, silence
hallucinated `you` and produced deceptively strong encoder ranks. That is an
important null control for the still-negative encoder-to-decoder pilot.

An owner-recorded flour/flower pair remains desirable:

```text
She carried the flour into the kitchen.
She picked the flower from the garden.
```

Screening with local TTS found that realized `flower` ranked
`15309 → 11452 → 229 → 1`, while the wrong homophone `flour` ranked better at
every fitted decoder layer. Re-record these lines under an explicit CC0 or
CC BY 4.0 release before treating the result as a publishable example; rerun all
numbers because speaker changes can alter the trajectory.

## Whisper encoder boundary

Do not use the current encoder lens as the positive hero. Across screened real
speech, median realized-token ranks remained in the thousands. Its timing also
uses Whisper-derived DTW and the encoder is bidirectional, so it cannot support
a real-time “the model knew this before hearing it” claim. Present it as an
experimental/negative panel until an independently aligned, larger-corpus lens
passes controls.

## Canonical TTS hero

Prompt: “A bright red train crossed the narrow bridge.”

Use zero-based speech-code index 8, displayed as `S9`, nominally
`0.32–0.36 s`. The realized acoustic code is ID `4106`.

| Readout | L0 | L4 | L8 | L12 | L16 | L20 | L22 | HEAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global rank | 3183 | 3137 | 1892 | 946 | 396 | 11 | 1 | 1 |
| Probability | 0.0016% | 0.0016% | 0.0053% | 0.0183% | 0.0453% | 1.915% | 5.514% | 13.959% |

This is the strongest single static story because it combines three different
questions without conflating their answers:

- **Fitted readout:** the realized code becomes progressively readable and is
  rank 1 by L22.
- **Local text sensitivity:** early gradient share peaks on ` bright` (L0
  28.5%, L4 25.4%); attention later concentrates on ` red` (L8 51.0%). These
  remain association/sensitivity diagnostics, not word-to-waveform alignment.
- **Causal intervention:** raw-head runner-up ID `4358` begins at rank 2 and
  12.639%, only `0.0993` logits behind ID `4106`. Residual steering at L20+L22
  needs just `0.001708984375` relative residual norm per edited coordinate to
  make ID `4358` rank 1 at 13.170% and emit it. The changed autoregressive suffix
  alters 43 later same-index codes; the result exactly matches direct code
  forcing for this run.

Package both baseline and steered WAVs. State that an acoustic-code ID has no
word or phoneme label and that downstream decoding is contextual.

Useful secondary TTS cases:

- “Tiny turtles travel together toward the tide.”: nearly monotonic realized
  code ranks `4149 → 3551 → 1658 → 176 → 18 → 2 → 1`.
- “Music fades as the evening grows quiet.”: non-monotonic ranks
  `5197 → 5262 → 3809 → 4392 → 1443 → 6 → 1`; this prevents the UI from
  implying that emergence must improve at every layer.

Across 378 screened interior positions from seven held-out/original prompts,
median fitted rank improved from `1785.5` at L0 to `12` at L22. Only 6.6% were
rank 1 at L22. The hero is therefore a strong illustration, not a representative
claim about every code position.

## Speech-to-speech: provisional only

The current LFM artifact is a one-clip integration fit, so it should not carry
the static site's primary scientific claim.

Best screening case: spoken “Say hello in one word.” The model replies “Hello!
How can I help you today?” and ends naturally at audio EOS. Realized lexical
readouts include:

```text
!       21679 → 2803 →     4 → 1 → 1
 How     6025 → 6216 →    34 → 1 → 1
Hello   28728 → 38641 → 13739 → 5 → 1
         L0      L4       L8  L12 L14
```

This is a useful mid-layer-emergence and instruction-following-failure example,
but its local macOS voice input is screening-only pending output-rights review.

Concise negative case: “What is two plus two? Answer with one word.” The actual
head emits `Four` at rank 1 and 66.6%, while the fitted lens still ranks it
`23420 → 41719 → 31297 → 2197 → 141`. This honestly demonstrates that the
one-clip projected lens can fail even when the model output is clear.

Before publication, fit and evaluate the planned multi-clip rank-512 LFM lens
on disjoint held-out inputs. The bundled `question.flac` is in the fitted
artifact's source example and may be shown only as an explicitly in-sample
integration case.

## Rights and provenance

- OpenSLR declares LibriSpeech SLR12 as CC BY 4.0. Ship utterance ID, reference
  transcript, source and license links, Panayotov et al. attribution, file hash,
  and any modification notice. See <https://www.openslr.org/12> and
  <https://creativecommons.org/licenses/by/4.0/>.
- Do not publish macOS `say` screening audio without a separate output-rights
  determination. Prefer owner/consenting-speaker recordings released as CC0 or
  CC BY 4.0.
- Chatterbox example text is project-authored and the original model card
  declares MIT, but complete the existing conversion/S3-tokenizer and derived
  audio review before packaging output. Label the audio synthetic and preserve
  model attribution.
- Complete the LFM Open License and derived-output review before packaging LFM
  speech.

## Static bundle contract

Each example directory should contain:

1. a compact immutable manifest with a stable example ID and teaching purpose;
2. input and/or output audio plus SHA-256 and source/license fields;
3. a reduced analysis payload with exact ranks, denominators, raw values,
   selected top-k entries, model/lens fingerprints, and warnings;
4. any trace or intervention payload, with ephemeral server `analysis_id`
   removed and parent/branch relationships replaced by stable IDs;
5. baseline and counterfactual audio as separate assets; and
6. a short “what to inspect / what this does not prove” annotation.

Do not bake only the selected coordinate. Preserve the full layer/position
matrix so visitors can inspect neighboring slices and discover counterexamples.

## Next implementation steps

- [x] Add deterministic, allowlist-based exporters for reduced static ASR,
  LFM, Chatterbox generation, and all-position Chatterbox trace payloads.
- [x] Add a shared static renderer for the three detailed explorer routes, with
  sample selection, synchronized positions, full cached matrices, local
  candidate tooltips, and pinned provenance.
- [x] Package ten rights-cleared LibriSpeech inputs for the detailed explorers,
  while retaining three examples per family in the curated findings bundle.
  Procedural silence remains a future null-control addition.
- [x] Add a static integrity validator covering hashes, schemas, matrix/trace
  completeness, media scope, no-index pages, and absence of live API calls.
- [ ] Obtain and rerun rights-cleared flour/flower recordings.
- [ ] Package the Chatterbox bridge baseline, trace, residual branch, and both
  WAVs after the derived-output review.
- [ ] Fit/evaluate a multi-clip LFM artifact before selecting a positive public
  speech-to-speech hero.
- [ ] Add pooled summary context beside every selected hero to disclose how
  unusual the highlighted trajectory is.
