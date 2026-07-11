# Pilot Results: Whisper Tiny English

Date: 2026-07-10

This is a plumbing and methodology pilot, not a scientific result. It proves
that fitting, saved artifacts, held-out evaluation, and the localhost explorer
operate on a real Whisper model. It also exposed a cross-stream centering error
whose correction materially weakened the encoder result.

## Setup

- Model: `openai/whisper-tiny.en`
- Hub revision: `87c7102498dcde7456f24cfd30239ca606ed9063`
- Artifact model fingerprint: `5a77767bfe9c962d`
- Architecture: 4 encoder blocks, 4 decoder blocks, width 384, vocabulary 51,864
- Hardware: Apple M2 Pro, 16 GB unified memory; macOS 26.5.1
- Runtime: Python 3.12, PyTorch 2.13.0, Transformers 5.13.0, NumPy 2.5.1,
  SciPy 1.18.0, SoundFile 0.14.0
- Device: MPS; fp32 model/gradients and CPU accumulation, fp16 saved lens
- Audio preprocessing: mono 16 kHz with polyphase band-limited resampling
- Corpus: the first 10 of 16 macOS `say` clips, voices Samantha and Moira at
  175 words/minute, approximately 2.5–3.1 seconds each
- Corpus fingerprint: `46640e2cbca442f599ef`
- Held out: final 6 clips, yielding 50 generated lexical token positions
- Decoder estimator: causal all-target sum, blocks 0–2 to block 3, dimension
  batch 8, reference-transcript teacher forcing
- Encoder estimators: global all-audio/all-target, and one generated token per
  clip with its Whisper cross-attention/DTW window plus 100 ms padding;
  dimension batch 4
- Encoder application: affine transport
  `J(h - mean_encoder) + mean_decoder_target`

Fit and evaluation clips were disjoint, and the evaluator now rejects exact
fit/evaluation overlap by default. Rank targets are Whisper's generated tokens,
not the references. Normalized ASR metrics against the references were WER
10.6% (5/47 words) and CER 3.8% (11/287 characters).

## Exact reproduction commands

```bash
.venv/bin/python scripts/make_macos_tts_corpus.py \
  --output-dir artifacts/tts_corpus --count 16

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-fit \
  artifacts/tts_corpus/manifest.jsonl \
  --output artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --limit 10 --encoder-estimator aligned

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-eval \
  artifacts/tts_corpus/manifest.jsonl \
  --lens artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --start 10 --limit 6 \
  --output artifacts/pilot/eval_aligned_affine.json
```

The current aligned artifact is 2.0 MB and has SHA-256
`489582ce50736e8e7b78b7cb9d0140b963d99c5d7784635ed93e71710ab293a2`.
Artifacts and generated speech remain ignored by Git.

## Runtime

- One-example decoder smoke fit: 3.8 seconds
- One-example global encoder smoke fit: 12.8 seconds
- Current 10-example aligned combined refit: 2 minutes 20 seconds
- Current 10-example global encoder-only refit: 1 minute 37 seconds
- Held-out analysis/evaluation: roughly 1–3 seconds per short clip after load

The decoder path detaches and reuses the 1,500-position encoder result. Encoder
fitting must retain the audio graph across each dimension-batched backward pass.
Peak memory and corpus-size convergence have not yet been measured.

## Decoder result

Full-vocabulary ranks of each generated token on the six held-out clips:

| Readout | Median rank | MRR | Top-10 |
|---|---:|---:|---:|
| Direct unembedding, decoder L0 | 7,678 | 0.009 | 4.0% |
| Jacobian lens, decoder L0 | 1,037 | 0.078 | 14.0% |
| Direct unembedding, decoder L1 | 7,520 | 0.021 | 8.0% |
| Jacobian lens, decoder L1 | 1,075 | 0.077 | 18.0% |
| Direct unembedding, decoder L2 | 9 | 0.360 | 52.0% |
| Jacobian lens, decoder L2 | 8 | 0.350 | 54.0% |
| Generated-token final-logit self-check | 1 | 1.000 | 100% |

The decoder J-lens substantially improves early-layer coordinate alignment over
direct unembedding and converges with it near the target block. This is
generated-token readout fidelity, not evidence that the J-lens is a superior
next-token predictor. The final-logit row is tautological: greedy-generated
tokens are scored against their own teacher-forced base-model logits.

## Encoder result after affine correction

Encoder and decoder residual streams do not share a meaningful zero. The first
pilot incorrectly applied the rectangular map as `Jh`; its apparently localized
lexical results are discarded. Refit artifacts store both activation means and
apply the affine first-order approximation.

Absolute affine ranks remain dominated by the corpus target-activation mean:

| Lens / layer | Aligned median | Aligned MRR | Aligned top-10 | Remote MRR |
|---|---:|---:|---:|---:|
| DTW-aligned L0 | 4,181 | 0.0156 | 2.0% | 0.0156 |
| DTW-aligned L1 | 3,978 | 0.0156 | 2.0% | 0.0156 |
| DTW-aligned L2 | 4,296 | 0.0157 | 2.0% | 0.0156 |
| DTW-aligned L3 | 4,346 | 0.0158 | 2.0% | 0.0156 |
| Global L0 | 4,829 | 0.0083 | 0.0% | 0.0083 |
| Global L1 | 4,840 | 0.0084 | 0.0% | 0.0083 |
| Global L2 | 4,800 | 0.0083 | 0.0% | 0.0083 |
| Global L3 | 4,906 | 0.0083 | 0.0% | 0.0083 |

Aligned and remote scores are nearly identical. Subtracting the fitted
target-mean readout isolates local baseline-relative changes for the browser,
but emitted-token recovery is also negative: every aligned layer for both
estimators has 0% top-10, and only two layer/estimator cells reach 2% top-100.
This pilot therefore provides **no evidence that the current reusable encoder
lens localizes lexical content**.

The aligned/global Jacobian matrix cosine still rises with depth:

| Encoder layer | Matrix cosine |
|---|---:|
| L0 | 0.365 |
| L1 | 0.526 |
| L2 | 0.704 |
| L3 | 0.847 |

That is a geometric property of the fitted maps, not localization evidence.
The DTW-aligned evaluation is also circular because Whisper cross-attention
supplies both fitting windows and held-out locations.

## Qualitative failures

- “played a familiar jazz melody” became “plate of familiar jasmeldi.”
- “left a striped umbrella” became “left is striped, umbrella.”
- The current encoder grid intentionally displays target-mean-relative token
  changes. These may be useful hypotheses to inspect, but there is no validated
  claim that their top words are recognized concepts or causal intermediates.
- Whisper's encoder is bidirectional, so even a genuinely localized readout
  would identify an audio location, not an online recognition instant.

## Next validation gates

1. Fit 10/25/50/100 natural-speech subsets and bootstrap clip-level intervals.
2. Fit independent corpus halves and measure matrix cosine and top-k stability.
3. Use an external forced aligner or known synthetic word boundaries so fitting
   and validation do not share Whisper's DTW signal.
4. Compare affine absolute, baseline-relative, global, aligned, `sum`/`mean`,
   and frozen-QK variants on the same split.
5. Separate correct and incorrect generated words and run silence, noise,
   homophone, temporal-order, and prefix controls.
6. Add matched-norm encoder-frame and decoder-position interventions. A readout
   remains descriptive until it passes this causal gate.
