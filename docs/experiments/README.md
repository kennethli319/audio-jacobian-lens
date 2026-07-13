# Experiment reports

These reports are the canonical home for measured outcomes. Setup and serving
instructions live in the repository's other documentation; the root README is
kept deliberately short.

| Report | Evidence status | Scope |
|---|---|---|
| [Whisper Tiny J-lens pilot](WHISPER_TINY_PILOT.md) | Preliminary, failure-inclusive plumbing pilot | First fitted Whisper decoder and encoder readouts, including the corrected negative encoder result |
| [Whisper encoder phonetic signatures](PHONETIC_SIGNATURES.md) | Exploratory development evidence; locked test untouched | Speaker-disjoint natural-speech tests of distributed phone information in encoder J-readouts |
| [Whisper fitted-phone steering](WHISPER_PHONE_STEERING.md) | Preliminary, one-clip causal existence results | Timed residual edits toward Yanny and Laurel, with fit replication and small matched-control sets |
| [LFM2.5 MLX vertical slice](LFM_VERTICAL_SLICE.md) | Integration smoke only; no positive interpretability result | Projected language-stream lens and generated-speech plumbing on one quantized checkpoint |
| [Chatterbox T3 fitted-code pilot](CHATTERBOX_T3_PILOT.md) | Preliminary held-out trend; validation incomplete | Corpus-fitted projected speech-code readouts on a quantized Chatterbox checkpoint |
| [Static example screening](STATIC_EXAMPLE_SCREENING.md) | Curatorial screening record | Evidence and rights review used to choose cached public examples |

## Reading the evidence labels

- **Preliminary** means the implementation and reported measurement worked,
  but important robustness checks remain open.
- **Exploratory** means hypotheses, timing, examples, or analyses were developed
  while looking at the same development material. It is not a locked confirmatory
  result.
- **Clip-specific existence result** means an intervention crossed a model
  decision boundary on one recording. It is not a universal model control axis.
- **Integration smoke** establishes that a pipeline runs end to end; it does not
  establish interpretability quality or scientific generalization.

Unless a report explicitly says otherwise, J-lens readouts are diagnostic
scores rather than calibrated probabilities, top vocabulary IDs are not the
model's verbalized thoughts, and decodability does not by itself establish
causal use.

## Reproduction and artifacts

The reports record exact checkpoint revisions, estimator choices, data
separation, and known failure modes. Some fitted tensors, row-level corpora, and
development search artifacts are intentionally excluded from Git. Public
cached explorers contain sanitized measured matrices, not a browser-side model
or downloadable fitting data.

Read [the methodology](../METHODOLOGY.md) for the common estimator and
interpretation contract, and [the project plan](../../PROJECT_PLAN.md) for the
chronological decision log and open gates.
