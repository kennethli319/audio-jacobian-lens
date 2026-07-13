# Experiment report: Chatterbox T3 fitted speech-code pilot

Date: 2026-07-10

Status: **preliminary held-out trend; validation incomplete**

## Question

Can a corpus-fitted projected transport make the realized Chatterbox acoustic
code increasingly readable through the causal T3 layers?

In the first quantized-checkpoint pilot, the realized code's rank improves
monotonically from early to late fitted layers on four prompts excluded from
fitting. This is encouraging same-stream evidence and a useful integration
check. It is not yet a stable mechanistic result: rank/seed, corpus-size,
direct-logit, shuffled, and unquantized comparisons remain open.

## Model and estimator

- Model: `mlx-community/chatterbox-turbo-8bit` at revision
  `2f2e21a03863f86a1274d1060dcc188e7cde77e1`.
- S3 tokenizer: `mlx-community/S3TokenizerV2` at revision
  `e0c9886f0e1c35ae85b1f27277416fb19fc72bec`.
- Stream: post-block T3 speech-prediction residuals at
  L0/L4/L8/L12/L16/L20/L22, transported to post-block L23 before final
  normalization and decoded through the real speech head.
- Estimator: uncentered, equal example mean, target reduction `sum`.
- Projection: seeded Hadamard rank 128/1,024, seed 29.

This is the same causal speech-code stream and is the fitted J-lens analogue.
The separate per-run text gradient and attention matrices are context-specific
diagnostics; they are not part of this corpus-averaged fitted result.

## Artifact provenance

| Field | Value |
|---|---|
| Fit corpus | 10 prompts, 480 selected speech positions |
| Projection time | 391.859 seconds, excluding 8.549 seconds of trajectory capture |
| Serialized size | 2,104,061 bytes, fp16 factors |
| Model compatibility fingerprint | `47f1c6108840fae0` |
| In-app lens fingerprint | `67da0e6fef27e310` |
| Serialized SHA-256 | `ebf46e3e088106e270eff676b09bac9ff3f5ff5206e9e1f924d692ee9a7b2aa8` |
| Prepared-example fingerprint | `9de6342be062cdf4efdf53ae01570cf47f96d0790cb6aa1093e253e4e03bedf7` |

The measured artifact is local and ignored by Git. The fingerprints identify
the file used for this report; they do not imply that the fitted tensor is
distributed in the repository.

## Held-out result

Four manifest records excluded from fitting produced 251 ordinary speech
codes. Rank is one-based among all 6,563 raw speech-head entries. Top-10 rate is
the fraction of positions where the realized code appears among the fitted
distribution's ten highest entries.

| Source layer | Median realized-code rank | Realized code in top 10 |
|---:|---:|---:|
| L0 | 1,593 | 0.4% |
| L4 | 1,376 | 0.4% |
| L8 | 952 | 2.0% |
| L12 | 466 | 6.4% |
| L16 | 187 | 9.6% |
| L20 | 23 | 32.3% |
| L22 | 12 | 45.8% |

Across those positions, the actual final head assigns the realized code a mean
probability of 15.85%; the fitted L22 readout assigns it 3.38% on average. The
rank trend says that the approximate transport becomes more predictive closer
to the T3 head. It does not show that fitted probabilities are calibrated or
that early code IDs have human-readable acoustic meanings.

## Interpretation boundary

Chatterbox speech-code IDs are learned codec symbols, not published words or
phonemes. A selected ID and its rank can be followed across layers, but naming
it as a specific sound without a separate validated decoder would overstate the
evidence.

The page also supports forced-code branches and residual steering. Those are
separate per-run intervention experiments, not validation of this fitted lens.
They follow one teacher-forced path, and the current residual branch lacks a
matched-random distribution and adaptive re-optimization after divergence.

## Open gates

- repeat across projection ranks and seeds;
- fit larger and independently sampled corpora;
- add direct/identity-logit and shuffled controls;
- compare with an unquantized checkpoint;
- build an explicit text-to-speech cross-lens if word-to-code attribution is
  the question; and
- extend beyond T3 to S3Gen and waveform generation before making waveform
  attribution claims.

The current checkpoint is an 8-bit affine MLX conversion, so its gradients and
reported ranks are revision-specific. See [`../CHATTERBOX.md`](../CHATTERBOX.md)
for exact fitting and serving instructions.
