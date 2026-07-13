# Experiment report: local MLX LFM2.5 vertical slice

Date: 2026-07-10

Status: **integration smoke only; no positive interpretability result**

## Purpose

This experiment tested whether the repository's fit, artifact, inference, and
explorer path could be extended from Whisper to a local quantized
speech-to-speech model. It succeeded as an end-to-end engineering slice over
LFM's language stream. It did not establish a useful fitted lens on held-out
speech, and it does not explain the generated waveform.

## Fixed target

- Model: `mlx-community/LFM2.5-Audio-1.5B-8bit`.
- Revision: `a569a7805a8e3eae954c244e54ba811d479a12c2`.
- Runtime: Apple silicon with MLX 0.32.0, MLX-Audio 0.4.5, MLX-LM 0.31.3,
  and Transformers 5.12.1.
- Captured stream: post-block residuals in the 16-layer, 2,048-wide LFM
  language backbone, decoded through its real final normalization and tied text
  head.

The played 24 kHz response follows separate Depthformer, audio-codebook, and
codec paths. Those paths are outside this fitted transport. Playback proves the
underlying model generated speech; it is not an attribution of its frames or
waveform samples.

## Estimator

A direct reverse-mode dense map would require 2,048 target probes per source
layer. The implemented approximation uses a seeded subset of an orthogonal
Hadamard basis and stores target and per-layer source factors:

```text
J_l approximately T.T @ S_l / projection_rank
```

Rank 2,048 is the complete-basis reconstruction. Lower ranks are
approximations whose token ordering can vary with rank and seed. Artifacts bind
the projection method, rank, seed, reduction, checkpoint revision, runtime,
tokenizer/config fingerprints, and fitting examples.

## What was observed

An early one-clip rank-64 Rademacher artifact produced poor lexical readouts.
It is a negative smoke result and is not evidence for an LFM lens.

The retained one-clip integration artifact used a rank-512 Hadamard projection
at seed 29 over L0/L4/L8/L12/L14 to target L15. Fitting took 147.90 seconds and
created a 12,588,293-byte local artifact. A same-clip localhost request then:

- returned HTTP 200 in 1.58 seconds with a 174,214-byte response;
- produced the partial text `I'm sorry, but I,`;
- returned 0.96 seconds of mono 24 kHz speech; and
- rendered all five projected language layers with synchronized token
  selection and explicit projection/text-head warnings.

These values verify serialization and serving on one machine; they are not
quality, latency, or generalization benchmarks. No disjoint held-out lens
evaluation was completed.

## Conclusion and open gates

The vertical slice demonstrates native MLX VJPs, deterministic interleaved
generation and replay, projected artifact compatibility, tied-head readouts,
and generated-speech playback. It provides no empirical basis yet for saying
that the displayed early-layer lexical candidates are stable or meaningful.

Required next work includes multiple fitting clips, disjoint held-out clips,
projection-rank and seed sensitivity, failure-inclusive evaluation, and fit
checkpointing. Model coverage must also be extended separately to the
FastConformer encoder, the 512-to-2,048 adapter, Depthformer, audio-codebook
heads, and decoded waveform if those components are to be interpreted.

Results are specific to this 8-bit conversion. Quantization can change the
Jacobian, so they must not be generalized to the BF16 model or another
conversion. See [`../MLX_LFM.md`](../MLX_LFM.md) for installation and serving
instructions and the model-license boundary.
