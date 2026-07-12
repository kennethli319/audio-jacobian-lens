# Local MLX LFM2.5 vertical slice

This optional path runs LiquidAI's speech-to-speech model locally on an
Apple-silicon Mac and reuses the explorer for a projected Jacobian lens over the
model's **language backbone and tied text head**. It also returns the model's
generated speech for playback. Playback is an output of the underlying model,
not an explanation of its audio-codebook or waveform-generation path.

The MLX path is a working vertical slice, not a completed speech-to-speech
interpretability result. It is independent of the Whisper backend and is not
deployed in the Linux Hugging Face Docker Space.

## Pinned runtime and checkpoint

Use an Apple-silicon Mac and Python 3.12. The `mlx` extra is deliberately pinned:

| Component | Pin |
|---|---|
| `mlx` | `0.32.0` |
| `mlx-audio` | `0.4.5` |
| `mlx-lm` | `0.31.3` |
| `transformers` | `5.12.1` |
| model | `mlx-community/LFM2.5-Audio-1.5B-8bit` |
| revision | `a569a7805a8e3eae954c244e54ba811d479a12c2` |

The MLX Transformers pin differs from the Whisper environment's verified
Transformers 5.13 installation. A dedicated environment avoids silently
changing the existing Whisper setup:

```bash
python3.12 -m venv .venv-mlx
UV_PROJECT_ENVIRONMENT=.venv-mlx uv sync --extra audio --extra mlx
```

Without `uv`, install the same project extras directly:

```bash
.venv-mlx/bin/python -m pip install -e '.[audio,mlx]'
```

The platform markers intentionally install MLX only on macOS `arm64`. The first
model-backed command downloads the pinned checkpoint from Hugging Face.

## Fit a projected language lens

This one-clip command exercises the complete fit path. It is a smoke test, not
a scientifically adequate corpus:

```bash
.venv-mlx/bin/audio-jlens-mlx-fit \
  samples/question.flac \
  --output artifacts/mlx/lfm2_5_audio_1clip_projected_rank512.pt \
  --model mlx-community/LFM2.5-Audio-1.5B-8bit \
  --revision a569a7805a8e3eae954c244e54ba811d479a12c2 \
  --source-layers 0,4,8,12,14 \
  --target-layer 15 \
  --projection-dim 512 \
  --projection-seed 0 \
  --target-reduction sum \
  --max-new-tokens 24
```

Pass multiple audio paths before `--output` for a pilot corpus. Keep fitting and
held-out evaluation clips disjoint; the three bundled LibriSpeech files are
convenient integration examples, not a representative evaluation set.

The command first generates a deterministic interleaved text/audio response,
then replays that path without a streaming cache. Ordinary generated text
positions provide the teacher-forced targets. The defaults capture post-block,
pre-final-normalization residuals at language layers 0, 4, 8, 12, and 14 and
target language layer 15. Same-stream centering is off unless `--center` is
explicitly supplied.

## Run the local explorer

Serve the fitted artifact on loopback:

```bash
.venv-mlx/bin/audio-jlens \
  --backend mlx-lfm \
  --model mlx-community/LFM2.5-Audio-1.5B-8bit \
  --revision a569a7805a8e3eae954c244e54ba811d479a12c2 \
  --lens artifacts/mlx/lfm2_5_audio_1clip_projected_rank512.pt \
  --lfm-max-new-tokens 512
```

`--lfm-max-new-tokens` is a serving-only emergency cap over the combined
interleaved sequence, not 512 text tokens and not a requested waveform
duration. With the pinned model's default cycle of six text positions followed
by 12 acoustic frames, text and speech consume the same counter. Generation
normally stops well below the ceiling when the model emits its final audio EOS;
the UI labels a ceiling hit as an incomplete response. This override does not
rewrite the generation policy or fingerprint stored in the fitted lens
artifact.

Open <http://127.0.0.1:8000>, choose or upload a short clip, and run the
analysis. The response contains:

- the exact mono 16 kHz model input and its waveform;
- the model's generated text and raw teacher-forced text-token diagnostics;
- projected language-layer J-lens readouts at generated text positions,
  including the realized token's exact raw score and competition rank even when
  it is outside the retained top-k; and
- the model's decoded 24 kHz speech response, when audio frames were generated.

The large token in each layer cell is that readout's top candidate. The smaller
`realized #N` label tracks the token that generation actually produced. For an
eligible lexical token, `N` is its exact rank among the 61,690-token lexical
display vocabulary; control or punctuation-only targets fall back to their
exact rank in the full 65,536-token vocabulary. The orange HEAD rank always
uses the full vocabulary. All ranks use one plus the count of strictly greater
scores, so tied scores share a competition rank.

There is no audio-time encoder grid or generated-token timing in this slice.
The explorer's reused "decoder" region means the causal LFM language backbone,
not a Whisper decoder or an attribution of the played speech.

## Projection method and rank audit

A dense map for one LFM source layer has `2048 × 2048` entries and a direct
reverse-mode fit would require 2,048 target probes. The implemented estimator
uses a seeded subset of an orthogonal Hadamard basis. For each selected target
probe, native `mx.vjp` measures the response at each source layer. The artifact
stores shared target factors `T` and per-layer source factors `S_l` and applies

```text
J_l ≈ Tᵀ S_l / k
```

where `k` is `--projection-dim`. Rank 2,048 uses the complete basis and
reconstructs the selected dense estimator; lower ranks are approximations whose
fine token ordering may change with rank and seed.

The earlier one-clip rank-64 Rademacher smoke artifact produced poor lexical
readouts and is not pilot evidence. The current implementation uses
`subsampled_hadamard_output_probe_vjp`, and **rank 512 is the minimum pilot
default**, not a quality guarantee. Any reported result still needs rank/seed
sensitivity, multiple fitting clips, held-out evaluation, and failure cases.

Artifacts record the projection method, rank, seed, reduction, exact checkpoint
revision, runtime versions, tokenizer/config fingerprints, and fitting-example
fingerprint. Do not treat artifacts with different methods, ranks, seeds, or
generation policies as interchangeable.

## Architecture and scope caveats

The pinned checkpoint is an 8-bit conversion. Its high-level route is a
17-layer, 512-wide FastConformer audio encoder, an adapter into a 16-layer,
2,048-wide hybrid LFM language backbone, and separate text and audio output
paths. Text logits use the final normalization and tied 65,536-token embedding.
Generated speech instead passes through a Depthformer and eight 2,049-token
audio-codebook heads before 24 kHz decoding.

Only the language-to-language transport and tied text head are implemented.
The following remain future work:

- FastConformer residual capture and audio-time alignment;
- encoder-to-language Jacobians across the 512-to-2,048 adapter;
- Depthformer and per-codebook audio-head targets;
- relating audio-codebook readouts to the decoded waveform; and
- rank-sensitive held-out evaluation, runtime/memory benchmarks, fit
  checkpointing, and larger-corpus convergence.

Quantization changes the checkpoint and potentially its Jacobians. Results are
specific to this exact 8-bit revision and must not be silently generalized to
the BF16 model or another conversion.

## License boundary

This repository's code is Apache-2.0 and `mlx-audio` is MIT, but the model
weights use the [LFM Open License
v1.0](https://github.com/Liquid4All/liquid-audio/blob/main/LICENSE), not either
code license. That license includes redistribution obligations and a commercial
use threshold tied to the user's legal entity and annual revenue. Review the
current model license before product use or redistributing weights or derived
artifacts. The bundled LibriSpeech examples are CC BY 4.0 as documented in
[`../samples/README.md`](../samples/README.md).
