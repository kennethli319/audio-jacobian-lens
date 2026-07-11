---
title: Audio Jacobian Lens
emoji: 🎧
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
fullWidth: true
header: mini
models:
  - openai/whisper-tiny.en
tags:
  - audio
  - automatic-speech-recognition
  - interpretability
---

# Audio Jacobian Lens

An experimental, local-first Jacobian-lens explorer for Whisper. This fork
preserves Anthropic's reference language-model implementation and adds:

- a Hugging Face Whisper encoder-decoder adapter;
- the paper-style causal decoder J-lens;
- an experimental encoder-to-decoder audio J-lens;
- an Apple-silicon MLX vertical slice for LFM2.5 Audio's language backbone;
- a local MLX Chatterbox-Turbo fitted speech-code J-lens plus per-run
  code-to-text sensitivity page;
- checkpointed fitting and held-out evaluation CLIs; and
- a localhost waveform, raw-output, encoder-grid, and decoder-grid explorer.

The project is based on
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), the
companion implementation for [*Verbalizable Representations Form a Global
Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html). The fork
was identical to upstream commit `581d398` before the audio work began.

Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) before continuing development. It is
the durable milestone, decision, and work log. The mathematical derivation and
interpretation rules are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The
separate [MLX LFM2.5 guide](docs/MLX_LFM.md) documents the local
speech-to-speech vertical slice and its narrower interpretation boundary.

## What the method says—and does not say

The initial intuition is mostly right: the lens makes an intermediate hidden
state readable in output-token vocabulary even though that state does not feed
the output head directly. The crucial missing step is a learned transport:

```text
J_l = E[∂h_target / ∂h_l]
lens_l(h) = unembed(J_l h)
```

`J_l` is an average first-order causal map over a corpus and valid positions.
The top tokens name directions that an activation is generally disposed to make
the model verbalize downstream. They are **not** calibrated probabilities that
Whisper will emit those tokens, and they are not a complete or unique transcript
of "what the model is thinking."

The site therefore keeps two metrics visibly separate:

- **Raw model probability:** Whisper's teacher-forced base-model distribution at
  a decoder step, shown before generation-time token processors and without a
  calibration claim.
- **J-lens salience:** a ranked vocabulary readout from an intermediate state,
  shown as a score/rank and never as a confidence percentage.

Whisper's encoder is bidirectional, so an encoder cell at 1.0 seconds may use
audio from later in the same window. Treat it as an audio-location view, not a
streaming belief trace.

## Quick start

Python 3.10+ is required. On this project machine, Python 3.12, PyTorch 2.13,
Transformers 5.13, and an Apple M2 Pro are verified.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[audio,dev]'
```

The checked-in `uv.lock` can instead be installed with
`uv sync --extra audio --extra dev`.

Start the frontend in demo-only mode (no model download or fitted lens needed):

```bash
.venv/bin/audio-jlens
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Three bundled natural
speech clips and the synthetic interactive demo work without supplying a local
audio file; model-backed analysis still requires a fitted lens. You can also
upload audio or record up to 30 seconds from the browser.

For live analysis, fit or supply a compatible lens and start:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --lens artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt
```

The server listens only on `127.0.0.1` by default. Uploaded audio is processed
locally and is not retained by the service. The model and processor are
downloaded from Hugging Face on first use. Audio is limited to 30 seconds.
`soundfile` handles WAV/AIFF/FLAC/OGG; the server falls back to a locally
installed `ffmpeg` for formats such as MP3/M4A and browser-recorded WebM.

### Apple-silicon MLX speech-to-speech slice

On an Apple-silicon Mac, use a separate environment for the exact optional MLX
stack and pinned 8-bit LFM2.5 Audio checkpoint:

```bash
python3.12 -m venv .venv-mlx
UV_PROJECT_ENVIRONMENT=.venv-mlx uv sync --extra audio --extra mlx

.venv-mlx/bin/audio-jlens-mlx-fit \
  samples/question.flac \
  --output artifacts/mlx/lfm2_5_audio_1clip_projected_rank512.pt \
  --model mlx-community/LFM2.5-Audio-1.5B-8bit \
  --revision a569a7805a8e3eae954c244e54ba811d479a12c2 \
  --projection-dim 512 \
  --max-new-tokens 24

.venv-mlx/bin/audio-jlens \
  --backend mlx-lfm \
  --model mlx-community/LFM2.5-Audio-1.5B-8bit \
  --revision a569a7805a8e3eae954c244e54ba811d479a12c2 \
  --lens artifacts/mlx/lfm2_5_audio_1clip_projected_rank512.pt \
  --lfm-max-new-tokens 512
```

This path captures the 16-layer LFM language backbone, fits a projected
language-to-language Jacobian, reads it through the tied text head, and plays
the model's generated speech. It does **not** yet fit the 17-layer
FastConformer, the audio adapter, Depthformer, or eight audio-codebook heads;
speech playback is not audio-output attribution. The one-clip command above is
only a vertical-slice smoke test. Rank 64 produced poor exploratory readouts;
rank 512 is the minimum pilot default and still needs rank/seed and held-out
validation. The serving budget is shared by interleaved text tokens and
acoustic frames. The 512-position default is an emergency ceiling rather than
a target duration: generation normally stops earlier when the model emits its
final audio EOS. See
[`docs/MLX_LFM.md`](docs/MLX_LFM.md) for exact package pins, projection details,
architecture caveats, and licensing.

### Apple-silicon Chatterbox fitted speech-code lens and text trace

The Chatterbox workspace now combines two separate diagnostics: a projected,
corpus-fitted J-lens over T3 speech-prediction residuals, and the existing
per-run gradient/attention trace from a selected speech code to input-text
positions. Fit the local pilot artifact with:

```bash
.venv-mlx/bin/audio-jlens-chatterbox-fit \
  artifacts/chatterbox/chatterbox_turbo_10prompt_rank128_seed29.pt \
  --manifest data/chatterbox_fit_prompts.jsonl \
  --split fit \
  --limit 10 \
  --model mlx-community/chatterbox-turbo-8bit \
  --revision 2f2e21a03863f86a1274d1060dcc188e7cde77e1 \
  --s3-tokenizer mlx-community/S3TokenizerV2 \
  --s3-tokenizer-revision e0c9886f0e1c35ae85b1f27277416fb19fc72bec \
  --generation-seed 7 \
  --max-speech-tokens 48 \
  --max-speech-positions 48 \
  --source-layers 0,4,8,12,16,20,22 \
  --target-layer 23 \
  --rank 128 \
  --projection-seed 29 \
  --target-reduction sum
```

Then load it in the dedicated local server:

```bash
.venv-mlx/bin/audio-jlens-chatterbox \
  --model mlx-community/chatterbox-turbo-8bit \
  --revision 2f2e21a03863f86a1274d1060dcc188e7cde77e1 \
  --s3-tokenizer mlx-community/S3TokenizerV2 \
  --s3-tokenizer-revision e0c9886f0e1c35ae85b1f27277416fb19fc72bec \
  --lens artifacts/chatterbox/chatterbox_turbo_10prompt_rank128_seed29.pt \
  --top-k 5
```

Open <http://127.0.0.1:8002/chatterbox>. Select a nominal 40 ms speech-code
slice to compare the fitted probability across T3 layers and generated speech
positions, then compare that code's separate per-run gradient across input-text
positions with causal self-attention. To test a candidate counterfactual, click
a non-realized ordinary code in the selected-position inspector and run the
separate branch action. It keeps the earlier code prefix, forces that ID at the
actual T3 output decision, greedily regenerates the suffix, and decodes a new
waveform. This is output-decision forcing, not fitted-residual steering. The
separate residual mode instead lets the user choose sampled post-block layers,
a forward speech-position span, and a norm cap. It adds context-specific
target-margin gradient directions to the real residual states, automatically
searches for a top-1/greedy flip without substituting the code, and reports the
target's before/after rank and probability through the fitted layers and HEAD.

The measured pilot used 10 fit prompts, 480 speech positions, rank `128/1024`,
projection seed `29`, and `391.859` seconds of projected VJPs. Its fp16 artifact
is `2,104,061` bytes; model fingerprint `47f1c6108840fae0`, in-app lens
fingerprint `67da0e6fef27e310`, and file SHA-256
`ebf46e3e088106e270eff676b09bac9ff3f5ff5206e9e1f924d692ee9a7b2aa8`.

On four disjoint held-out prompts (251 codes), realized-code median ranks from
L0 through L22 were `1593, 1376, 952, 466, 187, 23, 12`; top-10 rates were
`0.4%, 0.4%, 2.0%, 6.4%, 9.6%, 32.3%, 45.8%`. The actual final-head mean
probability was `15.85%`, versus fitted L22's `3.38%`. These are pilot results:
rank/seed replication, a direct logit-lens baseline, and an unquantized-model
comparison remain undone. Acoustic codes are not phonemes, the word-level
gradient/attention matrices are still local rather than fitted, and neither
view is end-to-end waveform attribution. See
[`docs/CHATTERBOX.md`](docs/CHATTERBOX.md) for the method, exact runtime pins,
full held-out table, security boundary, model provenance, and watermark caveat.

## Hugging Face Docker Space

The model-backed Protected Space is live at
[`kennethli319/audio-jacobian-lens`](https://huggingface.co/spaces/kennethli319/audio-jacobian-lens),
with the direct application at
[`kennethli319-audio-jacobian-lens.hf.space`](https://kennethli319-audio-jacobian-lens.hf.space).
The Space card at the top of this file selects the Docker SDK and port 7860.
The image prefetches the pinned public Whisper Tiny snapshot during its build,
so a Space wake-up does not wait for the model download before opening its HTTP
port.

Build and smoke-test the demo-only image first:

```bash
docker build -t audio-jacobian-lens .
docker run --rm -p 7860:7860 audio-jacobian-lens
```

Then, from a second terminal:

```bash
curl http://127.0.0.1:7860/api/status
```

`ready: false` is expected in this mode; the static explorer and synthetic demo
still work. To test model-backed analysis with the current local pilot artifact:

```bash
docker run --rm -p 7860:7860 \
  -v "$PWD/artifacts/pilot:/lens:ro" \
  -e JLENS_MODEL_REVISION=87c7102498dcde7456f24cfd30239ca606ed9063 \
  -e JLENS_LENS_PATH=/lens/whisper_tiny_en_10_tts_aligned.pt \
  audio-jacobian-lens
```

The `artifacts/` directory and all `*.pt` files are deliberately gitignored, so
that command does **not** make the lens part of a Space. For hosting, put a lens
you are allowed to distribute in a dedicated Hub model repository and configure
these Space Settings variables:

The current private deployment artifact is
`artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt` (2,075,933 bytes; SHA-256
`489582ce50736e8e7b78b7cb9d0140b963d99c5d7784635ed93e71710ab293a2`). It was
fitted on ten synthetic macOS TTS clips and carries the documented negative
encoder-pilot limitation. It is pinned at private artifact commit
`baf116d4f1dfb695b7c5e6ba18e8ed6b338ce696`; provenance and redistribution
review must be completed before making it public.

- `JLENS_MODEL_REVISION`: the pinned Whisper commit above;
- `JLENS_LENS_REPO_ID`: for example, `<user>/audio-jacobian-lens-artifacts`;
- `JLENS_LENS_FILENAME`: the combined `.pt` filename; and
- optionally `JLENS_LENS_REVISION`: preferably an immutable commit SHA.

For a private artifact repository, add a read-scoped `HF_TOKEN` as a Space
**Secret**, never as a public Variable or committed file. The container also
accepts `JLENS_LENS_PATH`, `JLENS_ENCODER_LENS_PATH`, and
`JLENS_DECODER_LENS_PATH` for files already present in the image or a local
mount.

This deployment uses **Protected** visibility. Protected keeps the repository
private but the running app remains public; a **Private** Space restricts the
app and cannot use a public embed or custom domain. To reproduce it, create a
Space with Docker as its SDK and push this repository:

```bash
hf auth login
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space HEAD:main
```

The current deployment runs on CPU Basic. Its first external `question.flac`
analysis completed in 4.44 seconds and returned a 472,738-byte result; this is a
smoke measurement, not a capacity benchmark. Benchmark longer clips and
concurrent use before selecting CPU Upgrade; Pro does not include paid Space hardware.
Upgraded hardware is billed while Starting or Running and does not sleep by
default, so set a custom sleep time in Settings. Do not select a GPU until the
image has been rebuilt and tested with CUDA-capable PyTorch. ZeroGPU is not an
option for this Docker/FastAPI app because it currently supports Gradio Spaces
only. See Hugging Face's [Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker)
and [hardware and billing](https://huggingface.co/docs/hub/spaces-gpus)
documentation.

### Pro custom domain and website embed

Custom domains require a Pro (or Team/Enterprise) subscription and a Public or
Protected Space. In the Space's **Settings → Custom Domain**, enter a subdomain
such as `lens.example.com`, then create this DNS record with the domain host:

```text
Type: CNAME
Name: lens
Target: hf.space
```

Wait for the Space setting to report **Ready**, then verify the app and browser
recording at `https://lens.example.com`. Link to that URL directly, or embed it
from an HTTPS page:

```html
<iframe
  src="https://lens.example.com"
  title="Audio Jacobian Lens"
  allow="microphone"
  loading="lazy"
  style="width: 100%; min-height: 900px; border: 0"
></iframe>
```

The direct `https://<space-subdomain>.hf.space` URL works in the same iframe.
See the official [custom-domain](https://huggingface.co/docs/hub/spaces-custom-domain)
and [embedding](https://huggingface.co/docs/hub/spaces-embed) guides.

### Cloud privacy and publication checks

Localhost's "processed locally" statement does not apply to a Space: uploaded
or recorded audio is sent to the Hugging Face-hosted container. This application
does not intentionally persist uploads or cache analysis results, but it returns
the normalized waveform to the browser for playback and the platform may process
request metadata under its own terms. The public server has a 64 MB request
limit, a 30-second decoded-audio limit, and one inference slot; it has no user
authentication or general rate limiter. Protected visibility does not change
that. Do not invite sensitive uploads until the public notice, access policy,
abuse controls, and cost limits are appropriate for the audience.

The code is Apache-2.0 and the bundled LibriSpeech samples are CC BY 4.0 with
attribution in [`samples/README.md`](samples/README.md). Whisper weights and a
fitted lens remain separately licensed artifacts. A lens file contains tensors
plus corpus/model fingerprints and fitting metadata, not the source audio, but
inspect that metadata and confirm the rights to its fitting data before
publishing it. A Protected repository is not a substitute for that review.

## Experimental causal trace

`audio-jlens-causal` makes matched-norm edits to real post-block encoder audio
positions or decoder prediction positions, then reruns every downstream layer.
Decoder schedules use source L0–L2 and leave final target L3 unedited. The CLI
reports per-token and full-candidate teacher-forced evidence, free generation,
propagation, and a matched-norm random-direction control. It is experimental
and also supports exploratory encoder directions averaged over decoded-token
prefix families such as `Y*` or `La*`. Those families are vocabulary groups,
not phonemes or encoder probabilities. Candidate comparisons use total
candidate-token path likelihood by default, show the absolute path likelihood,
retain mean-token scores as a secondary diagnostic, and exclude EOS. The trace
should not be described as
model confidence or a human perceptual mechanism. See
[the causal-trace protocol](docs/CAUSAL_TRACE.md) for the local command and
interpretation rules.

## Fit a Whisper lens

The fitting CLI consumes JSON Lines with paths relative to the manifest:

```json
{"audio":"audio/001.wav","text":"The spoken transcript.","speaker":"s01"}
```

Fit both streams:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-fit \
  data/my_audio/manifest.jsonl \
  --output artifacts/whisper_tiny_en.pt \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063
```

Defaults:

- decoder sources are blocks 0–2 and target is final block 3;
- decoder fitting uses all ordinary-text targets and the causal mask supplies
  the paper's `future >= source` relationship;
- encoder fitting samples one generated text token per clip, obtains its
  cross-attention/DTW timestamp, and averages source gradients only in the
  aligned audio window;
- output dimensions are batched 8-at-a-time for the decoder and 4-at-a-time for
  the encoder;
- interrupted fits resume from `.checkpoints/`; and
- the artifact records the model configuration, estimator recipe, corpus hash,
  weight revision/checksum, tokenizer and preprocessing fingerprints, exact
  fitting-example hashes, layer set, reduction, and example count.

Encoder artifacts additionally store source and target activation means. Their
cross-stream transport is `J(h - mean_encoder) + mean_decoder_target`; the
browser removes the fitted target-mean readout when ranking time-local encoder
changes. Decoder transport remains paper-compatible and linear.

Use `--encoder-estimator global` to fit the simpler all-audio/all-output
prototype for comparison. Use `--target-reduction mean` to normalize the
paper-style target sum by transcript length. These are methodological variants,
not interchangeable artifacts.

On macOS, this command creates a small synthetic plumbing corpus:

```bash
.venv/bin/python scripts/make_macos_tts_corpus.py \
  --output-dir artifacts/tts_corpus --count 16
```

Synthetic TTS is useful for end-to-end tests but is too narrow for scientific
claims. Fit/evaluation audio must be disjoint, and natural speech from multiple
speakers and domains is required before interpreting qualitative examples.

## Evaluate on held-out clips

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-eval \
  artifacts/tts_corpus/manifest.jsonl \
  --lens artifacts/whisper_tiny_en.pt \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --start 10 --limit 6 \
  --output artifacts/eval.json
```

The pilot evaluator reports full-vocabulary rank, mean reciprocal rank, and
top-k rates for:

- a generated-token final-logit self-check, explicitly labeled tautological;
- decoder J-lens versus direct unembedding at each source layer; and
- encoder absolute-affine and target-mean-relative ranks at aligned versus
  half-utterance-shifted audio frames.

The evaluator also reports punctuation-normalized WER/CER and rejects clips
whose hashes occur in the fitting artifact unless `--allow-overlap` is passed.
Rank metrics test readout behavior; they do not establish causal use.

## Current pilot — preliminary

On six held-out synthetic clips (50 generated lexical tokens), the decoder
J-lens improved top-10 recovery over direct unembedding at L0 (14% vs 4%) and
L1 (18% vs 8%); the two converged at L2 (54% vs 52%). Normalized WER was 10.6%.

The encoder result is negative after correcting cross-stream centering.
Aligned and remote absolute-affine ranks are nearly identical, and every
target-mean-relative aligned layer has 0% top-10 recovery. The earlier
uncentered pilot's apparently localized words are invalid and have been
discarded. See [`docs/PILOT_RESULTS.md`](docs/PILOT_RESULTS.md) for the full
failure-inclusive report.

## Local explorer

The FastAPI service exposes:

- `GET /api/status`
- `GET /api/samples`
- `GET /api/samples/{id}`
- `POST /api/analyze` with multipart field `audio` and optional
  `time_bin_overlap_seconds` (`0.02` by default; `0` disables overlap)
- `GET /api/docs`

The dependency-free frontend provides:

- one-click attributed LibriSpeech samples, local upload, and in-browser
  microphone recording;
- playback of the exact mono 16 kHz model input and waveform seeking;
- approximate DTW token timestamps, raw teacher-forced probability, entropy,
  and alternatives;
- a proportional encoder waveform-slice navigator plus a wrapping decoder-token
  navigator, both without horizontal matrix scrolling;
- 200 ms encoder pooling with a selectable 20 ms or zero overlap, exact range
  reporting, and effective geometry provenance for adaptively widened long clips;
- shared selection across the LM/output token, encoder slice, decoder position,
  dual waveform overlays, and audio seek position;
- a distinct decoder output-head card after L0–L2;
- independent, vocabulary-wide decoded-character filters for every encoder
  layer and decoder L0–L1, while decoder L2 and the output head stay unfiltered;
- absolute 0–100% probability gradients and separately labeled within-layer
  relative J-lens intensity—raw logits are never presented as percentages;
- fixed top-k inspection with keyboard and touch equivalents; and
- model/lens provenance plus interpretation warnings.

The sample audio is from LibriSpeech `dev-clean` under CC BY 4.0; exact
utterance IDs and attribution are in [`samples/README.md`](samples/README.md).

## Python API

```python
from transformers import AutoProcessor, WhisperForConditionalGeneration
from jlens import HFWhisperLensModel, WhisperJacobianLens

model_id = "openai/whisper-tiny.en"
revision = "87c7102498dcde7456f24cfd30239ca606ed9063"
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
hf_model = WhisperForConditionalGeneration.from_pretrained(
    model_id, revision=revision
)
model = HFWhisperLensModel(hf_model, processor, model_id=model_id)
lens = WhisperJacobianLens.load("artifacts/whisper_tiny_en.pt")
lens.validate_model(model)
```

The original decoder-only language-model API remains available:

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model")
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(
    "org/lens-repo", filename="model/lens.pt"
)
lens_logits, model_logits, _ = lens.apply(model, "A prompt", positions=[-1])
```

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
node --check web/app.js
```

Current baseline after the decoder L0/L1 length-filter addition: 103 tests
pass, including an explicit-VJP comparison on a rectangular tiny
encoder-decoder, an offline random Hugging Face Whisper model, and
Docker-entrypoint argument coverage.

## License and provenance

Code is Apache License 2.0; see [`LICENSE`](LICENSE). Existing source files
retain Anthropic's copyright notices. Whisper model code/weights and any audio
datasets are subject to their own licenses. No Whisper weights, private audio,
or fitted lens artifacts are committed to this repository.
