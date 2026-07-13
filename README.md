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

Audio Jacobian Lens is a working research fork that adapts the Jacobian-lens
method from language models to speech systems. The goal is a local, inspectable
workspace for asking when output-relevant information becomes readable inside
automatic speech recognition, speech-to-speech, and text-to-speech models—and
whether carefully chosen residual interventions can change the resulting path.

This repository currently contains five connected workspaces:

| Track | What is implemented | Current evidence boundary | Local page |
|---|---|---|---|
| **Whisper ASR** | Paper-style decoder J-lens, experimental encoder-to-decoder lens, raw output diagnostics, audio upload/samples/recording, synchronized waveform and token timelines | A small synthetic decoder pilot recovers some early lexical directions; the current cross-stream encoder result is negative and must not be treated as a phoneme or streaming-belief detector | [`:8000/`](http://127.0.0.1:8000/) |
| **LFM2.5 speech-to-speech** | Apple-silicon MLX generation, generated-speech playback, fitted readouts over the 16-layer language backbone | The retained lens is a one-clip integration pilot. It does not explain the FastConformer, audio adapter, acoustic codebooks, or played waveform | [`:8001/`](http://127.0.0.1:8001/) |
| **Chatterbox TTS** | Corpus-fitted T3 acoustic-code readouts, per-run text sensitivity and attention, forced-code branches, and residual steering with suffix regeneration | The ten-prompt rank-128 pilot is encouraging but incomplete; acoustic-code IDs are not words or phonemes, and the current work does not attribute S3Gen or waveform samples | [`:8002/chatterbox`](http://127.0.0.1:8002/chatterbox) |
| **Static review** | Primary full cached ASR and speech-to-speech explorers with every saved layer/position cell | Backend-free and safe to serve as static files. Each published explorer has ten reports. ASR combines attributed LibriSpeech examples with the unchanged Laurel/Yanny Audio S7; Speech uses the ten CC BY 4.0 LibriSpeech inputs. The Chatterbox/TTS pilot remains local until its acoustic-code readouts support a clearer interpretation | [Public review](https://kennethli319.github.io/audio-jacobian-lens/) |
| **Phonetic steering experiment** | A recorded fitted-phone encoder intervention integrated into the normal Laurel/Yanny layer explorer, including changed encoder, decoder, and HEAD states | The equal-strength Yanny recipe is the stronger one-clip result and reproduces with a second fitted lens; the Laurel route is target-conditioned and does not transfer exactly. Neither is a universal word-control axis | [Audio 10 replay](https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny) |

The integrated Laurel/Yanny sample uses the exact **Audio S7** from Hans Rutger
Bosker's [Laurel or Yanny? demo](https://hrbosker.github.io/demos/laurel-yanny/),
republished unchanged using the demo page's [CC BY
4.0](https://creativecommons.org/licenses/by/4.0/) notice; see also [Bosker
(2018)](https://doi.org/10.1121/1.5070144). Bosker describes the underlying
viral recording as originating from Vocabulary.com. That is source history
reported by Bosker, not a separate permission claim. The J-lens matrices,
phone-signature overlays, cached interventions, labels, and controls are this
project's additions.

The public review opens directly into the detailed ASR explorer. One consistent
top menu links ASR and Speech, including on narrow screens. The ASR
and Speech `/explorer/{family}/` aliases and findings URLs remain functional for
saved links, but findings are no longer promoted in the primary header. Public
TTS routes and caches are deliberately absent while the local Chatterbox pilot
remains scientifically underdetermined.

The older BPE/prefix Laurel/Yanny steering study is retained as a historical
baseline in [`web/causal.html`](web/causal.html). The fitted-phone follow-up is
summarized in [`docs/CAUSAL_TRACE.md`](docs/CAUSAL_TRACE.md) and replayed from
recorded, sanitized checkpoints inside ASR Audio 10. The source audio is
available there as the attributed, unchanged Audio S7. The retired standalone
checkpoint payload remains hash-pinned for reproducibility but is no longer a
public navigation destination.

## Start here: plans, evidence, and design contracts

The project is deliberately documented as an evolving research program rather
than only as a demo. **Read the project plan before making changes** and update
its milestone checklist, decisions, and work log when a development session
changes the state of the project.

| Document | Use it for |
|---|---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | Canonical goal, scientific contract, milestones M0–M9, implementation decisions, next tasks, and chronological work log |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Mathematical derivation, source/target streams, estimator definitions, rank semantics, and interpretation limits |
| [`docs/STATIC_SHOWCASE_CURATION.md`](docs/STATIC_SHOWCASE_CURATION.md) | Why each public example was selected, exact rank trajectories, controls, rights gates, and the static-bundle backlog |
| [`docs/PILOT_RESULTS.md`](docs/PILOT_RESULTS.md) | Failure-inclusive Whisper Tiny pilot results, including the corrected negative encoder finding |
| [`docs/MLX_LFM.md`](docs/MLX_LFM.md) | Pinned LFM2.5 MLX environment, projected-lens implementation, serving policy, and unfinished validation work |
| [`docs/CHATTERBOX.md`](docs/CHATTERBOX.md) | T3 sequence structure, fitted speech-code lens, local trace, intervention semantics, pilot evaluation, and model boundaries |
| [`docs/CAUSAL_TRACE.md`](docs/CAUSAL_TRACE.md) | Whisper residual-steering protocol, historical BPE/prefix results, fitted-phone follow-up, controls, and remaining causal gates |
| [`samples/README.md`](samples/README.md) | Bundled LibriSpeech provenance, licenses, utterance IDs, and attribution |

### What is established so far

- The reference implementation has been reproduced and the audio adaptation is
  specified and unit-tested.
- Whisper decoder readouts can recover some realized lexical directions before
  the final decoder block, but early readability varies sharply by token.
- The current Whisper encoder-to-decoder lens is a **negative pilot**: real
  speech ranks remain weak and silence can look spuriously strong.
- The Chatterbox T3 pilot shows acoustic-code readability improving through
  depth and includes a verified residual intervention, while rank/seed,
  unquantized-model, finite-difference, and S3-stage controls remain open.
- The LFM language-backbone path works end to end, but its one-clip fitted lens
  is integration evidence rather than a scientific result.
- The local explorers and backend-free cached explorers are usable now; older
  curated findings remain archived while the article carries the public
  interpretation. Public fitted-artifact and generated-audio distribution
  remain gated by the provenance reviews recorded in the plan.
- On one development Laurel/Yanny clip, an equal-strength fitted-phone schedule
  changes ordinary Whisper generation from `Lily!` to tokenizer-faithful
  `Yanny!`, reproduces with a second fitted phone lens, and is not matched by ten
  exact-budget random schedules. A separate exact Laurel route is explicitly
  target-conditioned and does not transfer exactly across lens fits.

The project is based on
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), the
companion implementation for [*Verbalizable Representations Form a Global
Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html). The fork
was identical to upstream commit `581d398` before the audio work began.

## What the method says—and does not say

The initial intuition is mostly right: the lens makes an intermediate hidden
state readable in output-token vocabulary even though that state does not feed
the output head directly. The crucial missing step is a learned transport:

```text
J_l = E[∂h_target / ∂h_l]
lens_l(h) = unembed(J_l h)
```

`J_l` is an average first-order causal map over a corpus and valid positions.
The top output symbols name directions that an activation is generally disposed
to make the model express downstream. They are **not** calibrated probabilities
that the model will emit those symbols, and they are not a complete or unique
transcript of "what the model is thinking."

The site therefore keeps three kinds of measurement visibly separate:

- **Raw model probability:** the base model's teacher-forced final-head
  distribution at a decoder or speech-code step, before generation-time token
  processors and without a calibration claim.
- **Fitted J-lens readout:** output-space logits and full-vocabulary ranks after
  the learned Jacobian transport. Whisper and LFM primarily show the raw score
  and rank. Chatterbox additionally shows the softmax of the fitted speech-head
  logits as a **fitted readout probability**; it is not the raw model's emission
  confidence.
- **Per-run diagnostics and interventions:** gradients, attention, direct code
  forcing, and residual steering answer different questions. They are never
  silently substituted for one another.

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

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Ten bundled natural
speech samples can be selected and played without supplying a local file. The
synthetic UI demo provides a complete backend-free analysis; analyzing a bundled,
uploaded, or recorded clip still requires a compatible fitted lens. You can
record up to 30 seconds from the browser.

The evidence Showcase is also available from the demo-only server at
[http://127.0.0.1:8000/showcase](http://127.0.0.1:8000/showcase). It makes no
model API calls: the local ASR, speech-to-speech, and TTS stories are frozen
curation records with their exact rank semantics, failure controls, and rights
status. Only ASR and speech-to-speech are currently published. See
[`docs/STATIC_SHOWCASE_CURATION.md`](docs/STATIC_SHOWCASE_CURATION.md) before
adding or replacing a public example.

The recorded fitted-phone intervention replay is available at
[http://127.0.0.1:8000/steering](http://127.0.0.1:8000/steering). It is also
backend-free: the controls select only measured Yanny and Laurel checkpoints,
not interpolated strengths or new inference runs.

The normal layer-by-layer version is the [integrated Laurel/Yanny ASR
Explorer](https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny).
Its Original, Yanny, and Laurel buttons swap complete cached encoder, decoder,
and HEAD matrices rather than drawing a client-side approximation. The Yanny
state is the cross-fit-reproduced open-loop result; the Laurel state remains a
target-conditioned, clip-specific existence result.

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
  --lfm-max-new-tokens 512 \
  --port 8001
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

A separate experimental follow-up now tests the fitted distributed phone signatures
themselves rather than BPE pieces. Because a phone prototype lives in the
51,864-dimensional readout basis, the implementation differentiates its
complete prototype score through the matching centered encoder lens, final
decoder normalization, and output head to propose a real 384-dimensional
encoder edit. On the first development-only Laurel/Yanny trace, this direction
raised both Whisper pieces of `Yanny`. The initial one-layer timing hypothesis
reached `Yelly!`/`Yay!`; a later post-hoc active-region, equal-strength
all-encoder-layer schedule generated tokenizer-faithful `Yanny!` on this clip
and survived replacement with an independently fitted phone lens. Ten matched
random schedules failed, but timing, reverse-sign, wrong-time, spectral,
larger-control, and held-out-audio gates remain open. The extended prototypes,
optimizer and runner outputs, and detailed private report stay under the
ignored private experiment tree and remain unpublished. Exact Audio S7 is
republished unchanged, with Bosker attribution and its CC BY 4.0 license, in
the [integrated ASR Audio 10
replay](https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny).
Its sanitized recorded conditions contain aggregate diagnostics, not live
inference, fitted artifacts, or private paths. The equal-strength cross-fit
Yanny result is the primary finding; the exact Laurel route is
target-conditioned and fails exact cross-fit transfer. The current boundary is
recorded in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

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

The Whisper/LFM FastAPI service exposes:

- `GET /api/status`
- `GET /api/samples`
- `GET /api/samples/{id}`
- `POST /api/analyze` with multipart field `audio` and optional
  `time_bin_overlap_seconds` (`0.02` by default; `0` disables overlap)
- `GET /api/docs`

The Chatterbox service has separate generation, fitted-readout, trace,
forced-branch, and residual-branch endpoints documented in
[`docs/CHATTERBOX.md`](docs/CHATTERBOX.md); its acoustic-code payloads should
not be interpreted using the Whisper text-token contract above.

The dependency-free frontend provides:

- one-click attributed LibriSpeech samples, local upload, and in-browser
  microphone recording;
- playback of the exact mono 16 kHz model input and waveform seeking;
- approximate DTW token timestamps, raw teacher-forced probability, entropy,
  and alternatives;
- a proportional encoder waveform-slice navigator plus a wrapping decoder-token
  navigator, both without horizontal matrix scrolling;
- 100 ms encoder pooling with a selectable 20 ms or zero overlap (80 ms default
  hop), exact range reporting, and effective geometry provenance for clips that
  exceed the 100-bin safety limit;
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

## Refresh and validate the backend-free explorers

The personal-site build uses already-inferred JSON only. With the ASR and
speech-to-speech servers ready on ports 8000–8001, and the private fitted-phone
artifacts available, regenerate the complete publication into a separate
static-site checkout with this ordered pipeline:

```bash
.venv/bin/python scripts/record_whisper_phone_steering_explorer.py \
  --audio samples/laurel-yanny.mp3 \
  --output artifacts/private/phonetic_encoder/causal/recorded_asr_steering_explorer_v1.json

.venv/bin/python scripts/export_static_explorer.py \
  --site-root ../kennethli319.github.io/audio-jacobian-lens

.venv/bin/python scripts/publish_static_asr_replay.py \
  artifacts/private/phonetic_encoder/causal/recorded_asr_steering_explorer_v1.json \
  ../kennethli319.github.io/audio-jacobian-lens

.venv/bin/python scripts/publish_static_phone_steering.py \
  --site-root ../kennethli319.github.io/audio-jacobian-lens

.venv/bin/python scripts/validate_static_explorer_site.py \
  ../kennethli319.github.io/audio-jacobian-lens
```

ASR and speech-to-speech are captured sequentially. The exporters whitelist
fields, pin model/lens provenance, remove ephemeral run IDs and generated audio,
and hash every report. ASR's larger exact-length token buckets are split into
compact sidecars and fetched only if a visitor enables the filter.
`export_static_explorer.py` deliberately preserves an already integrated,
manifest-bound Laurel/Yanny replay instead of replacing it with a plain ASR
run. On a clean build it emits an explicit reminder that
`publish_static_asr_replay.py` must run next; that publisher reduces the three
recorded runs, attaches the Original/Yanny/Laurel matrices, copies the exact
attributed Audio S7, and re-hashes the ASR manifest. Do not publish the
intermediate base export. The final validator is the release boundary.

The detailed explorer catalog lives in
`data/static_explorer_catalog_v2.json` and contains ten audio inputs plus ten
held-out or reviewed TTS prompts. The TTS prompts and
`scripts/export_static_tts_explorer.py` are retained for local research, but the
public release gate rejects any TTS route or cached report directory. It is
intentionally separate from `data/static_public_reports_v1.json`. ASR/LFM
exports can be resumed with repeatable `--only` and `--resume`; the resulting
published manifest must still contain the full ordered ten-report family.

The canonical static pages are the detailed explorers at the site root and
`/speech/`. They share the same header dimensions and two-item navigation; the
recorded steering experiment now lives inside ASR Audio 10. The retired
`/steering/` URL forwards saved links to that integrated example. The
ASR/Speech findings and `/explorer/{family}/` routes remain functional for
saved links but are omitted from the primary menu. The speech-to-speech explorer keeps every
saved top-token cell in one continuous, horizontally scrollable matrix with
fixed readable token columns. Its large cell label is the layer's top
candidate; the smaller `realized #N` label is the generated token's exact
competition rank from the complete saved readout, even when that token is
outside the five displayed candidates. Selecting either its generated-text
timeline or any layer/HEAD cell auto-reveals the matching timestep in both
contained scrollers without moving the page.
Each speech report also shows whether generation ended at a natural audio EOS
or exhausted its emergency step cap. A capped response is visibly marked as
possibly truncated and must not be read as a naturally completed answer.

The ASR decoder keeps the complete emitted sequence in one contained,
horizontally scrollable layer matrix. Large text is each layer's active top
candidate and the smaller `realized #N` badge is the exact rank of that
column's emitted token. A decoder column tracks its generated token directly;
an encoder window tracks the output token with greatest overlap under Whisper's
model-derived DTW timing. That encoder pairing is approximate and non-causal.
Phone Signature view is available on every cached ASR sample, so no sample is
singled out with a special feature badge.
When the optional character filter is active, both the top candidate and
realized rank are recomputed in the filtered vocabulary; an excluded realized
token is labeled `realized out`. The encoder layer matrix is independently
horizontally scrollable in both lexical-token and Phone Signature modes.
Selecting any token, waveform region, encoder cell, decoder cell, or HEAD cell
updates the shared coordinate and adjusts only those two matrix scrollers until
the synchronized encoder window and decoder/HEAD column are visible; it never
uses page-level scrolling for this reveal.

The `asr-laurel-yanny` report adds Original, Yanny, and Laurel cached replay
states to this same Explorer grammar. Switching state replaces the complete
saved encoder, decoder, and HEAD matrices while preserving synchronized
coordinates, so the visible top candidates, realized ranks, phone signatures,
and local candidate details all come from the selected recorded run.

The local Chatterbox/TTS workspace and exporter remain available for continued
experimentation, but no TTS page, findings page, alias, or cached TTS report is
part of the public site.

Before publishing, run the static-only integrity gate:

```bash
.venv/bin/python scripts/validate_static_explorer_site.py \
  ../kennethli319.github.io/audio-jacobian-lens
```

This verifies the canonical explorer, findings, and legacy-alias hierarchy;
the correct renderer on every no-index page; all 20 published detailed reports
and the separate ASR/Speech findings bundle; matrices, trace coverage, hashes,
every referenced input file; and the absence of model API calls, generated audio,
or ephemeral analysis identifiers. It also locks the steering payload to recorded-only
checkpoints, excludes source media and private artifacts from that replay
payload, verifies its asset hashes, and preserves the different evidence labels
for Yanny and Laurel. The separately served ASR Audio S7 is allowed only with
its unchanged-content hash and Bosker source, CC BY 4.0 license, and paper
attribution.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
node --check web/app.js
node --check web/chatterbox.js
node --check web/causal.js
node --check web/showcase.js
node --check web/steering.js
node --check web/workspace-nav.js
git diff --check
```

The suite includes explicit-VJP comparisons on deterministic tiny models,
offline Hugging Face Whisper coverage, MLX fitting and replay contracts,
Chatterbox branch and residual-steering tests, static-showcase integrity checks,
and Docker-entrypoint coverage. Platform-specific MLX tests may skip outside an
Apple-silicon environment. Do not commit `artifacts/`, model weights, fitted
lenses, local recordings, or generated evaluation outputs.

## License and provenance

Code is Apache License 2.0; see [`LICENSE`](LICENSE). Existing source files
retain Anthropic's copyright notices. Whisper, LFM, Chatterbox, tokenizer, and
audio-dataset artifacts are subject to their own licenses. No model weights,
private audio, or fitted lens artifacts are committed to this repository.
