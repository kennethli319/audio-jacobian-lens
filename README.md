# Audio Jacobian Lens

Audio Jacobian Lens adapts the Jacobian-lens method to speech models. It
provides local and cached explorers for inspecting when output-relevant
directions become readable across Whisper ASR and speech-to-speech model
layers, plus experimental tools for fitting lenses and testing residual
interventions.

Start with the [public cached explorer](https://kennethli319.github.io/audio-jacobian-lens/),
the [model-backed Hugging Face Space](https://huggingface.co/spaces/kennethli319/audio-jacobian-lens),
or the [integrated Laurel/Yanny replay](https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny).
The repository is based on
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) and
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html).

J-lens labels are fitted output-space readouts, not calibrated probabilities
or a literal transcript of what a model is thinking. Read
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) before interpreting them, and read
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) before changing the research workflow.

## Install

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[audio,dev]'
```

The checked-in lockfile can be used instead:

```bash
uv sync --extra audio --extra dev
```

## Run the local explorer

Start the frontend in demo mode. This needs no model download or fitted lens.

```bash
.venv/bin/audio-jlens
```

Open <http://127.0.0.1:8000>. Bundled samples and cached UI data work in this
mode; analyzing an uploaded or recorded clip requires a compatible fitted
lens.

For model-backed Whisper Tiny analysis:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --lens artifacts/whisper_tiny_en.pt
```

The server binds to `127.0.0.1` by default. It accepts audio up to 30 seconds;
WAV, AIFF, FLAC, and OGG use `soundfile`, while formats such as MP3, M4A, and
WebM require a local `ffmpeg` installation.

## Fit and evaluate a Whisper lens

The fitter consumes JSON Lines with audio paths relative to the manifest:

```json
{"audio":"audio/001.wav","text":"The spoken transcript.","speaker":"s01"}
```

Fit the encoder and decoder transports:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-fit \
  data/my_audio/manifest.jsonl \
  --output artifacts/whisper_tiny_en.pt \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063
```

Evaluate on disjoint held-out clips:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/audio-jlens-eval \
  data/my_audio/held_out.jsonl \
  --lens artifacts/whisper_tiny_en.pt \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --output artifacts/eval.json
```

Fitting resumes from `.checkpoints/` after interruption. Artifacts record the
model revision, estimator settings, corpus fingerprint, preprocessing details,
and fitting-example hashes. Evaluation rejects fitting/evaluation overlap by
default.

## Optional local workspaces

- Apple-silicon MLX speech-to-speech setup: [`docs/MLX_LFM.md`](docs/MLX_LFM.md)
- Chatterbox TTS fitted speech-code lens: [`docs/CHATTERBOX.md`](docs/CHATTERBOX.md)
- Whisper residual-intervention CLI: [`docs/CAUSAL_TRACE.md`](docs/CAUSAL_TRACE.md)

These workspaces use separate model stacks and should not be installed into the
base environment unless needed.

## Self-host with Docker

```bash
docker build -t audio-jacobian-lens .
docker run --rm -p 7860:7860 audio-jacobian-lens
```

From another terminal:

```bash
curl http://127.0.0.1:7860/api/status
```

This starts demo mode. Mount a fitted lens or configure a distributable Hub
artifact to enable model-backed analysis. See [`docs/HOSTING.md`](docs/HOSTING.md)
for Docker, Hugging Face Space, custom-domain, privacy, and artifact settings.
See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) to rebuild and validate the
backend-free public explorers.

## Documentation

- [`docs/README.md`](docs/README.md): documentation map
- [`docs/experiments/README.md`](docs/experiments/README.md): canonical experiment reports
- [`PROJECT_PLAN.md`](PROJECT_PLAN.md): milestones, decisions, and chronological work log
- [`samples/README.md`](samples/README.md): sample provenance and licenses

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

Platform-specific MLX tests may skip outside Apple silicon. Do not commit
`artifacts/`, model weights, fitted lenses, local recordings, or generated
evaluation outputs.

## License

Code is available under the [Apache License 2.0](LICENSE). Existing source files
retain their original notices. Model weights, tokenizers, fitted lenses, and
audio datasets remain subject to their own licenses.

## Citation

If Audio Jacobian Lens contributes to your work, please cite this repository
and the original Jacobian Lens paper:

```bibtex
@software{li2026audiojacobianlens,
  author={Li, Wang Yau Kenneth},
  title={Audio Jacobian Lens},
  year={2026},
  url={https://github.com/kennethli319/audio-jacobian-lens},
  license={Apache-2.0}
}

@article{gurnee2026verbalizable,
  author={Gurnee, Wes and Sofroniew, Nicholas and Pearce, Adam and Piotrowski, Mateusz and Kauvar, Isaac and Chen, Runjin and Soligo, Anna and Bogdan, Paul and Ong, Euan and Wang, Rowan and Thompson, Ben and Abrahams, David and Kantamneni, Subhash and Ameisen, Emmanuel and Batson, Joshua and Lindsey, Jack},
  title={Verbalizable Representations Form a Global Workspace in Language Models},
  journal={Transformer Circuits Thread},
  year={2026},
  url={https://transformer-circuits.pub/2026/workspace/index.html}
}
```
