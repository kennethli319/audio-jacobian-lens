# Publishing the cached explorers

The public site is a backend-free export of already-computed reports. Visitors
can switch samples, inspect layers, synchronize timelines, and replay recorded
interventions, but the static site never downloads a model, uploads user audio,
or performs inference.

## Public route contract

- `/audio-jacobian-lens/` is the canonical ten-report ASR explorer.
- `/audio-jacobian-lens/speech/` is the canonical ten-report speech-to-speech
  explorer.
- ASR Audio 10 (`asr-laurel-yanny`) contains Original, Yanny, and Laurel
  recorded states in the normal explorer layout.
- `/audio-jacobian-lens/steering/` is a no-index legacy redirect to ASR Audio
  10, not a separate application.
- Findings and `/explorer/{family}/` aliases remain only for saved links and are
  omitted from the primary navigation.
- TTS/Chatterbox is local-only. A release containing a TTS route, TTS report
  cache, or TTS navigation item must fail validation.

The primary header contains ASR and Speech links plus reciprocal resource tags
for the live ASR Hugging Face Space, the project notes, and the GitHub
self-hosting repository. Keep those tags on both canonical explorers and both
saved-link aliases. Do not add the retired standalone steering page or hidden
findings pages back to that menu.

## Prerequisites

Prepare a separate checkout of the personal static-site repository. Start the
model-backed ASR and speech-to-speech servers on ports `8000` and `8001`. The
recorded fitted-phone intervention also requires the private fitted artifacts
described in [`CAUSAL_TRACE.md`](CAUSAL_TRACE.md).

The media and report catalog is `data/static_explorer_catalog_v2.json`.
Publication provenance is in `data/static_public_reports_v1.json`. Confirm the
rights recorded in [`../samples/README.md`](../samples/README.md) before
changing either catalog.

## Ordered release pipeline

From this repository, replace the example site path below with the
`audio-jacobian-lens` directory inside the personal-site checkout:

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

Keep this order. `export_static_explorer.py` preserves a previously integrated,
manifest-bound replay, but a clean ASR export contains only the base report and
prints a reminder to run `publish_static_asr_replay.py`. That publisher reduces
the three recorded runs, attaches the full Original/Yanny/Laurel encoder,
decoder, and HEAD matrices, copies the attributed Audio S7, and re-hashes the
ASR manifest. Never publish the intermediate base export.

`publish_static_phone_steering.py` preserves the sanitized archival payload and
writes the old `/steering/` route as a redirect. It does not restore a
standalone steering UI.

ASR and speech-to-speech exports run sequentially. To recover from an
interrupted capture, use repeatable `--only <slug>` and `--resume` options with
`export_static_explorer.py`. The final manifests must still contain the complete
ordered ten-report family for ASR and Speech.

## Export boundary

The exporter whitelists fields, pins model and lens provenance, removes
ephemeral run IDs and generated audio, and hashes each report. Large ASR
character-filter token buckets are split into sidecars and fetched only when a
visitor enables that filter.

The recorded replay contains saved measurements only. Browser buttons replace
complete cached matrices; they do not infer, interpolate a new strength, expose
residual tensors, or load private fitted artifacts. The unchanged Laurel/Yanny
Audio S7 may be published only with the recorded Bosker source, CC BY 4.0
notice, paper attribution, and expected content hash.

The TTS catalog and `scripts/export_static_tts_explorer.py` remain available for
local research. They are deliberately outside the public release pipeline.

## Required validation

Always run the validator against the final static-site checkout:

```bash
.venv/bin/python scripts/validate_static_explorer_site.py \
  ../kennethli319.github.io/audio-jacobian-lens
```

The validation gate checks routes and renderers, all ASR and Speech reports,
matrices and trace coverage, report and asset hashes, referenced media,
recorded-only steering data, provenance labels, and the absence of model API
calls, generated audio, ephemeral identifiers, and public TTS material. A
successful export command without a successful final validation is not a
publishable release.
