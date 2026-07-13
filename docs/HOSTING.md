# Hosting Audio Jacobian Lens

Use this guide for the model-backed FastAPI application. To publish only cached
reports with no inference server, follow [`PUBLISHING.md`](PUBLISHING.md)
instead.

## Local Docker smoke test

Build and start the demo-only image:

```bash
docker build -t audio-jacobian-lens .
docker run --rm -p 7860:7860 audio-jacobian-lens
```

From another terminal:

```bash
curl http://127.0.0.1:7860/api/status
```

`ready: false` is expected when no fitted lens is configured. The cached
frontend still works.

To test a local combined Whisper lens without copying it into the image:

```bash
docker run --rm -p 7860:7860 \
  -v "$PWD/artifacts:/lens:ro" \
  -e JLENS_MODEL_REVISION=87c7102498dcde7456f24cfd30239ca606ed9063 \
  -e JLENS_LENS_PATH=/lens/whisper_tiny_en.pt \
  audio-jacobian-lens
```

The repository ignores `artifacts/` and `*.pt`; mounting a file locally does not
publish it.

## Lens and model settings

The container accepts these environment variables:

- `JLENS_MODEL` and `JLENS_MODEL_REVISION` select the model and immutable model
  revision.
- `JLENS_LENS_PATH` loads a combined lens already present in the container or a
  mounted volume.
- `JLENS_ENCODER_LENS_PATH` and `JLENS_DECODER_LENS_PATH` load separate local
  artifacts. Do not combine them with `JLENS_LENS_PATH`.
- `JLENS_LENS_REPO_ID` and `JLENS_LENS_FILENAME` download a lens from the Hub at
  startup. Set both together and do not also set `JLENS_LENS_PATH`.
- `JLENS_LENS_REVISION` should pin that Hub artifact to an immutable commit.
- `JLENS_LENS_REPO_TYPE` may be `model`, `dataset`, or `space`; it defaults to
  `model`.
- `JLENS_DEVICE`, `JLENS_TOP_K`, `JLENS_TIME_BIN_SECONDS`, and
  `JLENS_TIME_BIN_OVERLAP_SECONDS` override serving defaults.

A fitted lens is a separately licensed artifact. Inspect its metadata and
confirm the rights to its fitting corpus before uploading it anywhere.

## Hugging Face Docker Space

The repository's YAML frontmatter selects the Docker SDK and port `7860`. The
image prefetches the pinned public Whisper Tiny snapshot during its build. The
current deployment is
[`kennethli319/audio-jacobian-lens`](https://huggingface.co/spaces/kennethli319/audio-jacobian-lens),
with the direct application at
[`kennethli319-audio-jacobian-lens.hf.space`](https://kennethli319-audio-jacobian-lens.hf.space).

To deploy another Space, create it with the Docker SDK and push this repository:

```bash
hf auth login
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space HEAD:main
```

For a lens stored in a Hub model repository, add these Space **Variables**:

```text
JLENS_MODEL_REVISION=<immutable Whisper revision>
JLENS_LENS_REPO_ID=<user>/<artifact-repository>
JLENS_LENS_FILENAME=<combined-lens-filename.pt>
JLENS_LENS_REVISION=<immutable artifact commit>
```

If the artifact repository is private, add a read-scoped `HF_TOKEN` as a Space
**Secret**, never as a public Variable or committed file.

A Public or Protected Space can be viewed by anyone. Protected visibility keeps
the repository private but does not make the running application private. A
Private Space restricts the application and cannot serve as a public embed or
custom domain.

Hardware upgrades are billed while a Space is Starting or Running and may not
sleep by default. Configure a sleep time and load-test representative clips and
concurrency before upgrading. A Pro subscription does not include paid Space
hardware. Do not select GPU hardware until the image has been rebuilt and
tested with a CUDA-capable PyTorch stack. See Hugging Face's
[Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker) and
[hardware](https://huggingface.co/docs/hub/spaces-gpus) documentation.

## Custom domain and embedding

Hugging Face custom domains require Pro, Team, or Enterprise and a Public or
Protected Space. In **Space Settings → Custom Domain**, enter a subdomain, then
add this DNS record with the domain provider:

```text
Type: CNAME
Name: lens
Target: hf.space
```

After the Space setting reports **Ready**, link to the HTTPS URL directly or
embed it from an HTTPS page:

```html
<iframe
  src="https://lens.example.com"
  title="Audio Jacobian Lens"
  allow="microphone"
  loading="lazy"
  style="width: 100%; min-height: 900px; border: 0"
></iframe>
```

The direct `https://<space-subdomain>.hf.space` URL can be used in the same
iframe. See the official [custom-domain](https://huggingface.co/docs/hub/spaces-custom-domain)
and [embedding](https://huggingface.co/docs/hub/spaces-embed) guides.

## Privacy and publication boundary

The localhost statement that audio is processed locally does not apply to a
Space. Uploaded or recorded audio is sent to the Hugging Face-hosted container.
The application does not intentionally persist uploads or cache analysis
results, but it returns a normalized waveform for playback and the platform may
process request metadata under its own terms.

The public server accepts requests up to 64 MB, limits decoded audio to 30
seconds, and uses one inference slot. It has no user authentication or general
rate limiter. Do not invite sensitive uploads until the public notice, access
policy, abuse controls, and cost limits match the intended audience.

Code licensing does not grant rights to model weights, audio, or fitted lenses.
Bundled sample provenance is recorded in [`../samples/README.md`](../samples/README.md).
