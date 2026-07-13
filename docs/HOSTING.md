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
- `JLENS_PHONE_SIGNATURES_PATH` loads a matching local phone-prototype artifact.
  `JLENS_PHONE_SIGNATURES_FILENAME` downloads it from the same Hub repository
  and revision as the combined lens.
- `JLENS_ASR_ONLY=true` hides and blocks the Speech, TTS, Showcase, and
  steering pages for a focused hosted ASR explorer. It does not change the
  Whisper analysis itself.
- `JLENS_DEVICE`, `JLENS_TOP_K`, `JLENS_TIME_BIN_SECONDS`, and
  `JLENS_TIME_BIN_OVERLAP_SECONDS` override serving defaults.
- `JLENS_ANALYSIS_QUEUE_CAPACITY` enables a bounded FIFO for Whisper analysis.
  The value is the number of waiting jobs; one additional job may be running.
  Keep Uvicorn at one process so every request shares the same loaded model and
  queue.
- `JLENS_ANALYSIS_QUEUE_INITIAL_SECONDS` seeds the approximate browser wait
  estimate. Successful analyses replace it with an exponentially weighted
  recent average.

A fitted lens is a separately licensed artifact. Inspect its metadata and
confirm the rights to its fitting corpus before uploading it anywhere.

## Hugging Face Docker Space

The GitHub README intentionally contains no Hugging Face YAML frontmatter. The
Space-specific card at [`../deploy/huggingface/README.md`](../deploy/huggingface/README.md)
selects the Docker SDK and port `7860`; it must be uploaded as the Space
repository's root `README.md`. The image prefetches the pinned public Whisper
Tiny snapshot during its build. The current deployment is
[`kennethli319/audio-jacobian-lens`](https://huggingface.co/spaces/kennethli319/audio-jacobian-lens),
with the direct application at
[`kennethli319-audio-jacobian-lens.hf.space`](https://kennethli319-audio-jacobian-lens.hf.space).

To deploy another Space, create it with the Docker SDK. Upload the source while
excluding the GitHub README, then map the Space card to the root `README.md`:

```bash
hf auth login
hf upload <user>/<space-name> . . --repo-type space --exclude README.md
hf upload <user>/<space-name> deploy/huggingface/README.md README.md --repo-type space
```

Do not push the source branch directly over the Space's `main` branch: doing so
would replace the required Space card with GitHub's documentation-only README.

For a lens stored in a Hub model repository, add these Space **Variables**:

```text
JLENS_MODEL_REVISION=<immutable Whisper revision>
JLENS_LENS_REPO_ID=<user>/<artifact-repository>
JLENS_LENS_FILENAME=<combined-lens-filename.pt>
JLENS_LENS_REVISION=<immutable artifact commit>
JLENS_PHONE_SIGNATURES_FILENAME=<matching-phone-prototypes.pt>
JLENS_ASR_ONLY=true
JLENS_ANALYSIS_QUEUE_CAPACITY=4
JLENS_ANALYSIS_QUEUE_INITIAL_SECONDS=4
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
The application does not write uploads or reports to application storage. With
the queue enabled, it temporarily retains waiting audio and completed reports
in process memory as described below. Reports include a normalized waveform for
playback, and the platform may process request metadata under its own terms.

The public server accepts requests up to 64 MB, limits decoded audio to 30
seconds, and uses one inference slot. Its optional in-memory queue runs one job
at a time, holds at most the configured number of waiting uploads, caps waiting
audio at 64 MB total, and retains completed reports for at most ten minutes
(with earlier eviction under the terminal-result memory bounds). Raw audio is
dropped when a job starts or is cancelled; completed reports are retained in
memory only long enough for the submitting browser to fetch them.

The queue protects model memory and gives visitors a position and approximate
wait; it is not authentication, per-user fairness, or a general rate limiter.
Do not invite sensitive uploads until the public notice, access policy, abuse
controls, and cost limits match the intended audience.

Code licensing does not grant rights to model weights, audio, or fitted lenses.
Bundled sample provenance is recorded in [`../samples/README.md`](../samples/README.md).
