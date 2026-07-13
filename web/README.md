# Audio Jacobian Lens web client

This directory contains the dependency-free browser client for the local
Whisper Jacobian Lens API. Serve it from the same origin as the API so its
relative endpoints resolve correctly.

## API surface

- `GET /api/status` reports backend readiness.
- `GET /api/samples` returns
  `{samples: [{id, title, description, transcript, duration_seconds, filename,
  media_type, audio_url}]}`. Selecting a sample fetches `audio_url` and sends
  the resulting audio through the normal analysis request. An empty or missing
  sample catalog degrades to a quiet empty state.
- `POST /api/analyze` accepts multipart form data with audio in the `audio`
  field and optional `time_bin_overlap_seconds` (default `0.02`; use `0` for
  non-overlapping windows). Uploads, prepared samples, and microphone recordings
  all use this endpoint.

For a standalone visual preview, serve this directory with any static HTTP
server and choose **Load synthetic UI demo**. Demo analysis never contacts the
analysis endpoint.

The shared navigation links to the static **Showcase** at `/showcase`. The
legacy `/causal` route is an alias for the same replacement so saved links keep
working. The showcase compares frozen, curated ASR, speech-to-speech, and TTS
evidence without calling a model backend, and keeps readouts, raw heads,
diagnostics, and residual interventions semantically separate. Publication and
rights caveats travel with each example.

The earlier Laurel/Yanny causal experiment remains in `causal.html`,
`causal.js`, and `docs/CAUSAL_TRACE.md` as a reproducible project artifact; it
is no longer the primary public-facing study page.

## Interaction model

The primary lens views deliberately avoid a horizontally scrolling matrix.
The encoder uses one proportional waveform-slice slider and the decoder uses a
wrapping token navigator. A pinned selection synchronizes the output/LM-head
token, representative encoder bin, decoder position, waveform overlays, audio
seek position, and fixed layer comparisons. Blue marks the encoder bin; orange
marks the approximate aligned output-token interval. Pointer hover over the
encoder waveform remains a temporary local preview. Arrow, Home, and End keys
work in both source tabs and lens navigators.

Synchronization never invents timing. A token without a supported timing range
still selects the exact output-head and decoder position but leaves the encoder
unchanged. A waveform point in a gap selects the nearest reported token interval
and labels it as nearest/approximate. If token timing is wholly unavailable, the
pinned decoder/output token remains unchanged.

The decoder comparison appends a separate **Output head** card after its J-lens
source layers. That card displays the raw model output after decoder L3, final
normalization, and the LM head. It is visually and semantically distinct from
the L0–L2 J-lens cards.

The microphone flow uses `MediaRecorder`, stops just before the 30-second
backend limit, and exposes explicit recording, review, discard, and analyze
states. Other source controls are locked during capture, and the review action
retains its own recording file so a later sample or upload cannot replace it.
Browsers without the required media APIs, and denied microphone permissions,
receive an actionable fallback message.

## Metric and compatibility rules

The response schema is produced by `jlens/whisper_analysis.py` and mirrored by
the synthetic fixture in `app.js`. Cell arrays use `[layer][position]` ordering.

- Orange is reserved for raw teacher-forced model probability. It uses an
  absolute, labeled 0–100% scale.
- Blue is reserved for J-lens readouts. Card intensity is the score's rank
  **within the same layer across positions**. Encoder navigator-bar height is
  the mean of those independently normalized layer ranks; decoder navigator
  chips have no score intensity. Ranked candidates in the inspector use ordinal
  intensity. These colors are never percentages, cross-layer raw-score
  comparisons, confidence, or causal polarity. Raw scores remain visible as
  numbers.
- `metadata.streams` controls which sections appear. Older responses without
  the field are supported by inferring streams from non-empty layer and cell
  arrays.
- `metadata.display_vocabulary` records the lexical display filter. Ranks in
  cards and the inspector refer to that filtered vocabulary.
- Encoder cells additionally carry `top_tokens_by_length`, with a vocabulary-
  wide top-k for every exact decoded character length. The numeric encoder
  filter merges buckets `1…N` and reranks them, exactly matching a `−∞` mask on
  tokens longer than the session limit. Surrounding whitespace is ignored.
  This remains a phoneme-oriented exploration aid rather than a phoneme
  classifier or original J-Lens feature.
- Decoder L0 and L1 cells carry the same vocabulary-wide exact-length buckets.
  Their separate numeric filter merges and reranks buckets `1…N` before top-k;
  it never filters an already-truncated visible list. Decoder L2 and the output
  head remain unchanged as late-readout and actual-probability controls. This
  display adaptation does not alter Whisper generation.
- `encoder.pooling` records requested and effective window, overlap, and hop.
  The default is a 100 ms window with 20 ms overlap (80 ms hop). Long clips may
  use wider windows to cap the display at 100 bins. Overlapping bins are
  correlated and must not be read as independent evidence.
- Recognized stream score kinds are
  `target_mean_relative_logit_delta` for the encoder and `raw_readout_logit`
  for the decoder. Both are explained in the inspector and neither is rendered
  as probability.
- `transcription.timing_source` and `transcription.timing_quality` control
  whether approximate ranges and seeking are shown. Unavailable timing never
  receives invented exact-looking ranges.
- `audio.model_input_wav` is preferred for playback so the user hears the exact
  mono 16 kHz waveform supplied to Whisper. Legacy responses fall back to the
  source audio and say so explicitly.
