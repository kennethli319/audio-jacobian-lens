# Whisper residual-steering protocol

This guide describes how the repository edits Whisper residual states and how
to reproduce the historical token-direction traces. Measured outcomes,
controls, and their evidence labels live in the canonical
[`experiments/WHISPER_PHONE_STEERING.md`](experiments/WHISPER_PHONE_STEERING.md)
report.

An intervention is a causal edit to a real post-block residual state. It is not
an edit to an encoder token distribution: Whisper's encoder has no vocabulary
softmax. After the edit, the remaining encoder blocks, decoder, final
normalization, output head, and ordinary generation are rerun without changing
model weights.

## Encoder intervention

For encoder layer \(\ell\) and raw 20 ms positions \(S\), the runner applies

\[
h'^{E}_{\ell,s}=h^{E}_{\ell,s}+\delta_{\ell,s},\qquad s\in S.
\]

The edit direction may come from one of two experimental proposals:

- **Token or prefix direction:** transport one or more Whisper vocabulary
  directions back through the fitted encoder J-lens. These strings are
  tokenizer objects, not phoneme labels.
- **Fitted-phone direction:** differentiate a contrast between frozen phone
  prototypes through the matching encoder lens and final decoder
  normalization, yielding a direction in Whisper's 384-dimensional encoder
  residual space.

The second proposal uses the distributed Phone Signature readout; it does not
turn those cosine similarities into probabilities or a trained phoneme head.
Every schedule must state its layers, audio coordinates, direction source, and
residual-norm budget.

## Decoder intervention

Decoder steering uses autoregressive prediction positions instead of audio
time. If the forced prefix has length \(P\), target BPE piece \(c_j\) is edited
at post-block residual position \(P-1+j\):

\[
h'^{D}_{\ell,P-1+j}=h^{D}_{\ell,P-1+j}+\delta_{\ell,j}.
\]

The pilot decoder lens has source layers L0–L2 and target layer L3. L3 is a
downstream state, not another editable source Jacobian. A pre-hook tracks
absolute positions through Whisper's KV cache. Later-piece edits are applied
open-loop at their declared positions even if an earlier generated piece has
already diverged, so teacher-forced conditionals and free generation must be
reported separately.

## Candidate score

For candidate text \(c\) with pieces \(c_1,\ldots,c_n\), the primary
teacher-forced diagnostic is the complete token-path score

\[
\log P_{\mathrm{path}}(c\mid x)=\sum_i\log p(c_i\mid c_{<i},x).
\]

The restricted softmax in the replay normalizes these totals only over the
declared candidate set. It is not probability under Whisper's full output
space, excludes EOS and generation-time processors, and mechanically favors
shorter token paths. The trace therefore also retains absolute token
probabilities, ranks, a length-normalized diagnostic, and the separate free
generation outcome.

## Reproduce a token-direction trace

The attributed Laurel/Yanny source is already bundled as
`samples/laurel-yanny.mp3`; its provenance and reuse terms are in
[`../samples/README.md`](../samples/README.md).

```bash
PYTHONPATH="$PWD" .venv/bin/python -m jlens.causal_whisper_cli \
  --stream encoder \
  --audio samples/laurel-yanny.mp3 \
  --lens artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --model openai/whisper-tiny.en \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --device mps \
  --layers 1,2,3 \
  --start-seconds 0 --end-seconds 0.92 \
  --strength 0.20 \
  --positive ' Yanny' --negative ' Laurel' \
  --random-control-seed 7 \
  --output artifacts/causal/laurel-yanny-encoder.json
```

For decoder coordinates, switch to `--stream decoder --layers 0,1,2` and omit
the audio-time flags. Omitting `--piece-index` applies the schedule to every
positive-target BPE piece; repeat `--piece-index N` for a component condition.
Encoder and decoder percentage budgets are normalized in different residual
spaces, so equal percentages are not equal cross-stream doses.

### Tokenizer-faithful and explicit spans

With only `--start-seconds` and `--end-seconds`, the runner tokenizes the
positive target and divides the interval over its actual pieces. Use repeated
explicit segments only when the experiment calls for nonuniform or overlapping
coordinates:

```bash
--layers 1,2,3 \
--segment '0:0.35: Y' \
--segment '0.35:0.92:anny' \
--positive ' Yanny' --negative ' Laurel'
```

The segment text is still a Whisper token target, not a phone. A decomposition
such as `Y` / `an` / `ny` is a phoneme-inspired proxy and must be labeled as
such.

### Vocabulary-prefix families

Encoder-only prefix mode averages fitted source directions for all ordinary
tokens whose decoded text starts with the supplied substring after leading
whitespace is removed:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m jlens.causal_whisper_cli \
  --stream encoder --layers 1,2,3 \
  --audio samples/laurel-yanny.mp3 \
  --lens artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --device mps --strength 0.20 \
  --prefix-segment '0:0.46:Y' \
  --prefix-segment '0.46:0.92:anny' \
  --negative-prefix La \
  --positive ' Yanny' --negative ' Laurel' \
  --random-control-seed 7 \
  --output artifacts/causal/laurel-yanny-prefix-families.json
```

Prefix matching is case-sensitive, special tokens are excluded, and each
matching token receives equal weight. A textual prefix such as `Y` must never
be described as acoustic phone /j/. Family composition and overlap are stored
in the trace.

## Record and publish the fitted-phone replay

The full fitted-phone recorder consumes private fitted artifacts and writes a
sanitized report without fitted tensors:

```bash
.venv/bin/python scripts/record_whisper_phone_steering_explorer.py --help
.venv/bin/python scripts/publish_static_phone_steering.py \
  --source-root . \
  --site-root /path/to/static-site
```

The published replay contains only recorded Original, Yanny, and Laurel
checkpoints; it never interpolates an unmeasured strength. The canonical public
view is ASR Audio 10. `/steering/` is retained only as a redirect for old links.

## Interpretation requirements

- Include matched-norm random directions at the same layers, coordinates, and
  budget; one random seed is not a control distribution.
- Freeze direction construction, timing, and stopping criteria before the main
  evaluation, or label the outcome target-conditioned and exploratory.
- Report wrong-time, reverse-sign, unrelated-order, spectral, and held-out-audio
  checks where available.
- Treat a transcript flip as evidence for that exact intervention and
  checkpoint, not as a universal semantic control knob.
- Inspect absolute full-vocabulary scores alongside restricted candidate-set
  shares; damaging every alternative can inflate the latter.
- Keep fitted-phone cosine similarities, J-lens token readouts, raw head
  probabilities, and intervention outcomes visually and verbally distinct.
