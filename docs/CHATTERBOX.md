# Local MLX Chatterbox fitted speech-code J-lens and text trace

The Chatterbox page is a local text-to-speech diagnostic for Chatterbox-Turbo.
It now supports two distinct diagnostics through the causal T3 backbone:

1. a **corpus-fitted, projected speech-code J-lens** that transports an
   intermediate residual at a speech-prediction position to the final T3
   residual basis and decodes it through the model's own final normalization
   and speech head; and
2. a **per-run code-to-text cross-Jacobian sensitivity trace** that measures
   how one realized code's raw log-probability changes with input-text
   residuals and compares that sensitivity with causal self-attention.

Only the first item is fitted. The layer-by-input-token gradient and attention
matrices remain context-specific diagnostics; they are not corpus averages.
Neither view is a word aligner, phoneme decoder, causal contribution score, or
end-to-end attribution of waveform samples.

## Requirements and exact pins

The model-backed page requires an Apple-silicon Mac. MLX is not installed on
other platforms by the project's optional dependency markers.

| Component | Pin |
|---|---|
| Python | `3.12` recommended |
| `mlx` | `0.32.0` |
| `mlx-audio` | `0.4.5` |
| `mlx-lm` | `0.31.3` |
| `transformers` | `5.12.1` |
| model | [`mlx-community/chatterbox-turbo-8bit`](https://huggingface.co/mlx-community/chatterbox-turbo-8bit) |
| model revision | `2f2e21a03863f86a1274d1060dcc188e7cde77e1` |
| S3 tokenizer | [`mlx-community/S3TokenizerV2`](https://huggingface.co/mlx-community/S3TokenizerV2) |
| S3 tokenizer revision | `e0c9886f0e1c35ae85b1f27277416fb19fc72bec` |

Create a dedicated environment so these pins do not alter the verified Whisper
environment:

```bash
python3.12 -m venv .venv-mlx
UV_PROJECT_ENVIRONMENT=.venv-mlx uv sync --extra audio --extra mlx
```

Without `uv`, install the same project extras directly:

```bash
.venv-mlx/bin/python -m pip install -e '.[audio,mlx]'
```

## Fit the projected speech-code J-lens

The repository includes a small manifest with separate `fit` and `heldout`
records. This command reproduces the local ten-prompt pilot configuration:

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

The CLI records the exact teacher-forced text and speech-code trajectories in
an adjacent `.examples.jsonl` file. It does not run S3Gen or the vocoder while
fitting because those modules are outside the T3 residual-to-speech-head map.

For source layer `l`, selected speech-prediction positions `q_i`, and final T3
post-block residual `H_23`, the fitted estimator is:

```text
J_l = mean_examples mean_i d(sum_j H_23[q_j]) / d H_l[q_i]
```

The implementation probes its 1,024-dimensional output with a seeded subset
`R` of an orthogonal Hadamard basis, stores `S_l = R J_l`, and applies
`J_hat_l = R^T S_l / rank`. Rank 1,024 reconstructs this dense estimator
exactly; rank 128 is an approximation. One zero perturbation is inserted after
each requested source block, so one MLX VJP obtains every source layer for a
given probe.

The pilot artifact has this exact provenance:

| Field | Value |
|---|---|
| Fit corpus | 10 prompts, 480 selected speech positions |
| Source / target | post-block L0/L4/L8/L12/L16/L20/L22 to post-block L23, before final normalization |
| Estimator | uncentered, target `sum`, equal example mean |
| Projection | Hadamard rank `128/1024`, seed `29` |
| Projection time | `391.859` seconds, excluding `8.549` seconds of trajectory capture |
| Serialized size | `2,104,061` bytes, fp16 factors |
| Model compatibility fingerprint | `47f1c6108840fae0` |
| In-app fitted-lens fingerprint | `67da0e6fef27e310` |
| Serialized file SHA-256 | `ebf46e3e088106e270eff676b09bac9ff3f5ff5206e9e1f924d692ee9a7b2aa8` |
| Prepared-example fingerprint | `9de6342be062cdf4efdf53ae01570cf47f96d0790cb6aa1093e253e4e03bedf7` |

The artifact remains a local pilot and is ignored by version control; the
fingerprints identify the measured file rather than promising that it is
distributed with this repository.

## Run the local page

Load the fitted artifact with `--lens` while keeping all checkpoint identities
explicit:

```bash
.venv-mlx/bin/audio-jlens-chatterbox \
  --model mlx-community/chatterbox-turbo-8bit \
  --revision 2f2e21a03863f86a1274d1060dcc188e7cde77e1 \
  --s3-tokenizer mlx-community/S3TokenizerV2 \
  --s3-tokenizer-revision e0c9886f0e1c35ae85b1f27277416fb19fc72bec \
  --lens artifacts/chatterbox/chatterbox_turbo_10prompt_rank128_seed29.pt \
  --top-k 5 \
  --host 127.0.0.1 \
  --port 8002
```

Open <http://127.0.0.1:8002/chatterbox>. The first start downloads the pinned
model and tokenizer snapshots from Hugging Face; synthesis and tracing then run
locally through Apple Metal. The page also has a synthetic UI demo that does
not load the model. Every value in that demo is fabricated and is labeled as
such.

The model-backed input is limited to 240 characters and, after normalization,
64 text tokens. Generation is deterministic greedy decoding by default:

- seed `7`;
- at most `160` ordinary speech codes;
- temperature `1.0` and top-k `1`;
- repetition penalty `1.2`; and
- the checkpoint's built-in voice.

The probability shown for a selected code is the raw speech-head softmax before
the repetition penalty. It is not a calibrated probability of a word, phoneme,
or waveform slice.

With a compatible artifact loaded, the page also displays one compact blue
timeline strip per fitted layer. Each slice is the fitted full-speech-head
softmax probability assigned to the code that was actually generated at that
position. Its left edge and width use the same nominal time geometry as the
corresponding waveform slice; decoded audio after the final code remains
hatched and unassigned. Selecting a slice opens a candidate inspector with one
row per fitted layer plus the actual output head. Each row shows the bounded
top-k acoustic code IDs and full-vocabulary softmax percentages, and it always
includes the realized code with its global 6,563-entry rank even when that code
falls outside top-k. The orange output-head row is computed from the real raw
T3 speech-head logits before repetition penalty, temperature, top-k, or top-p;
the fitted and actual distributions must not be conflated.

Candidate IDs are not decoded as strings. They are learned acoustic symbols,
not words or published phonemes, and their audible realization depends on the
surrounding code sequence and voice conditioning. Human-readable labels would
require a separately validated alignment or probe.

### Forced-code continuation

After selecting a non-realized ordinary candidate in any fitted-layer or HEAD
row, the separate branch action runs this counterfactual:

1. keep every parent speech code before the selected index `t`;
2. force the selected code ID at the actual T3 output decision at `t`;
3. feed that code through T3's speech embedding and autoregressive cache;
4. greedily choose later codes after repetition penalty and temperature until
   stop or the configured token limit; and
5. replay the complete branch and decode a new waveform through S3Gen.

The main player becomes the branched waveform while the comparison keeps the
parent audio available. The new result is a complete analysis, so its actual
HEAD candidates, fitted J-lens strips, and per-run trace all use the branched
code path.

This operation changes an output decision; it does not edit an intermediate
T3 residual, steer the fitted layer that supplied the candidate, or mutate the
parent's stored logits. The page therefore keeps the candidate-source layer
separate from intervention provenance. The displayed parent rank, probability,
top-1 gap, and minimum additive bias are computed from the parent's raw full
speech-head distribution before repetition penalty, temperature, top-k, or
top-p. The bias value answers how much would have made the replacement the
unique raw-head winner; the implementation directly forces the ID rather than
injecting that bias. Start/stop control tokens and the already-realized code
cannot be selected.

### Gradient-proposed residual steering

The intervention-mode control keeps residual steering separate from forced-code
continuation. A selected candidate nominates target acoustic code `c`; it is not
inserted into the generated sequence. The user then chooses one or more sampled
post-block T3 layers, an anchor speech position, a consecutive forward span,
and a maximum relative residual-norm budget.

For parent speech position `j` and selected layer `l`, let `r_j` be the strongest
raw-head competitor other than `c`. The parent-path direction is

```text
g_l,j = d (z_c - z_r_j) / d H_l[j]
```

and the applied vector at shared strength `a` is

```text
delta_l,j(a) = a ||H_l[j]||_2 g_l,j / ||g_l,j||_2.
```

The server searches progressively larger `a` values, then refines the first
successful interval. Success requires `c` to be the unique raw speech-head
top-1 code at the anchor and also the greedy choice after repetition penalty
and temperature. If the requested cap is exhausted, the strongest attempted
branch is still returned and labeled unsuccessful. A code is never silently
forced in residual mode.

Every selected layer receives a separately normalized edit, so the common
strength is a per-coordinate fraction rather than the norm of the combined
multi-layer schedule. Codes before the anchor remain fixed. The selected
position and later suffix are greedily generated from the edited computation,
then the complete path is replayed with exactly the same residual schedule and
decoded through S3Gen.

Directions for later positions are computed on the parent teacher-forced path
and applied open-loop to the already-changing branch. This makes the schedule
reproducible, but it is not an adaptive optimizer after the branch diverges.
The result compares the target's exact probability and global rank before and
after at every loaded fitted layer and the actual HEAD across the requested
position window. Those fitted rows are diagnostics of the edited states; the
steering direction itself comes from the actual downstream raw-head margin.

Budgets greater than `0.5` times a coordinate's residual norm are flagged as
large, potentially off-manifold edits. A top-1 flip demonstrates what this
specific intervention caused in the quantized local model; it does not prove a
natural acoustic concept, phoneme direction, or robust mechanism.

## Ten-prompt pilot and held-out check

Four manifest records excluded from fitting produced 251 ordinary speech codes
in the held-out aggregate. Rank is one-based among all 6,563 raw speech-head
entries, and top-10 rate is the fraction of positions where the realized code
appeared in the fitted distribution's ten highest entries.

| Source layer | Median realized-code rank | Realized code in top 10 |
|---:|---:|---:|
| L0 | 1,593 | 0.4% |
| L4 | 1,376 | 0.4% |
| L8 | 952 | 2.0% |
| L12 | 466 | 6.4% |
| L16 | 187 | 9.6% |
| L20 | 23 | 32.3% |
| L22 | 12 | 45.8% |

Across those positions, the actual final head assigned the realized code a
mean probability of `15.85%`; the fitted L22 readout assigned it `3.38%` on
average. The monotonic rank improvement is encouraging evidence that the
transport becomes more predictive later in T3. It does not establish that the
probabilities are calibrated or that the pilot is scientifically stable.

## What the per-run text trace measures

Chatterbox-Turbo's T3 component is a 24-layer, 1,024-wide GPT-2-style causal
transformer. Its sequence has this layout:

```text
[voice conditioning | input text | previous generated speech codes]
```

There is no encoder-decoder cross-attention matrix between the text and speech
codes. A speech position can attend causally to the conditioning block, every
input-text position, and its previous speech codes.

The server records the raw logits while greedily generating the speech-code
path. It then replays that complete path with teacher forcing and checks that
the replay logits match the cached generation logits to a maximum absolute
error of `5e-4`. The replay exposes post-block residuals at every sequence
position.

For selected speech-code index `j`, generated code ID `s_j`, source layer `l`,
and input-text position `i`, the gradient diagnostic is:

```text
G_l(j, i) = || d log p(s_j | c_voice, text, s_<j) / d H_l[i] ||_2
```

`H_l` is the post-block residual for the whole concatenated sequence. The
remaining T3 layers, final normalization, and speech head are rerun inside an
MLX vector-Jacobian product. Only the gradient rows belonging to the input text
are displayed. The default page samples source layers 0, 4, 8, 12, 16, 20, and
22. Layer 23 is excluded: after that final block, the remaining normalization
and speech head are positionwise, so a different text position has zero
cross-position gradient to the selected speech position.

The displayed blue "gradient share" normalizes the unsigned L2 magnitudes only
across the visible text tokens for that layer. It is a visualization scale, not
percent causation. If `T` is the set of input-text positions, its exact value is

```text
S_l(j, i) = G_l(j, i) / sum_(r in T) G_l(j, r).
```

The explorer shows this quantity in a blue layer-by-token matrix and shows the
separately normalized within-text self-attention share in an aligned violet
matrix. Rows are sampled T3 layers, columns are tokenizer pieces, and every
cell retains its numeric value. Both color scales map the reported 0–100% share
directly; they are not rescaled to the largest cell or largest token in a row.
Long prompts are divided into repeated column bands so all coordinates remain
visible without horizontal page scrolling. Selecting any cell synchronizes the
focused layer, text token, upper token strip, and inspector.

The separately reported gradient text mass uses the full causal prefix as its
denominator. If `P_j` contains the voice-conditioning positions, input-text
positions, earlier speech-code positions, and the current speech prediction
position available to code `j`, then

```text
M_l^text(j) = sum_(i in T) G_l(j, i)
              / sum_(r in P_j) || d log p(s_j | c_voice, text, s_<j)
                                   / d H_l[r] ||_2.
```

Thus `M_l^text` is the text share of the sum of positionwise gradient norms;
it is not a vector decomposition or percent causation. Raw gradient norms also
depend on the scale and parameterization of each residual coordinate system.
They are not automatically comparable across distant layers, quantization
schemes, or checkpoints. Normalized shares remove a common scalar within one
row but do not remove that broader interpretation limit.

## Nominal output coordinates

The T3 speech vocabulary contains 6,561 ordinary acoustic code IDs. They are
produced at 25 Hz, so the page assigns code `j` the nominal interval:

```text
start = j / 25 seconds
end   = (j + 1) / 25 seconds
```

That is a nominal 40 ms coordinate. S3Gen upsamples each code to two 50 Hz
acoustic frames, but the current page does not differentiate those frames.
S3Gen's contextual encoder, mean-flow decoder, and vocoder can spread one
code's acoustic effect outside its assigned interval. The server also appends
three silence codes before waveform decoding, so decoded audio can extend past
the nominal content duration. The page reports that trailing duration.

Clicking the waveform therefore chooses the closest nominal T3 code; it does
not identify an exact waveform cause or an isolated 40 ms receptive field.

## Fitted readout, gradient, and attention answer different questions

The violet diagnostic reconstructs ordinary causal self-attention at the
selected T3 layer. It takes the selected speech position's query, attends to
the available prefix keys, averages over heads, and extracts the input-text
portion.

- The fitted J-lens asks which speech-code distribution is readable from one
  intermediate speech-position residual using a corpus-averaged transport.
- Gradient magnitude asks how locally sensitive the selected code's raw
  log-probability is to a residual perturbation at a text position after a
  particular layer.
- Attention weight reports routing association inside one attention sublayer.

Attention is not a gradient, contribution score, explanation, forced
alignment, or proof that a token was causally used. A large unsigned gradient
also does not reveal whether increasing a residual component would raise or
lower the selected probability. Agreement between the two views can be a useful
observation, but it does not turn either one into causal evidence.

## Current limitations

- The implementation has no S3 acoustic-frame Jacobian and no end-to-end
  text-to-mel or text-to-waveform attribution. The waveform is used only as a
  synchronized display for nominal T3 code coordinates.
- The fitted artifact is a ten-prompt, one-seed, rank-128 pilot. There is no
  rank sweep, projection-seed replication, direct/identity logit-lens baseline,
  or unquantized-checkpoint comparison yet. The four-prompt held-out aggregate
  is a useful plumbing and trend check, not a completed validation study.
- The input-text gradient and attention matrices remain a context-specific VJP
  trace and routing view. Loading the fitted speech-position artifact does not
  turn either matrix into a corpus-averaged text explanation.
- Acoustic code IDs are learned codec symbols. They have no published
  human-readable phoneme labels and must not be described as phoneme
  probabilities.
- The checkpoint is an 8-bit affine MLX conversion. Its gradients are specific
  to this quantized revision; an unquantized comparison is required before
  making scientific claims about Chatterbox generally.
- Each trace still follows one fixed, teacher-forced path. A forced-code branch
  can compare one changed output decision and its greedy autoregressive suffix,
  but it is not a search over alternatives or a sampling distribution. The
  separate residual branch is a real intermediate-state intervention, although
  its gradient directions are local to the parent path and its future schedule
  is open-loop after divergence.
- The residual experiment has no matched-random control distribution, adaptive
  future-position re-optimization, or unquantized-checkpoint replication yet.
  Large successful edits may be off-manifold rather than evidence that the same
  code is naturally represented at that coordinate.
- Only the checkpoint's built-in voice is exposed by this page. Voice-cloning
  uploads are outside the current interface.
- The service keeps at most two completed runs in process memory by default.
  Restarting the server or eviction removes them; there is no persistent run
  database.

## Local-only and security boundary

The dedicated server binds to `127.0.0.1` by default. Keep it on loopback. It
has one inference slot and input limits, but no user authentication, general
rate limiter, tenant isolation, or production abuse controls. Binding to
`0.0.0.0` or another non-loopback address exposes expensive model execution and
the generated audio endpoint; add authentication, TLS, request and concurrency
limits, and an appropriate reverse proxy before doing so.

Text prompts and generated runs are not intentionally written to disk by the
application. The generated WAV is returned to the browser, and the bounded run
cache stays in memory. Model and tokenizer files are downloaded to the local
Hugging Face cache on first use, so "local inference" does not mean the first
startup is network-free. This Apple-Metal backend is not part of the current
Linux Hugging Face Docker Space.

## Model provenance, licenses, and watermark boundary

Licenses attach to different artifacts independently:

| Artifact | Provenance | Declared license |
|---|---|---|
| this repository's code | Audio Jacobian Lens | Apache-2.0 |
| original model | [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo), revision `749d1c1a46eb10492095d68fbcf55691ccf137cd` | MIT |
| MLX conversion | [`mlx-community/chatterbox-turbo-8bit`](https://huggingface.co/mlx-community/chatterbox-turbo-8bit), revision `2f2e21a03863f86a1274d1060dcc188e7cde77e1`; its card says it was converted from the ResembleAI model with `mlx-audio` 0.2.8 | Apache-2.0 on the conversion's model card |
| MLX runtime code | [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio), installed here as 0.4.5 | MIT |
| S3 tokenizer weights | [`mlx-community/S3TokenizerV2`](https://huggingface.co/mlx-community/S3TokenizerV2), revision `e0c9886f0e1c35ae85b1f27277416fb19fc72bec` | no license metadata is declared on that model card; review before redistribution |

Do not assume this repository's Apache-2.0 license covers downloaded weights,
and do not bundle or redistribute those weights without reviewing their current
terms and notices.

ResembleAI describes PerTh watermarking for its official PyTorch generation
path. The MLX Chatterbox port used here does not invoke PerTh, so this project
does **not** claim that audio generated on the Chatterbox page is watermarked.
Do not present the local MLX output as carrying the official watermark.
