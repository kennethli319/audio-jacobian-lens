# Methodology: From Language-Model J-Lens to Speech Models

This note records what the published Jacobian lens does, what it does not do,
and how this project adapts it to Whisper ASR, LFM speech-to-speech, and
Chatterbox text-to-speech. It is the technical companion to
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

## 1. The published method

Let \(h_{\ell,t}\in\mathbb{R}^{d}\) be the residual stream at layer \(\ell\)
and token position \(t\), and let \(z_{L,t'}\) be a residual at a late target
layer. A prompt-specific first-order map is

\[
A_{\ell,t,t'} =
\frac{\partial z_{L,t'}}{\partial h_{\ell,t}}
\in\mathbb{R}^{d\times d}.
\]

The paper averages this causal derivative across a broad context distribution:

\[
J_\ell =
\mathbb{E}_{p,t,t'\geq t}
\left[
\frac{\partial z_{L,t'}}{\partial h_{\ell,t}}
\right].
\]

The released estimator implements the position reduction as a sum over valid
current-and-future target positions followed by a mean over valid source
positions:

\[
J_\ell^{(p)} =
\frac{1}{|S_p|}
\sum_{t\in S_p}
\sum_{t'\in S_p,\,t'\geq t}
\frac{\partial z_{L,t'}}{\partial h_{\ell,t}}.
\]

For each target residual dimension it injects the same one-hot cotangent at all
valid target positions, backpropagates once, and averages the resulting source
gradients over position. Several output dimensions are carried in separate
batch elements, so each prompt costs one forward pass and
\(\lceil d/\text{dim_batch}\rceil\) backward passes.

For a held-out activation, the lens replaces all downstream transformer blocks
with the average linear map and then uses the model's normal output path:

\[
\operatorname{lens}_\ell(h) =
\operatorname{softmax}
\left(W_U\,\operatorname{norm}(J_\ell h)\right).
\]

Ignoring the activation-dependent normalization scale, token \(v\)'s score is
proportional to

\[
\langle J_\ell^\top u_v, h\rangle,
\]

so \(J_\ell^\top u_v\) is the token-indexed J-lens direction in the source
layer's coordinates.

The main paper used 1,000 pretraining-like sequences of 128 tokens. It reports
useful gains with as few as 10 prompts and more modest improvement thereafter
on its Sonnet intermediate-recovery and causal evaluations. That result does
not establish that 10 Whisper clips are scientifically adequate.
The main Sonnet recipe targeted the penultimate block because the final block
could introduce next-token calibration artifacts. The public code defaults to
the final block but exposes the target as an option.

## 2. Correct interpretation

The user's initial description is directionally right: the method converts an
intermediate state into vocabulary-token readouts even though that state is not
directly connected to the output head.

The important correction is that it does not merely feed an early state through
the final output head, and it does not report what the model will say with a
calibrated confidence. It first transports the state through a context-averaged
Jacobian, which estimates a general first-order disposition to affect present
or future verbal output.

The safest language is:

> The top tokens name output concepts to which this internal activation is
> causally aligned on average across contexts.

That is different from each of the following:

- the model's actual next-token probability;
- a complete or unique decomposition of the hidden state;
- a context-specific nonlinear causal explanation;
- proof that the model consciously "thought" a word; or
- proof that a displayed direction was used in this example.

The paper strengthens selected interpretations with swaps, steering, ablation,
and clamping. Our audio project should do the same before making functional
claims.

## 3. Why J-space is not an ordinary subspace

There is one token direction per vocabulary item, normally many more directions
than residual dimensions. They can span the full residual stream, so their
ordinary linear span is not a selective space.

The paper instead treats J-space as points approximable by a sparse,
nonnegative combination of token directions, typically with at most about 25
active directions. Geometrically this is a union of low-dimensional cones. The
decomposition is non-unique because the dictionary is overcomplete and its
directions are correlated.

The appendix's displayed formal definition uses `span`/orthogonal projection
while its prose and experimental procedure specify nonnegative cones and
nonnegative gradient pursuit. We treat the experimental definition as the
operational one and keep the discrepancy visible rather than silently
resolving it.

## 4. Whisper has two different source streams

Whisper Tiny is an encoder-decoder transformer, not a decoder-only language
model. A 30-second log-Mel input has 3,000 feature frames; two convolutions
produce 1,500 encoder positions, one every 20 ms. Tiny has four encoder blocks,
four decoder blocks, residual width 384, and a vocabulary output head tied to
the decoder token embeddings. `whisper-tiny.en` has 51,864 vocabulary entries;
the multilingual `whisper-tiny` checkpoint has 51,865.

This produces two distinct questions.

### 4.1 Decoder Jacobian lens

For decoder residual \(h^{D}_{\ell,u}\) and late decoder residual
\(z^{D}_{L,u'}\), define

\[
J^{D}_\ell =
\mathbb{E}_{x,y,u,u'\geq u}
\left[
\frac{\partial z^{D}_{L,u'}}
     {\partial h^{D}_{\ell,u}}
\right].
\]

Here \(x\) is audio, \(y\) is the supplied decoder sequence, and \(u\) is a
decoder token position. This is the closest analogue to the published method.
The causal decoder mask automatically makes derivatives for \(u'<u\) zero.
Applying \(J^{D}_\ell\), the decoder final norm, and the tied output head yields
a token readout at each decoder position.

Viewing and fitting must record the decoder-prefix policy:

- **Teacher-forced prefix:** controlled and useful for evaluation, but may give
  the decoder correct earlier transcript tokens it would not generate itself.
- **Generated prefix:** faithful to the actual run, but mistakes change all
  downstream states and make comparisons harder.

We fit the decoder lens on teacher-forced prefixes and show generated-prefix
states in the main interactive view. A side-by-side teacher-forced view remains
a planned control.

Layer index `l` always refers to the residual returned **after** transformer
block `l`; the adapter's hooks and the target layer use this same post-block
convention.

### 4.2 Encoder-to-decoder Jacobian lens

For encoder residual \(h^{E}_{\ell,s}\) at audio-frame position \(s\), define

\[
J^{E}_\ell =
\mathbb{E}_{x,y,s,u}
\left[
\frac{\partial z^{D}_{L,u}}
     {\partial h^{E}_{\ell,s}}
\right].
\]

This map is generally rectangular,
\(J^{E}_\ell\in\mathbb{R}^{d_D\times d_E}\), although Whisper Tiny uses
\(d_D=d_E=384\). The token direction in encoder coordinates is
\((J^{E}_\ell)^\top u_v\).

There is no text-style \(u\geq s\) constraint: the encoder is bidirectional and
the decoder can cross-attend to every encoder position. We implement and retain
two distinct estimators.

The **global** estimator sums over valid decoder targets and averages over all
valid, unpadded encoder frames. It is the simplest fixed
audio-to-verbalization map but can blur temporal effects.

The default **aligned** estimator samples one generated ordinary-text target
\(u_p\) per clip, obtains its cross-attention/DTW time interval \(\tau_p\), and
averages only over a small source-frame window \(W(\tau_p)\):

\[
J^{EA}_\ell =
\mathbb{E}_{x,p,s\in W(\tau_p)}
\left[
\frac{\partial z^{D}_{L,p}}
     {\partial h^{E}_{\ell,s}}
\right].
\]

The target-selection rule is deterministic across the corpus so interrupted
fits resume exactly. Both estimators remain experimental. Future variants are
coarse relative-position lenses, frozen-QK gradients, and a context-specific
input Jacobian displayed separately as attribution rather than a reusable lens.

Unlike same-stream residual transport, encoder and decoder coordinates do not
share a meaningful origin. The fitted encoder artifact therefore also stores
the masked corpus means and applies the affine first-order approximation

\[
\widehat z^D_L(h^E_{\ell,s}) =
J^E_\ell\left(h^E_{\ell,s}-\mu^E_\ell\right)+\mu^D_L.
\]

The absolute vocabulary readout of this affine state can be dominated by the
corpus language prior in \(\mu^D_L\). The interactive encoder grid therefore
ranks the baseline-relative logit change

\[
\Delta q_v(h) =
q_v\!\left(J^E_\ell(h-\mu^E_\ell)+\mu^D_L\right)
- q_v(\mu^D_L),
\]

where \(q_v\) includes Whisper's final decoder normalization and output head.
This asks which lexical readouts change most relative to the fitted target
mean. Its sign is not labeled causal support or inhibition, and it remains an
uncalibrated first-order diagnostic.

The global estimator's paper-compatible `target_reduction=sum` makes an
example's target activation and derivative scale with its number of selected
transcript tokens. Final LayerNorm removes much of a common scalar, but this
still introduces transcript-length weighting and is reported as a recipe
choice. `target_reduction=mean` is the planned controlled comparison. For the
aligned one-target estimator, sum and mean coincide.

Because the aligned estimator takes both token identity and timing from
Whisper's own generated sequence, it can learn and display the model's
transcription errors. It is not anchored to the reference transcript.

### 4.3 Chatterbox T3 speech-position Jacobian lens

Chatterbox-Turbo places voice conditioning, text tokens, and previous speech
codes in one causal T3 sequence. Let (q_j) be the position whose residual
predicts ordinary speech code (s_j). The fitted same-stream map is

\[
J^{S}_\ell =
\mathbb{E}_{x,j}
\left[
\frac{\partial \sum_{k\in Q_x} H_{L,q_k}}
     {\partial H_{\ell,q_j}}
\right].
\]

The implementation injects one target probe at all selected target speech
positions, averages the resulting signed gradient over selected source speech
positions, and then averages examples equally. Causality makes gradients from
an earlier target position to a later source position zero, so the summed
target contains only self and future effects for each source. Norms and
absolute values are not taken before corpus averaging.

T3 is 1,024 dimensions wide. For a seeded Hadamard probe matrix
(R\in\{-1,+1\}^{k\times1024}), the artifact stores (S_\ell=R J^S_\ell)
and applies

\[
\widehat J^S_\ell h = \frac{1}{k}R^\top S_\ell h.
\]

Rank 1,024 is the complete orthogonal basis; lower ranks are approximate. The
first retained pilot uses rank 128 and the paper-style uncentered linear map.
The transported state is decoded through Chatterbox's real final T3
normalization and 6,563-way speech head:

\[
p^{\text{lens}}_\ell(c\mid q_j) =
\operatorname{softmax}
\left(W_{\text{speech}}\,\operatorname{ln_f}
\left(\widehat J^S_\ell H_{\ell,q_j}\right)+b\right)_c.
\]

This is a fitted readout distribution over acoustic code IDs. It is not the
base model's actual probability at layer \(\ell\), a phoneme distribution, or
text-to-waveform attribution. The separate text-position gradient matrices
remain context-specific cross-Jacobian diagnostics for one generated run.

## 5. Position masks and timing

The model consumes a padded 30-second feature tensor even for short clips. The
fit must derive a valid encoder mask from the unpadded feature length and reduce
it through the convolution stride. Silence padding must not dominate the
position average.

Decoder masks exclude:

- forced control-prefix states that do not correspond to a free text target;
- padding;
- any final state with no next-token target; and
- optionally timestamp/special tokens in semantic-only ablations.

Whisper's encoder position spacing is 20 ms, but late encoder states have a wide
receptive field and bidirectional attention. The site defaults to five-position
(100 ms) display windows with one-position (20 ms) overlap, giving an 80 ms
hop. A zero-overlap control is available at analysis time. For long clips the
window is widened enough to retain full coverage within 100 display bins; the
response records requested and effective window, overlap, and hop values.
Overlap makes boundary transitions smoother but does not create new temporal
resolution: adjacent readouts share residual positions and are correlated. The
site labels windows as audio locations, not instants when the model formed a
belief.

Generated text tokens can be aligned with audio using Whisper's token timestamp
support, which derives timing from cross-attention and dynamic time warping.
Using that same Whisper-derived alignment for both estimator fitting and
localization evaluation is circular. Release-quality localization evidence
must use external forced alignment or known independently generated boundaries.

The interactive site uses those ranges only for coordinated navigation. A
token selection maps its reported midpoint to a representative encoder display
bin; a waveform selection prefers a token interval that covers the selected
time. A gap uses the nearest reported interval and labels that relationship as
nearest and approximate; wholly missing timing is left unmatched. The blue
waveform overlay is the selected encoder bin and the orange band is the token
interval. No mapping is inferred from proportional sequence position.

## 6. Confidence and salience shown side by side

At each decoder step the site computes raw teacher-forced logits and displays:

- chosen-token probability and log-probability;
- entropy;
- top alternatives and probability margin; and
- top alternatives. Sequence/segment diagnostics remain planned.

Separately, each encoder or decoder lens cell should display:

- top token strings and ranks;
- raw or normalized lens scores;
- the realized output token's exact score/rank when that path retains it (now
  implemented for the Whisper ASR and LFM speech-to-speech matrices); and
- lens artifact provenance and corpus size.

The J-lens softmax can be useful for ranking but is not calibrated output
confidence. Whisper and LFM lens grids therefore retain raw readout scores.
Chatterbox's fitted panel is an explicit exception: it displays the full
speech-head softmax of fitted logits as a percent so it can be compared with
the same realized code under the real head. It is labeled fitted readout
probability rather than base-model confidence.

The interface therefore uses two intentionally different visual encodings.
Raw chosen-token probability uses orange. Whisper and LFM J-lens cells show a
raw logit-like score and use blue intensity normalized within each layer. The
Chatterbox fitted timeline instead puts its blue fitted-softmax slices and
orange real-head slices on one run-relative 0-to-maximum color domain. Exact
values remain available in the synchronized focus readout and accessible slice
labels rather than being printed inside every narrow mark. That shared domain
supports within-run comparison only; it is not comparable across runs or a
calibration claim.

For a selected Chatterbox speech position, a separate candidate inspector shows
the top-k IDs from each fitted full-head softmax and the actual raw output-head
softmax. It also shows the realized ID and its global full-vocabulary rank when
that ID is outside top-k. These are acoustic code candidates, not text-token,
word, or phoneme predictions; start/stop entries are labeled as control tokens.

The optional Chatterbox forced-code branch is a causal intervention on the
autoregressive output decision, not another J-lens score. For position `t`, it
copies parent codes `s_<t`, substitutes one selected ordinary ID for `s_t`,
feeds that ID back into T3, greedily regenerates `s_>t`, and decodes the complete
new code sequence. A fitted layer may nominate the candidate, but that layer's
residual is not edited. Candidate probability, global rank, top-1 logit gap,
and minimum unique-winner bias remain measurements of the parent's raw full
speech head before generation processors; the forced ID and branch waveform
are reported separately as intervention results.

Chatterbox residual steering is a second, non-equivalent intervention. At each
requested post-block layer and speech position it adds a normalized parent-path
gradient of the target-versus-strongest-other raw-head logit margin, scaled by
that coordinate's parent residual norm. A bounded line search looks for the
smallest shared relative strength that makes the target both unique raw top-1
and the processed greedy choice at the anchor. The output ID is never directly
substituted. Future-position directions are computed on the parent path and
applied open-loop after the branch may have diverged. Before/after fitted-lens
probabilities are measurements of the resulting states, not the direction used
to edit them. Raw and processed success, all attempted budgets, and per-coordinate
gradient/residual/delta norms must remain distinct in the report.

The raw model probability is also not the exact generation-time distribution:
Whisper applies suppression and timestamp rules while decoding. The UI states
that distinction explicitly.

For readability, the interactive grids rank over a documented display
vocabulary that removes control/timestamp, empty, replacement-character, and
punctuation-only tokens. Scores remain raw readout logits. The held-out rank
evaluator uses the full vocabulary; browser ranks must therefore be described
as display-filtered lexical ranks, not global vocabulary ranks.

The LFM speech-to-speech matrix computes its realized-token diagnostic from the
complete logits in the same pass as its bounded top-k. An eligible lexical
target uses its exact lexical-display competition rank; an ineligible control
or punctuation-only target uses its exact full-vocabulary rank instead. Both
rank spaces and denominators are retained in the payload. The actual output
HEAD always uses the full-vocabulary rank. No rank is inferred from whether the
target happens to appear in the visible top five.

Whisper decoder cells use the realized output token at the same generated
position. Whisper encoder cells require an approximate cross-stream pairing:
each pooled audio window selects the model-derived cross-attention/DTW token
interval with greatest temporal overlap, breaking ties by closest interval
midpoint and then earlier output position. The payload records that output
position plus overlap seconds/fraction. This is a display synchronization rule,
not an independent word boundary, causal attribution, or real-time belief
trace; Whisper's encoder is bidirectional.

Optional decoded-character filters are computed before top-k selection. The
server stores a vocabulary-wide top-k within every exact trimmed token length,
and the browser merges lengths `1…N` for the selected limit. This is equivalent
to assigning `−∞` to longer display-vocabulary tokens before ranking; it is not
the weaker operation of hiding long strings from an already truncated list.
The encoder can use this view at every displayed layer as a phoneme-oriented
exploration aid. The decoder exposes it only for L0 and L1; L2 and the output
head remain unfiltered late-readout and actual-probability controls. Neither
filter changes Whisper generation or creates a probability distribution.
For Whisper's realized-token badge, the server additionally retains the exact
rank for every cumulative `≤N` vocabulary. A lexical target longer than `N`, or
a nonlexical target excluded from the display vocabulary, has no filtered rank
and is shown as `out`; the unfiltered exact rank remains available in details.

## 7. Cross-page metric glossary

The explorers use a common presentation vocabulary while preserving the
different model architectures and estimators:

| Page | Analysis type | Output coordinate and probability | Intermediate diagnostic | Timing coordinate |
|---|---|---|---|---|
| Whisper ASR | A fitted, corpus-averaged decoder J-lens plus a separately fitted experimental encoder-to-decoder extension | Text-token probability from the Whisper LM head before suppression and timestamp rules | Raw decoder readout logit, or target-mean-relative encoder readout-logit delta; neither is a probability | Approximate generated-token interval, or an overlapping pooled input-audio window |
| LFM speech-to-speech | A rank-limited projected fitted J-lens over the causal language backbone | Generated-text-token probability from the tied text head on the realized interleaved path | Projected language J-lens readout logit; the acoustic codebooks and waveform path are outside the lens | Generated text position; response-token audio timing is unavailable |
| Chatterbox text-to-speech | A projected corpus-averaged J-lens over T3 speech-prediction positions, plus a separate per-run code-to-text cross-Jacobian | Generated speech-code probability and bounded top-k acoustic-code candidates from the real T3 speech head before generation processors; optional forced-code branching changes one output decision, while residual branching edits selected post-block states and lets the head decide | Fitted full-head top-k code candidates plus realized-code probability/rank; separately, raw \(\lVert\nabla_H\log p\rVert_2\), within-text gradient share, causal self-attention, and exact before/after target rank/probability for residual branches | Nominal 25 Hz speech-code step (40 ms), not an isolated waveform frame |

The shared metric terms mean:

- **Output-head probability** is a full-output softmax value from the base
  model's relevant head. It can use a percent sign, but it is not calibrated
  confidence and may precede generation-time processors.
- **J-lens readout score** is a fitted transport followed by an output-head
  readout. Raw logits and baseline-relative logit deltas do not use percent
  signs. When Chatterbox reports the full 6,563-way softmax of those fitted
  logits, that value may use a percent sign but remains a lens distribution,
  not the base model's actual emission probability or calibrated confidence.
- **Display intensity** is a within-view color or rank normalization used for
  navigation. It is not an additional model metric and cannot establish
  cross-layer magnitude.
- **Raw local sensitivity** is Chatterbox's unsigned gradient norm for one
  realized code path. It is separate from the fitted speech-position lens and
  is not an effect direction or causal contribution; residual-coordinate scale
  limits cross-layer comparison.
- **Normalized diagnostic share** states its denominator explicitly. A
  Chatterbox within-text gradient share sums to one only over displayed text
  positions; its text-mass ratio instead uses the full causal prefix. Such
  shares may use percentages but are not output probabilities.
- **Attention mass** is Chatterbox's mean-head causal self-attention routing
  association. It is shown separately from gradients and is not attribution,
  alignment, or proof of model use.
- **Layer `L`** denotes a post-block residual coordinate for J-lens and gradient
  sources. Chatterbox attention at `L` refers to that block's attention
  sublayer, so the two diagnostics must retain their distinct labels.

The corresponding implementation and additional Chatterbox equations are in
[`CHATTERBOX.md`](CHATTERBOX.md); LFM's projected estimator is detailed in
[`MLX_LFM.md`](MLX_LFM.md).

## 8. Validation ladder

1. **Numerical unit tests:** compare the batched estimator with an explicit
   autograd Jacobian on a deterministic tiny encoder-decoder.
2. **Identity/late-layer checks:** decoder maps should approach a residual-like
   transport near the target layer; masking and serialization must be exact.
3. **Held-out descriptive tests:** phoneme/word, homophone, silence, noise,
   temporal-order, and prefix-perturbation examples.
4. **Baselines:** actual logits, direct/untransported decoder unembedding, and
   context-specific gradient attribution where useful.
5. **Causal tests:** matched-norm ablation or steering along surfaced and control
   directions, scored on the relevant output tokens.
6. **Stability tests:** corpus-size convergence, speaker/domain shifts, time
   shifts, and generated-versus-gold prefixes.

## 9. Sources

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in
  Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- Anthropic,
  [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
- Radford et al., [Robust Speech Recognition via Large-Scale Weak
  Supervision](https://cdn.openai.com/papers/whisper.pdf)
- OpenAI, [`openai/whisper`](https://github.com/openai/whisper)
- Hugging Face,
  [Whisper model documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- Hugging Face,
  [`modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)
