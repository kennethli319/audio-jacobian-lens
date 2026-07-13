# Experiment report: fitted-phone residual steering in Whisper

Date: 2026-07-12

Status: **preliminary, clip-specific causal existence results**

## Question

Can a distributed fitted phone signature propose a real encoder-state edit
that changes Whisper's downstream recognition, rather than merely changing a
diagnostic label or adding an output-head bias?

On one ambiguous Laurel/Yanny recording, the answer is yes as an existence
result. A frozen equal-strength phone schedule changes ordinary greedy
generation from `Lily!` to `Yanny!` and transfers across two independently
fitted encoder lenses. A separate target-conditioned search also reaches
`Laurel`, but that result is explicitly weaker and does not transfer exactly.
Neither result is evidence of a universal word-control knob.

## Intervention

For a requested phone, the runner differentiates the complete fitted
phone-prototype contrast through the matching centered encoder lens, Whisper's
decoder final normalization, and output head. This produces a 384-dimensional
direction that is normalized and added to real post-block encoder residuals.
All subsequent encoder blocks, the complete decoder, the ordinary output head,
and greedy generation are then rerun.

There is no LM-head bias, forced output token, decoder edit, weight update, or
generation constraint. The phone prototype is a readout-space analysis object;
its pullback is a proposed direction, not a native phone axis guaranteed by the
architecture.

## Recording and baseline

The test clip is the unchanged **Audio S7** from Hans Rutger Bosker's
[Laurel or Yanny? demonstration](https://hrbosker.github.io/demos/laurel-yanny/).
Its reuse and attribution are documented in [`samples/README.md`](../../samples/README.md).
The model is `openai/whisper-tiny.en` at revision
`87c7102498dcde7456f24cfd30239ca606ed9063`.

Baseline greedy generation is `Lily!`. Whisper tokenizes `Yanny` as two
decisions, token 575 (` Y`) followed by token 7737 (`anny`), so it has no honest
single vocabulary rank. The second probability and rank below are
teacher-forced conditional on the first piece. ` Laurel` is one token, ID
43442.

| Baseline measure | Value |
|---|---:|
| `p(" Y")`; full-head rank | 12.3331%; #3 / 51,864 |
| `p("anny" | " Y")`; full-head rank | 0.26564%; #42 / 51,864 |
| two-piece `Yanny` path, excluding EOS | 0.032761% |
| `p(" Laurel")`; full-head rank | 0.0010898%; #2,463 / 51,864 |

## Equal-strength Yanny schedule

The fitted `Y / AE / N / IY` pullbacks are applied across the active 0.08–0.68
second word region at encoder L0, L1, L2, and L3. All 16 phone/layer bases
receive the same coefficient; no coefficient optimizer is involved. The phone
order and time spans were developed on this clip, so the experiment remains
post-hoc and clip-specific.

| Aggregate edited/reference residual norm | ` Y` | conditional `anny` | Two-piece path | Greedy output |
|---:|---:|---:|---:|---|
| Baseline | #3, 12.3331% | #42, 0.26564% | 0.032761% | `Lily!` |
| 3.1884155% | #1, 49.4310% | #2, 5.70938% | 2.82221% | `Yelly!` |
| 3.1884766% | #1, 49.4315% | #1, 5.70967% | 2.82238% | `Yanny!` |
| **3.5%** | **#1, 51.5962%** | **#1, 7.37133%** | **3.80332%** | **`Yanny!`** |

The numerical boundary is fragile, so 3.5% is the recorded demonstration
point. It raises the two-piece path 116.09 times over baseline and returns
ordinary IDs `[575, 7737, 0]`, corresponding to ` Y`, `anny`, and `!`.

### Checks run after freezing the schedule

- Replacing only A2's fitted phone directions with directions from the
  independently fitted A1 lens, without retuning spans, layers, or coefficient,
  still returns the same `Yanny!` IDs at the boundary and at 3.5%.
- A fresh CPU run reproduces the MPS output IDs and ranks.
- None of ten random schedules at the same coordinates and exact 3.5% norm
  generates `Yanny!`. Nine return `Lily!`, one returns `Yelly!`, and conditional
  `anny` ranks range from #38 to #55.

These checks distinguish the intervention from arbitrary equal-norm noise in a
small control set. Ten random directions are not a complete null distribution,
and A1/A2 share the same Whisper checkpoint and evaluation clip.

## Laurel search

The Laurel path uses 20 fitted `L / AO / R / AH / L` phone/layer directions over
the same active region, but their nonnegative coefficients were optimized
directly against the desired final-head token. At the recorded point:

| Measure | Baseline | Recorded Laurel intervention |
|---|---:|---:|
| Aggregate edited/reference residual norm | 0% | 14.5292% |
| `p(" Laurel")` | 0.0010898% | 10.6043% |
| Full-head rank | #2,463 / 51,864 | #1 / 51,864 |
| Greedy output | `Lily!` | `Laurel` |

Three separately optimized matched random 20-direction bases do not generate
Laurel even at larger 51.6–58.6% budgets, reaching ranks #64, #87, and #131.
However, replacing the fitted directions with the independent A1 set without
retuning gives `Lori`, with Laurel only at rank #10 and 1.6588%. The exact
Laurel flip is therefore a target-conditioned, clip-overfit existence result,
not a cross-fit control axis.

## What changed relative to the earlier token-direction attempt

The historical BPE/prefix intervention could raise candidate contrasts but
stopped at outputs such as `Yay!`; it treated vocabulary directions as the
steering basis. The fitted-phone intervention instead uses the distributed
top-100 phone signature, pulls that readout contrast back into encoder residual
space, edits timed acoustic positions, and then observes the unmodified
downstream token decisions. The successful run does not make the fitted phone
label a model probability or prove that this is Whisper's own causal
factorization.

## Public replay and open gates

The [cached ASR Audio 10 replay](https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny)
switches among complete recorded Original, Yanny, and Laurel encoder, decoder,
and HEAD matrices. It does not run Whisper in the browser, interpolate an
unmeasured strength, or expose the private fitted tensors.

Wrong-time, reverse-sign, unrelated-phone-order, spectral-variant, larger
random-control, held-out-audio, and held-out-model checks remain open. Until
those pass, the defensible conclusion is one-clip causal controllability in a
fitted phonetic subspace—not a general phoneme or word dial.
