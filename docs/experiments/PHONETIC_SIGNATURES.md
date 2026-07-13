# Experiment report: distributed phonetic signatures in Whisper

Date: 2026-07-12

Status: **exploratory development evidence; locked test split untouched**

## Question

Whisper's encoder positions are acoustic-time states, not decoded text-token
positions. Treating one highest-ranked vocabulary readout as the state’s
literal meaning is therefore especially fragile. This experiment asks whether
a *weighted pattern across many fitted J-lens vocabulary coordinates* is more
stable and phone-informative than a single top coordinate.

The result is positive at the level of decodability and matched similarity:
Whisper Tiny encoder states contain strong phone information, and the fitted
encoder-to-decoder lens exposes a reproducible, distributed phone-organized
pattern. This does not show that the ordinary meanings of individual
vocabulary IDs are phonemes, or that Whisper causally uses the displayed
coordinates.

## Setup and separation

- Model: `openai/whisper-tiny.en` at revision
  `87c7102498dcde7456f24cfd30239ca606ed9063`.
- Natural speech: LibriSpeech `dev-clean`, 2,703 utterances, about 5.4 hours.
- Phone boundaries: independently generated Montreal Forced Aligner
  TextGrids. Stressed ARPAbet labels were collapsed to 34 adequately supported
  phones.
- Eligible coordinates: phone midpoints lasting at least 60 ms, sampled on
  Whisper's native 20 ms encoder grid.
- Lens fits: two centered encoder-to-decoder lenses, A1 and A2, fitted on
  disjoint five-speaker halves.
- Prototype training and development evaluation used disjoint ten-speaker
  groups. A further ten-speaker locked test split was not loaded or evaluated.
- Residual probes used 15,687 training and 15,551 development phone rows.
  J-signatures used balanced sets of 3,400 training and 3,400 development rows,
  100 examples per phone with a per-speaker cap.

The fitted transport at encoder layer `l` is affine:

```text
transported = target_mean + J_l (h - source_mean_l)
```

The top-k representation retains the complete-vocabulary highest `k+1` raw
readout-logit deltas. A selected coordinate receives weight
`max(delta_j - delta_(k+1), 0)`, and the resulting sparse vector is L2
normalized. No character filter or softmax percentage is used. Each fitted
lens is compared with five random transports with the same singular values.

## Residual-state phone information

A frozen L2 multinomial logistic regression was trained on the prototype split
and evaluated on different speakers. This establishes decodability, not causal
use.

| Representation | 34-phone macro-F1 | Strict unseen-word macro-F1 | Cosine-centroid macro-F1 |
|---|---:|---:|---:|
| Central log-Mel | 31.9% | 32.9% | 22.9% |
| Encoder convolution | 49.9% | 49.0% | 42.3% |
| L0 residual | 68.0% | 66.1% | 65.0% |
| L1 residual | 82.8% | 80.7% | 80.4% |
| L2 residual | 89.1% | 86.9% | 85.4% |
| L3 residual | 91.0% | 88.2% | 83.8% |

## Distributed top-100 J-signatures

Phone prototypes were learned only on the training speakers and classified by
cosine similarity on development speakers. The strict subset excludes word
types seen anywhere in prototype training or either lens-fit corpus.

| Layer | Lens | Macro-F1 | Strict unseen-word F1 | Random mean / maximum |
|---|---|---:|---:|---:|
| L0 | A1 | 42.6% | 39.8% | 30.3% / 31.6% |
| L0 | A2 | 41.1% | 37.9% | 28.3% / 31.6% |
| L1 | A1 | 69.1% | 70.1% | 59.4% / 61.8% |
| L1 | A2 | 69.0% | 69.2% | 58.4% / 61.2% |
| L2 | A1 | 81.1% | 79.8% | 70.1% / 72.5% |
| L2 | A2 | 80.5% | 79.6% | 70.4% / 71.9% |
| L3 | A1 | 73.7% | 73.2% | 57.0% / 59.4% |
| L3 | A2 | 73.2% | 72.0% | 59.4% / 63.7% |

Random near-full-rank transports remain decodable because they preserve much
of the residual information. The relevant comparison is therefore the fitted
map's organization relative to spectrum-matched maps, not fitted versus chance.

### How much does top-k add at L2?

| Highest coordinates retained | A1 macro-F1 | A2 macro-F1 |
|---:|---:|---:|
| 1 | 63.5% | 63.6% |
| 2 | 70.1% | 69.1% |
| 5 | 75.1% | 74.0% |
| 10 | 76.5% | 75.5% |
| 25 | 78.3% | 77.6% |
| 50 | 80.0% | 78.9% |
| 100 | 81.1% | 80.5% |
| 200 | 82.4% | 81.7% |

Top-100 improves over top-1 by 17.6 and 16.9 macro-F1 points. The signal is not
only a dominant ID plus small corrections: retaining ranks 51–100 alone gives
79.4% / 79.6% macro-F1. The median highest-coordinate energy share is about
12%, median top-ten share is about 58%, and median inverse-Simpson effective
support is about 21 coordinates.

## Matched cross-speaker, cross-word ABX

The fixed evaluation contains 330 row-disjoint triplets, ten for each of 33
phones. Anchor and positive share a phone but differ in speaker and word. The
different-phone negative has the same broad manner class, shares the
positive's speaker, and uses a third word. No representation value enters
triplet selection.

| Lens | Layer | Fitted top-100 ABX | Random mean | Fitted minus random, speaker-clustered 95% CI |
|---|---:|---:|---:|---:|
| A1 | L0 | 63.8% | 59.2% | +4.6 points [-6.4, +14.7] |
| A2 | L0 | 61.8% | 56.9% | +4.9 points [-3.1, +13.6] |
| A1 | L1 | 76.8% | 72.1% | +4.8 points [-1.5, +10.9] |
| A2 | L1 | 77.7% | 71.9% | +5.8 points [-2.4, +12.6] |
| A1 | L2 | **91.5%** | 76.2% | **+15.3 points [+9.3, +21.9]** |
| A2 | L2 | **91.4%** | 76.6% | **+14.7 points [+9.3, +19.5]** |
| A1 | L3 | **90.8%** | 67.1% | **+23.6 points [+18.1, +29.3]** |
| A2 | L3 | **90.2%** | 68.3% | **+21.8 points [+16.5, +27.0]** |

The L2/L3 advantage replicates across the two independent lens fits. L0/L1 do
not show a reliable fitted-over-random advantage because their intervals cross
zero.

## Temporal and replication checks

At L2, classifying the aligned midpoint gives 81.1% / 80.5% macro-F1. Relabeling
the same signature with phones 80 ms earlier falls to 14.7% / 14.8%, and 80 ms
later falls to 11.0% / 10.7%. At ±160 ms it is about 2.1–2.2%. This sharp local
falloff argues against a global utterance or speaker-identity shortcut.

Across A1 and A2, the L2 full-logit phone prototypes have mean cosine 0.971,
top-100 prototypes have mean cosine 0.977, and same-state top-100 weighted
signatures have mean cosine 0.934. This is a lens-fit stability check, not an
independent-corpus replication.

## Interpretation and open gates

The experiment supports a narrow claim: a population pattern over many
vocabulary-aligned J-readout coordinates is a useful and reproducible basis for
describing local phonetic information. It does not assign a human phoneme
meaning to each vocabulary token, establish a calibrated phone probability, or
prove causal use.

The phone boundaries are automatic, the analysis is development-set
exploration, only five spectrum-matched random transports were used, and fine
coarticulation, stress, prosody, and microphone effects are not perfectly
matched. The untouched speaker split and larger null distributions remain
necessary confirmation work.

The browser's phone-signature labels are cosine similarity to frozen train-only
prototypes. Blue intensity is a within-layer display percentile, not a model
probability. The public prototype bank contains 34 normalized patterns derived
from 3,400 train-only states; it does not include row-level recordings or
identities.
