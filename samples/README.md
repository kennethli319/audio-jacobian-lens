# Bundled audio examples

These ten examples are unmodified 16 kHz mono FLAC excerpts from the
LibriSpeech `dev-clean` split, mirrored by Hugging Face's
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
fixture. LibriSpeech is distributed under
[CC BY 4.0](https://www.openslr.org/12/), is derived from public-domain
LibriVox audiobooks, and was prepared by Vassil Panayotov, Guoguo Chen, Daniel
Povey, and Sanjeev Khudanpur.

| Bundled file | LibriSpeech utterance ID | Reference transcript |
|---|---|---|
| `question.flac` | `1272-135031-0012` | WHERE IS MY BROTHER NOW |
| `universe.flac` | `1272-141231-0000` | A MAN SAID TO THE UNIVERSE SIR I EXIST |
| `buzzer.flac` | `1272-141231-0006` | THE BUZZER'S WHIRR TRIGGERED HIS MUSCLES INTO COMPLETE RELAXATION |
| `raps.flac` | `1272-135031-0003` | THE LITTLE GIRL HAD BEEN ASLEEP BUT SHE HEARD THE RAPS AND OPENED THE DOOR |
| `one-minute.flac` | `1272-141231-0004` | ONE MINUTE A VOICE SAID AND THE TIME BUZZER SOUNDED |
| `ten-seconds.flac` | `1272-141231-0016` | TEN SECONDS |
| `oh-no.flac` | `1272-135031-0018` | OH NO I'M QUITE SURE HE DIDN'T |
| `metal-forest.flac` | `1272-135031-0015` | THE METAL FOREST IS IN THE GREAT DOMED CAVERN THE LARGEST IN ALL OUR DOMINIONS REPLIED KALIKO |
| `inexhaustible.flac` | `1272-141231-0018` | A RED HAIRED MOUNTAIN OF A MAN WITH AN APPARENTLY INEXHAUSTIBLE STORE OF ENERGY |
| `impossible.flac` | `1272-141231-0024` | THIS IS PHYSICALLY IMPOSSIBLE WHEN CONSCIOUS |

They are included only as one-click UI examples. They are not part of the
10-clip macOS TTS fitting corpus or the six-clip pilot evaluation split.

The files are materialized reproducibly from the catalog-pinned Parquet and
verified by SHA-256:

```bash
.venv/bin/python scripts/materialize_static_audio_samples.py
```
