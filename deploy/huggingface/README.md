---
title: Audio Jacobian Lens
emoji: 🎧
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
fullWidth: true
header: mini
models:
  - openai/whisper-tiny.en
tags:
  - audio
  - automatic-speech-recognition
  - interpretability
---

# Audio Jacobian Lens

This is the Hugging Face Space card and deployment configuration for the live
Whisper ASR explorer.

- [Open the live explorer](https://kennethli319-audio-jacobian-lens.hf.space/)
- [Browse the cached static explorer](https://kennethli319.github.io/audio-jacobian-lens/)
- [Read the project notes](https://kennethli319.github.io/notes/audio-jacobian-lens/)
- [View the source and full documentation](https://github.com/kennethli319/audio-jacobian-lens)

The fitted J-lens readouts are interpretability diagnostics, not calibrated
phoneme probabilities or a literal transcript of what the model is thinking.
