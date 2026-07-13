# Documentation

This directory separates the project's research record from setup and release
instructions. Measured outcomes belong in `docs/experiments/`; implementation
guides describe how to reproduce or extend them without presenting a single
pilot as the project's conclusion.

Before changing the research workflow, read and update
[`PROJECT_PLAN.md`](../PROJECT_PLAN.md). It is the canonical milestone,
decision, and chronological work log.

## Research contracts

- [`METHODOLOGY.md`](METHODOLOGY.md) defines the fitted transports, source and
  target streams, rank semantics, centering, and interpretation limits.
- [`experiments/README.md`](experiments/README.md) indexes canonical,
  failure-inclusive experiment reports.
- [`CAUSAL_TRACE.md`](CAUSAL_TRACE.md) documents the Whisper residual-trace and
  steering protocol.

## Model workspaces

- [`MLX_LFM.md`](MLX_LFM.md) sets up the Apple-silicon LFM2.5
  speech-to-speech vertical slice.
- [`CHATTERBOX.md`](CHATTERBOX.md) sets up the Chatterbox T3 acoustic-code lens,
  trace, forced branch, and residual branch.
- [`../web/README.md`](../web/README.md) documents local web routes and frontend
  data contracts.

## Hosting and release

- [`HOSTING.md`](HOSTING.md) covers Docker, Hugging Face Spaces, custom domains,
  artifact configuration, and privacy boundaries.
- [`PUBLISHING.md`](PUBLISHING.md) is the ordered procedure for regenerating and
  validating the backend-free public explorers.
- [`../samples/README.md`](../samples/README.md) records media provenance,
  licenses, and attribution.

## Current public surface

The public site exposes the cached ASR explorer at the site root and the cached
speech-to-speech explorer at `/speech/`. Both contain ten saved reports and use
the same top-level navigation. The recorded Laurel/Yanny intervention is part
of ASR Audio 10; `/steering/` is retained only as a legacy redirect to that
integrated report. Findings and legacy explorer aliases remain available for
old links but are not promoted in the header.

TTS/Chatterbox remains a local research workspace. No TTS route, cached TTS
report, or TTS navigation item is included in the public build.
