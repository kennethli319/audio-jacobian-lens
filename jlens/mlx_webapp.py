# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Local explorer backend for the optional MLX LFM2.5 runtime."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from jlens.audio_io import decode_audio_bytes
from jlens.mlx_analysis import analyze_mlx_lfm_run
from jlens.mlx_lfm import (
    DEFAULT_LFM_MODEL_ID,
    DEFAULT_LFM_MODEL_REVISION,
    LFMGenerationConfig,
    MLXLFMModel,
)
from jlens.projected_lens import ProjectedCrossJacobianLens
from jlens.webapp import AnalysisBusyError


class MLXLFMAnalysisBackend:
    """Own one local MLX model/lens pair and serialize Metal inference."""

    # mlx-audio's compiled generation path retains the stream owned by the
    # thread that loaded the model. Uvicorn loads this backend and runs its
    # async endpoint on that same server thread.
    requires_server_thread = True

    def __init__(
        self,
        model: MLXLFMModel,
        lens: ProjectedCrossJacobianLens,
        *,
        top_k: int = 5,
    ) -> None:
        model.validate_projected_lens(lens)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.model = model
        self.lens = lens
        self.top_k = top_k
        # The fitted artifact fingerprints its generation policy. Keep that
        # policy distinct from an optional, serving-only duration budget so a
        # longer response does not bypass artifact compatibility validation.
        self.artifact_generation_config = getattr(model, "generation_config", None)
        self.serving_generation_config = self.artifact_generation_config
        # FastAPI executes synchronous endpoints in worker threads, while MLX
        # streams belong to the thread that creates them. Keep one stream per
        # worker instead of carrying the server thread's stream into a worker.
        self._thread_state = threading.local()
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        lens_path: str,
        model_id: str = DEFAULT_LFM_MODEL_ID,
        revision: str = DEFAULT_LFM_MODEL_REVISION,
        top_k: int = 5,
        serving_max_new_tokens: int | None = None,
    ) -> MLXLFMAnalysisBackend:
        if serving_max_new_tokens is not None and serving_max_new_tokens <= 0:
            raise ValueError("serving_max_new_tokens must be positive")
        lens = ProjectedCrossJacobianLens.load(lens_path)
        generation_payload = lens.metadata.get("generation", {})
        if not isinstance(generation_payload, dict):
            raise ValueError("MLX lens generation metadata must be a mapping")
        generation_config = LFMGenerationConfig(**generation_payload)
        model = MLXLFMModel.from_pretrained(
            model_id,
            revision=revision,
            generation_config=generation_config,
        )
        backend = cls(model, lens, top_k=top_k)
        if serving_max_new_tokens is not None:
            serving_config = replace(
                generation_config,
                max_new_tokens=serving_max_new_tokens,
            )
            model.generation_config = serving_config
            backend.serving_generation_config = serving_config
        return backend

    def status(self) -> dict[str, Any]:
        artifact_max_new_tokens = getattr(
            self.artifact_generation_config, "max_new_tokens", None
        )
        serving_max_new_tokens = getattr(
            self.serving_generation_config, "max_new_tokens", None
        )
        return {
            "ready": True,
            "backend": "mlx-lfm",
            "model_id": self.model.model_id,
            "device": "mlx-metal",
            "streams": ["decoder"],
            "capabilities": {
                "input_audio": True,
                "generated_text": True,
                "generated_audio": True,
                "language_jlens": True,
                "audio_encoder_jlens": False,
                "audio_codebook_jlens": False,
            },
            "generation": {
                "artifact_max_new_tokens": artifact_max_new_tokens,
                "serving_max_new_tokens": serving_max_new_tokens,
                "serving_override": (
                    artifact_max_new_tokens is not None
                    and serving_max_new_tokens != artifact_max_new_tokens
                ),
            },
            "message": (
                "projected LFM language lens ready on Apple Metal; "
                "audio-codebook lens deferred"
            ),
        }

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, Any]:
        del time_bin_overlap_seconds  # No encoder-time grid in the first slice.
        decoded = decode_audio_bytes(
            payload, target_rate=self.model.input_sample_rate
        )
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another MLX speech analysis is already running; retry shortly"
            )
        try:
            import mlx.core as mx

            stream = getattr(self._thread_state, "stream", None)
            if stream is None:
                stream = mx.new_thread_local_stream(mx.gpu)
                self._thread_state.stream = stream
            with mx.stream(stream):
                inputs = self.model.prepare_audio(
                    decoded.waveform,
                    sampling_rate=decoded.sampling_rate,
                    duration_seconds=decoded.duration_seconds,
                    decode_output_audio=True,
                )
                return analyze_mlx_lfm_run(
                    self.model,
                    self.lens,
                    inputs,
                    decoded.waveform,
                    top_k=self.top_k,
                )
        finally:
            self._lock.release()
