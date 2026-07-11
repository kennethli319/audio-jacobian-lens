# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""MLX adapter for LiquidAI's LFM2.5 speech-to-speech model.

The MLX runtime is deliberately optional.  Importing :mod:`jlens` on Linux or
in the existing Hugging Face Whisper deployment must not import ``mlx`` or
``mlx_audio``.  This module therefore imports the Apple-local runtime only when
an adapter is constructed or a method actually executes an MLX graph.

The first supported readout is the LFM text head.  Post-block language-model
residuals are captured before ``lfm.embedding_norm`` and decoded through that
normalization plus the model's tied text embedding.  Audio-codebook readouts
and the Depthformer are separate output spaces and are intentionally deferred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_LFM_MODEL_ID = "mlx-community/LFM2.5-Audio-1.5B-8bit"
DEFAULT_LFM_MODEL_REVISION = "a569a7805a8e3eae954c244e54ba811d479a12c2"
DEFAULT_LFM_SYSTEM_PROMPT = "Respond briefly with interleaved text and audio."
LFM_CAPTURE_CONVENTION = "post_block_pre_final_norm"
LFM_TEXT_HEAD = "text_token"
# Text-side control IDs from the pinned MLX-Audio LFM implementation. Audio
# codebook EOS is deliberately absent: its numeric ID belongs to the separate
# audio-code vocabulary and can also be an ordinary ID in the text vocabulary.
LFM_TEXT_CONTROL_TOKEN_IDS = frozenset({7, 128, 130})


def _is_ordinary_lfm_text_token(
    token_id: int,
    tokenizer_special_ids: set[int],
) -> bool:
    return (
        token_id not in tokenizer_special_ids
        and token_id not in LFM_TEXT_CONTROL_TOKEN_IDS
    )


def _runtime_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    vocabulary = tokenizer.get_vocab()
    for token, token_id in sorted(
        vocabulary.items(), key=lambda item: (int(item[1]), str(item[0]))
    ):
        digest.update(int(token_id).to_bytes(8, "big", signed=True))
        encoded = str(token).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    digest.update(
        json.dumps(
            sorted(int(token_id) for token_id in tokenizer.all_special_ids),
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()[:16]


def _as_float32_numpy(value: Any) -> np.ndarray:
    """Evaluate an MLX value and copy it into a host float32 array."""
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


@dataclass(frozen=True)
class LFMGenerationConfig:
    """Deterministic generation policy recorded in every fitted artifact."""

    system_prompt: str = DEFAULT_LFM_SYSTEM_PROMPT
    max_new_tokens: int = 72
    temperature: float = 0.0
    top_k: int = 1
    audio_temperature: float = 0.0
    audio_top_k: int = 1

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0 or self.audio_temperature < 0:
            raise ValueError("generation temperatures must be nonnegative")
        if self.top_k <= 0 or self.audio_top_k <= 0:
            raise ValueError("top-k generation limits must be positive")


def _interleaved_generation_diagnostics(
    *,
    max_new_tokens: int,
    generated_steps: int,
    text_tokens: int,
    audio_frames: int,
    audio_eos_seen: bool,
    final_step_audio_eos: bool,
    text_complete: bool,
) -> dict[str, int | bool | str]:
    """Describe why one MLX interleaved generation iterator stopped."""
    if final_step_audio_eos and text_complete:
        termination_reason = "audio_eos"
    elif generated_steps >= max_new_tokens:
        termination_reason = "budget_exhausted"
    else:
        termination_reason = "model_stop"
    return {
        "termination_reason": termination_reason,
        "budget_exhausted": termination_reason == "budget_exhausted",
        "max_new_tokens": max_new_tokens,
        "generated_steps": generated_steps,
        "text_tokens": text_tokens,
        "audio_frames": audio_frames,
        "audio_eos_seen": audio_eos_seen,
    }


@dataclass(frozen=True)
class LFMLensInputs:
    """One generated LFM conversation replayed for teacher-forced analysis."""

    text_tokens: Any
    audio_features: Any
    audio_codes: Any | None
    modalities: Any
    prediction_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    generated_text_token_ids: tuple[int, ...]
    generated_text: str
    generated_audio: np.ndarray | None
    generated_audio_sample_rate: int
    input_audio_positions: tuple[int, ...]
    duration_seconds: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if getattr(self.text_tokens, "ndim", None) != 2:
            raise ValueError("text_tokens must have shape [1, text_positions]")
        if getattr(self.audio_features, "ndim", None) != 3:
            raise ValueError("audio_features must have shape [1, frames, features]")
        if getattr(self.modalities, "ndim", None) != 2:
            raise ValueError("modalities must have shape [1, sequence_positions]")
        if self.modalities.shape[0] != 1:
            raise ValueError("LFMLensInputs currently requires batch size 1")
        if len(self.prediction_positions) != len(self.target_token_ids):
            raise ValueError("prediction positions must align with target token IDs")
        if not self.prediction_positions:
            raise ValueError("generated sequence has no ordinary text targets")
        total_positions = int(self.modalities.shape[1])
        if min(self.prediction_positions) < 0 or max(self.prediction_positions) >= total_positions:
            raise ValueError("prediction position is outside the interleaved sequence")
        if self.audio_codes is not None and (
            getattr(self.audio_codes, "ndim", None) != 3
            or self.audio_codes.shape[0] != 1
            or self.audio_codes.shape[2] != 8
        ):
            raise ValueError("audio_codes must have shape [1, frames, 8]")
        if self.generated_audio is not None and self.generated_audio.ndim != 1:
            raise ValueError("generated_audio must be a mono waveform")
        if self.generated_audio_sample_rate <= 0:
            raise ValueError("generated_audio_sample_rate must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class LFMForwardTrace:
    """Native MLX arrays needed for capture and VJP replay."""

    input_embeddings: Any
    attention_mask: Any
    convolution_mask: Any
    activations: dict[int, Any]


@dataclass(frozen=True)
class LFMCapturedRun:
    """Detached host tensors consumed by analysis and regression tests."""

    language_residuals: dict[int, torch.Tensor]
    actual_logits: torch.Tensor
    target_token_ids: torch.Tensor


class MLXLFMModel:
    """Lens-facing wrapper over ``mlx_audio``'s ``LFM2AudioModel``.

    The wrapper intentionally calls the explicit Python block loop rather than
    trying to emulate PyTorch hooks.  This also fixes the capture convention:
    each layer value is the residual after that block and before the final RMS
    normalization used by the tied text output head.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        model_id: str,
        model_revision: str,
        model_path: str | Path,
        generation_config: LFMGenerationConfig | None = None,
    ) -> None:
        if not hasattr(model, "audio_encoder") or not hasattr(model, "lfm"):
            raise TypeError("model must expose audio_encoder and lfm modules")
        if not hasattr(model.lfm, "layers") or not hasattr(
            model.lfm, "embedding_norm"
        ):
            raise TypeError("model.lfm must expose layers and embedding_norm")
        if not hasattr(model.lfm, "embed_tokens"):
            raise TypeError("model.lfm must expose the tied text embedding")
        if not hasattr(processor, "tokenizer"):
            raise TypeError("processor must expose a text tokenizer")

        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_path = Path(model_path)
        self.generation_config = generation_config or LFMGenerationConfig()

        self.n_encoder_layers = len(model.audio_encoder.layers)
        self.n_language_layers = len(model.lfm.layers)
        self.encoder_dim = int(model.config.encoder.d_model)
        self.language_dim = int(model.config.lfm.hidden_size)
        self.vocab_size = int(model.config.lfm.vocab_size)
        self.text_head_name = LFM_TEXT_HEAD
        self.input_sample_rate = int(model.config.preprocessor.sample_rate)
        self.encoder_seconds_per_position = float(
            model.config.preprocessor.window_stride
            * model.config.encoder.subsampling_factor
        )
        self.output_sample_rate = int(model.config.sample_rate)
        self.tokenizer_fingerprint = _tokenizer_fingerprint(self.tokenizer)

        config_path = self.model_path / "config.json"
        config_bytes = config_path.read_bytes()
        self.raw_config = json.loads(config_bytes)
        self.quantization = self.raw_config.get("quantization")
        self.model_config_fingerprint = hashlib.sha256(config_bytes).hexdigest()[:16]
        self.weights_fingerprint = f"hf-commit:{model_revision}"
        self.runtime_versions = {
            "mlx": _runtime_version("mlx"),
            "mlx_audio": _runtime_version("mlx-audio"),
            "mlx_lm": _runtime_version("mlx-lm"),
            "transformers": _runtime_version("transformers"),
        }
        fingerprint_payload = {
            "backend": "mlx",
            "model_id": model_id,
            "model_revision": model_revision,
            "weights_fingerprint": self.weights_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "capture_convention": LFM_CAPTURE_CONVENTION,
            "text_head": self.text_head_name,
            "generation": asdict(self.generation_config),
            "runtime_versions": self.runtime_versions,
        }
        rendered = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        )
        self.fingerprint = hashlib.sha256(rendered.encode()).hexdigest()[:16]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_LFM_MODEL_ID,
        *,
        revision: str = DEFAULT_LFM_MODEL_REVISION,
        generation_config: LFMGenerationConfig | None = None,
    ) -> MLXLFMModel:
        """Download a pinned checkpoint and construct the optional MLX runtime."""
        try:
            from huggingface_hub import snapshot_download
            from mlx_audio.sts.models.lfm_audio import (
                LFM2AudioModel,
                LFM2AudioProcessor,
            )
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "MLX LFM support is optional; install the project's 'mlx' extra "
                "on an Apple-silicon Mac"
            ) from exc

        model_path = snapshot_download(repo_id=model_id, revision=revision)
        model = LFM2AudioModel.from_pretrained(model_path)
        processor = LFM2AudioProcessor.from_pretrained(model_path)
        return cls(
            model,
            processor,
            model_id=model_id,
            model_revision=revision,
            model_path=model_path,
            generation_config=generation_config,
        )

    def __repr__(self) -> str:
        return (
            f"MLXLFMModel({self.model_id!r}, encoder_layers={self.n_encoder_layers}, "
            f"language_layers={self.n_language_layers}, "
            f"language_dim={self.language_dim})"
        )

    def lens_metadata(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "backend": "mlx",
            "model_family": "lfm2_audio",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.fingerprint,
            "weights_fingerprint": self.weights_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "runtime_versions": dict(self.runtime_versions),
            "quantization": self.quantization,
            "encoder_layers": self.n_encoder_layers,
            "language_layers": self.n_language_layers,
            "encoder_dim": self.encoder_dim,
            "language_dim": self.language_dim,
            "vocab_size": self.vocab_size,
            "capture_convention": LFM_CAPTURE_CONVENTION,
            "target_head": {
                "name": self.text_head_name,
                "semantic_kind": "text_token",
                "target_stream": "language",
                "vocab_size": self.vocab_size,
            },
            "deferred_heads": {
                "audio_codebooks": int(config.codebooks),
                "audio_vocab_per_codebook": int(config.audio_vocab_size),
            },
            "input_sample_rate": self.input_sample_rate,
            "encoder_seconds_per_position": self.encoder_seconds_per_position,
            "output_sample_rate": self.output_sample_rate,
            "generation": asdict(self.generation_config),
        }

    def prepare_audio(
        self,
        waveform: np.ndarray,
        *,
        sampling_rate: int,
        duration_seconds: float,
        decode_output_audio: bool = True,
    ) -> LFMLensInputs:
        """Generate a deterministic speech response and prepare its replay path."""
        import mlx.core as mx
        from mlx_audio.sts.models.lfm_audio import ChatState, LFMModality
        from mlx_audio.sts.models.lfm_audio.model import (
            AUDIO_EOS_TOKEN,
            TEXT_END_TOKEN,
        )

        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("waveform must be a non-empty mono float array")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text(self.generation_config.system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(mx.array(samples), sample_rate=sampling_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        prompt_positions = len(chat.modalities)

        generated_text_ids: list[int] = []
        ordinary_target_ids: list[int] = []
        prediction_positions: list[int] = []
        generated_steps = 0
        audio_eos_seen = False
        final_step_audio_eos = False
        text_complete = False
        special_ids = set(int(token_id) for token_id in self.tokenizer.all_special_ids)
        for token, modality in self.model.generate_interleaved(
            **dict(chat),
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_k=self.generation_config.top_k,
            audio_temperature=self.generation_config.audio_temperature,
            audio_top_k=self.generation_config.audio_top_k,
        ):
            mx.eval(token)
            generated_steps += 1
            final_step_audio_eos = False
            sequence_position = len(chat.modalities)
            chat.append(token, modality)
            if modality == LFMModality.TEXT:
                token_id = int(token.item())
                generated_text_ids.append(token_id)
                if token_id == TEXT_END_TOKEN:
                    text_complete = True
                if _is_ordinary_lfm_text_token(token_id, special_ids):
                    ordinary_target_ids.append(token_id)
                    prediction_positions.append(sequence_position - 1)
            elif int(token[0].item()) == AUDIO_EOS_TOKEN:
                audio_eos_seen = True
                final_step_audio_eos = True

        if not ordinary_target_ids:
            raise ValueError(
                "LFM generation produced no ordinary text tokens; increase "
                "max_new_tokens or review the generation prompt"
            )

        audio_codes = (
            None
            if not chat.audio_out_codes
            else mx.stack(chat.audio_out_codes, axis=0)[None, :, :]
        )
        generated_audio = None
        usable_audio_frames = [
            frame
            for frame in chat.audio_out_codes
            if int(frame[0].item()) != AUDIO_EOS_TOKEN
        ]
        if decode_output_audio and usable_audio_frames:
            codec_codes = mx.stack(usable_audio_frames, axis=1)[None, :, :]
            decoded = self.processor.decode_audio(codec_codes, codec="detokenizer")
            generated_audio = _as_float32_numpy(decoded).reshape(-1)

        modality_values = [int(value) for value in chat.modalities]
        input_audio_positions = tuple(
            index
            for index, value in enumerate(modality_values)
            if value == int(LFMModality.AUDIO_IN)
        )
        generated_text = self.tokenizer.decode(
            generated_text_ids, skip_special_tokens=True
        ).strip()
        generation_diagnostics = _interleaved_generation_diagnostics(
            max_new_tokens=self.generation_config.max_new_tokens,
            generated_steps=generated_steps,
            text_tokens=len(generated_text_ids),
            audio_frames=len(usable_audio_frames),
            audio_eos_seen=audio_eos_seen,
            final_step_audio_eos=final_step_audio_eos,
            text_complete=text_complete,
        )
        return LFMLensInputs(
            text_tokens=chat.get_text_tokens(),
            audio_features=chat.get_audio_features(),
            audio_codes=audio_codes,
            modalities=chat.get_modalities(),
            prediction_positions=tuple(prediction_positions),
            target_token_ids=tuple(ordinary_target_ids),
            generated_text_token_ids=tuple(generated_text_ids),
            generated_text=generated_text,
            generated_audio=generated_audio,
            generated_audio_sample_rate=self.output_sample_rate,
            input_audio_positions=input_audio_positions,
            duration_seconds=float(duration_seconds),
            metadata={
                "prompt_positions": prompt_positions,
                "generation": asdict(self.generation_config),
                "generation_diagnostics": generation_diagnostics,
                "output_audio_frames": len(usable_audio_frames),
            },
        )

    def build_interleaved_embeddings(self, inputs: LFMLensInputs) -> Any:
        """Build the exact teacher-forced mixed-modality embedding sequence."""
        return self.model._build_interleaved_embeddings(
            inputs.text_tokens,
            inputs.audio_features,
            inputs.audio_codes,
            inputs.modalities,
        )

    def forward_trace(self, inputs: LFMLensInputs) -> LFMForwardTrace:
        """Evaluate and retain every post-block, pre-final-norm LFM residual."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        embeddings = self.build_interleaved_embeddings(inputs)
        attention_mask = create_attention_mask(embeddings, None)
        convolution_mask = create_ssm_mask(embeddings, None)
        hidden = embeddings
        activations: dict[int, Any] = {}
        for layer_index, layer in enumerate(self.model.lfm.layers):
            mask = attention_mask if layer.is_attention_layer else convolution_mask
            hidden = layer(hidden, mask, cache=None)
            activations[layer_index] = hidden
        mx.eval(embeddings, *activations.values())
        return LFMForwardTrace(
            input_embeddings=embeddings,
            attention_mask=attention_mask,
            convolution_mask=convolution_mask,
            activations=activations,
        )

    def unembed_mx(self, residual: Any) -> Any:
        """Apply the native final norm and tied text head to an MLX residual."""
        normalized = self.model.lfm.embedding_norm(residual)
        return self.model.lfm.embed_tokens.as_linear(normalized)

    def unembed(self, residual: Any) -> torch.Tensor:
        """Torch-facing bridge used by the existing top-k analysis helpers."""
        import mlx.core as mx

        if torch.is_tensor(residual):
            host = residual.detach().float().cpu().numpy()
            mlx_residual = mx.array(host)
        elif isinstance(residual, np.ndarray):
            mlx_residual = mx.array(residual.astype(np.float32, copy=False))
        else:
            mlx_residual = residual
        logits = self.unembed_mx(mlx_residual)
        return torch.from_numpy(_as_float32_numpy(logits))

    def capture(self, inputs: LFMLensInputs) -> LFMCapturedRun:
        """Return host residuals and actual text-head logits for generated text."""
        import mlx.core as mx

        trace = self.forward_trace(inputs)
        positions = mx.array(inputs.prediction_positions, dtype=mx.int32)
        final_residual = trace.activations[self.n_language_layers - 1][
            :, positions, :
        ]
        actual_logits = self.unembed_mx(final_residual)[0]
        residuals = {
            layer: torch.from_numpy(_as_float32_numpy(value[0]))
            for layer, value in trace.activations.items()
        }
        return LFMCapturedRun(
            language_residuals=residuals,
            actual_logits=torch.from_numpy(_as_float32_numpy(actual_logits)),
            target_token_ids=torch.tensor(inputs.target_token_ids, dtype=torch.long),
        )

    def validate_projected_lens(self, lens: Any) -> None:
        """Reject a factorized artifact fitted for another runtime/checkpoint."""
        expected = lens.metadata.get("model_fingerprint")
        if expected != self.fingerprint:
            raise ValueError(
                "lens/model fingerprint mismatch: "
                f"lens={expected!r}, model={self.fingerprint!r}"
            )
        if lens.source_stream != "language" or lens.target_stream != "language":
            raise ValueError("the first MLX LFM lens must be language->language")
        if lens.source_dim != self.language_dim or lens.target_dim != self.language_dim:
            raise ValueError("projected lens dimensions do not match the LFM backbone")
