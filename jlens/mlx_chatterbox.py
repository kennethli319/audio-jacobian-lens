# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Local MLX Chatterbox-Turbo frame-to-text sensitivity adapter.

Chatterbox T3 is a causal decoder over one concatenated sequence:

``[voice conditioning | input text | previous speech codes]``.

There is no encoder/decoder cross-attention matrix.  For a selected generated
speech code this module replays the exact generated path and uses native MLX
VJPs to measure the L2 norm of the chosen code log-probability gradient at each
input-text position.  This is a per-run cross-Jacobian sensitivity diagnostic,
not the corpus-averaged vocabulary J-lens from the paper and not a causal
contribution score.

The optional MLX runtime is imported lazily so the existing Whisper server and
Linux deployment remain usable without MLX or MLX-Audio installed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from jlens.whisper_analysis import waveform_envelope, waveform_wav_data_url

DEFAULT_CHATTERBOX_MODEL_ID = "mlx-community/chatterbox-turbo-8bit"
DEFAULT_CHATTERBOX_MODEL_REVISION = (
    "2f2e21a03863f86a1274d1060dcc188e7cde77e1"
)
DEFAULT_S3_TOKENIZER_ID = "mlx-community/S3TokenizerV2"
DEFAULT_S3_TOKENIZER_REVISION = (
    "e0c9886f0e1c35ae85b1f27277416fb19fc72bec"
)
DEFAULT_CHATTERBOX_LAYERS = (0, 4, 8, 12, 16, 20, 22)
CHATTERBOX_CAPTURE_CONVENTION = "t3_post_block_speech_prediction_v1"
SPEECH_CODE_RATE_HZ = 25.0
MEL_FRAME_RATE_HZ = 50.0
SPEECH_VOCAB_SIZE = 6561
SPEECH_HEAD_CANDIDATE_SCHEMA_VERSION = 1
RAW_SPEECH_HEAD_NORMALIZATION = (
    "full_speech_head_softmax_before_generation_processors"
)
FORCED_CODE_BIAS_EPSILON = 1e-6
FORCED_CODE_BRANCH_METHOD = "force_output_code_then_greedy_suffix_redecode"
RESIDUAL_BRANCH_METHOD = "parent_path_local_margin_gradient_calibrated"
MAX_RESIDUAL_FORWARD_SPAN = 8
MAX_RELATIVE_RESIDUAL_NORM = 2.0
RESIDUAL_CALIBRATION_LEVELS = 8
RESIDUAL_CALIBRATION_REFINEMENTS = 4
RESIDUAL_NORM_EPSILON = 1e-12


def _runtime_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _file_sha256(path: Path) -> str:
    """Fingerprint a small provenance file without loading it into memory."""
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax_row(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _log_softmax_value(logits: np.ndarray, index: int) -> float:
    """Return one log-softmax value without underflowing through probability space."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logits must be a one-dimensional row")
    if not 0 <= index < values.size:
        raise IndexError("logit index is outside the row")
    maximum = float(np.max(values))
    log_normalizer = maximum + math.log(float(np.exp(values - maximum).sum()))
    return float(values[index] - log_normalizer)


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    scores = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(scores.sum())
    if not math.isfinite(total) or total <= 0:
        return np.zeros_like(scores, dtype=np.float64)
    return scores / total


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
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class ChatterboxGenerationConfig:
    """Deterministic local generation policy recorded in every run."""

    seed: int = 7
    max_speech_tokens: int = 160
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.2

    def __post_init__(self) -> None:
        if self.max_speech_tokens <= 0:
            raise ValueError("max_speech_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass
class ChatterboxCapturedRun:
    """One generated waveform plus evaluated replay states kept in RAM."""

    raw_text: str
    normalized_text: str
    text_token_ids: tuple[int, ...]
    text_tokens: list[dict[str, Any]]
    speech_code_ids: tuple[int, ...]
    raw_logits: np.ndarray
    waveform: np.ndarray
    sample_rate: int
    condition_length: int
    speech_start: int
    input_residual: Any
    post_block_residuals: dict[int, Any]
    replay_max_abs_error: float

    @property
    def duration_seconds(self) -> float:
        return float(self.waveform.size / self.sample_rate)


class MLXChatterboxModel:
    """Lens-facing wrapper around MLX-Audio's Chatterbox-Turbo model."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        model_revision: str,
        model_path: str | Path,
        s3_tokenizer_id: str,
        s3_tokenizer_revision: str,
        s3_tokenizer_path: str | Path,
        generation_config: ChatterboxGenerationConfig | None = None,
    ) -> None:
        if not hasattr(model, "t3") or not hasattr(model, "s3gen"):
            raise TypeError("model must expose Chatterbox T3 and S3Gen modules")
        if not hasattr(model.t3, "tfmr") or not hasattr(model.t3.tfmr, "h"):
            raise TypeError("model.t3 must expose the GPT-2 block list")
        if model.tokenizer is None:
            raise TypeError("model must expose the Chatterbox text tokenizer")
        if getattr(model, "_conds", None) is None:
            raise ValueError("the selected Chatterbox checkpoint has no built-in voice")

        self.model = model
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_path = Path(model_path)
        self.s3_tokenizer_id = s3_tokenizer_id
        self.s3_tokenizer_revision = s3_tokenizer_revision
        self.s3_tokenizer_path = Path(s3_tokenizer_path)
        self.generation_config = generation_config or ChatterboxGenerationConfig()
        self.tokenizer = model.tokenizer
        self.n_layers = len(model.t3.tfmr.h)
        self.hidden_size = int(model.t3.dim)
        self.n_heads = int(model.t3.tfmr.config.n_head)
        self.speech_vocab_size = int(model.t3.hp.speech_tokens_dict_size)
        self.sample_rate = int(model.sample_rate)
        self.tokenizer_fingerprint = _tokenizer_fingerprint(self.tokenizer)
        self.default_lens_layers = DEFAULT_CHATTERBOX_LAYERS

        config_bytes = (self.model_path / "config.json").read_bytes()
        self.raw_config = json.loads(config_bytes)
        self.quantization = self.raw_config.get("quantization")
        self.model_config_fingerprint = hashlib.sha256(config_bytes).hexdigest()[:16]
        self.weights_fingerprint = f"hf-commit:{model_revision}"
        self.voice_conditioning_fingerprint = _file_sha256(
            self.model_path / "conds.safetensors"
        )
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
            "s3_tokenizer_id": s3_tokenizer_id,
            "s3_tokenizer_revision": s3_tokenizer_revision,
            "model_config_fingerprint": self.model_config_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "voice_conditioning_fingerprint": self.voice_conditioning_fingerprint,
            "capture_convention": CHATTERBOX_CAPTURE_CONVENTION,
            "runtime_versions": self.runtime_versions,
        }
        rendered = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        )
        self.fingerprint = hashlib.sha256(rendered.encode()).hexdigest()[:16]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_CHATTERBOX_MODEL_ID,
        *,
        revision: str = DEFAULT_CHATTERBOX_MODEL_REVISION,
        s3_tokenizer_id: str = DEFAULT_S3_TOKENIZER_ID,
        s3_tokenizer_revision: str = DEFAULT_S3_TOKENIZER_REVISION,
        generation_config: ChatterboxGenerationConfig | None = None,
    ) -> MLXChatterboxModel:
        """Load exact local MLX model and S3-tokenizer revisions."""
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from mlx_audio.tts.models.chatterbox_turbo.models.s3tokenizer import (
                S3TokenizerV2,
            )
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "MLX Chatterbox support requires the project's 'mlx' extra on "
                "an Apple-silicon Mac"
            ) from exc

        model_path = Path(snapshot_download(repo_id=model_id, revision=revision))
        s3_path = Path(
            snapshot_download(
                repo_id=s3_tokenizer_id,
                revision=s3_tokenizer_revision,
                allow_patterns=["model.safetensors", "config.json"],
            )
        )
        model = load_model(model_path)

        # MLX-Audio's post-load hook currently resolves the S3 tokenizer from
        # an unpinned Hub branch. Replace it with the explicitly pinned weights
        # before any generation so the final runtime is reproducible.
        pinned_s3 = S3TokenizerV2("speech_tokenizer_v2_25hz")
        s3_weights = mx.load(str(s3_path / "model.safetensors"))
        if hasattr(pinned_s3, "sanitize"):
            s3_weights = pinned_s3.sanitize(s3_weights)
        pinned_s3.load_weights(list(s3_weights.items()), strict=False)
        mx.eval(pinned_s3.parameters())
        model._s3tokenizer = pinned_s3
        model.eval()

        return cls(
            model,
            model_id=model_id,
            model_revision=revision,
            model_path=model_path,
            s3_tokenizer_id=s3_tokenizer_id,
            s3_tokenizer_revision=s3_tokenizer_revision,
            s3_tokenizer_path=s3_path,
            generation_config=generation_config,
        )

    def metadata(self) -> dict[str, Any]:
        special_token_ids = self.special_speech_token_ids()
        return {
            "backend": "mlx",
            "model_family": "chatterbox_turbo",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.fingerprint,
            "weights_fingerprint": self.weights_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "voice_conditioning_fingerprint": self.voice_conditioning_fingerprint,
            "s3_tokenizer_id": self.s3_tokenizer_id,
            "s3_tokenizer_revision": self.s3_tokenizer_revision,
            "runtime_versions": dict(self.runtime_versions),
            "quantization": self.quantization,
            "t3_layers": self.n_layers,
            "t3_width": self.hidden_size,
            "attention_heads": self.n_heads,
            "speech_vocab_size": self.speech_vocab_size,
            "valid_speech_codes": SPEECH_VOCAB_SIZE,
            "capture_convention": CHATTERBOX_CAPTURE_CONVENTION,
            "target_head": {
                "name": "t3.speech_head",
                "semantic_kind": "speech_code",
                "target_stream": "t3_speech_position",
                "vocab_size": self.speech_vocab_size,
                "valid_ordinary_codes": SPEECH_VOCAB_SIZE,
                "special_token_ids": special_token_ids,
            },
            "speech_code_rate_hz": SPEECH_CODE_RATE_HZ,
            "mel_frame_rate_hz": MEL_FRAME_RATE_HZ,
            "sample_rate": self.sample_rate,
            "generation": asdict(self.generation_config),
        }

    def special_speech_token_ids(self) -> dict[str, int]:
        """Return checkpoint-declared speech control IDs when they are valid."""
        hp = self.model.t3.hp
        resolved: dict[str, int] = {}
        for label, attribute in (
            ("start", "start_speech_token"),
            ("stop", "stop_speech_token"),
        ):
            value = getattr(hp, attribute, None)
            if isinstance(value, (int, np.integer)) and 0 <= int(value) < int(
                self.speech_vocab_size
            ):
                resolved[label] = int(value)
        return resolved

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        cleaned = text.strip()
        if len(cleaned) > 240:
            raise ValueError("text must be at most 240 characters for frame tracing")
        return cleaned

    def _tokenize_text(self, raw_text: str) -> tuple[str, Any, list[dict[str, Any]]]:
        import mlx.core as mx
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import punc_norm

        normalized = punc_norm(raw_text)
        encoded = self.tokenizer(
            normalized,
            return_tensors="np",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        token_ids_np = np.asarray(encoded.input_ids, dtype=np.int32)
        if token_ids_np.shape[1] > 64:
            raise ValueError("normalized text exceeds the 64-token trace limit")
        offsets = np.asarray(encoded.offset_mapping, dtype=np.int32)[0]
        tokens: list[dict[str, Any]] = []
        for index, (token_id, offset) in enumerate(
            zip(token_ids_np[0].tolist(), offsets.tolist(), strict=True)
        ):
            start, end = (int(offset[0]), int(offset[1]))
            piece = normalized[start:end]
            if not piece:
                piece = self.tokenizer.decode(
                    [int(token_id)], clean_up_tokenization_spaces=False
                )
            tokens.append(
                {
                    "index": index,
                    "id": int(token_id),
                    "text": piece,
                    "char_start": start,
                    "char_end": end,
                }
            )
        return normalized, mx.array(token_ids_np), tokens

    def _generate_codes(self, text_ids: Any) -> tuple[list[int], np.ndarray]:
        """Run deterministic greedy T3 generation while recording raw logits."""
        import mlx.core as mx

        config = self.generation_config
        mx.random.seed(config.seed)
        t3 = self.model.t3
        bos = mx.full((1, 1), t3.hp.start_speech_token, dtype=mx.int32)
        embeds, _ = t3.prepare_input_embeds(self.model._conds.t3, text_ids, bos)
        hidden, cache = t3.tfmr(inputs_embeds=embeds, cache=None)
        generated: list[int] = []
        raw_logits: list[np.ndarray] = []

        for _ in range(config.max_speech_tokens):
            logits = t3.speech_head(hidden[:, -1, :])
            mx.eval(logits)
            raw_logits.append(np.asarray(logits, dtype=np.float32)[0].copy())

            processed = logits
            if generated and config.repetition_penalty != 1.0:
                processed = t3._apply_repetition_penalty(
                    processed,
                    mx.array([generated], dtype=mx.int32),
                    config.repetition_penalty,
                )
            if config.temperature != 1.0:
                processed = processed / config.temperature

            if config.top_k == 1:
                next_token = mx.argmax(processed, axis=-1)[:, None]
            else:
                next_token = t3._sample_token(
                    logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    generated_tokens=(
                        None
                        if not generated
                        else mx.array([generated], dtype=mx.int32)
                    ),
                    repetition_penalty=config.repetition_penalty,
                )
            mx.eval(next_token)
            token_id = int(next_token[0, 0])
            if token_id == t3.hp.stop_speech_token:
                break
            if token_id >= SPEECH_VOCAB_SIZE:
                raise RuntimeError(
                    f"Chatterbox generated unsupported speech token {token_id}"
                )
            generated.append(token_id)
            hidden, cache = t3.tfmr(
                inputs_embeds=t3.speech_emb(next_token), cache=cache
            )

        if not generated:
            raise RuntimeError("Chatterbox generated no ordinary speech codes")
        return generated, np.stack(raw_logits[: len(generated)])

    def _forced_code_provenance(
        self,
        run: ChatterboxCapturedRun,
        speech_code_index: int,
        replacement_code_id: int,
    ) -> dict[str, Any]:
        """Describe one forced output decision in the parent's raw head space."""
        if not 0 <= speech_code_index < len(run.speech_code_ids):
            raise ValueError("speech_code_index is outside the generated sequence")
        if isinstance(replacement_code_id, bool) or not isinstance(
            replacement_code_id, (int, np.integer)
        ):
            raise ValueError("replacement_code_id must be an integer")

        replacement_code_id = int(replacement_code_id)
        special_ids = self.special_speech_token_ids()
        if replacement_code_id in special_ids.values():
            label = next(
                name
                for name, token_id in special_ids.items()
                if token_id == replacement_code_id
            )
            raise ValueError(
                f"replacement_code_id is the {label} control token; choose an "
                "ordinary acoustic code"
            )
        ordinary_vocab_size = min(SPEECH_VOCAB_SIZE, self.speech_vocab_size)
        if not 0 <= replacement_code_id < ordinary_vocab_size:
            raise ValueError(
                "replacement_code_id must be an ordinary acoustic code in "
                f"[0, {ordinary_vocab_size - 1}]"
            )

        original_id = int(run.speech_code_ids[speech_code_index])
        if replacement_code_id == original_id:
            raise ValueError(
                "replacement_code_id is already the realized code at this position"
            )

        logits = np.asarray(run.raw_logits[speech_code_index], dtype=np.float64)
        if logits.ndim != 1 or logits.size != self.speech_vocab_size:
            raise ValueError(
                "parent raw speech-head logits do not match the speech vocabulary"
            )
        if not np.isfinite(logits).all():
            raise ValueError("parent raw speech-head logits must all be finite")

        probabilities = _softmax_row(logits)
        raw_top1_code_id = int(np.argmax(logits))
        replacement_log_probability = _log_softmax_value(
            logits, replacement_code_id
        )
        replacement_logit = float(logits[replacement_code_id])
        raw_top1_logit = float(logits[raw_top1_code_id])
        replacement_rank = (
            int(np.count_nonzero(logits > replacement_logit)) + 1
        )

        # This is the smallest additive offset that makes the replacement a
        # *unique* raw-head winner.  Excluding the candidate itself matters when
        # it is already top-1; in that case no bias is needed unless it is tied.
        other_logits = np.delete(logits, replacement_code_id)
        competing_logit = float(np.max(other_logits))
        minimum_bias = max(
            0.0,
            competing_logit
            - replacement_logit
            + FORCED_CODE_BIAS_EPSILON,
        )

        return {
            "schema_version": 1,
            "kind": "forced_speech_code_autoregressive_branch",
            "method": FORCED_CODE_BRANCH_METHOD,
            "speech_code_index": speech_code_index,
            "prefix_length": speech_code_index,
            "original_realized_code_id": original_id,
            "replacement_code_id": replacement_code_id,
            "raw_top1_code_id": raw_top1_code_id,
            "raw_top1_probability": float(probabilities[raw_top1_code_id]),
            "replacement_raw_probability": float(
                math.exp(replacement_log_probability)
            ),
            "replacement_raw_log_probability": replacement_log_probability,
            "replacement_global_rank": replacement_rank,
            "replacement_to_raw_top1_logit_gap": (
                raw_top1_logit - replacement_logit
            ),
            "minimum_additive_bias_to_be_unique_raw_top1": minimum_bias,
            "additive_bias_epsilon": FORCED_CODE_BIAS_EPSILON,
            "raw_head_normalization": RAW_SPEECH_HEAD_NORMALIZATION,
            "suffix_policy": (
                "argmax_after_repetition_penalty_and_temperature"
            ),
            "regenerated_suffix_start_index": speech_code_index + 1,
            "parent_speech_code_count": len(run.speech_code_ids),
        }

    def branch_synthesis(
        self,
        parent: ChatterboxCapturedRun,
        speech_code_index: int,
        replacement_code_id: int,
    ) -> tuple[ChatterboxCapturedRun, dict[str, Any]]:
        """Force one emitted code, then greedily regenerate and decode its suffix.

        This intervenes on an autoregressive output decision.  It does not edit
        an intermediate T3 residual or a fitted-lens distribution.
        """
        import mlx.core as mx

        intervention = self._forced_code_provenance(
            parent, speech_code_index, replacement_code_id
        )
        replacement_code_id = int(replacement_code_id)
        config = self.generation_config
        text_ids = mx.array([list(parent.text_token_ids)], dtype=mx.int32)
        mx.random.seed(config.seed)

        t3 = self.model.t3
        bos = mx.full((1, 1), t3.hp.start_speech_token, dtype=mx.int32)
        embeds, _ = t3.prepare_input_embeds(self.model._conds.t3, text_ids, bos)
        hidden, cache = t3.tfmr(inputs_embeds=embeds, cache=None)
        generated: list[int] = []
        raw_logits: list[np.ndarray] = []

        for position in range(config.max_speech_tokens):
            logits = t3.speech_head(hidden[:, -1, :])
            mx.eval(logits)
            raw_logits.append(np.asarray(logits, dtype=np.float32)[0].copy())

            if position < speech_code_index:
                token_id = int(parent.speech_code_ids[position])
            elif position == speech_code_index:
                token_id = replacement_code_id
            else:
                processed = logits
                if generated and config.repetition_penalty != 1.0:
                    processed = t3._apply_repetition_penalty(
                        processed,
                        mx.array([generated], dtype=mx.int32),
                        config.repetition_penalty,
                    )
                if config.temperature != 1.0:
                    processed = processed / config.temperature
                next_token = mx.argmax(processed, axis=-1)[:, None]
                mx.eval(next_token)
                token_id = int(next_token[0, 0])

            if token_id == t3.hp.stop_speech_token:
                break
            if not 0 <= token_id < SPEECH_VOCAB_SIZE:
                raise RuntimeError(
                    f"Chatterbox branch generated unsupported speech token {token_id}"
                )
            generated.append(token_id)
            next_token = mx.array([[token_id]], dtype=mx.int32)
            hidden, cache = t3.tfmr(
                inputs_embeds=t3.speech_emb(next_token), cache=cache
            )

        if len(generated) <= speech_code_index:
            raise RuntimeError(
                "forced Chatterbox branch ended before the replacement position"
            )
        branch_logits = np.stack(raw_logits[: len(generated)])
        replay = self._replay(text_ids, generated, branch_logits)
        waveform = self._decode_waveform(generated)
        branch = ChatterboxCapturedRun(
            raw_text=parent.raw_text,
            normalized_text=parent.normalized_text,
            text_token_ids=parent.text_token_ids,
            text_tokens=[dict(token) for token in parent.text_tokens],
            speech_code_ids=tuple(generated),
            raw_logits=branch_logits,
            waveform=waveform,
            sample_rate=self.sample_rate,
            condition_length=replay[0],
            speech_start=replay[1],
            input_residual=replay[2],
            post_block_residuals=replay[3],
            replay_max_abs_error=replay[4],
        )
        intervention["branch_speech_code_count"] = len(branch.speech_code_ids)
        return branch, intervention

    def _validate_residual_branch_request(
        self,
        parent: ChatterboxCapturedRun,
        speech_code_index: int,
        target_code_id: int,
        layers: Sequence[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> tuple[int, int, tuple[int, ...], int, float]:
        if isinstance(speech_code_index, bool) or not isinstance(
            speech_code_index, (int, np.integer)
        ):
            raise ValueError("speech_code_index must be an integer")
        speech_code_index = int(speech_code_index)
        if not 0 <= speech_code_index < len(parent.speech_code_ids):
            raise ValueError("speech_code_index is outside the generated sequence")

        if isinstance(target_code_id, bool) or not isinstance(
            target_code_id, (int, np.integer)
        ):
            raise ValueError("target_code_id must be an integer")
        target_code_id = int(target_code_id)
        special_ids = self.special_speech_token_ids()
        if target_code_id in special_ids.values():
            label = next(
                name
                for name, token_id in special_ids.items()
                if token_id == target_code_id
            )
            raise ValueError(
                f"target_code_id is the {label} control token; choose an "
                "ordinary acoustic code"
            )
        ordinary_vocab_size = min(SPEECH_VOCAB_SIZE, self.speech_vocab_size)
        if not 0 <= target_code_id < ordinary_vocab_size:
            raise ValueError(
                "target_code_id must be an ordinary acoustic code in "
                f"[0, {ordinary_vocab_size - 1}]"
            )

        if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
            raise ValueError("layers must be a non-empty sequence of integers")
        resolved_layers: list[int] = []
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
                raise ValueError("layers must contain only integers")
            resolved_layers.append(int(layer))
        if not resolved_layers:
            raise ValueError("layers must be non-empty")
        if len(set(resolved_layers)) != len(resolved_layers):
            raise ValueError("layers must not contain duplicates")
        resolved_layers.sort()
        if resolved_layers[0] < 0 or resolved_layers[-1] >= self.n_layers:
            raise ValueError(
                f"layers must be post-block T3 indices in [0, {self.n_layers - 1}]"
            )

        if isinstance(forward_span, bool) or not isinstance(
            forward_span, (int, np.integer)
        ):
            raise ValueError("forward_span must be an integer")
        forward_span = int(forward_span)
        if not 1 <= forward_span <= MAX_RESIDUAL_FORWARD_SPAN:
            raise ValueError(
                f"forward_span must be in [1, {MAX_RESIDUAL_FORWARD_SPAN}]"
            )
        if speech_code_index + forward_span > len(parent.speech_code_ids):
            raise ValueError(
                "speech_code_index + forward_span exceeds the parent speech "
                "sequence"
            )

        if isinstance(max_relative_residual_norm, bool) or not isinstance(
            max_relative_residual_norm, (int, float, np.integer, np.floating)
        ):
            raise ValueError("max_relative_residual_norm must be numeric")
        max_relative_residual_norm = float(max_relative_residual_norm)
        if (
            not math.isfinite(max_relative_residual_norm)
            or not 0 < max_relative_residual_norm <= MAX_RELATIVE_RESIDUAL_NORM
        ):
            raise ValueError(
                "max_relative_residual_norm must be finite and in "
                f"(0, {MAX_RELATIVE_RESIDUAL_NORM}]"
            )
        return (
            speech_code_index,
            target_code_id,
            tuple(resolved_layers),
            forward_span,
            max_relative_residual_norm,
        )

    def _processed_greedy_code_id(
        self, logits: np.ndarray, generated: Sequence[int]
    ) -> int | None:
        """Reproduce the deterministic generation processors before argmax."""
        processed = np.asarray(logits, dtype=np.float64).copy()
        config = self.generation_config
        if generated and config.repetition_penalty != 1.0:
            valid_ids = {
                int(token_id)
                for token_id in generated
                if 0 <= int(token_id) < processed.size
            }
            for token_id in valid_ids:
                if processed[token_id] < 0:
                    processed[token_id] *= config.repetition_penalty
                else:
                    processed[token_id] /= config.repetition_penalty
        if config.temperature != 1.0:
            processed /= config.temperature
        token_id = int(np.argmax(processed))
        if token_id == self.model.t3.hp.stop_speech_token:
            return None
        return token_id

    def _residual_attempt_payload(
        self,
        logits: np.ndarray,
        *,
        attempt_index: int,
        relative_residual_norm: float,
        target_code_id: int,
        generated_prefix: Sequence[int],
    ) -> dict[str, Any]:
        row = np.asarray(logits, dtype=np.float64)
        probabilities = _softmax_row(row)
        target_log_probability = _log_softmax_value(row, target_code_id)
        target_logit = float(row[target_code_id])
        other_logits = np.delete(row, target_code_id)
        strongest_other = float(np.max(other_logits))
        raw_top1_code_id = int(np.argmax(row))
        target_is_raw_top1 = bool(target_logit > strongest_other)
        processed_id = self._processed_greedy_code_id(row, generated_prefix)
        processed_equals_target = processed_id == target_code_id
        return {
            "attempt_index": attempt_index,
            "relative_residual_norm": float(relative_residual_norm),
            "target_probability": float(probabilities[target_code_id]),
            "target_log_probability": target_log_probability,
            "target_rank": int(np.count_nonzero(row > target_logit)) + 1,
            "raw_top1_code_id": raw_top1_code_id,
            "target_logit_margin_to_strongest_other": (
                target_logit - strongest_other
            ),
            "target_is_raw_top1": target_is_raw_top1,
            "processed_greedy_code_id": processed_id,
            "processed_greedy_equals_target": processed_equals_target,
            "success": target_is_raw_top1 and processed_equals_target,
        }

    def _residual_directions(
        self,
        parent: ChatterboxCapturedRun,
        target_code_id: int,
        layers: Sequence[int],
        positions: Sequence[int],
    ) -> tuple[
        dict[tuple[int, int], np.ndarray],
        list[dict[str, Any]],
    ]:
        """Compute parent-path local margin gradients at post-block coordinates."""
        import mlx.core as mx

        transformer = self.model.t3.tfmr
        layer_to_input = {layer: index for index, layer in enumerate(layers)}
        directions: dict[tuple[int, int], np.ndarray] = {}
        diagnostics: list[dict[str, Any]] = []
        zero_deltas = [
            mx.zeros_like(parent.post_block_residuals[layer]) for layer in layers
        ]

        for speech_position in positions:
            query_position = parent.speech_start + speech_position
            parent_logits = np.asarray(
                parent.raw_logits[speech_position], dtype=np.float64
            )
            competitor_logits = parent_logits.copy()
            competitor_logits[target_code_id] = -np.inf
            competitor_code_id = int(np.argmax(competitor_logits))

            def margin(
                *deltas: Any,
                query: int = query_position,
                competitor: int = competitor_code_id,
            ) -> Any:
                residual = parent.input_residual
                for layer_index, block in enumerate(transformer.h):
                    residual, _ = block(residual, cache=None)
                    delta_index = layer_to_input.get(layer_index)
                    if delta_index is not None:
                        residual = residual + deltas[delta_index]
                final = transformer.ln_f(residual)
                logits = self.model.t3.speech_head(
                    final[:, query, :]
                )[0]
                return logits[target_code_id] - logits[competitor]

            _, gradients = mx.vjp(margin, zero_deltas, [mx.array(1.0)])
            vectors = [gradient[0, query_position, :] for gradient in gradients]
            residual_vectors = [
                parent.post_block_residuals[layer][0, query_position, :]
                for layer in layers
            ]
            mx.eval(*vectors, *residual_vectors)
            for layer, vector, baseline_vector in zip(
                layers, vectors, residual_vectors, strict=True
            ):
                vector_np = np.asarray(vector, dtype=np.float64)
                gradient_norm = float(np.linalg.norm(vector_np))
                baseline_norm = float(
                    np.linalg.norm(np.asarray(baseline_vector, dtype=np.float64))
                )
                if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise RuntimeError(
                        "residual steering gradient is zero or non-finite at "
                        f"L{layer}, speech position {speech_position}"
                    )
                if not math.isfinite(baseline_norm) or baseline_norm <= 0:
                    raise RuntimeError(
                        "parent residual norm is zero or non-finite at "
                        f"L{layer}, speech position {speech_position}"
                    )
                directions[(layer, speech_position)] = (
                    vector_np / gradient_norm
                ).astype(np.float32)
                diagnostics.append(
                    {
                        "layer": layer,
                        "speech_code_index": speech_position,
                        "competitor_code_id": competitor_code_id,
                        "gradient_l2_norm": gradient_norm,
                        "baseline_residual_l2_norm": baseline_norm,
                    }
                )
            mx.clear_cache()
        return directions, diagnostics

    def _new_t3_cache(self) -> list[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.n_layers)]

    def _residual_transformer_step(
        self,
        inputs_embeds: Any,
        cache: list[Any] | None,
        layer_edits: dict[int, Any],
    ) -> tuple[Any, list[Any]]:
        """Run one prompt/chunk and add edits after selected T3 blocks."""
        import mlx.core as mx

        transformer = self.model.t3.tfmr
        if cache is None:
            cache = self._new_t3_cache()
            past_length = 0
        else:
            past_length = int(cache[0].offset)
        length = int(inputs_embeds.shape[1])
        positions = mx.arange(past_length, past_length + length)
        residual = inputs_embeds + transformer.wpe(positions)
        for layer_index, block in enumerate(transformer.h):
            residual, _ = block(residual, cache=cache[layer_index])
            delta = layer_edits.get(layer_index)
            if delta is not None:
                if length == 1:
                    residual = residual + delta[None, None, :]
                else:
                    residual = mx.concatenate(
                        [
                            residual[:, :-1, :],
                            residual[:, -1:, :] + delta[None, None, :],
                        ],
                        axis=1,
                    )
        return transformer.ln_f(residual), cache

    def _scaled_residual_edits(
        self,
        directions: dict[tuple[int, int], np.ndarray],
        coordinate_diagnostics: Sequence[dict[str, Any]],
        layers: Sequence[int],
        speech_position: int,
        relative_strength: float,
    ) -> dict[int, Any]:
        import mlx.core as mx

        norm_by_coordinate = {
            (int(entry["layer"]), int(entry["speech_code_index"])): float(
                entry["baseline_residual_l2_norm"]
            )
            for entry in coordinate_diagnostics
        }
        return {
            layer: mx.array(
                directions[(layer, speech_position)]
                * norm_by_coordinate[(layer, speech_position)]
                * relative_strength
            )
            for layer in layers
        }

    def _residual_anchor_logits(
        self,
        parent: ChatterboxCapturedRun,
        speech_code_index: int,
        layers: Sequence[int],
        directions: dict[tuple[int, int], np.ndarray],
        coordinate_diagnostics: Sequence[dict[str, Any]],
        relative_strength: float,
    ) -> np.ndarray:
        import mlx.core as mx

        text_ids = mx.array([list(parent.text_token_ids)], dtype=mx.int32)
        bos = mx.full(
            (1, 1), self.model.t3.hp.start_speech_token, dtype=mx.int32
        )
        prefix = mx.array(
            [list(parent.speech_code_ids[:speech_code_index])], dtype=mx.int32
        )
        speech_inputs = mx.concatenate([bos, prefix], axis=1)
        embeds, _ = self.model.t3.prepare_input_embeds(
            self.model._conds.t3, text_ids, speech_inputs
        )
        edits = self._scaled_residual_edits(
            directions,
            coordinate_diagnostics,
            layers,
            speech_code_index,
            relative_strength,
        )
        hidden, _ = self._residual_transformer_step(embeds, None, edits)
        logits = self.model.t3.speech_head(hidden[:, -1, :])
        mx.eval(logits)
        result = np.asarray(logits, dtype=np.float32)[0].copy()
        mx.clear_cache()
        return result

    def _calibrate_residual_strength(
        self,
        parent: ChatterboxCapturedRun,
        speech_code_index: int,
        target_code_id: int,
        layers: Sequence[int],
        directions: dict[tuple[int, int], np.ndarray],
        coordinate_diagnostics: Sequence[dict[str, Any]],
        max_relative_residual_norm: float,
    ) -> tuple[float, list[dict[str, Any]], bool]:
        generated_prefix = parent.speech_code_ids[:speech_code_index]
        attempts: list[dict[str, Any]] = [
            self._residual_attempt_payload(
                parent.raw_logits[speech_code_index],
                attempt_index=0,
                relative_residual_norm=0.0,
                target_code_id=target_code_id,
                generated_prefix=generated_prefix,
            )
        ]
        if attempts[0]["success"]:
            return 0.0, attempts, True

        strengths = [
            max_relative_residual_norm
            * (2.0 ** (index - RESIDUAL_CALIBRATION_LEVELS + 1))
            for index in range(RESIDUAL_CALIBRATION_LEVELS)
        ]
        last_failed_strength = 0.0
        successful_strength: float | None = None
        for strength in strengths:
            logits = self._residual_anchor_logits(
                parent,
                speech_code_index,
                layers,
                directions,
                coordinate_diagnostics,
                strength,
            )
            attempt = self._residual_attempt_payload(
                logits,
                attempt_index=len(attempts),
                relative_residual_norm=strength,
                target_code_id=target_code_id,
                generated_prefix=generated_prefix,
            )
            attempts.append(attempt)
            if attempt["success"]:
                successful_strength = strength
                break
            last_failed_strength = strength

        if successful_strength is not None:
            low = last_failed_strength
            high = successful_strength
            for _ in range(RESIDUAL_CALIBRATION_REFINEMENTS):
                strength = (low + high) / 2.0
                logits = self._residual_anchor_logits(
                    parent,
                    speech_code_index,
                    layers,
                    directions,
                    coordinate_diagnostics,
                    strength,
                )
                attempt = self._residual_attempt_payload(
                    logits,
                    attempt_index=len(attempts),
                    relative_residual_norm=strength,
                    target_code_id=target_code_id,
                    generated_prefix=generated_prefix,
                )
                attempts.append(attempt)
                if attempt["success"]:
                    high = strength
                else:
                    low = strength
            return high, attempts, True

        best = max(
            (
                attempt
                for attempt in attempts
                if float(attempt["relative_residual_norm"]) > 0
            ),
            key=lambda attempt: (
                bool(attempt["processed_greedy_equals_target"]),
                bool(attempt["target_is_raw_top1"]),
                float(attempt["target_logit_margin_to_strongest_other"]),
                float(attempt["target_probability"]),
            ),
        )
        return float(best["relative_residual_norm"]), attempts, False

    def _replay_with_residual_edits(
        self,
        text_ids: Any,
        speech_code_ids: Sequence[int],
        raw_logits: np.ndarray,
        *,
        layers: Sequence[int],
        positions: Sequence[int],
        directions: dict[tuple[int, int], np.ndarray],
        coordinate_diagnostics: Sequence[dict[str, Any]],
        relative_strength: float,
    ) -> tuple[int, int, Any, dict[int, Any], float]:
        """Replay the exact open-loop schedule and retain edited post-block states."""
        import mlx.core as mx

        t3 = self.model.t3
        codes = mx.array([list(speech_code_ids)], dtype=mx.int32)
        bos = mx.full((1, 1), t3.hp.start_speech_token, dtype=mx.int32)
        speech_inputs = mx.concatenate([bos, codes[:, :-1]], axis=1)
        embeds, condition_length = t3.prepare_input_embeds(
            self.model._conds.t3, text_ids, speech_inputs
        )
        transformer = t3.tfmr
        residual = embeds + transformer.wpe(mx.arange(embeds.shape[1]))
        input_residual = residual
        speech_start = condition_length + int(text_ids.shape[1])
        norm_by_coordinate = {
            (int(entry["layer"]), int(entry["speech_code_index"])): float(
                entry["baseline_residual_l2_norm"]
            )
            for entry in coordinate_diagnostics
        }
        post_block: dict[int, Any] = {}
        for layer_index, block in enumerate(transformer.h):
            residual, _ = block(residual, cache=None)
            if layer_index in layers and relative_strength > 0:
                delta_matrix = np.zeros(
                    (int(residual.shape[1]), self.hidden_size), dtype=np.float32
                )
                for speech_position in positions:
                    if speech_position >= len(speech_code_ids):
                        continue
                    delta_matrix[speech_start + speech_position] = (
                        directions[(layer_index, speech_position)]
                        * norm_by_coordinate[(layer_index, speech_position)]
                        * relative_strength
                    )
                residual = residual + mx.array(delta_matrix)[None, :, :]
            post_block[layer_index] = residual
        final = transformer.ln_f(residual)
        replay_logits = t3.speech_head(
            final[:, speech_start : speech_start + len(speech_code_ids), :]
        )
        mx.eval(replay_logits, *post_block.values())
        replay_np = np.asarray(replay_logits, dtype=np.float32)[0]
        max_error = float(np.max(np.abs(replay_np - raw_logits)))
        if max_error > 5e-4:
            raise RuntimeError(
                "residual-steered full replay diverged from incremental cache "
                f"generation (max abs logit error {max_error:.6g})"
            )
        return (
            condition_length,
            speech_start,
            input_residual,
            post_block,
            max_error,
        )

    @staticmethod
    def _first_suffix_divergence(
        parent_ids: Sequence[int], branch_ids: Sequence[int]
    ) -> int | None:
        for index, (parent_id, branch_id) in enumerate(
            zip(parent_ids, branch_ids, strict=False)
        ):
            if int(parent_id) != int(branch_id):
                return index
        if len(parent_ids) != len(branch_ids):
            return min(len(parent_ids), len(branch_ids))
        return None

    def residual_branch_synthesis(
        self,
        parent: ChatterboxCapturedRun,
        speech_code_index: int,
        target_code_id: int,
        layers: Sequence[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> tuple[ChatterboxCapturedRun, dict[str, Any]]:
        """Gradient-propose post-block edits and regenerate without forcing a code."""
        import mlx.core as mx

        (
            speech_code_index,
            target_code_id,
            resolved_layers,
            forward_span,
            max_relative_residual_norm,
        ) = self._validate_residual_branch_request(
            parent,
            speech_code_index,
            target_code_id,
            layers,
            forward_span,
            max_relative_residual_norm,
        )
        requested_positions = tuple(
            range(speech_code_index, speech_code_index + forward_span)
        )
        directions, coordinate_diagnostics = self._residual_directions(
            parent,
            target_code_id,
            resolved_layers,
            requested_positions,
        )
        (
            chosen_strength,
            calibration_attempts,
            calibration_succeeded,
        ) = self._calibrate_residual_strength(
            parent,
            speech_code_index,
            target_code_id,
            resolved_layers,
            directions,
            coordinate_diagnostics,
            max_relative_residual_norm,
        )

        config = self.generation_config
        mx.random.seed(config.seed)
        text_ids = mx.array([list(parent.text_token_ids)], dtype=mx.int32)
        bos = mx.full(
            (1, 1), self.model.t3.hp.start_speech_token, dtype=mx.int32
        )
        prefix_ids = list(parent.speech_code_ids[:speech_code_index])
        prefix = mx.array([prefix_ids], dtype=mx.int32)
        speech_inputs = mx.concatenate([bos, prefix], axis=1)
        embeds, condition_length = self.model.t3.prepare_input_embeds(
            self.model._conds.t3, text_ids, speech_inputs
        )
        anchor_edits = self._scaled_residual_edits(
            directions,
            coordinate_diagnostics,
            resolved_layers,
            speech_code_index,
            chosen_strength,
        )
        hidden, cache = self._residual_transformer_step(
            embeds, None, anchor_edits
        )
        speech_start = condition_length + len(parent.text_token_ids)
        prompt_logits = self.model.t3.speech_head(
            hidden[:, speech_start:, :]
        )
        mx.eval(prompt_logits)
        raw_rows = [
            row.copy()
            for row in np.asarray(prompt_logits, dtype=np.float32)[0]
        ]
        generated = prefix_ids.copy()
        evaluated_positions = [speech_code_index]
        anchor_token_id = self._processed_greedy_code_id(
            raw_rows[speech_code_index], generated
        )
        stopped = anchor_token_id is None
        if anchor_token_id is not None:
            if not 0 <= anchor_token_id < SPEECH_VOCAB_SIZE:
                raise RuntimeError(
                    "residual branch generated unsupported speech token "
                    f"{anchor_token_id}"
                )
            generated.append(anchor_token_id)

        while not stopped and len(generated) < config.max_speech_tokens:
            speech_position = len(generated)
            current_token = mx.array([[generated[-1]]], dtype=mx.int32)
            edits = (
                self._scaled_residual_edits(
                    directions,
                    coordinate_diagnostics,
                    resolved_layers,
                    speech_position,
                    chosen_strength,
                )
                if speech_position in requested_positions
                else {}
            )
            hidden, cache = self._residual_transformer_step(
                self.model.t3.speech_emb(current_token), cache, edits
            )
            logits = self.model.t3.speech_head(hidden[:, -1, :])
            mx.eval(logits)
            row = np.asarray(logits, dtype=np.float32)[0].copy()
            raw_rows.append(row)
            if speech_position in requested_positions:
                evaluated_positions.append(speech_position)
            token_id = self._processed_greedy_code_id(row, generated)
            if token_id is None:
                stopped = True
                break
            if not 0 <= token_id < SPEECH_VOCAB_SIZE:
                raise RuntimeError(
                    f"residual branch generated unsupported speech token {token_id}"
                )
            generated.append(token_id)

        if not generated:
            raise RuntimeError(
                "residual steering caused Chatterbox to emit stop before any "
                "ordinary speech code"
            )
        branch_logits = np.stack(raw_rows[: len(generated)])
        replay = self._replay_with_residual_edits(
            text_ids,
            generated,
            branch_logits,
            layers=resolved_layers,
            positions=requested_positions,
            directions=directions,
            coordinate_diagnostics=coordinate_diagnostics,
            relative_strength=chosen_strength,
        )
        waveform = self._decode_waveform(generated)
        branch = ChatterboxCapturedRun(
            raw_text=parent.raw_text,
            normalized_text=parent.normalized_text,
            text_token_ids=parent.text_token_ids,
            text_tokens=[dict(token) for token in parent.text_tokens],
            speech_code_ids=tuple(generated),
            raw_logits=branch_logits,
            waveform=waveform,
            sample_rate=self.sample_rate,
            condition_length=replay[0],
            speech_start=replay[1],
            input_residual=replay[2],
            post_block_residuals=replay[3],
            replay_max_abs_error=replay[4],
        )

        applied_position_set = (
            set(evaluated_positions) if chosen_strength > 0 else set()
        )
        coordinate_payload: list[dict[str, Any]] = []
        for coordinate in coordinate_diagnostics:
            applied = int(coordinate["speech_code_index"]) in applied_position_set
            baseline_norm = float(coordinate["baseline_residual_l2_norm"])
            coordinate_payload.append(
                {
                    **coordinate,
                    "applied_delta_l2_norm": (
                        baseline_norm * chosen_strength if applied else 0.0
                    ),
                    "applied_relative_residual_norm": (
                        chosen_strength if applied else 0.0
                    ),
                    "applied": applied,
                }
            )
        selected_attempt = min(
            (
                attempt
                for attempt in calibration_attempts
                if math.isclose(
                    float(attempt["relative_residual_norm"]),
                    chosen_strength,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ),
            key=lambda attempt: int(attempt["attempt_index"]),
        )
        warnings = [
            "This is a context-specific gradient-proposed post-block residual intervention, not a fitted-lens causal map and not direct logit replacement.",
            "Future position directions are computed on the parent teacher-forced path and then reused open-loop after the branch diverges; they are not dynamically re-estimated.",
            "Raw-head top-1 and the processed greedy token are separate because repetition penalty can change the emitted code.",
        ]
        if max_relative_residual_norm > 0.5:
            warnings.append(
                "The requested residual budget exceeds 0.5× the parent residual norm and can move activations substantially off the observed model manifold."
            )
        return branch, {
            "schema_version": 1,
            "kind": "t3_post_block_residual_steering_branch",
            "method": RESIDUAL_BRANCH_METHOD,
            "speech_code_index": speech_code_index,
            "target_code_id": target_code_id,
            "original_realized_code_id": int(
                parent.speech_code_ids[speech_code_index]
            ),
            "layers": list(resolved_layers),
            "forward_span": forward_span,
            "requested_positions": list(requested_positions),
            "applied_positions": sorted(applied_position_set),
            "coordinate": "post_t3_block_output_at_speech_prediction_position",
            "direction_objective": (
                "target_raw_logit_minus_parent_strongest_non_target_raw_logit"
            ),
            "direction_source": "parent_teacher_forced_path",
            "future_direction_policy": (
                "position_specific_parent_path_direction_applied_on_dynamic_branch_path"
            ),
            "suffix_policy": (
                "argmax_after_repetition_penalty_and_temperature"
            ),
            "max_relative_residual_norm": max_relative_residual_norm,
            "strength_budget_kind": (
                "per_coordinate_delta_l2_over_parent_residual_l2"
            ),
            "chosen_relative_residual_norm": chosen_strength,
            "target_became_raw_top1": bool(
                selected_attempt["target_is_raw_top1"]
            ),
            "processed_greedy_code_id_at_anchor": selected_attempt[
                "processed_greedy_code_id"
            ],
            "processed_greedy_code_id": selected_attempt[
                "processed_greedy_code_id"
            ],
            "processed_greedy_equals_target": bool(
                selected_attempt["processed_greedy_equals_target"]
            ),
            "calibration_status": (
                "succeeded" if calibration_succeeded else "budget_exhausted"
            ),
            "calibration_attempts": calibration_attempts,
            "coordinates": coordinate_payload,
            "parent_speech_code_count": len(parent.speech_code_ids),
            "branch_speech_code_count": len(branch.speech_code_ids),
            "branch_emitted_code_id_at_start": (
                int(branch.speech_code_ids[speech_code_index])
                if speech_code_index < len(branch.speech_code_ids)
                else None
            ),
            "first_suffix_divergence_index": self._first_suffix_divergence(
                parent.speech_code_ids, branch.speech_code_ids
            ),
            "limitations": warnings,
        }

    def _replay(
        self,
        text_ids: Any,
        speech_code_ids: Sequence[int],
        raw_logits: np.ndarray,
    ) -> tuple[int, int, Any, dict[int, Any], float]:
        import mlx.core as mx

        t3 = self.model.t3
        codes = mx.array([list(speech_code_ids)], dtype=mx.int32)
        bos = mx.full((1, 1), t3.hp.start_speech_token, dtype=mx.int32)
        speech_inputs = mx.concatenate([bos, codes[:, :-1]], axis=1)
        embeds, condition_length = t3.prepare_input_embeds(
            self.model._conds.t3, text_ids, speech_inputs
        )
        transformer = t3.tfmr
        residual = embeds + transformer.wpe(mx.arange(embeds.shape[1]))
        input_residual = residual
        post_block: dict[int, Any] = {}
        for layer_index, block in enumerate(transformer.h):
            residual, _ = block(residual, cache=None)
            post_block[layer_index] = residual
        final = transformer.ln_f(residual)
        speech_start = condition_length + int(text_ids.shape[1])
        replay_logits = t3.speech_head(
            final[:, speech_start : speech_start + len(speech_code_ids), :]
        )
        mx.eval(replay_logits, *post_block.values())
        replay_np = np.asarray(replay_logits, dtype=np.float32)[0]
        max_error = float(np.max(np.abs(replay_np - raw_logits)))
        if max_error > 5e-4:
            raise RuntimeError(
                "teacher-forced Chatterbox replay diverged from cached generation "
                f"(max abs logit error {max_error:.6g})"
            )
        return (
            condition_length,
            speech_start,
            input_residual,
            post_block,
            max_error,
        )

    def _decode_waveform(self, speech_code_ids: Sequence[int]) -> np.ndarray:
        import mlx.core as mx
        from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3GEN_SIL

        codes = mx.array(list(speech_code_ids), dtype=mx.int32)
        silence = mx.array([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL], dtype=mx.int32)
        decoder_input = mx.concatenate([codes, silence])[None, :]
        waveform, _ = self.model.s3gen.inference(
            speech_tokens=decoder_input,
            ref_dict=self.model._conds.gen,
            n_cfm_timesteps=2,
        )
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        mx.eval(waveform)
        return np.asarray(waveform, dtype=np.float32).reshape(-1).copy()

    def _capture_text_run(
        self, text: str, *, decode_waveform: bool
    ) -> ChatterboxCapturedRun:
        raw_text = self._validate_text(text)
        normalized, text_ids, text_tokens = self._tokenize_text(raw_text)
        speech_codes, raw_logits = self._generate_codes(text_ids)
        replay = self._replay(text_ids, speech_codes, raw_logits)
        waveform = (
            self._decode_waveform(speech_codes)
            if decode_waveform
            else np.empty(0, dtype=np.float32)
        )
        return ChatterboxCapturedRun(
            raw_text=raw_text,
            normalized_text=normalized,
            text_token_ids=tuple(int(value) for value in np.asarray(text_ids)[0]),
            text_tokens=text_tokens,
            speech_code_ids=tuple(speech_codes),
            raw_logits=raw_logits,
            waveform=waveform,
            sample_rate=self.sample_rate,
            condition_length=replay[0],
            speech_start=replay[1],
            input_residual=replay[2],
            post_block_residuals=replay[3],
            replay_max_abs_error=replay[4],
        )

    def capture_for_fitting(self, text: str) -> ChatterboxCapturedRun:
        """Generate and replay a T3 path without running S3Gen or the vocoder."""
        return self._capture_text_run(text, decode_waveform=False)

    def synthesize(self, text: str) -> ChatterboxCapturedRun:
        return self._capture_text_run(text, decode_waveform=True)

    def generation_payload(self, run: ChatterboxCapturedRun) -> dict[str, Any]:
        codes: list[dict[str, Any]] = []
        for index, (token_id, logits) in enumerate(
            zip(run.speech_code_ids, run.raw_logits, strict=True)
        ):
            raw_log_probability = _log_softmax_value(logits, token_id)
            codes.append(
                {
                    "index": index,
                    "id": token_id,
                    "start_seconds": index / SPEECH_CODE_RATE_HZ,
                    "end_seconds": (index + 1) / SPEECH_CODE_RATE_HZ,
                    "mel_start": index * 2,
                    "mel_end": index * 2 + 2,
                    "raw_probability": float(math.exp(raw_log_probability)),
                    "raw_log_probability": raw_log_probability,
                }
            )
        nominal_duration = len(codes) / SPEECH_CODE_RATE_HZ
        return {
            "schema_version": 3,
            "model": self.metadata(),
            "input": {
                "raw_text": run.raw_text,
                "normalized_text": run.normalized_text,
                "tokens": run.text_tokens,
            },
            "output": {
                "audio_data_url": waveform_wav_data_url(
                    run.waveform, sampling_rate=run.sample_rate
                ),
                "sample_rate": run.sample_rate,
                "duration_seconds": run.duration_seconds,
                "waveform": waveform_envelope(run.waveform),
                "speech_codes": codes,
                "nominal_content_duration_seconds": nominal_duration,
                "trailing_audio_seconds": max(
                    0.0, run.duration_seconds - nominal_duration
                ),
            },
            "replay": {
                "policy": "teacher_forced_full_sequence",
                "max_abs_logit_error": run.replay_max_abs_error,
            },
            "warnings": [
                "A selected waveform slice is mapped to its nominal 25 Hz T3 speech code. S3Gen and the vocoder mix context beyond an exact 40 ms boundary.",
                "Gradient values are context-specific sensitivity of the chosen raw speech-code log-probability to input-text residuals. They are not probabilities, causal contributions, or the paper's corpus-averaged J-lens.",
                "The self-attention view is ordinary causal self-attention over conditioning, text, and prior speech—not encoder-decoder cross-attention.",
                "Speech-code IDs have no published phoneme or human-readable semantic labels.",
                "The pinned checkpoint is an 8-bit MLX conversion; quantization can change gradients and requires an unquantized sensitivity check before scientific claims.",
                "The MLX port does not apply the official PyTorch implementation's PerTh output watermark, so this page does not claim generated audio is watermarked.",
            ],
        }

    def speech_head_candidate_payload(
        self,
        run: ChatterboxCapturedRun,
        *,
        top_k: int,
    ) -> dict[str, Any]:
        """Summarize the cached raw speech head without generation processing.

        The full vocabulary participates in each softmax and global rank.  Only
        ``top_k`` entries per generated position are serialized so a long local
        synthesis cannot turn into an unbounded response.
        """
        logits = np.asarray(run.raw_logits, dtype=np.float64)
        target_ids = np.asarray(run.speech_code_ids, dtype=np.int64)
        vocab_size = int(self.speech_vocab_size)
        if logits.ndim != 2:
            raise ValueError("raw speech-head logits must be a two-dimensional matrix")
        expected_shape = (target_ids.size, vocab_size)
        if logits.shape != expected_shape:
            raise ValueError(
                "raw speech-head logits have shape "
                f"{logits.shape}, expected {expected_shape}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("raw speech-head logits must all be finite")
        if target_ids.size and (
            int(target_ids.min()) < 0 or int(target_ids.max()) >= vocab_size
        ):
            raise ValueError("generated speech-code ID is outside the speech head")
        if not 1 <= int(top_k) <= vocab_size:
            raise ValueError(f"top_k must be in [1, {vocab_size}]")

        resolved_top_k = int(top_k)
        special_token_ids = self.special_speech_token_ids()
        special_by_id = {
            token_id: label for label, token_id in special_token_ids.items()
        }
        target_probabilities: list[float] = []
        target_log_probabilities: list[float] = []
        target_ranks: list[int] = []
        top_codes: list[list[dict[str, Any]]] = []

        for target_id, row in zip(target_ids.tolist(), logits, strict=True):
            probabilities = _softmax_row(row)
            target_log_probability = _log_softmax_value(row, int(target_id))
            target_probabilities.append(float(math.exp(target_log_probability)))
            target_log_probabilities.append(target_log_probability)
            target_ranks.append(int(np.count_nonzero(row > row[target_id])) + 1)

            vocabulary_ids = np.arange(vocab_size, dtype=np.int64)
            ordered_ids = np.lexsort((vocabulary_ids, -row))[
                :resolved_top_k
            ].tolist()
            position_candidates: list[dict[str, Any]] = []
            for candidate_id in ordered_ids:
                candidate: dict[str, Any] = {
                    "id": candidate_id,
                    "probability": float(probabilities[candidate_id]),
                }
                special_label = special_by_id.get(candidate_id)
                if special_label is not None:
                    candidate["special_token"] = special_label
                position_candidates.append(candidate)
            top_codes.append(position_candidates)

        return {
            "schema_version": SPEECH_HEAD_CANDIDATE_SCHEMA_VERSION,
            "top_k": resolved_top_k,
            "vocab_size": vocab_size,
            "target_ids": target_ids.tolist(),
            "target_probabilities": target_probabilities,
            "target_log_probabilities": target_log_probabilities,
            "target_ranks": target_ranks,
            "top_codes": top_codes,
            "source_coordinate": "final_t3_speech_prediction_position",
            "target_head": "t3_speech_head_after_final_norm",
            "normalization": RAW_SPEECH_HEAD_NORMALIZATION,
            "special_token_ids": special_token_ids,
            "generation_processors_excluded": [
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
            ],
            "warnings": [
                "These are raw speech-head probabilities before repetition penalty, temperature, top-k, or top-p processing.",
                "Speech-code IDs are learned acoustic symbols, not words or published phoneme labels.",
            ],
        }

    def _attention_trace(
        self, run: ChatterboxCapturedRun, layer: int, speech_code_index: int
    ) -> tuple[np.ndarray, float]:
        import mlx.core as mx

        transformer = self.model.t3.tfmr
        block = transformer.h[layer]
        source = (
            run.input_residual
            if layer == 0
            else run.post_block_residuals[layer - 1]
        )
        normalized = block.ln_1(source)
        qkv = block.attn.c_attn(normalized)
        q, k, _ = mx.split(qkv, 3, axis=-1)
        batch, length, _ = q.shape
        heads = block.attn.num_heads
        width = block.attn.head_dim
        q = q.reshape(batch, length, heads, width).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, heads, width).transpose(0, 2, 1, 3)
        query_position = run.speech_start + speech_code_index
        scores = (
            q[:, :, query_position : query_position + 1, :]
            @ k[:, :, : query_position + 1, :].transpose(0, 1, 3, 2)
        ) * block.attn.scale
        weights = mx.softmax(scores, axis=-1).mean(axis=1)[0, 0]
        text_start = run.condition_length
        text_end = text_start + len(run.text_token_ids)
        text_weights = weights[text_start:text_end]
        mx.eval(text_weights)
        text_np = np.asarray(text_weights, dtype=np.float64)
        return _normalize_nonnegative(text_np), float(text_np.sum())

    def trace(
        self,
        run: ChatterboxCapturedRun,
        speech_code_index: int,
        *,
        layers: Sequence[int] = DEFAULT_CHATTERBOX_LAYERS,
    ) -> dict[str, Any]:
        import mlx.core as mx

        if not 0 <= speech_code_index < len(run.speech_code_ids):
            raise ValueError("speech_code_index is outside the generated sequence")
        resolved_layers = sorted({int(layer) for layer in layers})
        if not resolved_layers or resolved_layers[0] < 0:
            raise ValueError("select at least one nonnegative T3 source layer")
        if resolved_layers[-1] >= self.n_layers - 1:
            raise ValueError(
                f"T3 source layers must be below final layer {self.n_layers - 1}"
            )

        target_id = run.speech_code_ids[speech_code_index]
        text_start = run.condition_length
        text_end = text_start + len(run.text_token_ids)
        query_position = run.speech_start + speech_code_index
        transformer = self.model.t3.tfmr
        gradient_l2: list[list[float]] = []
        gradient_share: list[list[float]] = []
        gradient_text_mass: list[float] = []
        attention_share: list[list[float]] = []
        attention_text_mass: list[float] = []
        target_log_probabilities: list[float] = []

        for layer in resolved_layers:
            source = run.post_block_residuals[layer]

            def selected_log_probability(
                source_residual: Any, start_layer: int = layer
            ) -> Any:
                residual = source_residual
                for suffix_layer in range(start_layer + 1, self.n_layers):
                    residual, _ = transformer.h[suffix_layer](
                        residual, cache=None
                    )
                residual = transformer.ln_f(residual)
                logits = self.model.t3.speech_head(
                    residual[:, query_position, :]
                )[0]
                return logits[target_id] - mx.logsumexp(logits)

            values, gradients = mx.vjp(
                selected_log_probability, [source], [mx.array(1.0)]
            )
            position_norms = mx.sqrt(mx.sum(gradients[0][0] ** 2, axis=-1))
            text_norms = position_norms[text_start:text_end]
            prefix_norms = position_norms[: query_position + 1]
            mx.eval(values[0], text_norms, prefix_norms)
            text_np = np.asarray(text_norms, dtype=np.float64)
            prefix_total = float(np.asarray(prefix_norms, dtype=np.float64).sum())
            gradient_l2.append(text_np.tolist())
            gradient_share.append(_normalize_nonnegative(text_np).tolist())
            gradient_text_mass.append(
                0.0 if prefix_total <= 0 else float(text_np.sum() / prefix_total)
            )
            target_log_probabilities.append(float(values[0].item()))

            attention, attention_mass = self._attention_trace(
                run, layer, speech_code_index
            )
            attention_share.append(attention.tolist())
            attention_text_mass.append(attention_mass)
            mx.clear_cache()

        code_logits = run.raw_logits[speech_code_index]
        raw_probability = float(_softmax_row(code_logits)[target_id])
        return {
            "selection": {
                "speech_code_index": speech_code_index,
                "speech_code_id": target_id,
                "start_seconds": speech_code_index / SPEECH_CODE_RATE_HZ,
                "end_seconds": (speech_code_index + 1) / SPEECH_CODE_RATE_HZ,
                "mel_start": speech_code_index * 2,
                "mel_end": speech_code_index * 2 + 2,
                "raw_probability": raw_probability,
            },
            "layers": resolved_layers,
            "text_tokens": run.text_tokens,
            "gradient_l2": gradient_l2,
            "gradient_share": gradient_share,
            "gradient_text_mass": gradient_text_mass,
            "attention_share": attention_share,
            "attention_text_mass": attention_text_mass,
            "target_log_probability": target_log_probabilities,
            "score_kind": "chosen_speech_code_raw_log_probability_gradient_l2",
            "attention_kind": "text_prefix_causal_self_attention",
            "warnings": [
                "Within-text shares are normalized only for display; retain raw gradient norms and the text-versus-prefix mass when comparing layers.",
                "Attention weights show routing association and are not causal attribution.",
                "The final T3 block is excluded because the remaining final normalization and speech head are positionwise; its text-position gradient to a different speech position is zero.",
            ],
        }
