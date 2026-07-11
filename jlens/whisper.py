# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face Whisper adapter for audio Jacobian-lens experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from jlens.hooks import ActivationRecorder


def _expand_single_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] != 1:
        raise ValueError(
            "Jacobian examples must have batch size 1 before dimension batching; "
            f"got {tensor.shape[0]}"
        )
    return tensor.expand(batch_size, *tensor.shape[1:])


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash vocabulary identity plus the active Whisper task prefix."""
    digest = hashlib.sha256()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        for token, token_id in sorted(
            vocabulary.items(), key=lambda item: (item[1], item[0])
        ):
            digest.update(int(token_id).to_bytes(8, "big", signed=True))
            encoded = str(token).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
    else:
        digest.update(type(tokenizer).__qualname__.encode())
        digest.update(
            json.dumps(
                list(getattr(tokenizer, "all_special_ids", [])),
                separators=(",", ":"),
            ).encode()
        )
    digest.update(
        json.dumps(
            list(getattr(tokenizer, "prefix_tokens", [])),
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()[:16]


def _local_weights_fingerprint(model: torch.nn.Module) -> str:
    """Hash local weights when no immutable Hub revision identifies them."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        values = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(values.tobytes())
    return f"sha256:{digest.hexdigest()[:20]}"


def _stable_config_fingerprint(payload: dict[str, Any]) -> str:
    """Hash forward-relevant config while excluding administrative defaults."""
    ignored = {
        "_name_or_path",
        "alignment_heads",
        "architectures",
        "begin_suppress_tokens",
        "forced_decoder_ids",
        "id2label",
        "is_multilingual",
        "label2id",
        "language",
        "output_attentions",
        "output_hidden_states",
        "problem_type",
        "return_dict",
        "suppress_tokens",
        "task",
        "torch_dtype",
        "transformers_version",
        "use_cache",
    }
    semantic = {key: value for key, value in payload.items() if key not in ignored}
    rendered = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class WhisperLensInputs:
    """One audio/decoder example prepared for fitting or lens application.

    ``decoder_target_ids[:, i]`` is the token predicted by the decoder state at
    ``decoder_input_ids[:, i]``. Masks are over post-convolution encoder
    positions and decoder prediction positions respectively.
    """

    input_features: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_target_ids: torch.Tensor
    encoder_position_mask: torch.Tensor
    decoder_position_mask: torch.Tensor
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_features.ndim != 3:
            raise ValueError("input_features must have shape [batch, mel, frames]")
        if self.decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [batch, positions]")
        if self.decoder_target_ids.shape != self.decoder_input_ids.shape:
            raise ValueError("decoder_target_ids must align with decoder_input_ids")
        if self.encoder_position_mask.ndim != 2:
            raise ValueError("encoder_position_mask must have shape [batch, positions]")
        if self.decoder_position_mask.shape != self.decoder_input_ids.shape:
            raise ValueError("decoder_position_mask must align with decoder_input_ids")
        batch_sizes = {
            self.input_features.shape[0],
            self.decoder_input_ids.shape[0],
            self.encoder_position_mask.shape[0],
            self.decoder_position_mask.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("all WhisperLensInputs tensors must share a batch size")
        if not bool(self.encoder_position_mask.any()):
            raise ValueError("encoder_position_mask selects no positions")
        if not bool(self.decoder_position_mask.any()):
            raise ValueError("decoder_position_mask selects no positions")

    @property
    def batch_size(self) -> int:
        return int(self.input_features.shape[0])

    def to(self, device: torch.device | str) -> WhisperLensInputs:
        """Move tensor fields to ``device`` without mutating the example."""
        return replace(
            self,
            input_features=self.input_features.to(device),
            decoder_input_ids=self.decoder_input_ids.to(device),
            decoder_target_ids=self.decoder_target_ids.to(device),
            encoder_position_mask=self.encoder_position_mask.to(device),
            decoder_position_mask=self.decoder_position_mask.to(device),
        )

    def expand_batch(self, batch_size: int) -> WhisperLensInputs:
        """View one example as ``batch_size`` identical examples for VJPs."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return replace(
            self,
            input_features=_expand_single_batch(self.input_features, batch_size),
            decoder_input_ids=_expand_single_batch(
                self.decoder_input_ids, batch_size
            ),
            decoder_target_ids=_expand_single_batch(
                self.decoder_target_ids, batch_size
            ),
            encoder_position_mask=_expand_single_batch(
                self.encoder_position_mask, batch_size
            ),
            decoder_position_mask=_expand_single_batch(
                self.decoder_position_mask, batch_size
            ),
        )


def downsample_feature_mask(
    feature_mask: torch.Tensor, *, encoder_positions: int
) -> torch.Tensor:
    """Convert Whisper's 10 ms feature mask to its 20 ms encoder positions.

    Whisper's first convolution has stride one and its second has stride two.
    Valid feature frames are left-aligned by the feature extractor, so a length
    reduction is both clearer and less error-prone than pooling boolean values.
    """
    if feature_mask.ndim != 2:
        raise ValueError("feature_mask must have shape [batch, feature_frames]")
    valid_feature_frames = feature_mask.to(torch.long).sum(dim=1)
    valid_encoder_positions = torch.div(
        valid_feature_frames + 1, 2, rounding_mode="floor"
    ).clamp(max=encoder_positions)
    positions = torch.arange(encoder_positions, device=feature_mask.device)
    return positions[None, :] < valid_encoder_positions[:, None]


def prediction_mask_from_targets(
    target_ids: torch.Tensor,
    *,
    special_token_ids: set[int],
    include_eos: bool = False,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Select decoder positions whose *next target* is an ordinary text token."""
    mask = torch.ones_like(target_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        if include_eos and eos_token_id is not None and token_id == eos_token_id:
            continue
        mask &= target_ids != token_id
    return mask


class HFWhisperLensModel:
    """Lens-facing wrapper over ``WhisperForConditionalGeneration``.

    The wrapper retains the full encoder-decoder computation. In particular it
    does not call the decoder in isolation without ``encoder_hidden_states``, a
    tempting shortcut that would silently remove the audio conditioning.
    """

    def __init__(
        self,
        hf_model: torch.nn.Module,
        processor: Any,
        *,
        model_id: str | None = None,
    ) -> None:
        if not all(hasattr(hf_model, name) for name in ("model", "proj_out", "config")):
            raise TypeError(
                "hf_model must be a WhisperForConditionalGeneration-like model"
            )
        if not all(hasattr(hf_model.model, name) for name in ("encoder", "decoder")):
            raise TypeError("hf_model.model must expose encoder and decoder")

        self._hf_model = hf_model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_id = model_id or getattr(hf_model.config, "_name_or_path", None)
        if not self.model_id:
            self.model_id = type(hf_model).__name__

        hf_model.eval()
        for parameter in hf_model.parameters():
            parameter.requires_grad_(False)

        self.encoder = hf_model.model.encoder
        self.decoder = hf_model.model.decoder
        self.encoder_layers = self.encoder.layers
        self.decoder_layers = self.decoder.layers
        self.decoder_norm = self.decoder.layer_norm
        self.output_head = hf_model.proj_out

        self.model_revision = getattr(hf_model.config, "_commit_hash", None)
        self.weights_fingerprint = (
            f"hf-commit:{self.model_revision}"
            if self.model_revision
            else _local_weights_fingerprint(hf_model)
        )
        self.decoder_prefix_ids = list(
            getattr(self.tokenizer, "prefix_tokens", [])
        )
        self.tokenizer_fingerprint = _tokenizer_fingerprint(self.tokenizer)
        feature_extractor = getattr(processor, "feature_extractor", None)
        feature_payload = (
            {}
            if feature_extractor is None
            else getattr(feature_extractor, "to_dict", lambda: {})()
        )
        self.feature_extractor_fingerprint = _stable_config_fingerprint(
            feature_payload
        )
        generation_config = getattr(hf_model, "generation_config", None)
        generation_payload = (
            {} if generation_config is None else generation_config.to_dict()
        )
        self.generation_policy = {
            key: generation_payload.get(key)
            for key in (
                "alignment_heads",
                "begin_suppress_tokens",
                "forced_decoder_ids",
                "is_multilingual",
                "language",
                "suppress_tokens",
                "task",
            )
            if generation_payload.get(key) is not None
        }
        generation_json = json.dumps(
            self.generation_policy,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        self.generation_config_fingerprint = hashlib.sha256(
            generation_json.encode()
        ).hexdigest()[:16]

        config = hf_model.config
        self.model_config_fingerprint = _stable_config_fingerprint(
            config.to_dict()
        )
        self.n_encoder_layers = int(config.encoder_layers)
        self.n_decoder_layers = int(config.decoder_layers)
        self.encoder_dim = int(config.d_model)
        self.decoder_dim = int(config.d_model)
        self.vocab_size = int(config.vocab_size)
        self.max_source_positions = int(config.max_source_positions)
        self.max_target_positions = int(config.max_target_positions)
        if len(self.encoder_layers) != self.n_encoder_layers:
            raise ValueError("encoder layer count disagrees with model config")
        if len(self.decoder_layers) != self.n_decoder_layers:
            raise ValueError("decoder layer count disagrees with model config")

    def __repr__(self) -> str:
        return (
            f"HFWhisperLensModel({self.model_id!r}, "
            f"encoder_layers={self.n_encoder_layers}, "
            f"decoder_layers={self.n_decoder_layers}, d_model={self.decoder_dim})"
        )

    @property
    def input_device(self) -> torch.device:
        return self.encoder.conv1.weight.device

    @property
    def unembedding_weight(self) -> torch.Tensor:
        return self.output_head.weight

    @property
    def fingerprint(self) -> str:
        """Stable configuration fingerprint used for lens compatibility checks."""
        relevant = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "weights_fingerprint": self.weights_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "feature_extractor_fingerprint": self.feature_extractor_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "decoder_prefix_ids": self.decoder_prefix_ids,
        }
        payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def lens_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_fingerprint": self.fingerprint,
            "encoder_layers": self.n_encoder_layers,
            "decoder_layers": self.n_decoder_layers,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "vocab_size": self.vocab_size,
            "model_revision": self.model_revision,
            "weights_fingerprint": self.weights_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "feature_extractor_fingerprint": self.feature_extractor_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "decoder_prefix_ids": self.decoder_prefix_ids,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "generation_policy": self.generation_policy,
        }

    def prepare_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int,
        sequence_ids: torch.Tensor,
        include_eos_target: bool = False,
        duration_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WhisperLensInputs:
        """Create one fitting/viewing example from audio and a full token sequence.

        ``sequence_ids`` must contain the decoder start/prefix, transcript, and
        usually EOS. Everything except its last token becomes decoder input;
        everything except its first token is the aligned prediction target.
        """
        if sequence_ids.ndim == 1:
            sequence_ids = sequence_ids.unsqueeze(0)
        if sequence_ids.ndim != 2 or sequence_ids.shape[0] != 1:
            raise ValueError("sequence_ids must have shape [tokens] or [1, tokens]")
        if sequence_ids.shape[1] < 2:
            raise ValueError("sequence_ids needs at least two tokens")
        if self.decoder_prefix_ids:
            prefix_length = len(self.decoder_prefix_ids)
            actual_prefix = sequence_ids[0, :prefix_length].tolist()
            if actual_prefix != self.decoder_prefix_ids:
                raise ValueError(
                    "decoder sequence prefix does not match the adapter's active "
                    f"language/task policy: expected {self.decoder_prefix_ids}, "
                    f"got {actual_prefix}"
                )

        features = self.processor.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features
        expected_feature_frames = self.max_source_positions * 2
        if input_features.shape[-1] != expected_feature_frames:
            raise ValueError(
                f"Whisper input must be padded to {expected_feature_frames} feature "
                f"frames, got {input_features.shape[-1]}"
            )

        feature_mask = getattr(features, "attention_mask", None)
        if feature_mask is None:
            if duration_seconds is None:
                raise ValueError(
                    "feature extractor returned no attention mask; pass duration_seconds"
                )
            valid_features = min(
                expected_feature_frames,
                max(1, round(duration_seconds / 0.01)),
            )
            feature_mask = torch.zeros(1, expected_feature_frames, dtype=torch.long)
            feature_mask[:, :valid_features] = 1
        encoder_mask = downsample_feature_mask(
            feature_mask, encoder_positions=self.max_source_positions
        )

        decoder_input_ids = sequence_ids[:, :-1].to(torch.long)
        decoder_target_ids = sequence_ids[:, 1:].to(torch.long)
        special_ids = set(int(token_id) for token_id in self.tokenizer.all_special_ids)
        decoder_mask = prediction_mask_from_targets(
            decoder_target_ids,
            special_token_ids=special_ids,
            include_eos=include_eos_target,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        if not bool(decoder_mask.any()):
            raise ValueError("token sequence has no ordinary text prediction targets")

        return WhisperLensInputs(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_target_ids=decoder_target_ids,
            encoder_position_mask=encoder_mask,
            decoder_position_mask=decoder_mask,
            duration_seconds=duration_seconds,
            metadata=dict(metadata or {}),
        )

    def forward(self, inputs: WhisperLensInputs) -> Any:
        """Run the full bare Whisper model with hooks visible and no KV cache."""
        return self._hf_model.model(
            input_features=inputs.input_features,
            decoder_input_ids=inputs.decoder_input_ids,
            use_cache=False,
        )

    def encode_audio(self, inputs: WhisperLensInputs) -> torch.Tensor:
        """Return the final normalized encoder states for decoder-only fitting."""
        return self.encoder(inputs.input_features).last_hidden_state

    def forward_decoder(
        self,
        inputs: WhisperLensInputs,
        encoder_hidden_states: torch.Tensor,
    ) -> Any:
        """Run the decoder against precomputed audio states without a KV cache."""
        return self.decoder(
            input_ids=inputs.decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        )

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply Whisper's final decoder LayerNorm and tied output head."""
        device = self.output_head.weight.device
        dtype = self.output_head.weight.dtype
        normalized = self.decoder_norm(residual.to(device=device, dtype=dtype))
        return self.output_head(normalized)

    def generate(self, input_features: torch.Tensor, **kwargs: Any) -> Any:
        """Delegate to Whisper's custom generation method."""
        return self._hf_model.generate(input_features, **kwargs)

    @torch.no_grad()
    def capture(
        self,
        inputs: WhisperLensInputs,
        *,
        encoder_layers: list[int] | None = None,
        decoder_layers: list[int] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """Capture held-out residuals plus actual raw decoder logits."""
        encoder_layers = list(range(self.n_encoder_layers)) if encoder_layers is None else encoder_layers
        decoder_layers = list(range(self.n_decoder_layers)) if decoder_layers is None else decoder_layers
        inputs = inputs.to(self.input_device)
        with (
            ActivationRecorder(self.encoder_layers, at=encoder_layers) as encoder_recorder,
            ActivationRecorder(self.decoder_layers, at=decoder_layers) as decoder_recorder,
        ):
            outputs = self.forward(inputs)
        encoder = {
            layer: encoder_recorder.activations[layer].detach()
            for layer in encoder_layers
        }
        decoder = {
            layer: decoder_recorder.activations[layer].detach()
            for layer in decoder_layers
        }
        logits = self.output_head(outputs.last_hidden_state).detach()
        return encoder, decoder, logits
