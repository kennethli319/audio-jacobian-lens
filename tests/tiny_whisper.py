"""Deterministic tiny encoder-decoder for Whisper estimator tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jlens.whisper import WhisperLensInputs


class _ResidualEncoderBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.08)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        global_context = hidden.mean(dim=1, keepdim=True)
        return hidden + self.linear(hidden) + 0.03 * global_context


class _ResidualDecoderBlock(nn.Module):
    def __init__(self, decoder_width: int, encoder_width: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(decoder_width, decoder_width, bias=False)
        self.cross_linear = nn.Linear(encoder_width, decoder_width, bias=False)
        with torch.no_grad():
            self.self_linear.weight.mul_(0.07)
            self.cross_linear.weight.mul_(0.06)

    def forward(
        self, hidden: torch.Tensor, encoder_hidden: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        )[None, :, None]
        causal_context = hidden.cumsum(dim=1) / positions
        audio_context = encoder_hidden.mean(dim=1, keepdim=True)
        return (
            hidden
            + self.self_linear(hidden)
            + 0.04 * causal_context
            + self.cross_linear(audio_context)
        )


class TinyWhisperLensModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        encoder_dim: int = 3,
        decoder_dim: int = 5,
        source_positions: int = 6,
        vocab_size: int = 13,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.model_id = "tiny-whisper-test"
        self.fingerprint = "tiny-whisper-test-v1"
        self.n_encoder_layers = encoder_layers
        self.n_decoder_layers = decoder_layers
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_source_positions = source_positions
        self.max_target_positions = 16
        self.vocab_size = vocab_size
        self.encoder_layers = nn.ModuleList(
            [_ResidualEncoderBlock(encoder_dim) for _ in range(encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [
                _ResidualDecoderBlock(decoder_dim, encoder_dim)
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_embed = nn.Embedding(vocab_size, decoder_dim)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.output_head = nn.Linear(decoder_dim, vocab_size, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @property
    def input_device(self) -> torch.device:
        return self.decoder_embed.weight.device

    def forward(self, inputs: WhisperLensInputs):
        encoder_hidden = inputs.input_features.permute(0, 2, 1)
        for block in self.encoder_layers:
            encoder_hidden = block(encoder_hidden)
        return self.forward_decoder(inputs, encoder_hidden)

    def encode_audio(self, inputs: WhisperLensInputs) -> torch.Tensor:
        encoder_hidden = inputs.input_features.permute(0, 2, 1)
        for block in self.encoder_layers:
            encoder_hidden = block(encoder_hidden)
        return encoder_hidden

    def forward_decoder(
        self,
        inputs: WhisperLensInputs,
        encoder_hidden: torch.Tensor,
    ):
        decoder_hidden = self.decoder_embed(inputs.decoder_input_ids)
        for block in self.decoder_layers:
            decoder_hidden = block(decoder_hidden, encoder_hidden)
        return SimpleNamespace(last_hidden_state=self.decoder_norm(decoder_hidden))

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.decoder_norm(residual))

    def lens_metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_fingerprint": self.fingerprint,
            "encoder_layers": self.n_encoder_layers,
            "decoder_layers": self.n_decoder_layers,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "vocab_size": self.vocab_size,
        }


def tiny_inputs(
    model: TinyWhisperLensModel,
    *,
    decoder_positions: int = 5,
) -> WhisperLensInputs:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(
        1,
        model.encoder_dim,
        model.max_source_positions,
        generator=generator,
    )
    decoder_ids = torch.arange(decoder_positions).remainder(model.vocab_size)[None]
    targets = (decoder_ids + 1).remainder(model.vocab_size)
    encoder_mask = torch.tensor(
        [[True] * (model.max_source_positions - 1) + [False]]
    )
    decoder_mask = torch.tensor([[False] + [True] * (decoder_positions - 1)])
    return WhisperLensInputs(
        input_features=features,
        decoder_input_ids=decoder_ids,
        decoder_target_ids=targets,
        encoder_position_mask=encoder_mask,
        decoder_position_mask=decoder_mask,
        duration_seconds=0.1,
    )
