from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_fitting import jacobians_for_whisper_example


def random_hf_whisper() -> HFWhisperLensModel:
    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    hf_model = WhisperForConditionalGeneration(config)
    tokenizer = SimpleNamespace(all_special_ids=[0, 1, 2], eos_token_id=2)
    processor = SimpleNamespace(tokenizer=tokenizer)
    return HFWhisperLensModel(
        hf_model, processor, model_id="random-hf-whisper-test"
    )


def random_inputs() -> WhisperLensInputs:
    return WhisperLensInputs(
        input_features=torch.randn(1, 4, 8),
        decoder_input_ids=torch.tensor([[1, 3, 4, 5]]),
        decoder_target_ids=torch.tensor([[3, 4, 5, 2]]),
        encoder_position_mask=torch.tensor([[True, True, True, False]]),
        decoder_position_mask=torch.tensor([[True, True, True, False]]),
    )


def test_hf_adapter_captures_both_stacks_and_actual_logits():
    model = random_hf_whisper()
    encoder, decoder, logits = model.capture(random_inputs())
    assert {layer: value.shape for layer, value in encoder.items()} == {
        0: (1, 4, 8),
        1: (1, 4, 8),
    }
    assert {layer: value.shape for layer, value in decoder.items()} == {
        0: (1, 4, 8),
        1: (1, 4, 8),
    }
    assert logits.shape == (1, 4, 32)


def test_hf_adapter_runs_cross_stream_estimator():
    model = random_hf_whisper()
    result = jacobians_for_whisper_example(
        model,
        random_inputs(),
        encoder_source_layers=[0, 1],
        decoder_source_layers=[0],
        target_decoder_layer=1,
        dim_batch=4,
    )
    assert all(matrix.shape == (8, 8) for matrix in result.encoder.values())
    assert result.decoder[0].shape == (8, 8)
    assert all(
        torch.isfinite(matrix).all()
        for matrix in [*result.encoder.values(), *result.decoder.values()]
    )


def test_local_weight_fingerprint_distinguishes_checkpoints():
    first = random_hf_whisper()
    second = random_hf_whisper()
    assert first.weights_fingerprint.startswith("sha256:")
    assert first.fingerprint != second.fingerprint
