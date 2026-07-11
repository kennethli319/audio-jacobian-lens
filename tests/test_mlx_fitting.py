from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.mlx_fitting import (
    PROJECTION_METHOD,
    hadamard_target_factors,
    projected_vjps_for_lfm_example,
)
from jlens.mlx_lfm import (
    LFM_TEXT_CONTROL_TOKEN_IDS,
    LFMForwardTrace,
    LFMGenerationConfig,
    LFMLensInputs,
    _interleaved_generation_diagnostics,
    _is_ordinary_lfm_text_token,
)
from jlens.projected_lens import ProjectedCrossJacobianLens


def fake_inputs(*, width: int = 4) -> LFMLensInputs:
    return LFMLensInputs(
        text_tokens=np.zeros((1, 3), dtype=np.int64),
        audio_features=np.zeros((1, 2, width), dtype=np.float32),
        audio_codes=None,
        modalities=np.ones((1, 3), dtype=np.int64),
        prediction_positions=(0, 1),
        target_token_ids=(1, 2),
        generated_text_token_ids=(1, 2),
        generated_text="tiny",
        generated_audio=None,
        generated_audio_sample_rate=24_000,
        input_audio_positions=(),
        duration_seconds=0.2,
        metadata={},
    )


def test_hadamard_target_factors_are_seeded_orthogonal_and_complete():
    full = hadamard_target_factors(8, 8, seed=3)
    torch.testing.assert_close(full.T @ full / 8, torch.eye(8))
    assert torch.equal(full, hadamard_target_factors(8, 8, seed=3))
    assert not torch.equal(full, hadamard_target_factors(8, 8, seed=4))
    with pytest.raises(ValueError, match="power of two"):
        hadamard_target_factors(7, 4, seed=0)
    with pytest.raises(ValueError, match=r"\[1, target_dim\]"):
        hadamard_target_factors(8, 9, seed=0)


def test_lfm_input_and_generation_policy_validation():
    assert LFMGenerationConfig().top_k == 1
    with pytest.raises(ValueError, match="system_prompt"):
        LFMGenerationConfig(system_prompt=" ")
    with pytest.raises(ValueError, match="ordinary text targets"):
        LFMLensInputs(
            **{
                **fake_inputs().__dict__,
                "prediction_positions": (),
                "target_token_ids": (),
            }
        )


def test_lfm_text_control_ids_are_excluded_even_if_tokenizer_omits_them():
    tokenizer_special_ids = {1, 2}
    generated_text_ids = [42, 7, 128, 130, 2, 2048]

    ordinary = [
        token_id
        for token_id in generated_text_ids
        if _is_ordinary_lfm_text_token(token_id, tokenizer_special_ids)
    ]

    assert ordinary == [42, 2048]
    assert LFM_TEXT_CONTROL_TOKEN_IDS == {7, 128, 130}


@pytest.mark.parametrize(
    (
        "generated_steps",
        "audio_eos_seen",
        "final_step_audio_eos",
        "text_complete",
        "reason",
    ),
    [
        (72, False, False, False, "budget_exhausted"),
        (64, True, True, True, "audio_eos"),
        (8, False, False, False, "model_stop"),
        (72, True, False, True, "budget_exhausted"),
        (72, True, True, False, "budget_exhausted"),
    ],
)
def test_interleaved_generation_diagnostics_report_exact_stop_reason(
    generated_steps, audio_eos_seen, final_step_audio_eos, text_complete, reason
):
    diagnostics = _interleaved_generation_diagnostics(
        max_new_tokens=72,
        generated_steps=generated_steps,
        text_tokens=6,
        audio_frames=generated_steps - 6 - int(audio_eos_seen),
        audio_eos_seen=audio_eos_seen,
        final_step_audio_eos=final_step_audio_eos,
        text_complete=text_complete,
    )

    assert diagnostics == {
        "termination_reason": reason,
        "budget_exhausted": reason == "budget_exhausted",
        "max_new_tokens": 72,
        "generated_steps": generated_steps,
        "text_tokens": 6,
        "audio_frames": generated_steps - 6 - int(audio_eos_seen),
        "audio_eos_seen": audio_eos_seen,
    }


def test_tiny_mlx_output_probe_vjps_reconstruct_explicit_jacobian():
    mx = pytest.importorskip("mlx.core")

    class LinearLayer:
        is_attention_layer = True

        def __init__(self, weight):
            self.weight = mx.array(weight)

        def __call__(self, hidden, mask=None, cache=None):
            del mask, cache
            return hidden @ self.weight.T

    weights = [
        np.array(
            [[1.0, 0.2, 0.0, 0.1], [0.0, 1.1, -0.1, 0.0], [0.3, 0.0, 0.9, 0.0], [0.0, -0.2, 0.1, 1.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.9, 0.0, 0.2, 0.0], [0.1, 1.0, 0.0, -0.1], [0.0, 0.2, 1.1, 0.0], [-0.1, 0.0, 0.0, 0.8]],
            dtype=np.float32,
        ),
        np.array(
            [[1.0, -0.1, 0.0, 0.2], [0.0, 0.8, 0.2, 0.0], [0.1, 0.0, 1.0, -0.2], [0.0, 0.3, 0.0, 0.9]],
            dtype=np.float32,
        ),
    ]
    layers = [LinearLayer(weight) for weight in weights]

    class TinyMLXAdapter:
        n_language_layers = 3
        language_dim = 4
        model = SimpleNamespace(lfm=SimpleNamespace(layers=layers))

        @staticmethod
        def forward_trace(inputs):
            del inputs
            hidden = mx.array(
                [[[0.2, -0.4, 0.7, 0.1], [0.5, 0.3, -0.2, 0.8], [0.0, 0.1, 0.2, 0.3]]]
            )
            activations = {}
            for index, layer in enumerate(layers):
                hidden = layer(hidden)
                activations[index] = hidden
            mx.eval(*activations.values())
            return LFMForwardTrace(hidden, None, None, activations)

    probes = hadamard_target_factors(4, 4, seed=9)
    result = projected_vjps_for_lfm_example(
        TinyMLXAdapter(),
        fake_inputs(),
        probes,
        source_layers=[0],
        target_layer=2,
    )
    lens = ProjectedCrossJacobianLens(
        probes,
        result.source_factors,
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream="language",
        target_stream="language",
        projection_method=PROJECTION_METHOD,
    )
    explicit = torch.from_numpy(weights[2] @ weights[1])
    residual = torch.tensor([[0.4, -0.2, 0.1, 0.8]])
    torch.testing.assert_close(
        lens.transport(residual, 0), residual @ explicit.T, rtol=1e-5, atol=1e-5
    )
