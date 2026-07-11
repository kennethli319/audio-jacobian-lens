from __future__ import annotations

import numpy as np
import torch

from jlens.mlx_analysis import analyze_mlx_lfm_run
from jlens.mlx_lfm import LFMCapturedRun, LFMLensInputs
from jlens.projected_lens import ProjectedCrossJacobianLens


class _Tokenizer:
    all_special_ids = []

    @staticmethod
    def decode(token_ids, **kwargs):
        del kwargs
        return ["a", "b", "c", "d"][int(token_ids[0])]


class _Model:
    tokenizer = _Tokenizer()
    vocab_size = 4
    input_sample_rate = 16_000
    fingerprint = "tiny-mlx"

    @staticmethod
    def validate_projected_lens(lens):
        assert lens.metadata["model_fingerprint"] == "tiny-mlx"

    @staticmethod
    def capture(inputs):
        del inputs
        residual = torch.tensor(
            [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]
        )
        return LFMCapturedRun(
            language_residuals={0: residual, 1: residual + 0.5},
            actual_logits=torch.tensor(
                [[4.0, 1.0, 0.0, -1.0], [0.0, 5.0, 1.0, -2.0]]
            ),
            target_token_ids=torch.tensor([0, 1]),
        )

    @staticmethod
    def unembed(residual):
        return residual

    @staticmethod
    def lens_metadata():
        return {
            "backend": "mlx",
            "model_family": "lfm2_audio",
            "model_id": "tiny-mlx-speech",
        }


def test_mlx_payload_reuses_decoder_schema_and_exposes_generated_audio():
    target_factors = torch.eye(4) * 2
    lens = ProjectedCrossJacobianLens(
        target_factors,
        {0: target_factors, 1: target_factors},
        n_examples=2,
        source_dim=4,
        target_dim=4,
        source_stream="language",
        target_stream="language",
        projection_method="subsampled_hadamard_output_probe_vjp",
        metadata={
            "model_fingerprint": "tiny-mlx",
            "target_layer": 2,
            "projection_seed": 9,
            "generation": {"max_new_tokens": 18},
        },
    )
    inputs = LFMLensInputs(
        text_tokens=np.zeros((1, 3), dtype=np.int64),
        audio_features=np.zeros((1, 2, 4), dtype=np.float32),
        audio_codes=None,
        modalities=np.ones((1, 3), dtype=np.int64),
        prediction_positions=(0, 1),
        target_token_ids=(0, 1),
        generated_text_token_ids=(0, 1),
        generated_text="ab",
        generated_audio=np.zeros(2_400, dtype=np.float32),
        generated_audio_sample_rate=24_000,
        input_audio_positions=(),
        duration_seconds=0.2,
        metadata={
            "generation": {"max_new_tokens": 72},
            "generation_diagnostics": {
                "termination_reason": "audio_eos",
                "budget_exhausted": False,
                "max_new_tokens": 72,
                "generated_steps": 64,
                "text_tokens": 6,
                "audio_frames": 57,
                "audio_eos_seen": True,
            },
        },
    )

    payload = analyze_mlx_lfm_run(
        _Model(), lens, inputs, np.zeros(3_200, dtype=np.float32), top_k=2
    )

    assert payload["metadata"]["streams"] == ["decoder"]
    assert payload["metadata"]["projection"] == {
        "method": "subsampled_hadamard_output_probe_vjp",
        "rank": 4,
        "target_dim": 4,
        "seed": 9,
        "dense_at_full_rank": True,
    }
    assert payload["metadata"]["capabilities"]["audio_codebook_jlens"] is False
    assert payload["metadata"]["artifact_generation"] == {"max_new_tokens": 18}
    assert payload["metadata"]["serving_generation"] == {"max_new_tokens": 72}
    assert payload["metadata"]["generation_diagnostics"] == {
        "termination_reason": "audio_eos",
        "budget_exhausted": False,
        "max_new_tokens": 72,
        "generated_steps": 64,
        "text_tokens": 6,
        "audio_frames": 57,
        "audio_eos_seen": True,
    }
    assert payload["decoder"]["layers"] == [0, 1]
    assert len(payload["decoder"]["cells"]) == 2
    assert payload["transcription"]["text"] == "ab"
    assert payload["transcription"]["tokens"][0]["probability"] > 0.9
    assert payload["audio"]["model_output_wav"].startswith("data:audio/wav;base64,")
    assert payload["audio"]["model_output_duration_seconds"] == 0.1


def test_rademacher_projection_is_not_claimed_dense_at_full_rank():
    factors = torch.eye(4) * 2
    lens = ProjectedCrossJacobianLens(
        factors,
        {0: factors},
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream="language",
        target_stream="language",
        projection_method="rademacher_output_probe_vjp",
        metadata={
            "model_fingerprint": "tiny-mlx",
            "target_layer": 1,
            "projection_seed": 4,
        },
    )
    inputs = LFMLensInputs(
        text_tokens=np.zeros((1, 3), dtype=np.int64),
        audio_features=np.zeros((1, 2, 4), dtype=np.float32),
        audio_codes=None,
        modalities=np.ones((1, 3), dtype=np.int64),
        prediction_positions=(0, 1),
        target_token_ids=(0, 1),
        generated_text_token_ids=(0, 1),
        generated_text="ab",
        generated_audio=None,
        generated_audio_sample_rate=24_000,
        input_audio_positions=(),
        duration_seconds=0.2,
        metadata={},
    )

    payload = analyze_mlx_lfm_run(
        _Model(), lens, inputs, np.zeros(3_200, dtype=np.float32), top_k=2
    )

    assert payload["metadata"]["projection"]["dense_at_full_rank"] is False
