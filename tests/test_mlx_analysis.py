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


class _RankTokenizer:
    all_special_ids = [3]

    @staticmethod
    def decode(token_ids, **kwargs):
        del kwargs
        return ["the", "x", "y", "<|control|>", "z"][int(token_ids[0])]


class _RankModel:
    tokenizer = _RankTokenizer()
    vocab_size = 5
    input_sample_rate = 16_000
    fingerprint = "rank-mlx"

    @staticmethod
    def validate_projected_lens(lens):
        assert lens.metadata["model_fingerprint"] == "rank-mlx"

    @staticmethod
    def capture(inputs):
        del inputs
        early = torch.tensor(
            [
                [1.0, 3.0, 1.0, 2.0, 0.0],
                [2.0, 2.0, 1.0, 0.0, 0.0],
            ]
        )
        late = torch.tensor(
            [
                [3.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 3.0],
            ]
        )
        return LFMCapturedRun(
            language_residuals={0: early, 1: late},
            actual_logits=early,
            target_token_ids=torch.tensor([0, 3]),
        )

    @staticmethod
    def unembed(residual):
        return residual

    @staticmethod
    def lens_metadata():
        return {
            "backend": "mlx",
            "model_family": "lfm2_audio",
            "model_id": "rank-mlx-speech",
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


def test_mlx_payload_reports_exact_realized_ranks_from_full_logits():
    # The factorized map is identity: (h @ I) @ (5I) / rank 5 == h.
    target_factors = torch.eye(5) * 5
    source_factors = torch.eye(5)
    lens = ProjectedCrossJacobianLens(
        target_factors,
        {0: source_factors, 1: source_factors},
        n_examples=1,
        source_dim=5,
        target_dim=5,
        source_stream="language",
        target_stream="language",
        projection_method="subsampled_hadamard_output_probe_vjp",
        metadata={
            "model_fingerprint": "rank-mlx",
            "target_layer": 2,
        },
    )
    inputs = LFMLensInputs(
        text_tokens=np.zeros((1, 2), dtype=np.int64),
        audio_features=np.zeros((1, 2, 5), dtype=np.float32),
        audio_codes=None,
        modalities=np.ones((1, 2), dtype=np.int64),
        prediction_positions=(0, 1),
        target_token_ids=(0, 3),
        generated_text_token_ids=(0, 3),
        generated_text="the<|control|>",
        generated_audio=None,
        generated_audio_sample_rate=24_000,
        input_audio_positions=(),
        duration_seconds=0.1,
        metadata={},
    )

    payload = analyze_mlx_lfm_run(
        _RankModel(), lens, inputs, np.zeros(1_600, dtype=np.float32), top_k=1
    )

    head_the, head_control = payload["transcription"]["tokens"]
    assert head_the["top_tokens"][0]["id"] == 1
    assert head_the["rank"] == 3
    assert head_the["rank_denominator"] == 5
    assert head_the["rank_space"] == "full_model_vocabulary"
    assert head_the["full_vocabulary_rank"] == 3
    assert head_the["full_vocabulary_denominator"] == 5
    assert head_the["rank_tie_policy"] == "1_plus_count_strictly_greater"
    assert head_the["score_kind"] == "raw_teacher_forced_probability"
    assert head_control["rank"] == 4

    early_the = payload["decoder"]["cells"][0][0]["realized_token"]
    assert early_the == {
        "id": 0,
        "text": "the",
        "score": 1.0,
        "rank": 2,
        "rank_denominator": 4,
        "rank_space": "lexical_display_vocabulary",
        "display_vocabulary_rank": 2,
        "display_vocabulary_denominator": 4,
        "full_vocabulary_rank": 3,
        "full_vocabulary_denominator": 5,
        "rank_tie_policy": "1_plus_count_strictly_greater",
        "score_kind": "raw_readout_logit",
        "vocabulary_filter": {
            "display_lexical_filter_applied": True,
            "display_lexical_eligible": True,
            "character_length_filter_applied": False,
            "decoded_character_length": 3,
            "character_length_constraint": None,
        },
    }
    early_control = payload["decoder"]["cells"][0][1]["realized_token"]
    assert early_control["id"] == 3
    assert early_control["rank"] == 4
    assert early_control["rank_denominator"] == 5
    assert early_control["rank_space"] == "full_model_vocabulary"
    assert early_control["display_vocabulary_rank"] is None
    assert early_control["display_vocabulary_denominator"] == 4
    assert early_control["full_vocabulary_rank"] == 4
    assert early_control["vocabulary_filter"]["display_lexical_eligible"] is False

    late_the = payload["decoder"]["cells"][1][0]["realized_token"]
    late_control = payload["decoder"]["cells"][1][1]["realized_token"]
    assert late_the["rank"] == 1
    assert late_the["rank_space"] == "lexical_display_vocabulary"
    assert late_control["rank"] == 1
    assert late_control["rank_space"] == "full_model_vocabulary"
    assert late_control["display_vocabulary_rank"] is None

    # The bounded top-token lists remain display-filtered and unchanged in
    # shape; realized ranks are an independent full-logit diagnostic.
    for layer_cells in payload["decoder"]["cells"]:
        for cell in layer_cells:
            assert len(cell["top_tokens"]) == 1
            assert cell["top_tokens"][0]["id"] != 3


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
