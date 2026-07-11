from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.chatterbox_fitting import (
    CHATTERBOX_CAPTURE_CONVENTION,
    CHATTERBOX_SOURCE_STREAM,
    CHATTERBOX_TARGET_STREAM,
    ChatterboxLensExample,
    chatterbox_examples_fingerprint,
    fit_mlx_chatterbox_speech_lens,
    projected_vjps_for_chatterbox_example,
    validate_chatterbox_speech_lens,
)
from jlens.mlx_fitting import PROJECTION_METHOD, hadamard_target_factors
from jlens.projected_lens import ProjectedCrossJacobianLens


def _example(*, selected: tuple[int, ...] | None = None) -> ChatterboxLensExample:
    return ChatterboxLensExample(
        text_token_ids=(4,),
        speech_code_ids=(2, 3),
        selected_code_indices=selected,
        raw_text="Tiny.",
        normalized_text="Tiny.",
        record_id="tiny-1",
    )


def test_chatterbox_lens_example_validation_and_fingerprint():
    example = _example()
    assert example.code_indices == (0, 1)
    assert _example(selected=(1,)).code_indices == (1,)
    assert chatterbox_examples_fingerprint([example]) == chatterbox_examples_fingerprint(
        [example]
    )
    assert chatterbox_examples_fingerprint([example]) != chatterbox_examples_fingerprint(
        [_example(selected=(1,))]
    )
    with pytest.raises(ValueError, match="sorted"):
        _example(selected=(1, 0))
    with pytest.raises(ValueError, match="exceed"):
        _example(selected=(2,))
    with pytest.raises(ValueError, match="at least one"):
        chatterbox_examples_fingerprint([])


class _ArtifactModel:
    hidden_size = 4
    n_layers = 3

    def __init__(self, fingerprint: str = "tiny-model") -> None:
        self.fingerprint = fingerprint

    def metadata(self):
        return {
            "model_fingerprint": self.fingerprint,
            "model_id": "tiny/chatterbox",
            "speech_vocab_size": 6,
        }


def _artifact(*, fingerprint: str = "tiny-model") -> ProjectedCrossJacobianLens:
    factors = hadamard_target_factors(4, 4, seed=3)
    return ProjectedCrossJacobianLens(
        factors,
        {0: factors.clone(), 1: factors.clone()},
        n_examples=2,
        source_dim=4,
        target_dim=4,
        source_stream=CHATTERBOX_SOURCE_STREAM,
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
        metadata={
            "model_fingerprint": fingerprint,
            "source_layers": [0, 1],
            "target_layer": 2,
            "projection_method": PROJECTION_METHOD,
            "capture_convention": CHATTERBOX_CAPTURE_CONVENTION,
            "centered": False,
            "target_head": {
                "name": "t3.speech_head",
                "semantic_kind": "speech_code",
                "vocab_size": 6,
                "valid_ordinary_codes": 6,
            },
        },
    )


def test_chatterbox_projected_artifact_roundtrip_and_validation(tmp_path):
    path = tmp_path / "chatterbox-lens.pt"
    _artifact().save(path, dtype=torch.float32)
    loaded = ProjectedCrossJacobianLens.load(path)
    validate_chatterbox_speech_lens(_ArtifactModel(), loaded)
    assert loaded.source_layers == [0, 1]
    assert loaded.n_examples == 2
    torch.testing.assert_close(loaded.target_factors, _artifact().target_factors)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_chatterbox_speech_lens(_ArtifactModel("other-model"), loaded)
    wrong_stream = ProjectedCrossJacobianLens(
        loaded.target_factors,
        loaded.source_factors,
        n_examples=loaded.n_examples,
        source_dim=4,
        target_dim=4,
        source_stream="text",
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
        metadata=loaded.metadata,
    )
    with pytest.raises(ValueError, match="speech positions"):
        validate_chatterbox_speech_lens(_ArtifactModel(), wrong_stream)

    wrong_centering = _artifact()
    wrong_centering.metadata["centered"] = True
    with pytest.raises(ValueError, match="centering metadata"):
        validate_chatterbox_speech_lens(_ArtifactModel(), wrong_centering)

    wrong_head = _artifact()
    wrong_head.metadata["target_head"]["semantic_kind"] = "text"
    with pytest.raises(ValueError, match="target-head metadata"):
        validate_chatterbox_speech_lens(_ArtifactModel(), wrong_head)

    factors = hadamard_target_factors(4, 4, seed=3)
    late_source = ProjectedCrossJacobianLens(
        factors,
        {2: factors.clone()},
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream=CHATTERBOX_SOURCE_STREAM,
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
        metadata={
            **_artifact().metadata,
            "source_layers": [2],
        },
    )
    with pytest.raises(ValueError, match="precede"):
        validate_chatterbox_speech_lens(_ArtifactModel(), late_source)


def _tiny_mlx_model(mx):
    weights = [
        np.array(
            [
                [1.0, 0.2, 0.0, 0.1],
                [0.0, 1.1, -0.1, 0.0],
                [0.3, 0.0, 0.9, 0.0],
                [0.0, -0.2, 0.1, 1.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.9, 0.0, 0.2, 0.0],
                [0.1, 1.0, 0.0, -0.1],
                [0.0, 0.2, 1.1, 0.0],
                [-0.1, 0.0, 0.0, 0.8],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [1.0, -0.1, 0.0, 0.2],
                [0.0, 0.8, 0.2, 0.0],
                [0.1, 0.0, 1.0, -0.2],
                [0.0, 0.3, 0.0, 0.9],
            ],
            dtype=np.float32,
        ),
    ]

    class LinearBlock:
        def __init__(self, weight):
            self.weight = mx.array(weight)

        def __call__(self, hidden, cache=None):
            del cache
            return hidden @ self.weight.T, None

    class PositionEmbedding:
        @staticmethod
        def __call__(positions):
            return mx.zeros((positions.shape[0], 4))

    class TinyT3:
        hp = SimpleNamespace(start_speech_token=5)

        def __init__(self):
            self.tfmr = SimpleNamespace(
                h=[LinearBlock(weight) for weight in weights],
                wpe=PositionEmbedding(),
            )

        @staticmethod
        def prepare_input_embeds(condition, text_ids, speech_inputs):
            del condition
            length = 1 + int(text_ids.shape[1]) + int(speech_inputs.shape[1])
            values = np.arange(length * 4, dtype=np.float32).reshape(1, length, 4)
            return mx.array(values / 10.0), 1

    class TinyModel(_ArtifactModel):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                t3=TinyT3(),
                _conds=SimpleNamespace(t3=object()),
            )

    return TinyModel(), weights


def test_joint_chatterbox_vjp_reconstructs_all_source_layer_jacobians():
    mx = pytest.importorskip("mlx.core")
    model, weights = _tiny_mlx_model(mx)
    probes = hadamard_target_factors(4, 4, seed=7)
    result = projected_vjps_for_chatterbox_example(
        model,
        _example(),
        probes,
        source_layers=[0, 1],
        target_layer=2,
    )
    lens = ProjectedCrossJacobianLens(
        probes,
        result.source_factors,
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream=CHATTERBOX_SOURCE_STREAM,
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
    )
    residual = torch.tensor([[0.4, -0.2, 0.1, 0.8]])
    expected = {
        0: torch.from_numpy(weights[2] @ weights[1]),
        1: torch.from_numpy(weights[2]),
    }
    for layer in (0, 1):
        torch.testing.assert_close(
            lens.transport(residual, layer),
            residual @ expected[layer].T,
            rtol=1e-5,
            atol=1e-5,
        )

    mean_result = projected_vjps_for_chatterbox_example(
        model,
        _example(),
        probes,
        source_layers=[0, 1],
        target_layer=2,
        target_reduction="mean",
    )
    for layer in (0, 1):
        torch.testing.assert_close(
            mean_result.source_factors[layer],
            result.source_factors[layer] / 2,
        )
    torch.testing.assert_close(mean_result.target_mean, result.target_mean / 2)


def test_tiny_chatterbox_fit_averages_and_roundtrips(tmp_path):
    mx = pytest.importorskip("mlx.core")
    model, _ = _tiny_mlx_model(mx)
    lens = fit_mlx_chatterbox_speech_lens(
        model,
        [_example(), _example(selected=(0,))],
        source_layers=[0, 1],
        target_layer=2,
        projection_dim=4,
        projection_seed=2,
        center=True,
        artifact_metadata={"manifest": "tiny.jsonl"},
    )
    assert lens.n_examples == 2
    assert lens.projection_dim == 4
    assert lens.metadata["target_positions"] == 3
    assert lens.metadata["vjp_source_strategy"] == "post_block_zero_delta_injection"
    assert lens.metadata["artifact_metadata"] == {"manifest": "tiny.jsonl"}
    assert lens.source_means is not None
    assert lens.target_mean is not None
    validate_chatterbox_speech_lens(model, lens)

    path = tmp_path / "fit.pt"
    lens.save(path, dtype=torch.float32)
    loaded = ProjectedCrossJacobianLens.load(path)
    validate_chatterbox_speech_lens(model, loaded)
    for layer in lens.source_layers:
        torch.testing.assert_close(
            loaded.source_factors[layer], lens.source_factors[layer]
        )
        torch.testing.assert_close(loaded.source_means[layer], lens.source_means[layer])
    torch.testing.assert_close(loaded.target_mean, lens.target_mean)
