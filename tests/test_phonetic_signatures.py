from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from jlens.cross_lens import CrossJacobianLens
from jlens.phonetic_signatures import (
    PhoneSignaturePrototypes,
    encoder_lens_fingerprint,
)


class _LinearReadoutModel:
    vocab_size = 6
    fingerprint = "phone-signature-test-v1"

    _weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    @classmethod
    def unembed(cls, residuals: torch.Tensor) -> torch.Tensor:
        return residuals @ cls._weight.T.to(residuals.device)


class _LayerNormReadoutModel:
    vocab_size = 6
    fingerprint = "phone-signature-layer-norm-test-v1"

    _weight = torch.tensor(
        [
            [1.0, 0.5, -0.25],
            [0.0, 1.0, 0.0],
            [0.5, -0.5, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    @classmethod
    def unembed(cls, residuals: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(residuals, (3,))
        return normalized @ cls._weight.T.to(residuals.device)


class _ZeroGradientReadoutModel:
    vocab_size = 6
    fingerprint = "phone-signature-test-v1"

    @classmethod
    def unembed(cls, residuals: torch.Tensor) -> torch.Tensor:
        zero = residuals.sum(dim=-1, keepdim=True) * 0
        return zero.expand(*residuals.shape[:-1], cls.vocab_size)


def _lens(*, scale: float = 1.0) -> CrossJacobianLens:
    return CrossJacobianLens(
        {0: torch.eye(3) * scale},
        n_examples=4,
        source_dim=3,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        source_means={0: torch.zeros(3)},
        target_mean=torch.zeros(3),
        metadata={"estimator_name": "phone-signature-test"},
    )


def _prototypes(lens: CrossJacobianLens) -> PhoneSignaturePrototypes:
    values = torch.zeros(2, _LinearReadoutModel.vocab_size)
    values[0, 0] = 1
    values[1, 1] = 1
    return PhoneSignaturePrototypes(
        {0: values},
        labels=["AA", "B"],
        signature_top_k=2,
        vocab_size=_LinearReadoutModel.vocab_size,
        model_fingerprint=_LinearReadoutModel.fingerprint,
        encoder_lens_fingerprint_value=encoder_lens_fingerprint(lens),
        metadata={"source": {"split": "train"}},
    )


def _layer_norm_lens() -> CrossJacobianLens:
    return CrossJacobianLens(
        {
            0: torch.tensor(
                [
                    [1.0, 0.2, 0.0],
                    [0.0, 0.8, -0.1],
                    [0.3, 0.0, 1.2],
                ]
            )
        },
        n_examples=4,
        source_dim=3,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        source_means={0: torch.tensor([0.1, -0.2, 0.3])},
        target_mean=torch.tensor([0.4, -0.5, 0.8]),
        metadata={"estimator_name": "phone-signature-layer-norm-test"},
    )


def _dense_prototypes(lens: CrossJacobianLens) -> PhoneSignaturePrototypes:
    values = torch.zeros(2, _LayerNormReadoutModel.vocab_size)
    values[0, 0] = 2**-0.5
    values[0, 2] = 2**-0.5
    values[1, 1] = 1
    return PhoneSignaturePrototypes(
        {0: values},
        labels=["AE", "L"],
        signature_top_k=2,
        vocab_size=_LayerNormReadoutModel.vocab_size,
        model_fingerprint=_LayerNormReadoutModel.fingerprint,
        encoder_lens_fingerprint_value=encoder_lens_fingerprint(lens),
    )


def test_phone_signature_scores_match_rank_thresholded_sparse_cosine():
    lens = _lens()
    prototypes = _prototypes(lens)
    rows = prototypes.score_layer(
        _LinearReadoutModel(),
        lens,
        torch.tensor([[3.0, 2.0, 1.0]]),
        layer=0,
        top_n=2,
    )

    assert [candidate["phone"] for candidate in rows[0]] == ["AA", "B"]
    assert rows[0][0]["similarity"] == torch.tensor(2 / 5**0.5).item()
    assert rows[0][1]["similarity"] == torch.tensor(1 / 5**0.5).item()
    assert [candidate["rank"] for candidate in rows[0]] == [1, 2]
    assert all(candidate["rank_denominator"] == 2 for candidate in rows[0])
    assert all(candidate["is_probability"] is False for candidate in rows[0])


def test_phone_signature_zero_norm_returns_no_arbitrary_phone():
    lens = _lens(scale=0.0)
    prototypes = _prototypes(lens)
    rows = prototypes.score_layer(
        _LinearReadoutModel(),
        lens,
        torch.tensor([[3.0, 2.0, 1.0]]),
        layer=0,
    )
    assert rows == [[]]


def test_phone_prototype_logit_pullback_returns_source_gradient_and_metadata():
    lens = _lens()
    prototypes = _prototypes(lens)
    result = prototypes.prototype_logit_pullback(
        _LinearReadoutModel(),
        lens,
        torch.tensor([3.0, 2.0, 1.0]),
        layer=0,
        target_phone="AA",
    )

    torch.testing.assert_close(result.direction, torch.tensor([1.0, -1.0, 0.0]))
    assert result.direction.requires_grad is False
    assert result.layer == 0
    assert result.target_phone == "AA"
    assert result.target_readout_score == pytest.approx(3.0)
    assert result.other_mean_readout_score == pytest.approx(2.0)
    assert result.objective_value == pytest.approx(1.0)
    assert result.competing_phone_count == 1
    assert result.gradient_l2_norm == pytest.approx(2**0.5)
    assert result.objective_kind == (
        "target_phone_prototype_minus_other_phone_mean_readout_logit"
    )
    assert result.is_probability is False


def test_phone_prototype_pullback_uses_full_pattern_transport_and_final_norm():
    lens = _layer_norm_lens()
    prototypes = _dense_prototypes(lens)
    reference = torch.tensor([0.9, -0.1, 1.4])
    result = prototypes.prototype_logit_pullback(
        _LayerNormReadoutModel(),
        lens,
        reference,
        layer=0,
        target_phone="AE",
    )

    source = reference.clone().requires_grad_(True)
    transported = lens.transport(source.unsqueeze(0), 0)
    logits = _LayerNormReadoutModel.unembed(transported)
    baseline = _LayerNormReadoutModel.unembed(lens.target_mean.unsqueeze(0))
    contrast = prototypes.prototypes[0][0] - prototypes.prototypes[0][1]
    objective = ((logits - baseline) * contrast).sum()
    (expected,) = torch.autograd.grad(objective, source)

    torch.testing.assert_close(result.direction, expected)
    assert torch.isfinite(result.direction).all()
    assert result.direction.norm() > 0


@pytest.mark.parametrize(
    ("reference", "layer", "target_phone", "message"),
    [
        (torch.zeros(2), 0, "AA", "reference_residual"),
        (torch.zeros(3), 1, "AA", "no layer 1"),
        (torch.zeros(3), 0, "Y", "no label 'Y'"),
        (torch.tensor([0.0, float("nan"), 0.0]), 0, "AA", "non-finite"),
    ],
)
def test_phone_prototype_pullback_rejects_invalid_coordinate(
    reference: torch.Tensor,
    layer: int,
    target_phone: str,
    message: str,
):
    lens = _lens()
    with pytest.raises(ValueError, match=message):
        _prototypes(lens).prototype_logit_pullback(
            _LinearReadoutModel(),
            lens,
            reference,
            layer=layer,
            target_phone=target_phone,
        )


def test_phone_prototype_pullback_rejects_zero_gradient():
    lens = _lens()
    with pytest.raises(ValueError, match="zero norm"):
        _prototypes(lens).prototype_logit_pullback(
            _ZeroGradientReadoutModel(),
            lens,
            torch.ones(3),
            layer=0,
            target_phone="AA",
        )


def test_phone_signature_artifact_round_trip_and_compatibility(tmp_path):
    lens = _lens()
    prototypes = _prototypes(lens)
    path = tmp_path / "phones.pt"
    prototypes.save(path, dtype=torch.float32)
    loaded = PhoneSignaturePrototypes.load(path)

    loaded.validate(model=_LinearReadoutModel(), encoder_lens=lens)
    assert loaded.labels == ["AA", "B"]
    assert loaded.public_metadata()["score_kind"] == (
        "phone_prototype_cosine_similarity"
    )


def test_phone_signature_rejects_a_different_encoder_lens():
    lens = _lens()
    prototypes = _prototypes(lens)
    try:
        prototypes.validate(
            model=_LinearReadoutModel(),
            encoder_lens=_lens(scale=0.5),
        )
    except ValueError as error:
        assert "different encoder lens" in str(error)
    else:
        raise AssertionError("mismatched encoder lens was accepted")
