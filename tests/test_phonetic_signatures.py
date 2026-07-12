from __future__ import annotations

import torch

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
