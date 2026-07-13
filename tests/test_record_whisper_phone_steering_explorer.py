from __future__ import annotations

import numpy as np
import pytest
import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.whisper import WhisperLensInputs
from jlens.whisper_analysis import WhisperAnalysisCapture, analyze_whisper_run
from jlens.whisper_lens import WhisperJacobianLens
from scripts.record_whisper_phone_steering_explorer import (
    _canonical_condition_metadata,
    _scaled_laurel_coefficients,
)


class _Tokenizer:
    all_special_ids = [0]
    _tokens = ["<|start|>", " a", "b", "cc", "d", "e"]

    def decode(self, token_ids, **_kwargs):
        return self._tokens[int(token_ids[0])]


class _NoCaptureModel:
    model_id = "captured-test"
    fingerprint = "captured-test-v1"
    vocab_size = 6
    tokenizer = _Tokenizer()

    @staticmethod
    def unembed(hidden):
        return hidden

    @staticmethod
    def capture(*_args, **_kwargs):
        raise AssertionError("pre-captured analysis must not recapture the model")


def _inputs() -> WhisperLensInputs:
    return WhisperLensInputs(
        input_features=torch.zeros(1, 2, 4),
        decoder_input_ids=torch.tensor([[0, 1]]),
        decoder_target_ids=torch.tensor([[1, 2]]),
        encoder_position_mask=torch.tensor([[True, True]]),
        decoder_position_mask=torch.tensor([[True, True]]),
        duration_seconds=0.04,
    )


def _lens() -> WhisperJacobianLens:
    common = {0: torch.eye(6)}
    encoder = CrossJacobianLens(
        common,
        n_examples=1,
        source_dim=6,
        target_dim=6,
        source_stream="encoder",
        target_stream="decoder",
        source_means={0: torch.zeros(6)},
        target_mean=torch.zeros(6),
        metadata={"estimator_name": "test"},
    )
    decoder = CrossJacobianLens(
        common,
        n_examples=1,
        source_dim=6,
        target_dim=6,
        source_stream="decoder",
        target_stream="decoder",
        metadata={"estimator_name": "test"},
    )
    return WhisperJacobianLens(
        encoder=encoder,
        decoder=decoder,
        model_metadata={"model_fingerprint": _NoCaptureModel.fingerprint},
        estimator_metadata={},
    )


def test_analysis_can_serialize_strictly_validated_pre_captured_tensors():
    capture = WhisperAnalysisCapture(
        encoder_activations={0: torch.ones(1, 2, 6)},
        decoder_activations={0: torch.ones(1, 2, 6)},
        actual_logits=torch.zeros(1, 2, 6),
    )
    report = analyze_whisper_run(
        _NoCaptureModel(),
        _lens(),
        _inputs(),
        np.zeros(640, dtype=np.float32),
        captured=capture,
        top_k=2,
        time_bin_seconds=0.02,
        time_bin_overlap_seconds=0.0,
    )
    assert report["encoder"]["layers"] == [0]
    assert report["decoder"]["layers"] == [0]
    assert report["transcription"]["text"] == " ab"


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (
            WhisperAnalysisCapture({}, {0: torch.ones(1, 2, 6)}, torch.zeros(1, 2, 6)),
            "captured encoder layers",
        ),
        (
            WhisperAnalysisCapture(
                {0: torch.ones(1, 2, 6)},
                {0: torch.ones(1, 2, 6)},
                torch.zeros(1, 1, 6),
            ),
            "captured output logits has shape",
        ),
        (
            WhisperAnalysisCapture(
                {0: torch.full((1, 2, 6), float("nan"))},
                {0: torch.ones(1, 2, 6)},
                torch.zeros(1, 2, 6),
            ),
            "contains non-finite values",
        ),
    ],
)
def test_pre_captured_analysis_rejects_misaligned_or_nonfinite_tensors(
    capture, message
):
    with pytest.raises(ValueError, match=message):
        capture.validate(model=_NoCaptureModel(), lens=_lens(), inputs=_inputs())


def test_recorder_copies_public_evidence_and_scales_frozen_laurel_ray():
    target = {
        "label": "Laurel",
        "evidence": {"tier": "target_conditioned_clip_specific_existence"},
        "method": {"kind": "phone_pullback"},
        "layers": [0, 1],
        "schedule": [
            {"phone": "L", "start_position": 4, "end_position": 9},
            {"phone": "AO", "start_position": 9, "end_position": 18},
        ],
        "checkpoints": [
            {
                "id": "recommended",
                "recorded": True,
                "interpolated": False,
                "generated": {"text": "Laurel", "token_ids": [43442]},
                "budget_fraction": 0.145,
                "coefficient_scale": 0.7,
            }
        ],
    }
    condition = _canonical_condition_metadata({"targets": {"laurel": target}}, "laurel")
    coefficients = [
        {
            "layer": layer,
            "segment_index": segment_index,
            "phone": segment["phone"],
            "start_position": segment["start_position"],
            "end_position": segment["end_position"],
            "coefficient": 0.1 * (1 + layer + segment_index),
        }
        for layer in condition["layers"]
        for segment_index, segment in enumerate(condition["schedule"])
    ]
    recipe = {
        "format": "audio-jlens-private-laurel-phone-basis-search-v1",
        "best": {"coefficients": coefficients},
    }
    scaled = _scaled_laurel_coefficients(recipe, condition, scale=0.7)
    assert condition["evidence"] == target["evidence"]
    assert [item["coefficient"] for item in scaled] == pytest.approx(
        [item["source_coefficient"] * 0.7 for item in scaled]
    )
    assert [(item["layer"], item["segment_index"]) for item in scaled] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
