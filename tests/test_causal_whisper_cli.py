from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from jlens.causal_whisper_cli import (
    _candidate_outcomes,
    _parse_segments,
    _span_from_seconds,
    _steered_generation,
    _tokenizer_faithful_segments,
)


def test_candidate_outcomes_makes_total_path_primary_and_retains_mean(monkeypatch):
    scores = {
        " A": SimpleNamespace(
            text=" A",
            token_ids=(0,),
            token_log_probabilities=(-2.0,),
            total_log_probability=-2.0,
            mean_log_probability=-2.0,
        ),
        " B": SimpleNamespace(
            text=" B",
            token_ids=(1, 2),
            token_log_probabilities=(-1.1, -1.1),
            total_log_probability=-2.2,
            mean_log_probability=-1.1,
        ),
    }
    monkeypatch.setattr(
        "jlens.causal_whisper_cli.score_candidate_text",
        lambda _model, _inputs, text, **_kwargs: scores[text],
    )

    outcomes = _candidate_outcomes(
        SimpleNamespace(),
        SimpleNamespace(),
        positive_text=" A",
        negative_text=" B",
        comparison_texts=[" A", " B"],
        logits=torch.zeros(1, 1, 3),
        first_prediction_position=0,
    )

    first, second = outcomes["comparison_set"]
    assert outcomes["comparison_primary_metric"].startswith("restricted_total")
    assert outcomes["candidate_token_paths_include_eos"] is False
    assert outcomes["total_log_probability_positive_minus_negative"] == pytest.approx(
        0.2
    )
    assert outcomes["mean_log_probability_positive_minus_negative"] == pytest.approx(
        -0.9
    )
    assert first["restricted_total_log_probability_softmax"] > second[
        "restricted_total_log_probability_softmax"
    ]
    assert first["restricted_mean_log_probability_softmax"] < second[
        "restricted_mean_log_probability_softmax"
    ]


def test_span_from_seconds_maps_to_exact_20ms_positions():
    assert _span_from_seconds(
        start_seconds=0.18, end_seconds=0.38, valid_positions=47
    ) == (9, 19)
    assert _span_from_seconds(
        start_seconds=0.9, end_seconds=1.1, valid_positions=47
    ) == (45, 47)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-0.01, 0.2, "nonnegative"), (0.2, 0.2, "greater"), (1.0, 1.2, "no valid")],
)
def test_span_from_seconds_rejects_invalid_or_empty_spans(start, end, message):
    with pytest.raises(ValueError, match=message):
        _span_from_seconds(start_seconds=start, end_seconds=end, valid_positions=10)


def test_parse_segments_preserves_target_piece_spaces():
    assert _parse_segments(["0:0.3: Y", "0.3:0.9:anny"]) == [
        (0.0, 0.3, " Y"),
        (0.3, 0.9, "anny"),
    ]


def test_parse_segments_rejects_invalid_shape():
    with pytest.raises(ValueError, match="start:end:text"):
        _parse_segments(["0:0.3"])


def test_tokenizer_faithful_segments_divide_the_span_by_bpe_piece():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            assert text == " Yanny"
            assert add_special_tokens is False
            return {"input_ids": [10, 11]}

        def decode(self, token_ids, **_kwargs):
            return {10: " Y", 11: "anny"}[token_ids[0]]

    model = SimpleNamespace(tokenizer=Tokenizer())
    assert _tokenizer_faithful_segments(
        model, " Yanny", start_seconds=0.2, end_seconds=0.6
    ) == [(0.2, 0.4, " Y"), (0.4, 0.6, "anny")]


def test_steered_generation_applies_each_aggregated_layer_delta_once(monkeypatch):
    entered_layers = []

    class FakeResidualAdder:
        def __init__(self, _layers, *, layer, delta):
            entered_layers.append((layer, delta))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Model:
        encoder_layers = object()

        def generate(self, input_features, *, attention_mask):
            assert input_features == "features"
            assert attention_mask == "mask"
            return "generated"

    monkeypatch.setattr(
        "jlens.causal_whisper_cli.ResidualAdder", FakeResidualAdder
    )
    schedule = SimpleNamespace(
        interventions=(
            SimpleNamespace(layer=1),
            SimpleNamespace(layer=1),
            SimpleNamespace(layer=2),
        )
    )
    deltas = {1: "combined-l1", 2: "combined-l2"}

    assert _steered_generation(
        Model(),
        input_features="features",
        attention_mask="mask",
        schedule=schedule,
        deltas=deltas,
    ) == "generated"
    assert entered_layers == [(1, "combined-l1"), (2, "combined-l2")]
