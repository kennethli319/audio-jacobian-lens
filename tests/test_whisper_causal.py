from __future__ import annotations

import pytest
import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.hooks import DecoderResidualScheduleAdder
from jlens.whisper_causal import (
    DecoderIntervention,
    DecoderInterventionSchedule,
    EncoderIntervention,
    EncoderInterventionSchedule,
    candidate_text_token_ids,
    decoder_lens_contrast_direction,
    encoder_lens_contrast_direction,
    prepare_candidate_inputs,
    random_encoder_direction,
    run_decoder_intervention_schedule,
    run_encoder_intervention,
    run_encoder_intervention_schedule,
    score_candidate_text,
    vocabulary_token_ids_starting_with,
)
from jlens.whisper_lens import WhisperJacobianLens
from tests.test_whisper_hf_adapter import random_hf_whisper, random_inputs
from tests.tiny_whisper import TinyWhisperLensModel, tiny_inputs


class _CandidateTokenizer:
    values = {" Yanny": [3, 4], " Laurel": [5]}

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": self.values[text]}


def test_encoder_intervention_recomputes_all_downstream_states():
    model = TinyWhisperLensModel(encoder_layers=3, decoder_layers=3)
    trace = run_encoder_intervention(
        model,
        tiny_inputs(model),
        EncoderIntervention(
            layer=1,
            start_position=1,
            end_position=3,
            direction=torch.tensor([1.0, -2.0, 0.5]),
            strength=0.2,
        ),
    )

    torch.testing.assert_close(
        trace.steered_encoder[1] - trace.baseline_encoder[1], trace.delta
    )
    assert trace.delta[:, 0].abs().sum() == 0
    assert trace.delta[:, 3:].abs().sum() == 0
    assert trace.encoder_change_norms()[2].max() > 0
    assert trace.decoder_change_norms()[0].max() > 0
    assert (trace.steered_logits - trace.baseline_logits).abs().max() > 0


def test_zero_strength_is_a_noop():
    model = TinyWhisperLensModel()
    trace = run_encoder_intervention(
        model,
        tiny_inputs(model),
        EncoderIntervention(
            layer=0,
            start_position=0,
            end_position=2,
            direction=torch.tensor([1.0, 0.0, 0.0]),
            strength=0.0,
        ),
    )
    assert trace.delta.abs().sum() == 0
    for layer in trace.baseline_encoder:
        torch.testing.assert_close(
            trace.steered_encoder[layer], trace.baseline_encoder[layer]
        )
    for layer in trace.baseline_decoder:
        torch.testing.assert_close(
            trace.steered_decoder[layer], trace.baseline_decoder[layer]
        )
    torch.testing.assert_close(trace.steered_logits, trace.baseline_logits)


def test_ordered_schedule_applies_multiple_encoder_edits():
    model = TinyWhisperLensModel(encoder_layers=3, decoder_layers=2)
    schedule = EncoderInterventionSchedule(
        (
            EncoderIntervention(0, 0, 2, torch.tensor([1.0, 0.0, 0.0]), 0.1),
            EncoderIntervention(1, 0, 2, torch.tensor([0.0, 1.0, 0.0]), 0.1),
            EncoderIntervention(2, 0, 2, torch.tensor([0.0, 0.0, 1.0]), 0.1),
        )
    )
    trace = run_encoder_intervention_schedule(model, tiny_inputs(model), schedule)
    torch.testing.assert_close(
        trace.steered_encoder[0] - trace.baseline_encoder[0], trace.deltas[0]
    )
    assert set(trace.deltas) == {0, 1, 2}
    assert trace.encoder_change_norms()[2].max() > trace.deltas[2].norm()
    assert trace.decoder_change_norms()[1].max() > 0
    with pytest.raises(ValueError, match="multiple"):
        _ = trace.intervention


def test_schedule_allows_same_layer_multiple_time_spans_but_rejects_reordering():
    first = EncoderIntervention(0, 0, 1, torch.ones(3), 0.1)
    second = EncoderIntervention(1, 0, 1, torch.ones(3), 0.1)
    same_layer_other_span = EncoderIntervention(0, 1, 2, torch.ones(3), 0.1)
    schedule = EncoderInterventionSchedule((first, same_layer_other_span, second))
    assert len(schedule.interventions) == 3
    with pytest.raises(ValueError, match="ordered"):
        EncoderInterventionSchedule((second, first))


def test_same_layer_piecewise_edits_are_combined_once_at_that_layer():
    model = TinyWhisperLensModel(encoder_layers=2, decoder_layers=1)
    schedule = EncoderInterventionSchedule(
        (
            EncoderIntervention(0, 0, 1, torch.tensor([1.0, 0.0, 0.0]), 0.1),
            EncoderIntervention(0, 1, 2, torch.tensor([0.0, 1.0, 0.0]), 0.1),
            EncoderIntervention(1, 0, 2, torch.tensor([0.0, 0.0, 1.0]), 0.1),
        )
    )
    trace = run_encoder_intervention_schedule(model, tiny_inputs(model), schedule)
    torch.testing.assert_close(
        trace.steered_encoder[0] - trace.baseline_encoder[0], trace.deltas[0]
    )
    assert trace.deltas[0][:, 0].norm() > 0
    assert trace.deltas[0][:, 1].norm() > 0


def test_decoder_schedule_edits_exact_positions_and_downstream_layers():
    model = TinyWhisperLensModel(encoder_layers=2, decoder_layers=3)
    inputs = tiny_inputs(model)
    schedule = DecoderInterventionSchedule(
        (
            DecoderIntervention(0, 1, torch.arange(1, 6).float(), 0.1),
            DecoderIntervention(0, 2, torch.arange(5, 0, -1).float(), 0.1),
            DecoderIntervention(1, 1, torch.ones(5), 0.1),
        )
    )
    trace = run_decoder_intervention_schedule(model, inputs, schedule)

    layer_zero_change = trace.steered_decoder[0] - trace.baseline_decoder[0]
    torch.testing.assert_close(layer_zero_change[:, 0], torch.zeros_like(layer_zero_change[:, 0]))
    torch.testing.assert_close(
        layer_zero_change[:, 1], trace.vectors[0][1].unsqueeze(0)
    )
    torch.testing.assert_close(
        layer_zero_change[:, 2], trace.vectors[0][2].unsqueeze(0)
    )
    assert trace.decoder_change_norms()[2].max() > 0
    torch.testing.assert_close(
        trace.steered_logits[:, 0], trace.baseline_logits[:, 0]
    )
    assert (trace.steered_logits - trace.baseline_logits).abs().max() > 0


def test_decoder_vectors_score_candidates_with_different_sequence_lengths():
    model = TinyWhisperLensModel(decoder_layers=3)
    model.tokenizer = _CandidateTokenizer()
    model.decoder_prefix_ids = [0]
    inputs = tiny_inputs(model)
    positive_inputs, _, first_position = prepare_candidate_inputs(
        model, inputs, " Yanny"
    )
    schedule = DecoderInterventionSchedule(
        (
            DecoderIntervention(0, first_position, torch.ones(5), 0.1),
            DecoderIntervention(0, first_position + 1, -torch.ones(5), 0.1),
        )
    )
    trace = run_decoder_intervention_schedule(model, positive_inputs, schedule)

    baseline = score_candidate_text(model, inputs, " Laurel")
    steered = score_candidate_text(
        model, inputs, " Laurel", decoder_vectors=trace.vectors
    )
    assert len(baseline.token_log_probabilities) == 1
    assert steered.total_log_probability != pytest.approx(
        baseline.total_log_probability
    )


def test_decoder_position_adder_matches_full_and_cached_hf_execution():
    model = random_hf_whisper()
    input_ids = torch.tensor([[1, 3, 4]])
    encoder_hidden = torch.randn(1, 4, model.decoder_dim)
    vectors = {
        0: {
            1: torch.linspace(-0.2, 0.2, model.decoder_dim),
            2: torch.linspace(0.3, -0.3, model.decoder_dim),
        }
    }

    with DecoderResidualScheduleAdder(
        model.decoder,
        model.decoder_layers,
        vectors_by_layer=vectors,
    ):
        full = model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        ).last_hidden_state

    with DecoderResidualScheduleAdder(
        model.decoder,
        model.decoder_layers,
        vectors_by_layer=vectors,
    ):
        prefix = model.decoder(
            input_ids=input_ids[:, :2],
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )
        final = model.decoder(
            input_ids=input_ids[:, 2:],
            encoder_hidden_states=encoder_hidden,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).last_hidden_state

    torch.testing.assert_close(final[:, 0], full[:, 2])


def test_encoder_intervention_handles_huggingface_tuple_outputs():
    model = random_hf_whisper()
    trace = run_encoder_intervention(
        model,
        random_inputs(),
        EncoderIntervention(
            layer=0,
            start_position=0,
            end_position=2,
            direction=torch.arange(1, 9, dtype=torch.float32),
            strength=0.1,
        ),
    )
    torch.testing.assert_close(
        trace.steered_encoder[0] - trace.baseline_encoder[0], trace.delta
    )
    assert trace.encoder_change_norms()[1].max() > 0
    assert trace.decoder_change_norms()[1].max() > 0


def test_intervention_rejects_invalid_window_and_direction():
    model = TinyWhisperLensModel()
    with pytest.raises(ValueError, match="outside"):
        run_encoder_intervention(
            model,
            tiny_inputs(model),
            EncoderIntervention(
                layer=0,
                start_position=0,
                end_position=99,
                direction=torch.ones(3),
                strength=0.1,
            ),
        )
    with pytest.raises(ValueError, match="nonzero"):
        run_encoder_intervention(
            model,
            tiny_inputs(model),
            EncoderIntervention(
                layer=0,
                start_position=0,
                end_position=1,
                direction=torch.zeros(3),
                strength=0.1,
            ),
        )


def test_encoder_lens_contrast_direction_subtracts_candidate_means():
    matrix = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    encoder = CrossJacobianLens(
        {0: matrix},
        n_examples=1,
        source_dim=3,
        target_dim=5,
        source_stream="encoder",
        target_stream="decoder",
    )
    lens = WhisperJacobianLens(
        encoder=encoder,
        model_metadata={"model_id": "test"},
        estimator_metadata={},
    )
    model = type(
        "Model",
        (),
        {
            "vocab_size": 4,
            "unembedding_weight": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
        },
    )()
    direction = encoder_lens_contrast_direction(
        model,
        lens,
        layer=0,
        positive_token_ids=[2, 3],
        negative_token_ids=[0],
    )
    expected = (matrix[2] + matrix[3]) / 2 - matrix[0]
    torch.testing.assert_close(direction, expected)
    with pytest.raises(ValueError, match="must differ"):
        encoder_lens_contrast_direction(
            model,
            lens,
            layer=0,
            positive_token_ids=[2, 3],
            negative_token_ids=[3, 2],
        )


def test_decoder_lens_contrast_direction_subtracts_candidate_means():
    matrix = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    decoder = CrossJacobianLens(
        {0: matrix},
        n_examples=1,
        source_dim=5,
        target_dim=5,
        source_stream="decoder",
        target_stream="decoder",
    )
    lens = WhisperJacobianLens(
        decoder=decoder,
        model_metadata={"model_id": "test"},
        estimator_metadata={},
    )
    model = type(
        "Model",
        (),
        {
            "vocab_size": 4,
            "unembedding_weight": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
        },
    )()
    direction = decoder_lens_contrast_direction(
        model,
        lens,
        layer=0,
        positive_token_ids=[2, 3],
        negative_token_ids=[0],
    )
    expected = (matrix[2] + matrix[3]) / 2 - matrix[0]
    torch.testing.assert_close(direction, expected)


def test_candidate_scores_exclude_prefix_and_support_intervention():
    model = TinyWhisperLensModel()
    model.tokenizer = _CandidateTokenizer()
    model.decoder_prefix_ids = [0]
    inputs = tiny_inputs(model)
    baseline = score_candidate_text(model, inputs, " Yanny")
    assert candidate_text_token_ids(model, " Laurel") == [5]
    assert baseline.token_ids == (3, 4)
    assert len(baseline.token_ranks) == 2
    assert all(1 <= rank <= baseline.rank_denominator for rank in baseline.token_ranks)
    assert baseline.rank_denominator == model.vocab_size
    assert baseline.mean_log_probability == pytest.approx(
        baseline.total_log_probability / 2
    )

    trace = run_encoder_intervention(
        model,
        inputs,
        EncoderIntervention(
            layer=0,
            start_position=0,
            end_position=2,
            direction=torch.tensor([1.0, 1.0, -1.0]),
            strength=0.2,
        ),
    )
    steered = score_candidate_text(
        model,
        inputs,
        " Yanny",
        intervention=trace.intervention,
        delta=trace.delta,
    )
    assert steered.total_log_probability != pytest.approx(
        baseline.total_log_probability
    )


def test_random_direction_is_reproducible_and_nonzero():
    first = random_encoder_direction(4, seed=7)
    second = random_encoder_direction(4, seed=7)
    third = random_encoder_direction(4, seed=8)
    torch.testing.assert_close(first, second)
    assert first.norm() > 0
    assert not torch.equal(first, third)


def test_vocabulary_prefix_family_ignores_only_leading_whitespace():
    class Tokenizer:
        all_special_ids = [0]
        pieces = {0: "<special>", 1: " Y", 2: "You", 3: " y", 4: "XY"}

        def decode(self, token_ids, **_kwargs):
            return self.pieces[token_ids[0]]

    model = type(
        "PrefixModel",
        (),
        {"tokenizer": Tokenizer(), "vocab_size": 5},
    )()
    assert vocabulary_token_ids_starting_with(model, "Y") == [1, 2]
    assert vocabulary_token_ids_starting_with(model, " Y") == [1, 2]
    assert vocabulary_token_ids_starting_with(
        model, " Y", strip_leading_whitespace=False
    ) == [1]
    with pytest.raises(ValueError, match="no ordinary"):
        vocabulary_token_ids_starting_with(model, "La")
