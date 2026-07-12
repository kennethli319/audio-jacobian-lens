from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.phonetic_signatures import (
    PhoneSignaturePrototypes,
    encoder_lens_fingerprint,
)
from jlens.whisper import (
    WhisperLensInputs,
    downsample_feature_mask,
    prediction_mask_from_targets,
)
from jlens.whisper_analysis import (
    _encoder_realized_token_alignment,
    _transcript_and_confidence,
    analyze_whisper_run,
    display_token_lengths,
)
from jlens.whisper_lens import WhisperJacobianLens, pool_sequence_residuals


def test_feature_mask_downsamples_to_20ms_encoder_positions():
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )
    actual = downsample_feature_mask(mask, encoder_positions=4)
    expected = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_prediction_mask_is_based_on_next_target_tokens():
    targets = torch.tensor([[100, 7, 8, 101, 9]])
    mask = prediction_mask_from_targets(
        targets,
        special_token_ids={100, 101},
    )
    torch.testing.assert_close(mask, torch.tensor([[False, True, True, False, True]]))


def test_pool_sequence_residuals_tracks_boundaries():
    residuals = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    mask = torch.tensor([True, True, True, True, True, False])
    pooled = pool_sequence_residuals(residuals, mask, positions_per_bin=2)
    torch.testing.assert_close(
        pooled.residuals,
        torch.stack(
            [residuals[0:2].mean(0), residuals[2:4].mean(0), residuals[4:5].mean(0)]
        ),
    )
    assert pooled.start_positions == [0, 2, 4]
    assert pooled.end_positions == [2, 4, 5]


def test_pool_sequence_residuals_supports_default_200ms_windows():
    residuals = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    mask = torch.ones(20, dtype=torch.bool)
    pooled = pool_sequence_residuals(
        residuals,
        mask,
        positions_per_bin=10,
        stride_positions=9,
    )
    torch.testing.assert_close(
        pooled.residuals,
        torch.stack(
            [
                residuals[0:10].mean(0),
                residuals[9:19].mean(0),
                residuals[18:20].mean(0),
            ]
        ),
    )
    assert pooled.start_positions == [0, 9, 18]
    assert pooled.end_positions == [10, 19, 20]


def test_pool_rejects_stride_larger_than_window():
    with pytest.raises(ValueError, match="cannot exceed"):
        pool_sequence_residuals(
            torch.zeros(4, 2),
            torch.ones(4, dtype=torch.bool),
            positions_per_bin=2,
            stride_positions=3,
        )


def test_pool_rejects_holey_mask():
    with pytest.raises(ValueError, match="contiguous"):
        pool_sequence_residuals(
            torch.zeros(4, 2),
            torch.tensor([True, False, True, False]),
            positions_per_bin=2,
        )


class _Tokenizer:
    all_special_ids = [0, 1]

    @staticmethod
    def decode(token_ids, **_kwargs):
        return str(token_ids[0])


class _DisplayTokenizer:
    all_special_ids = [0]
    _tokens = [
        "<|startoftranscript|>",
        " hello",
        " th",
        " a",
        "!",
        "語",
        "ab",
        "abc",
        "�",
    ]

    def decode(self, token_ids, **_kwargs):
        return self._tokens[token_ids[0]]


def test_display_token_lengths_cover_the_full_vocabulary():
    tokenizer = _DisplayTokenizer()
    lengths = display_token_lengths(tokenizer, len(tokenizer._tokens))
    assert lengths.tolist() == [0, 5, 2, 1, 0, 1, 2, 3, 0]


def test_decoder_length_buckets_apply_only_to_l0_and_l1():
    tokenizer = _DisplayTokenizer()
    vocab_size = len(tokenizer._tokens)
    residual = torch.arange(vocab_size, dtype=torch.float32)

    class _Model:
        model_id = "decoder-filter-test"
        fingerprint = "decoder-filter-test-v1"

        def __init__(self):
            self.tokenizer = tokenizer
            self.vocab_size = vocab_size

        @staticmethod
        def unembed(hidden):
            return hidden

        @staticmethod
        def capture(inputs, *, encoder_layers, decoder_layers):
            assert encoder_layers == []
            activations = {
                layer: residual.repeat(inputs.decoder_input_ids.shape[1], 1)[None]
                for layer in decoder_layers
            }
            actual_logits = torch.zeros(
                1, inputs.decoder_input_ids.shape[1], vocab_size
            )
            return {}, activations, actual_logits

    decoder_lens = CrossJacobianLens(
        {layer: torch.eye(vocab_size) for layer in range(3)},
        n_examples=1,
        source_dim=vocab_size,
        target_dim=vocab_size,
        source_stream="decoder",
        target_stream="decoder",
        metadata={"estimator_name": "test"},
    )
    lens = WhisperJacobianLens(
        decoder=decoder_lens,
        model_metadata={"model_fingerprint": _Model.fingerprint},
        estimator_metadata={},
    )
    inputs = WhisperLensInputs(
        input_features=torch.zeros(1, 2, 2),
        decoder_input_ids=torch.tensor([[0, 1, 2]]),
        decoder_target_ids=torch.tensor([[1, 2, 4]]),
        encoder_position_mask=torch.tensor([[True, True]]),
        decoder_position_mask=torch.tensor([[True, True, True]]),
        duration_seconds=0.04,
    )

    payload = analyze_whisper_run(
        _Model(), lens, inputs, np.zeros(640, dtype=np.float32), top_k=2
    )

    assert payload["metadata"]["decoder_token_length_filter"] == {
        "policy": "exact_decoded_character_length_buckets",
        "eligible_source_layers": [0, 1],
        "maximum_available_length": 5,
        "character_count_ignores_surrounding_whitespace": True,
    }
    assert payload["metadata"]["display_vocabulary"] == {
        "policy": "alphanumeric_lexical_tokens",
        "full_vocabulary_size": 9,
        "display_vocabulary_size": 6,
        "exact_decoded_character_length_counts": {
            "1": 2,
            "2": 2,
            "3": 1,
            "5": 1,
        },
        "maximum_decoded_character_length_counts": {
            "1": 2,
            "2": 4,
            "3": 5,
            "4": 5,
            "5": 6,
        },
    }
    assert payload["metadata"]["candidate_rank_semantics"] == {
        "method": "1_plus_count_strictly_greater",
        "ties": "equal scores share the same competition rank",
        "lens_primary_space": "lexical_display_vocabulary",
        "output_head_primary_space": "full_model_vocabulary",
        "character_filter_merge": (
            "merge disjoint exact-length buckets, sort by score, and rank by "
            "strictly greater scores"
        ),
        "realized_maximum_length_space": (
            "maximum_decoded_character_length_vocabulary"
        ),
        "realized_maximum_length_denominator_source": (
            "display_vocabulary.maximum_decoded_character_length_counts"
        ),
    }
    for layer_index in (0, 1):
        for cell in payload["decoder"]["cells"][layer_index]:
            assert set(cell["top_tokens_by_length"]) == {"1", "2", "3", "5"}
            for length, candidates in cell["top_tokens_by_length"].items():
                assert all(
                    len(candidate["text"].strip()) == int(length)
                    for candidate in candidates
                )
                for candidate in candidates:
                    assert candidate["rank_space"] == (
                        "exact_decoded_character_length_bucket"
                    )
                    assert candidate["rank_denominator"] == len(
                        [
                            token
                            for token in tokenizer._tokens
                            if token.strip().isalnum()
                            and len(token.strip()) == int(length)
                        ]
                    )
                    assert candidate["display_vocabulary_denominator"] == 6
                    assert candidate["full_vocabulary_denominator"] == 9
                    assert candidate["rank_tie_policy"] == (
                        "1_plus_count_strictly_greater"
                    )
                    assert candidate["score_kind"] == "raw_readout_logit"
                    assert candidate["vocabulary_filter"] == {
                        "display_lexical_filter_applied": True,
                        "character_length_filter_applied": True,
                        "decoded_character_length": int(length),
                        "character_length_constraint": {
                            "operator": "exact",
                            "value": int(length),
                        },
                    }

    assert all(
        "top_tokens_by_length" not in cell for cell in payload["decoder"]["cells"][2]
    )
    assert all(
        "top_tokens_by_length" not in token
        for token in payload["transcription"]["tokens"]
    )
    assert [
        [candidate["id"] for candidate in cells[0]["top_tokens"]]
        for cells in payload["decoder"]["cells"]
    ] == [[7, 6], [7, 6], [7, 6]]
    for layer_index, layer_cells in enumerate(payload["decoder"]["cells"]):
        for position_index, cell in enumerate(layer_cells):
            assert cell["position_index"] == position_index
            assert cell["time_window"] == {
                "start_seconds": None,
                "end_seconds": None,
                "timing_source": "unavailable",
            }
            assert cell["candidate_space"] == {
                "primary_rank_space": "lexical_display_vocabulary",
                "primary_rank_denominator": 6,
                "display_lexical_filter_applied": True,
                "character_length_filter_available": layer_index in (0, 1),
                "character_length_filter_policy": (
                    "exact_decoded_character_length_buckets"
                    if layer_index in (0, 1)
                    else None
                ),
            }
            candidate = cell["top_tokens"][0]
            assert candidate["rank"] == 1
            assert candidate["rank_denominator"] == 6
            assert candidate["rank_space"] == "lexical_display_vocabulary"
            assert candidate["display_vocabulary_rank"] == 1
            assert candidate["full_vocabulary_rank"] == 2
            assert candidate["vocabulary_filter"][
                "character_length_filter_applied"
            ] is False
            realized_token = cell["realized_token"]
            expected_id = payload["decoder"]["positions"][position_index]["token_id"]
            assert realized_token["id"] == expected_id
            assert realized_token["score_kind"] == "raw_readout_logit"
            assert realized_token["rank_space"] == (
                "full_model_vocabulary"
                if expected_id == 4
                else "lexical_display_vocabulary"
            )
            if layer_index in (0, 1):
                by_maximum = cell["realized_rank_by_max_length"]
                assert set(by_maximum) == {"1", "2", "3", "4", "5"}
                target_length = len(realized_token["text"].strip())
                assert all(
                    by_maximum[str(limit)] is None
                    for limit in range(1, target_length)
                )
                filtered_rank = by_maximum[str(target_length)]
                if expected_id == 4:
                    assert filtered_rank is None
                else:
                    assert isinstance(filtered_rank, int)
                    denominator = payload["metadata"]["display_vocabulary"][
                        "maximum_decoded_character_length_counts"
                    ][str(target_length)]
                    assert 1 <= filtered_rank <= denominator
            else:
                assert "realized_rank_by_max_length" not in cell

    for layer_index in (0, 1):
        first, second, punctuation = payload["decoder"]["cells"][layer_index]
        assert first["realized_rank_by_max_length"] == {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": 6,
        }
        assert second["realized_rank_by_max_length"] == {
            "1": None,
            "2": 4,
            "3": 5,
            "4": 5,
            "5": 5,
        }
        assert punctuation["realized_token"]["rank_space"] == (
            "full_model_vocabulary"
        )
        assert punctuation["realized_rank_by_max_length"] == {
            str(length): None for length in range(1, 6)
        }

    realized = payload["transcription"]["tokens"][0]
    assert realized["rank"] == 1
    assert realized["rank_denominator"] == 9
    assert realized["rank_space"] == "full_model_vocabulary"
    assert realized["full_vocabulary_rank"] == 1
    assert realized["full_vocabulary_denominator"] == 9
    assert realized["rank_tie_policy"] == "1_plus_count_strictly_greater"
    assert realized["score_kind"] == "raw_teacher_forced_probability"
    assert realized["candidate_space"] == {
        "primary_rank_space": "full_model_vocabulary",
        "primary_rank_denominator": 9,
        "display_lexical_filter_applied": False,
        "character_length_filter_available": False,
        "character_length_filter_policy": None,
    }
    for candidate in realized["top_tokens"]:
        assert candidate["rank"] == 1
        assert candidate["rank_denominator"] == 9
        assert candidate["rank_space"] == "full_model_vocabulary"
        assert candidate["full_vocabulary_rank"] == 1
        assert candidate["full_vocabulary_denominator"] == 9
        assert candidate["score_kind"] == "raw_teacher_forced_probability"
        assert "log_probability" in candidate
        assert candidate["vocabulary_filter"][
            "display_lexical_filter_applied"
        ] is False


def test_encoder_length_buckets_and_waveform_windows_apply_to_every_layer():
    tokenizer = _DisplayTokenizer()
    vocab_size = len(tokenizer._tokens)
    residual = torch.arange(vocab_size, dtype=torch.float32)

    class _Model:
        model_id = "encoder-filter-test"
        fingerprint = "encoder-filter-test-v1"

        def __init__(self):
            self.tokenizer = tokenizer
            self.vocab_size = vocab_size

        @staticmethod
        def unembed(hidden):
            return hidden

        @staticmethod
        def capture(inputs, *, encoder_layers, decoder_layers):
            assert encoder_layers == [0, 2]
            assert decoder_layers == []
            activations = {
                layer: residual.repeat(
                    inputs.encoder_position_mask.shape[1], 1
                )[None]
                for layer in encoder_layers
            }
            actual_logits = torch.zeros(
                1, inputs.decoder_input_ids.shape[1], vocab_size
            )
            return activations, {}, actual_logits

    encoder_lens = CrossJacobianLens(
        {layer: torch.eye(vocab_size) for layer in (0, 2)},
        n_examples=1,
        source_dim=vocab_size,
        target_dim=vocab_size,
        source_stream="encoder",
        target_stream="decoder",
        source_means={
            layer: torch.zeros(vocab_size) for layer in (0, 2)
        },
        target_mean=torch.zeros(vocab_size),
        metadata={"estimator_name": "test"},
    )
    lens = WhisperJacobianLens(
        encoder=encoder_lens,
        model_metadata={"model_fingerprint": _Model.fingerprint},
        estimator_metadata={},
    )
    phone_values = torch.zeros(2, vocab_size)
    phone_values[0, 8] = 1
    phone_values[1, 7] = 1
    phone_prototypes = PhoneSignaturePrototypes(
        {layer: phone_values for layer in (0, 2)},
        labels=["AA", "B"],
        signature_top_k=2,
        vocab_size=vocab_size,
        model_fingerprint=_Model.fingerprint,
        encoder_lens_fingerprint_value=encoder_lens_fingerprint(encoder_lens),
    )
    inputs = WhisperLensInputs(
        input_features=torch.zeros(1, 2, 2),
        decoder_input_ids=torch.tensor([[0, 1]]),
        decoder_target_ids=torch.tensor([[1, 2]]),
        encoder_position_mask=torch.tensor([[True, True]]),
        decoder_position_mask=torch.tensor([[True, True]]),
        duration_seconds=0.04,
    )

    payload = analyze_whisper_run(
        _Model(),
        lens,
        inputs,
        np.zeros(640, dtype=np.float32),
        top_k=2,
        phone_signature_prototypes=phone_prototypes,
    )

    assert payload["encoder"]["layers"] == [0, 2]
    assert payload["encoder"]["time_bins"] == [
        {"start_seconds": 0.0, "end_seconds": 0.04}
    ]
    for layer_cells in payload["encoder"]["cells"]:
        assert len(layer_cells) == 1
        cell = layer_cells[0]
        assert set(cell["top_tokens_by_length"]) == {"1", "2", "3", "5"}
        assert cell["time_window"] == {
            "start_seconds": 0.0,
            "end_seconds": 0.04,
            "timing_source": "encoder_pooling_window",
        }
        assert cell["candidate_space"][
            "character_length_filter_available"
        ] is True
        assert cell["top_tokens"][0]["score_kind"] == (
            "target_mean_relative_logit_delta"
        )
        assert cell["phone_signature_usable"] is True
        assert [candidate["phone"] for candidate in cell["phone_signatures"]] == [
            "AA",
            "B",
        ]
        assert all(
            candidate["score_kind"] == "phone_prototype_cosine_similarity"
            for candidate in cell["phone_signatures"]
        )
        assert "realized_token" not in cell
    assert payload["metadata"]["phone_signature"]["available"] is True
    assert payload["metadata"]["phone_signature"]["phone_inventory_size"] == 2
    assert payload["metadata"]["phone_signature"][
        "effective_display_window_seconds"
    ] == 0.2


def test_encoder_realized_rank_uses_maximum_overlap_output_token():
    tokenizer = _DisplayTokenizer()
    vocab_size = len(tokenizer._tokens)
    residual = torch.arange(vocab_size, dtype=torch.float32)

    class _Model:
        model_id = "encoder-realized-rank-test"
        fingerprint = "encoder-realized-rank-test-v1"

        def __init__(self):
            self.tokenizer = tokenizer
            self.vocab_size = vocab_size

        @staticmethod
        def unembed(hidden):
            return hidden

        @staticmethod
        def capture(inputs, *, encoder_layers, decoder_layers):
            assert encoder_layers == [0]
            assert decoder_layers == []
            activations = {
                0: residual.repeat(inputs.encoder_position_mask.shape[1], 1)[None]
            }
            actual_logits = torch.zeros(
                1, inputs.decoder_input_ids.shape[1], vocab_size
            )
            return activations, {}, actual_logits

    encoder_lens = CrossJacobianLens(
        {0: torch.eye(vocab_size)},
        n_examples=1,
        source_dim=vocab_size,
        target_dim=vocab_size,
        source_stream="encoder",
        target_stream="decoder",
        source_means={0: torch.zeros(vocab_size)},
        target_mean=torch.zeros(vocab_size),
        metadata={"estimator_name": "test"},
    )
    lens = WhisperJacobianLens(
        encoder=encoder_lens,
        model_metadata={"model_fingerprint": _Model.fingerprint},
        estimator_metadata={},
    )
    inputs = WhisperLensInputs(
        input_features=torch.zeros(1, 2, 8),
        decoder_input_ids=torch.tensor([[0, 1]]),
        decoder_target_ids=torch.tensor([[1, 2]]),
        encoder_position_mask=torch.tensor([[True, True, True, True]]),
        decoder_position_mask=torch.tensor([[True, True]]),
        duration_seconds=0.08,
    )

    payload = analyze_whisper_run(
        _Model(),
        lens,
        inputs,
        np.zeros(1280, dtype=np.float32),
        token_timestamps=torch.tensor([[0.0, 0.0, 0.04]]),
        top_k=2,
        time_bin_seconds=0.02,
        time_bin_overlap_seconds=0.0,
    )

    assert payload["encoder"]["realized_token_alignment"] == {
        "method": "maximum_token_interval_overlap",
        "tie_break": "closest_interval_midpoint_then_lower_token_position",
        "timing_source": "whisper_cross_attention_dtw",
        "timing_quality": "model_derived",
        "interpretation": "approximate_non_causal_synchronization",
    }
    cells = payload["encoder"]["cells"][0]
    assert [cell["realized_token_position"] for cell in cells] == [0, 0, 1, 1]
    assert [cell["realized_token"]["id"] for cell in cells] == [1, 1, 2, 2]
    assert [cell["realized_token_alignment"]["match"] for cell in cells] == [
        "overlapping",
        "overlapping",
        "overlapping",
        "overlapping",
    ]
    assert [
        cell["realized_token_alignment"]["overlap_fraction_of_window"]
        for cell in cells
    ] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    for cell in cells:
        realized = cell["realized_token"]
        assert realized["score_kind"] == "target_mean_relative_logit_delta"
        assert realized["rank_space"] == "lexical_display_vocabulary"
        filtered_rank = cell["realized_rank_by_max_length"][
            str(len(realized["text"].strip()))
        ]
        assert isinstance(filtered_rank, int)


def test_encoder_realized_alignment_prefers_overlap_then_midpoint_then_position():
    tokens = [
        {"start_seconds": 4.9, "end_seconds": 5.1},
        {"start_seconds": 0.0, "end_seconds": 4.95},
        {"start_seconds": 6.0, "end_seconds": 7.0},
    ]
    maximum_overlap = _encoder_realized_token_alignment(
        [{"start_seconds": 0.0, "end_seconds": 10.0}], tokens
    )[0]
    assert maximum_overlap["token_position"] == 1
    assert maximum_overlap["overlap_seconds"] == pytest.approx(4.95)

    closest_midpoint = _encoder_realized_token_alignment(
        [{"start_seconds": 0.0, "end_seconds": 4.0}],
        [
            {"start_seconds": 0.0, "end_seconds": 1.0},
            {"start_seconds": 1.5, "end_seconds": 2.5},
        ],
    )[0]
    assert closest_midpoint["token_position"] == 1

    lower_position = _encoder_realized_token_alignment(
        [{"start_seconds": 0.0, "end_seconds": 2.0}],
        [
            {"start_seconds": 0.0, "end_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0},
        ],
    )[0]
    assert lower_position["token_position"] == 0

    nearest = _encoder_realized_token_alignment(
        [{"start_seconds": 3.8, "end_seconds": 4.0}],
        [
            {"start_seconds": 0.0, "end_seconds": 1.0},
            {"start_seconds": 5.0, "end_seconds": 6.0},
        ],
    )[0]
    assert nearest["token_position"] == 1
    assert nearest["match"] == "nearest"
    assert nearest["overlap_seconds"] == 0.0


def _analysis_inputs() -> WhisperLensInputs:
    return WhisperLensInputs(
        input_features=torch.zeros(1, 2, 4),
        decoder_input_ids=torch.tensor([[1, 2]]),
        decoder_target_ids=torch.tensor([[2, 0]]),
        encoder_position_mask=torch.tensor([[True, False]]),
        decoder_position_mask=torch.tensor([[True, False]]),
        duration_seconds=2.0,
    )


def test_token_timestamps_are_clamped_and_never_fabricated():
    model = SimpleNamespace(tokenizer=_Tokenizer())
    logits = torch.zeros(1, 2, 4)
    transcript, _ = _transcript_and_confidence(
        model,
        _analysis_inputs(),
        logits,
        token_timestamps=torch.tensor([[0.0, 10.0, 10.0]]),
        duration_seconds=2.0,
        top_k=2,
    )
    assert transcript["tokens"][0]["start_seconds"] == 2.0
    assert transcript["tokens"][0]["end_seconds"] == 2.0
    assert transcript["timing_source"] == "whisper_cross_attention_dtw"

    no_timing, _ = _transcript_and_confidence(
        model,
        _analysis_inputs(),
        logits,
        token_timestamps=None,
        duration_seconds=2.0,
        top_k=2,
    )
    assert no_timing["tokens"][0]["start_seconds"] is None
    assert no_timing["tokens"][0]["end_seconds"] is None
    assert no_timing["timing_source"] == "unavailable"


def test_output_head_reports_full_vocabulary_competition_ranks_with_ties():
    model = SimpleNamespace(tokenizer=_Tokenizer())
    logits = torch.tensor(
        [[[4.0, 3.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )

    transcript, _ = _transcript_and_confidence(
        model,
        _analysis_inputs(),
        logits,
        token_timestamps=None,
        duration_seconds=2.0,
        top_k=3,
    )

    realized = transcript["tokens"][0]
    assert realized["id"] == 2
    assert realized["rank"] == 2
    assert realized["rank_denominator"] == 4
    assert realized["rank_space"] == "full_model_vocabulary"
    assert realized["rank_tie_policy"] == "1_plus_count_strictly_greater"
    ranks_by_id = {
        candidate["id"]: candidate["rank"]
        for candidate in realized["top_tokens"]
    }
    assert ranks_by_id == {0: 1, 1: 2, 2: 2}
    for candidate in realized["top_tokens"]:
        assert candidate["rank_denominator"] == 4
        assert candidate["full_vocabulary_rank"] == candidate["rank"]
        assert candidate["full_vocabulary_denominator"] == 4
        assert candidate["rank_space"] == "full_model_vocabulary"
        assert candidate["vocabulary_filter"] == {
            "display_lexical_filter_applied": False,
            "character_length_filter_applied": False,
            "decoded_character_length": len(candidate["text"].strip()),
            "character_length_constraint": None,
        }
