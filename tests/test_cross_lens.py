from __future__ import annotations

import pytest
import torch

from jlens.cross_lens import CrossJacobianLens
from jlens.whisper_lens import lens_topk, lens_topk_grouped


def make_lens(value: float = 1.0, *, n_examples: int = 2) -> CrossJacobianLens:
    return CrossJacobianLens(
        {0: torch.full((3, 2), value), 2: torch.eye(3, 2)},
        n_examples=n_examples,
        source_dim=2,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        metadata={"model": "tiny"},
    )


def make_centered_lens(
    value: float = 1.0,
    *,
    n_examples: int = 2,
    source_mean_value: float = 0.0,
    target_mean_value: float = 0.0,
) -> CrossJacobianLens:
    return CrossJacobianLens(
        {0: torch.full((3, 2), value), 2: torch.eye(3, 2)},
        n_examples=n_examples,
        source_dim=2,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        source_means={
            0: torch.full((2,), source_mean_value),
            2: torch.full((2,), source_mean_value),
        },
        target_mean=torch.full((3,), target_mean_value),
        metadata={"model": "tiny"},
    )


def test_rectangular_transport_orientation():
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lens = CrossJacobianLens(
        {1: matrix},
        n_examples=1,
        source_dim=2,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
    )
    source = torch.tensor([[7.0, 11.0]])
    torch.testing.assert_close(lens.transport(source, 1), source @ matrix.T)


def test_centered_transport_is_affine_and_anchors_fitted_means():
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    source_mean = torch.tensor([2.0, -1.0])
    target_mean = torch.tensor([7.0, 11.0, 13.0])
    lens = CrossJacobianLens(
        {1: matrix},
        n_examples=1,
        source_dim=2,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        source_means={1: source_mean},
        target_mean=target_mean,
    )
    source = torch.tensor([[7.0, 11.0]])
    expected = (source - source_mean) @ matrix.T + target_mean
    torch.testing.assert_close(lens.transport(source, 1), expected)
    torch.testing.assert_close(lens.transport(source_mean, 1), target_mean)


def test_vocabulary_directions_are_unembedding_times_jacobian():
    lens = make_lens()
    unembedding = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.testing.assert_close(
        lens.vocabulary_directions(unembedding, 0),
        unembedding @ lens.jacobians[0],
    )


def test_save_load_and_weighted_merge(tmp_path):
    first = make_lens(1.0, n_examples=1)
    second = make_lens(3.0, n_examples=3)
    path = tmp_path / "cross.pt"
    first.save(str(path), dtype=torch.float32)
    loaded = CrossJacobianLens.load(str(path))
    torch.testing.assert_close(loaded.jacobians[0], first.jacobians[0])
    assert loaded.metadata == first.metadata

    merged = CrossJacobianLens.merge([first, second])
    torch.testing.assert_close(merged.jacobians[0], torch.full((3, 2), 2.5))
    assert merged.n_examples == 4


def test_centering_roundtrip_and_weighted_merge(tmp_path):
    first = make_centered_lens(
        1.0,
        n_examples=1,
        source_mean_value=1.0,
        target_mean_value=10.0,
    )
    second = make_centered_lens(
        3.0,
        n_examples=3,
        source_mean_value=5.0,
        target_mean_value=2.0,
    )
    path = tmp_path / "centered.pt"
    first.save(str(path), dtype=torch.float32)
    loaded = CrossJacobianLens.load(str(path))
    assert loaded.source_means is not None
    assert loaded.target_mean is not None
    torch.testing.assert_close(loaded.source_means[0], first.source_means[0])
    torch.testing.assert_close(loaded.target_mean, first.target_mean)

    merged = CrossJacobianLens.merge([first, second])
    assert merged.source_means is not None
    assert merged.target_mean is not None
    torch.testing.assert_close(merged.source_means[0], torch.full((2,), 4.0))
    torch.testing.assert_close(merged.target_mean, torch.full((3,), 4.0))


def test_merge_keeps_stable_metadata_and_aggregates_shard_provenance():
    first = make_lens(n_examples=2)
    first.metadata.update(
        corpus_fingerprint="corpus-a",
        manifest_name="a.jsonl",
        requested_examples=2,
    )
    second = make_lens(n_examples=3)
    second.metadata.update(
        corpus_fingerprint="corpus-b",
        manifest_name="b.jsonl",
        requested_examples=3,
    )
    merged = CrossJacobianLens.merge([first, second])
    assert merged.metadata["model"] == "tiny"
    assert "corpus_fingerprint" not in merged.metadata
    assert merged.metadata["shard_provenance"] == [
        {
            "corpus_fingerprint": "corpus-a",
            "manifest_name": "a.jsonl",
            "requested_examples": 2,
            "n_examples": 2,
        },
        {
            "corpus_fingerprint": "corpus-b",
            "manifest_name": "b.jsonl",
            "requested_examples": 3,
            "n_examples": 3,
        },
    ]

    third = make_lens(n_examples=1)
    third.metadata.update(corpus_fingerprint="corpus-c")
    remerged = CrossJacobianLens.merge([merged, third])
    assert len(remerged.metadata["shard_provenance"]) == 3


def test_v1_state_is_always_loaded_as_legacy_uncentered():
    state = make_centered_lens().state_dict(dtype=torch.float32)
    state["format_version"] = 1
    loaded = CrossJacobianLens.from_state_dict(state)
    assert loaded.source_means is None
    assert loaded.target_mean is None


def test_shape_and_metadata_mismatches_are_rejected():
    with pytest.raises(ValueError, match="expected"):
        CrossJacobianLens(
            {0: torch.zeros(2, 2)},
            n_examples=1,
            source_dim=2,
            target_dim=3,
            source_stream="encoder",
            target_stream="decoder",
        )
    other = make_lens()
    incompatible = CrossJacobianLens(
        other.jacobians,
        n_examples=1,
        source_dim=2,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        metadata={"model": "different"},
    )
    with pytest.raises(ValueError, match="disagree"):
        CrossJacobianLens.merge([other, incompatible])
    with pytest.raises(ValueError, match="both be provided"):
        CrossJacobianLens(
            other.jacobians,
            n_examples=1,
            source_dim=2,
            target_dim=3,
            source_stream="encoder",
            target_stream="decoder",
            source_means={0: torch.zeros(2), 2: torch.zeros(2)},
        )
    with pytest.raises(ValueError, match="centered and uncentered"):
        CrossJacobianLens.merge([other, make_centered_lens()])


def test_topk_can_remove_centered_target_language_prior():
    lens = CrossJacobianLens(
        {0: torch.eye(3)},
        n_examples=1,
        source_dim=3,
        target_dim=3,
        source_stream="encoder",
        target_stream="decoder",
        source_means={0: torch.zeros(3)},
        target_mean=torch.tensor([10.0, 0.0, 0.0]),
    )

    class _IdentityReadout:
        vocab_size = 3

        @staticmethod
        def unembed(residual):
            return residual

    residual = torch.tensor([[1.0, 2.0, 3.0]])
    absolute = lens_topk(_IdentityReadout(), lens, residual, layer=0, top_k=1)
    delta = lens_topk(
        _IdentityReadout(),
        lens,
        residual,
        layer=0,
        top_k=1,
        subtract_target_baseline=True,
    )
    assert int(absolute.token_ids[0, 0]) == 0
    assert int(delta.token_ids[0, 0]) == 2


def test_topk_masks_the_full_vocabulary_before_ranking():
    lens = CrossJacobianLens(
        {0: torch.eye(4)},
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream="encoder",
        target_stream="decoder",
    )

    class _IdentityReadout:
        vocab_size = 4

        @staticmethod
        def unembed(residual):
            return residual

    # Tokens 0 and 1 dominate the unmasked logits. Masking them to -inf must
    # promote even the negative-scoring eligible token into the returned top-k.
    residual = torch.tensor([[12.0, 11.0, -4.0, 0.5]])
    result = lens_topk(
        _IdentityReadout(),
        lens,
        residual,
        layer=0,
        top_k=2,
        token_mask=torch.tensor([False, False, True, True]),
    )
    assert result.token_ids.tolist() == [[3, 2]]
    torch.testing.assert_close(result.scores, torch.tensor([[0.5, -4.0]]))


def test_grouped_topk_preserves_exact_length_bucket_candidates():
    lens = CrossJacobianLens(
        {0: torch.eye(5)},
        n_examples=1,
        source_dim=5,
        target_dim=5,
        source_stream="encoder",
        target_stream="decoder",
    )

    class _IdentityReadout:
        vocab_size = 5

        @staticmethod
        def unembed(residual):
            return residual

    result = lens_topk_grouped(
        _IdentityReadout(),
        lens,
        torch.tensor([[10.0, 9.0, 8.0, 0.5, -4.0]]),
        layer=0,
        top_k=2,
        token_mask=torch.tensor([True, True, True, True, True]),
        token_groups={
            1: torch.tensor([False, False, False, True, True]),
            2: torch.tensor([False, True, True, False, False]),
        },
    )
    assert result.overall.token_ids.tolist() == [[0, 1]]
    assert result.groups[1].token_ids.tolist() == [[3, 4]]
    assert result.groups[2].token_ids.tolist() == [[1, 2]]
    torch.testing.assert_close(
        result.groups[1].scores, torch.tensor([[0.5, -4.0]])
    )


def test_topk_reports_exact_active_display_and_full_competition_ranks():
    lens = CrossJacobianLens(
        {0: torch.eye(4)},
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream="encoder",
        target_stream="decoder",
    )

    class _IdentityReadout:
        vocab_size = 4

        @staticmethod
        def unembed(residual):
            return residual

    result = lens_topk(
        _IdentityReadout(),
        lens,
        torch.tensor([[10.0, 9.0, 9.0, 8.0]]),
        layer=0,
        top_k=3,
        token_mask=torch.tensor([False, True, True, True]),
    )
    by_id = {
        int(token_id): (
            int(rank),
            int(display_rank),
            int(full_rank),
        )
        for token_id, rank, display_rank, full_rank in zip(
            result.token_ids[0],
            result.ranks[0],
            result.display_vocabulary_ranks[0],
            result.full_vocabulary_ranks[0],
            strict=True,
        )
    }
    assert by_id == {1: (1, 1, 2), 2: (1, 1, 2), 3: (3, 3, 4)}
    assert result.rank_denominator == 3
    assert result.display_vocabulary_denominator == 3
    assert result.full_vocabulary_denominator == 4


def test_topk_reports_exact_selected_readouts_even_outside_topk_and_mask():
    lens = CrossJacobianLens(
        {0: torch.eye(5)},
        n_examples=1,
        source_dim=5,
        target_dim=5,
        source_stream="encoder",
        target_stream="decoder",
    )

    class _IdentityReadout:
        vocab_size = 5

        @staticmethod
        def unembed(residual):
            return residual

    result = lens_topk(
        _IdentityReadout(),
        lens,
        torch.tensor(
            [
                [1.0, 3.0, 1.0, 2.0, 0.0],
                [2.0, 2.0, 1.0, 0.0, 0.0],
            ]
        ),
        layer=0,
        top_k=1,
        token_mask=torch.tensor([True, True, True, False, True]),
        selected_token_ids=torch.tensor([0, 3]),
        position_chunk_size=1,
    )

    assert result.token_ids[0].tolist() == [1]
    assert all(3 not in row for row in result.token_ids.tolist())
    selected = result.selected_readouts
    assert selected is not None
    assert selected.token_ids.tolist() == [0, 3]
    torch.testing.assert_close(selected.scores, torch.tensor([1.0, 0.0]))
    # Token 0 ties token 2, so both receive lexical-display rank 2. Token 3 is
    # outside the display mask, and zero is the documented internal sentinel.
    assert selected.display_vocabulary_ranks.tolist() == [2, 0]
    assert selected.display_vocabulary_eligible.tolist() == [True, False]
    assert selected.display_vocabulary_denominator == 4
    assert selected.full_vocabulary_ranks.tolist() == [3, 4]
    assert selected.full_vocabulary_denominator == 5


def test_grouped_topk_reports_bucket_display_and_full_denominators():
    lens = CrossJacobianLens(
        {0: torch.eye(4)},
        n_examples=1,
        source_dim=4,
        target_dim=4,
        source_stream="encoder",
        target_stream="decoder",
    )

    class _IdentityReadout:
        vocab_size = 4

        @staticmethod
        def unembed(residual):
            return residual

    result = lens_topk_grouped(
        _IdentityReadout(),
        lens,
        torch.tensor([[10.0, 9.0, 9.0, 8.0]]),
        layer=0,
        top_k=2,
        token_mask=torch.tensor([False, True, True, True]),
        token_groups={
            1: torch.tensor([False, False, True, True]),
        },
    )
    bucket = result.groups[1]
    assert bucket.token_ids.tolist() == [[2, 3]]
    assert bucket.ranks.tolist() == [[1, 2]]
    assert bucket.display_vocabulary_ranks.tolist() == [[1, 3]]
    assert bucket.full_vocabulary_ranks.tolist() == [[2, 4]]
    assert bucket.rank_denominator == 2
    assert bucket.display_vocabulary_denominator == 3
    assert bucket.full_vocabulary_denominator == 4
