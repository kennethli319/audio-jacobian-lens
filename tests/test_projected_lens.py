from __future__ import annotations

import pytest
import torch

from jlens.projected_lens import ProjectedCrossJacobianLens


def source_factors_from_output_probes(
    jacobian: torch.Tensor, target_factors: torch.Tensor
) -> torch.Tensor:
    return target_factors @ jacobian


def make_projected_lens(
    response_value: float = 1.0,
    *,
    n_examples: int = 1,
    centered: bool = False,
    projection_method: str = "output_probe_vjp",
    metadata: dict | None = None,
) -> ProjectedCrossJacobianLens:
    target_factors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]
    )
    source_means = {0: torch.tensor([2.0, -1.0])} if centered else None
    target_mean = torch.tensor([3.0, 4.0, 5.0]) if centered else None
    return ProjectedCrossJacobianLens(
        target_factors,
        {0: torch.full((3, 2), response_value)},
        n_examples=n_examples,
        source_dim=2,
        target_dim=3,
        source_stream="speech_encoder",
        target_stream="text_decoder",
        projection_method=projection_method,
        source_means=source_means,
        target_mean=target_mean,
        metadata={"backend": "tiny", **(metadata or {})},
    )


def test_scaled_identity_target_factors_exactly_reconstruct_dense_transport():
    jacobian = torch.tensor(
        [[1.0, 2.0, -1.0], [3.0, -2.0, 0.5], [0.0, 4.0, 2.0], [-1.0, 1.0, 3.0]]
    )
    target_dim, source_dim = jacobian.shape
    target_factors = torch.eye(target_dim) * target_dim**0.5
    source_factors = source_factors_from_output_probes(jacobian, target_factors)
    lens = ProjectedCrossJacobianLens(
        target_factors,
        {2: source_factors},
        n_examples=1,
        source_dim=source_dim,
        target_dim=target_dim,
        source_stream="encoder",
        target_stream="decoder",
        projection_method="output_probe_vjp",
    )

    residual = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 2.0, 1.0]])
    torch.testing.assert_close(lens.transport(residual, 2), residual @ jacobian.T)

    unembedding = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.testing.assert_close(
        lens.vocabulary_directions(unembedding, 2), unembedding @ jacobian
    )


def test_random_gaussian_output_probes_approximate_dense_transport():
    generator = torch.Generator().manual_seed(17)
    source_dim = 7
    target_dim = 5
    projection_dim = 8192
    jacobian = torch.randn(target_dim, source_dim, generator=generator)
    target_factors = torch.randn(
        projection_dim, target_dim, generator=generator
    )
    source_factors = source_factors_from_output_probes(jacobian, target_factors)
    lens = ProjectedCrossJacobianLens(
        target_factors,
        {0: source_factors},
        n_examples=1,
        source_dim=source_dim,
        target_dim=target_dim,
        source_stream="encoder",
        target_stream="decoder",
        projection_method="output_probe_vjp",
    )
    residual = torch.randn(32, source_dim, generator=generator)
    expected = residual @ jacobian.T
    actual = lens.transport(residual, 0)
    relative_error = (actual - expected).norm() / expected.norm()
    assert float(relative_error) < 0.05


def test_source_probe_jvp_factors_use_the_same_neutral_representation():
    jacobian = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]])
    source_dim = jacobian.shape[1]
    source_factors = torch.eye(source_dim) * source_dim**0.5
    target_factors = source_factors @ jacobian.T
    lens = ProjectedCrossJacobianLens(
        target_factors,
        {1: source_factors},
        n_examples=1,
        source_dim=source_dim,
        target_dim=jacobian.shape[0],
        source_stream="encoder",
        target_stream="decoder",
        projection_method="source_probe_jvp",
    )
    residual = torch.tensor([[3.0, -2.0]])
    torch.testing.assert_close(lens.transport(residual, 1), residual @ jacobian.T)


def test_affine_transport_anchors_means_and_roundtrips_safely(tmp_path):
    lens = make_projected_lens(centered=True, metadata={"nested": {"z": 2, "a": 1}})
    assert lens.source_means is not None
    assert lens.target_mean is not None
    torch.testing.assert_close(
        lens.transport(lens.source_means[0], 0), lens.target_mean
    )

    state = lens.state_dict(dtype=torch.float32)
    assert list(state["source_factors"]) == [0]
    assert state["projection_method"] == "output_probe_vjp"
    assert state["metadata"]["projection_method"] == "output_probe_vjp"
    assert list(state["metadata"]["nested"]) == ["a", "z"]
    path = tmp_path / "projected.pt"
    lens.save(path, dtype=torch.float32)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["format"] == "projected-cross-jacobian-lens"
    loaded = ProjectedCrossJacobianLens.load(path)
    torch.testing.assert_close(loaded.target_factors, lens.target_factors)
    torch.testing.assert_close(loaded.source_factors[0], lens.source_factors[0])
    torch.testing.assert_close(loaded.source_means[0], lens.source_means[0])
    torch.testing.assert_close(loaded.target_mean, lens.target_mean)
    assert loaded.metadata == lens.metadata


def test_weighted_merge_requires_compatible_factors_method_and_metadata():
    first = make_projected_lens(1.0, n_examples=1)
    second = make_projected_lens(3.0, n_examples=3)
    merged = ProjectedCrossJacobianLens.merge([first, second])
    assert merged.n_examples == 4
    torch.testing.assert_close(merged.source_factors[0], torch.full((3, 2), 2.5))

    different_target_factors = make_projected_lens(3.0)
    different_target_factors.target_factors[0, 0] += 1.0
    with pytest.raises(ValueError, match="different target factors"):
        ProjectedCrossJacobianLens.merge([first, different_target_factors])

    different_metadata = make_projected_lens(3.0, metadata={"revision": "other"})
    with pytest.raises(ValueError, match="metadata"):
        ProjectedCrossJacobianLens.merge([first, different_metadata])

    different_method = make_projected_lens(
        3.0, projection_method="external_low_rank_factorization"
    )
    with pytest.raises(ValueError, match="projection method"):
        ProjectedCrossJacobianLens.merge([first, different_method])


def test_strict_shape_and_state_validation():
    target_factors = torch.zeros(4, 2)
    with pytest.raises(ValueError, match=r"source_factors\[0\].*expected"):
        ProjectedCrossJacobianLens(
            target_factors,
            {0: torch.zeros(3, 3)},
            n_examples=1,
            source_dim=3,
            target_dim=2,
            source_stream="encoder",
            target_stream="decoder",
            projection_method="output_probe_vjp",
        )
    with pytest.raises(ValueError, match="both be provided"):
        ProjectedCrossJacobianLens(
            target_factors,
            {0: torch.zeros(4, 3)},
            n_examples=1,
            source_dim=3,
            target_dim=2,
            source_stream="encoder",
            target_stream="decoder",
            projection_method="output_probe_vjp",
            source_means={0: torch.zeros(3)},
        )

    state = make_projected_lens().state_dict(dtype=torch.float32)
    state["source_layers"] = [9]
    with pytest.raises(ValueError, match="source_layers"):
        ProjectedCrossJacobianLens.from_state_dict(state)
