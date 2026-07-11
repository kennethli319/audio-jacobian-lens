from __future__ import annotations

import pytest
import torch

from jlens.hooks import ActivationRecorder
from jlens.whisper_fitting import (
    fit_whisper,
    jacobians_for_whisper_example,
)
from jlens.whisper_lens import WhisperJacobianLens

from .tiny_whisper import TinyWhisperLensModel, tiny_inputs


def explicit_rows(model, inputs, *, encoder_layers, decoder_layers, target_layer):
    encoder_recorder = ActivationRecorder(
        model.encoder_layers,
        at=encoder_layers,
        start_graph_at=min(encoder_layers) if encoder_layers else None,
    )
    decoder_recorder = ActivationRecorder(
        model.decoder_layers,
        at=sorted({*decoder_layers, target_layer}),
        start_graph_at=(
            min(decoder_layers) if decoder_layers and not encoder_layers else None
        ),
    )
    with encoder_recorder, decoder_recorder, torch.enable_grad():
        model.forward(inputs)
        target = decoder_recorder.activations[target_layer]
        named = [
            ("encoder", layer, encoder_recorder.activations[layer])
            for layer in encoder_layers
        ] + [
            ("decoder", layer, decoder_recorder.activations[layer])
            for layer in decoder_layers
        ]
        matrices = {
            (stream, layer): torch.zeros(
                model.decoder_dim,
                model.encoder_dim if stream == "encoder" else model.decoder_dim,
            )
            for stream, layer, _ in named
        }
        target_positions = inputs.decoder_position_mask[0].nonzero(as_tuple=True)[0]
        for output_dim in range(model.decoder_dim):
            scalar = target[0, target_positions, output_dim].sum()
            gradients = torch.autograd.grad(
                scalar,
                [tensor for _, _, tensor in named],
                retain_graph=(output_dim < model.decoder_dim - 1),
            )
            for (stream, layer, _), gradient in zip(named, gradients, strict=True):
                mask = (
                    inputs.encoder_position_mask[0]
                    if stream == "encoder"
                    else inputs.decoder_position_mask[0]
                )
                matrices[(stream, layer)][output_dim] = gradient[0, mask].mean(0)
    return matrices


def test_dimension_batched_estimator_matches_explicit_vjps():
    model = TinyWhisperLensModel()
    inputs = tiny_inputs(model)
    result = jacobians_for_whisper_example(
        model,
        inputs,
        encoder_source_layers=[0, 2],
        decoder_source_layers=[0, 1],
        target_decoder_layer=2,
        dim_batch=3,
    )
    expected = explicit_rows(
        model,
        inputs,
        encoder_layers=[0, 2],
        decoder_layers=[0, 1],
        target_layer=2,
    )
    for layer, matrix in result.encoder.items():
        torch.testing.assert_close(matrix, expected[("encoder", layer)])
    for layer, matrix in result.decoder.items():
        torch.testing.assert_close(matrix, expected[("decoder", layer)])


def test_mean_target_reduction_scales_sum_estimator():
    model = TinyWhisperLensModel()
    inputs = tiny_inputs(model)
    summed = jacobians_for_whisper_example(
        model,
        inputs,
        encoder_source_layers=[1],
        decoder_source_layers=[],
        target_decoder_layer=2,
        dim_batch=2,
        target_reduction="sum",
    )
    meaned = jacobians_for_whisper_example(
        model,
        inputs,
        encoder_source_layers=[1],
        decoder_source_layers=[],
        target_decoder_layer=2,
        dim_batch=2,
        target_reduction="mean",
    )
    count = int(inputs.decoder_position_mask.sum())
    torch.testing.assert_close(meaned.encoder[1], summed.encoder[1] / count)
    torch.testing.assert_close(
        meaned.target_activation_mean,
        summed.target_activation_mean / count,
    )


def test_decoder_causality_blocks_later_source_positions():
    model = TinyWhisperLensModel()
    inputs = tiny_inputs(model)
    selected_target = 2
    decoder_mask = torch.zeros_like(inputs.decoder_position_mask)
    decoder_mask[:, selected_target] = True
    inputs = inputs.__class__(
        input_features=inputs.input_features,
        decoder_input_ids=inputs.decoder_input_ids,
        decoder_target_ids=inputs.decoder_target_ids,
        encoder_position_mask=inputs.encoder_position_mask,
        decoder_position_mask=decoder_mask,
    )
    recorder = ActivationRecorder(
        model.decoder_layers,
        at=[0, 2],
        start_graph_at=0,
    )
    with recorder, torch.enable_grad():
        model.forward(inputs)
        target = recorder.activations[2][0, selected_target].sum()
        gradient = torch.autograd.grad(target, recorder.activations[0])[0][0]
    assert torch.count_nonzero(gradient[selected_target + 1 :]) == 0


def test_fit_checkpoint_bundle_roundtrip_and_resume(tmp_path):
    model = TinyWhisperLensModel()
    examples = [tiny_inputs(model), tiny_inputs(model)]
    checkpoint = tmp_path / "fit.pt"
    bundle = fit_whisper(
        model,
        examples,
        encoder_source_layers=[0, 2],
        decoder_source_layers=[0, 1],
        target_decoder_layer=2,
        dim_batch=3,
        checkpoint_path=str(checkpoint),
    )
    assert bundle.encoder is not None
    assert bundle.decoder is not None
    assert bundle.encoder.jacobians[0].shape == (
        model.decoder_dim,
        model.encoder_dim,
    )
    assert bundle.decoder.jacobians[0].shape == (
        model.decoder_dim,
        model.decoder_dim,
    )
    assert bundle.encoder.n_examples == 2
    assert bundle.encoder.source_means is not None
    assert bundle.encoder.target_mean is not None
    assert bundle.decoder.source_means is None

    encoder_recorder = ActivationRecorder(model.encoder_layers, at=[0, 2])
    decoder_recorder = ActivationRecorder(model.decoder_layers, at=[2])
    with encoder_recorder, decoder_recorder, torch.no_grad():
        model.forward(examples[0])
    encoder_positions = examples[0].encoder_position_mask[0]
    decoder_positions = examples[0].decoder_position_mask[0]
    for layer in (0, 2):
        expected_source_mean = (
            encoder_recorder.activations[layer][0, encoder_positions]
            .float()
            .mean(dim=0)
        )
        torch.testing.assert_close(
            bundle.encoder.source_means[layer], expected_source_mean
        )
        # The fitted source origin must land exactly on the target origin.
        torch.testing.assert_close(
            bundle.encoder.transport(bundle.encoder.source_means[layer], layer),
            bundle.encoder.target_mean,
        )
    expected_target_mean = (
        decoder_recorder.activations[2][0, decoder_positions].float().sum(dim=0)
    )
    torch.testing.assert_close(bundle.encoder.target_mean, expected_target_mean)

    path = tmp_path / "lens.pt"
    bundle.save(str(path), dtype=torch.float32)
    loaded = WhisperJacobianLens.load(str(path))
    loaded.validate_model(model)
    torch.testing.assert_close(loaded.encoder.jacobians[2], bundle.encoder.jacobians[2])
    assert loaded.encoder.source_means is not None
    assert loaded.encoder.target_mean is not None
    torch.testing.assert_close(
        loaded.encoder.source_means[2], bundle.encoder.source_means[2]
    )
    torch.testing.assert_close(loaded.encoder.target_mean, bundle.encoder.target_mean)

    legacy_state = bundle.state_dict(dtype=torch.float32)
    legacy_state["format_version"] = 1
    legacy_state["encoder"]["format_version"] = 1
    legacy_state["decoder"]["format_version"] = 1
    legacy = WhisperJacobianLens.from_state_dict(legacy_state)
    assert legacy.encoder is not None
    assert legacy.encoder.source_means is None
    assert legacy.encoder.target_mean is None

    resumed = fit_whisper(
        model,
        examples,
        encoder_source_layers=[0, 2],
        decoder_source_layers=[0, 1],
        target_decoder_layer=2,
        dim_batch=3,
        checkpoint_path=str(checkpoint),
        resume=True,
    )
    torch.testing.assert_close(
        resumed.decoder.jacobians[1], bundle.decoder.jacobians[1]
    )
    assert resumed.encoder.source_means is not None
    assert resumed.encoder.target_mean is not None
    torch.testing.assert_close(
        resumed.encoder.source_means[0], bundle.encoder.source_means[0]
    )
    torch.testing.assert_close(resumed.encoder.target_mean, bundle.encoder.target_mean)


def test_whisper_bundle_merge_accepts_different_shard_provenance():
    model = TinyWhisperLensModel()
    inputs = tiny_inputs(model)
    first = fit_whisper(
        model,
        [inputs],
        encoder_source_layers=[0],
        decoder_source_layers=[],
        target_decoder_layer=2,
        dim_batch=3,
        estimator_name="encoder-aligned",
        artifact_metadata={
            "corpus_fingerprint": "corpus-a",
            "manifest_name": "a.jsonl",
            "requested_examples": 1,
        },
    )
    second = fit_whisper(
        model,
        [inputs, inputs],
        encoder_source_layers=[0],
        decoder_source_layers=[],
        target_decoder_layer=2,
        dim_batch=3,
        estimator_name="encoder-aligned",
        artifact_metadata={
            "corpus_fingerprint": "corpus-b",
            "manifest_name": "b.jsonl",
            "requested_examples": 2,
        },
    )
    merged = WhisperJacobianLens.merge([first, second])
    assert merged.encoder is not None
    assert merged.encoder.n_examples == 3
    assert len(merged.encoder.metadata["shard_provenance"]) == 2
    assert len(merged.estimator_metadata["shard_provenance"]) == 2

    # Per-stream metadata produced by combine_streams remains mergeable too.
    combined_first = WhisperJacobianLens.combine_streams(encoder_bundle=first)
    combined_second = WhisperJacobianLens.combine_streams(encoder_bundle=second)
    combined_merged = WhisperJacobianLens.merge([combined_first, combined_second])
    encoder_metadata = combined_merged.estimator_metadata["encoder"]
    assert len(encoder_metadata["shard_provenance"]) == 2
    assert combined_merged.estimator_metadata["decoder"] is None


def test_checkpoint_resume_rejects_changed_example_content_or_order(tmp_path):
    model = TinyWhisperLensModel()
    first = tiny_inputs(model)
    second = tiny_inputs(model)
    second.input_features[0, 0, 0] += 1.0
    checkpoint = tmp_path / "ordered-fit.pt"
    fit_whisper(
        model,
        [first, second],
        encoder_source_layers=[0],
        decoder_source_layers=[],
        target_decoder_layer=2,
        dim_batch=3,
        checkpoint_path=str(checkpoint),
    )
    with pytest.raises(ValueError, match="configuration does not match"):
        fit_whisper(
            model,
            [second, first],
            encoder_source_layers=[0],
            decoder_source_layers=[],
            target_decoder_layer=2,
            dim_batch=3,
            checkpoint_path=str(checkpoint),
            resume=True,
        )
