from __future__ import annotations

import contextlib
import math
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.chatterbox_fitting import (
    CHATTERBOX_CAPTURE_CONVENTION,
    CHATTERBOX_SOURCE_STREAM,
    CHATTERBOX_TARGET_STREAM,
)
from jlens.chatterbox_server import _parser
from jlens.chatterbox_webapp import (
    MLXChatterboxBackend,
    UnknownChatterboxRunError,
)
from jlens.mlx_chatterbox import (
    FORCED_CODE_BIAS_EPSILON,
    FORCED_CODE_BRANCH_METHOD,
    MAX_RELATIVE_RESIDUAL_NORM,
    MAX_RESIDUAL_FORWARD_SPAN,
    MEL_FRAME_RATE_HZ,
    RAW_SPEECH_HEAD_NORMALIZATION,
    RESIDUAL_BRANCH_METHOD,
    SPEECH_CODE_RATE_HZ,
    SPEECH_HEAD_CANDIDATE_SCHEMA_VERSION,
    ChatterboxCapturedRun,
    ChatterboxGenerationConfig,
    MLXChatterboxModel,
    _log_softmax_value,
    _normalize_nonnegative,
    _softmax_row,
    _tokenizer_fingerprint,
)
from jlens.mlx_fitting import PROJECTION_METHOD
from jlens.projected_lens import ProjectedCrossJacobianLens
from jlens.webapp import AnalysisBusyError


class _Tokenizer:
    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = vocabulary or {"alpha": 0, "beta": 1}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)


def _adapter(tmp_path) -> MLXChatterboxModel:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type":"chatterbox_turbo","quantization":{"bits":8}}',
        encoding="utf-8",
    )
    model = SimpleNamespace(
        t3=SimpleNamespace(
            tfmr=SimpleNamespace(
                h=[object() for _ in range(24)],
                config=SimpleNamespace(n_head=16),
            ),
            dim=1024,
            hp=SimpleNamespace(speech_tokens_dict_size=6563),
        ),
        s3gen=object(),
        tokenizer=_Tokenizer(),
        _conds=object(),
        sample_rate=24_000,
    )
    return MLXChatterboxModel(
        model,
        model_id="local/chatterbox",
        model_revision="model-revision",
        model_path=model_path,
        s3_tokenizer_id="local/s3",
        s3_tokenizer_revision="s3-revision",
        s3_tokenizer_path=tmp_path / "s3",
        generation_config=ChatterboxGenerationConfig(max_speech_tokens=12),
    )


def test_generation_config_validates_every_sampling_bound() -> None:
    assert ChatterboxGenerationConfig().top_k == 1
    for kwargs, message in (
        ({"max_speech_tokens": 0}, "max_speech_tokens"),
        ({"temperature": 0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.01}, "top_p"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
    ):
        with pytest.raises(ValueError, match=message):
            ChatterboxGenerationConfig(**kwargs)


def test_softmax_and_nonnegative_normalization_are_stable() -> None:
    probabilities = _softmax_row(np.array([1000.0, 1001.0, 999.0]))
    assert probabilities.sum() == pytest.approx(1.0)
    assert int(probabilities.argmax()) == 1
    assert np.isfinite(probabilities).all()
    assert _log_softmax_value(np.array([0.0, -1000.0]), 1) == pytest.approx(
        -1000.0
    )

    np.testing.assert_allclose(
        _normalize_nonnegative(np.array([-4.0, 1.0, 3.0])),
        np.array([0.0, 0.25, 0.75]),
    )
    np.testing.assert_array_equal(
        _normalize_nonnegative(np.array([-1.0, 0.0])),
        np.zeros(2),
    )


def test_tokenizer_fingerprint_is_order_independent_and_content_sensitive() -> None:
    forward = _Tokenizer({"alpha": 4, "beta": 8})
    reverse = _Tokenizer({"beta": 8, "alpha": 4})
    changed = _Tokenizer({"alpha": 4, "beta!": 8})

    assert _tokenizer_fingerprint(forward) == _tokenizer_fingerprint(reverse)
    assert _tokenizer_fingerprint(forward) != _tokenizer_fingerprint(changed)
    assert len(_tokenizer_fingerprint(forward)) == 16


def test_adapter_metadata_and_text_validation_need_no_mlx_runtime(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    metadata = adapter.metadata()

    assert metadata["model_family"] == "chatterbox_turbo"
    assert metadata["model_revision"] == "model-revision"
    assert metadata["s3_tokenizer_revision"] == "s3-revision"
    assert metadata["t3_layers"] == 24
    assert metadata["t3_width"] == 1024
    assert metadata["attention_heads"] == 16
    assert metadata["quantization"] == {"bits": 8}
    assert metadata["speech_code_rate_hz"] == SPEECH_CODE_RATE_HZ
    assert metadata["mel_frame_rate_hz"] == MEL_FRAME_RATE_HZ
    assert metadata["generation"]["max_speech_tokens"] == 12

    assert adapter._validate_text("  say this  ") == "say this"
    with pytest.raises(ValueError, match="non-empty"):
        adapter._validate_text("   ")
    with pytest.raises(ValueError, match="240"):
        adapter._validate_text("x" * 241)


def test_generation_payload_preserves_code_time_and_probability_schema(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    logits = np.array(
        [[3.0, 1.0, -2.0, 0.0], [-1.0, 0.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    run = ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(1,),
        text_tokens=[
            {
                "index": 0,
                "id": 1,
                "text": "Hello",
                "char_start": 0,
                "char_end": 5,
            }
        ],
        speech_code_ids=(0, 2),
        raw_logits=logits,
        waveform=np.zeros(2400, dtype=np.float32),
        sample_rate=24_000,
        condition_length=2,
        speech_start=3,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=1e-6,
    )

    payload = adapter.generation_payload(run)

    assert payload["schema_version"] == 3
    assert payload["input"]["raw_text"] == "Hello"
    assert payload["input"]["normalized_text"] == "Hello."
    assert payload["output"]["sample_rate"] == 24_000
    assert payload["output"]["duration_seconds"] == pytest.approx(0.1)
    assert payload["output"]["nominal_content_duration_seconds"] == pytest.approx(
        0.08
    )
    assert payload["output"]["trailing_audio_seconds"] == pytest.approx(0.02)
    assert payload["output"]["audio_data_url"].startswith("data:audio/wav;base64,")
    assert [code["id"] for code in payload["output"]["speech_codes"]] == [0, 2]
    assert payload["output"]["speech_codes"][0] == {
        "index": 0,
        "id": 0,
        "start_seconds": 0.0,
        "end_seconds": 0.04,
        "mel_start": 0,
        "mel_end": 2,
        "raw_probability": pytest.approx(float(_softmax_row(logits[0])[0])),
        "raw_log_probability": pytest.approx(
            math.log(float(_softmax_row(logits[0])[0]))
        ),
    }
    assert payload["output"]["speech_codes"][1]["start_seconds"] == 0.04
    assert payload["output"]["speech_codes"][1]["mel_start"] == 2
    assert payload["replay"]["policy"] == "teacher_forced_full_sequence"
    assert len(payload["warnings"]) >= 5


def test_actual_speech_head_candidates_use_full_softmax_rank_and_bounded_top_k(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.speech_vocab_size = 4
    adapter.model.t3.hp.speech_tokens_dict_size = 4
    adapter.model.t3.hp.start_speech_token = 2
    adapter.model.t3.hp.stop_speech_token = 3
    logits = np.array(
        [[3.0, 1.0, 2.0, 0.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    run = ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(1,),
        text_tokens=[],
        speech_code_ids=(0, 3),
        raw_logits=logits,
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=24_000,
        condition_length=2,
        speech_start=3,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=0.0,
    )

    candidates = adapter.speech_head_candidate_payload(run, top_k=2)

    assert candidates["schema_version"] == SPEECH_HEAD_CANDIDATE_SCHEMA_VERSION
    assert candidates["top_k"] == 2
    assert candidates["vocab_size"] == 4
    assert candidates["target_ids"] == [0, 3]
    assert candidates["target_ranks"] == [1, 4]
    assert candidates["normalization"] == RAW_SPEECH_HEAD_NORMALIZATION
    assert candidates["special_token_ids"] == {"start": 2, "stop": 3}
    assert candidates["generation_processors_excluded"] == [
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
    ]
    expected = np.stack([_softmax_row(row) for row in logits])
    assert candidates["target_probabilities"] == pytest.approx(
        [expected[0, 0], expected[1, 3]]
    )
    assert candidates["target_log_probabilities"] == pytest.approx(
        [math.log(expected[0, 0]), math.log(expected[1, 3])]
    )
    assert [entry["id"] for entry in candidates["top_codes"][0]] == [0, 2]
    assert [
        entry["probability"] for entry in candidates["top_codes"][0]
    ] == pytest.approx(expected[0, [0, 2]])
    assert candidates["top_codes"][0][1]["special_token"] == "start"
    assert len(candidates["top_codes"]) == len(run.speech_code_ids)
    assert all(len(position) == 2 for position in candidates["top_codes"])


def test_actual_speech_head_candidate_schema_rejects_misaligned_or_invalid_data(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.speech_vocab_size = 4
    base = ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(1,),
        text_tokens=[],
        speech_code_ids=(0,),
        raw_logits=np.zeros((1, 3), dtype=np.float32),
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=24_000,
        condition_length=2,
        speech_start=3,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=0.0,
    )

    with pytest.raises(ValueError, match="expected"):
        adapter.speech_head_candidate_payload(base, top_k=2)
    base.raw_logits = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="top_k"):
        adapter.speech_head_candidate_payload(base, top_k=0)
    base.raw_logits[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.speech_head_candidate_payload(base, top_k=2)


def _forced_code_parent(adapter: MLXChatterboxModel) -> ChatterboxCapturedRun:
    logits = np.zeros((3, adapter.speech_vocab_size), dtype=np.float32)
    logits[1, 7] = 4.0
    logits[1, 9] = 2.0
    logits[1, 3] = 1.0
    return ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(1,),
        text_tokens=[],
        speech_code_ids=(2, 3, 4),
        raw_logits=logits,
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=24_000,
        condition_length=2,
        speech_start=3,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=0.0,
    )


def test_forced_code_provenance_uses_parent_raw_head_and_unique_winner_bias(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    parent = _forced_code_parent(adapter)

    provenance = adapter._forced_code_provenance(parent, 1, 9)

    probabilities = _softmax_row(parent.raw_logits[1])
    assert provenance["schema_version"] == 1
    assert provenance["kind"] == "forced_speech_code_autoregressive_branch"
    assert provenance["method"] == FORCED_CODE_BRANCH_METHOD
    assert provenance["speech_code_index"] == 1
    assert provenance["prefix_length"] == 1
    assert provenance["original_realized_code_id"] == 3
    assert provenance["replacement_code_id"] == 9
    assert provenance["raw_top1_code_id"] == 7
    assert provenance["raw_top1_probability"] == pytest.approx(probabilities[7])
    assert provenance["replacement_raw_probability"] == pytest.approx(
        probabilities[9]
    )
    assert provenance["replacement_raw_log_probability"] == pytest.approx(
        math.log(probabilities[9])
    )
    assert provenance["replacement_global_rank"] == 2
    assert provenance["replacement_to_raw_top1_logit_gap"] == pytest.approx(2.0)
    assert provenance[
        "minimum_additive_bias_to_be_unique_raw_top1"
    ] == pytest.approx(2.0 + FORCED_CODE_BIAS_EPSILON)
    assert provenance["additive_bias_epsilon"] == FORCED_CODE_BIAS_EPSILON
    assert provenance["raw_head_normalization"] == RAW_SPEECH_HEAD_NORMALIZATION
    assert provenance["suffix_policy"] == (
        "argmax_after_repetition_penalty_and_temperature"
    )
    assert provenance["regenerated_suffix_start_index"] == 2
    assert provenance["parent_speech_code_count"] == 3


def test_forced_code_provenance_rejects_invalid_or_nonreplacement_codes(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.model.t3.hp.start_speech_token = 6561
    adapter.model.t3.hp.stop_speech_token = 6562
    parent = _forced_code_parent(adapter)

    for index, replacement, message in (
        (3, 9, "speech_code_index"),
        (1, 3, "already the realized"),
        (1, 6561, "start control token"),
        (1, 6562, "stop control token"),
        (1, 6563, "ordinary acoustic code"),
        (1, -1, "ordinary acoustic code"),
        (1, True, "must be an integer"),
    ):
        with pytest.raises(ValueError, match=message):
            adapter._forced_code_provenance(parent, index, replacement)


def test_forced_code_branch_preserves_prefix_forces_decision_and_greedily_replays(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter(tmp_path)
    adapter.speech_vocab_size = 6
    adapter.model.t3.hp.speech_tokens_dict_size = 6
    adapter.model.t3.hp.start_speech_token = 4
    adapter.model.t3.hp.stop_speech_token = 5
    adapter.model._conds = SimpleNamespace(t3=object(), gen=object())
    adapter.generation_config = ChatterboxGenerationConfig(
        max_speech_tokens=6,
        repetition_penalty=1.2,
    )

    class _Random:
        seeds: list[int] = []

        @classmethod
        def seed(cls, seed: int) -> None:
            cls.seeds.append(seed)

    core = types.ModuleType("mlx.core")
    core.int32 = np.int32
    core.array = np.array
    core.full = np.full
    core.argmax = np.argmax
    core.eval = lambda *values: None
    core.random = _Random
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class _Transformer:
        def __call__(self, *, inputs_embeds, cache):
            return np.asarray(inputs_embeds, dtype=np.float32), object()

    penalty_prefixes: list[list[int]] = []
    t3 = adapter.model.t3
    t3.tfmr = _Transformer()
    t3.prepare_input_embeds = lambda conds, text_ids, bos: (
        np.array([[[-1.0]]], dtype=np.float32),
        2,
    )
    t3.speech_emb = lambda token: np.asarray(token, dtype=np.float32)[..., None]

    def speech_head(hidden):
        previous = int(hidden[0, -1])
        logits = np.zeros((1, 6), dtype=np.float32)
        top_by_previous = {-1: 2, 2: 0, 1: 3, 3: 5}
        logits[0, top_by_previous[previous]] = 4.0
        if previous == 2:
            logits[0, 1] = 3.0
        return logits

    def repetition_penalty(logits, generated, penalty):
        assert penalty == 1.2
        penalty_prefixes.append(np.asarray(generated)[0].tolist())
        return logits

    t3.speech_head = speech_head
    t3._apply_repetition_penalty = repetition_penalty
    replay_calls: list[tuple[list[int], np.ndarray]] = []

    def replay(text_ids, codes, raw_logits):
        replay_calls.append((list(codes), raw_logits.copy()))
        return 2, 3, "input", {0: "layer"}, 0.0

    monkeypatch.setattr(adapter, "_replay", replay)
    monkeypatch.setattr(
        adapter,
        "_decode_waveform",
        lambda codes: np.asarray(codes, dtype=np.float32),
    )
    parent = ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(11,),
        text_tokens=[{"index": 0, "id": 11, "text": "Hello"}],
        speech_code_ids=(2, 0),
        raw_logits=np.array(
            [
                [0, 0, 4, 0, 0, 0],
                [4, 3, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=24_000,
        condition_length=2,
        speech_start=3,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=0.0,
    )

    branch, intervention = adapter.branch_synthesis(parent, 1, 1)

    assert branch.speech_code_ids == (2, 1, 3)
    assert branch.speech_code_ids[:1] == parent.speech_code_ids[:1]
    assert branch.speech_code_ids[1] == 1
    np.testing.assert_array_equal(branch.waveform, np.array([2, 1, 3]))
    assert replay_calls[0][0] == [2, 1, 3]
    assert replay_calls[0][1].shape == (3, 6)
    assert penalty_prefixes == [[2, 1], [2, 1, 3]]
    assert _Random.seeds == [adapter.generation_config.seed]
    assert intervention["original_realized_code_id"] == 0
    assert intervention["replacement_code_id"] == 1
    assert intervention["branch_speech_code_count"] == 3


def test_residual_branch_request_validation_is_parent_bounded_and_budgeted(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.model.t3.hp.start_speech_token = 6561
    adapter.model.t3.hp.stop_speech_token = 6562
    parent = _forced_code_parent(adapter)

    resolved = adapter._validate_residual_branch_request(
        parent, 1, 3, [8, 0, 4], 2, 0.5
    )
    assert resolved == (1, 3, (0, 4, 8), 2, 0.5)
    assert MAX_RESIDUAL_FORWARD_SPAN == 8
    assert MAX_RELATIVE_RESIDUAL_NORM == 2.0

    invalid = (
        (3, 3, [0], 1, 0.5, "speech_code_index"),
        (1, 6561, [0], 1, 0.5, "start control token"),
        (1, 6562, [0], 1, 0.5, "stop control token"),
        (1, 6563, [0], 1, 0.5, "ordinary acoustic code"),
        (1, 3, [], 1, 0.5, "layers must be non-empty"),
        (1, 3, [0, 0], 1, 0.5, "duplicates"),
        (1, 3, [-1], 1, 0.5, "post-block"),
        (1, 3, [24], 1, 0.5, "post-block"),
        (1, 3, [0], 0, 0.5, "forward_span"),
        (1, 3, [0], 9, 0.5, "forward_span"),
        (2, 3, [0], 2, 0.5, "exceeds the parent"),
        (1, 3, [0], 1, 0.0, "relative_residual_norm"),
        (1, 3, [0], 1, 2.01, "relative_residual_norm"),
    )
    for index, target, layers, span, budget, message in invalid:
        with pytest.raises(ValueError, match=message):
            adapter._validate_residual_branch_request(
                parent, index, target, layers, span, budget
            )


def test_residual_attempt_distinguishes_raw_top1_from_processed_greedy(
    tmp_path,
) -> None:
    adapter = _adapter(tmp_path)
    adapter.speech_vocab_size = 4
    adapter.model.t3.hp.stop_speech_token = 3
    adapter.generation_config = ChatterboxGenerationConfig(
        repetition_penalty=2.0
    )

    attempt = adapter._residual_attempt_payload(
        np.array([0.0, 10.0, 9.0, -2.0]),
        attempt_index=0,
        relative_residual_norm=0.0,
        target_code_id=1,
        generated_prefix=(1,),
    )

    assert attempt["target_rank"] == 1
    assert attempt["target_is_raw_top1"] is True
    assert attempt["processed_greedy_code_id"] == 2
    assert attempt["processed_greedy_equals_target"] is False
    assert attempt["success"] is False

    successful = adapter._residual_attempt_payload(
        np.array([0.0, 20.0, 9.0, -2.0]),
        attempt_index=1,
        relative_residual_norm=0.2,
        target_code_id=1,
        generated_prefix=(1,),
    )
    assert successful["target_is_raw_top1"] is True
    assert successful["processed_greedy_code_id"] == 1
    assert successful["processed_greedy_equals_target"] is True
    assert successful["success"] is True


def test_residual_calibration_refines_success_and_uses_positive_budget_failure(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter(tmp_path)
    parent = _forced_code_parent(adapter)
    adapter.speech_vocab_size = 3
    adapter.model.t3.hp.stop_speech_token = 2
    adapter.generation_config = ChatterboxGenerationConfig(
        repetition_penalty=1.0
    )
    parent.speech_code_ids = (0, 0, 0)
    parent.raw_logits = np.array(
        [[2.0, 0.0, -4.0], [2.0, 0.0, -4.0], [2.0, 0.0, -4.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        adapter,
        "_residual_anchor_logits",
        lambda parent, index, layers, directions, diagnostics, strength: np.array(
            [2.0, strength * 10.0, -4.0], dtype=np.float32
        ),
    )
    chosen, attempts, succeeded = adapter._calibrate_residual_strength(
        parent, 1, 1, [0], {}, [], 1.0
    )
    assert succeeded is True
    assert 0 < chosen < 1.0
    assert attempts[0]["relative_residual_norm"] == 0
    assert any(attempt["success"] for attempt in attempts[1:])

    monkeypatch.setattr(
        adapter,
        "_residual_anchor_logits",
        lambda parent, index, layers, directions, diagnostics, strength: np.array(
            [2.0, strength * 0.1, -4.0], dtype=np.float32
        ),
    )
    chosen, attempts, succeeded = adapter._calibrate_residual_strength(
        parent, 1, 1, [0], {}, [], 0.5
    )
    assert succeeded is False
    assert chosen > 0
    assert chosen <= 0.5
    assert attempts[0]["relative_residual_norm"] == 0


def test_residual_replay_reapplies_deltas_and_stores_edited_post_block_states(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter(tmp_path)
    adapter.hidden_size = 2
    adapter.n_layers = 2
    adapter.speech_vocab_size = 4

    core = types.ModuleType("mlx.core")
    core.int32 = np.int32
    core.array = np.array
    core.full = np.full
    core.concatenate = np.concatenate
    core.arange = np.arange
    core.eval = lambda *values: None
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class _Block:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def __call__(self, residual, cache=None):
            return residual + self.offset, cache

    transformer = SimpleNamespace(
        h=[_Block(1.0), _Block(0.0)],
        wpe=lambda positions: np.zeros((len(positions), 2), dtype=np.float32),
        ln_f=lambda residual: residual,
    )
    t3 = adapter.model.t3
    t3.tfmr = transformer
    t3.hp.start_speech_token = 3
    adapter.model._conds = SimpleNamespace(t3=object(), gen=object())

    def prepare_input_embeds(conds, text_ids, speech_inputs):
        length = int(text_ids.shape[1] + speech_inputs.shape[1])
        return np.zeros((1, length, 2), dtype=np.float32), 0

    t3.prepare_input_embeds = prepare_input_embeds

    def speech_head(residual):
        logits = np.zeros((*residual.shape[:-1], 4), dtype=np.float32)
        logits[..., 0] = residual[..., 0]
        return logits

    t3.speech_head = speech_head
    text_ids = np.array([[9]], dtype=np.int32)
    directions = {
        (0, 0): np.array([1.0, 0.0], dtype=np.float32),
        (0, 1): np.array([1.0, 0.0], dtype=np.float32),
    }
    diagnostics = [
        {
            "layer": 0,
            "speech_code_index": position,
            "baseline_residual_l2_norm": 2.0,
        }
        for position in (0, 1)
    ]
    raw_logits = np.array(
        [[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    replay = adapter._replay_with_residual_edits(
        text_ids,
        [0, 0],
        raw_logits,
        layers=[0],
        positions=[0, 1],
        directions=directions,
        coordinate_diagnostics=diagnostics,
        relative_strength=0.5,
    )

    assert replay[4] == 0
    edited_layer = np.asarray(replay[3][0])
    assert edited_layer[0, 1:, 0].tolist() == [2.0, 2.0]
    assert np.asarray(replay[3][1])[0, 1:, 0].tolist() == [2.0, 2.0]


def test_residual_branch_regenerates_from_anchor_without_forcing_target(
    tmp_path, monkeypatch
) -> None:
    adapter = _adapter(tmp_path)
    adapter.speech_vocab_size = 6
    adapter.model.t3.hp.speech_tokens_dict_size = 6
    adapter.model.t3.hp.start_speech_token = 4
    adapter.model.t3.hp.stop_speech_token = 5
    adapter.model._conds = SimpleNamespace(t3=object(), gen=object())
    adapter.generation_config = ChatterboxGenerationConfig(max_speech_tokens=6)

    class _Random:
        @staticmethod
        def seed(seed):
            assert seed == adapter.generation_config.seed

    core = types.ModuleType("mlx.core")
    core.int32 = np.int32
    core.array = np.array
    core.full = np.full
    core.concatenate = np.concatenate
    core.eval = lambda *values: None
    core.random = _Random
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    t3 = adapter.model.t3

    def prepare(conds, text_ids, speech_inputs):
        length = 2 + int(speech_inputs.shape[1])
        return np.zeros((1, length, 1), dtype=np.float32), 1

    t3.prepare_input_embeds = prepare
    t3.speech_emb = lambda token: np.asarray(token, dtype=np.float32)[..., None]

    def speech_head(hidden):
        logits = np.zeros((*hidden.shape[:-1], 6), dtype=np.float32)
        for index in np.ndindex(hidden.shape[:-1]):
            state = int(hidden[index + (0,)])
            top_id = {0: 0, 1: 1, 2: 3, 3: 5}.get(state, 0)
            logits[index + (top_id,)] = 4.0
        return logits

    t3.speech_head = speech_head
    directions = {
        (0, 1): np.ones(adapter.hidden_size, dtype=np.float32),
        (0, 2): np.ones(adapter.hidden_size, dtype=np.float32),
    }
    coordinates = [
        {
            "layer": 0,
            "speech_code_index": position,
            "competitor_code_id": 1,
            "gradient_l2_norm": 1.0,
            "baseline_residual_l2_norm": 10.0,
        }
        for position in (1, 2)
    ]
    monkeypatch.setattr(
        adapter,
        "_residual_directions",
        lambda parent, target, layers, positions: (directions, coordinates),
    )
    attempts = [
        {
            "attempt_index": 0,
            "relative_residual_norm": 0.25,
            "target_probability": 0.1,
            "target_log_probability": math.log(0.1),
            "target_rank": 2,
            "raw_top1_code_id": 1,
            "target_logit_margin_to_strongest_other": -1.0,
            "target_is_raw_top1": False,
            "processed_greedy_code_id": 1,
            "processed_greedy_equals_target": False,
            "success": False,
        }
    ]
    monkeypatch.setattr(
        adapter,
        "_calibrate_residual_strength",
        lambda *args: (0.25, attempts, False),
    )
    step_calls = 0

    def transformer_step(inputs, cache, edits):
        nonlocal step_calls
        if cache is None:
            hidden = np.zeros_like(inputs)
            hidden[0, -2, 0] = 0
            hidden[0, -1, 0] = 1
        else:
            step_calls += 1
            hidden = np.array([[[2 if step_calls == 1 else 3]]], dtype=np.float32)
        return hidden, object()

    monkeypatch.setattr(adapter, "_residual_transformer_step", transformer_step)
    replay_calls: list[dict[str, object]] = []

    def replay(text_ids, codes, logits, **kwargs):
        replay_calls.append({"codes": list(codes), **kwargs})
        return 1, 2, "input", {0: "edited"}, 0.0

    monkeypatch.setattr(adapter, "_replay_with_residual_edits", replay)
    monkeypatch.setattr(
        adapter,
        "_decode_waveform",
        lambda codes: np.asarray(codes, dtype=np.float32),
    )
    parent = ChatterboxCapturedRun(
        raw_text="Hello",
        normalized_text="Hello.",
        text_token_ids=(9,),
        text_tokens=[],
        speech_code_ids=(0, 1, 1),
        raw_logits=np.zeros((3, 6), dtype=np.float32),
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=24_000,
        condition_length=1,
        speech_start=2,
        input_residual=None,
        post_block_residuals={},
        replay_max_abs_error=0.0,
    )

    branch, intervention = adapter.residual_branch_synthesis(
        parent, 1, 2, [0], 2, 0.5
    )

    assert branch.speech_code_ids == (0, 1, 3)
    assert branch.speech_code_ids[:1] == parent.speech_code_ids[:1]
    assert branch.speech_code_ids[1] != 2
    assert intervention["target_code_id"] == 2
    assert intervention["method"] == RESIDUAL_BRANCH_METHOD
    assert intervention["calibration_status"] == "budget_exhausted"
    assert intervention["processed_greedy_equals_target"] is False
    assert intervention["branch_emitted_code_id_at_start"] == 1
    assert intervention["requested_positions"] == [1, 2]
    assert intervention["applied_positions"] == [1, 2]
    assert replay_calls[0]["positions"] == (1, 2)
    assert replay_calls[0]["relative_strength"] == 0.25


class _BackendModel:
    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.traced: list[tuple[object, int]] = []

    @staticmethod
    def metadata() -> dict[str, object]:
        return {"model_id": "fake-chatterbox", "t3_layers": 24}

    def synthesize(self, text: str) -> object:
        self.synthesized.append(text)
        return SimpleNamespace(text=text)

    @staticmethod
    def generation_payload(run: object) -> dict[str, object]:
        return {"schema_version": 3, "generated_text": run.text}

    def trace(self, run: object, index: int) -> dict[str, object]:
        self.traced.append((run, index))
        return {"selection": {"speech_code_index": index}, "source": run.text}


class _CandidateBackendModel(_BackendModel):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_top_k: list[int] = []

    @staticmethod
    def generation_payload(run: object) -> dict[str, object]:
        return {
            "schema_version": 3,
            "generated_text": run.text,
            "output": {"speech_codes": [{"id": 1}]},
        }

    def speech_head_candidate_payload(
        self, run: object, *, top_k: int
    ) -> dict[str, object]:
        self.candidate_top_k.append(top_k)
        return {
            "schema_version": 1,
            "top_k": top_k,
            "target_ids": [1],
            "top_codes": [[{"id": 1, "probability": 0.75}]],
            "source": run.text,
        }


class _BranchBackendModel(_CandidateBackendModel):
    def __init__(self) -> None:
        super().__init__()
        self.branched: list[tuple[object, int, int]] = []

    def synthesize(self, text: str) -> object:
        self.synthesized.append(text)
        return SimpleNamespace(text=text, speech_code_ids=(1, 2))

    @staticmethod
    def generation_payload(run: object) -> dict[str, object]:
        return {
            "schema_version": 3,
            "generated_text": run.text,
            "output": {
                "speech_codes": [
                    {"index": index, "id": token_id}
                    for index, token_id in enumerate(run.speech_code_ids)
                ]
            },
        }

    def speech_head_candidate_payload(
        self, run: object, *, top_k: int
    ) -> dict[str, object]:
        self.candidate_top_k.append(top_k)
        return {
            "schema_version": 1,
            "top_k": top_k,
            "target_ids": list(run.speech_code_ids),
            "top_codes": [
                [{"id": token_id, "probability": 0.75}]
                for token_id in run.speech_code_ids
            ],
        }

    def branch_synthesis(
        self, parent: object, speech_code_index: int, replacement_code_id: int
    ) -> tuple[object, dict[str, object]]:
        self.branched.append(
            (parent, speech_code_index, replacement_code_id)
        )
        if replacement_code_id == parent.speech_code_ids[speech_code_index]:
            raise ValueError("replacement is already realized")
        codes = (
            *parent.speech_code_ids[:speech_code_index],
            replacement_code_id,
            3,
        )
        run = SimpleNamespace(text=parent.text, speech_code_ids=codes)
        return run, {
            "schema_version": 1,
            "kind": "forced_speech_code_autoregressive_branch",
            "method": FORCED_CODE_BRANCH_METHOD,
            "speech_code_index": speech_code_index,
            "prefix_length": speech_code_index,
            "original_realized_code_id": parent.speech_code_ids[
                speech_code_index
            ],
            "replacement_code_id": replacement_code_id,
            "raw_top1_code_id": 9,
            "replacement_raw_probability": 0.1,
            "replacement_raw_log_probability": math.log(0.1),
            "replacement_global_rank": 2,
            "replacement_to_raw_top1_logit_gap": 0.4,
            "minimum_additive_bias_to_be_unique_raw_top1": 0.400001,
            "additive_bias_epsilon": FORCED_CODE_BIAS_EPSILON,
            "raw_head_normalization": RAW_SPEECH_HEAD_NORMALIZATION,
            "suffix_policy": (
                "argmax_after_repetition_penalty_and_temperature"
            ),
            "regenerated_suffix_start_index": speech_code_index + 1,
            "parent_speech_code_count": len(parent.speech_code_ids),
            "branch_speech_code_count": len(codes),
        }


class _FittedBackendModel(_BackendModel):
    hidden_size = 4
    n_layers = 3
    speech_vocab_size = 4

    @staticmethod
    def metadata() -> dict[str, object]:
        return {
            "model_id": "fake-chatterbox",
            "model_fingerprint": "fake-model-v1",
            "t3_layers": 3,
            "t3_width": 4,
            "speech_vocab_size": 4,
            "valid_speech_codes": 4,
        }

    def synthesize(self, text: str) -> object:
        self.synthesized.append(text)
        return SimpleNamespace(text=text, speech_code_ids=(1, 3))


class _ResidualBackendModel(_FittedBackendModel):
    def __init__(self) -> None:
        super().__init__()
        self.residual_branched: list[tuple[object, int, int, tuple[int, ...], int, float]] = []
        self.residual_error: Exception | None = None

    def synthesize(self, text: str) -> object:
        self.synthesized.append(text)
        return SimpleNamespace(
            text=text,
            speech_code_ids=(1, 3),
            raw_logits=np.array(
                [[1.0, 3.0, 2.0, 0.0], [3.0, 2.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )

    @staticmethod
    def generation_payload(run: object) -> dict[str, object]:
        return {
            "schema_version": 3,
            "generated_text": run.text,
            "output": {
                "speech_codes": [
                    {"index": index, "id": token_id}
                    for index, token_id in enumerate(run.speech_code_ids)
                ]
            },
        }

    @staticmethod
    def speech_head_candidate_payload(
        run: object, *, top_k: int
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "top_k": top_k,
            "target_ids": list(run.speech_code_ids),
            "top_codes": [
                [{"id": int(np.argmax(row)), "probability": 0.5}]
                for row in run.raw_logits
            ],
        }

    def residual_branch_synthesis(
        self,
        parent: object,
        speech_code_index: int,
        target_code_id: int,
        layers: list[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> tuple[object, dict[str, object]]:
        if self.residual_error is not None:
            raise self.residual_error
        self.residual_branched.append(
            (
                parent,
                speech_code_index,
                target_code_id,
                tuple(layers),
                forward_span,
                max_relative_residual_norm,
            )
        )
        run = SimpleNamespace(
            text=parent.text,
            speech_code_ids=(2,),
            raw_logits=np.array([[0.0, 1.0, 4.0, -1.0]], dtype=np.float32),
        )
        positions = list(
            range(speech_code_index, speech_code_index + forward_span)
        )
        coordinates = [
            {
                "layer": layer,
                "speech_code_index": position,
                "competitor_code_id": 1,
                "gradient_l2_norm": 2.0,
                "baseline_residual_l2_norm": 8.0,
                "applied_delta_l2_norm": 2.0,
                "applied_relative_residual_norm": 0.25,
                "applied": position < len(run.speech_code_ids),
            }
            for layer in layers
            for position in positions
        ]
        attempt = {
            "attempt_index": 1,
            "relative_residual_norm": 0.25,
            "target_probability": 0.8,
            "target_log_probability": math.log(0.8),
            "target_rank": 1,
            "raw_top1_code_id": target_code_id,
            "target_logit_margin_to_strongest_other": 1.0,
            "target_is_raw_top1": True,
            "processed_greedy_code_id": target_code_id,
            "processed_greedy_equals_target": True,
            "success": True,
        }
        return run, {
            "schema_version": 1,
            "kind": "t3_post_block_residual_steering_branch",
            "method": RESIDUAL_BRANCH_METHOD,
            "speech_code_index": speech_code_index,
            "target_code_id": target_code_id,
            "original_realized_code_id": parent.speech_code_ids[
                speech_code_index
            ],
            "layers": list(layers),
            "forward_span": forward_span,
            "requested_positions": positions,
            "applied_positions": [speech_code_index],
            "coordinate": "post_t3_block_output_at_speech_prediction_position",
            "direction_objective": (
                "target_raw_logit_minus_parent_strongest_non_target_raw_logit"
            ),
            "direction_source": "parent_teacher_forced_path",
            "future_direction_policy": (
                "position_specific_parent_path_direction_applied_on_dynamic_branch_path"
            ),
            "suffix_policy": "argmax_after_repetition_penalty_and_temperature",
            "max_relative_residual_norm": max_relative_residual_norm,
            "chosen_relative_residual_norm": 0.25,
            "target_became_raw_top1": True,
            "processed_greedy_code_id_at_anchor": target_code_id,
            "processed_greedy_code_id": target_code_id,
            "processed_greedy_equals_target": True,
            "calibration_status": "succeeded",
            "calibration_attempts": [attempt],
            "coordinates": coordinates,
            "parent_speech_code_count": len(parent.speech_code_ids),
            "branch_speech_code_count": len(run.speech_code_ids),
            "branch_emitted_code_id_at_start": target_code_id,
            "first_suffix_divergence_index": speech_code_index,
            "limitations": ["test residual branch"],
        }


def _backend_lens() -> ProjectedCrossJacobianLens:
    factors = torch.eye(4) * 2
    return ProjectedCrossJacobianLens(
        factors,
        {0: factors.clone(), 1: factors.clone()},
        n_examples=3,
        source_dim=4,
        target_dim=4,
        source_stream=CHATTERBOX_SOURCE_STREAM,
        target_stream=CHATTERBOX_TARGET_STREAM,
        projection_method=PROJECTION_METHOD,
        metadata={
            "model_fingerprint": "fake-model-v1",
            "source_layers": [0, 1],
            "target_layer": 2,
            "projection_method": PROJECTION_METHOD,
            "projection_seed": 9,
            "dense_at_full_rank": True,
            "estimator": "projected_average_jacobian",
            "examples_fingerprint": "examples-v1",
            "capture_convention": CHATTERBOX_CAPTURE_CONVENTION,
            "centered": False,
            "target_head": {
                "name": "t3.speech_head",
                "semantic_kind": "speech_code",
                "vocab_size": 4,
                "valid_ordinary_codes": 4,
            },
        },
    )


@pytest.fixture
def fake_mlx_runtime(monkeypatch):
    core = types.ModuleType("mlx.core")
    core.gpu = object()
    streams: list[tuple[int, object]] = []

    def new_thread_local_stream(device):
        assert device is core.gpu
        stream = object()
        streams.append((threading.get_ident(), stream))
        return stream

    core.new_thread_local_stream = new_thread_local_stream
    core.stream = contextlib.nullcontext
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return streams


def test_backend_status_distinguishes_absent_and_loaded_speech_lens() -> None:
    model = _FittedBackendModel()
    without_lens = MLXChatterboxBackend(model).status()
    with_lens = MLXChatterboxBackend(model, _backend_lens()).status()

    assert without_lens["capabilities"]["fitted_speech_code_jlens"] is False
    assert without_lens["speech_code_jlens"] is None
    assert "no compatible fitted" in without_lens["message"]

    assert with_lens["capabilities"]["fitted_speech_code_jlens"] is True
    assert with_lens["message"].startswith(
        "Chatterbox fitted speech-code J-lens"
    )
    summary = with_lens["speech_code_jlens"]
    assert summary["format"] == "projected-cross-jacobian-lens"
    assert len(summary["fingerprint"]) == 16
    assert summary["source_layers"] == [0, 1]
    assert summary["target_layer"] == 2
    assert summary["n_examples"] == 3
    assert summary["centered"] is False
    assert summary["model_fingerprint"] == "fake-model-v1"
    assert summary["examples_fingerprint"] == "examples-v1"
    assert summary["projection"] == {
        "method": PROJECTION_METHOD,
        "rank": 4,
        "target_dim": 4,
        "seed": 9,
        "dense_at_full_rank": True,
    }


def test_backend_constructor_validates_lens_and_top_k_bounds() -> None:
    model = _FittedBackendModel()
    incompatible = _backend_lens()
    incompatible.metadata["model_fingerprint"] = "another-model"

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        MLXChatterboxBackend(model, incompatible)
    for top_k in (0, -1, 21):
        with pytest.raises(ValueError, match=r"top_k must be in \[1, 20\]"):
            MLXChatterboxBackend(model, top_k=top_k)


def test_fitted_generation_payload_uses_full_softmax_and_full_ranks(
    monkeypatch, fake_mlx_runtime
) -> None:
    import jlens.chatterbox_webapp as module

    model = _FittedBackendModel()
    lens = _backend_lens()
    logits = {
        0: torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]]
        ),
        1: torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 3.0, 2.0]]
        ),
    }
    calls: list[tuple[object, object, object]] = []

    def fitted_logits(model_arg, lens_arg, run_arg):
        calls.append((model_arg, lens_arg, run_arg))
        return logits

    monkeypatch.setattr(module, "chatterbox_speech_lens_logits", fitted_logits)
    backend = MLXChatterboxBackend(model, lens, top_k=2)

    generated = backend.synthesize("Read fitted speech codes.")

    assert len(calls) == 1
    assert calls[0][0] is model
    assert calls[0][1] is lens
    fitted = generated["fitted_speech_code_jlens"]
    assert fitted["schema_version"] == 1
    assert fitted["layers"] == [0, 1]
    assert fitted["target_ids"] == [1, 3]
    assert fitted["target_ranks"] == [[2, 4], [3, 3]]
    assert fitted["normalization"] == (
        "full_speech_head_softmax_before_generation_processors"
    )
    assert fitted["artifact"] == backend.status()["speech_code_jlens"]

    expected_log_probabilities = [
        layer_logits.log_softmax(dim=-1)
        for layer_logits in logits.values()
    ]
    for layer_index, expected in enumerate(expected_log_probabilities):
        target_ids = torch.tensor([1, 3])
        expected_targets = expected.gather(1, target_ids[:, None])[:, 0]
        assert fitted["target_log_probabilities"][layer_index] == pytest.approx(
            expected_targets.tolist()
        )
        assert fitted["target_probabilities"][layer_index] == pytest.approx(
            expected_targets.exp().tolist()
        )

    first_top = fitted["top_codes"][0][0]
    assert [entry["id"] for entry in first_top] == [0, 1]
    full_probabilities = logits[0][0].softmax(dim=-1)
    assert [entry["probability"] for entry in first_top] == pytest.approx(
        full_probabilities[[0, 1]].tolist()
    )
    assert sum(entry["probability"] for entry in first_top) == pytest.approx(
        float(full_probabilities[[0, 1]].sum())
    )
    assert sum(entry["probability"] for entry in first_top) < 1
    assert 3 not in [entry["id"] for entry in fitted["top_codes"][0][1]]
    assert fitted["target_ranks"][0][1] == 4


def test_chatterbox_parser_accepts_fitted_lens_and_top_k() -> None:
    defaults = _parser().parse_args([])
    configured = _parser().parse_args(
        ["--lens", "artifacts/chatterbox.pt", "--top-k", "9"]
    )

    assert defaults.lens is None
    assert defaults.top_k == 5
    assert configured.lens == "artifacts/chatterbox.pt"
    assert configured.top_k == 9


def test_backend_delegates_and_reuses_its_thread_local_stream(fake_mlx_runtime) -> None:
    model = _BackendModel()
    backend = MLXChatterboxBackend(model, max_cached_runs=2)

    generated = backend.synthesize("First sentence")
    traced = backend.trace(generated["analysis_id"], 3)
    backend.trace(generated["analysis_id"], 4)

    assert generated["schema_version"] == 3
    assert generated["generated_text"] == "First sentence"
    assert "fitted_speech_code_jlens" not in generated
    assert traced == {
        "analysis_id": generated["analysis_id"],
        "selection": {"speech_code_index": 3},
        "source": "First sentence",
    }
    assert model.synthesized == ["First sentence"]
    assert [index for _, index in model.traced] == [3, 4]
    assert len(fake_mlx_runtime) == 1


def test_backend_attaches_actual_head_candidates_with_its_configured_top_k(
    fake_mlx_runtime,
) -> None:
    model = _CandidateBackendModel()
    backend = MLXChatterboxBackend(model, top_k=7)

    generated = backend.synthesize("Inspect actual candidates")

    assert backend.status()["capabilities"]["speech_head_candidates"] is True
    assert model.candidate_top_k == [7]
    assert generated["output"]["speech_head_candidates"] == {
        "schema_version": 1,
        "top_k": 7,
        "target_ids": [1],
        "top_codes": [[{"id": 1, "probability": 0.75}]],
        "source": "Inspect actual candidates",
    }


def test_backend_cache_is_bounded_and_trace_refreshes_lru(fake_mlx_runtime) -> None:
    backend = MLXChatterboxBackend(_BackendModel(), max_cached_runs=2)
    first = backend.synthesize("first")["analysis_id"]
    second = backend.synthesize("second")["analysis_id"]

    backend.trace(first, 0)
    third = backend.synthesize("third")["analysis_id"]

    assert set(backend._runs) == {first, third}
    with pytest.raises(UnknownChatterboxRunError):
        backend.trace(second, 0)
    assert backend.trace(first, 1)["source"] == "first"


def test_backend_forced_code_branch_returns_full_payload_and_keeps_parent_pair(
    fake_mlx_runtime,
) -> None:
    model = _BranchBackendModel()
    backend = MLXChatterboxBackend(model, max_cached_runs=1, top_k=4)
    parent_id = backend.synthesize("branch this")["analysis_id"]

    result = backend.branch(parent_id, 1, 8)

    branch_id = result["analysis_id"]
    assert branch_id != parent_id
    assert set(backend._runs) == {parent_id, branch_id}
    assert result["schema_version"] == 3
    assert [code["id"] for code in result["output"]["speech_codes"]] == [
        1,
        8,
        3,
    ]
    assert result["output"]["speech_head_candidates"]["target_ids"] == [
        1,
        8,
        3,
    ]
    intervention = result["intervention"]
    assert intervention["parent_analysis_id"] == parent_id
    assert intervention["original_realized_code_id"] == 2
    assert intervention["replacement_code_id"] == 8
    assert intervention["method"] == FORCED_CODE_BRANCH_METHOD
    assert model.branched[0][1:] == (1, 8)
    assert backend.status()["capabilities"]["forced_code_branching"] is True

    traced = backend.trace(branch_id, 1)
    assert traced["intervention"] == intervention
    assert traced["source"] == "branch this"


def test_backend_forced_code_branch_maps_unknown_busy_and_model_validation(
    fake_mlx_runtime,
) -> None:
    backend = MLXChatterboxBackend(_BranchBackendModel(), max_cached_runs=2)
    with pytest.raises(UnknownChatterboxRunError):
        backend.branch("missing", 0, 2)

    parent_id = backend.synthesize("cached")["analysis_id"]
    with pytest.raises(ValueError, match="already realized"):
        backend.branch(parent_id, 0, 1)

    backend._lock.acquire()
    try:
        with pytest.raises(AnalysisBusyError, match="retry shortly"):
            backend.branch(parent_id, 0, 2)
    finally:
        backend._lock.release()


def test_backend_residual_branch_returns_full_diagnostics_and_preserves_pair(
    monkeypatch, fake_mlx_runtime
) -> None:
    import jlens.chatterbox_webapp as module

    model = _ResidualBackendModel()
    lens = _backend_lens()

    def fitted_logits(model_arg, lens_arg, run):
        assert model_arg is model
        assert lens_arg is lens
        if len(run.speech_code_ids) == 2:
            return {
                0: torch.tensor([[3.0, 2.0, 1.0, 0.0], [1.0, 2.0, 3.0, 0.0]]),
                1: torch.tensor([[2.0, 1.0, 4.0, 0.0], [0.0, 3.0, 2.0, 1.0]]),
            }
        return {
            0: torch.tensor([[0.0, 1.0, 4.0, 2.0]]),
            1: torch.tensor([[0.0, 2.0, 5.0, 1.0]]),
        }

    monkeypatch.setattr(module, "chatterbox_speech_lens_logits", fitted_logits)
    backend = MLXChatterboxBackend(model, lens, max_cached_runs=1, top_k=2)
    parent_id = backend.synthesize("residual branch")["analysis_id"]

    result = backend.residual_branch(parent_id, 0, 2, [0, 1], 2, 0.75)

    branch_id = result["analysis_id"]
    assert branch_id != parent_id
    assert set(backend._runs) == {parent_id, branch_id}
    assert backend.status()["capabilities"]["residual_code_steering"] is True
    assert result["output"]["speech_head_candidates"]["target_ids"] == [2]
    assert result["fitted_speech_code_jlens"]["layers"] == [0, 1]
    intervention = result["intervention"]
    assert intervention["parent_analysis_id"] == parent_id
    assert intervention["method"] == RESIDUAL_BRANCH_METHOD
    diagnostics = intervention["target_diagnostics"]
    assert diagnostics["positions"] == [0, 1]
    assert diagnostics["fitted_layers"] == [0, 1]
    assert len(diagnostics["before_probabilities"]) == 2
    assert len(diagnostics["after_probabilities"]) == 2
    assert diagnostics["after_probabilities"][0][1] is None
    assert diagnostics["after_ranks"][1][1] is None
    assert diagnostics["head_after_probabilities"][1] is None
    assert diagnostics["head_after_ranks"][1] is None
    assert diagnostics["parent_realized_ids"] == [1, 3]
    assert diagnostics["branch_realized_ids"] == [2, None]
    assert diagnostics["edited_coordinates"] == [
        {"layer": 0, "speech_code_index": 0},
        {"layer": 1, "speech_code_index": 0},
    ]
    assert diagnostics["first_suffix_divergence_index"] == 0

    traced = backend.trace(branch_id, 0)
    assert traced["intervention"] == intervention


def test_backend_residual_branch_maps_unknown_busy_and_model_validation(
    monkeypatch, fake_mlx_runtime
) -> None:
    import jlens.chatterbox_webapp as module

    model = _ResidualBackendModel()
    monkeypatch.setattr(
        module,
        "chatterbox_speech_lens_logits",
        lambda model, lens, run: {},
    )
    backend = MLXChatterboxBackend(model, max_cached_runs=2)
    with pytest.raises(UnknownChatterboxRunError):
        backend.residual_branch("missing", 0, 2, [0], 1, 0.5)

    parent_id = backend.synthesize("cached")["analysis_id"]
    model.residual_error = ValueError("invalid residual layer")
    with pytest.raises(ValueError, match="invalid residual layer"):
        backend.residual_branch(parent_id, 0, 2, [0], 1, 0.5)
    model.residual_error = None

    backend._lock.acquire()
    try:
        with pytest.raises(AnalysisBusyError, match="retry shortly"):
            backend.residual_branch(parent_id, 0, 2, [0], 1, 0.5)
    finally:
        backend._lock.release()


def test_backend_reports_capabilities_and_rejects_overlap(fake_mlx_runtime) -> None:
    backend = MLXChatterboxBackend(_BackendModel(), max_cached_runs=1)
    status = backend.status()
    assert status["backend"] == "mlx-chatterbox-turbo"
    assert status["capabilities"]["code_to_text_gradient"] is True
    assert status["capabilities"]["s3_frame_jacobian"] is False

    generated = backend.synthesize("cached")
    backend._lock.acquire()
    try:
        with pytest.raises(AnalysisBusyError, match="retry shortly"):
            backend.synthesize("blocked")
        with pytest.raises(AnalysisBusyError, match="retry shortly"):
            backend.trace(generated["analysis_id"], 0)
    finally:
        backend._lock.release()


def test_backend_rejects_invalid_cache_capacity() -> None:
    with pytest.raises(ValueError, match="max_cached_runs"):
        MLXChatterboxBackend(_BackendModel(), max_cached_runs=0)
