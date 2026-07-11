from __future__ import annotations

import contextlib
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from jlens.mlx_webapp import MLXLFMAnalysisBackend
from jlens.webapp import (
    DEFAULT_LFM_SERVING_MAX_NEW_TOKENS,
    AnalysisBusyError,
    _parser,
)


class _Model:
    model_id = "tiny-local-mlx"
    input_sample_rate = 16_000

    def __init__(self) -> None:
        self.prepare_threads: list[int] = []

    @staticmethod
    def validate_projected_lens(lens) -> None:
        assert lens == "lens"

    def prepare_audio(self, waveform, **kwargs):
        self.prepare_threads.append(threading.get_ident())
        return {"waveform": waveform, **kwargs}


@pytest.fixture
def fake_mlx_runtime(monkeypatch):
    core = types.ModuleType("mlx.core")
    core.gpu = object()
    created_streams: list[tuple[int, object]] = []

    def new_thread_local_stream(device):
        assert device is core.gpu
        stream = object()
        created_streams.append((threading.get_ident(), stream))
        return stream

    core.new_thread_local_stream = new_thread_local_stream
    core.stream = contextlib.nullcontext
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return created_streams


def test_mlx_backend_creates_and_reuses_one_stream_per_worker_thread(
    monkeypatch, fake_mlx_runtime
):
    import jlens.mlx_webapp as module

    model = _Model()
    monkeypatch.setattr(
        module,
        "decode_audio_bytes",
        lambda payload, target_rate: SimpleNamespace(
            waveform=payload,
            sampling_rate=target_rate,
            duration_seconds=0.25,
        ),
    )
    monkeypatch.setattr(
        module,
        "analyze_mlx_lfm_run",
        lambda model, lens, inputs, waveform, top_k: {
            "thread": threading.get_ident(),
            "top_k": top_k,
        },
    )
    backend = MLXLFMAnalysisBackend(model, "lens", top_k=3)

    main_result = backend.analyze(b"main")
    backend.analyze(b"main-again")
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_result = executor.submit(backend.analyze, b"worker").result()
        executor.submit(backend.analyze, b"worker-again").result()

    assert main_result["top_k"] == 3
    assert main_result["thread"] != worker_result["thread"]
    assert [thread for thread, _ in fake_mlx_runtime] == [
        main_result["thread"],
        worker_result["thread"],
    ]
    assert model.prepare_threads == [
        main_result["thread"],
        main_result["thread"],
        worker_result["thread"],
        worker_result["thread"],
    ]


def test_mlx_backend_reports_capabilities_and_rejects_overlap(monkeypatch):
    import jlens.mlx_webapp as module

    model = _Model()
    monkeypatch.setattr(
        module,
        "decode_audio_bytes",
        lambda payload, target_rate: SimpleNamespace(
            waveform=payload,
            sampling_rate=target_rate,
            duration_seconds=0.25,
        ),
    )
    backend = MLXLFMAnalysisBackend(model, "lens")
    status = backend.status()

    assert status["backend"] == "mlx-lfm"
    assert status["capabilities"]["generated_audio"] is True
    assert status["capabilities"]["audio_codebook_jlens"] is False
    backend._lock.acquire()
    try:
        with pytest.raises(AnalysisBusyError, match="already running"):
            backend.analyze(b"audio")
    finally:
        backend._lock.release()


def test_mlx_backend_applies_serving_budget_after_artifact_validation(monkeypatch):
    import jlens.mlx_webapp as module

    lens = SimpleNamespace(metadata={"generation": {"max_new_tokens": 18}})
    validated_configs = []

    class LoadedModel:
        model_id = "tiny-local-mlx"
        input_sample_rate = 16_000

        def __init__(self, generation_config):
            self.generation_config = generation_config

        def validate_projected_lens(self, loaded_lens):
            assert loaded_lens is lens
            validated_configs.append(self.generation_config)

    def load_model(model_id, *, revision, generation_config):
        assert model_id == "tiny-local-mlx"
        assert revision == "revision"
        return LoadedModel(generation_config)

    monkeypatch.setattr(
        module.ProjectedCrossJacobianLens,
        "load",
        lambda path: lens,
    )
    monkeypatch.setattr(
        module,
        "MLXLFMModel",
        SimpleNamespace(from_pretrained=load_model),
    )

    backend = MLXLFMAnalysisBackend.load(
        lens_path="lens.pt",
        model_id="tiny-local-mlx",
        revision="revision",
        serving_max_new_tokens=512,
    )

    assert [config.max_new_tokens for config in validated_configs] == [18]
    assert backend.artifact_generation_config.max_new_tokens == 18
    assert backend.model.generation_config.max_new_tokens == 512
    assert backend.status()["generation"] == {
        "artifact_max_new_tokens": 18,
        "serving_max_new_tokens": 512,
        "serving_override": True,
    }


def test_lfm_serving_budget_cli_defaults_to_eos_first_emergency_cap_and_is_positive():
    defaults = _parser().parse_args(["--backend", "mlx-lfm"])
    configured = _parser().parse_args(
        ["--backend", "mlx-lfm", "--lfm-max-new-tokens", "96"]
    )

    assert defaults.lfm_max_new_tokens == DEFAULT_LFM_SERVING_MAX_NEW_TOKENS == 512
    assert configured.lfm_max_new_tokens == 96
    with pytest.raises(SystemExit):
        _parser().parse_args(["--lfm-max-new-tokens", "0"])
