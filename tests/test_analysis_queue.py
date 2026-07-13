from __future__ import annotations

import gc
import sys
import threading
import time
import weakref
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from jlens.analysis_queue import (
    AnalysisJobCancelledError,
    AnalysisJobFailedError,
    AnalysisJobNotReadyError,
    AnalysisJobQueue,
    AnalysisQueueFullError,
)
from jlens.webapp import create_app


class _ControlledBackend:
    """Backend whose forwards advance only when a test releases them."""

    def __init__(self, payloads: tuple[bytes, ...]) -> None:
        self.started = {payload: threading.Event() for payload in payloads}
        self.release = {payload: threading.Event() for payload in payloads}
        self.calls: list[bytes] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def status(self) -> dict[str, object]:
        return {"ready": True, "model_id": "controlled-test-backend"}

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, object]:
        with self._lock:
            self.calls.append(payload)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started[payload].set()
        try:
            if not self.release[payload].wait(timeout=5):
                raise RuntimeError(f"test did not release {payload!r}")
            return {
                "payload": payload.decode(),
                "overlap": time_bin_overlap_seconds,
            }
        finally:
            with self._lock:
                self.active -= 1


class _FailingBackend:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, object]:
        self.calls.append(payload)
        if payload == b"bad-input":
            raise ValueError("the uploaded audio is invalid")
        if payload == b"backend-error":
            raise RuntimeError("inference failed")
        return {"payload": payload.decode()}


class _FastRouteBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, float | None]] = []

    def status(self) -> dict[str, object]:
        return {"ready": True, "model_id": "fast-test-backend"}

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, object]:
        self.calls.append((payload, time_bin_overlap_seconds))
        return {
            "received": len(payload),
            "overlap": time_bin_overlap_seconds,
        }


class _WeakAnalysis(dict[str, object]):
    """Weak-referenceable analysis tree for worker-frame retention checks."""


class _RetentionBackend:
    def __init__(self) -> None:
        self.analysis_ref: weakref.ReferenceType[_WeakAnalysis] | None = None

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, object]:
        analysis = _WeakAnalysis(marker=payload.decode())
        self.analysis_ref = weakref.ref(analysis)
        return analysis


def _eventually(
    callback: Callable[[], object],
    predicate: Callable[[object], bool],
    *,
    timeout: float = 2.0,
) -> object:
    deadline = time.monotonic() + timeout
    last: object = None
    while time.monotonic() < deadline:
        last = callback()
        if predicate(last):
            return last
        threading.Event().wait(0.002)
    pytest.fail(f"condition was not met; last value was {last!r}")


def test_queue_runs_strict_fifo_with_only_one_active_analysis() -> None:
    backend = _ControlledBackend((b"first", b"second", b"third"))
    queue = AnalysisJobQueue(backend, capacity=3, initial_estimate_seconds=9)
    try:
        first = queue.submit(b"first", time_bin_overlap_seconds=0.01)
        assert backend.started[b"first"].wait(timeout=1)

        second = queue.submit(b"second", time_bin_overlap_seconds=0.02)
        third = queue.submit(b"third", time_bin_overlap_seconds=0.03)
        assert queue.get(second["id"])["queue_position"] == 1
        assert queue.get(second["id"])["jobs_ahead"] == 1
        assert queue.get(third["id"])["queue_position"] == 2
        assert queue.get(third["id"])["jobs_ahead"] == 2
        with pytest.raises(AnalysisJobNotReadyError):
            queue.result(second["id"])

        backend.release[b"first"].set()
        assert backend.started[b"second"].wait(timeout=1)
        backend.release[b"second"].set()
        assert backend.started[b"third"].wait(timeout=1)
        backend.release[b"third"].set()

        assert queue.wait(first["id"], timeout=1) == {
            "payload": "first",
            "overlap": 0.01,
        }
        assert queue.wait(second["id"], timeout=1) == {
            "payload": "second",
            "overlap": 0.02,
        }
        assert queue.wait(third["id"], timeout=1) == {
            "payload": "third",
            "overlap": 0.03,
        }
        assert backend.calls == [b"first", b"second", b"third"]
        assert backend.max_active == 1
        assert queue.status()["sample_count"] == 3
    finally:
        queue.close()


def test_queue_enforces_waiting_job_and_pending_byte_bounds() -> None:
    backend = _ControlledBackend((b"active", b"queued", b"replacement"))
    queue = AnalysisJobQueue(
        backend,
        capacity=1,
        max_queued_bytes=64,
    )
    try:
        active = queue.submit(b"active")
        assert backend.started[b"active"].wait(timeout=1)
        queued = queue.submit(b"queued")

        with pytest.raises(AnalysisQueueFullError, match="queue is full"):
            queue.submit(b"replacement")

        assert queue.cancel(queued["id"])["state"] == "cancelled"
        replacement = queue.submit(b"replacement")
        assert queue.get(replacement["id"])["state"] == "queued"

        backend.release[b"active"].set()
        assert backend.started[b"replacement"].wait(timeout=1)
        backend.release[b"replacement"].set()
        assert queue.wait(active["id"], timeout=1)["payload"] == "active"
        assert queue.wait(replacement["id"], timeout=1)["payload"] == "replacement"
    finally:
        queue.close()

    byte_backend = _ControlledBackend((b"a", b"12", b"x"))
    byte_queue = AnalysisJobQueue(
        byte_backend,
        capacity=2,
        max_queued_bytes=2,
    )
    try:
        running = byte_queue.submit(b"a")
        assert byte_backend.started[b"a"].wait(timeout=1)
        pending = byte_queue.submit(b"12")
        with pytest.raises(AnalysisQueueFullError, match="upload-memory limit"):
            byte_queue.submit(b"x")
        assert byte_queue.cancel(pending["id"])["state"] == "cancelled"
        byte_backend.release[b"a"].set()
        assert byte_queue.wait(running["id"], timeout=1)["payload"] == "a"
    finally:
        byte_queue.close()


def test_queue_cancels_waiting_and_running_jobs_without_starting_next_early() -> None:
    backend = _ControlledBackend((b"active", b"waiting"))
    queue = AnalysisJobQueue(backend, capacity=2)
    try:
        active = queue.submit(b"active")
        assert backend.started[b"active"].wait(timeout=1)
        waiting = queue.submit(b"waiting")

        cancelled_waiting = queue.cancel(waiting["id"])
        assert cancelled_waiting["state"] == "cancelled"
        with pytest.raises(AnalysisJobCancelledError):
            queue.result(waiting["id"])
        assert not backend.started[b"waiting"].is_set()

        cancelling = queue.cancel(active["id"])
        assert cancelling["state"] == "cancelling"
        with pytest.raises(AnalysisJobNotReadyError):
            queue.result(active["id"])
        backend.release[b"active"].set()
        _eventually(
            lambda: queue.get(active["id"]),
            lambda value: value["state"] == "cancelled",  # type: ignore[index]
        )
        with pytest.raises(AnalysisJobCancelledError):
            queue.wait(active["id"], timeout=1)
        assert backend.calls == [b"active"]
    finally:
        queue.close()


def test_queue_shutdown_discards_an_in_flight_result() -> None:
    backend = _ControlledBackend((b"active",))
    queue = AnalysisJobQueue(backend, capacity=1)
    active = queue.submit(b"active")
    assert backend.started[b"active"].wait(timeout=1)

    queue.close(timeout=0)
    assert queue.get(active["id"])["state"] == "cancelling"
    backend.release[b"active"].set()

    _eventually(
        lambda: queue.get(active["id"]),
        lambda value: value["state"] == "cancelled",  # type: ignore[index]
    )
    with pytest.raises(AnalysisJobCancelledError):
        queue.result(active["id"])


def test_idle_worker_releases_raw_payload_and_python_analysis_tree() -> None:
    marker = b"private-audio-marker-that-must-not-stay-in-the-worker-frame"
    backend = _RetentionBackend()
    queue = AnalysisJobQueue(backend, capacity=1)
    try:
        job = queue.submit(marker)
        assert queue.wait(job["id"], timeout=1) == {"marker": marker.decode()}

        def worker_released_private_locals() -> bool:
            gc.collect()
            worker = queue._worker
            if worker is None or worker.ident is None:
                return False
            frame = sys._current_frames().get(worker.ident)
            if frame is None:
                return False
            direct_bytes = [
                value
                for value in frame.f_locals.values()
                if isinstance(value, (bytes, bytearray))
            ]
            return backend.analysis_ref is not None and (
                backend.analysis_ref() is None
                and all(marker not in value for value in direct_bytes)
            )

        _eventually(worker_released_private_locals, bool)
    finally:
        queue.close()


def test_idle_worker_expires_terminal_results_without_another_request() -> None:
    backend = _RetentionBackend()
    queue = AnalysisJobQueue(
        backend,
        capacity=1,
        result_ttl_seconds=0.05,
    )
    try:
        job = queue.submit(b"expires-while-idle")
        assert queue.wait(job["id"], timeout=1)["marker"] == "expires-while-idle"

        _eventually(lambda: job["id"] not in queue._jobs, bool, timeout=1)
    finally:
        queue.close()


def test_queue_recovers_after_client_and_backend_failures() -> None:
    backend = _FailingBackend()
    queue = AnalysisJobQueue(backend, capacity=3)
    try:
        invalid = queue.submit(b"bad-input")
        broken = queue.submit(b"backend-error")
        valid = queue.submit(b"good")

        with pytest.raises(AnalysisJobFailedError) as invalid_error:
            queue.wait(invalid["id"], timeout=1)
        assert invalid_error.value.client_error is True
        assert str(invalid_error.value) == "the uploaded audio is invalid"

        with pytest.raises(AnalysisJobFailedError) as backend_error:
            queue.wait(broken["id"], timeout=1)
        assert backend_error.value.client_error is False
        assert str(backend_error.value) == "inference failed"

        assert queue.wait(valid["id"], timeout=1) == {"payload": "good"}
        assert backend.calls == [b"bad-input", b"backend-error", b"good"]
        assert queue.status()["sample_count"] == 1
    finally:
        queue.close()


def test_queue_routes_expose_status_position_result_cancel_and_full_response(
    tmp_path,
) -> None:
    backend = _ControlledBackend((b"one", b"two", b"three"))
    app = create_app(
        backend,
        web_dir=tmp_path,
        analysis_queue_capacity=1,
        analysis_queue_initial_seconds=7,
    )

    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["analysis_queue"] == {
            "enabled": True,
            "capacity": 1,
            "queued": 0,
            "running": False,
            "average_seconds": 7.0,
            "sample_count": 0,
        }

        first = client.post(
            "/api/analysis/jobs",
            files={"audio": ("one.wav", b"one", "audio/wav")},
            data={"time_bin_overlap_seconds": "0.02"},
        )
        assert first.status_code == 202
        assert first.headers["cache-control"] == "no-store"
        first_job = first.json()
        assert first.headers["location"] == first_job["status_url"]
        assert backend.started[b"one"].wait(timeout=1)

        running = client.get(first_job["status_url"])
        assert running.status_code == 200
        assert running.json()["state"] == "running"
        assert client.get(first_job["result_url"]).status_code == 409

        second = client.post(
            "/api/analysis/jobs",
            files={"audio": ("two.wav", b"two", "audio/wav")},
        )
        assert second.status_code == 202
        second_job = client.get(second.json()["status_url"]).json()
        assert second_job["state"] == "queued"
        assert second_job["queue_position"] == 1
        assert second_job["jobs_ahead"] == 1
        assert second_job["estimated_wait_seconds"] is not None

        full = client.post(
            "/api/analysis/jobs",
            files={"audio": ("three.wav", b"three", "audio/wav")},
        )
        assert full.status_code == 429
        assert full.headers["retry-after"] == "7"
        assert "queue is full" in full.json()["detail"]

        # The legacy endpoint is deliberately funneled through the same queue.
        legacy_full = client.post(
            "/api/analyze",
            files={"audio": ("three.wav", b"three", "audio/wav")},
        )
        assert legacy_full.status_code == 429
        assert backend.calls == [b"one"]

        cancelled = client.delete(second.json()["status_url"])
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelled"
        assert client.get(second.json()["result_url"]).status_code == 410

        backend.release[b"one"].set()
        completed = _eventually(
            lambda: client.get(first_job["status_url"]),
            lambda value: value.json()["state"] == "succeeded",  # type: ignore[union-attr]
        )
        assert completed.headers["cache-control"] == "no-store"  # type: ignore[union-attr]
        result = client.get(first_job["result_url"])
        assert result.status_code == 200
        assert result.json() == {"payload": "one", "overlap": 0.02}
        assert client.get("/api/status").json()["analysis_queue"][
            "sample_count"
        ] == 1


def test_legacy_analyze_route_waits_for_and_returns_queued_result(tmp_path) -> None:
    backend = _FastRouteBackend()
    app = create_app(
        backend,
        web_dir=tmp_path,
        analysis_queue_capacity=2,
        analysis_queue_initial_seconds=4,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            files={"audio": ("legacy.wav", b"legacy", "audio/wav")},
            data={"time_bin_overlap_seconds": "0.03"},
        )

        assert response.status_code == 200
        assert response.json() == {"received": 6, "overlap": 0.03}
        assert backend.calls == [(b"legacy", 0.03)]
        queue_status = client.get("/api/status").json()["analysis_queue"]
        assert queue_status["sample_count"] == 1
        assert queue_status["queued"] == 0
        assert queue_status["running"] is False


def test_queue_routes_apply_origin_and_request_size_guards(tmp_path) -> None:
    app = create_app(
        _FastRouteBackend(),
        web_dir=tmp_path,
        analysis_queue_capacity=1,
    )
    with TestClient(app) as client:
        cross_origin_post = client.post(
            "/api/analysis/jobs",
            files={"audio": ("cross-origin.wav", b"audio", "audio/wav")},
            headers={"origin": "https://example.invalid"},
        )
        cross_origin_get = client.get(
            "/api/analysis/jobs/not-a-job",
            headers={"origin": "https://example.invalid"},
        )
        oversized = client.post(
            "/api/analysis/jobs",
            content=b"not parsed",
            headers={"content-length": str(66 * 1024 * 1024)},
        )

        assert cross_origin_post.status_code == 403
        assert cross_origin_get.status_code == 403
        assert oversized.status_code == 413
        assert oversized.json()["detail"] == "analysis request exceeds the 64 MB limit"


def test_queue_configuration_rejects_missing_or_thread_affine_backend(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="requires a model backend"):
        create_app(None, web_dir=tmp_path, analysis_queue_capacity=1)

    backend = _FastRouteBackend()
    backend.requires_server_thread = True  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="requires the server thread"):
        create_app(backend, web_dir=tmp_path, analysis_queue_capacity=1)
