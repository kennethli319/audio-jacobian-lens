# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Bounded, in-memory FIFO queue for expensive audio-model analysis."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


class QueueBackend(Protocol):
    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, Any]: ...


class AnalysisQueueFullError(RuntimeError):
    """Raised when accepting another upload would exceed a queue bound."""


class AnalysisJobNotFoundError(KeyError):
    """Raised for unknown or expired opaque job identifiers."""


class AnalysisJobNotReadyError(RuntimeError):
    """Raised when a result is requested before its job has finished."""


class AnalysisJobCancelledError(RuntimeError):
    """Raised when a cancelled job's result is requested."""


class AnalysisJobFailedError(RuntimeError):
    """Raised when analysis failed and therefore has no result."""

    def __init__(self, message: str, *, client_error: bool = False) -> None:
        super().__init__(message)
        self.client_error = client_error


@dataclass
class _AnalysisJob:
    job_id: str
    payload: bytes | None
    time_bin_overlap_seconds: float | None
    submitted_at: float
    state: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    result: bytes | None = None
    error: str | None = None
    client_error: bool = False
    discard_result: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class AnalysisJobQueue:
    """Run one model call at a time while bounding pending upload memory.

    ``capacity`` counts waiting jobs; one additional job may be running. Raw
    audio is discarded as soon as a worker starts it, and terminal results are
    retained only briefly so polling clients can retrieve them.
    """

    def __init__(
        self,
        backend: QueueBackend,
        *,
        capacity: int = 4,
        max_queued_bytes: int = 64 * 1024 * 1024,
        result_ttl_seconds: float = 600.0,
        max_terminal_jobs: int = 16,
        max_retained_result_bytes: int = 64 * 1024 * 1024,
        initial_estimate_seconds: float = 6.0,
        ewma_alpha: float = 0.3,
    ) -> None:
        if capacity <= 0:
            raise ValueError("analysis queue capacity must be positive")
        if max_queued_bytes <= 0:
            raise ValueError("analysis queue byte limit must be positive")
        if result_ttl_seconds <= 0:
            raise ValueError("analysis queue result TTL must be positive")
        if max_terminal_jobs <= 0:
            raise ValueError("analysis queue terminal-job limit must be positive")
        if max_retained_result_bytes <= 0:
            raise ValueError("analysis queue result-byte limit must be positive")
        if initial_estimate_seconds <= 0:
            raise ValueError("analysis queue estimate must be positive")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("analysis queue EWMA alpha must be in (0, 1]")

        self.backend = backend
        self.capacity = capacity
        self.max_queued_bytes = max_queued_bytes
        self.result_ttl_seconds = result_ttl_seconds
        self.max_terminal_jobs = max_terminal_jobs
        self.max_retained_result_bytes = max_retained_result_bytes
        self._average_seconds = float(initial_estimate_seconds)
        self._ewma_alpha = float(ewma_alpha)
        self._sample_count = 0
        self._condition = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, _AnalysisJob] = {}
        self._active_id: str | None = None
        self._queued_bytes = 0
        self._closed = False
        self._worker: threading.Thread | None = None

    def _now(self) -> float:
        return time.monotonic()

    def _start_worker_locked(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="audio-jlens-analysis-queue",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload = bytes(payload)
        with self._condition:
            self._purge_locked(self._now())
            if self._closed:
                raise AnalysisQueueFullError("analysis queue is shutting down")
            if len(self._pending) >= self.capacity:
                raise AnalysisQueueFullError(
                    "the analysis queue is full; try again after a current job finishes"
                )
            if self._queued_bytes + len(payload) > self.max_queued_bytes:
                raise AnalysisQueueFullError(
                    "the analysis queue has reached its upload-memory limit; try again shortly"
                )
            job_id = secrets.token_urlsafe(24)
            job = _AnalysisJob(
                job_id=job_id,
                payload=payload,
                time_bin_overlap_seconds=time_bin_overlap_seconds,
                submitted_at=self._now(),
            )
            self._jobs[job_id] = job
            self._pending.append(job_id)
            self._queued_bytes += len(payload)
            self._start_worker_locked()
            self._condition.notify_all()
            return self._snapshot_locked(job, self._now())

    def status(self) -> dict[str, Any]:
        with self._condition:
            self._purge_locked(self._now())
            return {
                "enabled": True,
                "capacity": self.capacity,
                "queued": len(self._pending),
                "running": self._active_id is not None,
                "average_seconds": round(self._average_seconds, 1),
                "sample_count": self._sample_count,
            }

    def get(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            now = self._now()
            self._purge_locked(now)
            job = self._jobs.get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            return self._snapshot_locked(job, now)

    def result(self, job_id: str) -> bytes:
        with self._condition:
            self._purge_locked(self._now())
            job = self._jobs.get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            if job.state in {"queued", "running", "cancelling"}:
                raise AnalysisJobNotReadyError("analysis is still queued or running")
            if job.state == "cancelled":
                raise AnalysisJobCancelledError("analysis job was cancelled")
            if job.state == "failed":
                raise AnalysisJobFailedError(
                    job.error or "audio analysis failed",
                    client_error=job.client_error,
                )
            if job.result is None:
                raise AnalysisJobNotReadyError("analysis result is unavailable")
            return job.result

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            done = job.done
        if not done.wait(timeout):
            raise AnalysisJobNotReadyError("timed out waiting for audio analysis")
        return json.loads(self.result(job_id))

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            now = self._now()
            self._purge_locked(now)
            job = self._jobs.get(job_id)
            if job is None:
                raise AnalysisJobNotFoundError(job_id)
            if job.state == "queued":
                self._pending.remove(job_id)
                if job.payload is not None:
                    self._queued_bytes -= len(job.payload)
                job.payload = None
                job.state = "cancelled"
                job.finished_at = now
                job.done.set()
            elif job.state == "running":
                # PyTorch forwards are not safely preemptible. Mark the result
                # for disposal and let the single worker release the model.
                job.state = "cancelling"
                job.discard_result = True
            self._condition.notify_all()
            return self._snapshot_locked(job, now)

    def close(self, timeout: float = 1.0) -> None:
        with self._condition:
            self._closed = True
            now = self._now()
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                if active is not None and active.state == "running":
                    active.state = "cancelling"
                    active.discard_result = True
            while self._pending:
                job = self._jobs[self._pending.popleft()]
                if job.payload is not None:
                    self._queued_bytes -= len(job.payload)
                job.payload = None
                job.state = "cancelled"
                job.finished_at = now
                job.done.set()
            self._condition.notify_all()
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, timeout))

    def _snapshot_locked(
        self,
        job: _AnalysisJob,
        now: float,
    ) -> dict[str, Any]:
        queue_position: int | None = None
        jobs_ahead = 0
        estimated_wait: float | None = None
        estimated_remaining: float | None = None
        elapsed = 0.0

        if job.state == "queued":
            try:
                queue_position = self._pending.index(job.job_id) + 1
            except ValueError:
                queue_position = None
            if queue_position is not None:
                active_remaining = self._active_remaining_locked(now)
                jobs_ahead = (1 if self._active_id is not None else 0) + max(
                    0, queue_position - 1
                )
                estimated_wait = (
                    active_remaining
                    + max(0, queue_position - 1) * self._average_seconds
                )
                estimated_remaining = estimated_wait + self._average_seconds
        elif job.state in {"running", "cancelling"}:
            if job.started_at is not None:
                elapsed = max(0.0, now - job.started_at)
            # The browser presents this as time until completion once running.
            estimated_wait = max(0.0, self._average_seconds - elapsed)
            estimated_remaining = estimated_wait
        elif job.started_at is not None and job.finished_at is not None:
            elapsed = max(0.0, job.finished_at - job.started_at)

        payload: dict[str, Any] = {
            "id": job.job_id,
            "job_id": job.job_id,
            "state": job.state,
            "queue_position": queue_position,
            "jobs_ahead": jobs_ahead,
            "estimated_wait_seconds": (
                None if estimated_wait is None else round(estimated_wait, 1)
            ),
            "estimated_remaining_seconds": (
                None if estimated_remaining is None else round(estimated_remaining, 1)
            ),
            "estimated_job_seconds": round(self._average_seconds, 1),
            "elapsed_seconds": round(elapsed, 1),
            "status_url": f"/api/analysis/jobs/{job.job_id}",
            "result_url": f"/api/analysis/jobs/{job.job_id}/result",
        }
        if job.error:
            payload["error"] = job.error
        return payload

    def _active_remaining_locked(self, now: float) -> float:
        if self._active_id is None:
            return 0.0
        active = self._jobs.get(self._active_id)
        if active is None or active.started_at is None:
            return self._average_seconds
        return max(0.0, self._average_seconds - (now - active.started_at))

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    now = self._now()
                    self._purge_locked(now)
                    self._condition.wait(
                        timeout=self._next_expiry_delay_locked(now),
                    )
                if self._closed and not self._pending:
                    return
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                payload = job.payload or b""
                self._queued_bytes -= len(payload)
                job.payload = None
                job.state = "running"
                job.started_at = self._now()
                self._active_id = job_id
                self._condition.notify_all()

            result, error, client_error = self._execute_job(job, payload)

            # The long-lived worker frame must not keep a private upload alive
            # while it waits for the next job. The Python analysis tree lives
            # only inside _execute_job; clear its serialized input/output here
            # after transferring the retained result to the bounded job store.
            payload = b""
            result_for_job = result
            result = None
            job_for_completion = job
            job = None
            with self._condition:
                finished_at = self._now()
                duration = max(
                    0.0,
                    finished_at
                    - (job_for_completion.started_at or finished_at),
                )
                if error is None and not job_for_completion.discard_result:
                    job_for_completion.result = result_for_job
                    job_for_completion.state = "succeeded"
                    self._sample_count += 1
                    self._average_seconds = (
                        self._ewma_alpha * max(0.1, duration)
                        + (1 - self._ewma_alpha) * self._average_seconds
                    )
                elif job_for_completion.discard_result:
                    job_for_completion.state = "cancelled"
                else:
                    job_for_completion.state = "failed"
                    job_for_completion.error = error
                    job_for_completion.client_error = client_error
                job_for_completion.finished_at = finished_at
                self._active_id = None
                job_for_completion.done.set()
                self._purge_locked(finished_at)
                self._condition.notify_all()

            result_for_job = None
            error = None
            job_for_completion = None

    def _execute_job(
        self,
        job: _AnalysisJob,
        payload: bytes,
    ) -> tuple[bytes | None, str | None, bool]:
        """Run and serialize one analysis in a short-lived stack frame."""

        result: bytes | None = None
        error: str | None = None
        client_error = False
        try:
            analysis = self.backend.analyze(
                payload,
                time_bin_overlap_seconds=job.time_bin_overlap_seconds,
            )
            # Retaining the Python tree costs several times more memory
            # than its wire representation. Cache compact JSON instead.
            result = json.dumps(
                analysis,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except ValueError as exc:
            error = str(exc)
            client_error = True
        except Exception as exc:  # noqa: BLE001 - isolate the queue worker
            error = str(exc) or "audio analysis failed"
            LOGGER.exception("Queued audio analysis failed")
        return result, error, client_error

    def _next_expiry_delay_locked(self, now: float) -> float | None:
        expiries = [
            (job.finished_at or now) + self.result_ttl_seconds
            for job in self._jobs.values()
            if job.state in {"succeeded", "failed", "cancelled"}
            and job.finished_at is not None
        ]
        if not expiries:
            return None
        return max(0.0, min(expiries) - now)

    def _purge_locked(self, now: float) -> None:
        terminal = [
            job
            for job in self._jobs.values()
            if job.state in {"succeeded", "failed", "cancelled"}
            and job.finished_at is not None
        ]
        expired_ids = {
            job.job_id
            for job in terminal
            if now - (job.finished_at or now) >= self.result_ttl_seconds
        }
        retained = sorted(
            (job for job in terminal if job.job_id not in expired_ids),
            key=lambda item: item.finished_at or 0.0,
            reverse=True,
        )
        retained_bytes = 0
        retained_count = 0
        for job in retained:
            result_bytes = len(job.result) if job.result is not None else 0
            exceeds_count = retained_count >= self.max_terminal_jobs
            exceeds_bytes = (
                retained_count > 0
                and retained_bytes + result_bytes > self.max_retained_result_bytes
            )
            if exceeds_count or exceeds_bytes:
                expired_ids.add(job.job_id)
                continue
            retained_count += 1
            retained_bytes += result_bytes
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)
