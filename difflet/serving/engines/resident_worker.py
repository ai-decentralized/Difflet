"""Resident worker serving engine.

The parent process owns admission, timeout, and HTTP lifecycle. The child worker
process owns model loading and generation.
"""

from __future__ import annotations

import asyncio
import importlib
import multiprocessing as mp
import queue
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from difflet.serving.errors import DiffletServingError, request_cancelled
from difflet.serving.types import DiffletGenerateOutput, DiffletGenerateRequest, ServingProfile
from difflet.serving.types import CancellationSignal, WorkerRequestContext


@dataclass(frozen=True)
class ResidentWorkerConfig:
    max_running_requests: int = 1
    max_queued_requests: int = 8
    queue_timeout: float = 30.0
    request_timeout: float = 300.0
    worker_cancel_timeout: float = 10.0
    worker_restart_timeout: float = 900.0


class ResidentWorkerServingEngine:
    """Serialized resident-worker engine for one loaded serving profile."""

    def __init__(
        self,
        *,
        profile: ServingProfile,
        orchestrator_factory: str,
        config: ResidentWorkerConfig | None = None,
    ) -> None:
        self.profile = profile
        self.orchestrator_factory = orchestrator_factory
        self.config = config or ResidentWorkerConfig()
        if self.config.max_running_requests != 1:
            raise ValueError("P0 ResidentWorkerServingEngine requires max_running_requests=1")
        self._worker = _ResidentWorkerProcess(
            profile=profile,
            orchestrator_factory=orchestrator_factory,
            startup_timeout=self.config.worker_restart_timeout,
        )
        self._run_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._pending = 0
        self._draining = False
        self._recovering = False
        self._unrecoverable = False
        self._closed = False

    async def start(self) -> None:
        await asyncio.to_thread(self._worker.start)
        self._unrecoverable = False

    async def shutdown(self) -> None:
        self._draining = True
        self._closed = True
        await asyncio.to_thread(self._worker.shutdown)

    @property
    def ready(self) -> bool:
        return (not self._closed) and (not self._recovering) and self._worker.is_ready()

    @property
    def healthy(self) -> bool:
        if self._closed or self._unrecoverable:
            return False
        if self._recovering:
            return True
        return self._worker.is_healthy()

    async def generate(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
        if self._draining:
            raise DiffletServingError(503, "engine_draining", "engine is draining", "server_error")
        if self._recovering:
            raise DiffletServingError(503, "engine_recovering", "worker is recovering", "server_error")
        if not self._worker.is_ready():
            raise DiffletServingError(503, "engine_unavailable", "worker is not ready", "server_error")

        received_at = time.monotonic()
        deadline = received_at + float(self.config.request_timeout)
        await self._admit_or_raise()
        lock_acquired = False
        try:
            try:
                wait_timeout = min(
                    float(self.config.queue_timeout),
                    max(deadline - time.monotonic(), 0.0),
                )
                await asyncio.wait_for(
                    self._run_lock.acquire(),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError as exc:
                if time.monotonic() >= deadline:
                    raise DiffletServingError(
                        504,
                        "request_timeout",
                        "request timed out",
                        "server_error",
                    ) from exc
                raise DiffletServingError(
                    429,
                    "queue_timeout",
                    "request waited too long for the resident worker",
                ) from exc
            lock_acquired = True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiffletServingError(504, "request_timeout", "request timed out", "server_error")

            run_task = asyncio.create_task(
                self._run_one(request, deadline_monotonic=deadline),
                name=f"difflet-generate-{request.request_id}",
            )
            release_lock = True
            try:
                return await asyncio.wait_for(asyncio.shield(run_task), timeout=remaining)
            except asyncio.CancelledError:
                release_lock = False
                self._start_inflight_recovery(run_task, reason="caller_cancelled")
                raise
            except asyncio.TimeoutError as exc:
                release_lock = False
                self._start_inflight_recovery(run_task, reason="timeout")
                raise DiffletServingError(504, "request_timeout", "request timed out", "server_error") from exc
            except DiffletServingError as exc:
                if _requires_worker_recovery(exc):
                    release_lock = False
                    self._start_inflight_recovery(run_task, reason=exc.code)
                raise
            finally:
                if release_lock and lock_acquired:
                    self._run_lock.release()
        finally:
            await self._release_admission()

    async def _run_one(
        self,
        request: DiffletGenerateRequest,
        *,
        deadline_monotonic: float,
    ) -> DiffletGenerateOutput:
        return await asyncio.to_thread(
            self._worker.run_generation,
            request,
            deadline_monotonic,
        )

    async def _admit_or_raise(self) -> None:
        async with self._admission_lock:
            capacity = self.config.max_running_requests + self.config.max_queued_requests
            if self._pending >= capacity:
                raise DiffletServingError(429, "queue_full", "resident worker queue is full")
            self._pending += 1

    async def _release_admission(self) -> None:
        async with self._admission_lock:
            if self._pending > 0:
                self._pending -= 1

    def _start_inflight_recovery(self, run_task: asyncio.Task, *, reason: str) -> None:
        self._recovering = True
        asyncio.create_task(
            self._recover_worker(run_task, reason=reason),
            name=f"difflet-worker-recovery-{reason}",
        )

    async def _recover_worker(self, run_task: asyncio.Task, *, reason: str) -> None:
        clean_cancel = False
        try:
            if reason in {"caller_cancelled", "timeout", "request_timeout"}:
                self._worker.cancel_inflight()
                clean_cancel = await self._wait_for_terminal_state(run_task)
            if not clean_cancel:
                await asyncio.to_thread(self._worker.terminate)
                with suppress(BaseException):
                    await asyncio.wait_for(run_task, timeout=1.0)
                await asyncio.wait_for(
                    asyncio.to_thread(self._worker.start),
                    timeout=float(self.config.worker_restart_timeout),
                )
            if not self._worker.is_ready():
                raise DiffletServingError(
                    503,
                    "engine_unavailable",
                    "worker did not become ready after recovery",
                    "server_error",
                )
            self._recovering = False
        except Exception:
            self._worker.mark_error()
            self._unrecoverable = True
            self._recovering = False
        finally:
            if self._run_lock.locked():
                self._run_lock.release()

    async def _wait_for_terminal_state(self, run_task: asyncio.Task) -> bool:
        try:
            await asyncio.wait_for(
                run_task,
                timeout=float(self.config.worker_cancel_timeout),
            )
            return False
        except DiffletServingError as exc:
            if exc.code == "request_cancelled" or exc.status_code < 500:
                return self._worker.is_ready() and self._worker.is_healthy()
            return False
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False


def _requires_worker_recovery(exc: DiffletServingError) -> bool:
    return exc.code == "request_timeout" or exc.status_code >= 500


class _ResidentWorkerProcess:
    def __init__(
        self,
        *,
        profile: ServingProfile,
        orchestrator_factory: str,
        startup_timeout: float,
    ) -> None:
        self.profile = profile
        self.orchestrator_factory = orchestrator_factory
        self.startup_timeout = float(startup_timeout)
        self._ctx = mp.get_context("spawn")
        self._cmd_q: mp.Queue | None = None
        self._cancel_q: mp.Queue | None = None
        self._reply_q: mp.Queue | None = None
        self._process: mp.Process | None = None
        self._inflight_request_id: str | None = None
        self.ready = False
        self.healthy = False

    def start(self) -> None:
        self.terminate()
        self._cmd_q = self._ctx.Queue()
        self._cancel_q = self._ctx.Queue()
        self._reply_q = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(
                self.orchestrator_factory,
                self.profile,
                self._cmd_q,
                self._cancel_q,
                self._reply_q,
            ),
            daemon=True,
        )
        self._process.start()
        reply = self._get_reply(timeout=self.startup_timeout)
        if reply.get("type") != "ready":
            self.mark_error()
            raise _error_from_reply(reply)
        self.ready = True
        self.healthy = True

    def run_generation(
        self,
        request: DiffletGenerateRequest,
        deadline_monotonic: float,
    ) -> DiffletGenerateOutput:
        cmd_q = self._cmd_q
        reply_q = self._reply_q
        process = self._process
        if cmd_q is None or reply_q is None or process is None:
            raise DiffletServingError(503, "engine_unavailable", "worker is not started", "server_error")
        self._inflight_request_id = request.request_id
        cmd_q.put(
            {
                "type": "generate",
                "request": request,
                "deadline": deadline_monotonic,
            }
        )
        while True:
            timeout = max(min(deadline_monotonic - time.monotonic(), 0.1), 0.01)
            if not process.is_alive():
                self.mark_error()
                raise DiffletServingError(503, "engine_unavailable", "worker exited", "server_error")
            try:
                reply = reply_q.get(timeout=timeout)
            except queue.Empty:
                continue
            if reply.get("request_id") != request.request_id:
                continue
            if reply.get("type") == "generation_ok":
                self._clear_inflight(request.request_id)
                return reply["output"]
            if reply.get("type") == "cancel_ack":
                self._clear_inflight(request.request_id)
                raise request_cancelled("worker acknowledged request cancellation")
            self._clear_inflight(request.request_id)
            raise _error_from_reply(reply)

    def cancel_inflight(self) -> None:
        if self._cancel_q is not None and self._inflight_request_id is not None:
            self._cancel_q.put({"type": "cancel", "request_id": self._inflight_request_id})

    def shutdown(self) -> None:
        if self._cmd_q is not None and self._process is not None and self._process.is_alive():
            self._cmd_q.put({"type": "shutdown"})
            self._process.join(timeout=5)
        self.terminate()

    def terminate(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._clear_inflight()
        self.ready = False
        self.healthy = False

    def mark_error(self) -> None:
        self.ready = False
        self.healthy = False

    def is_ready(self) -> bool:
        self._refresh_process_state()
        return self.ready

    def is_healthy(self) -> bool:
        self._refresh_process_state()
        return self.healthy

    def _get_reply(self, *, timeout: float) -> dict[str, Any]:
        if self._reply_q is None:
            raise RuntimeError("reply queue is not initialized")
        try:
            return self._reply_q.get(timeout=timeout)
        except queue.Empty as exc:
            raise DiffletServingError(
                503,
                "engine_unavailable",
                "worker startup timed out",
                "server_error",
            ) from exc

    def _clear_inflight(self, request_id: str | None = None) -> None:
        if request_id is None or request_id == self._inflight_request_id:
            self._inflight_request_id = None

    def _refresh_process_state(self) -> None:
        if self._process is not None and self.ready and not self._process.is_alive():
            self.mark_error()


def _worker_main(
    orchestrator_factory: str,
    profile: ServingProfile,
    cmd_q: mp.Queue,
    cancel_q: mp.Queue,
    reply_q: mp.Queue,
) -> None:
    orchestrator = None
    try:
        factory = _load_factory(orchestrator_factory)
        orchestrator = factory(model_id=profile.model_id)
        orchestrator.load(profile)
        orchestrator.smoke()
        reply_q.put({"type": "ready"})
        while True:
            cmd = cmd_q.get()
            cmd_type = cmd.get("type")
            if cmd_type == "shutdown":
                if orchestrator is not None:
                    orchestrator.shutdown()
                reply_q.put({"type": "shutdown_ok"})
                return
            if cmd_type != "generate":
                continue
            request: DiffletGenerateRequest = cmd["request"]
            context = WorkerRequestContext.with_timeout(
                request.request_id,
                max(float(cmd["deadline"]) - time.monotonic(), 0.0),
                cancellation=CancellationSignal(
                    external_event=_QueueCancellationSource(cancel_q, request.request_id)
                ),
            )
            try:
                output = asyncio.run(orchestrator.generate(request, context))
                reply_q.put(
                    {
                        "type": "generation_ok",
                        "request_id": request.request_id,
                        "output": output,
                    }
                )
            except asyncio.CancelledError:
                reply_q.put({"type": "cancel_ack", "request_id": request.request_id})
            except BaseException as exc:
                reply_q.put(_reply_from_error(exc, request_id=request.request_id))
    except BaseException as exc:
        reply_q.put(_reply_from_error(exc))


def _load_factory(path: str):
    module_name, sep, attr = path.partition(":")
    if sep != ":":
        raise ValueError(f"invalid factory reference {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class _QueueCancellationSource:
    def __init__(self, cancel_q: mp.Queue, request_id: str) -> None:
        self.cancel_q = cancel_q
        self.request_id = request_id
        self._cancelled = False

    def is_set(self) -> bool:
        if self._cancelled:
            return True
        while True:
            try:
                msg = self.cancel_q.get_nowait()
            except queue.Empty:
                break
            if msg.get("request_id") == self.request_id:
                self._cancelled = True
        return self._cancelled


def _reply_from_error(exc: BaseException, *, request_id: str | None = None) -> dict[str, Any]:
    if isinstance(exc, DiffletServingError):
        return {
            "type": "error",
            "request_id": request_id,
            "status_code": exc.status_code,
            "code": exc.code,
            "message": exc.message,
            "error_type": exc.error_type,
        }
    return {
        "type": "error",
        "request_id": request_id,
        "status_code": 503,
        "code": "engine_unavailable",
        "message": f"{type(exc).__name__}: {exc}",
        "error_type": "server_error",
        "traceback": traceback.format_exc(),
    }


def _error_from_reply(reply: dict[str, Any]) -> DiffletServingError:
    return DiffletServingError(
        int(reply.get("status_code", 503)),
        str(reply.get("code", "engine_unavailable")),
        str(reply.get("message", "worker error")),
        str(reply.get("error_type", "server_error")),
    )
