"""Resident worker serving engine.

The parent process owns admission, timeout, and HTTP lifecycle. The child worker
process owns model loading and generation.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing as mp
import multiprocessing.process as mp_process
import os
import queue
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable

from difflet.serving.errors import DiffletServingError, internal_error, request_cancelled
from difflet.serving.engines.stage_pipeline import StagePipelineEngine
from difflet.serving.options import validate_worker_heartbeat_interval
from difflet.serving.types import (
    CancellationSignal,
    DiffletGenerateRequest,
    GenerateOutput,
    ResolvedRuntimeBundle,
    WorkerRequestContext,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResidentWorkerConfig:
    max_running_requests: int = 1
    max_queued_requests: int = 8
    queue_timeout: float = 30.0
    request_timeout: float = 300.0
    worker_cancel_timeout: float = 10.0
    worker_restart_timeout: float = 900.0
    worker_heartbeat_interval: float = 30.0

    def __post_init__(self) -> None:
        validate_worker_heartbeat_interval(self.worker_heartbeat_interval)


class ResidentWorkerServingEngine:
    """Serialized resident-worker engine for one loaded serving profile."""

    def __init__(
        self,
        *,
        runtime: ResolvedRuntimeBundle,
        orchestrator_factory: str,
        config: ResidentWorkerConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.profile = runtime.profile
        self.orchestrator_factory = orchestrator_factory
        self.config = config or ResidentWorkerConfig()
        if self.config.max_running_requests != 1:
            raise ValueError("P0 ResidentWorkerServingEngine requires max_running_requests=1")
        self._run_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._pending = 0
        self._worker = _ResidentWorkerProcess(
            runtime=runtime,
            orchestrator_factory=orchestrator_factory,
            startup_timeout=self.config.worker_restart_timeout,
            heartbeat_interval=self.config.worker_heartbeat_interval,
            admission_snapshot=self._admission_snapshot,
        )
        self._draining = False
        self._recovering = False
        self._unrecoverable = False
        self._closed = False
        self._draining_event = asyncio.Event()
        self._recovery_done = asyncio.Event()
        self._recovery_done.set()
        self._startup_task: asyncio.Task[None] | None = None
        self._recovery_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._closed or self._draining:
            raise _engine_draining()
        async with self._lifecycle_lock:
            if self._closed or self._draining:
                raise _engine_draining()
            await self._start_worker_uncancellable()
            if self._closed or self._draining:
                # A concurrent shutdown owns the final worker teardown.
                raise _engine_draining()
            self._unrecoverable = False

    async def shutdown(self) -> None:
        self._draining = True
        self._closed = True
        self._draining_event.set()
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(
                self._shutdown_worker_lifecycle(),
                name="difflet-worker-shutdown",
            )
            self._shutdown_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Worker lifecycle calls run in threads and cannot be cancelled.
            # Complete teardown before propagating cancellation to the caller.
            with suppress(BaseException):
                await task
            raise

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

    async def generate(self, request: DiffletGenerateRequest) -> GenerateOutput:
        if self._draining:
            raise DiffletServingError(503, "engine_draining", "engine is draining", "server_error")
        if self._recovering:
            raise DiffletServingError(
                503, "engine_recovering", "worker is recovering", "server_error"
            )
        if not self._worker.is_ready():
            raise DiffletServingError(
                503, "engine_unavailable", "worker is not ready", "server_error"
            )

        logger.info(
            "engine.generate admitted request_id=%s model=%s queued=%d/%d",
            request.request_id,
            request.model,
            self._pending,
            self.config.max_running_requests + self.config.max_queued_requests,
        )
        received_at = time.monotonic()
        deadline = received_at + float(self.config.request_timeout)
        await self._admit_or_raise()
        logger.info("engine.queue_acquired request_id=%s", request.request_id)
        lock_acquired = False
        try:
            try:
                logger.debug(
                    "engine.wait_lock start request_id=%s timeout=%.3f",
                    request.request_id,
                    float(self.config.queue_timeout),
                )
                wait_timeout = min(
                    float(self.config.queue_timeout),
                    max(deadline - time.monotonic(), 0.0),
                )
                await self._acquire_run_lock_or_drain(wait_timeout)
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
            logger.info("engine.lock_acquired request_id=%s", request.request_id)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "engine.request_timeout request_id=%s before_start", request.request_id
                )
                raise DiffletServingError(
                    504, "request_timeout", "request timed out", "server_error"
                )

            logger.info("engine.run_start request_id=%s", request.request_id)
            run_task = asyncio.create_task(
                self._run_one(request, deadline_monotonic=deadline),
                name=f"difflet-generate-{request.request_id}",
            )
            draining_task = asyncio.create_task(
                self._draining_event.wait(),
                name=f"difflet-draining-{request.request_id}",
            )
            release_lock = True
            try:
                done, _ = await asyncio.wait(
                    (run_task, draining_task),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if run_task in done:
                    logger.info("engine.run_complete request_id=%s", request.request_id)
                    return run_task.result()
                if draining_task in done:
                    run_task.cancel()
                    with suppress(BaseException):
                        await run_task
                    logger.warning("engine.draining_wait_cancel request_id=%s", request.request_id)
                    raise _engine_draining()
                raise asyncio.TimeoutError
            except asyncio.CancelledError:
                release_lock = not self._start_inflight_recovery(
                    run_task, reason="caller_cancelled"
                )
                raise
            except asyncio.TimeoutError as exc:
                release_lock = not self._start_inflight_recovery(run_task, reason="timeout")
                logger.warning("engine.request_timeout request_id=%s", request.request_id)
                raise DiffletServingError(
                    504, "request_timeout", "request timed out", "server_error"
                ) from exc
            except DiffletServingError as exc:
                if _requires_worker_recovery(exc):
                    release_lock = not self._start_inflight_recovery(run_task, reason=exc.code)
                logger.warning(
                    "engine.request_failed request_id=%s code=%s status=%s",
                    request.request_id,
                    exc.code,
                    exc.status_code,
                )
                raise
            finally:
                draining_task.cancel()
                with suppress(BaseException):
                    await draining_task
                if release_lock and lock_acquired:
                    self._run_lock.release()
        finally:
            await self._release_admission()

    async def _run_one(
        self,
        request: DiffletGenerateRequest,
        *,
        deadline_monotonic: float,
    ) -> GenerateOutput:
        return await asyncio.to_thread(
            self._worker.run_generation,
            request,
            deadline_monotonic,
        )

    async def _admit_or_raise(self) -> None:
        async with self._admission_lock:
            capacity = self.config.max_running_requests + self.config.max_queued_requests
            if self._pending >= capacity:
                logger.warning("engine.queue_full current=%d capacity=%d", self._pending, capacity)
                raise DiffletServingError(429, "queue_full", "resident worker queue is full")
            logger.debug("engine.queue_admit before=%d", self._pending)
            self._pending += 1
            logger.debug("engine.queue_admit after=%d", self._pending)

    async def _acquire_run_lock_or_drain(self, timeout: float) -> None:
        logger.debug("engine.acquire_lock timeout=%.3f", timeout)
        lock_task = asyncio.create_task(self._run_lock.acquire())
        draining_task = asyncio.create_task(self._draining_event.wait())
        lock_transferred = False
        try:
            done, _ = await asyncio.wait(
                (lock_task, draining_task),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if draining_task in done:
                raise _engine_draining()
            if lock_task in done:
                lock_transferred = True
                logger.debug("engine.acquire_lock granted")
                return
            raise asyncio.TimeoutError
        finally:
            if not lock_transferred:
                if not lock_task.done():
                    lock_task.cancel()
                    with suppress(BaseException):
                        await lock_task
                elif (
                    not lock_task.cancelled()
                    and lock_task.exception() is None
                    and lock_task.result()
                    and self._run_lock.locked()
                ):
                    self._run_lock.release()
            draining_task.cancel()
            with suppress(BaseException):
                await draining_task

    async def _release_admission(self) -> None:
        async with self._admission_lock:
            if self._pending > 0:
                self._pending -= 1

    def _admission_snapshot(self) -> dict[str, int]:
        pending_requests = self._pending
        running_requests = min(int(self._run_lock.locked()), pending_requests)
        queued_requests = max(pending_requests - running_requests, 0)
        return {
            "running_requests": running_requests,
            "queued_requests": queued_requests,
            "pending_requests": pending_requests,
            "request_capacity": (
                self.config.max_running_requests + self.config.max_queued_requests
            ),
        }

    def _start_inflight_recovery(self, run_task: asyncio.Task, *, reason: str) -> bool:
        if self._closed or self._draining:
            return False
        if self._recovery_task is not None and not self._recovery_task.done():
            return True
        logger.warning("engine.recovery_start request reason=%s", reason)
        self._recovering = True
        self._recovery_done.clear()
        task = asyncio.create_task(
            self._recover_worker(run_task, reason=reason),
            name=f"difflet-worker-recovery-{reason}",
        )
        self._recovery_task = task
        task.add_done_callback(self._clear_recovery_task)
        return True

    async def _recover_worker(self, run_task: asyncio.Task, *, reason: str) -> None:
        clean_cancel = False
        fence_established = False
        try:
            if self._closed or self._draining:
                await self._terminate_worker_for_fence()
                fence_established = True
                return
            if reason in {"caller_cancelled", "timeout", "request_timeout"}:
                self._worker.cancel_inflight()
                clean_cancel = await self._wait_for_terminal_state(run_task)
                logger.info("engine.recovery_cancel_done reason=%s clean=%s", reason, clean_cancel)
            if clean_cancel:
                fence_established = True
            else:
                async with self._lifecycle_lock:
                    await self._run_worker_call_uncancellable(
                        self._worker.terminate,
                        name="difflet-worker-recovery-terminate",
                    )
                    fence_established = True
                    # The process is confirmed dead. Drain the parent-side thread
                    # before reusing worker fields for a replacement process.
                    with suppress(BaseException):
                        await asyncio.shield(run_task)
                    if self._closed or self._draining:
                        return
                    await self._start_worker_uncancellable()
                if self._closed or self._draining:
                    return
                logger.info("engine.recovery_worker_restarted reason=%s", reason)

            if self._closed or self._draining:
                return
            if not self._worker.is_ready():
                logger.error("engine.recovery_failed_not_ready reason=%s", reason)
                raise DiffletServingError(
                    503,
                    "engine_unavailable",
                    "worker did not become ready after recovery",
                    "server_error",
                )
        except BaseException:
            logger.exception("engine.recovery_failed reason=%s", reason)
            self._worker.mark_error()
            self._unrecoverable = True
            # Never fail open: an exception in cancellation, termination, or
            # restart still has to prove that no worker can write the old target.
            fence_established = False
            try:
                await self._terminate_worker_for_fence()
                fence_established = True
            except BaseException:
                logger.critical(
                    "engine.recovery_fence_failed reason=%s",
                    reason,
                    exc_info=True,
                )
        finally:
            if fence_established:
                self._recovering = False
                self._recovery_done.set()
                if self._run_lock.locked():
                    self._run_lock.release()

    async def _shutdown_worker_lifecycle(self) -> None:
        await asyncio.sleep(0)
        startup_task = self._startup_task
        if startup_task is not None and startup_task is not asyncio.current_task():
            with suppress(BaseException):
                await asyncio.shield(startup_task)
        recovery_task = self._recovery_task
        if recovery_task is not None and recovery_task is not asyncio.current_task():
            with suppress(BaseException):
                await asyncio.shield(recovery_task)

        async with self._lifecycle_lock:
            await self._run_worker_call_uncancellable(
                self._worker.shutdown,
                name="difflet-worker-final-shutdown",
            )
        self._recovering = False
        self._recovery_done.set()
        if self._run_lock.locked():
            self._run_lock.release()

    async def _start_worker_uncancellable(self) -> None:
        task = asyncio.create_task(
            asyncio.to_thread(self._worker.start),
            name="difflet-worker-start",
        )
        self._startup_task = task
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                with suppress(BaseException):
                    await task
                raise
        finally:
            if self._startup_task is task:
                self._startup_task = None

    async def _run_worker_call_uncancellable(
        self,
        call: Callable[[], Any],
        *,
        name: str,
    ) -> Any:
        task = asyncio.create_task(asyncio.to_thread(call), name=name)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await task
            raise

    async def _terminate_worker_for_fence(self) -> None:
        async with self._lifecycle_lock:
            await self._run_worker_call_uncancellable(
                self._worker.terminate,
                name="difflet-worker-fence-terminate",
            )

    def _clear_recovery_task(self, task: asyncio.Task[None]) -> None:
        if self._recovery_task is task:
            self._recovery_task = None

    async def wait_for_recovery(self, timeout: float | None = None) -> None:
        """Fence caller-owned cleanup until cancellation acknowledgement or restart."""

        if not self._recovery_done.is_set():
            waiter = self._recovery_done.wait()
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=float(timeout))
        if self._unrecoverable or not self._worker.is_ready():
            raise DiffletServingError(
                503,
                "engine_unavailable",
                "worker did not recover to a ready state",
                "server_error",
            )

    async def _wait_for_terminal_state(self, run_task: asyncio.Task) -> bool:
        try:
            await asyncio.wait_for(
                asyncio.shield(run_task),
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


def _engine_draining() -> DiffletServingError:
    return DiffletServingError(503, "engine_draining", "engine is draining", "server_error")


class _ResidentWorkerProcess:
    def __init__(
        self,
        *,
        runtime: ResolvedRuntimeBundle,
        orchestrator_factory: str,
        startup_timeout: float,
        heartbeat_interval: float,
        admission_snapshot: Callable[[], dict[str, int]],
    ) -> None:
        self.runtime = runtime
        self.orchestrator_factory = orchestrator_factory
        self.startup_timeout = float(startup_timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self._admission_snapshot = admission_snapshot
        self._ctx = mp.get_context("spawn")
        self._cmd_q: mp.Queue | None = None
        self._cancel_q: mp.Queue | None = None
        self._reply_q: mp.Queue | None = None
        self._status_recv: Connection | None = None
        self._status_send: Connection | None = None
        self._process: mp_process.BaseProcess | None = None
        self._status_stop: threading.Event | None = None
        self._status_thread: threading.Thread | None = None
        self._inflight_request_id: str | None = None
        self.ready = False
        self.healthy = False

    def start(self) -> None:
        logger.info(
            "worker_process_start model=%s profile_identity=%s startup_timeout=%.1f",
            self.runtime.profile.model_id,
            self.runtime.runtime_plan.profile_identity,
            self.startup_timeout,
        )
        self.terminate()
        self._cmd_q = self._ctx.Queue()
        self._cancel_q = self._ctx.Queue()
        self._reply_q = self._ctx.Queue()
        self._status_recv, self._status_send = self._ctx.Pipe(duplex=False)
        self._start_status_consumer()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(
                self.orchestrator_factory,
                self.runtime,
                self._cmd_q,
                self._cancel_q,
                self._reply_q,
                self._status_send,
                self.heartbeat_interval,
            ),
            daemon=True,
        )
        try:
            self._process.start()
            self._status_send.close()
            self._status_send = None
            reply = self._get_reply(timeout=self.startup_timeout)
            if reply.get("type") != "ready":
                self.mark_error()
                raise _error_from_reply(reply)
            self.ready = True
            self.healthy = True
            logger.info("worker_process_ready model=%s", self.runtime.profile.model_id)
        except BaseException:
            self.terminate()
            raise

    def run_generation(
        self,
        request: DiffletGenerateRequest,
        deadline_monotonic: float,
    ) -> GenerateOutput:
        logger.info(
            "worker_process_run_start request_id=%s model=%s deadline_remaining=%.3f",
            request.request_id,
            request.model,
            deadline_monotonic - time.monotonic(),
        )
        cmd_q = self._cmd_q
        reply_q = self._reply_q
        process = self._process
        if cmd_q is None or reply_q is None or process is None:
            raise DiffletServingError(
                503, "engine_unavailable", "worker is not started", "server_error"
            )
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
                raise DiffletServingError(
                    503, "engine_unavailable", "worker exited", "server_error"
                )
            try:
                reply = reply_q.get(timeout=timeout)
            except queue.Empty:
                continue
            if reply.get("request_id") != request.request_id:
                continue
            if reply.get("type") == "generation_ok":
                self._clear_inflight(request.request_id)
                logger.info("worker_process_generation_ok request_id=%s", request.request_id)
                return reply["output"]
            if reply.get("type") == "cancel_ack":
                self._clear_inflight(request.request_id)
                logger.warning(
                    "worker_process_generation_cancelled request_id=%s", request.request_id
                )
                raise request_cancelled("worker acknowledged request cancellation")
            self._clear_inflight(request.request_id)
            logger.error("worker_process_generation_error request_id=%s", request.request_id)
            raise _error_from_reply(reply)

    def cancel_inflight(self) -> None:
        logger.info("worker_process_cancel_inflight request_id=%s", self._inflight_request_id)
        if self._cancel_q is not None and self._inflight_request_id is not None:
            self._cancel_q.put({"type": "cancel", "request_id": self._inflight_request_id})

    def shutdown(self) -> None:
        logger.info(
            "worker_process_shutdown requested model=%s alive=%s",
            self.runtime.profile.model_id,
            self._process is not None and self._process.is_alive(),
        )
        if self._cmd_q is not None and self._process is not None and self._process.is_alive():
            self._cmd_q.put({"type": "shutdown"})
            self._process.join(timeout=5)
        self.terminate()

    def terminate(self) -> None:
        logger.info("worker_process_terminate model=%s", self.runtime.profile.model_id)
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                logger.error(
                    "worker_process_terminate_escalate model=%s",
                    self.runtime.profile.model_id,
                )
                kill = getattr(process, "kill", None)
                if kill is not None:
                    kill()
                else:  # pragma: no cover - supported Python processes expose kill
                    process.terminate()
                # A recovery fence cannot be opened while the old process may
                # still write its output target. Wait for confirmed process death.
                process.join()
            if process.is_alive():  # defensive for non-standard process doubles
                raise RuntimeError("resident worker did not terminate")
        self._process = None
        self._stop_status_consumer()
        self._close_queues()
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

    def _start_status_consumer(self) -> None:
        logger.debug("worker_status_consumer_start model=%s", self.runtime.profile.model_id)
        self._status_stop = threading.Event()
        self._status_thread = threading.Thread(
            target=self._consume_status,
            name="difflet-worker-status",
            daemon=True,
        )
        self._status_thread.start()

    def _stop_status_consumer(self) -> None:
        logger.debug("worker_status_consumer_stop model=%s", self.runtime.profile.model_id)
        if self._status_stop is not None:
            self._status_stop.set()
        if self._status_thread is not None:
            self._status_thread.join(timeout=1.0)
        if self._status_recv is not None:
            self._status_recv.close()
            self._status_recv = None
        if self._status_send is not None:
            self._status_send.close()
            self._status_send = None
        self._status_stop = None
        self._status_thread = None

    def _close_queues(self) -> None:
        for name in ("_cmd_q", "_cancel_q", "_reply_q"):
            channel = getattr(self, name)
            if channel is None:
                continue
            with suppress(Exception):
                channel.close()
            with suppress(Exception):
                channel.join_thread()
            setattr(self, name, None)

    def _consume_status(self) -> None:
        status_recv = self._status_recv
        stop = self._status_stop
        if status_recv is None or stop is None:
            return
        last_heartbeat = time.monotonic()
        last_event: dict[str, Any] | None = None
        stale_reported = False
        last_state_signature = None
        last_state_start = time.monotonic()
        last_summary = time.monotonic()
        heartbeat_summary_interval = self.heartbeat_interval
        while not stop.is_set():
            try:
                if not status_recv.poll(0.2):
                    raise TimeoutError
                event = status_recv.recv()
            except TimeoutError:
                if (
                    not stale_reported
                    and time.monotonic() - last_heartbeat >= 3 * self.heartbeat_interval
                ):
                    logger.warning(
                        "worker heartbeat stale last_state=%s request_id=%s stage=%s",
                        (last_event or {}).get("state"),
                        (last_event or {}).get("request_id"),
                        (last_event or {}).get("stage"),
                        extra={"worker_heartbeat_stale": last_event or {}},
                    )
                    stale_reported = True
                continue
            except (EOFError, OSError):
                return
            if event.get("type") != "worker_heartbeat":
                continue
            event = {**event, **self._admission_snapshot()}
            last_heartbeat = time.monotonic()
            last_event = event
            stale_reported = False
            state_signature = (event.get("state"), event.get("request_id"), event.get("stage"))
            now = time.monotonic()
            summary_due = now - last_summary >= heartbeat_summary_interval
            if state_signature != last_state_signature:
                if last_state_signature is not None:
                    logger.info(
                        "worker heartbeat state_dwell_seconds=%.2f from_state=%s to_state=%s",
                        now - last_state_start,
                        last_state_signature[0],
                        state_signature[0],
                        extra={"worker_heartbeat": event},
                    )
                logger.info(
                    "worker heartbeat transition state=%s request_id=%s stage=%s "
                    "running_requests=%d queued_requests=%d pending_requests=%d capacity=%d",
                    event.get("state"),
                    event.get("request_id"),
                    event.get("stage"),
                    event["running_requests"],
                    event["queued_requests"],
                    event["pending_requests"],
                    event["request_capacity"],
                    extra={"worker_heartbeat": event},
                )
                last_summary = now
                last_state_start = now
                last_state_signature = state_signature
            elif summary_due:
                logger.info(
                    "worker heartbeat alive state=%s request_id=%s stage=%s model=%s profile=%s "
                    "running_requests=%d queued_requests=%d pending_requests=%d capacity=%d",
                    event.get("state"),
                    event.get("request_id"),
                    event.get("stage"),
                    event.get("model_id"),
                    event.get("profile_identity"),
                    event["running_requests"],
                    event["queued_requests"],
                    event["pending_requests"],
                    event["request_capacity"],
                    extra={"worker_heartbeat": event},
                )
                last_summary = now


def _worker_main(
    orchestrator_factory: str,
    runtime: ResolvedRuntimeBundle,
    cmd_q: mp.Queue,
    cancel_q: mp.Queue,
    reply_q: mp.Queue,
    status_conn: Connection,
    heartbeat_interval: float,
) -> None:
    stage_engine: StagePipelineEngine | None = None
    try:
        logger.info("worker_main start model=%s", runtime.profile.model_id)
        _apply_worker_runtime_environment(runtime)
    except BaseException as exc:
        logger.exception("worker_main environment_apply_failed model=%s", runtime.profile.model_id)
        reply_q.put(_reply_from_error(exc))
        status_conn.close()
        return
    try:
        validate_worker_heartbeat_interval(heartbeat_interval)
    except (TypeError, ValueError) as exc:
        logger.error(
            "worker_main invalid_heartbeat_interval model=%s interval=%s",
            runtime.profile.model_id,
            heartbeat_interval,
        )
        reply_q.put(_reply_from_error(exc))
        status_conn.close()
        return
    heartbeat_stop = threading.Event()
    heartbeat_state = _WorkerHeartbeatState(
        model_id=runtime.profile.model_id,
        profile_identity=runtime.runtime_plan.profile_identity,
    )
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(status_conn, heartbeat_state, heartbeat_stop, heartbeat_interval),
        name="difflet-worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        logger.info("worker_main loading stage_adapter=%s", orchestrator_factory)
        factory = _load_factory(orchestrator_factory)
        adapter = factory(model_id=runtime.profile.model_id)
        stage_engine = StagePipelineEngine(runtime=runtime, adapter=adapter)
        logger.info("worker_main stage_adapter_created model=%s", runtime.profile.model_id)
        heartbeat_state.set(state="loading")
        asyncio.run(stage_engine.start())
        logger.info("worker_main stage_engine_ready model=%s", runtime.profile.model_id)
        heartbeat_state.set(state="ready")
        reply_q.put({"type": "ready"})
        while True:
            cmd = cmd_q.get()
            cmd_type = cmd.get("type")
            if cmd_type == "shutdown":
                heartbeat_state.set(state="draining")
                logger.info("worker_main shutdown_command model=%s", runtime.profile.model_id)
                if stage_engine is not None:
                    asyncio.run(stage_engine.shutdown())
                reply_q.put({"type": "shutdown_ok"})
                return
            if cmd_type != "generate":
                continue
            request: DiffletGenerateRequest = cmd["request"]
            logger.info(
                "worker_main generate_start request_id=%s model=%s",
                request.request_id,
                runtime.profile.model_id,
            )
            heartbeat_state.set(state="busy", request_id=request.request_id)
            context = WorkerRequestContext.with_timeout(
                request.request_id,
                max(float(cmd["deadline"]) - time.monotonic(), 0.0),
                cancellation=CancellationSignal(
                    external_event=_QueueCancellationSource(cancel_q, request.request_id)
                ),
                stage_callback=lambda stage_id: heartbeat_state.set(
                    state="busy", request_id=request.request_id, stage=stage_id
                ),
            )
            try:
                if stage_engine is None:
                    raise RuntimeError("stage engine is not initialized")
                output = asyncio.run(stage_engine.generate(request, context))
                logger.info("worker_main generate_ok request_id=%s", request.request_id)
                reply_q.put(
                    {
                        "type": "generation_ok",
                        "request_id": request.request_id,
                        "output": output,
                    }
                )
            except asyncio.CancelledError:
                logger.warning("worker_main generate_cancelled request_id=%s", request.request_id)
                reply_q.put({"type": "cancel_ack", "request_id": request.request_id})
            except BaseException as exc:
                logger.exception("worker_main generate_error request_id=%s", request.request_id)
                reply_q.put(_reply_from_error(exc, request_id=request.request_id))
            finally:
                heartbeat_state.set(state="ready")
    except BaseException as exc:
        logger.exception("worker_main failed model=%s", runtime.profile.model_id)
        heartbeat_state.set(state="error")
        reply_q.put(_reply_from_error(exc))
    finally:
        if stage_engine is not None:
            with suppress(BaseException):
                asyncio.run(stage_engine.shutdown())
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        logger.info("worker_main exit model=%s", runtime.profile.model_id)
        status_conn.close()


def _apply_worker_runtime_environment(runtime: ResolvedRuntimeBundle) -> None:
    plan = runtime.runtime_plan
    if plan.mode != "resident":
        raise ValueError(f"resident worker cannot apply {plan.mode!r} runtime plan")
    if len(plan.allocations) != 1:
        raise ValueError(
            f"P0 resident worker requires exactly one allocation; found {len(plan.allocations)}"
        )

    allocation = plan.allocations[0]
    num_cores = int(allocation.effective_num_cores)
    if num_cores <= 0:
        raise ValueError("resident worker effective_num_cores must be positive")
    if int(allocation.world_size) <= 0 or int(allocation.world_size) > num_cores:
        raise ValueError("resident worker allocation world_size must fit effective_num_cores")

    available_core_ids = tuple(int(core_id) for core_id in plan.environment.available_core_ids)
    if any(core_id < 0 for core_id in available_core_ids):
        raise ValueError("resident worker available_core_ids must be nonnegative")
    if len(available_core_ids) != len(set(available_core_ids)):
        raise ValueError("resident worker available_core_ids must be unique")
    if len(available_core_ids) < num_cores:
        raise ValueError(
            "resident worker allocation exceeds available cores: "
            f"requires {num_cores}, has {len(available_core_ids)}"
        )
    visible_core_ids = available_core_ids[:num_cores]
    virtual_core_size = _validated_optional_positive_int(
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        allocation.effective_virtual_core_size,
    )
    logical_nc_config = _validated_optional_positive_int(
        "NEURON_LOGICAL_NC_CONFIG",
        allocation.effective_logical_nc_config,
    )

    child = plan.environment.child_distributed
    if (child.world_size, child.local_world_size, child.rank, child.local_rank) != (1, 1, 0, 0):
        raise ValueError("P0 resident worker child distributed environment must be 1/1/0/0")

    os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join(str(core_id) for core_id in visible_core_ids)
    os.environ["NEURON_RT_NUM_CORES"] = str(num_cores)
    _set_or_clear_int_environment(
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        virtual_core_size,
    )
    _set_or_clear_int_environment(
        "NEURON_LOGICAL_NC_CONFIG",
        logical_nc_config,
    )
    os.environ["WORLD_SIZE"] = str(child.world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(child.local_world_size)
    os.environ["RANK"] = str(child.rank)
    os.environ["LOCAL_RANK"] = str(child.local_rank)


def _validated_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive when configured")
    return value


def _set_or_clear_int_environment(name: str, value: int | None) -> None:
    if value is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = str(value)


class _WorkerHeartbeatState:
    def __init__(self, *, model_id: str, profile_identity: str) -> None:
        self._lock = threading.Lock()
        self._model_id = model_id
        self._profile_identity = profile_identity
        self._state = "starting"
        self._request_id: str | None = None
        self._stage: str | None = None

    def set(
        self,
        *,
        state: str,
        request_id: str | None = None,
        stage: str | None = None,
    ) -> None:
        with self._lock:
            self._state = state
            self._request_id = request_id
            self._stage = stage

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "worker_heartbeat",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "model_id": self._model_id,
                "profile_identity": self._profile_identity,
                "state": self._state,
                "request_id": self._request_id,
                "stage": self._stage,
            }


def _heartbeat_loop(
    status_conn: Connection,
    state: _WorkerHeartbeatState,
    stop: threading.Event,
    interval: float,
) -> None:
    os.set_blocking(status_conn.fileno(), False)
    while not stop.is_set():
        try:
            status_conn.send(state.snapshot())
        except (BlockingIOError, BrokenPipeError, OSError):
            pass
        stop.wait(interval)


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
    if isinstance(exc, DiffletServingError) and exc.code in _PUBLIC_WORKER_ERROR_CODES:
        return {
            "type": "error",
            "request_id": request_id,
            "status_code": exc.status_code,
            "code": exc.code,
            "message": exc.message,
            "error_type": exc.error_type,
        }
    logger.error(
        "worker operation failed request_id=%s",
        request_id,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    public = internal_error()
    return {
        "type": "error",
        "request_id": request_id,
        "status_code": public.status_code,
        "code": public.code,
        "message": public.message,
        "error_type": public.error_type,
    }


def _error_from_reply(reply: dict[str, Any]) -> DiffletServingError:
    code = str(reply.get("code", "internal_error"))
    if code == "internal_error":
        return internal_error()
    if code not in _PUBLIC_WORKER_ERROR_CODES:
        return internal_error()
    return DiffletServingError(
        int(reply.get("status_code", 500)),
        code,
        str(reply.get("message", "Request failed")),
        str(reply.get("error_type", "server_error")),
    )


_PUBLIC_WORKER_ERROR_CODES = frozenset(
    {
        "feature_not_supported",
        "invalid_extra_body",
        "invalid_prompt",
        "profile_mismatch",
        "prompt_too_long",
        "request_cancelled",
        "unsupported_input_modality",
        "unsupported_modality",
    }
)
