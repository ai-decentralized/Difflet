"""Shared lifecycle and admission owner for synchronous and asynchronous video work."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import logging
import math
import os
import stat
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from difflet.serving.errors import DiffletServingError, internal_error, request_cancelled
from difflet.serving.types import (
    DiffletGenerateRequest,
    FileBackedGenerateOutput,
    FileOutputTarget,
)
from difflet.serving.video_jobs import (
    InvalidVideoJobTransition,
    InMemoryVideoJobRepository,
    VideoJob,
    VideoJobCreate,
    VideoJobError,
    VideoJobNotFound,
    VideoJobPage,
    VideoJobStateConflict,
)
from difflet.serving.video_storage import (
    LocalVideoArtifactStore,
    VideoArtifact,
    VideoArtifactLease,
    VideoArtifactTarget,
)

logger = logging.getLogger(__name__)


class _WorkQueue(deque["_VideoWorkItem"]):
    def empty(self) -> bool:
        return not self


@dataclass(frozen=True, slots=True)
class VideoGenerationResult:
    request_id: str
    artifact: VideoArtifact
    media_metadata: dict[str, Any]
    inference_time_s: float


@dataclass(slots=True)
class _VideoWorkItem:
    key: str
    request: DiffletGenerateRequest
    job_backed: bool
    future: asyncio.Future[VideoGenerationResult] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    submission_done: asyncio.Event = field(default_factory=asyncio.Event)
    create_task: asyncio.Task[VideoJob] | None = None
    task: asyncio.Task[None] | None = None
    target: VideoArtifactTarget | None = None
    artifact: VideoArtifact | None = None
    cancelled: bool = False
    delete_requested: bool = False
    publication_claimed: bool = False
    deadline_expired: bool = False
    finished: bool = False
    enqueued_monotonic: float = field(default_factory=time.monotonic)
    phase: str = "creating"
    deadline_task: asyncio.Task[None] | None = None
    deadline_monotonic: float | None = None
    storage_state: str = "unreserved"
    accounted_bytes: int = 0
    job_slot_owned: bool = False


_ROOT_LEASE_REGISTRY_LOCK = threading.Lock()
_ROOT_LEASE_REGISTRY: set[Path] = set()


class _VideoArtifactRootLease:
    """Exclusive process/host lease for the ephemeral media root."""

    def __init__(
        self,
        *,
        artifacts: LocalVideoArtifactStore,
    ) -> None:
        paths = {artifacts.root / ".difflet-video-service.lock"}
        self._paths = tuple(sorted(paths, key=str))
        self._fds: list[int] = []
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        if self._held:
            return
        with _ROOT_LEASE_REGISTRY_LOCK:
            conflict = next(
                (path for path in self._paths if path in _ROOT_LEASE_REGISTRY),
                None,
            )
            if conflict is not None:
                raise RuntimeError(
                    f"video service storage is already owned by another instance: {conflict}"
                )
            _ROOT_LEASE_REGISTRY.update(self._paths)

        opened: list[int] = []
        try:
            for path in self._paths:
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                opened.append(fd)
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or int(info.st_nlink) != 1:
                    raise RuntimeError(f"video service lock is not a private regular file: {path}")
                os.fchmod(fd, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    raise RuntimeError(
                        f"video service storage is already owned by another instance: {path}"
                    ) from exc
        except BaseException:
            for fd in reversed(opened):
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                with suppress(OSError):
                    os.close(fd)
            with _ROOT_LEASE_REGISTRY_LOCK:
                _ROOT_LEASE_REGISTRY.difference_update(self._paths)
            raise
        self._fds = opened
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        for fd in reversed(self._fds):
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(fd)
        self._fds.clear()
        self._held = False
        with _ROOT_LEASE_REGISTRY_LOCK:
            _ROOT_LEASE_REGISTRY.difference_update(self._paths)


class VideoGenerationService:
    """One bounded FIFO for both Videos API submission styles.

    The resident engine remains the sole model owner, while this service is the
    sole parent-side admission owner.  It also owns the cancellation fence and
    the staging-to-committed publication protocol.
    """

    def __init__(
        self,
        *,
        engine,
        jobs: InMemoryVideoJobRepository,
        artifacts: LocalVideoArtifactStore,
        max_queued_requests: int,
        queue_timeout_s: float,
        request_timeout_s: float,
        recovery_timeout_s: float,
        retention_seconds: int = 25 * 60 * 60,
        max_jobs: int = 4096,
        sweep_interval_s: float = 5 * 60.0,
    ) -> None:
        if isinstance(max_queued_requests, bool) or max_queued_requests < 0:
            raise ValueError("max_queued_requests must be a nonnegative integer")
        if not math.isfinite(recovery_timeout_s) or recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be positive")
        if (
            not math.isfinite(queue_timeout_s)
            or queue_timeout_s <= 0
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("video queue and request timeouts must be positive")
        if (
            isinstance(retention_seconds, bool)
            or not isinstance(retention_seconds, int)
            or retention_seconds <= 0
            or isinstance(max_jobs, bool)
            or not isinstance(max_jobs, int)
            or max_jobs <= 0
            or not math.isfinite(sweep_interval_s)
            or sweep_interval_s <= 0
        ):
            raise ValueError("video retention, job cap, and sweep interval must be positive")
        self.engine = engine
        self.jobs = jobs
        self.artifacts = artifacts
        self.capacity = 1 + int(max_queued_requests)
        self.queue_timeout_s = float(queue_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.recovery_timeout_s = float(recovery_timeout_s)
        self.retention_seconds = int(retention_seconds)
        self.max_jobs = int(max_jobs)
        self.sweep_interval_s = float(sweep_interval_s)
        self.storage_margin_bytes = max(1024**3, self.artifacts.max_artifact_bytes)
        self._queue: _WorkQueue = _WorkQueue()
        self._queue_ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._content_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._items: dict[str, _VideoWorkItem] = {}
        self._pending = 0
        self._dispatcher: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._accepting = False
        self._closed = False
        self._root_lease = _VideoArtifactRootLease(artifacts=artifacts)
        self._active_reserved_bytes = 0
        self._retained_bytes = 0
        self._job_slots_reserved = 0

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("video generation service is closed")
            if self._dispatcher is not None:
                return
            self._root_lease.acquire()
            try:
                # Jobs are deliberately process-local.  Every service lifecycle
                # starts empty, and any files left by an unclean prior process
                # are unreachable orphans that can be removed immediately.
                await _run_thread_fenced(self.jobs.clear)
                sweep = await _run_thread_fenced(self.artifacts.purge_all_managed)
            except BaseException:
                self._root_lease.release()
                raise
            logger.info(
                "video.service_start ephemeral_jobs=true staging_removed=%d artifacts_removed=%d",
                len(sweep.staging_removed),
                len(sweep.artifacts_removed),
            )
            self._accepting = True
            self._job_slots_reserved = 0
            self._dispatcher = asyncio.create_task(
                self._dispatch_loop(), name="difflet-video-dispatcher"
            )
            self._sweeper = asyncio.create_task(self._sweep_loop(), name="difflet-video-sweeper")

    async def shutdown(self) -> None:
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(
                self._shutdown_impl(),
                name="difflet-video-shutdown",
            )
            self._shutdown_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await _await_task_fenced(task)
            raise

    async def _shutdown_impl(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            async with self._lock:
                self._accepting = False
                items = tuple(self._items.values())
                self._queue_ready.set()
            for item in items:
                await self._cancel_item(item, delete_requested=False)
            dispatcher = self._dispatcher
            if dispatcher is not None:
                with suppress(BaseException):
                    await dispatcher
            sweeper = self._sweeper
            if sweeper is not None:
                sweeper.cancel()
                with suppress(BaseException):
                    await sweeper
            self._sweeper = None
            self._dispatcher = None
            self._closed = True
            try:
                try:
                    if self._root_lease.held:
                        async with self._content_lock:
                            sweep = await _run_thread_fenced(self.artifacts.purge_all_managed)
                        logger.info(
                            "video.service_stop ephemeral_jobs=true staging_removed=%d "
                            "artifacts_removed=%d",
                            len(sweep.staging_removed),
                            len(sweep.artifacts_removed),
                        )
                finally:
                    # close() clears all process-local job metadata.
                    await _run_thread_fenced(self.jobs.close)
                    self._job_slots_reserved = 0
            finally:
                self._root_lease.release()

    async def create_async(
        self,
        request: DiffletGenerateRequest,
        *,
        deadline: float | None = None,
    ) -> VideoJob:
        self._require_video_request(request)
        await self.sweep_expired()
        item = _VideoWorkItem(
            key=request.request_id,
            request=request,
            job_backed=True,
            deadline_monotonic=deadline,
        )
        await self._reserve(item)
        try:
            return await self._create_async_reserved(item)
        finally:
            item.submission_done.set()

    async def _create_async_reserved(self, item: _VideoWorkItem) -> VideoJob:
        request = item.request
        deadline = self._item_deadline(item)
        create_task = asyncio.create_task(
            asyncio.to_thread(
                self.jobs.create,
                VideoJobCreate(
                    id=request.request_id,
                    model=request.model,
                    prompt=request.prompt,
                    request=_job_request_metadata(request),
                    user=request.video.user if request.video is not None else None,
                ),
            ),
            name=f"difflet-video-create-{request.request_id}",
        )
        item.create_task = create_task
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            job = await asyncio.wait_for(asyncio.shield(create_task), timeout=remaining)
        except asyncio.TimeoutError:
            async with self._lock:
                item.deadline_expired = True
                item.cancelled = True
            timed_out_job: VideoJob | None = None
            try:
                timed_out_job = await _await_task_fenced(create_task)
            except BaseException:
                pass
            async with self._lock:
                if item.create_task is create_task:
                    item.create_task = None
            if timed_out_job is not None:
                await self._mark_failed(item.key, _timeout_job_error())
            await self._finish_item(item)
            raise _request_timeout_error()
        except asyncio.CancelledError:
            # Repository work already dispatched to a thread cannot be
            # force-cancelled. Fence it before releasing admission, then remove
            # any job that was created but never enqueued.
            async with self._lock:
                item.cancelled = True
                item.delete_requested = True
            cancelled_job: VideoJob | None = None
            try:
                cancelled_job = await _await_task_fenced(create_task)
            except BaseException:
                pass
            async with self._lock:
                if item.create_task is create_task:
                    item.create_task = None
            if cancelled_job is not None:
                with suppress(Exception):
                    await asyncio.to_thread(
                        self.jobs.delete,
                        cancelled_job.id,
                        expected_statuses="queued",
                    )
            await self._finish_item(item)
            raise
        except BaseException:
            async with self._lock:
                if item.create_task is create_task:
                    item.create_task = None
            await self._finish_item(item)
            raise

        rejection: DiffletServingError | None = None
        async with self._lock:
            if item.create_task is create_task:
                item.create_task = None
            if time.monotonic() >= deadline:
                item.deadline_expired = True
                item.cancelled = True
                rejection = _request_timeout_error()
            elif item.finished or item.cancelled or not self._accepting or self._closed:
                item.cancelled = True
                rejection = _service_unavailable_error()
            else:
                item.phase = "queued"
                self._start_deadline_watch(item)
                self._queue.append(item)
                self._queue_ready.set()
        if rejection is not None:
            if item.deadline_expired:
                await self._mark_failed(item.key, _timeout_job_error())
                await self._finish_item(item)
            else:
                # Shutdown owns terminal persistence for admitted work and does
                # not close the repository until this submission unwinds.
                await item.done.wait()
            raise rejection
        return job

    async def generate_sync(
        self,
        request: DiffletGenerateRequest,
        *,
        deadline: float | None = None,
    ) -> VideoGenerationResult:
        self._require_video_request(request)
        future: asyncio.Future[VideoGenerationResult] = asyncio.get_running_loop().create_future()
        item = _VideoWorkItem(
            key=request.request_id,
            request=request,
            job_backed=False,
            future=future,
            deadline_monotonic=deadline,
        )
        await self._reserve(item)
        rejection: DiffletServingError | None = None
        async with self._lock:
            if item.finished or not self._accepting or self._closed:
                item.cancelled = True
                rejection = _service_unavailable_error()
            else:
                item.phase = "queued"
                self._start_deadline_watch(item)
                self._queue.append(item)
                self._queue_ready.set()
        if rejection is not None:
            await self._finish_item(item)
            raise rejection
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(_consume_future_exception)
            async with self._lock:
                item.delete_requested = True
                queued = item.phase == "queued" and item.task is None
                if queued:
                    item.cancelled = True
                    with suppress(ValueError):
                        self._queue.remove(item)
                    if not self._queue:
                        if self._accepting:
                            self._queue_ready.clear()
                        else:
                            self._queue_ready.set()
            if queued:
                if not future.done():
                    future.set_exception(
                        request_cancelled("Video generation was cancelled before dispatch")
                    )
                await self._finish_item(item)
            raise

    async def get_job(self, video_id: str) -> VideoJob:
        try:
            job = await asyncio.to_thread(self.jobs.get, video_id)
        except (TypeError, ValueError) as exc:
            raise DiffletServingError(404, "video_not_found", "Video job was not found") from exc
        if job is None:
            raise DiffletServingError(404, "video_not_found", "Video job was not found")
        return job

    async def list_jobs(self, *, limit: int = 20, after: str | None = None) -> VideoJobPage:
        return await asyncio.to_thread(self.jobs.list, limit=limit, after=after)

    async def open_content(self, video_id: str) -> tuple[VideoJob, VideoArtifactLease]:
        async with self._content_lock:
            job = await self.get_job(video_id)
            if job.status in {"queued", "in_progress"}:
                raise DiffletServingError(
                    409,
                    "video_not_ready",
                    "Video generation has not completed",
                )
            if job.status == "failed":
                raise DiffletServingError(
                    422,
                    "video_generation_failed",
                    "Video generation failed; inspect the job metadata for details",
                )
            if not job.artifact_key:
                raise DiffletServingError(
                    500,
                    "video_artifact_missing",
                    "Completed video metadata has no artifact",
                    "server_error",
                )
            try:
                lease = await self._open_artifact_lease(job.artifact_key)
            except Exception as exc:
                logger.exception("video.content_open_failed video_id=%s", video_id)
                raise DiffletServingError(
                    500,
                    "video_artifact_missing",
                    "Completed video artifact is unavailable",
                    "server_error",
                ) from exc
            return job, lease

    async def open_sync_result(self, result: VideoGenerationResult) -> VideoArtifactLease:
        """Open a sync artifact without leaking its descriptor on task cancellation."""

        return await self._open_artifact_lease(result.artifact.key)

    async def delete_job(self, video_id: str) -> VideoJob:
        job = await self.get_job(video_id)
        async with self._lock:
            item = self._items.get(video_id)
            if item is not None and not item.finished:
                if item.phase != "queued" or item.task is not None:
                    raise DiffletServingError(
                        409,
                        "video_in_progress",
                        "Video generation is already in progress",
                    )
                item.cancelled = True
                item.delete_requested = True
                with suppress(ValueError):
                    self._queue.remove(item)
                if not self._queue:
                    if self._accepting:
                        self._queue_ready.clear()
                    else:
                        self._queue_ready.set()
                deleted = self.jobs.delete(video_id, expected_statuses="queued")
            else:
                deleted = None
        if deleted is not None:
            assert item is not None
            await self._finish_item(item)
            return deleted

        mutation_task = asyncio.create_task(
            self._delete_persisted_job(video_id),
            name=f"difflet-video-delete-{video_id}",
        )
        try:
            return await asyncio.shield(mutation_task)
        except asyncio.CancelledError:
            # Job deletion cannot be rolled back by cancelling its parent task.
            # Complete the metadata/artifact transaction before propagating.
            with suppress(BaseException):
                await _await_task_fenced(mutation_task)
            raise

    async def _delete_persisted_job(self, video_id: str) -> VideoJob:
        async with self._content_lock:
            current = await _run_thread_fenced(self.jobs.get, video_id)
            if current is None:
                # A concurrent DELETE won the physical removal race.
                raise DiffletServingError(404, "video_not_found", "Video job was not found")
            # The metadata is process-local and cheap to remove, while artifact
            # unlink/fsync may fail. Delete the artifact first so a storage error
            # leaves a retrievable job and lets the client retry DELETE.
            if current.artifact_key:
                await self._delete_artifact_with_retries(current.artifact_key)
            deleted = await _run_thread_fenced(self.jobs.delete, video_id)
            if deleted is None:
                raise DiffletServingError(404, "video_not_found", "Video job was not found")
            await self._release_job_slot_count()
            if deleted.artifact_size_bytes:
                await self._release_retained_bytes(deleted.artifact_size_bytes)
            return deleted

    async def delete_sync_result(self, result: VideoGenerationResult) -> None:
        deleted = await asyncio.to_thread(self.artifacts.delete, result.artifact.key)
        if deleted:
            await self._release_retained_bytes(result.artifact.size_bytes)

    async def _delete_artifact_with_retries(self, artifact_key: str) -> None:
        for attempt in range(3):
            try:
                await _run_thread_fenced(self.artifacts.delete, artifact_key)
                return
            except Exception:
                if attempt == 2:
                    logger.critical(
                        "video.artifact_delete_failed artifact_key=%s",
                        artifact_key,
                        exc_info=True,
                    )
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    async def _open_artifact_lease(self, artifact_key: str) -> VideoArtifactLease:
        open_task = asyncio.create_task(
            asyncio.to_thread(self.artifacts.open, artifact_key),
            name=f"difflet-video-open-{artifact_key}",
        )
        try:
            return await asyncio.shield(open_task)
        except asyncio.CancelledError:
            # A local filesystem open dispatched to a thread cannot be cancelled.
            # Fence it and close any resulting descriptor before propagating the
            # caller's cancellation.
            lease: VideoArtifactLease | None = None
            try:
                lease = await _await_task_fenced(open_task)
            except Exception:
                pass
            if lease is not None:
                lease.close()
            raise

    async def _reserve(self, item: _VideoWorkItem) -> None:
        async with self._lock:
            if not self._accepting or self._closed:
                raise DiffletServingError(
                    503,
                    "video_service_unavailable",
                    "Video generation service is not accepting requests",
                    "server_error",
                )
            if self._pending >= self.capacity:
                raise DiffletServingError(
                    429,
                    "queue_full",
                    "Video generation queue is full",
                )
            if item.job_backed and self._job_slots_reserved >= self.max_jobs:
                raise DiffletServingError(
                    429,
                    "video_retention_full",
                    "Video job retention capacity is exhausted",
                )
            free_bytes = self.artifacts.available_bytes()
            needed = (
                self._active_reserved_bytes
                + self.artifacts.max_artifact_bytes
                + self.storage_margin_bytes
            )
            if free_bytes < needed:
                logger.error(
                    "video.storage_full free_bytes=%d active_reserved_bytes=%d "
                    "required_bytes=%d margin_bytes=%d",
                    free_bytes,
                    self._active_reserved_bytes,
                    needed,
                    self.storage_margin_bytes,
                )
                raise DiffletServingError(
                    507,
                    "video_storage_full",
                    "Video storage safety reserve is unavailable",
                    "server_error",
                )
            if free_bytes <= 2 * self.storage_margin_bytes:
                logger.warning(
                    "video.storage_pressure free_bytes=%d active_reserved_bytes=%d "
                    "margin_bytes=%d",
                    free_bytes,
                    self._active_reserved_bytes,
                    self.storage_margin_bytes,
                )
            if item.key in self._items:
                raise DiffletServingError(409, "video_id_conflict", "Video request ID exists")
            self._pending += 1
            self._items[item.key] = item
            if item.job_backed:
                item.job_slot_owned = True
                self._job_slots_reserved += 1
            item.storage_state = "reserved"
            item.accounted_bytes = self.artifacts.max_artifact_bytes
            self._active_reserved_bytes += item.accounted_bytes

    async def _dispatch_loop(self) -> None:
        while True:
            await self._queue_ready.wait()
            async with self._lock:
                if not self._queue:
                    self._queue_ready.clear()
                    if not self._accepting:
                        return
                    continue
                item = self._queue.popleft()
                if not self._queue:
                    if self._accepting:
                        self._queue_ready.clear()
                    else:
                        self._queue_ready.set()
                queue_expired = False
                if item.finished or item.cancelled:
                    claimed = False
                else:
                    queue_expired = (
                        time.monotonic() - item.enqueued_monotonic > self.queue_timeout_s
                    )
                    if queue_expired:
                        claimed = False
                    elif item.job_backed:
                        # This repository CAS and dequeue share the same lock as
                        # queued DELETE, which is their linearization point.
                        self.jobs.mark_in_progress(item.key)
                        item.phase = "preparing"
                        claimed = True
                    else:
                        item.phase = "preparing"
                        claimed = True
            if not claimed:
                if not item.finished and not item.cancelled and queue_expired:
                    await self._fail_queued_item(
                        item,
                        DiffletServingError(
                            429,
                            "queue_timeout",
                            "Video request waited too long in the generation queue",
                        ),
                    )
                    continue
                await self._finish_item(item)
                continue
            item.task = asyncio.create_task(
                self._execute_item(item), name=f"difflet-video-{item.key}"
            )
            try:
                await item.task
            except asyncio.CancelledError:
                # `_execute_item` performs the worker fence and cleanup.
                pass
            except BaseException:
                logger.exception("video.dispatch_unhandled request_id=%s", item.key)
            finally:
                item.task = None

    async def _execute_item(self, item: _VideoWorkItem) -> None:
        keep_artifact = False
        worker_output_fenced = True
        started = time.perf_counter()
        deadline = self._item_deadline(item)
        try:
            if item.cancelled:
                raise asyncio.CancelledError
            target = await asyncio.to_thread(self.artifacts.allocate_staging, item.key)
            item.target = target
            if item.cancelled:
                raise asyncio.CancelledError
            request = _with_output_target(item.request, target)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiffletServingError(
                    504,
                    "request_timeout",
                    "Video request timed out",
                    "server_error",
                )
            try:
                async with self._lock:
                    if item.cancelled:
                        raise asyncio.CancelledError
                    item.phase = "engine"
                worker_output_fenced = False
                output = await asyncio.wait_for(self.engine.generate(request), timeout=remaining)
            except asyncio.TimeoutError as exc:
                await self._wait_for_engine_fence()
                worker_output_fenced = True
                raise DiffletServingError(
                    504,
                    "request_timeout",
                    "Video request timed out",
                    "server_error",
                ) from exc
            except asyncio.CancelledError:
                raise
            except BaseException:
                # A non-cancellation engine result is terminal: the worker has
                # returned an error and can no longer write this target.  Still
                # wait for any resident-engine restart before draining backlog.
                await self._wait_for_engine_fence()
                worker_output_fenced = True
                raise
            worker_output_fenced = True
            async with self._lock:
                if item.cancelled:
                    raise asyncio.CancelledError
                if item.delete_requested and not item.job_backed:
                    raise asyncio.CancelledError
                item.phase = "publishing"
            if not isinstance(output, FileBackedGenerateOutput):
                raise TypeError("video engine must return FileBackedGenerateOutput")
            if output.mime_type != "video/mp4" or output.output_format != "mp4":
                raise ValueError("video engine returned an unsupported media format")
            actual_size = await asyncio.to_thread(
                self.artifacts.validate_staging,
                target,
                reported_path=output.path,
                expected_size_bytes=output.size_bytes,
            )
            media = await asyncio.to_thread(
                _validate_media,
                Path(output.path),
                item.request,
            )
            _compare_worker_media(output, media)
            async with self._content_lock:
                commit_fn = getattr(self.artifacts, "commit_local", None) if not item.job_backed else None
                if commit_fn is None:
                    commit_fn = self.artifacts.commit
                artifact = await asyncio.to_thread(
                    commit_fn,
                    target,
                    reported_path=output.path,
                    expected_size_bytes=actual_size,
                )
            item.artifact = artifact
            await self._commit_storage_reservation(item, artifact.size_bytes)
            inference_time = time.perf_counter() - started
            media_dict = _media_metadata_dict(media)
            media_dict.update(
                {
                    "inference_time_s": inference_time,
                    "file_name": artifact.key,
                }
            )
            artifact_url = getattr(self.artifacts, "url", lambda _key: None)(artifact.key)
            if artifact_url:
                media_dict["url"] = artifact_url
            result = VideoGenerationResult(
                request_id=item.request.request_id,
                artifact=artifact,
                media_metadata=media_dict,
                inference_time_s=inference_time,
            )
            # Claim terminal publication under a short lock. The repository CAS
            # stays outside this lock so DELETE can enter its bounded wait. Once
            # claimed, completion has won and DELETE removes the terminal job and
            # artifact afterward.
            async with self._lock:
                if item.cancelled:
                    if item.deadline_expired:
                        raise _request_timeout_error()
                    raise asyncio.CancelledError
                if item.delete_requested and not item.job_backed:
                    raise asyncio.CancelledError
                if time.monotonic() >= deadline:
                    item.cancelled = True
                    item.deadline_expired = True
                    raise _request_timeout_error()
                item.publication_claimed = True
                item.phase = "terminal"
            if item.job_backed:
                await asyncio.to_thread(
                    self.jobs.mark_completed,
                    item.key,
                    artifact_key=artifact.key,
                    artifact_size_bytes=artifact.size_bytes,
                    media_metadata=media_dict,
                    expires_at=int(time.time()) + self.retention_seconds,
                )
            elif item.future is not None and not item.future.done():
                item.future.set_result(result)
            keep_artifact = True
        except asyncio.CancelledError:
            if not worker_output_fenced:
                await self._wait_for_engine_fence()
                worker_output_fenced = True
            if item.job_backed and not item.delete_requested:
                await self._mark_failed(
                    item.key,
                    (
                        _timeout_job_error()
                        if item.deadline_expired
                        else VideoJobError(
                            code="request_cancelled",
                            message="Video generation was cancelled",
                        )
                    ),
                )
            if item.future is not None and not item.future.done():
                item.future.set_exception(
                    _request_timeout_error()
                    if item.deadline_expired
                    else request_cancelled("Video generation was cancelled")
                )
            raise
        except BaseException as exc:
            error = _public_job_error(exc)
            logger.exception("video.generation_failed request_id=%s code=%s", item.key, error.code)
            if item.job_backed and not item.delete_requested:
                await self._mark_failed(item.key, error)
            if item.future is not None and not item.future.done():
                item.future.set_exception(_public_exception(exc))
        finally:
            if not keep_artifact and worker_output_fenced:
                if item.artifact is not None:
                    with suppress(Exception):
                        deleted = await asyncio.to_thread(self.artifacts.delete, item.artifact.key)
                        if deleted:
                            await self._release_item_storage(item)
                if item.target is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(self.artifacts.delete_staging, item.target)
            elif not worker_output_fenced:
                logger.critical(
                    "video.staging_quarantined_without_worker_fence request_id=%s",
                    item.key,
                )
            await self._finish_item(item)

    async def _cancel_item(
        self,
        item: _VideoWorkItem,
        *,
        delete_requested: bool,
    ) -> None:
        async with self._lock:
            completion_won = item.publication_claimed
            if not completion_won:
                item.cancelled = True
            item.delete_requested = item.delete_requested or delete_requested
            task = item.task
            create_task = item.create_task
            phase = item.phase
            finished = item.finished
        if finished:
            await self._delete_abandoned_sync_artifact(item, delete_requested=delete_requested)
            return
        if create_task is not None:
            # Repository creation already dispatched to a thread must finish
            # before shutdown clears the store. The submitting coroutine observes
            # `cancelled` and cannot enqueue after this point.
            with suppress(BaseException):
                await _await_task_fenced(create_task)
        if task is None:
            if item.job_backed and not item.delete_requested:
                await self._mark_failed(
                    item.key,
                    (
                        _timeout_job_error()
                        if item.deadline_expired
                        else VideoJobError(
                            code="server_shutdown",
                            message=(
                                "Video generation stopped because the server is shutting down"
                            ),
                        )
                    ),
                )
            if item.future is not None and not item.future.done():
                item.future.set_exception(request_cancelled("Video generation was cancelled"))
            await self._finish_item(item)
            if create_task is not None:
                await item.submission_done.wait()
            return
        if phase == "engine" and not completion_won:
            task.cancel()
            with suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(task), timeout=self.recovery_timeout_s)
        else:
            # Parent-side repository/filesystem work runs in threads and cannot
            # be force-cancelled safely. Let that small critical section finish;
            # `_execute_item` observes `cancelled` at each boundary.
            with suppress(BaseException):
                await asyncio.wait_for(item.done.wait(), timeout=self.recovery_timeout_s)
        if item.done.is_set():
            await self._delete_abandoned_sync_artifact(item, delete_requested=delete_requested)

    async def _fail_queued_item(
        self,
        item: _VideoWorkItem,
        error: DiffletServingError,
    ) -> None:
        if item.job_backed:
            await self._mark_failed(
                item.key,
                VideoJobError(
                    code=error.code,
                    message=error.message,
                    error_type=error.error_type,
                ),
            )
        if item.future is not None and not item.future.done():
            item.future.set_exception(error)
        await self._finish_item(item)

    async def _wait_for_engine_fence(self) -> None:
        wait_for_recovery = getattr(self.engine, "wait_for_recovery", None)
        if wait_for_recovery is None:
            return
        try:
            # API cancellation waits are bounded separately.  Staging cleanup is
            # not: wait until the old worker acknowledges cancellation or has
            # been terminated/restarted.
            await wait_for_recovery(timeout=None)
        except DiffletServingError:
            # `wait_for_recovery` raises after the recovery fence when the worker
            # is unrecoverable.  Cleanup is still safe because the old worker is
            # no longer allowed to write the parent-owned target.
            logger.exception("video.worker_recovery_failed")

    def _start_deadline_watch(self, item: _VideoWorkItem) -> None:
        item.deadline_task = asyncio.create_task(
            self._deadline_watch(item),
            name=f"difflet-video-deadline-{item.key}",
        )

    async def _deadline_watch(self, item: _VideoWorkItem) -> None:
        deadline = self._item_deadline(item)
        try:
            await asyncio.sleep(max(deadline - time.monotonic(), 0.0))
        except asyncio.CancelledError:
            return

        async with self._lock:
            if item.finished or item.publication_claimed:
                return
            item.deadline_expired = True
            item.cancelled = True
            task = item.task
            phase = item.phase

        # Stop device work before any potentially blocking metadata update.
        # The execution task owns the recovery fence and staging cleanup.
        if task is not None and phase == "engine":
            task.cancel()

        error = _request_timeout_error()
        if item.job_backed:
            try:
                await self._mark_failed(item.key, _timeout_job_error())
            except Exception:
                logger.critical(
                    "video.deadline_persistence_failed request_id=%s",
                    item.key,
                    exc_info=True,
                )
        if item.future is not None and not item.future.done():
            item.future.set_exception(error)
        if task is None:
            await self._finish_item(item)

    async def _delete_abandoned_sync_artifact(
        self,
        item: _VideoWorkItem,
        *,
        delete_requested: bool,
    ) -> None:
        if item.job_backed or not delete_requested or item.artifact is None:
            return
        with suppress(Exception):
            await asyncio.to_thread(self.artifacts.delete, item.artifact.key)

    async def _mark_failed(self, video_id: str, error: VideoJobError) -> None:
        for attempt in range(3):
            try:
                try:
                    await _run_thread_fenced(
                        self.jobs.mark_failed,
                        video_id,
                        error=error,
                        expires_at=int(time.time()) + self.retention_seconds,
                    )
                except TypeError as exc:
                    if "expires_at" not in str(exc):
                        raise
                    # Compatibility for small repository test doubles written
                    # before terminal expiry became part of the contract.
                    await _run_thread_fenced(self.jobs.mark_failed, video_id, error=error)
                return
            except (VideoJobNotFound, VideoJobStateConflict, InvalidVideoJobTransition):
                # DELETE or another terminal CAS may have won the race.
                logger.info("video.mark_failed_skipped video_id=%s", video_id)
                return
            except Exception:
                if attempt == 2:
                    logger.critical(
                        "video.mark_failed_persistence_error video_id=%s",
                        video_id,
                        exc_info=True,
                    )
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    async def _finish_item(self, item: _VideoWorkItem) -> None:
        deadline_task: asyncio.Task[None] | None = None
        owns_finish = False
        async with self._lock:
            if not item.finished:
                owns_finish = True
                item.finished = True
                if self._items.get(item.key) is item:
                    self._items.pop(item.key, None)
                with suppress(ValueError):
                    self._queue.remove(item)
                if not self._queue:
                    if self._accepting:
                        self._queue_ready.clear()
                    else:
                        self._queue_ready.set()
                self._pending = max(self._pending - 1, 0)
                if item.storage_state == "reserved":
                    self._active_reserved_bytes = max(
                        self._active_reserved_bytes - item.accounted_bytes, 0
                    )
                    item.storage_state = "released"
                    item.accounted_bytes = 0
                if item.job_slot_owned and self.jobs.get(item.key) is None:
                    self._job_slots_reserved = max(self._job_slots_reserved - 1, 0)
                    item.job_slot_owned = False
                item.phase = "done"
                deadline_task = item.deadline_task
                item.deadline_task = None
        if not owns_finish:
            await item.done.wait()
            return
        if deadline_task is not None and deadline_task is not asyncio.current_task():
            deadline_task.cancel()
            with suppress(BaseException):
                await asyncio.shield(deadline_task)
        item.done.set()

    def _item_deadline(self, item: _VideoWorkItem) -> float:
        return (
            item.deadline_monotonic
            if item.deadline_monotonic is not None
            else item.enqueued_monotonic + self.request_timeout_s
        )

    async def _commit_storage_reservation(
        self,
        item: _VideoWorkItem,
        actual_bytes: int,
    ) -> None:
        async with self._lock:
            if item.storage_state != "reserved":
                raise RuntimeError("video storage reservation is not active")
            self._active_reserved_bytes = max(self._active_reserved_bytes - item.accounted_bytes, 0)
            self._retained_bytes += actual_bytes
            item.storage_state = "retained"
            item.accounted_bytes = actual_bytes

    async def _release_item_storage(self, item: _VideoWorkItem) -> None:
        async with self._lock:
            if item.storage_state == "reserved":
                self._active_reserved_bytes = max(
                    self._active_reserved_bytes - item.accounted_bytes, 0
                )
            elif item.storage_state == "retained":
                self._retained_bytes = max(self._retained_bytes - item.accounted_bytes, 0)
            item.storage_state = "released"
            item.accounted_bytes = 0

    async def _release_retained_bytes(self, size_bytes: int) -> None:
        async with self._lock:
            self._retained_bytes = max(self._retained_bytes - int(size_bytes), 0)

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.sweep_interval_s)
                await self.sweep_expired()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("video.retention_sweep_failed")

    async def sweep_expired(self, *, now: int | None = None) -> int:
        async with self._content_lock:
            expired = await _run_thread_fenced(self.jobs.list_expired, now=now)
            removed = 0
            for job in expired:
                try:
                    if job.artifact_key:
                        await self._delete_artifact_with_retries(job.artifact_key)
                    deleted = await _run_thread_fenced(
                        self.jobs.delete,
                        job.id,
                        expected_statuses=("completed", "failed"),
                    )
                except Exception:
                    logger.exception("video.retention_delete_failed video_id=%s", job.id)
                    continue
                if deleted is not None:
                    removed += 1
                    await self._release_job_slot_count()
                    if deleted.artifact_size_bytes:
                        await self._release_retained_bytes(deleted.artifact_size_bytes)
            if removed:
                logger.info("video.retention_sweep removed_jobs=%d", removed)
            return removed

    async def _release_job_slot_count(self) -> None:
        async with self._lock:
            self._job_slots_reserved = max(self._job_slots_reserved - 1, 0)

    @staticmethod
    def _require_video_request(request: DiffletGenerateRequest) -> None:
        if request.video is None or request.output_format != "mp4":
            raise TypeError("VideoGenerationService requires a normalized MP4 request")


def _with_output_target(
    request: DiffletGenerateRequest,
    target: VideoArtifactTarget,
) -> DiffletGenerateRequest:
    assert request.video is not None
    return replace(
        request,
        video=replace(
            request.video,
            output_target=FileOutputTarget(staging_path=str(target.staging_path)),
        ),
    )


def _job_request_metadata(request: DiffletGenerateRequest) -> dict[str, Any]:
    assert request.video is not None
    return {
        "width": request.width,
        "height": request.height,
        "num_frames": request.video.num_frames,
        "fps": request.video.fps,
        "seconds": request.video.requested_seconds,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "guidance_scale_2": request.video.guidance_scale_2,
        "boundary_ratio": request.video.boundary_ratio,
        "flow_shift": request.video.flow_shift,
        "negative_prompt": request.video.negative_prompt,
        "seed": request.seed,
    }


def _validate_media(path: Path, request: DiffletGenerateRequest):
    from difflet.serving.video_media import validate_mp4

    assert request.video is not None
    return validate_mp4(
        path,
        expected_width=request.width,
        expected_height=request.height,
        expected_num_frames=request.video.num_frames,
        expected_fps=request.video.fps,
    )


def _media_metadata_dict(media: Any) -> dict[str, Any]:
    if hasattr(media, "__dataclass_fields__"):
        return asdict(media)
    if isinstance(media, dict):
        return dict(media)
    return {
        name: getattr(media, name)
        for name in ("width", "height", "num_frames", "fps", "duration_s")
    }


def _compare_worker_media(output: FileBackedGenerateOutput, media: Any) -> None:
    actual = _media_metadata_dict(media)
    expected = {
        "width": output.width,
        "height": output.height,
        "num_frames": output.num_frames,
    }
    for name, value in expected.items():
        if int(actual[name]) != int(value):
            raise ValueError(f"worker-reported video {name} does not match encoded MP4")
    if abs(float(actual["fps"]) - float(output.fps)) > 1e-6:
        raise ValueError("worker-reported video fps does not match encoded MP4")
    if abs(float(actual["duration_s"]) - float(output.duration_s)) > 1e-6:
        raise ValueError("worker-reported video duration does not match encoded MP4")


def _public_job_error(exc: BaseException) -> VideoJobError:
    if isinstance(exc, DiffletServingError):
        return VideoJobError(code=exc.code, message=exc.message, error_type=exc.error_type)
    return VideoJobError(
        code="internal_error",
        message="Internal model execution error",
        error_type="server_error",
    )


def _public_exception(exc: BaseException) -> DiffletServingError:
    if isinstance(exc, DiffletServingError):
        return exc
    return internal_error()


def _request_timeout_error() -> DiffletServingError:
    return DiffletServingError(
        504,
        "request_timeout",
        "Video request timed out",
        "server_error",
    )


def _service_unavailable_error() -> DiffletServingError:
    return DiffletServingError(
        503,
        "video_service_unavailable",
        "Video generation service is not accepting requests",
        "server_error",
    )


def _timeout_job_error() -> VideoJobError:
    error = _request_timeout_error()
    return VideoJobError(
        code=error.code,
        message=error.message,
        error_type=error.error_type,
    )


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    """Drain a sync result abandoned by a disconnected/cancelled caller."""

    if future.cancelled():
        return
    with suppress(BaseException):
        future.exception()


async def _await_task_fenced(task: asyncio.Task[Any]) -> Any:
    """Wait for non-cancellable thread-backed work despite repeated task cancellation."""

    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _run_thread_fenced(function, /, *args, **kwargs):
    """Run local blocking work and never leave its thread mutating after cancellation."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(BaseException):
            await _await_task_fenced(task)
        raise


__all__ = ["VideoGenerationResult", "VideoGenerationService"]
