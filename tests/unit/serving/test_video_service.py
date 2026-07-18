from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from difflet.serving.errors import DiffletServingError
from difflet.serving.types import (
    DiffletGenerateRequest,
    FileBackedGenerateOutput,
    VideoGenerateOptions,
)
from difflet.serving.video_jobs import (
    InMemoryVideoJobRepository,
    VideoJobCreate,
    VideoJobRepositoryError,
)
from difflet.serving.video_service import VideoGenerationService
from difflet.serving.video_storage import LocalVideoArtifactStore, S3VideoArtifactStore


def _request(request_id: str, *, prompt: str = "a tiny video") -> DiffletGenerateRequest:
    return DiffletGenerateRequest(
        request_id=request_id,
        model="test/video",
        prompt=prompt,
        height=16,
        width=16,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
        output_format="mp4",
        video=VideoGenerateOptions(num_frames=2, fps=2, user="caller-1"),
    )


def _write_tiny_mp4(path: Path, request: DiffletGenerateRequest) -> bytes:
    """Write a two-frame PyAV MP4, with a dependency-light local fallback.

    PyAV is a runtime dependency of Difflet, but the lightweight developer test
    environment used for repository-only checks may omit it.  In that environment
    the tests patch only the media decoder boundary while retaining the real
    staging, commit, lease, and deletion paths.
    """

    assert request.video is not None
    if importlib.util.find_spec("av") is None:
        payload = f"test-mp4:{request.request_id}".encode()
        path.write_bytes(payload)
        return payload

    import av

    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=request.video.fps)
        stream.width = request.width
        stream.height = request.height
        stream.pix_fmt = "yuv420p"
        for index in range(request.video.num_frames):
            frame = av.VideoFrame(request.width, request.height, "yuv420p")
            for plane_index, plane in enumerate(frame.planes):
                level = 16 + index if plane_index == 0 else 128
                plane.update(bytes([level]) * plane.buffer_size)
            frame.pts = index
            frame.time_base = Fraction(1, request.video.fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


def _patch_media_validation_without_pyav(monkeypatch) -> None:
    if importlib.util.find_spec("av") is not None:
        return

    import difflet.serving.video_service as video_service_module
    from difflet.serving.video_media import VideoMediaMetadata

    def _validate(path: Path, request: DiffletGenerateRequest) -> VideoMediaMetadata:
        assert request.video is not None
        return VideoMediaMetadata(
            width=request.width,
            height=request.height,
            num_frames=request.video.num_frames,
            fps=float(request.video.fps),
            duration_s=request.video.num_frames / request.video.fps,
            codec_name="test-fallback",
            pixel_format="yuv420p",
            size_bytes=path.stat().st_size,
        )

    monkeypatch.setattr(video_service_module, "_validate_media", _validate)


class _FakeVideoEngine:
    healthy = True
    ready = True

    def __init__(
        self,
        *,
        blocked_ids: set[str] | None = None,
        failing_prompts: set[str] | None = None,
    ) -> None:
        self.blocked_ids = blocked_ids or set()
        self.failing_prompts = failing_prompts or set()
        self.release = asyncio.Event()
        self.started = defaultdict(asyncio.Event)
        self.calls: list[str] = []
        self.cancelled: set[str] = set()
        self.payloads: dict[str, bytes] = {}
        self.recovery_waits = 0
        self.start_calls = 0
        self.shutdown_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def generate(self, request: DiffletGenerateRequest) -> FileBackedGenerateOutput:
        assert request.video is not None
        assert request.video.output_target is not None
        self.calls.append(request.request_id)
        self.started[request.request_id].set()
        try:
            if request.request_id in self.blocked_ids:
                await self.release.wait()
            if request.prompt in self.failing_prompts:
                raise RuntimeError("private fake-engine failure")
            path = Path(request.video.output_target.staging_path)
            payload = _write_tiny_mp4(path, request)
            self.payloads[request.request_id] = payload
            return FileBackedGenerateOutput(
                path=str(path),
                mime_type="video/mp4",
                output_format="mp4",
                size_bytes=len(payload),
                width=request.width,
                height=request.height,
                num_frames=request.video.num_frames,
                fps=float(request.video.fps),
                duration_s=request.video.num_frames / request.video.fps,
            )
        except asyncio.CancelledError:
            self.cancelled.add(request.request_id)
            raise

    async def wait_for_recovery(self, *, timeout: float | None) -> None:
        assert timeout is None
        self.recovery_waits += 1


class _BlockingValidationStore(LocalVideoArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.validation_entered = threading.Event()
        self.release_validation = threading.Event()
        self.validation_calls = 0

    def validate_staging(self, *args, **kwargs):
        self.validation_calls += 1
        if self.validation_calls == 1:
            self.validation_entered.set()
            assert self.release_validation.wait(timeout=3.0)
        return super().validate_staging(*args, **kwargs)


class _BlockingCommitStore(LocalVideoArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.commit_entered = threading.Event()
        self.release_commit = threading.Event()

    def commit(self, *args, **kwargs):
        self.commit_entered.set()
        assert self.release_commit.wait(timeout=3.0)
        return super().commit(*args, **kwargs)


class _BlockingFailureRepository(InMemoryVideoJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.failure_entered = threading.Event()
        self.release_failure = threading.Event()
        self._blocked_once = False

    def mark_failed(
        self, video_id, *, error, expected_statuses=("queued", "in_progress"), now=None
    ):
        if error.code == "request_timeout" and not self._blocked_once:
            self._blocked_once = True
            self.failure_entered.set()
            assert self.release_failure.wait(timeout=3.0)
        return super().mark_failed(
            video_id,
            error=error,
            expected_statuses=expected_statuses,
            now=now,
        )


class _BlockAfterFailureRepository(InMemoryVideoJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.failed_persisted = threading.Event()
        self.release_failure_return = threading.Event()

    def mark_failed(
        self, video_id, *, error, expected_statuses=("queued", "in_progress"), now=None
    ):
        failed = super().mark_failed(
            video_id,
            error=error,
            expected_statuses=expected_statuses,
            now=now,
        )
        self.failed_persisted.set()
        assert self.release_failure_return.wait(timeout=3.0)
        return failed


class _BlockingCreateReturnRepository(InMemoryVideoJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.committed = threading.Event()
        self.release_return = threading.Event()

    def create(self, create: VideoJobCreate):
        job = super().create(create)
        self.committed.set()
        assert self.release_return.wait(timeout=3.0)
        return job


class _FailCompletionRepository(InMemoryVideoJobRepository):
    def mark_completed(self, *args, **kwargs):
        raise OSError("completion persistence unavailable")


class _FailOnceDeleteStore(LocalVideoArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.delete_calls = 0

    def delete(self, artifact_key: str) -> bool:
        self.delete_calls += 1
        if self.delete_calls == 1:
            raise OSError("simulated transient unlink failure")
        return super().delete(artifact_key)


class _ControlledDeleteStore(LocalVideoArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.delete_calls = 0
        self.fail_deletes = True

    def delete(self, artifact_key: str) -> bool:
        self.delete_calls += 1
        if self.fail_deletes:
            raise OSError("simulated persistent unlink failure")
        return super().delete(artifact_key)


class _PendingCleanupStore(LocalVideoArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.retry_calls = 0

    def retry_pending_remote_deletes(self) -> int:
        self.retry_calls += 1
        return 1


class _BlockingS3Client:
    def __init__(self) -> None:
        self.upload_started = threading.Event()
        self.release_upload = threading.Event()
        self.deletes: list[tuple[str, str]] = []

    def upload_file(self, filename, bucket, key, *, ExtraArgs):
        assert Path(filename).is_file()
        assert ExtraArgs == {"ContentType": "video/mp4"}
        self.upload_started.set()
        assert self.release_upload.wait(timeout=3.0)

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        assert operation == "get_object"
        return f"https://example.test/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def delete_object(self, *, Bucket, Key):
        self.deletes.append((Bucket, Key))


class _BlockingS3Store:
    bucket = "video-bucket"
    prefix = "difflet"

    def __init__(self, client: _BlockingS3Client) -> None:
        self.client = client

    def _client(self):
        return self.client


class _DeleteRetryS3Client:
    def __init__(self) -> None:
        self.delete_errors: list[Exception] = []
        self.deletes: list[tuple[str, str]] = []

    def upload_file(self, filename, bucket, key, *, ExtraArgs):
        assert Path(filename).is_file()
        assert ExtraArgs == {"ContentType": "video/mp4"}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        assert operation == "get_object"
        return f"https://example.test/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def delete_object(self, *, Bucket, Key):
        self.deletes.append((Bucket, Key))
        if self.delete_errors:
            raise self.delete_errors.pop(0)


def _service(tmp_path, engine: _FakeVideoEngine, *, max_queued_requests: int = 2):
    jobs = InMemoryVideoJobRepository()
    artifacts = LocalVideoArtifactStore(tmp_path / "media")
    service = VideoGenerationService(
        engine=engine,
        jobs=jobs,
        artifacts=artifacts,
        max_queued_requests=max_queued_requests,
        queue_timeout_s=2.0,
        request_timeout_s=2.0,
        recovery_timeout_s=1.0,
    )
    return service, jobs, artifacts


async def _wait_for_status(
    service: VideoGenerationService,
    video_id: str,
    expected: str,
):
    deadline = asyncio.get_running_loop().time() + 3.0
    while True:
        job = await service.get_job(video_id)
        if job.status == expected:
            return job
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"{video_id} remained {job.status!r}; expected {expected!r}")
        await asyncio.sleep(0.01)


async def _wait_for_reserved(service: VideoGenerationService, request_id: str) -> None:
    deadline = asyncio.get_running_loop().time() + 3.0
    while request_id not in service._items:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"{request_id} was not admitted")
        await asyncio.sleep(0)


def test_sync_and_async_share_one_capacity_domain(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_async", "video_sync"})
        service, jobs, artifacts = _service(tmp_path, engine, max_queued_requests=1)
        await service.start()
        try:
            created = await service.create_async(_request("video_async"))
            assert created.status == "queued"
            await engine.started["video_async"].wait()

            sync_task = asyncio.create_task(service.generate_sync(_request("video_sync")))
            await _wait_for_reserved(service, "video_sync")

            with pytest.raises(DiffletServingError) as rejected:
                await service.create_async(_request("video_over_capacity"))
            assert (rejected.value.status_code, rejected.value.code) == (429, "queue_full")

            engine.release.set()
            sync_result = await sync_task
            completed = await _wait_for_status(service, "video_async", "completed")

            assert jobs.count() == 1
            assert completed.artifact_key == "video_async.mp4"
            assert artifacts.get(sync_result.artifact.key) is not None
            await service.delete_sync_result(sync_result)
            assert artifacts.get(sync_result.artifact.key) is None
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_local_sync_cleanup_failure_is_deferred_and_releases_accounting(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_sync_local_orphan"
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _ControlledDeleteStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            result = await service.generate_sync(_request(request_id))
            assert service._retained_bytes == result.artifact.size_bytes

            await service.delete_sync_result(result)
            assert artifacts.delete_calls == 3
            assert artifacts.get(result.artifact.key) is not None
            assert service._retained_bytes == result.artifact.size_bytes

            artifacts.fail_deletes = False
            assert await service.sweep_expired(now=0) == 0
            assert artifacts.get(result.artifact.key) is None
            assert service._retained_bytes == 0
            assert artifacts.retry_pending_artifact_deletes() == ()
        finally:
            artifacts.fail_deletes = False
            await service.shutdown()

    asyncio.run(_run())


def test_queue_timeout_uses_async_job_and_sync_http_error_channels(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_blocker"})
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=2,
            queue_timeout_s=0.02,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request("video_blocker"))
            await engine.started["video_blocker"].wait()

            queued = await service.create_async(_request("video_async_queue_timeout"))
            assert queued.status == "queued"
            sync_task = asyncio.create_task(
                service.generate_sync(_request("video_sync_queue_timeout"))
            )
            await _wait_for_reserved(service, "video_sync_queue_timeout")

            await asyncio.sleep(0.05)
            engine.release.set()

            failed = await _wait_for_status(
                service,
                "video_async_queue_timeout",
                "failed",
            )
            assert failed.error is not None
            assert failed.error.code == "queue_timeout"
            assert failed.error.error_type == "invalid_request_error"

            with pytest.raises(DiffletServingError) as sync_timeout:
                await sync_task
            assert (sync_timeout.value.status_code, sync_timeout.value.code) == (
                429,
                "queue_timeout",
            )
        finally:
            engine.release.set()
            await service.shutdown()

    asyncio.run(_run())


def test_queued_video_uses_queue_timeout_before_execution_timeout(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_queue_blocker"})
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=0.5,
            request_timeout_s=0.02,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(
                _request("video_queue_blocker"),
                deadline=time.monotonic() + 1.0,
            )
            await engine.started["video_queue_blocker"].wait()

            queued = await service.create_async(_request("video_queue_survivor"))
            assert queued.status == "queued"
            await asyncio.sleep(0.05)
            assert jobs.require("video_queue_survivor").status == "queued"

            engine.release.set()
            completed = await _wait_for_status(
                service,
                "video_queue_survivor",
                "completed",
            )
            assert completed.error is None
        finally:
            engine.release.set()
            await service.shutdown()

    asyncio.run(_run())


def test_async_jobs_transition_from_queued_to_completed_or_sanitized_failed(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(failing_prompts={"fail"})
        service, _, artifacts = _service(tmp_path, engine)
        await service.start()
        try:
            completed_create = await service.create_async(_request("video_ok"))
            failed_create = await service.create_async(_request("video_fail", prompt="fail"))
            assert completed_create.status == failed_create.status == "queued"

            completed = await _wait_for_status(service, "video_ok", "completed")
            failed = await _wait_for_status(service, "video_fail", "failed")

            assert completed.started_at is not None
            assert completed.completed_at is not None
            assert completed.artifact_key == "video_ok.mp4"
            assert completed.media_metadata is not None
            assert completed.media_metadata["num_frames"] == 2
            assert artifacts.require(completed.artifact_key).size_bytes > 0

            assert failed.error is not None
            assert failed.error.code == "internal_error"
            assert failed.error.message == "Internal model execution error"
            assert "private" not in failed.error.message
            assert failed.artifact_key is None
            assert tuple(artifacts.staging_root.iterdir()) == ()
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_sync_generation_creates_no_job_row_and_explicit_cleanup_removes_artifact(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        service, jobs, artifacts = _service(tmp_path, engine)
        await service.start()
        try:
            result = await service.generate_sync(_request("video_sync_only"))

            assert jobs.count() == 0
            assert artifacts.require(result.artifact.key).size_bytes == result.artifact.size_bytes
            with artifacts.open(result.artifact.key) as lease:
                assert (
                    b"".join(lease.iter_chunks(chunk_size=7)) == engine.payloads["video_sync_only"]
                )

            await service.delete_sync_result(result)
            assert artifacts.get(result.artifact.key) is None
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_sync_caller_cancellation_only_removes_work_before_dispatch(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run_queued() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_blocker"})
        service, _, artifacts = _service(tmp_path / "queued", engine)
        await service.start()
        try:
            await service.create_async(_request("video_blocker"))
            await engine.started["video_blocker"].wait()
            sync = asyncio.create_task(service.generate_sync(_request("video_sync_queued")))
            await _wait_for_reserved(service, "video_sync_queued")
            while service._items["video_sync_queued"].phase != "queued":
                await asyncio.sleep(0)
            sync.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sync
            assert "video_sync_queued" not in engine.calls
            assert "video_sync_queued" not in service._items
            assert artifacts.get("video_sync_queued.mp4") is None
        finally:
            engine.release.set()
            await service.shutdown()

    async def _run_started() -> None:
        request_id = "video_sync_started"
        engine = _FakeVideoEngine(blocked_ids={request_id})
        service, jobs, artifacts = _service(tmp_path / "started", engine)
        await service.start()
        try:
            sync = asyncio.create_task(service.generate_sync(_request(request_id)))
            await engine.started[request_id].wait()
            item = service._items[request_id]
            sync.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sync
            assert request_id not in engine.cancelled
            assert item.done.is_set() is False

            engine.release.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)
            assert request_id not in engine.cancelled
            assert jobs.count() == 0
            assert artifacts.get(f"{request_id}.mp4") is None
        finally:
            engine.release.set()
            await service.shutdown()

    asyncio.run(_run_queued())
    asyncio.run(_run_started())


def test_shutdown_drops_completed_jobs_and_outputs(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_ephemeral_lifecycle"
        first, first_jobs, first_artifacts = _service(tmp_path, _FakeVideoEngine())
        await first.start()
        await first.create_async(_request(request_id))
        completed = await _wait_for_status(first, request_id, "completed")
        assert first_jobs.require(request_id) == completed
        assert first_artifacts.get(f"{request_id}.mp4") is not None

        await first.shutdown()

        with pytest.raises(VideoJobRepositoryError):
            first_jobs.count()
        assert tuple(first_artifacts.artifact_root.iterdir()) == ()
        assert tuple(first_artifacts.staging_root.iterdir()) == ()

        second, second_jobs, second_artifacts = _service(tmp_path, _FakeVideoEngine())
        await second.start()
        try:
            assert second_jobs.count() == 0
            assert second_artifacts.get(f"{request_id}.mp4") is None
        finally:
            await second.shutdown()

    asyncio.run(_run())


def test_start_discards_preexisting_jobs_and_orphan_artifacts(tmp_path):
    async def _run() -> None:
        engine = _FakeVideoEngine()
        service, jobs, artifacts = _service(tmp_path, engine)
        for video_id in ("video_stale_queued", "video_stale_running"):
            jobs.create(
                VideoJobCreate(
                    id=video_id,
                    model="test/video",
                    prompt="stale",
                    request={"width": 16, "height": 16},
                )
            )
        jobs.mark_in_progress("video_stale_running")
        orphan = artifacts.artifact_root / "video_orphan.mp4"
        orphan.write_bytes(b"orphan")

        await service.start()
        try:
            assert jobs.count() == 0
            assert orphan.exists() is False
            assert engine.calls == []
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_second_service_cannot_sweep_an_active_shared_media_root(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_owned_by_first_service"
        first_engine = _FakeVideoEngine(blocked_ids={request_id})
        first, first_jobs, first_artifacts = _service(tmp_path, first_engine)
        await first.start()

        second_engine = _FakeVideoEngine()
        second_jobs = InMemoryVideoJobRepository()
        second_artifacts = LocalVideoArtifactStore(tmp_path / "media")
        second = VideoGenerationService(
            engine=second_engine,
            jobs=second_jobs,
            artifacts=second_artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        try:
            await first.create_async(_request(request_id))
            await first_engine.started[request_id].wait()
            item = first._items[request_id]
            assert item.target is not None
            staging_path = item.target.staging_path

            with pytest.raises(RuntimeError, match="already owned"):
                await second.start()
            await second.shutdown()

            assert first_jobs.require(request_id).status == "in_progress"
            assert staging_path.exists()
            assert first._pending == 1
        finally:
            await second.shutdown()
            await first.shutdown()

    asyncio.run(_run())


def test_delete_removes_queued_work_and_rejects_running_work(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_running", "video_queued"})
        service, jobs, artifacts = _service(tmp_path, engine, max_queued_requests=1)
        await service.start()
        try:
            await service.create_async(_request("video_running"))
            await engine.started["video_running"].wait()
            await service.create_async(_request("video_queued"))
            assert jobs.require("video_queued").status == "queued"

            deleted_queued = await service.delete_job("video_queued")
            assert deleted_queued.status == "queued"
            assert "video_queued" not in engine.calls

            with pytest.raises(DiffletServingError) as running:
                await service.delete_job("video_running")
            assert (running.value.status_code, running.value.code) == (
                409,
                "video_in_progress",
            )
            assert "video_running" not in engine.cancelled
            assert engine.recovery_waits == 0

            engine.release.set()
            await _wait_for_status(service, "video_running", "completed")
            deleted_running = await service.delete_job("video_running")
            assert deleted_running.status == "completed"
            assert jobs.count() == 0
            assert tuple(artifacts.staging_root.iterdir()) == ()
            assert tuple(artifacts.artifact_root.iterdir()) == ()
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_delete_during_publication_returns_conflict_and_generation_completes(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _BlockingValidationStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=0.02,
        )
        await service.start()
        try:
            await service.create_async(_request("video_delete_publish_race"))
            assert await asyncio.to_thread(artifacts.validation_entered.wait, 3.0)
            item = service._items["video_delete_publish_race"]

            with pytest.raises(DiffletServingError) as deleting:
                await asyncio.wait_for(service.delete_job("video_delete_publish_race"), timeout=1.0)
            assert (deleting.value.status_code, deleting.value.code) == (
                409,
                "video_in_progress",
            )
            assert item.cancelled is False

            artifacts.release_validation.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)

            assert jobs.require("video_delete_publish_race").status == "completed"
            assert artifacts.get("video_delete_publish_race.mp4") is not None
            assert tuple(artifacts.staging_root.iterdir()) == ()

            deleted = await service.delete_job("video_delete_publish_race")
            assert deleted.status == "completed"
            assert jobs.get("video_delete_publish_race") is None
        finally:
            artifacts.release_validation.set()
            await service.shutdown()

    asyncio.run(_run())


def test_delete_returns_409_but_preserves_staging_until_recovery_fence(
    tmp_path,
):
    class _DelayedFenceEngine:
        healthy = True
        ready = True

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.fence_wait_started = asyncio.Event()
            self.release_fence = asyncio.Event()
            self.fenced = False

        async def generate(
            self,
            request: DiffletGenerateRequest,
        ) -> FileBackedGenerateOutput:
            assert request.video is not None
            assert request.video.output_target is not None
            path = Path(request.video.output_target.staging_path)
            payload = _write_tiny_mp4(path, request)
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def wait_for_recovery(self, *, timeout: float | None) -> None:
            assert timeout is None
            self.fence_wait_started.set()
            await self.release_fence.wait()
            self.fenced = True

    async def _run() -> None:
        engine = _DelayedFenceEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=0.02,
        )
        await service.start()
        try:
            await service.create_async(_request("video_delayed_fence"))
            await asyncio.wait_for(engine.started.wait(), timeout=3.0)
            item = service._items["video_delayed_fence"]
            assert item.target is not None
            staging_path = item.target.staging_path

            with pytest.raises(DiffletServingError) as deleting:
                await asyncio.wait_for(
                    service.delete_job("video_delayed_fence"),
                    timeout=1.0,
                )
            assert (deleting.value.status_code, deleting.value.code) == (
                409,
                "video_in_progress",
            )
            assert engine.fence_wait_started.is_set() is False
            assert engine.fenced is False
            assert item.done.is_set() is False
            assert service._pending == 1
            assert staging_path.exists()
            assert jobs.require("video_delayed_fence").status == "in_progress"

        finally:
            engine.release_fence.set()
            await service.shutdown()

    asyncio.run(_run())


def test_sync_deadline_expires_while_publication_validation_is_blocked(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _BlockingValidationStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.1,
            recovery_timeout_s=0.02,
        )
        await service.start()
        try:
            generation = asyncio.create_task(
                service.generate_sync(_request("video_sync_publish_timeout"))
            )
            assert await asyncio.to_thread(artifacts.validation_entered.wait, 3.0)
            item = service._items["video_sync_publish_timeout"]
            assert item.target is not None
            staging_path = item.target.staging_path

            with pytest.raises(DiffletServingError) as timed_out:
                await asyncio.wait_for(generation, timeout=1.0)
            assert (timed_out.value.status_code, timed_out.value.code) == (
                504,
                "request_timeout",
            )

            # The caller receives its deadline while the non-cancellable thread
            # remains fenced in validation.  It still owns admission and staging.
            assert item.deadline_expired is True
            assert item.done.is_set() is False
            assert service._pending == 1
            assert staging_path.exists()
            assert artifacts.get("video_sync_publish_timeout.mp4") is None
            assert jobs.count() == 0

            artifacts.release_validation.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)
            assert service._pending == 0
            assert staging_path.exists() is False
            assert artifacts.get("video_sync_publish_timeout.mp4") is None
            assert tuple(artifacts.staging_root.iterdir()) == ()
            assert tuple(artifacts.artifact_root.iterdir()) == ()
        finally:
            artifacts.release_validation.set()
            await service.shutdown()

    asyncio.run(_run())


def test_sync_disconnect_during_local_commit_removes_committed_artifact(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_sync_disconnect_commit"
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _BlockingCommitStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            generation = asyncio.create_task(service.generate_sync(_request(request_id)))
            assert await asyncio.to_thread(artifacts.commit_entered.wait, 3.0)
            item = service._items[request_id]
            assert item.publication_claimed is True
            assert item.phase == "publishing_local"

            generation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await generation
            assert item.delete_requested is True

            artifacts.release_commit.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)

            assert artifacts.get(f"{request_id}.mp4") is None
            assert tuple(artifacts.staging_root.iterdir()) == ()
            assert tuple(artifacts.artifact_root.iterdir()) == ()
            assert service._retained_bytes == 0
        finally:
            artifacts.release_commit.set()
            await service.shutdown()

    asyncio.run(_run())


def test_async_deadline_fails_job_while_publication_validation_is_blocked(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _BlockingValidationStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.1,
            recovery_timeout_s=0.02,
        )
        await service.start()
        try:
            created = await service.create_async(_request("video_async_publish_timeout"))
            assert created.status == "queued"
            assert await asyncio.to_thread(artifacts.validation_entered.wait, 3.0)
            item = service._items["video_async_publish_timeout"]
            assert item.target is not None
            staging_path = item.target.staging_path

            failed = await asyncio.wait_for(
                _wait_for_status(service, "video_async_publish_timeout", "failed"),
                timeout=1.0,
            )
            assert failed.error is not None
            assert (failed.error.code, failed.error.error_type) == (
                "request_timeout",
                "server_error",
            )
            assert item.deadline_expired is True
            assert item.done.is_set() is False
            assert service._pending == 1
            assert staging_path.exists()
            assert artifacts.get("video_async_publish_timeout.mp4") is None

            artifacts.release_validation.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)
            final = jobs.require("video_async_publish_timeout")
            assert final.status == "failed"
            assert final.error is not None
            assert final.error.code == "request_timeout"
            assert final.artifact_key is None
            assert service._pending == 0
            assert staging_path.exists() is False
            assert artifacts.get("video_async_publish_timeout.mp4") is None
            assert tuple(artifacts.staging_root.iterdir()) == ()
            assert tuple(artifacts.artifact_root.iterdir()) == ()
        finally:
            artifacts.release_validation.set()
            await service.shutdown()

    asyncio.run(_run())


def test_async_local_publication_claim_survives_s3_past_request_deadline(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        client = _BlockingS3Client()
        artifacts = S3VideoArtifactStore(
            tmp_path / "media",
            s3=_BlockingS3Store(client),  # type: ignore[arg-type]
            retention_seconds=90_000,
        )
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.2,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            created = await service.create_async(_request("video_s3_past_deadline"))
            assert created.status == "queued"
            assert await asyncio.to_thread(client.upload_started.wait, 3.0)

            item = service._items["video_s3_past_deadline"]
            assert item.publication_claimed is True
            assert item.phase == "publishing_remote"
            assert item.artifact is not None
            assert artifacts.require(item.artifact.key) == item.artifact

            # The original total request deadline expires while S3 is blocked.
            # A valid local MP4 has already won completion and must not be
            # converted to a failed job or removed.
            await asyncio.sleep(0.25)
            in_progress = jobs.require("video_s3_past_deadline")
            assert in_progress.status == "in_progress"
            assert item.deadline_expired is False
            assert artifacts.get("video_s3_past_deadline.mp4") is not None

            client.release_upload.set()
            completed = await _wait_for_status(
                service,
                "video_s3_past_deadline",
                "completed",
            )
            assert completed.error is None
            assert completed.media_metadata is not None
            assert completed.media_metadata["url"].startswith("https://example.test/")
        finally:
            client.release_upload.set()
            await service.shutdown()

    asyncio.run(_run())


def test_failed_completion_persistence_transfers_orphan_cleanup_to_store(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_completion_orphan"
        engine = _FakeVideoEngine()
        jobs = _FailCompletionRepository()
        client = _DeleteRetryS3Client()
        client.delete_errors = [
            OSError(f"initial orphan cleanup attempt {attempt} unavailable") for attempt in range(3)
        ]
        artifacts = S3VideoArtifactStore(
            tmp_path / "media",
            s3=_BlockingS3Store(client),  # type: ignore[arg-type]
        )
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(request_id))
            item = service._items[request_id]
            await asyncio.wait_for(item.done.wait(), timeout=3.0)

            failed = jobs.require(request_id)
            assert failed.status == "failed"
            assert failed.artifact_key is None
            assert artifacts.get(f"{request_id}.mp4") is not None
            assert service._retained_bytes > 0
            assert len(client.deletes) == 3

            # The failed row cannot identify the committed artifact. Store-owned
            # orphan cleanup retries remote removal, then local removal, and
            # reports the released bytes back to the service.
            assert await service.sweep_expired(now=0) == 0
            assert artifacts.get(f"{request_id}.mp4") is None
            assert service._retained_bytes == 0
            assert len(client.deletes) == 4
            assert jobs.require(request_id).status == "failed"
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_deadline_cancels_engine_before_blocking_failure_persistence(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_deadline_db_block"
        engine = _FakeVideoEngine(blocked_ids={request_id})
        jobs = _BlockingFailureRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.05,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(request_id))
            await engine.started[request_id].wait()
            item = service._items[request_id]
            assert await asyncio.to_thread(jobs.failure_entered.wait, 3.0)

            deadline = asyncio.get_running_loop().time() + 1.0
            while request_id not in engine.cancelled:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "engine cancellation waited for metadata failure persistence"
                    )
                await asyncio.sleep(0)

            assert jobs.release_failure.is_set() is False
            jobs.release_failure.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)
            failed = jobs.require(request_id)
            assert failed.status == "failed"
            assert failed.error is not None
            assert failed.error.code == "request_timeout"
        finally:
            jobs.release_failure.set()
            await service.shutdown()

    asyncio.run(_run())


def test_delete_is_bounded_after_terminal_publication_is_claimed(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    class _BlockingCompletionRepository(InMemoryVideoJobRepository):
        def __init__(self) -> None:
            super().__init__()
            self.completion_entered = threading.Event()
            self.release_completion = threading.Event()

        def mark_completed(self, *args, **kwargs):
            self.completion_entered.set()
            assert self.release_completion.wait(timeout=3.0)
            return super().mark_completed(*args, **kwargs)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = _BlockingCompletionRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=0.02,
        )
        await service.start()
        try:
            await service.create_async(_request("video_claimed_publication"))
            assert await asyncio.to_thread(jobs.completion_entered.wait, 3.0)
            item = service._items["video_claimed_publication"]
            assert item.publication_claimed is True
            assert item.phase == "terminal"
            assert item.artifact is not None

            # Completion already won the linearization race, but its metadata CAS
            # must not hold the service lock and make DELETE wait without a bound.
            with pytest.raises(DiffletServingError) as deleting:
                await asyncio.wait_for(
                    service.delete_job("video_claimed_publication"),
                    timeout=1.0,
                )
            assert (deleting.value.status_code, deleting.value.code) == (
                409,
                "video_in_progress",
            )
            assert item.cancelled is False
            assert item.done.is_set() is False
            assert jobs.require("video_claimed_publication").status == "in_progress"
            assert artifacts.get("video_claimed_publication.mp4") is not None

            jobs.release_completion.set()
            await asyncio.wait_for(item.done.wait(), timeout=3.0)
            completed = jobs.require("video_claimed_publication")
            assert completed.status == "completed"
            assert completed.artifact_key == "video_claimed_publication.mp4"

            deleted = await service.delete_job("video_claimed_publication")
            assert deleted.status == "completed"
            assert jobs.get("video_claimed_publication") is None
            assert artifacts.get("video_claimed_publication.mp4") is None
        finally:
            jobs.release_completion.set()
            await service.shutdown()

    asyncio.run(_run())


def test_content_states_and_open_lease_survive_completed_job_delete(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine(
            blocked_ids={"video_pending"},
            failing_prompts={"fail"},
        )
        service, jobs, artifacts = _service(tmp_path, engine)
        await service.start()
        try:
            await service.create_async(_request("video_pending"))
            await engine.started["video_pending"].wait()
            with pytest.raises(DiffletServingError) as pending:
                await service.open_content("video_pending")
            assert (pending.value.status_code, pending.value.code) == (409, "video_not_ready")
            with pytest.raises(DiffletServingError) as deleting:
                await service.delete_job("video_pending")
            assert (deleting.value.status_code, deleting.value.code) == (
                409,
                "video_in_progress",
            )
            engine.release.set()
            await _wait_for_status(service, "video_pending", "completed")
            await service.delete_job("video_pending")

            await service.create_async(_request("video_failed", prompt="fail"))
            await _wait_for_status(service, "video_failed", "failed")
            with pytest.raises(DiffletServingError) as failed:
                await service.open_content("video_failed")
            assert (failed.value.status_code, failed.value.code) == (
                422,
                "video_generation_failed",
            )

            await service.create_async(_request("video_completed"))
            await _wait_for_status(service, "video_completed", "completed")
            completed, lease = await service.open_content("video_completed")
            expected_payload = engine.payloads["video_completed"]

            deleted = await service.delete_job("video_completed")
            assert deleted == completed
            assert jobs.get("video_completed") is None
            assert artifacts.get(completed.artifact_key) is None
            assert b"".join(lease.iter_chunks(chunk_size=5)) == expected_payload
            assert lease.closed is True
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_delete_retries_artifact_cleanup_before_metadata_removal(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_delete_retry"
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _FailOnceDeleteStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(request_id))
            await _wait_for_status(service, request_id, "completed")

            deleted = await service.delete_job(request_id)

            assert deleted.status == "completed"
            assert jobs.get(request_id) is None
            assert artifacts.get(f"{request_id}.mp4") is None
            assert artifacts.delete_calls == 2
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_delete_failure_keeps_job_and_artifact_retryable(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_delete_failure"
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _ControlledDeleteStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(request_id))
            completed = await _wait_for_status(service, request_id, "completed")
            assert completed.artifact_key is not None

            with pytest.raises(OSError, match="persistent unlink failure"):
                await service.delete_job(request_id)

            assert artifacts.delete_calls == 3
            assert jobs.require(request_id) == completed
            assert artifacts.get(completed.artifact_key) is not None

            artifacts.fail_deletes = False
            deleted = await service.delete_job(request_id)

            assert deleted == completed
            assert jobs.get(request_id) is None
            assert artifacts.get(completed.artifact_key) is None
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_delete_stale_queued_item_after_deadline_persisted_failed_job(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        blocker_id = "video_delete_deadline_blocker"
        expired_id = "video_delete_deadline_expired"
        engine = _FakeVideoEngine(blocked_ids={blocker_id})
        jobs = _BlockAfterFailureRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(blocker_id))
            await engine.started[blocker_id].wait()
            await service.create_async(
                _request(expired_id),
                deadline=asyncio.get_running_loop().time() + 0.05,
            )
            assert await asyncio.to_thread(jobs.failed_persisted.wait, 3.0)

            item = service._items[expired_id]
            assert item.phase == "queued"
            assert jobs.require(expired_id).status == "failed"

            deletion = asyncio.create_task(service.delete_job(expired_id))
            await asyncio.sleep(0)
            jobs.release_failure_return.set()
            deleted = await asyncio.wait_for(deletion, timeout=3.0)

            assert deleted.status == "failed"
            assert jobs.get(expired_id) is None
            assert expired_id not in service._items
            assert service._job_slots_reserved == 1
        finally:
            jobs.release_failure_return.set()
            engine.release.set()
            await service.shutdown()

    asyncio.run(_run())


def test_failed_s3_delete_background_sweep_preserves_job_and_content(
    tmp_path,
    monkeypatch,
):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        request_id = "video_s3_delete_failure"
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        client = _DeleteRetryS3Client()
        artifacts = S3VideoArtifactStore(
            tmp_path / "media",
            s3=_BlockingS3Store(client),  # type: ignore[arg-type]
        )
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            await service.create_async(_request(request_id))
            completed = await _wait_for_status(service, request_id, "completed")
            assert completed.artifact_key is not None
            assert completed.media_metadata is not None
            original_url = completed.media_metadata["url"]

            client.delete_errors = [OSError(f"remote delete {attempt}") for attempt in range(3)]
            with pytest.raises(OSError, match="remote delete"):
                await service.delete_job(request_id)

            assert len(client.deletes) == 3
            assert jobs.require(request_id) == completed
            assert artifacts.get(completed.artifact_key) is not None

            # Store-level orphan cleanup must not take ownership from the live
            # job transaction after an HTTP DELETE failure.
            assert await service.sweep_expired(now=0) == 0
            assert len(client.deletes) == 3
            assert jobs.require(request_id).media_metadata["url"] == original_url
            _, lease = await service.open_content(request_id)
            lease.close()

            deleted = await service.delete_job(request_id)
            assert deleted == completed
            assert jobs.get(request_id) is None
            assert artifacts.get(completed.artifact_key) is None
            assert len(client.deletes) == 4
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_shutdown_fences_inflight_create_and_never_returns_a_stale_queued_job(tmp_path):
    async def _run() -> None:
        request_id = "video_create_shutdown_race"
        engine = _FakeVideoEngine()
        jobs = _BlockingCreateReturnRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        create = asyncio.create_task(service.create_async(_request(request_id)))
        assert await asyncio.to_thread(jobs.committed.wait, 3.0)

        shutdown = asyncio.create_task(service.shutdown())
        deadline = asyncio.get_running_loop().time() + 1.0
        while service._accepting:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("shutdown did not stop admission")
            await asyncio.sleep(0)
        jobs.release_return.set()

        with pytest.raises(DiffletServingError) as rejected:
            await create
        assert (rejected.value.status_code, rejected.value.code) == (
            503,
            "video_service_unavailable",
        )
        await shutdown

        # Shutdown clears and closes the process-local store. A new lifecycle
        # starts with no knowledge of the previous request.
        with pytest.raises(VideoJobRepositoryError):
            jobs.count()
        with InMemoryVideoJobRepository() as fresh:
            assert fresh.count() == 0
        assert tuple(artifacts.artifact_root.iterdir()) == ()
        assert tuple(artifacts.staging_root.iterdir()) == ()
        assert service._pending == 0
        assert service._queue.empty()
        assert engine.calls == []

    asyncio.run(_run())


def test_async_create_uses_total_deadline_before_queueing(tmp_path):
    async def _run() -> None:
        request_id = "video_create_deadline"
        engine = _FakeVideoEngine()
        jobs = _BlockingCreateReturnRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.05,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            create = asyncio.create_task(service.create_async(_request(request_id)))
            assert await asyncio.to_thread(jobs.committed.wait, 3.0)
            item = service._items[request_id]
            deadline = asyncio.get_running_loop().time() + 1.0
            while not item.deadline_expired:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("create did not observe the total request deadline")
                await asyncio.sleep(0)
            jobs.release_return.set()

            with pytest.raises(DiffletServingError) as timed_out:
                await create
            assert (timed_out.value.status_code, timed_out.value.code) == (
                504,
                "request_timeout",
            )
            failed = jobs.require(request_id)
            assert failed.status == "failed"
            assert failed.error is not None
            assert failed.error.code == "request_timeout"
            assert service._pending == 0
            assert engine.calls == []
        finally:
            jobs.release_return.set()
            await service.shutdown()

    asyncio.run(_run())


def test_cancelled_async_create_fences_store_and_removes_unenqueued_job(tmp_path):
    class _SlowCreateRepository(InMemoryVideoJobRepository):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def create(self, create: VideoJobCreate):
            self.entered.set()
            assert self.release.wait(timeout=3.0)
            return super().create(create)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = _SlowCreateRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            create_task = asyncio.create_task(service.create_async(_request("video_cancelled")))
            assert await asyncio.to_thread(jobs.entered.wait, 3.0)
            create_task.cancel()
            jobs.release.set()
            with pytest.raises(asyncio.CancelledError):
                await create_task

            assert jobs.count() == 0
            assert service._pending == 0
            assert "video_cancelled" not in service._items
            assert engine.calls == []
        finally:
            jobs.release.set()
            await service.shutdown()

    asyncio.run(_run())


def test_cancel_after_create_commit_before_enqueue_releases_all_capacity(tmp_path):
    async def _run() -> None:
        request_id = "video_cancelled_after_create"
        engine = _FakeVideoEngine()
        jobs = _BlockingCreateReturnRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        lock_held = False
        try:
            create = asyncio.create_task(service.create_async(_request(request_id)))
            assert await asyncio.to_thread(jobs.committed.wait, 3.0)

            await service._lock.acquire()
            lock_held = True
            jobs.release_return.set()
            await asyncio.sleep(0.01)
            create.cancel()
            service._lock.release()
            lock_held = False

            with pytest.raises(asyncio.CancelledError):
                await create

            assert jobs.count() == 0
            assert request_id not in service._items
            assert service._pending == 0
            assert service._job_slots_reserved == 0
            assert service._active_reserved_bytes == 0
            assert engine.calls == []
        finally:
            if lock_held:
                service._lock.release()
            jobs.release_return.set()
            await service.shutdown()

    asyncio.run(_run())


def test_terminal_ttl_sweeper_removes_metadata_and_artifact(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
            retention_seconds=1,
            sweep_interval_s=60.0,
        )
        await service.start()
        try:
            await service.create_async(_request("video_ttl"))
            completed = await _wait_for_status(service, "video_ttl", "completed")
            assert completed.expires_at is not None
            assert artifacts.get("video_ttl.mp4") is not None

            assert await service.sweep_expired(now=completed.expires_at) == 1
            assert jobs.get("video_ttl") is None
            assert artifacts.get("video_ttl.mp4") is None
            assert service._retained_bytes == 0
            assert service._job_slots_reserved == 0
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_retention_sweep_clears_active_item_job_slot_ownership(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    class _RecoveryBlockedEngine(_FakeVideoEngine):
        def __init__(self) -> None:
            super().__init__(blocked_ids={"video_expired_active"})
            self.recovery_entered = asyncio.Event()
            self.release_recovery = asyncio.Event()

        async def wait_for_recovery(self, *, timeout: float | None) -> None:
            assert timeout is None
            self.recovery_waits += 1
            self.recovery_entered.set()
            await self.release_recovery.wait()

    async def _run() -> None:
        engine = _RecoveryBlockedEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=0.02,
            recovery_timeout_s=1.0,
            retention_seconds=1,
            max_jobs=2,
            sweep_interval_s=60.0,
        )
        await service.start()
        try:
            await service.create_async(_request("video_expired_active"))
            await asyncio.wait_for(engine.recovery_entered.wait(), timeout=3.0)
            failed = await _wait_for_status(service, "video_expired_active", "failed")
            assert failed.expires_at is not None
            active_item = service._items[failed.id]

            await service.create_async(
                _request("video_retained"),
                deadline=time.monotonic() + 5.0,
            )
            assert service._job_slots_reserved == 2

            assert await service.sweep_expired(now=failed.expires_at) == 1
            assert jobs.get(failed.id) is None
            assert jobs.get("video_retained") is not None
            assert active_item.job_slot_owned is False
            assert service._job_slots_reserved == 1

            engine.release_recovery.set()
            await asyncio.wait_for(active_item.done.wait(), timeout=3.0)
            assert jobs.count() == 1
            assert service._job_slots_reserved == 1
        finally:
            engine.release.set()
            engine.release_recovery.set()
            await service.shutdown()

    asyncio.run(_run())


def test_retention_sweeper_retries_store_owned_remote_cleanup(tmp_path):
    async def _run() -> None:
        engine = _FakeVideoEngine()
        jobs = InMemoryVideoJobRepository()
        artifacts = _PendingCleanupStore(tmp_path / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=1,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        await service.start()
        try:
            assert await service.sweep_expired() == 0
            assert artifacts.retry_calls == 1
        finally:
            await service.shutdown()

    asyncio.run(_run())


def test_async_job_cap_and_cumulative_storage_reservation_are_enforced(tmp_path, monkeypatch):
    _patch_media_validation_without_pyav(monkeypatch)

    async def _run_job_cap() -> None:
        class _CreateBlockedBeforeCommit(InMemoryVideoJobRepository):
            def __init__(self) -> None:
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def create(self, create: VideoJobCreate):
                self.entered.set()
                assert self.release.wait(timeout=3.0)
                return super().create(create)

        engine = _FakeVideoEngine()
        jobs = _CreateBlockedBeforeCommit()
        artifacts = LocalVideoArtifactStore(tmp_path / "cap" / "media")
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=2,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
            max_jobs=1,
        )
        await service.start()
        first = asyncio.create_task(service.create_async(_request("video_cap_1")))
        try:
            assert await asyncio.to_thread(jobs.entered.wait, 3.0)
            assert jobs.count() == 0
            with pytest.raises(DiffletServingError) as full:
                await service.create_async(_request("video_cap_2"))
            assert (full.value.status_code, full.value.code) == (
                429,
                "video_retention_full",
            )
            jobs.release.set()
            await first
        finally:
            jobs.release.set()
            engine.release.set()
            await service.shutdown()

    async def _run_storage() -> None:
        engine = _FakeVideoEngine(blocked_ids={"video_storage_1"})
        jobs = InMemoryVideoJobRepository()
        artifacts = LocalVideoArtifactStore(tmp_path / "storage" / "media", max_artifact_bytes=100)
        service = VideoGenerationService(
            engine=engine,
            jobs=jobs,
            artifacts=artifacts,
            max_queued_requests=2,
            queue_timeout_s=2.0,
            request_timeout_s=2.0,
            recovery_timeout_s=1.0,
        )
        monkeypatch.setattr(
            artifacts,
            "available_bytes",
            lambda: service.storage_margin_bytes + 150,
        )
        await service.start()
        try:
            await service.create_async(_request("video_storage_1"))
            await engine.started["video_storage_1"].wait()
            with pytest.raises(DiffletServingError) as full:
                await service.create_async(_request("video_storage_2"))
            assert (full.value.status_code, full.value.code) == (
                507,
                "video_storage_full",
            )
        finally:
            engine.release.set()
            await service.shutdown()

    asyncio.run(_run_job_cap())
    asyncio.run(_run_storage())
