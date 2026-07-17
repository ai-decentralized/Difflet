"""FastAPI app factory for Difflet serving."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextlib import suppress
from pathlib import Path
from typing import Any, Coroutine, TypeVar

from difflet.serving.artifact_store import S3ArtifactStore
from difflet.serving.errors import DiffletServingError, internal_error, request_cancelled
from difflet.serving.model_registry import ResolvedServingModel
from difflet.serving.openai.serving_chat import (
    generate_chat_completion,
    normalize_chat_request,
)
from difflet.serving.options import ServeOptions
from difflet.serving.validation import BoundedValidationExecutor

logger = logging.getLogger(__name__)

_ResultT = TypeVar("_ResultT")
_DISCONNECT_POLL_SECONDS = 0.1


async def _wait_for_disconnect(request) -> None:
    # The request body has already been parsed before FastAPI enters this route,
    # so the only relevant subsequent ASGI message is ``http.disconnect``.
    # Avoid Starlette's zero-timeout ``is_disconnected()`` probe: with some
    # AnyIO/TestClient combinations it can remain blocked after the body is read.
    while True:
        try:
            message = await asyncio.wait_for(
                request.receive(),
                timeout=_DISCONNECT_POLL_SECONDS,
            )
        except asyncio.TimeoutError:
            continue
        if message.get("type") == "http.disconnect":
            return


async def _run_until_disconnect(
    operation: Coroutine[Any, Any, _ResultT],
    *,
    request,
) -> _ResultT:
    operation_task = asyncio.create_task(operation, name="difflet-http-chat-completion")
    disconnect_task = asyncio.create_task(
        _wait_for_disconnect(request),
        name="difflet-http-disconnect",
    )
    try:
        done, _ = await asyncio.wait(
            (operation_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return operation_task.result()

        disconnect_task.result()
        operation_task.cancel()
        with suppress(BaseException):
            await operation_task
        raise request_cancelled("client disconnected")
    finally:
        for task in (operation_task, disconnect_task):
            if not task.done():
                task.cancel()
        for task in (operation_task, disconnect_task):
            with suppress(BaseException):
                await task


async def _run_with_request_timeout(
    operation: Coroutine[Any, Any, _ResultT],
    *,
    timeout_s: float,
) -> _ResultT:
    try:
        return await asyncio.wait_for(operation, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise DiffletServingError(
            504,
            "request_timeout",
            "Request timed out",
            "server_error",
        ) from exc


def create_app(
    *,
    options: ServeOptions,
    resolved_model: ResolvedServingModel,
    engine,
    request_validator=None,
    artifact_store=None,
    video_service=None,
    video_jobs=None,
    video_artifact_store=None,
):
    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("fastapi is required for `difflet serve`") from exc

    class _CleanupStreamingResponse(StreamingResponse):
        """Run async cleanup even when ASGI header/body sending is interrupted."""

        def __init__(self, *args, cleanup, **kwargs):
            super().__init__(*args, **kwargs)
            self._cleanup = cleanup

        async def __call__(self, scope, receive, send):
            try:
                return await super().__call__(scope, receive, send)
            finally:
                await _run_cleanup_fenced(self._cleanup())

    async def _run_cleanup_fenced(cleanup: Coroutine[Any, Any, None]) -> None:
        cleanup_task = asyncio.create_task(cleanup, name="difflet-video-response-cleanup")
        while True:
            try:
                await asyncio.shield(cleanup_task)
                return
            except asyncio.CancelledError:
                if cleanup_task.done():
                    cleanup_task.result()
                    return

    is_video = resolved_model.metadata.output_modality == "video"
    validation_executor = BoundedValidationExecutor(
        max_workers=options.validation_workers,
        max_waiting=options.validation_max_waiting,
        timeout_s=options.validation_timeout,
    )
    if not is_video:
        if artifact_store is None:
            artifact_store = S3ArtifactStore.from_env_if_configured(
                client_timeout=options.artifact_store_timeout
            )
        image_response_mode = "s3" if artifact_store is not None else "inline"
        logger.info("serving image_response_mode=%s", image_response_mode)
    elif video_service is None:
        from difflet.serving.video_jobs import InMemoryVideoJobRepository
        from difflet.serving.video_service import VideoGenerationService
        from difflet.serving.video_storage import LocalVideoArtifactStore, S3VideoArtifactStore

        video_root = _video_service_root(options, resolved_model)
        video_jobs = video_jobs or InMemoryVideoJobRepository()
        if video_artifact_store is None:
            s3_store = S3ArtifactStore.from_env_if_configured(
                client_timeout=options.artifact_store_timeout
            )
            video_artifact_store = (
                S3VideoArtifactStore(
                    video_root / "media",
                    s3=s3_store,
                    retention_seconds=options.video_retention_seconds,
                )
                if s3_store is not None
                else LocalVideoArtifactStore(video_root / "media")
            )
            logger.info("serving video_artifact_store=%s", type(video_artifact_store).__name__)
        video_service = VideoGenerationService(
            engine=engine,
            jobs=video_jobs,
            artifacts=video_artifact_store,
            max_queued_requests=options.max_queued_requests,
            queue_timeout_s=options.queue_timeout,
            request_timeout_s=options.request_timeout,
            recovery_timeout_s=(options.worker_cancel_timeout + options.worker_restart_timeout),
            retention_seconds=options.video_retention_seconds,
            max_jobs=options.video_max_jobs,
            sweep_interval_s=options.video_sweep_interval_seconds,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("serving startup begin model=%s", resolved_model.model_id)
        try:
            await validation_executor.start(request_validator)
            await engine.start()
        except BaseException:
            try:
                if video_service is not None:
                    await video_service.shutdown()
            finally:
                # Resident startup is cancellation-fenced and may have made a
                # child ready before re-raising cancellation.  Always perform
                # the matching teardown on every failed startup path.
                try:
                    await engine.shutdown()
                finally:
                    await validation_executor.shutdown()
            raise
        try:
            if video_service is not None:
                await video_service.start()
            logger.info("serving startup complete model=%s", resolved_model.model_id)
            yield
        finally:
            logger.info("serving shutdown begin model=%s", resolved_model.model_id)
            try:
                if video_service is not None:
                    await video_service.shutdown()
            finally:
                try:
                    await engine.shutdown()
                finally:
                    await validation_executor.shutdown()
            logger.info("serving shutdown complete model=%s", resolved_model.model_id)

    app = FastAPI(title="Difflet Serving", lifespan=lifespan)
    app.state.video_service = video_service
    app.state.validation_executor = validation_executor

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "http.request_validation_failed path=%s method=%s validation_errors=%d",
            request.url.path,
            request.method,
            len(exc.errors()),
        )
        error = DiffletServingError(
            400,
            "invalid_request",
            "request body is not valid JSON",
        )
        return JSONResponse(status_code=error.status_code, content=error.to_payload())

    @app.exception_handler(DiffletServingError)
    async def _serving_error_handler(request: Request, exc: DiffletServingError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.get("/health")
    async def health():
        if not engine.healthy:
            return JSONResponse(status_code=503, content={"status": "unhealthy"})
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not engine.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "model": resolved_model.model_id},
            )
        return {"status": "ready", "model": resolved_model.model_id}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": resolved_model.model_id,
                    "object": "model",
                    "owned_by": "difflet",
                    "difflet_model_type": resolved_model.metadata.model_type,
                }
            ],
        }

    if not is_video:

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request, payload: Any = Body(default=None)):
            start = time.perf_counter()
            request_id = str(uuid.uuid4())
            client = request.client.host if request.client else None
            logger.info(
                "http.request_started path=%s method=%s model=%s request_id=%s client=%s",
                "/v1/chat/completions",
                "POST",
                resolved_model.model_id,
                request_id,
                client,
            )
            try:
                deadline = time.monotonic() + options.request_timeout
                normalized = normalize_chat_request(
                    payload,
                    resolved_model=resolved_model,
                    request_id=request_id,
                )
                if request_validator is not None:
                    await validation_executor.run(
                        request_validator.validate,
                        normalized,
                        deadline=deadline,
                    )
                operation = generate_chat_completion(
                    payload,
                    resolved_model=resolved_model,
                    request_id=request_id,
                    engine=engine,
                    request_validator=None,
                    artifact_store=artifact_store,
                    artifact_ttl_seconds=options.artifact_ttl_seconds,
                    artifact_store_timeout=options.artifact_store_timeout,
                    normalized_request=normalized,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DiffletServingError(
                        504, "request_timeout", "Request timed out", "server_error"
                    )
                response = await _run_until_disconnect(
                    _run_with_request_timeout(operation, timeout_s=remaining),
                    request=request,
                )
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.info(
                    "http.request_completed path=%s status=%s model=%s "
                    "request_id=%s duration_ms=%.2f",
                    "/v1/chat/completions",
                    200,
                    resolved_model.model_id,
                    request_id,
                    duration_ms,
                )
                return response
            except DiffletServingError as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.warning(
                    "http.request_failed path=%s status=%s model=%s request_id=%s "
                    "duration_ms=%.2f code=%s message=%s",
                    "/v1/chat/completions",
                    exc.status_code,
                    resolved_model.model_id,
                    request_id,
                    duration_ms,
                    exc.code,
                    exc.message,
                )
                raise
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                logger.exception(
                    "http.request_error path=%s status=%s model=%s request_id=%s "
                    "duration_ms=%.2f",
                    "/v1/chat/completions",
                    500,
                    resolved_model.model_id,
                    request_id,
                    duration_ms,
                )
                raise internal_error() from exc

    else:
        from difflet.serving.openai.serving_video import (
            normalize_video_multipart_request,
            video_delete_response,
            video_job_to_response,
            video_jobs_to_list_response,
        )
        from difflet.serving.video_jobs import new_video_job_id

        assert video_service is not None

        async def _normalize_video(request: Request, request_id: str):
            normalized = await normalize_video_multipart_request(
                request,
                resolved_model=resolved_model,
                request_id=request_id,
            )
            deadline = time.monotonic() + options.request_timeout
            if request_validator is not None:
                await validation_executor.run(
                    request_validator.validate,
                    normalized,
                    deadline=deadline,
                )
            return normalized, deadline

        @app.post("/v1/videos")
        async def create_video(request: Request):
            video_id = new_video_job_id()
            try:
                normalized, deadline = await _normalize_video(request, video_id)
                job = await video_service.create_async(normalized, deadline=deadline)
                return JSONResponse(
                    status_code=200,
                    content=video_job_to_response(job).model_dump(mode="json"),
                )
            except DiffletServingError:
                raise
            except Exception as exc:
                logger.exception("video.create_failed video_id=%s", video_id)
                raise internal_error() from exc

        @app.post("/v1/videos/sync")
        async def create_video_sync(request: Request):
            request_id = f"video_sync_{uuid.uuid4().hex}"
            lease = None
            result = None

            async def _cleanup_sync_result() -> None:
                nonlocal lease, result
                current_lease = lease
                current_result = result
                lease = None
                result = None
                if current_lease is not None:
                    current_lease.close()
                if current_result is None:
                    return
                try:
                    await video_service.delete_sync_result(current_result)
                except Exception:
                    logger.exception(
                        "video.sync_cleanup_failed request_id=%s artifact=%s",
                        request_id,
                        current_result.artifact.key,
                    )

            try:
                normalized, deadline = await _normalize_video(request, request_id)
                result = await _run_until_disconnect(
                    video_service.generate_sync(normalized, deadline=deadline),
                    request=request,
                )
                lease = await video_service.open_sync_result(result)

                return _CleanupStreamingResponse(
                    lease.iter_chunks(),
                    media_type="video/mp4",
                    headers={
                        "Content-Length": str(result.artifact.size_bytes),
                        "X-Request-Id": request_id,
                        "X-Model": resolved_model.model_id,
                        "X-Inference-Time-S": f"{result.inference_time_s:.6f}",
                    },
                    cleanup=_cleanup_sync_result,
                )
            except asyncio.CancelledError:
                await _run_cleanup_fenced(_cleanup_sync_result())
                raise
            except DiffletServingError:
                raise
            except Exception as exc:
                await _run_cleanup_fenced(_cleanup_sync_result())
                logger.exception("video.sync_failed request_id=%s", request_id)
                raise internal_error() from exc

        @app.get("/v1/videos")
        async def list_videos(request: Request):
            limit, after = _video_list_query(request)
            try:
                page = await video_service.list_jobs(limit=limit, after=after)
                response = video_jobs_to_list_response(page.data, has_more=page.has_more)
                return JSONResponse(content=response.model_dump(mode="json"))
            except DiffletServingError:
                raise
            except (TypeError, ValueError) as exc:
                raise DiffletServingError(400, "invalid_request", str(exc)) from exc
            except Exception as exc:
                logger.exception("video.list_failed")
                raise internal_error() from exc

        @app.get("/v1/videos/{video_id}/content")
        async def video_content(video_id: str):
            job, lease = await video_service.open_content(video_id)

            async def _close_content_lease() -> None:
                lease.close()

            return _CleanupStreamingResponse(
                lease.iter_chunks(),
                media_type="video/mp4",
                headers={
                    "Content-Length": str(lease.artifact.size_bytes),
                    "Content-Disposition": f'attachment; filename="{job.id}.mp4"',
                },
                cleanup=_close_content_lease,
            )

        @app.get("/v1/videos/{video_id}")
        async def retrieve_video(video_id: str):
            try:
                job = await video_service.get_job(video_id)
                return JSONResponse(content=video_job_to_response(job).model_dump(mode="json"))
            except DiffletServingError:
                raise
            except Exception as exc:
                logger.exception("video.retrieve_failed video_id=%s", video_id)
                raise internal_error() from exc

        @app.delete("/v1/videos/{video_id}")
        async def delete_video(video_id: str):
            try:
                deleted = await video_service.delete_job(video_id)
                response = video_delete_response(deleted.id)
                return JSONResponse(content=response.model_dump(mode="json"))
            except DiffletServingError:
                raise
            except Exception as exc:
                logger.exception("video.delete_failed video_id=%s", video_id)
                raise internal_error() from exc

    return app


def _video_service_root(options: ServeOptions, resolved_model: ResolvedServingModel) -> Path:
    cache_root = Path(options.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
    profile = resolved_model.profile
    identity = json.dumps(
        {
            "model": profile.model_id,
            "revision": profile.revision,
            "height": profile.height,
            "width": profile.width,
            "num_frames": profile.num_frames,
            "fps": profile.output_fps,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:24]
    return cache_root / "serving" / "videos" / digest


def _video_list_query(request) -> tuple[int, str | None]:
    query = request.query_params
    unknown = set(query) - {"limit", "after"}
    if unknown:
        raise DiffletServingError(
            400,
            "invalid_request",
            f"unsupported list query field {sorted(unknown)[0]!r}",
        )
    for field in ("limit", "after"):
        if len(query.getlist(field)) > 1:
            raise DiffletServingError(
                400,
                "invalid_request",
                f"list query field {field!r} is repeated",
            )
    raw_limit = query.get("limit", "20")
    if not raw_limit.isdigit():
        raise DiffletServingError(400, "invalid_request", "limit must be an integer")
    limit = int(raw_limit)
    if not 1 <= limit <= 100:
        raise DiffletServingError(
            400,
            "invalid_request",
            "limit must satisfy 1 <= limit <= 100",
        )
    after = query.get("after")
    if after is not None and not after:
        raise DiffletServingError(400, "invalid_request", "after must not be empty")
    return limit, after
