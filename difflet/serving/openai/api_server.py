"""FastAPI app factory for Difflet serving."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextlib import suppress
from typing import Any, Coroutine, TypeVar

from difflet.serving.artifact_store import R2ArtifactStore
from difflet.serving.errors import DiffletServingError, internal_error, request_cancelled
from difflet.serving.model_registry import ResolvedServingModel
from difflet.serving.openai.serving_chat import generate_chat_completion
from difflet.serving.options import ServeOptions

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


def create_app(
    *,
    options: ServeOptions,
    resolved_model: ResolvedServingModel,
    engine,
    request_validator=None,
    artifact_store=None,
):
    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("fastapi is required for `difflet serve`") from exc

    if artifact_store is None:
        artifact_store = R2ArtifactStore.from_env_if_configured(
            client_timeout=options.artifact_store_timeout
        )
    image_response_mode = "r2" if artifact_store is not None else "inline"
    logger.info("serving image_response_mode=%s", image_response_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("serving startup begin model=%s", resolved_model.model_id)
        await engine.start()
        try:
            logger.info("serving startup complete model=%s", resolved_model.model_id)
            yield
        finally:
            logger.info("serving shutdown begin model=%s", resolved_model.model_id)
            await engine.shutdown()
            logger.info("serving shutdown complete model=%s", resolved_model.model_id)

    app = FastAPI(title="Difflet Serving", lifespan=lifespan)

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
            operation = generate_chat_completion(
                payload,
                resolved_model=resolved_model,
                request_id=request_id,
                engine=engine,
                request_validator=request_validator,
                artifact_store=artifact_store,
                artifact_ttl_seconds=options.artifact_ttl_seconds,
                artifact_store_timeout=options.artifact_store_timeout,
            )
            response = await _run_until_disconnect(operation, request=request)
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "http.request_completed path=%s status=%s model=%s request_id=%s duration_ms=%.2f",
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
                "http.request_failed path=%s status=%s model=%s request_id=%s duration_ms=%.2f code=%s message=%s",
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
                "http.request_error path=%s status=%s model=%s request_id=%s duration_ms=%.2f",
                "/v1/chat/completions",
                500,
                resolved_model.model_id,
                request_id,
                duration_ms,
            )
            raise internal_error() from exc

    return app
