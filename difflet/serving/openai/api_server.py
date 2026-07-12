"""FastAPI app factory for Difflet serving."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from difflet.serving.artifact_store import MemoryArtifactStore, R2ArtifactStore
from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import ResolvedServingModel
from difflet.serving.openai.serving_chat import generate_chat_completion
from difflet.serving.options import ServeOptions

logger = logging.getLogger(__name__)


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
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("fastapi is required for `difflet serve`") from exc

    if artifact_store is None:
        artifact_store = (
            MemoryArtifactStore()
            if options.artifact_store == "memory"
            else R2ArtifactStore.from_env(client_timeout=options.artifact_store_timeout)
        )

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
    async def chat_completions(payload: Any = Body(default=None), request: Request | None = None):
        start = time.perf_counter()
        request_id = payload.get("id") if isinstance(payload, dict) and "id" in payload else None
        client = request.client.host if request is not None and request.client else None
        logger.info(
            "http.request_started path=%s method=%s model=%s request_id=%s client=%s",
            "/v1/chat/completions",
            "POST",
            resolved_model.model_id,
            request_id,
            client,
        )
        try:
            response = await generate_chat_completion(
                payload,
                resolved_model=resolved_model,
                engine=engine,
                request_validator=request_validator,
                artifact_store=artifact_store,
                artifact_ttl_seconds=options.artifact_ttl_seconds,
                artifact_store_timeout=options.artifact_store_timeout,
            )
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
            raise

    return app
