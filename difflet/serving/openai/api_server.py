"""FastAPI app factory for Difflet serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from difflet.serving.artifact_store import MemoryArtifactStore, R2ArtifactStore
from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import ResolvedServingModel
from difflet.serving.openai.serving_chat import generate_chat_completion
from difflet.serving.options import ServeOptions


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
        await engine.start()
        try:
            yield
        finally:
            await engine.shutdown()

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
    async def chat_completions(payload: Any = Body(default=None)):
        return await generate_chat_completion(
            payload,
            resolved_model=resolved_model,
            engine=engine,
            request_validator=request_validator,
            artifact_store=artifact_store,
            artifact_ttl_seconds=options.artifact_ttl_seconds,
            artifact_store_timeout=options.artifact_store_timeout,
        )

    return app
