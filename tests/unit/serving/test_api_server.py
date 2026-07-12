from __future__ import annotations

import asyncio

import pytest

from difflet.serving.openai.api_server import create_app
from difflet.serving.options import ServeOptions
from difflet.serving.model_registry import resolve_serving_model


class _UnhealthyEngine:
    healthy = False
    ready = False

    async def start(self):
        return None

    async def shutdown(self):
        return None


def test_health_returns_503_when_engine_unhealthy():
    app = create_app(
        options=ServeOptions(model_id="black-forest-labs/FLUX.1-dev", artifact_store="memory"),
        resolved_model=resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev")),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")

    response = asyncio.run(health_route.endpoint())

    assert response.status_code == 503


@pytest.mark.parametrize("payload", [[], None, "not-an-object", 42])
def test_chat_completions_non_object_json_uses_difflet_error_contract(payload):
    from fastapi.testclient import TestClient

    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
        artifact_store="memory",
    )
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "request body must be an object",
            "type": "invalid_request_error",
            "code": "invalid_request",
        }
    }
