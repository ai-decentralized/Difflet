from __future__ import annotations

import asyncio
import inspect

import pytest

from difflet.serving.openai.api_server import create_app
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import ServeOptions
from difflet.serving.model_registry import resolve_serving_model


class _UnhealthyEngine:
    healthy = False
    ready = False

    async def start(self):
        return None

    async def shutdown(self):
        return None


class _UnexpectedFailureEngine(_UnhealthyEngine):
    healthy = True
    ready = True

    async def generate(self, request):
        raise RuntimeError("private backend details")


class _SuccessfulEngine(_UnexpectedFailureEngine):
    async def generate(self, request):
        from difflet.serving.types import DiffletGenerateOutput

        return DiffletGenerateOutput(
            data=b"png",
            mime_type="image/png",
            output_format="png",
        )


class _BlockingEngine(_UnexpectedFailureEngine):
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, request):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _CancellationFencedStartEngine(_UnhealthyEngine):
    def __init__(self):
        self.started = asyncio.Event()
        self.release_start = asyncio.Event()
        self.shutdown_calls = 0

    async def start(self):
        self.started.set()
        try:
            await self.release_start.wait()
        except asyncio.CancelledError:
            # Model the resident engine's non-cancellable process startup fence.
            await self.release_start.wait()
            raise

    async def shutdown(self):
        self.shutdown_calls += 1


class _DisconnectRequest:
    client = None

    def __init__(self, engine: _BlockingEngine):
        self.engine = engine

    async def receive(self):
        await self.engine.started.wait()
        return {"type": "http.disconnect"}


def test_health_returns_503_when_engine_unhealthy():
    app = create_app(
        options=ServeOptions(model_id="black-forest-labs/FLUX.1-dev"),
        resolved_model=resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev")),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")

    response = asyncio.run(health_route.endpoint())

    assert response.status_code == 503


def test_api_key_auth_is_optional_and_protects_only_v1_routes():
    from fastapi.testclient import TestClient

    open_options = ServeOptions(model_id="black-forest-labs/FLUX.1-dev")
    open_app = create_app(
        options=open_options,
        resolved_model=resolve_serving_model(open_options),
        engine=_SuccessfulEngine(),
        artifact_store=object(),
    )
    with TestClient(open_app) as client:
        assert client.get("/v1/models").status_code == 200

    protected_options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
        api_key="test-secret",
    )
    assert (
        resolve_serving_model(protected_options).profile
        == resolve_serving_model(open_options).profile
    )
    protected_app = create_app(
        options=protected_options,
        resolved_model=resolve_serving_model(protected_options),
        engine=_SuccessfulEngine(),
        artifact_store=object(),
    )
    with TestClient(protected_app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").json() == {"error": "Unauthorized"}
        assert client.get("/v1/models").status_code == 401
        assert (
            client.post(
                "/v1/chat/completions",
                content=b"not-json",
                headers={"Content-Type": "application/json"},
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/models",
                headers={"Authorization": "Basic test-secret"},
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-secret"},
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/models",
                headers={"Authorization": "bEaReR test-secret"},
            ).status_code
            == 200
        )
        assert client.options("/v1/models").status_code != 401


def test_cancelled_engine_start_is_always_torn_down():
    async def _run() -> None:
        options = ServeOptions(model_id="black-forest-labs/FLUX.1-dev")
        engine = _CancellationFencedStartEngine()
        app = create_app(
            options=options,
            resolved_model=resolve_serving_model(options),
            engine=engine,
            artifact_store=object(),
        )
        context = app.router.lifespan_context(app)
        startup = asyncio.create_task(context.__aenter__())
        await engine.started.wait()
        startup.cancel()
        engine.release_start.set()

        with pytest.raises(asyncio.CancelledError):
            await startup
        assert engine.shutdown_calls == 1

    asyncio.run(_run())


@pytest.mark.parametrize("payload", [[], None, "not-an-object", 42])
def test_chat_completions_non_object_json_uses_difflet_error_contract(payload):
    from fastapi.testclient import TestClient

    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
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


def test_chat_completions_invalid_json_uses_difflet_error_contract():
    from fastapi.testclient import TestClient

    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
    )
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )

    invalid_json = b'{"messages":[{"role":"user","content":"line one\nline two"}]}'
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=invalid_json,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "request body is not valid JSON",
            "type": "invalid_request_error",
            "code": "invalid_request",
        }
    }
    assert "detail" not in response.json()


def test_chat_completions_requires_fastapi_request_injection():
    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
    )
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/v1/chat/completions"
    )

    request_parameter = inspect.signature(route.endpoint).parameters["request"]

    assert request_parameter.default is inspect.Parameter.empty


def test_chat_completions_rejects_client_request_body_id():
    from fastapi.testclient import TestClient

    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
    )
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_UnhealthyEngine(),
        artifact_store=None,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "id": "client-request-id",
                "model": options.model_id,
                "messages": [{"role": "user", "content": "a cat"}],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "feature_not_supported"


def test_chat_completions_returns_data_url_without_s3(monkeypatch):
    from fastapi.testclient import TestClient

    for name in (
        "DIFFLET_S3_BUCKET",
        "DIFFLET_S3_ENDPOINT_URL",
        "DIFFLET_S3_REGION",
        "DIFFLET_S3_ACCESS_KEY_ID",
        "DIFFLET_S3_SECRET_ACCESS_KEY",
        "DIFFLET_S3_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    options = ServeOptions(model_id="black-forest-labs/FLUX.1-dev")
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_SuccessfulEngine(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": options.model_id,
                "messages": [{"role": "user", "content": "a cat"}],
            },
        )

    assert response.status_code == 200
    url = response.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    assert url == "data:image/png;base64,cG5n"


def test_chat_completions_unexpected_failure_uses_sanitized_error_contract(caplog):
    from fastapi.testclient import TestClient

    options = ServeOptions(
        model_id="black-forest-labs/FLUX.1-dev",
    )
    app = create_app(
        options=options,
        resolved_model=resolve_serving_model(options),
        engine=_UnexpectedFailureEngine(),
        artifact_store=None,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": options.model_id,
                "messages": [{"role": "user", "content": "a cat"}],
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "Internal model execution error",
            "type": "server_error",
            "code": "internal_error",
        }
    }
    assert "private backend details" not in response.text
    assert "private backend details" in caplog.text


def test_chat_completions_disconnect_cancels_and_awaits_generation():
    async def _run():
        options = ServeOptions(
            model_id="black-forest-labs/FLUX.1-dev",
        )
        engine = _BlockingEngine()
        app = create_app(
            options=options,
            resolved_model=resolve_serving_model(options),
            engine=engine,
            artifact_store=None,
        )
        route = next(
            route for route in app.routes if getattr(route, "path", None) == "/v1/chat/completions"
        )
        payload = {
            "model": options.model_id,
            "messages": [{"role": "user", "content": "a cat"}],
        }

        with pytest.raises(DiffletServingError) as exc:
            await route.endpoint(payload=payload, request=_DisconnectRequest(engine))

        assert exc.value.code == "request_cancelled"
        assert engine.cancelled.is_set()
        current = asyncio.current_task()
        leaked = [
            task
            for task in asyncio.all_tasks()
            if task is not current and task.get_name().startswith("difflet-http-")
        ]
        assert leaked == []

    asyncio.run(_run())
