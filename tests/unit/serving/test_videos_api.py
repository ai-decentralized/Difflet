from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

import pytest

from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.openai.api_server import create_app
from difflet.serving.options import ServeOptions
from difflet.serving.video_jobs import InMemoryVideoJobRepository
from difflet.serving.video_service import VideoGenerationService
from difflet.serving.video_storage import LocalVideoArtifactStore, S3VideoArtifactStore
from tests.unit.serving.test_video_service import (
    _FakeVideoEngine,
    _patch_media_validation_without_pyav,
    _request,
)
from tests.unit.serving.test_video_storage import _FakeS3Client, _FakeS3Store


class _NoopVideoService:
    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _PromptControlledEngine(_FakeVideoEngine):
    async def generate(self, request):
        if request.prompt == "hold":
            self.blocked_ids.add(request.request_id)
        return await super().generate(request)


def _resolved_video(tmp_path):
    options = ServeOptions(
        model_id="Lightricks/LTX-2",
        cache_dir=str(tmp_path),
        max_queued_requests=4,
    )
    resolved = resolve_serving_model(options)
    resolved = replace(
        resolved,
        profile=replace(
            resolved.profile,
            width=16,
            height=16,
            num_frames=2,
            output_fps=2,
        ),
    )
    return options, resolved


def _video_app(tmp_path, monkeypatch, *, engine=None, artifacts=None):
    _patch_media_validation_without_pyav(monkeypatch)
    options, resolved = _resolved_video(tmp_path)
    engine = engine or _FakeVideoEngine()
    jobs = InMemoryVideoJobRepository()
    artifacts = artifacts or LocalVideoArtifactStore(tmp_path / "media")
    service = VideoGenerationService(
        engine=engine,
        jobs=jobs,
        artifacts=artifacts,
        max_queued_requests=options.max_queued_requests,
        queue_timeout_s=options.effective_queue_timeout("video"),
        request_timeout_s=options.request_timeout,
        recovery_timeout_s=1.0,
    )
    app = create_app(
        options=options,
        resolved_model=resolved,
        engine=engine,
        video_service=service,
    )
    return app, resolved, service, jobs, artifacts, engine


def _multipart(*items: tuple[str, str]):
    return [(name, (None, value)) for name, value in items]


def _wait_for_status(client, video_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 3.0
    while True:
        response = client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] == expected:
            return body
        if time.monotonic() >= deadline:
            raise AssertionError(f"{video_id} remained {body['status']!r}; expected {expected!r}")
        time.sleep(0.01)


def _route_methods(app, prefix: str) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in app.routes
        if getattr(route, "path", "").startswith(prefix)
        for method in getattr(route, "methods", ())
    }


def test_video_models_register_exactly_six_video_methods_and_image_models_only_chat(
    tmp_path,
):
    video_options, video_resolved = _resolved_video(tmp_path / "video")
    video_app = create_app(
        options=video_options,
        resolved_model=video_resolved,
        engine=_FakeVideoEngine(),
        video_service=_NoopVideoService(),
    )

    image_options = ServeOptions(model_id="black-forest-labs/FLUX.1-dev")
    image_app = create_app(
        options=image_options,
        resolved_model=resolve_serving_model(image_options),
        engine=_FakeVideoEngine(),
        artifact_store=object(),
    )

    assert _route_methods(video_app, "/v1/videos") == {
        ("POST", "/v1/videos"),
        ("POST", "/v1/videos/sync"),
        ("GET", "/v1/videos"),
        ("GET", "/v1/videos/{video_id}"),
        ("GET", "/v1/videos/{video_id}/content"),
        ("DELETE", "/v1/videos/{video_id}"),
    }
    assert _route_methods(video_app, "/v1/chat/completions") == set()
    assert _route_methods(image_app, "/v1/chat/completions") == {("POST", "/v1/chat/completions")}
    assert _route_methods(image_app, "/v1/videos") == set()


def test_sync_video_returns_raw_mp4_headers_and_leaves_no_job_or_artifact(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    app, resolved, _, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/sync",
            files=_multipart(
                ("model", resolved.model_id),
                ("prompt", "sync clip"),
                ("size", "16x16"),
                ("num_frames", "2"),
                ("fps", "2"),
            ),
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert response.headers["x-model"] == resolved.model_id
        assert response.headers["x-request-id"].startswith("video_sync_")
        assert float(response.headers["x-inference-time-s"]) >= 0
        assert int(response.headers["content-length"]) == len(response.content)
        assert response.content == engine.payloads[response.headers["x-request-id"]]
        assert jobs.count() == 0
        assert tuple(artifacts.artifact_root.iterdir()) == ()
        assert tuple(artifacts.staging_root.iterdir()) == ()

    assert engine.start_calls == engine.shutdown_calls == 1


def test_oversized_integer_form_field_returns_400_without_admission(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    app, resolved, _, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            files=_multipart(
                ("model", resolved.model_id),
                ("prompt", "bounded integer"),
                ("seed", "9" * 10_000),
            ),
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_extra_body"
        assert jobs.count() == 0
        assert engine.calls == []
        assert tuple(artifacts.artifact_root.iterdir()) == ()


def test_oversized_list_limit_returns_stable_400(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app, _, _, jobs, _, engine = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/v1/videos", params={"limit": "9" * 10_000})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert jobs.count() == 0
        assert engine.calls == []


def test_sync_video_cleans_artifact_when_asgi_send_fails(tmp_path, monkeypatch):
    import difflet.serving.openai.serving_video as serving_video
    from starlette.requests import Request

    async def fake_normalize(raw_request, *, resolved_model, request_id):
        return _request(request_id)

    monkeypatch.setattr(serving_video, "normalize_video_multipart_request", fake_normalize)
    app, _, service, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/videos/sync"
    )

    async def _run() -> None:
        never_received = asyncio.Event()

        async def receive():
            await never_received.wait()

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/videos/sync",
                "raw_path": b"/v1/videos/sync",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            },
            receive,
        )

        async def fail_body_send(message):
            if message["type"] == "http.response.body":
                raise RuntimeError("simulated ASGI send failure")

        await engine.start()
        await service.start()
        try:
            response = await endpoint(request)
            with pytest.raises(Exception, match="simulated ASGI send failure"):
                await response(request.scope, receive, fail_body_send)

            assert jobs.count() == 0
            assert tuple(artifacts.artifact_root.iterdir()) == ()
            assert tuple(artifacts.staging_root.iterdir()) == ()
        finally:
            await service.shutdown()
            await engine.shutdown()

    asyncio.run(_run())


def test_sync_video_cleanup_survives_repeated_response_cancellation(tmp_path, monkeypatch):
    import difflet.serving.openai.serving_video as serving_video
    from starlette.requests import Request

    async def fake_normalize(raw_request, *, resolved_model, request_id):
        return _request(request_id)

    monkeypatch.setattr(serving_video, "normalize_video_multipart_request", fake_normalize)
    app, _, service, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/videos/sync"
    )
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    real_delete_sync_result = service.delete_sync_result

    async def blocking_delete_sync_result(result):
        cleanup_entered.set()
        await release_cleanup.wait()
        await real_delete_sync_result(result)

    monkeypatch.setattr(service, "delete_sync_result", blocking_delete_sync_result)

    async def _run() -> None:
        never_received = asyncio.Event()

        async def receive():
            await never_received.wait()

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/videos/sync",
                "raw_path": b"/v1/videos/sync",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            },
            receive,
        )

        async def send(message):
            return None

        await engine.start()
        await service.start()
        try:
            response = await endpoint(request)
            response_task = asyncio.create_task(response(request.scope, receive, send))
            await asyncio.wait_for(cleanup_entered.wait(), timeout=3.0)
            assert len(tuple(artifacts.artifact_root.iterdir())) == 1

            response_task.cancel()
            await asyncio.sleep(0)
            response_task.cancel()
            await asyncio.sleep(0)
            assert response_task.done() is False
            assert len(tuple(artifacts.artifact_root.iterdir())) == 1

            release_cleanup.set()
            await asyncio.wait_for(response_task, timeout=3.0)
            assert jobs.count() == 0
            assert tuple(artifacts.artifact_root.iterdir()) == ()
            assert tuple(artifacts.staging_root.iterdir()) == ()
        finally:
            release_cleanup.set()
            await service.shutdown()
            await engine.shutdown()

    asyncio.run(_run())


def test_sync_video_cleans_artifact_when_cancelled_during_open(tmp_path, monkeypatch):
    import difflet.serving.openai.serving_video as serving_video
    from starlette.requests import Request

    async def fake_normalize(raw_request, *, resolved_model, request_id):
        return _request(request_id)

    monkeypatch.setattr(serving_video, "normalize_video_multipart_request", fake_normalize)
    app, _, service, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/videos/sync"
    )
    open_entered = threading.Event()
    release_open = threading.Event()
    real_open = artifacts.open

    def blocking_open(artifact_key):
        open_entered.set()
        assert release_open.wait(timeout=3.0)
        return real_open(artifact_key)

    monkeypatch.setattr(artifacts, "open", blocking_open)

    async def _run() -> None:
        never_received = asyncio.Event()

        async def receive():
            await never_received.wait()

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/videos/sync",
                "raw_path": b"/v1/videos/sync",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            },
            receive,
        )

        await engine.start()
        await service.start()
        try:
            route_task = asyncio.create_task(endpoint(request))
            assert await asyncio.to_thread(open_entered.wait, 3.0)
            route_task.cancel()
            release_open.set()
            with pytest.raises(asyncio.CancelledError):
                await route_task

            assert jobs.count() == 0
            assert tuple(artifacts.artifact_root.iterdir()) == ()
            assert tuple(artifacts.staging_root.iterdir()) == ()
        finally:
            release_open.set()
            await service.shutdown()
            await engine.shutdown()

    asyncio.run(_run())


def test_async_video_multipart_status_list_content_headers_and_delete(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    app, resolved, _, jobs, artifacts, engine = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        created_response = client.post(
            "/v1/videos",
            files=_multipart(
                ("model", resolved.model_id),
                ("prompt", "async clip"),
                ("size", "16x16"),
                ("num_frames", "2"),
                ("fps", "2"),
                ("user", "api-caller"),
            ),
        )
        assert created_response.status_code == 200
        created = created_response.json()
        assert created["object"] == "video"
        assert created["status"] == "queued"
        assert created["model"] == resolved.model_id
        assert created["size"] == "16x16"
        assert set(created) == {
            "id",
            "object",
            "status",
            "model",
            "prompt",
            "size",
            "seconds",
            "progress",
            "quality",
            "created_at",
            "completed_at",
            "remixed_from_video_id",
            "error",
            "url",
            "expires_at",
        }
        video_id = created["id"]

        completed = _wait_for_status(client, video_id, "completed")
        assert completed["progress"] == 100
        assert "duration_s" not in completed
        assert "file_name" not in completed
        assert "file_size_bytes" not in completed
        assert "inference_time_s" not in completed

        listed_response = client.get("/v1/videos", params={"limit": 1})
        assert listed_response.status_code == 200
        listed = listed_response.json()
        assert listed["object"] == "list"
        assert listed["first_id"] == listed["last_id"] == video_id
        assert listed["has_more"] is False
        assert [item["id"] for item in listed["data"]] == [video_id]

        content = client.get(f"/v1/videos/{video_id}/content")
        assert content.status_code == 200
        assert content.headers["content-type"] == "video/mp4"
        assert content.headers["content-disposition"] == (f'attachment; filename="{video_id}.mp4"')
        assert int(content.headers["content-length"]) == len(content.content)
        assert content.content == engine.payloads[video_id]

        deleted_response = client.delete(f"/v1/videos/{video_id}")
        assert deleted_response.status_code == 200
        assert deleted_response.json() == {
            "id": video_id,
            "deleted": True,
            "object": "video.deleted",
        }
        assert jobs.get(video_id) is None
        assert artifacts.get(f"{video_id}.mp4") is None
        assert client.get(f"/v1/videos/{video_id}").status_code == 404


def test_async_s3_publication_failure_completes_with_local_content(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    s3_client = _FakeS3Client(upload_error=OSError("upload unavailable"))
    artifacts = S3VideoArtifactStore(
        tmp_path / "media",
        s3=_FakeS3Store(s3_client),  # type: ignore[arg-type]
    )
    app, resolved, _, _, _, engine = _video_app(
        tmp_path,
        monkeypatch,
        artifacts=artifacts,
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/videos",
            files=_multipart(
                ("model", resolved.model_id),
                ("prompt", "local fallback"),
            ),
        ).json()
        completed = _wait_for_status(client, created["id"], "completed")

        assert completed["url"] is None
        content = client.get(f"/v1/videos/{created['id']}/content")
        assert content.status_code == 200
        assert content.content == engine.payloads[created["id"]]


def test_video_jobs_and_outputs_disappear_after_server_lifecycle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    first_app, resolved, _, _, first_artifacts, _ = _video_app(tmp_path, monkeypatch)
    with TestClient(first_app) as first_client:
        created_response = first_client.post(
            "/v1/videos",
            files=_multipart(
                ("model", resolved.model_id),
                ("prompt", "ephemeral clip"),
                ("size", "16x16"),
                ("num_frames", "2"),
                ("fps", "2"),
            ),
        )
        assert created_response.status_code == 200
        video_id = created_response.json()["id"]
        _wait_for_status(first_client, video_id, "completed")
        assert first_artifacts.get(f"{video_id}.mp4") is not None

    assert first_artifacts.get(f"{video_id}.mp4") is None

    second_app, _, _, second_jobs, second_artifacts, _ = _video_app(tmp_path, monkeypatch)
    with TestClient(second_app) as second_client:
        listed = second_client.get("/v1/videos")
        retrieved = second_client.get(f"/v1/videos/{video_id}")
        content = second_client.get(f"/v1/videos/{video_id}/content")

        assert listed.status_code == 200
        assert listed.json()["data"] == []
        assert retrieved.status_code == 404
        assert content.status_code == 404
        assert second_jobs.count() == 0
        assert second_artifacts.get(f"{video_id}.mp4") is None

    assert tuple(tmp_path.rglob("*.sqlite*")) == ()


def test_content_endpoint_distinguishes_pending_failed_and_completed_jobs(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    engine = _PromptControlledEngine(failing_prompts={"fail"})
    app, _, _, _, _, _ = _video_app(tmp_path, monkeypatch, engine=engine)
    with TestClient(app) as client:
        pending = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "hold")),
        ).json()
        pending_content = client.get(f"/v1/videos/{pending['id']}/content")
        assert pending_content.status_code == 409
        assert pending_content.json()["error"]["code"] == "video_not_ready"
        deleting = client.delete(f"/v1/videos/{pending['id']}")
        assert deleting.status_code == 409
        assert deleting.json()["error"]["code"] == "video_in_progress"
        engine.release.set()
        _wait_for_status(client, pending["id"], "completed")
        assert client.delete(f"/v1/videos/{pending['id']}").status_code == 200

        failed = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "fail")),
        ).json()
        _wait_for_status(client, failed["id"], "failed")
        failed_content = client.get(f"/v1/videos/{failed['id']}/content")
        assert failed_content.status_code == 422
        assert failed_content.json()["error"]["code"] == "video_generation_failed"

        completed = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "complete")),
        ).json()
        _wait_for_status(client, completed["id"], "completed")
        assert client.get(f"/v1/videos/{completed['id']}/content").status_code == 200


def test_video_endpoints_enforce_multipart_fields_and_list_query_contract(
    tmp_path,
    monkeypatch,
):
    from fastapi.testclient import TestClient

    app, _, _, jobs, _, _ = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        wrong_type = client.post("/v1/videos", json={"prompt": "not multipart"})
        assert wrong_type.status_code == 400
        assert wrong_type.json()["error"]["code"] == "invalid_request"

        repeated = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "first"), ("prompt", "second")),
        )
        assert repeated.status_code == 400
        assert repeated.json()["error"]["code"] == "invalid_request"

        unsupported = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "clip"), ("generate_sound", "false")),
        )
        assert unsupported.status_code == 400
        assert unsupported.json()["error"]["code"] == "feature_not_supported"

        invalid_list = client.get("/v1/videos", params={"limit": 0})
        assert invalid_list.status_code == 400
        assert invalid_list.json()["error"]["code"] == "invalid_request"

        unknown_list_field = client.get("/v1/videos", params={"status": "queued"})
        assert unknown_list_field.status_code == 400
        assert unknown_list_field.json()["error"]["code"] == "invalid_request"
        assert jobs.count() == 0


def test_video_multipart_body_part_field_and_file_limits(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app, _, _, jobs, _, _ = _video_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        declared_oversize = client.post(
            "/v1/videos",
            content=b"ignored",
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "content-length": str(1024 * 1024 + 1),
            },
        )
        assert declared_oversize.status_code == 413
        assert declared_oversize.json()["error"]["code"] == "request_too_large"

        large_part = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "x" * (256 * 1024 + 1))),
        )
        assert large_part.status_code == 413
        assert large_part.json()["error"]["code"] == "request_too_large"

        too_many_fields = client.post(
            "/v1/videos",
            files=_multipart(("prompt", "clip"), *[("user", str(i)) for i in range(32)]),
        )
        assert too_many_fields.status_code == 413
        assert too_many_fields.json()["error"]["code"] == "request_too_large"

        upload = client.post(
            "/v1/videos",
            files={"input_reference": ("reference.png", b"png", "image/png")},
        )
        assert upload.status_code == 400
        assert upload.json()["error"]["code"] == "feature_not_supported"
        assert jobs.count() == 0
