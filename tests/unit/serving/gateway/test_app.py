from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from difflet.serving.gateway import app as gateway_app
from difflet.serving.gateway.app import create_gateway_app
from difflet.serving.gateway.fleet import FleetConfig, FleetEntry

FLUX = "black-forest-labs/FLUX.1-dev"
HV = "hunyuanvideo-community/HunyuanVideo"
LTX = "Lightricks/LTX-2"
NOT_FOUND = {"error": {"message": "Video job was not found", "type": "invalid_request_error",
                       "code": "video_not_found"}}


def _config(tmp_path):
    image = ("--height", "1024", "--width", "1024")
    return FleetConfig(host="0.0.0.0", port=8000, runs_dir=tmp_path, models=(
        FleetEntry(FLUX, 0, 8100, Path("/c/0"), image),
        FleetEntry(HV, 3, 8103, Path("/c/3"), ("--shapes", "320x512x61,320x512x33")),
        FleetEntry(LTX, 4, 8104, Path("/c/4"),
                   ("--height", "512", "--width", "768", "--num-frames", "121"))))


class Backends:
    """The fake ``difflet serve`` processes, one per port: records every request it gets."""

    def __init__(self):
        self.seen: list[httpx.Request] = []
        self.routes: dict[tuple[int, str, str], httpx.Response] = {}
        self.down: set[int] = set()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.seen.append(request)
        if request.url.port in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        key = (request.url.port, request.method, request.url.path)
        answer = self.routes.get(key) or httpx.Response(404, json=NOT_FOUND)

        async def body():  # streamed like a real socket (a bytes body is read eagerly)
            yield answer.content

        return httpx.Response(answer.status_code, headers=answer.headers, content=body())

    def ports(self):
        return [r.url.port for r in self.seen]


@pytest.fixture
def backends():
    return Backends()


def _client(tmp_path, backends, *, api_key=None, statuses=None):
    statuses = statuses or {FLUX: "ready", HV: "ready", LTX: "ready"}

    async def status_fn():
        return dict(statuses)

    app = create_gateway_app(_config(tmp_path), api_key=api_key, status_fn=status_fn,
                             client=httpx.AsyncClient(transport=httpx.MockTransport(backends)))
    return TestClient(app)


def _chat(model=FLUX):
    return {"model": model, "messages": [{"role": "user", "content": "a boat"}],
            "extra_body": {"height": 1024, "width": 1024}}


def _multipart(fields, files=()):
    boundary = "difflet-test-boundary"
    out = b""
    for name, value in fields:
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n").encode()
    for name, filename, data in files:
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n').encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return out, f"multipart/form-data; boundary={boundary}"


def _code(resp):
    return resp.json()["error"]["code"]


def test_chat_routes_by_model(tmp_path, backends):
    backends.routes[(8100, "POST", "/v1/chat/completions")] = httpx.Response(
        200, json={"id": "chatcmpl-1"}, headers={"X-Request-Id": "r1"})
    body = json.dumps(_chat(FLUX + "/")).encode()
    resp = _client(tmp_path, backends).post(
        "/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.json() == {"id": "chatcmpl-1"}
    assert resp.headers["x-request-id"] == "r1"
    assert backends.ports() == [8100]
    assert backends.seen[0].content == body


def test_chat_model_required(tmp_path, backends):
    client = _client(tmp_path, backends)
    assert _code(client.post("/v1/chat/completions", json={"messages": []})) == "model_required"
    resp = client.post("/v1/chat/completions", content=b"not json")
    assert resp.status_code == 400 and _code(resp) == "model_required"
    assert backends.seen == []


def test_chat_unknown_model_404(tmp_path, backends):
    resp = _client(tmp_path, backends).post("/v1/chat/completions", json=_chat("nobody/none"))
    assert resp.status_code == 404 and _code(resp) == "model_not_found"
    assert FLUX in resp.json()["error"]["message"]


def test_chat_video_model_wrong_endpoint(tmp_path, backends):
    resp = _client(tmp_path, backends).post("/v1/chat/completions", json=_chat(HV))
    assert resp.status_code == 400 and _code(resp) == "wrong_endpoint"
    assert "/v1/videos" in resp.json()["error"]["message"]
    assert backends.seen == []


def test_video_post_with_image_model_wrong_endpoint(tmp_path, backends):
    body, ctype = _multipart([("model", FLUX), ("prompt", "x")])
    resp = _client(tmp_path, backends).post("/v1/videos", content=body,
                                            headers={"Content-Type": ctype})
    assert resp.status_code == 400 and _code(resp) == "wrong_endpoint"


def test_multipart_with_file_forwarded_byte_identical(tmp_path, backends):
    backends.routes[(8104, "POST", "/v1/videos")] = httpx.Response(200, json={"id": "video_gen_1"})
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
    body, ctype = _multipart([("model", LTX), ("prompt", "a hill"), ("size", "768x512")],
                             files=[("first_frame", "f.png", png)])
    resp = _client(tmp_path, backends).post("/v1/videos", content=body,
                                            headers={"Content-Type": ctype})
    assert resp.status_code == 200 and resp.json() == {"id": "video_gen_1"}
    (seen,) = backends.seen
    assert seen.url.port == 8104
    assert seen.content == body
    assert seen.headers["content-type"] == ctype


def test_video_post_model_required(tmp_path, backends):
    body, ctype = _multipart([("prompt", "x")])
    client = _client(tmp_path, backends)
    assert _code(client.post("/v1/videos", content=body,
                             headers={"Content-Type": ctype})) == "model_required"
    assert _code(client.post("/v1/videos/sync", content=b"garbage",
                             headers={"Content-Type": "multipart/form-data; boundary=x"})
                 ) == "model_required"


def test_videos_sync_streams_bytes(tmp_path, backends):
    mp4 = b"\x00\x00\x00\x18ftypmp42" * 5000
    backends.routes[(8103, "POST", "/v1/videos/sync")] = httpx.Response(
        200, content=mp4, headers={"Content-Type": "video/mp4", "X-Model": HV})
    body, ctype = _multipart([("model", HV), ("prompt", "x")])
    resp = _client(tmp_path, backends).post("/v1/videos/sync", content=body,
                                            headers={"Content-Type": ctype})
    assert resp.status_code == 200
    assert resp.content == mp4
    assert resp.headers["content-type"] == "video/mp4" and resp.headers["x-model"] == HV


def test_backend_error_passed_through(tmp_path, backends):
    error = {"error": {"message": "queue is full", "type": "server_error", "code": "queue_full"}}
    backends.routes[(8100, "POST", "/v1/chat/completions")] = httpx.Response(429, json=error)
    resp = _client(tmp_path, backends).post("/v1/chat/completions", json=_chat())
    assert resp.status_code == 429 and resp.json() == error


def test_body_too_large_413(tmp_path, backends, monkeypatch):
    monkeypatch.setattr(gateway_app, "MAX_BODY_BYTES", 100)
    resp = _client(tmp_path, backends).post("/v1/chat/completions", content=b"x" * 101)
    assert resp.status_code == 413 and _code(resp) == "request_too_large"
    assert backends.seen == []


def test_authorization_not_forwarded(tmp_path, backends):
    backends.routes[(8100, "POST", "/v1/chat/completions")] = httpx.Response(200, json={})
    _client(tmp_path, backends).post("/v1/chat/completions", json=_chat(), headers={
        "Authorization": "Bearer k", "X-Request-Id": "abc", "Connection": "keep-alive"})
    (seen,) = backends.seen
    assert "authorization" not in seen.headers
    assert seen.headers["x-request-id"] == "abc"
    assert seen.headers["host"] == "127.0.0.1:8100"


def test_api_key_required_on_v1_only(tmp_path, backends):
    client = _client(tmp_path, backends, api_key="k")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer k"}).status_code == 200
    for path in ("/", "/health", "/ready"):
        assert client.get(path).status_code == 200


def test_video_id_fan_out(tmp_path, backends):
    backends.routes[(8104, "GET", "/v1/videos/video_gen_1")] = httpx.Response(
        200, json={"id": "video_gen_1", "status": "completed"})
    resp = _client(tmp_path, backends).get("/v1/videos/video_gen_1")
    assert resp.status_code == 200 and resp.json()["status"] == "completed"
    assert backends.ports() == [8103, 8104]  # video backends only, in config order


def test_video_content_streams_from_owner(tmp_path, backends):
    mp4 = b"mp4-bytes" * 10000
    backends.routes[(8103, "GET", "/v1/videos/video_gen_2/content")] = httpx.Response(
        200, content=mp4, headers={"Content-Type": "video/mp4"})
    resp = _client(tmp_path, backends).get("/v1/videos/video_gen_2/content")
    assert resp.status_code == 200 and resp.content == mp4
    assert backends.ports() == [8103]


def test_video_delete_fan_out(tmp_path, backends):
    backends.routes[(8104, "DELETE", "/v1/videos/video_gen_3")] = httpx.Response(
        200, json={"id": "video_gen_3", "deleted": True})
    resp = _client(tmp_path, backends).delete("/v1/videos/video_gen_3")
    assert resp.status_code == 200 and resp.json()["deleted"] is True
    assert [r.method for r in backends.seen] == ["DELETE", "DELETE"]


def test_unknown_video_id_404(tmp_path, backends):
    resp = _client(tmp_path, backends).get("/v1/videos/video_gen_x")
    assert resp.status_code == 404 and resp.json() == NOT_FOUND


def test_video_lookup_with_a_backend_down_is_503(tmp_path, backends):
    backends.down.add(8104)
    resp = _client(tmp_path, backends).get("/v1/videos/video_gen_x")
    assert resp.status_code == 503 and _code(resp) == "model_unavailable"
    assert LTX in resp.json()["error"]["message"]


def test_list_videos_requires_model_and_strips_it(tmp_path, backends):
    backends.routes[(8103, "GET", "/v1/videos")] = httpx.Response(200, json={"data": []})
    client = _client(tmp_path, backends)
    assert _code(client.get("/v1/videos")) == "model_required"
    resp = client.get("/v1/videos", params={"model": HV, "limit": "5", "after": "video_gen_1"})
    assert resp.status_code == 200 and resp.json() == {"data": []}
    (seen,) = backends.seen
    assert seen.url.port == 8103
    assert dict(seen.url.params) == {"limit": "5", "after": "video_gen_1"}


def test_backend_down_503(tmp_path, backends):
    backends.down.add(8100)
    resp = _client(tmp_path, backends).post("/v1/chat/completions", json=_chat())
    assert resp.status_code == 503 and _code(resp) == "model_unavailable"
    assert "down" in resp.json()["error"]["message"]


def test_ready_aggregate(tmp_path, backends):
    assert _client(tmp_path, backends).get("/ready").status_code == 200
    statuses = {FLUX: "ready", HV: "loading", LTX: "ready"}
    resp = _client(tmp_path, backends, statuses=statuses).get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready", "models": statuses}


def test_models_lists_config(tmp_path, backends):
    statuses = {FLUX: "ready", HV: "loading", LTX: "down"}
    data = _client(tmp_path, backends, statuses=statuses).get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == [FLUX, HV, LTX]
    assert data[0]["difflet_kind"] == "image" and data[0]["difflet_shapes"] == ["1024x1024"]
    assert data[1]["difflet_shapes"] == ["320x512x61", "320x512x33"]
    assert data[1]["difflet_device"] == 3 and data[1]["difflet_status"] == "loading"
    assert backends.seen == []


def test_page_served(tmp_path, backends):
    resp = _client(tmp_path, backends).get("/")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    assert "Difflet models" in resp.text and "320×512×61" in resp.text


def test_compare_page_served_and_linked_without_the_api_key(tmp_path, backends):
    client = _client(tmp_path, backends, api_key="secret")
    assert 'href="compare"' in client.get("/").text
    resp = client.get("/compare")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    assert "RTX PRO 6000" in resp.text and backends.seen == []


def test_client_disconnect_cancels_upstream(tmp_path):
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def backend(request):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def status_fn():
            return {}

        app = create_gateway_app(_config(tmp_path), status_fn=status_fn,
                                 client=httpx.AsyncClient(transport=httpx.MockTransport(backend)))
        body = json.dumps(_chat()).encode()
        pending = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if pending:
                return pending.pop()
            await started.wait()  # the client goes away while the backend is still working
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
                 "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
                 "headers": [(b"host", b"gw"), (b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 5000), "server": ("gw", 8000)}
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        assert cancelled.is_set()

    asyncio.run(run())


def test_stats_count_successful_generations_across_restarts(tmp_path, backends):
    backends.routes[(8100, "POST", "/v1/chat/completions")] = httpx.Response(200, json={"ok": 1})
    with _client(tmp_path, backends) as client:
        assert client.get("/stats").json() == {"total_runs": 0}
        assert client.post("/v1/chat/completions", json=_chat()).status_code == 200
        assert client.post("/v1/chat/completions", json=_chat(LTX)).status_code == 400  # not counted
        assert client.get("/stats").json() == {"total_runs": 1}

    with _client(tmp_path, backends) as client:  # a restarted gateway keeps the count
        assert client.get("/stats").json() == {"total_runs": 1}
        assert 'id="runs">1<' in client.get("/").text
