from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from difflet.serving.gateway import app as gateway_app
from difflet.serving.gateway.app import client_ip, client_key, create_gateway_app
from difflet.serving.gateway.fleet import FleetConfig, FleetEntry, Limits, Wait
from difflet.serving.gateway.page import _duration, model_defaults

FLUX = "black-forest-labs/FLUX.1-dev"
HV = "hunyuanvideo-community/HunyuanVideo"
LTX = "Lightricks/LTX-2"
NOT_FOUND = {"error": {"message": "Video job was not found", "type": "invalid_request_error",
                       "code": "video_not_found"}}


def _config(tmp_path, **gateway):
    image = ("--height", "1024", "--width", "1024")
    return FleetConfig(host="0.0.0.0", port=8000, runs_dir=tmp_path, models=(
        FleetEntry(FLUX, 0, 8100, Path("/c/0"), image),
        FleetEntry(HV, 3, 8103, Path("/c/3"), ("--shapes", "320x512x61,320x512x33")),
        FleetEntry(LTX, 4, 8104, Path("/c/4"),
                   ("--height", "512", "--width", "768", "--num-frames", "121"))), **gateway)


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


def _client(tmp_path, backends, *, api_key=None, statuses=None, config=None, peer=None):
    statuses = statuses or {FLUX: "ready", HV: "ready", LTX: "ready"}

    async def status_fn():
        return dict(statuses)

    app = create_gateway_app(config or _config(tmp_path), api_key=api_key, status_fn=status_fn,
                             client=httpx.AsyncClient(transport=httpx.MockTransport(backends)))
    return TestClient(app, client=peer or ("testclient", 50000))


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
    for path in ("/", "/health", "/ready", "/ui/models"):
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


def test_queue_counts_active_video_jobs_from_the_backend(tmp_path, backends):
    jobs = [{"id": f"video_gen_{i}", "status": s}
            for i, s in enumerate(("queued", "in_progress", "completed", "failed", "queued"))]
    backends.routes[(8103, "GET", "/v1/videos")] = httpx.Response(
        200, json={"object": "list", "data": jobs, "has_more": False})
    backends.down.add(8104)  # a backend that does not answer shows an empty queue
    with _client(tmp_path, backends) as client:
        assert client.get("/queue").json() == {"models": {FLUX: 0, HV: 3, LTX: 0}}
    listed = [r for r in backends.seen if r.url.port == 8103]
    assert listed[0].url.params["limit"] == "100"


def test_polled_queue_and_statuses_are_shared_between_pages(tmp_path, backends, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(gateway_app, "monotonic", lambda: clock[0])
    backends.routes[(8103, "GET", "/v1/videos")] = httpx.Response(
        200, json={"object": "list", "data": [], "has_more": False})
    with _client(tmp_path, backends) as client:
        for _ in range(5):  # five tabs polling at once
            client.get("/queue")
        assert sum(1 for r in backends.seen if r.url.port == 8103) == 1
        clock[0] += gateway_app._POLL_CACHE_S
        client.get("/queue")
        assert sum(1 for r in backends.seen if r.url.port == 8103) == 2


def test_queue_counts_a_held_request_until_its_backend_answers(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(gateway_app, "monotonic", lambda: clock[0])
    release = asyncio.Event()

    async def backend(request: httpx.Request) -> httpx.Response:
        await request.aread()
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404, json=NOT_FOUND)
        await release.wait()  # the backend is still generating

        async def body():  # streamed like a real socket, as the Backends fixture does
            yield b'{"ok": 1}'

        return httpx.Response(200, headers={"content-type": "application/json"}, content=body())

    async def ready():
        return {}

    app = create_gateway_app(_config(tmp_path), status_fn=ready,
                             client=httpx.AsyncClient(transport=httpx.MockTransport(backend)))

    async def fresh_queue(client):  # past the shared poll window, so /queue asks again
        clock[0] += gateway_app._POLL_CACHE_S
        return (await client.get("/queue")).json()["models"][FLUX]

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://gw") as client:
            pending = asyncio.create_task(client.post("/v1/chat/completions", json=_chat()))
            for _ in range(100):
                if await fresh_queue(client) == 1:
                    break
                await asyncio.sleep(0.01)
            assert await fresh_queue(client) == 1
            release.set()
            assert (await pending).status_code == 200
            assert await fresh_queue(client) == 0

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


SITE = "https://difflet.yottalabs.ai"


def test_cors_preflight_from_the_page_origin_skips_the_api_key(tmp_path, backends):
    client = _client(tmp_path, backends, api_key="k",
                     config=_config(tmp_path, cors_origins=(SITE,)))
    resp = client.options("/v1/chat/completions", headers={
        "Origin": SITE, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-difflet-client"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == SITE
    assert "x-difflet-client" in resp.headers["access-control-allow-headers"].lower()
    assert "DELETE" in resp.headers["access-control-allow-methods"]
    assert backends.seen == []

    denied = client.get("/v1/models", headers={"Origin": SITE})  # a 401 the page can read
    assert denied.status_code == 401 and denied.headers["access-control-allow-origin"] == SITE
    opened = client.get("/ui/models", headers={"Origin": SITE})
    assert opened.headers["access-control-allow-origin"] == SITE
    exposed = opened.headers["access-control-expose-headers"].lower()
    assert "retry-after" in exposed and "x-request-id" in exposed


def test_cors_refuses_another_origin(tmp_path, backends):
    client = _client(tmp_path, backends, config=_config(tmp_path, cors_origins=(SITE,)))
    resp = client.options("/v1/chat/completions", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in resp.headers
    assert "access-control-allow-origin" not in client.get(
        "/ui/models", headers={"Origin": "https://evil.example"}).headers


def test_no_cors_headers_without_origins(tmp_path, backends):
    resp = _client(tmp_path, backends).get("/ui/models", headers={"Origin": SITE})
    assert "access-control-allow-origin" not in resp.headers


def test_ui_models_lists_the_page_models_with_live_status(tmp_path, backends):
    statuses = {FLUX: "ready", HV: "loading", LTX: "down"}
    config = _config(tmp_path, limits=Limits(client_per_hour=5), wait=Wait(60, 120, 300))
    body = _client(tmp_path, backends, statuses=statuses, config=config).get("/ui/models").json()
    assert body["wait"] == {"notice_s": 60, "suggest_s": 120, "reject_s": 300}
    assert body["limits"] == {"client_active": 1, "client_per_hour": 5, "ip_per_hour": 2000}
    flux, hv, ltx = body["models"]
    assert flux == {"id": FLUX, "label": "FLUX.1-dev", "kind": "image", "device": 0,
                    "shapes": [[1024, 1024, None]], "status": "ready", "steps": 28,
                    "guidance": 3.5, "estimate": None}
    assert (hv["id"], hv["status"], hv["shapes"]) == (HV, "loading", [[320, 512, 61], [320, 512, 33]])
    assert ltx["status"] == "down"
    assert backends.seen == []


def _request(peer, *headers):
    return Request({"type": "http", "client": (peer, 1234),
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers]})


def test_client_ip_trusts_forwarded_for_only_behind_a_loopback_peer():
    xff = ("X-Forwarded-For", "6.6.6.6, 1.2.3.4")
    assert client_ip(_request("127.0.0.1", xff)) == "1.2.3.4"  # the proxy appended the last one
    assert client_ip(_request("::1", ("X-Forwarded-For", "6.6.6.6"),
                              ("X-Forwarded-For", "1.2.3.4"))) == "1.2.3.4"
    assert client_ip(_request("9.9.9.9", xff)) == "9.9.9.9"  # a direct client cannot pick one
    assert client_ip(_request("127.0.0.1")) == "127.0.0.1"
    assert client_ip(_request("testclient", xff)) == "testclient"


@pytest.mark.parametrize("given, key", [
    ("abcd1234", "abcd1234"), ("A_b-" * 16, "A_b-" * 16),
    ("abc", "ip:9.9.9.9"), ("x" * 65, "ip:9.9.9.9"), ("../../etc", "ip:9.9.9.9"),
    ("abcd 1234", "ip:9.9.9.9"), ("abcd1234\n", "ip:9.9.9.9")])
def test_client_key_is_the_header_or_the_ip(given, key):
    assert client_key(_request("9.9.9.9", ("X-Difflet-Client", given))) == key
    assert client_key(_request("9.9.9.9")) == "ip:9.9.9.9"


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(gateway_app, "monotonic", lambda: now[0])
    return now


def _as(client_id, **headers):
    return {"X-Difflet-Client": client_id, **headers}


def test_ip_limit_counts_the_forwarded_client_and_comes_first(tmp_path, backends, clock):
    backends.routes[(8100, "POST", "/v1/chat/completions")] = httpx.Response(200, json={"ok": 1})
    config = _config(tmp_path, limits=Limits(client_per_hour=1, ip_per_hour=2))
    client = _client(tmp_path, backends, config=config, peer=("127.0.0.1", 40000))
    venue = {"X-Forwarded-For": "1.2.3.4"}
    for who in ("client-aaaa", "client-bbbb"):
        assert client.post("/v1/chat/completions", json=_chat(),
                           headers=_as(who, **venue)).status_code == 200
    clock[0] += 100
    resp = client.post("/v1/chat/completions", json=_chat(), headers=_as("client-aaaa", **venue))
    assert resp.status_code == 429 and _code(resp) == "ip_rate_limited"  # before client_rate_limited
    assert resp.headers["retry-after"] == "3500"
    assert "2 generations per hour" in resp.json()["error"]["message"]
    assert client.post("/v1/chat/completions", json=_chat(),
                       headers=_as("client-cccc", **{"X-Forwarded-For": "5.6.7.8"})
                       ).status_code == 200  # another network
    assert len(backends.seen) == 3


def test_client_hourly_limit_slides_and_counts_accepted_only(tmp_path, backends, clock):
    route = (8100, "POST", "/v1/chat/completions")
    config = _config(tmp_path, limits=Limits(client_per_hour=2))
    client = _client(tmp_path, backends, config=config)
    backends.routes[route] = httpx.Response(429, json={"error": {"code": "queue_full"}})
    assert client.post("/v1/chat/completions", json=_chat(), headers=_as("me-123456")
                       ).status_code == 429  # refused by the backend: not counted
    backends.routes[route] = httpx.Response(200, json={"ok": 1})
    for _ in range(2):
        assert client.post("/v1/chat/completions", json=_chat(),
                           headers=_as("me-123456")).status_code == 200
        clock[0] += 10
    resp = client.post("/v1/chat/completions", json=_chat(), headers=_as("me-123456"))
    assert resp.status_code == 429 and _code(resp) == "client_rate_limited"
    assert resp.headers["retry-after"] == "3580"  # the first one leaves the window at 1000 + 3600
    assert client.post("/v1/chat/completions", json=_chat(),
                       headers=_as("you-123456")).status_code == 200  # another client
    clock[0] = 1000.0 + 3600
    assert client.post("/v1/chat/completions", json=_chat(),
                       headers=_as("me-123456")).status_code == 200
    assert client.post("/v1/chat/completions", json=_chat(), headers=_as("me-123456")
                       ).status_code == 429  # 1010 and 4600 are in the window
    assert len(backends.seen) == 5


def test_client_busy_while_the_gateway_holds_its_request(tmp_path):
    release = asyncio.Event()

    async def backend(request: httpx.Request) -> httpx.Response:
        await request.aread()
        if request.url.path == "/v1/videos":
            return httpx.Response(200, json={"data": []})
        await release.wait()  # the backend is still generating

        async def body():  # streamed like a real socket, as the Backends fixture does
            yield b'{"ok": 1}'

        return httpx.Response(200, content=body())

    async def ready():
        return {}

    app = create_gateway_app(_config(tmp_path), status_fn=ready,
                             client=httpx.AsyncClient(transport=httpx.MockTransport(backend)))
    video, ctype = _multipart([("model", HV), ("prompt", "x")])

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://gw") as client:
            pending = asyncio.create_task(client.post("/v1/chat/completions", json=_chat(),
                                                      headers=_as("me-123456")))
            for _ in range(100):
                if (await client.get("/queue")).json()["models"][FLUX] == 1:
                    break
                await asyncio.sleep(0.01)
            busy = await client.post("/v1/videos/sync", content=video,
                                     headers=_as("me-123456", **{"Content-Type": ctype}))
            assert busy.status_code == 429 and _code(busy) == "client_busy"
            assert "must finish first" in busy.json()["error"]["message"]
            assert int(busy.headers["retry-after"]) > 0
            release.set()
            assert (await pending).status_code == 200
            assert (await client.post("/v1/chat/completions", json=_chat(),
                                      headers=_as("me-123456"))).status_code == 200

    asyncio.run(run())


def test_client_busy_while_its_async_video_job_is_queued(tmp_path, backends, clock):
    listing = (8103, "GET", "/v1/videos")
    backends.routes[(8103, "POST", "/v1/videos")] = httpx.Response(
        200, json={"id": "video_gen_a", "status": "queued"})
    backends.routes[(8104, "POST", "/v1/videos")] = httpx.Response(200, json={"id": "video_gen_b"})
    backends.routes[listing] = httpx.Response(200, json={"data": [
        {"id": "video_gen_a", "status": "queued"}, {"id": "video_gen_x", "status": "queued"}]})
    client = _client(tmp_path, backends)
    hv, ctype = _multipart([("model", HV), ("prompt", "x")])
    ltx, _ = _multipart([("model", LTX), ("prompt", "x")])
    me = _as("me-123456", **{"Content-Type": ctype})

    first = client.post("/v1/videos", content=hv, headers=me)
    assert first.status_code == 200 and first.json()["id"] == "video_gen_a"  # relayed unchanged
    busy = client.post("/v1/videos", content=ltx, headers=me)
    assert busy.status_code == 429 and _code(busy) == "client_busy"
    assert client.post("/v1/videos", content=ltx, headers=_as("you-123456", **{
        "Content-Type": ctype})).status_code == 200  # someone else's job does not count

    backends.down.add(8103)  # a backend that does not answer: its job is kept, not counted
    assert client.post("/v1/videos", content=ltx, headers=me).status_code == 200
    backends.down.discard(8103)
    backends.routes[(8104, "GET", "/v1/videos")] = httpx.Response(200, json={"data": [
        {"id": "video_gen_b", "status": "completed"}]})
    assert _code(client.post("/v1/videos", content=ltx, headers=me)) == "client_busy"

    backends.routes[listing] = httpx.Response(200, json={"data": [
        {"id": "video_gen_a", "status": "completed"}]})
    assert client.post("/v1/videos", content=hv, headers=me).status_code == 200  # released
    clock[0] += 24 * 3600 + 1  # still queued a day later: forgotten
    backends.routes[listing] = httpx.Response(200, json={"data": [
        {"id": "video_gen_a", "status": "queued"}]})
    assert client.post("/v1/videos", content=ltx, headers=me).status_code == 200


def test_model_busy_names_shorter_waits(tmp_path, backends):
    hv_each = 10 + 1.0 * model_defaults(HV)[0]
    image = ("--height", "1024", "--width", "1024")
    config = FleetConfig(host="0.0.0.0", port=8000, runs_dir=tmp_path, models=(
        FleetEntry(FLUX, 0, 8100, Path("/c/0"), image),
        FleetEntry(HV, 3, 8103, Path("/c/3"), ("--shapes", "320x512x61"), estimate=(10, 1.0)),
        FleetEntry(LTX, 4, 8104, Path("/c/4"),
                   ("--height", "512", "--width", "768", "--num-frames", "121"),
                   estimate=(5, 0.5))), wait=Wait(1, 1, int(2.5 * hv_each)))
    backends.routes[(8103, "GET", "/v1/videos")] = httpx.Response(200, json={"data": [
        {"id": f"video_gen_{i}", "status": "queued"} for i in range(3)]})
    backends.routes[(8104, "GET", "/v1/videos")] = httpx.Response(200, json={"data": []})
    body, ctype = _multipart([("model", HV), ("prompt", "x")])
    resp = _client(tmp_path, backends, config=config).post(
        "/v1/videos", content=body, headers={"Content-Type": ctype})
    assert resp.status_code == 429 and _code(resp) == "model_busy"
    message = resp.json()["error"]["message"]
    assert HV in message and f"{LTX} (no wait)" in message and FLUX not in message
    assert f"over {_duration(int(2.5 * hv_each))}" in message  # the configured threshold
    assert resp.headers["retry-after"] == str(round(hv_each))  # one request ahead must finish
    assert [r.method for r in backends.seen] == ["GET"] * len(backends.seen)  # nothing forwarded
