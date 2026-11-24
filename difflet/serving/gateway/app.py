"""The gateway: one endpoint in front of the fleet's ``difflet serve`` backends (design §6.4).

It reads the request's ``model`` to pick the backend and forwards the request bytes unchanged;
the backend's status and body come back unchanged. It never times out a forwarded request on its
own (a video can wait hours in its backend's queue); a client that goes away while the backend is
still working closes the upstream request, so the backend cancels it.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote, urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.requests import Request as StarletteRequest

from difflet.serving.auth import ApiKeyAuthenticationMiddleware
from difflet.serving.errors import DiffletServingError
from difflet.serving.gateway.fleet import FleetConfig, FleetEntry
from difflet.serving.gateway.launcher import read_pid
from difflet.serving.gateway.page import render_page, ui_models
from difflet.serving.gateway.status import Status, fleet_status
from difflet.serving.openai.api_server import _run_until_disconnect
from difflet.serving.openai.serving_video import MAX_KEYFRAME_FORM_BODY_BYTES

#: The largest request body any backend accepts (a video form with two keyframes).
MAX_BODY_BYTES = MAX_KEYFRAME_FORM_BODY_BYTES
_HOP_BY_HOP = {b"connection", b"keep-alive", b"transfer-encoding", b"te", b"trailer", b"upgrade",
               b"proxy-authorization", b"proxy-authenticate"}
_DROP_REQUEST = _HOP_BY_HOP | {b"host", b"content-length", b"authorization"}
_DROP_RESPONSE = _HOP_BY_HOP | {b"date", b"server"}  # uvicorn sends its own


def create_gateway_app(
    config: FleetConfig,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    status_fn: Callable[[], Awaitable[dict[str, Status]]] | None = None,
) -> FastAPI:
    http = client

    @asynccontextmanager
    async def lifespan(app):
        nonlocal http
        if client is not None:
            yield
            return
        # No read timeout: the backend owns its queue and request timeouts. No pool cap: a slow
        # video download must not make every other request wait for a connection.
        http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0),
                                 limits=httpx.Limits(max_connections=None))
        try:
            yield
        finally:
            await http.aclose()

    async def statuses() -> dict[str, Status]:
        return await (status_fn() if status_fn is not None else fleet_status(config, http))

    runs = RunCounter(config.runs_dir / "gateway-stats.json")
    #: Requests the gateway holds open per model (chat completions, /v1/videos/sync): a backend
    #: answers them only when the generation ends, so each counts while it waits and while it runs.
    inflight = {entry.model_id: 0 for entry in config.models}
    app = FastAPI(title="Difflet gateway", lifespan=lifespan)
    if api_key:
        app.add_middleware(ApiKeyAuthenticationMiddleware, api_key=api_key)
    if config.cors_origins:
        # Added last, so outermost: a preflight never meets the API key, and a 401 still carries
        # the CORS headers the page needs to read it.
        app.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins),
                           allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                           allow_headers=["Content-Type", "Authorization", "X-Difflet-Client"],
                           expose_headers=["Retry-After", "X-Request-Id"])

    @app.exception_handler(DiffletServingError)
    async def _serving_error(request: Request, exc: DiffletServingError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    def route(model, kind: str) -> FleetEntry:
        served = ", ".join(e.model_id for e in config.models)
        if not isinstance(model, str) or not model:
            raise DiffletServingError(400, "model_required",
                                      f"the request must name its model; served: {served}")
        entry = config.entry(model)
        if entry is None:
            raise DiffletServingError(404, "model_not_found",
                                      f"model {model!r} is not served here; served: {served}")
        if entry.kind != kind:
            right = ("a video model; use /v1/videos" if entry.kind == "video"
                     else "an image model; use /v1/chat/completions")
            raise DiffletServingError(400, "wrong_endpoint", f"{entry.model_id} is {right}")
        return entry

    async def send(request: Request, entry: FleetEntry, path: str, *, body: bytes = b"",
                   query: str | None = None) -> httpx.Response:
        query = request.url.query if query is None else query
        upstream = http.build_request(
            request.method, entry.base_url + path + (f"?{query}" if query else ""),
            headers=[(k, v) for k, v in request.headers.raw if k.lower() not in _DROP_REQUEST],
            content=body)
        try:
            return await _run_until_disconnect(http.send(upstream, stream=True), request=request)
        except httpx.TransportError as exc:
            loading = read_pid(config.pid_file(entry)) is not None
            raise DiffletServingError(
                503, "model_unavailable",
                f"{entry.model_id} is not answering: its backend is "
                + ("still loading; try again later" if loading else "down"),
                "server_error") from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await _read_body(request)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        entry = route(payload.get("model") if isinstance(payload, dict) else None, "image")
        return _relay(runs.count(await held(request, entry, "/v1/chat/completions", body)))

    @app.post("/v1/videos")
    @app.post("/v1/videos/sync")
    async def create_video(request: Request):
        body = await _read_body(request)
        entry = route(await _form_model(request, body), "video")
        if request.url.path.endswith("/sync"):
            return _relay(runs.count(await held(request, entry, request.url.path, body)))
        return _relay(runs.count(await send(request, entry, request.url.path, body=body)))

    async def held(request: Request, entry: FleetEntry, path: str, body: bytes) -> httpx.Response:
        """``send`` for a request the backend answers when its generation ends, counted in
        ``inflight`` until then."""
        inflight[entry.model_id] += 1
        try:
            return await send(request, entry, path, body=body)
        finally:
            inflight[entry.model_id] -= 1

    async def active(entry: FleetEntry) -> int:
        """Requests running or waiting on ``entry``: the ones the gateway holds open, plus a video
        backend's async jobs still queued or in progress (its own list, so API users count too)."""
        count = inflight[entry.model_id]
        if entry.kind != "video":
            return count
        try:
            # shortcut: the newest 100 jobs only (the list is newest first); page with `after`
            # if a queue can exceed 100.
            resp = await http.get(entry.base_url + "/v1/videos", params={"limit": 100},
                                  timeout=5.0)
            if resp.status_code == 200:
                count += sum(1 for job in resp.json().get("data", [])
                             if job.get("status") in ("queued", "in_progress"))
        except (httpx.HTTPError, ValueError, AttributeError):
            pass  # a backend that does not answer has no visible queue
        return count

    @app.get("/v1/videos")
    async def list_videos(request: Request):
        entry = route(request.query_params.get("model"), "video")
        # The backend rejects query fields it does not know.
        query = urlencode([(k, v) for k, v in request.query_params.multi_items() if k != "model"])
        return _relay(await send(request, entry, "/v1/videos", query=query))

    async def find_video(request: Request, path: str):
        """Ask each video backend in turn; job ids do not name their model."""
        unavailable = None
        for entry in config.video_entries():
            try:
                resp = await send(request, entry, path)
            except DiffletServingError as exc:
                if exc.code != "model_unavailable":
                    raise
                unavailable = unavailable or exc
                continue
            if resp.status_code != 404:
                return _relay(resp)
            await resp.aclose()
        if unavailable is not None:  # the job may be on the backend that did not answer
            raise unavailable
        raise DiffletServingError(404, "video_not_found", "Video job was not found")

    @app.get("/v1/videos/{video_id}")
    @app.delete("/v1/videos/{video_id}")
    async def video(request: Request, video_id: str):
        return await find_video(request, f"/v1/videos/{quote(video_id, safe='')}")

    @app.get("/v1/videos/{video_id}/content")
    async def video_content(request: Request, video_id: str):
        return await find_video(request, f"/v1/videos/{quote(video_id, safe='')}/content")

    @app.get("/v1/models")
    async def models():
        found = await statuses()
        return {"object": "list", "data": [{
            "id": e.model_id,
            "object": "model",
            "owned_by": "difflet",
            "difflet_kind": e.kind,
            "difflet_device": e.device,
            "difflet_shapes": ["x".join(str(n) for n in s if n is not None) for s in e.shapes],
            "difflet_status": found.get(e.model_id, "down"),
        } for e in config.models]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        found = await statuses()
        all_ready = all(found.get(e.model_id) == "ready" for e in config.models)
        return JSONResponse(status_code=200 if all_ready else 503, content={
            "status": "ready" if all_ready else "not_ready", "models": found})

    @app.get("/stats")
    async def stats():
        return {"total_runs": runs.total}

    @app.get("/queue")
    async def queue():
        counts = await asyncio.gather(*(active(entry) for entry in config.models))
        return {"models": {e.model_id: n for e, n in zip(config.models, counts)}}

    @app.get("/ui/models")
    async def ui():
        return ui_models(config, await statuses())

    @app.get("/", response_class=HTMLResponse)
    async def page(request: Request):
        return HTMLResponse(render_page(config, await statuses(), base_url=str(request.base_url),
                                        total_runs=runs.total))

    return app


class RunCounter:
    """Generations the gateway passed on successfully (an image answered 200, a video job
    accepted), kept across gateway restarts in one JSON file. A display figure: a file that cannot
    be read or written never fails a request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.total = int(json.loads(path.read_text())["total_runs"])
        except (OSError, ValueError, KeyError, TypeError):
            self.total = 0

    def count(self, response: httpx.Response) -> httpx.Response:
        if response.status_code == 200:
            self.total += 1
            try:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"total_runs": self.total}))
                tmp.replace(self.path)
            except OSError:
                pass
        return response


async def _read_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise DiffletServingError(413, "request_too_large",
                                      f"request body exceeds {MAX_BODY_BYTES // 2**20} MiB")
    return bytes(body)


async def _form_model(request: Request, body: bytes):
    """The ``model`` field of a form body that was already read (None when there is none)."""
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    try:
        async with StarletteRequest(request.scope, receive).form() as form:
            return form.get("model")
    except Exception:  # a malformed form names no model; the caller answers model_required
        return None


def _relay(resp: httpx.Response) -> StreamingResponse:
    async def chunks():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    out = StreamingResponse(chunks(), status_code=resp.status_code)
    out.raw_headers = [(k.lower(), v) for k, v in resp.headers.raw
                       if k.lower() not in _DROP_RESPONSE]
    return out
