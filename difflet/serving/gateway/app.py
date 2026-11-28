"""The gateway: one endpoint in front of the fleet's ``difflet serve`` backends (design §6.4).

It reads the request's ``model`` to pick the backend and forwards the request bytes unchanged;
the backend's status and body come back unchanged. It never times out a forwarded request on its
own (a video can wait hours in its backend's queue); a client that goes away while the backend is
still working closes the upstream request, so the backend cancels it.

Generation submissions pass the fleet's limits first (``submit``): per client IP and per client
per hour, one request at a time per client, and no model whose estimated wait is too long.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from email.message import Message
from time import monotonic
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
from difflet.serving.gateway.page import (_duration, entry_defaults, render_info, render_page,
                                          ui_models)
from difflet.serving.gateway.status import Status, fleet_status
from difflet.serving.openai.api_server import _run_until_disconnect
from difflet.serving.openai.serving_video import MAX_KEYFRAME_FORM_BODY_BYTES

#: The largest request body any backend accepts (a video form with two keyframes).
MAX_BODY_BYTES = MAX_KEYFRAME_FORM_BODY_BYTES
_HOP_BY_HOP = {b"connection", b"keep-alive", b"transfer-encoding", b"te", b"trailer", b"upgrade",
               b"proxy-authorization", b"proxy-authenticate"}
_DROP_REQUEST = _HOP_BY_HOP | {b"host", b"content-length", b"authorization"}
_DROP_RESPONSE = _HOP_BY_HOP | {b"date", b"server"}  # uvicorn sends its own
_CLIENT_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
_ACTIVE_JOB = ("queued", "in_progress")
_HOUR_S = 3600.0
#: How long the gateway remembers a client's async video job that is still listed as active.
_JOB_MEMORY_S = 24 * 3600.0
#: How long the answers every open page polls (/queue, the backend statuses) are shared.
_POLL_CACHE_S = 2.0


def _shared(seconds: float, fetch: Callable[[], Awaitable]) -> Callable[[], Awaitable]:
    """``fetch`` called at most once per ``seconds``; callers in between, and callers waiting
    while a fetch runs, get its result."""
    lock = asyncio.Lock()
    value, fetched = None, -math.inf

    async def get():
        nonlocal value, fetched
        async with lock:
            if monotonic() - fetched >= seconds:
                value, fetched = await fetch(), monotonic()
        return value

    return get


@dataclass
class RateLimited(DiffletServingError):
    """A 429 the gateway answers itself, with ``Retry-After: retry_after`` (seconds)."""

    retry_after: float = 60.0


def client_ip(request: Request) -> str:
    """The client's address: behind a proxy on this host (a loopback peer), the last
    X-Forwarded-For entry, the one that proxy appended; otherwise the peer itself."""
    peer = request.client.host if request.client else ""
    forwarded = ",".join(request.headers.getlist("x-forwarded-for")).split(",")[-1].strip()
    try:
        behind_proxy = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        behind_proxy = False
    return forwarded if forwarded and behind_proxy else peer


def client_key(request: Request) -> str:
    """Who the per-client limits count: the page's ``X-Difflet-Client`` id, else the IP."""
    given = request.headers.get("x-difflet-client", "")
    return given if _CLIENT_ID.fullmatch(given) else f"ip:{client_ip(request)}"


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

    #: Requests the gateway holds open per model (chat completions, /v1/videos/sync): a backend
    #: answers them only when the generation ends, so each counts while it waits and while it runs.
    inflight = {entry.model_id: 0 for entry in config.models}
    # shortcut: the limits' state is in memory, so it resets when the gateway restarts and holds
    # for one gateway process only; move it to a shared store if the gateway is ever replicated.
    #: Times of the submissions a backend accepted (HTTP 200) in the last hour, per IP / client.
    ip_accepted: dict[str, deque[float]] = {}
    client_accepted: dict[str, deque[float]] = {}
    #: Model ids of the requests each client has in the gateway now (held open, or being admitted).
    client_held: dict[str, list[str]] = {}
    #: Async video jobs each client created: job id -> (model id, when).
    client_jobs: dict[str, dict[str, tuple[str, float]]] = {}
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
        headers = ({"Retry-After": str(max(1, math.ceil(exc.retry_after)))}
                   if isinstance(exc, RateLimited) else None)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(),
                            headers=headers)

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
        return await submit(request, entry, "/v1/chat/completions", body, hold=True)

    @app.post("/v1/videos")
    @app.post("/v1/videos/sync")
    async def create_video(request: Request):
        body = await _read_body(request)
        model, fields = await _form_model(request, body)
        entry = route(model, "video")
        if entry.default_negative_prompt is not None and "negative_prompt" not in fields:
            body = _with_form_field(request.headers.get("content-type", ""), body,
                                    "negative_prompt", entry.default_negative_prompt)
        return await submit(request, entry, request.url.path, body,
                            hold=request.url.path.endswith("/sync"))

    async def submit(request: Request, entry: FleetEntry, path: str, body: bytes, *,
                     hold: bool) -> StreamingResponse:
        """Forward a generation once the limits let it through, in this order: the client IP's
        and the client's submissions in the last hour, the client's requests running or waiting,
        the model's estimated wait. ``hold``: the backend answers when the generation ends, so the
        request counts in ``inflight`` until then; otherwise it answers with an async job."""
        ip, key = client_ip(request), client_key(request)
        check_rate(ip_accepted, ip, config.limits.ip_per_hour, "ip_rate_limited",
                   "This network has")
        check_rate(client_accepted, key, config.limits.client_per_hour, "client_rate_limited",
                   "You have")
        mine = client_held.setdefault(key, [])
        ahead = list(mine)
        mine.append(entry.model_id)  # before any await, so a parallel submission sees this one
        raw = None
        try:
            ahead += await running_jobs(key)
            if len(ahead) >= config.limits.client_active:
                raise RateLimited(
                    429, "client_busy",
                    "You already have a request running or waiting; it must finish first",
                    retry_after=max((s for m in ahead if (s := each(config.entry(m))) is not None),
                                    default=60.0))
            await check_wait(entry)
            if hold:
                inflight[entry.model_id] += 1
            try:
                resp = await send(request, entry, path, body=body)
            finally:
                if hold:
                    inflight[entry.model_id] -= 1
            if resp.status_code == 200:
                now = monotonic()
                ip_accepted.setdefault(ip, deque()).append(now)
                client_accepted.setdefault(key, deque()).append(now)
                if not hold:  # an async video job: remember it until it finishes
                    raw = await _read_raw(resp)
                    if (job_id := _job_id(raw)) is not None:
                        client_jobs.setdefault(key, {})[job_id] = (entry.model_id, now)
        finally:
            mine.remove(entry.model_id)
            if not mine:
                del client_held[key]
        return _relay(resp, raw)

    def check_rate(windows: dict[str, deque[float]], key: str, limit: int, code: str,
                   who: str) -> None:
        now = monotonic()
        times = windows.get(key)
        while times and now - times[0] >= _HOUR_S:
            times.popleft()
        if times is not None and not times:
            del windows[key]
        elif times is not None and len(times) >= limit:
            retry = times[len(times) - limit] + _HOUR_S - now  # when one leaves the window
            raise RateLimited(429, code, f"{who} reached the limit of {limit} generations per "
                              f"hour; try again in about {_duration(retry)}", retry_after=retry)

    def each(entry: FleetEntry) -> float | None:
        """Seconds one request takes on ``entry`` at its default steps (measured), None without
        an estimate."""
        steps = entry_defaults(entry)[0]
        if entry.estimate is None or not steps:
            return None
        return entry.estimate[0] + entry.estimate[1] * steps

    async def check_wait(entry: FleetEntry) -> None:
        """Refuse ``entry`` when the requests ahead, at their default steps, take longer than
        ``wait.reject_s``, naming up to two ready models of its kind with a shorter wait."""
        seconds = each(entry)
        if seconds is None:
            return
        wait = await active(entry) * seconds
        if wait <= config.wait.reject_s:
            return
        found = await statuses()
        others = [e for e in config.models if e is not entry and e.kind == entry.kind
                  and found.get(e.model_id) == "ready" and each(e) is not None]
        waits = [n * each(e) for e, n in zip(others, await asyncio.gather(*map(active, others)))]
        shorter = sorted((w, e.model_id) for e, w in zip(others, waits) if w < wait)[:2]
        tip = ("Models with a shorter wait: " + ", ".join(
            f"{m} ({'no wait' if w == 0 else 'about ' + _duration(w)})" for w, m in shorter) + "."
            if shorter else "Try again later.")
        raise RateLimited(
            429, "model_busy",
            f"{entry.model_id} is too busy right now: the estimated wait is about "
            f"{_duration(wait)}, over {_duration(config.wait.reject_s)}. {tip}",
            # until enough requests ahead finish for the wait to drop under the threshold
            retry_after=math.ceil((wait - config.wait.reject_s) / seconds) * seconds)

    async def running_jobs(key: str) -> list[str]:
        """Model ids of the async video jobs ``key`` created that their backend still lists as
        queued or in progress. A job listed as finished, not listed, or older than a day is
        forgotten; one on a backend that does not answer is kept but not counted."""
        jobs = client_jobs.get(key, {})
        now = monotonic()
        for job_id, (_, created) in list(jobs.items()):
            if now - created > _JOB_MEMORY_S:
                del jobs[job_id]
        # Only the jobs known before the listings are fetched: a job recorded meanwhile may be
        # newer than its listing.
        known = list(jobs.items())
        models = sorted({model for _, (model, _) in known})
        listed = dict(zip(models, await asyncio.gather(
            *(video_jobs(config.entry(m)) for m in models))))
        running = []
        for job_id, (model, _) in known:
            if listed[model] is None:
                continue
            if listed[model].get(job_id) in _ACTIVE_JOB:
                running.append(model)
            else:
                jobs.pop(job_id, None)
        return running

    async def video_jobs(entry: FleetEntry) -> dict[str, str] | None:
        """Job id -> status from a video backend's own list, None when it does not answer."""
        try:
            # shortcut: the newest 100 jobs only (the list is newest first); page with `after`
            # if a queue can exceed 100.
            resp = await http.get(entry.base_url + "/v1/videos", params={"limit": 100},
                                  timeout=5.0)
            if resp.status_code == 200:
                return {job.get("id"): job.get("status") for job in resp.json().get("data", [])}
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return None

    async def active(entry: FleetEntry) -> int:
        """Requests running or waiting on ``entry``: the ones the gateway holds open, plus a video
        backend's async jobs still queued or in progress (its own list, so API users count too)."""
        count = inflight[entry.model_id]
        if entry.kind == "video":  # a backend that does not answer has no visible queue
            jobs = await video_jobs(entry) or {}
            count += sum(1 for status in jobs.values() if status in _ACTIVE_JOB)
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

    async def queue_counts() -> dict[str, int]:
        counts = await asyncio.gather(*(active(entry) for entry in config.models))
        return {e.model_id: n for e, n in zip(config.models, counts)}

    # What every open page polls (/queue every 5 s, statuses every 15 s) is shared for a moment,
    # so a full venue of tabs costs the backends one probe round per interval, not one per tab.
    # The limit checks above read the backends directly.
    polled_statuses = _shared(_POLL_CACHE_S, statuses)
    polled_queue = _shared(_POLL_CACHE_S, queue_counts)

    @app.get("/v1/models")
    async def models():
        found = await polled_statuses()
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
        found = await polled_statuses()
        all_ready = all(found.get(e.model_id) == "ready" for e in config.models)
        return JSONResponse(status_code=200 if all_ready else 503, content={
            "status": "ready" if all_ready else "not_ready", "models": found})

    @app.get("/queue")
    async def queue():
        return {"models": await polled_queue()}

    @app.get("/ui/models")
    async def ui():
        return ui_models(config, await polled_statuses())

    @app.get("/", response_class=HTMLResponse)
    async def page():
        return HTMLResponse(render_page(config, await polled_statuses()))

    @app.get("/info", response_class=HTMLResponse)
    async def info(request: Request):
        return HTMLResponse(render_info(config, await polled_statuses(),
                                        base_url=str(request.base_url)))

    return app


async def _read_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise DiffletServingError(413, "request_too_large",
                                      f"request body exceeds {MAX_BODY_BYTES // 2**20} MiB")
    return bytes(body)


async def _form_model(request: Request, body: bytes) -> tuple[str | None, frozenset[str]]:
    """The ``model`` field of a form body that was already read (None when there is none), and
    the names of all its fields."""
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    try:
        async with StarletteRequest(request.scope, receive).form() as form:
            return form.get("model"), frozenset(form.keys())
    except Exception:  # a malformed form names no model; the caller answers model_required
        return None, frozenset()


def _with_form_field(content_type: str, body: bytes, name: str, value: str) -> bytes:
    """``body`` (a form that parsed) with one more text field, in its own encoding."""
    message = Message()
    message["content-type"] = content_type
    if message.get_content_type() == "application/x-www-form-urlencoded":
        return body + (b"&" if body else b"") + urlencode({name: value}).encode()
    boundary = message.get_param("boundary")
    end = body.rfind(f"--{boundary}--".encode()) if boundary else -1
    if end < 0:
        return body
    part = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n").encode()
    return body[:end] + part + body[end:]


async def _read_raw(resp: httpx.Response) -> bytes:
    """A streamed response's bytes as sent (no decoding), for a body to look at and relay."""
    try:
        return b"".join([chunk async for chunk in resp.aiter_raw()])
    finally:
        await resp.aclose()


def _job_id(raw: bytes) -> str | None:
    try:
        job_id = json.loads(raw).get("id")
    except (ValueError, AttributeError):
        return None
    return job_id if isinstance(job_id, str) else None


def _relay(resp: httpx.Response, raw: bytes | None = None) -> StreamingResponse:
    """``resp`` back to the client as the backend sent it; ``raw``: its body, already read."""
    async def chunks():
        if raw is not None:
            yield raw
            return
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    out = StreamingResponse(chunks(), status_code=resp.status_code)
    out.raw_headers = [(k.lower(), v) for k, v in resp.headers.raw
                       if k.lower() not in _DROP_RESPONSE]
    return out
