# Multi-model gateway

`difflet gateway` puts one endpoint in front of several `difflet serve` backends, each resident on
its own Neuron device. A client names the model in its request (`model` in the chat-completions
JSON, or the `model` field of a video form); the gateway forwards the request to that model's
backend and returns the backend's answer unchanged. A page at `/` generates images and videos
with any ready model, and `/info` lists every model, its shapes, its status and an example request; a
static copy of the `/` page can be hosted elsewhere ([Public page](#public-page-static-hosting-cors-and-limits)).

One fleet file drives everything: which backends `up` starts, where the gateway routes, and what
the page shows. An example for a trn3 with six models on devices 0–5 is
[`examples/serving/fleet.yaml`](../../examples/serving/fleet.yaml).

To bring the public server (difflet.yottalabs.ai) up on a new machine, from mounting the disks
to TLS and DNS, follow [gateway-setup.md](gateway-setup.md).

## Start and stop

Run these from the device venv: `up` starts each backend with the interpreter it runs under.

```bash
difflet gateway up     --config fleet.yaml   # start the backends, one at a time
nohup difflet gateway serve --config fleet.yaml > gateway.out 2>&1 &   # the endpoint
difflet gateway status --config fleet.yaml   # model, device, port, status per backend
difflet gateway down   --config fleet.yaml   # stop every backend
difflet gateway page --api-base https://api.example.com --out site/index.html   # static page
```

- `up` goes through the entries in order. It skips a backend that is already running, refuses a
  device another process holds (a Neuron core cannot be shared) or a port already taken, starts
  the backend detached, and waits up to an hour for its port to open before starting the next one,
  so the weight loads do not share the disk. A backend that exits first is reported with the last
  40 lines of its log. The port opens only after the model is loaded and its smoke generation has
  run.
- Backends outlive the gateway: restarting `difflet gateway serve` (for a config or page change)
  reloads no model. Stop the backends with `down`; it sends SIGTERM to each backend's process
  group and SIGKILL to whatever is left after 60 s.
- Each backend runs in `<runs_dir>/<slug>/` with `serve.pid`, `serve.out` (its console) and
  `logs/` there.
- Every backend must find its compiled artifacts in its `cache_dir` and its weights in the hub
  cache: backends run with `HF_HUB_OFFLINE=1`, so a missing snapshot fails instead of
  downloading. A wrong `cache_dir` makes the backend compile at startup (`serve.out` shows it).

## Fleet file

```yaml
gateway:
  host: 127.0.0.1
  port: 8000
  cors_origins: ["https://page.example.com"]
  limits: {client_active: 1, client_per_hour: 20, ip_per_hour: 2000}
  wait: {notice_s: 1800, suggest_s: 1800, reject_s: 7200}
runs_dir: /mnt/nvme/difflet-fleet/runs
models:
  - model_id: black-forest-labs/FLUX.1-dev
    slug: flux
    device: 0
    port: 8100
    cache_dir: /mnt/nvme/difflet-fleet/cache/flux
    env: {HF_HUB_CACHE: /mnt/nvme/hub}
    serve_args: [--tp-degree, "4", --height, "1024", --width, "1024"]
    estimate: {base_s: 0.7, per_step_s: 0.278}
```

| Field | Meaning |
|---|---|
| `gateway.host`, `gateway.port` | Where the gateway listens (default `0.0.0.0:8000`; `127.0.0.1` behind a proxy on the host) |
| `gateway.cors_origins` | Origins whose pages may call the gateway, such as `https://page.example.com` (default none: no CORS headers) |
| `gateway.limits` | Per-client and per-IP caps on generations ([Limits for a public page](#limits-for-a-public-page)); each field optional |
| `gateway.wait` | Estimated waits at which the page warns and the gateway refuses ([Wait thresholds](#wait-thresholds)); each field optional |
| `runs_dir` | Parent of each backend's working directory (required) |
| `model_id` | A checkpoint `difflet serve` serves (`model_id` with a trailing `/` routes the same) |
| `device` | The Neuron device the backend gets, all its cores |
| `port` | The backend's port on `127.0.0.1` |
| `cache_dir` | The backend's `--cache-dir`: compiled artifacts and generated videos |
| `serve_args` | Every other `difflet serve` flag; the shape flags here are what the page shows |
| `slug` | Optional name of the working directory; default: the model id, lower-cased, with `/` and `.` as `-` |
| `env` | Optional extra environment for this backend (`HF_HUB_CACHE`, `PYTHONPATH`, ...) |
| `estimate` | Optional, measured: a request takes about `base_s + per_step_s * steps` seconds. The page shows it; the wait thresholds use it at the entry's default steps (`defaults`) |
| `defaults` | Optional `{steps, guidance}`: what the page sends when its Steps or Guidance field is empty, and the steps the estimates and wait thresholds assume. Unset, the backend's own default (the model's spec). `guidance` is refused for a model that takes none (Qwen-Image, MiniMax-H3). API requests that omit them still get the backend's default |

The file is refused, with the field named, when a field is unknown, a model, device, port or
slug repeats, the gateway's port is also a backend's, `serve_args` sets `--host`, `--port`,
`--model-id` or `--cache-dir` (the launcher sets them), an entry has no shape (`--height` and
`--width`, plus `--num-frames` for a video model, or `--shapes`), the model is not servable, an
origin is not `http(s)://host[:port]` (no path, no trailing `/`), a limit or wait is not a
positive integer, or the waits are not `notice_s <= suggest_s <= reject_s`.

Each backend is started as

```text
taskset -c <the device's NUMA-node CPUs> <python> -m difflet.cli.main serve \
  --model-id <model_id> --host 127.0.0.1 --port <port> --cache-dir <cache_dir> <serve_args>
```

with the launcher's environment, then the entry's `env`, then these, which always win:
`NEURON_RT_VISIBLE_CORES=<the device's cores>`, `DIFFLET_BACKEND=neuron`, `NEURON_NUMA_PIN=0`,
`HF_HUB_OFFLINE=1` and `DIFFLET_API_KEY=` (empty: the gateway does not forward the client's
`Authorization`, so a key inherited by a backend would refuse every request). `neuronx-cc`'s
directory is put first on `PATH` when it is not on it already.

## Routes

| Route | Behavior |
|---|---|
| `POST /v1/chat/completions` | Routed by the JSON body's `model` (an image model) |
| `POST /v1/videos`, `POST /v1/videos/sync` | Routed by the form's `model` (a video model); the form bytes, keyframe files included, are forwarded unchanged |
| `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}` | Asked of each video backend in turn; the first answer that is not `404` is returned |
| `GET /v1/videos?model=...` | That model's job list; `model` is required and removed before forwarding (`limit`, `after` pass through) |
| `GET /v1/models` | Every configured model with `difflet_kind`, `difflet_device`, `difflet_shapes` (`HxW` or `HxWxF`) and `difflet_status` |
| `GET /ready` | `200` when every model is ready, else `503`; the body lists each model's status |
| `GET /health` | `200` while the gateway runs |
| `GET /ui/models` | The page's model list (`id`, `label`, `kind`, `device`, `shapes`, `status`, default `steps` and `guidance`, `estimate`), plus `wait` and `limits` |
| `GET /queue` | Per model, the requests running or waiting: held by the gateway, plus a video backend's queued or in-progress jobs |
| `GET /` | The generation page |
| `GET /info` | Every model's kind, device, shapes, estimated time and status, and an example request per model |

Backend answers, errors included, come back with their status and body unchanged. The gateway's
own errors use the backends' error format:

| Status | Code | When |
|---|---|---|
| 400 | `model_required` | The request names no model |
| 404 | `model_not_found` | No backend serves that model; the message lists the served ones |
| 400 | `wrong_endpoint` | A video model on chat completions, or an image model on `/v1/videos` |
| 503 | `model_unavailable` | The backend does not answer: still loading (its process is up) or down. For a video id lookup: no backend owns the id and at least one did not answer |
| 404 | `video_not_found` | No video backend knows the id |
| 413 | `request_too_large` | The body is larger than any backend accepts (40 MiB) |
| 429 | `ip_rate_limited`, `client_rate_limited`, `client_busy`, `model_busy` | A generation refused by a limit ([Limits for a public page](#limits-for-a-public-page)); with `Retry-After` |

The gateway never times a request out: a backend owns its queue and request timeouts, and a
video can wait for hours in its queue. When the client disconnects while the backend is still
working, the gateway closes the upstream request and the backend cancels it.

## Status

| Status | Meaning |
|---|---|
| ready | The backend's process is alive and `/ready` answers `200` for its own model |
| loading | The process is alive and its port is not open yet (loading weights, or compiling) |
| recovering | `/ready` answers `503`: the backend is restarting its worker after a failure |
| down | No live process for the entry's pid file, or the port answers for another model |

## API key

`--api-key KEY` (or `DIFFLET_API_KEY`) makes every `/v1` route require
`Authorization: Bearer KEY`; `/`, `/info`, `/health`, `/ready`, `/ui/models` and `/queue` stay open. It is one shared key, as for `difflet serve`; put the port behind a security group or a
TLS-terminating proxy. The backends listen on `127.0.0.1` without a key. The page sends no key,
so it works only with a gateway started without one.

## Public page: static hosting, CORS and limits

The page can be served by a static host on another domain, calling the gateway through a
TLS-terminating proxy on the gateway's host. For example, a page on a static host at
`https://page.example.com` and the gateway at `https://api.example.com`:

1. In the fleet file: `gateway.host: 127.0.0.1` (only the proxy reaches the gateway) and
   `cors_origins: ["https://page.example.com"]`.
2. Caddy on the gateway's host, with ports 80 and 443 open (Caddy gets the certificate itself):

   ```caddyfile
   api.example.com {
       reverse_proxy 127.0.0.1:8000 {
           flush_interval -1
       }
   }
   ```

   `flush_interval -1` passes streamed video bytes on at once. Caddy sets no read timeout by
   default, and none must be added: a `/v1/videos/sync` request or a chat completion can wait in
   its model's queue and run for many minutes before the backend answers.
3. Write the page and deploy the directory as a static site (no build step):

   ```bash
   difflet gateway page --api-base https://api.example.com --out site/index.html
   ```

   The file is self-contained (inline style and script, no external assets). It embeds no
   models: at start and every 15 s it loads them, with the wait thresholds and limits, from
   `/ui/models`. The page served at the gateway's own `/` keeps working on the same origin.

With `cors_origins` set, the gateway answers CORS preflights from those origins (methods `GET`,
`POST`, `DELETE`, `OPTIONS`; headers `Content-Type`, `Authorization`, `X-Difflet-Client`; it
exposes `Retry-After` and `X-Request-Id`). The CORS layer runs before the API-key check, so a
preflight never gets a `401`. Other origins get no CORS headers, so a browser on them cannot read
the answers; this does not stop non-browser clients.

### Limits for a public page

Generation submissions (`POST /v1/chat/completions`, `/v1/videos`, `/v1/videos/sync`) are checked
after the model is routed and before they are forwarded, in this order. Each refusal is a `429`
with a `Retry-After` (seconds) and a message saying why; the page shows the message.

| Code | Refused when | Default |
|---|---|---|
| `ip_rate_limited` | The client IP had `limits.ip_per_hour` accepted submissions in the last hour | 2000 |
| `client_rate_limited` | The client had `limits.client_per_hour` accepted submissions in the last hour | 20 |
| `client_busy` | The client already has `limits.client_active` requests running or waiting | 1 |
| `model_busy` | The model's estimated wait is over `wait.reject_s`; the message gives the wait and the threshold and names up to two ready models of the same kind with a shorter wait | 7200 s |

- **The client** is the `X-Difflet-Client` header when it matches `[A-Za-z0-9_-]{8,64}` (the page
  sends a random id kept in the browser's local storage), else `ip:<client IP>`. API clients
  without the header share one client per IP: at an event where everyone has the venue's IP, they
  should each send their own id. The id is not authentication: a client that changes it gets a
  fresh per-client count, and `ip_per_hour` is what still bounds it.
- **The client IP** is the last `X-Forwarded-For` entry when the request comes from a loopback
  address (the proxy on this host appends the real client), else the connecting address.
- **Running or waiting** counts the client's requests the gateway holds open (chat completions,
  `/v1/videos/sync`) and the async video jobs it created (`POST /v1/videos`) that their backend
  still lists as queued or in progress. A job is forgotten once it is listed as finished, no
  longer listed, or a day old; a job on a backend that does not answer is not counted.
- **The hourly counts** slide over the last hour and count only submissions a backend accepted
  (`200`); refused ones, by a limit or by the backend, do not count.
- The counts live in the gateway's memory: they start over when it restarts, and hold for one
  gateway process only.

### Wait thresholds

A model's estimated wait is its requests running or waiting (as `GET /queue` counts them) times
one request at its default steps, from the entry's `estimate`. A model without an `estimate` has
no estimated wait and is never refused for it.

| Field | Default | From this wait |
|---|---|---|
| `wait.notice_s` | 1800 (30 min) | The page shows "Long wait: about X before it starts" under the queue line |
| `wait.suggest_s` | 1800 (30 min) | It adds "Consider another model:" and up to two ready models of the same kind with shorter waits (click to select one); with the defaults, together with the notice |
| `wait.reject_s` | 7200 (2 h) | The page says the model is too busy right now (over the threshold, e.g. "over 2 h") and disables Run; the gateway refuses a submission with `model_busy` |

Every threshold the page and the messages show is the configured value, not a fixed text.

## Limits

- Each model runs one request at a time; others wait in its backend's queue
  (`--max-queued-requests`, default 8) and get `429` when the queue is full or the queue timeout
  passes (30 s for images, 24 h for videos). Different models run in parallel.
- A backend serves only the shapes it was started with; any other gets `profile_mismatch`.
- One backend per model, and no queue in the gateway.
- A `difflet serve` process that dies stays down until `up` is run again (`difflet serve` does
  restart its own worker after a worker failure).
- All backends on one NUMA node share its CPUs, and each worker rank sizes its thread count from
  the host's CPU count, so host-side stages of several models running at once (text encoders, a
  host VAE) contend for CPUs.
- Linux only: the launcher uses `taskset` and `/proc`.
