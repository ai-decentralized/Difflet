# Multi-model gateway

`difflet gateway` puts one endpoint in front of several `difflet serve` backends, each resident on
its own Neuron device. A client names the model in its request (`model` in the chat-completions
JSON, or the `model` field of a video form); the gateway forwards the request to that model's
backend and returns the backend's answer unchanged. A page at `/` lists every model, its shapes,
its status and an example request.

One fleet file drives everything: which backends `up` starts, where the gateway routes, and what
the page shows. An example for a trn3 with six models on devices 0–5 is
[`examples/serving/fleet.yaml`](../../examples/serving/fleet.yaml).

## Start and stop

Run these from the device venv: `up` starts each backend with the interpreter it runs under.

```bash
difflet gateway up     --config fleet.yaml   # start the backends, one at a time
nohup difflet gateway serve --config fleet.yaml > gateway.out 2>&1 &   # the endpoint
difflet gateway status --config fleet.yaml   # model, device, port, status per backend
difflet gateway down   --config fleet.yaml   # stop every backend
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
gateway: {host: 0.0.0.0, port: 8000}
runs_dir: /mnt/nvme/difflet-fleet/runs
models:
  - model_id: black-forest-labs/FLUX.1-dev
    slug: flux
    device: 0
    port: 8100
    cache_dir: /mnt/nvme/difflet-fleet/cache/flux
    env: {HF_HUB_CACHE: /mnt/nvme/hub}
    serve_args: [--tp-degree, "4", --height, "1024", --width, "1024"]
```

| Field | Meaning |
|---|---|
| `gateway.host`, `gateway.port` | Where the gateway listens (default `0.0.0.0:8000`) |
| `runs_dir` | Parent of each backend's working directory (required) |
| `model_id` | A checkpoint `difflet serve` serves (`model_id` with a trailing `/` routes the same) |
| `device` | The Neuron device the backend gets, all its cores |
| `port` | The backend's port on `127.0.0.1` |
| `cache_dir` | The backend's `--cache-dir`: compiled artifacts and generated videos |
| `serve_args` | Every other `difflet serve` flag; the shape flags here are what the page shows |
| `slug` | Optional name of the working directory; default: the model id, lower-cased, with `/` and `.` as `-` |
| `env` | Optional extra environment for this backend (`HF_HUB_CACHE`, `PYTHONPATH`, ...) |

The file is refused, with the field named, when a model, device, port or slug repeats, the
gateway's port is also a backend's, `serve_args` sets `--host`, `--port`, `--model-id` or
`--cache-dir` (the launcher sets them), an entry has no shape (`--height` and `--width`, plus
`--num-frames` for a video model, or `--shapes`), or the model is not servable.

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
| `GET /` | The models page |

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
`Authorization: Bearer KEY`; `/`, `/health` and `/ready` stay open. It is one shared key, as for
`difflet serve`; put the port behind a security group or a TLS-terminating proxy. The backends
listen on `127.0.0.1` without a key.

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
