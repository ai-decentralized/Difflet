# Multi-model gateway: every servable model resident on one trn3 — design

**Date:** 2026-11-20
**Status:** design settled 2026-11-20 (approach A, shared-host question answered); implementation plan next.
No code until the user confirms the implementation plan.
**Fact-check baseline:** main `eaf4c003`. Items marked "verified" were checked against the code or the host.

Markers: **[User decision]** the user decided it in the discussion; **[My choice]** a default I picked, open to
change; **[Open]** needs the user's answer.

## 1. Purpose

Host every Difflet model that can serve today on the team's trn3.48xlarge, each resident on its own device, and
put one endpoint in front of them. A client sends one request that says which model to use and what image or
video to make; the endpoint hands it to that model and returns the result. A page on the same endpoint tells
users which models are up and which shapes each one serves.

## 2. Current state (verified)

- `difflet serve` (`difflet/cli/serve.py`) already is a single-model resident server: one model, one immutable
  profile compiled at startup, an OpenAI-style FastAPI app (`difflet/serving/openai/api_server.py`).
  - Images: `POST /v1/chat/completions` (JSON). `model` is optional there; a wrong one gets `400 model_not_served`.
  - Videos: `POST /v1/videos` (async job), `POST /v1/videos/sync`, `GET /v1/videos`, `GET /v1/videos/{id}`,
    `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}`. Bodies are `multipart/form-data` only and may carry
    keyframe images (MiniMax-H3 `first_frame` / `last_frame`). Job ids are `video_gen_<uuid4 hex>`
    (`video_jobs.py:131`): they do not name the model.
  - `GET /health`, `GET /ready`, `GET /v1/models` (one entry). The port opens only after load and the smoke
    generation pass: the FastAPI lifespan awaits `engine.start()` (`api_server.py:187-191`) before uvicorn binds.
    So while a backend loads its port refuses connections; `/ready` answers 503 only during worker recovery.
  - `GET /v1/videos` accepts only `limit` and `after` query fields; anything else is `400` (`api_server.py:538`).
  - A request outside the compiled shapes gets `profile_mismatch`; HTTP never triggers compilation.
  - One request at a time per server; others queue (`--max-queued-requests`, default 8) and get `429` after the
    queue timeout: 30 s for images, 24 h for videos (`options.py:143-146`). `--request-timeout` (300 s) counts
    from when the request leaves the queue. `--dp-degree`/replicas exist on GPU backends only.
  - Binds `0.0.0.0:8091` by default; optional shared API key from `--api-key`, else `DIFFLET_API_KEY`
    (`serve.py:157-159`), enforced by `ApiKeyAuthenticationMiddleware` (`serving/auth.py`).
  - Writes logs to `./logs/` (5 MB × up to 1024 files) and loads `./.env` from its working directory.
  - Video results are written under `<cache_dir>/serving/videos/` and kept 25 h by default
    (`--video-retention-seconds`); without S3 configured (no `.env` exists today) they stay on local disk.
  - Download and compile policies are automatic (`serve.py:155-156`): a wrong `--cache-dir` silently compiles at
    startup, and a missing hub snapshot downloads.
  - Each worker rank sets torch threads to `os.cpu_count() / ranks` (`resident_worker.py:1072`), not its CPU
    affinity; Wan's fp32 umT5 output bits depend on that count (`backends/neuron/core/model_application.py:74`).
- It honors `NEURON_RT_VISIBLE_CORES` (`common/neuron_cores.py`), so several servers can run side by side, each
  pinned to its own device.
- Models with Trainium serving metadata (`serving/model_registry.py:205`): FLUX.1-dev, Qwen-Image, Wan (2.1 and
  2.2 checkpoints — 2.2 is "experimental … not MVP-qualified because its second transformer is not loaded",
  `common/registry/wan.py:14-20`), HunyuanVideo, LTX-2, MiniMax-H3. HunyuanVideo 1.5 has none. Registering metadata is not proof
  that each one serves on the native `neuron` backend on trn3 — see section 7.
- Host: trn3.48xlarge, 16 devices × 4 logical cores (LNC=2), 144 GB HBM each; device d = cores `4d..4d+3`;
  devices 0–7 on NUMA node 1 (CPUs 48–95,144–191). Device venv `/home/ubuntu/venvs/native-fp8`.
- `fastapi`, `uvicorn`, `python-multipart`, `PyYAML` are declared dependencies; `httpx 0.28.1` is in
  `requirements-neuron-native.lock` but not declared in `pyproject.toml`.
- Hub weights live in `~/.cache/huggingface/hub` on the root EBS disk (~125 MB/s, 96% full, 79 GB free);
  compiled caches are spread over `~/difflet-cache-{video,fp8,cold}` (root disk) and `/mnt/nvme/difflet-cache-*`
  (`/mnt/nvme`: 849 GB free on 2026-11-20).
- Every Neuron runtime process also opens `/dev/neuron0` (`bench/host.py:355`).

So the per-model servers exist; what is missing is the front door that routes by `model`, the page, and a way
to start the six servers with the right pinning.

**Finding (2026-11-20, after the cache inventory) — blocks five of the six models.** On this host's native
`neuron` backend only MiniMax-H3 can run `difflet serve` (its stage is `placement="neuron_native"`, added in
#158). The serve adapters of FLUX, Qwen-Image, Wan, HunyuanVideo and LTX-2 take the Trainium NxD AoT path:
FLUX/LTX-2 compile through `_serving_compile_environment`, which forces `WORLD_SIZE=1`
(`common/orchestrators/flux.py:155-178`), and `NeuronBackend.prepare_runtime` refuses tp4 under it
(`backends/neuron/runtime.py:89-97`); Qwen imports `neuronx_distributed_inference`
(`serving/orchestrators/qwen_image.py:410`); Wan and HunyuanVideo import the Trainium application/config. No venv
on the host has `neuronx_distributed`. Those five run natively only through `difflet generate` (torchrun ranks
each calling `DiffletPipeline.from_pretrained`, `cli/orchestrators/torchrun.py:423`). The resident worker already
runs one process per rank on the neuron backend (`supports_torchrun_mpmd`, `resident_worker.py:545-563`), so native
serving for them is feasible but is new serving code — a separate project. Also: no serve-layout artifacts exist
on the host, and native NEFF reuse is keyed by graph bytes, so every backend's first start compiles; the only
MiniMax-H3 serve graphs were built at tp4×cp2 (two devices), not tp4 on one device.

## 3. Goals, non-goals, anti-goals

**Goals**

1. Every model that serves today is resident on its own device **[User decision]**: one device per model, as many
   as are supported, starting at device 0 **[User decision]**.
2. One endpoint; the request's `model` field picks the model **[User decision, from the original ask]**.
3. Shapes are the ones already compiled; the exact list is the user's to set later **[User decision]**.
4. A page shows every model's shape(s) so users know what to ask for **[User decision]**.
5. Different models run in parallel; each model runs one request at a time **[User decision — understood and
   accepted]**.

**Non-goals (v1)**

- Several replicas of one model, load balancing between them, autoscaling.
- Per-user accounts, quotas, billing. One shared API key, as `difflet serve` has.
- HunyuanVideo 1.5 (no Trainium serving).
- Compiling new shapes on request; changing the engine or `difflet serve`.
- Restarting a crashed backend automatically (see 6.3).

**Anti-goals**

- The page lists a shape the server does not actually serve.
- Taking devices another session on this host is using.
- A gateway restart forcing every model to reload (minutes each).

## 4. Constraints

- No change to `difflet serve` behavior; the gateway only talks HTTP to it.
- Backend errors reach the client unchanged (status and body); gateway errors use the same payload as
  `DiffletServingError.to_payload()`.
- Long requests: a video can wait in its queue for hours and then run for minutes. The gateway must never time
  out a forwarded request on its own, and a client disconnect must reach the backend (it cancels on disconnect).
- Shared host: every launch checks devices and CPUs are idle first (team rule) and the HY4 lock — section 7.

## 5. Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Gateway inside Difflet: `difflet gateway --config fleet.yaml`, plus a launcher that starts each `difflet serve` from the same file | Same code, error format and tests as the servers; the page and the servers read one config, so they cannot disagree. Cost: some deployment code in the engine repo |
| B | Same code in a separate deploy repo calling an installed Difflet | Engine stays engine-only; two repos to keep in step, error format copied, two CI setups |
| C | nginx routing by path (`/flux/v1/...`), static HTML page | No Python, but the model is picked by URL, not by the request's `model`; nginx cannot read a multipart field; the static page can drift from what runs |

The user picked **A** **[User decision]**.

## 6. Design

### 6.1 Topology

```
client ──► difflet gateway  0.0.0.0:8000   (API key here, page at /)
             ├─► difflet serve FLUX.1-dev    127.0.0.1:8100  device 0  cores 0-3
             ├─► difflet serve Qwen-Image    127.0.0.1:8101  device 1  cores 4-7
             ├─► difflet serve Wan 2.1       127.0.0.1:8102  device 2  cores 8-11
             ├─► difflet serve HunyuanVideo  127.0.0.1:8103  device 3  cores 12-15
             ├─► difflet serve LTX-2         127.0.0.1:8104  device 4  cores 16-19
             └─► difflet serve MiniMax-H3    127.0.0.1:8105  device 5  cores 20-23
```

Device order follows the list above **[My choice]**. Backends bind to `127.0.0.1` so only the gateway is
reachable from outside **[My choice]**. Ports `8100 + device` **[My choice]**.

Wan is served as `Wan-AI/Wan2.1-T2V-14B-Diffusers` only **[My choice]**: it is the qualified serving checkpoint;
2.2 serves with one of its two transformers, so its output is not real Wan 2.2. Adding 2.2 later is one more
config entry on device 6.

### 6.2 One config file

`fleet.yaml` is the single source for the launcher, the gateway's routing, and the page **[My choice]**. An
example ships at `examples/serving/fleet.yaml`; the real one lives wherever the operator keeps it.

```yaml
gateway: {host: 0.0.0.0, port: 8000}
runs_dir: /mnt/nvme/difflet-fleet/runs           # per-model working dirs, logs, pid files
models:
  - model_id: black-forest-labs/FLUX.1-dev
    device: 0
    port: 8100
    cache_dir: /mnt/nvme/difflet-fleet/cache/flux  # holds this shape's compiled artifacts and generated outputs
    serve_args: [--tp-degree, "4", --height, "1024", --width, "1024"]
  # ... one entry per model
```

Cache dirs and `runs_dir` go on `/mnt/nvme` **[My choice]**: generated videos land under `<cache_dir>` and the
root disk has 79 GB free. The inventory step copies each model's existing compiled artifacts for its chosen shape
into its fleet cache dir, so no backend compiles at startup. Video retention stays at the 25 h default; S3 stays
off **[My choice]**.

The shapes shown on the page are read from `serve_args` (`--height/--width/--num-frames` or `--shapes`), the same
arguments the launcher passes to `difflet serve`, so the page cannot list a shape the server was not started with.
The gateway also cross-checks each backend's `/v1/models` id against `model_id` at startup and marks a
mismatch as down.

### 6.3 Launcher

A small command, `difflet gateway launch --config fleet.yaml` **[My choice; the name is the builder's]**, starts
one detached `difflet serve` per entry and exits. Per entry it sets:

- `NEURON_RT_VISIBLE_CORES=4d..4d+3`, `NEURON_NUMA_PIN=0`, `taskset` to the device's NUMA node CPUs,
  `DIFFLET_BACKEND=neuron`, `PATH` with `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin` first (torch-neuronx
  spawns `neuronx-cc` by name);
- `--host 127.0.0.1 --port <port> --model-id <id> --cache-dir <cache_dir>` plus `serve_args`;
- `DIFFLET_API_KEY=""` (the backends must not inherit the gateway's key — the gateway strips `Authorization`,
  so an inherited key would make every `/v1` call 401) and `HF_HUB_OFFLINE=1` (a missing snapshot fails loudly
  instead of downloading onto the full root disk);
- its own working directory `<runs_dir>/<model-slug>/`, so each server's `logs/` stays separate; no `.env` is
  linked in (none exists; if S3 is wanted later, the launcher passes those variables explicitly);
- a pid file there, so `launch --stop` / `--status` and the gateway's page can find it.

Before starting an entry it refuses if that device is busy (`fuser /dev/neuron<d>`, checking holders'
`NEURON_RT_VISIBLE_CORES` as `difflet/bench/host.py` already does — reuse that code). Processes from the fleet's
own pid files are not conflicts. CPU business is reported, not refused **[My choice]**: the team's "devices and
CPUs idle" rule guards timing runs, and a resident service is not one. The bench `check_idle` CPU test would
refuse entry 2 because of entry 1, since devices 0–5 share NUMA node 1's CPUs.

All six backends are pinned to NUMA node 1's CPUs and keep the engine's thread count **[My choice]**.
`shortcut:` each rank runs `192 / ranks` threads on a 96-CPU node, so host-side stages of several models at
once (text encoders, a host VAE) contend; upgrade by making the worker size its threads from its CPU affinity
(an engine change, and it moves Wan's umT5 output bits, so it needs its own verification).

Starting is sequential by default: the launcher waits for each backend's port to answer before starting the
next, so six loads do not share the disk at once **[My choice]**. A backend whose startup log shows compilation is
reported as a wrong `cache_dir`.

Coordination with the HY4 lock is section 7's open question; the launcher implements whatever the user picks.

Backends outlive the gateway: restarting the gateway (config tweak, page change) does not reload any model
**[My choice — anti-goal 3]**. No automatic restart of a crashed backend in v1; `difflet serve` already restarts
its own worker on worker failure. `shortcut:` if a whole `difflet serve` process dies, the page shows it down
until someone reruns the launcher; add a systemd unit per model if that proves too manual.

### 6.4 Gateway routes

| Route | Behavior |
|---|---|
| `POST /v1/chat/completions` | Parse JSON, read `model` — required here (`400 model_required`), unknown `404 model_not_found`, a video model `400 wrong_endpoint` (pointing at `/v1/videos`). Forward the body as received and return the backend's response unchanged |
| `POST /v1/videos`, `POST /v1/videos/sync` | Buffer the raw body; parse the multipart form only to read `model` (same three errors); forward the raw bytes with the original `Content-Type`. Stream `/sync` bytes back |
| `GET /v1/videos/{id}`, `GET .../content`, `DELETE /v1/videos/{id}` | Ask the video backends in turn; the first non-404 answer wins; all 404 → `404`. Content is streamed **[My choice: stateless, survives a gateway restart; 4 local calls per lookup]** |
| `GET /v1/videos` | Requires `?model=`; forwarded to that backend with `model` removed from the query (the backend rejects unknown fields) **[My choice: avoids merging paginated lists across backends]** |
| `GET /v1/models` | One entry per configured model: the backend's entry plus `difflet_device`, `difflet_shapes`, `difflet_ready` |
| `GET /health` | Gateway process alive |
| `GET /ready` | Per-model readiness from each backend's `/ready`; `200` when all are ready, else `503`, body lists each **[My choice]** |
| `GET /` | The page (6.5) |

Forwarding: strip hop-by-hop headers and the client's `Authorization` (backends have no key); keep
`Content-Type`, `X-Request-Id`, etc. Backend unreachable → `503 model_unavailable` (with "loading" in the message
when its pid is alive). No read timeout on any forwarded request — the backend owns queue and request timeouts;
only the connect timeout is short. When the client disconnects, close the upstream request so the backend cancels. Use `httpx.AsyncClient` (declare `httpx` in `pyproject.toml`; it is already in the lock).

### 6.5 The page

Server-rendered HTML at `GET /`, no JavaScript framework, rendered on each load **[My choice]**:

| Model | Kind | Device | Shapes (H×W[×F]) | Status | Example |
|---|---|---|---|---|---|
| black-forest-labs/FLUX.1-dev | image | 0 | 1024×1024 | ready | `curl …` |

- Status combines the launcher's pid file and the backend's port (a loading backend refuses connections, so HTTP
  alone cannot tell loading from down): pid alive + `/ready` 200 → **ready**; pid alive + connection refused →
  **loading**; pid alive + 503 → **recovering**; no live pid → **down**. The gateway reads pid files from the
  config's `runs_dir`.
- The `/v1/models` id check runs whenever a backend first answers, not once at gateway startup (most backends
  are still loading then).
- One copy-paste example request per model, filled with that model's first shape.
- A line stating "one request at a time per model; others queue" and the 429 behavior.
- Open without the API key, like `/health` and `/ready` **[My choice — it lists models and shapes only]**.

### 6.6 Auth

`--api-key` / `DIFFLET_API_KEY` on the gateway, reusing `ApiKeyAuthenticationMiddleware`; backends run without a
key on loopback. Same caveat as today: one shared key; put the port behind a security group or TLS proxy.

## 7. Device and host constraints

- **Shared host [Open].** On 2026-11-19 the user said, through session `ubuntu-ee` (HY4 kernel campaign), no more
  Difflet device work on this box until told otherwise. Each HY4 whole-node lane takes
  `/home/ubuntu/ws/trainium-kernel-agent/locks/corenode.lock` and then waits until `fuser /dev/neuron*` is empty
  (`.../examples/hy4-mxfp8/e2e_lane.sh:8-9`). Every Neuron runtime process also opens `/dev/neuron0`, so **while
  the fleet is up, every HY4 whole-node job waits** — the node cannot be shared partially. And if the launcher only
  peeks at the lock, a HY4 lane can take it and then hang on our devices with nobody told.
  **[User decision, 2026-11-20]** "我们忽略其他的task会用这台机器，我们先站上device，我后续会通知大家的": the fleet
  takes devices 0–5 now; the launcher does not touch `corenode.lock`; the user tells the other sessions. The
  launcher still refuses a device whose cores another process holds — the Neuron runtime cannot share cores, so
  starting there would only fail later with a less clear error **[My choice]**.
- **Fit on one device [to verify].** Not yet run on trn3: whether every serving stage of each model fits in one
  device's 4 cores and 144 GB — especially MiniMax-H3 (TP4 DiT plus video and audio VAEs) and Wan (a registry
  layout that wants more than 4 cores would be refused). Step 1 of the plan starts each model alone on its device;
  a model that does not fit is reported back, not forced.
- **Weights on slow disk [Open].** Hub weights on the root EBS disk load at ~125 MB/s; six models starting at once
  share that. Options: start them one at a time (6.3's default), or point `HF_HOME` at a copy on `/mnt/nvme`
  (849 GB free, shared with other sessions). Builder default: sequential start from EBS **[My choice]**.
- **Compiled shapes [User decision + to inventory].** Plan step: list, per model, which shapes the existing caches
  hold for the `neuron` backend at tp4, and give the list to the user to choose from. Until then the config uses
  one compiled shape per model.

## 8. Testing

- Unit (CPU, no device): gateway routing against fake backends (small FastAPI apps or `httpx.MockTransport`) —
  model required / unknown / wrong endpoint, multipart forwarded byte-identical, video id fan-out, backend
  down → 503, `/ready` aggregation, page lists exactly the config's shapes, API key enforced on `/v1`.
  Follow `tests/unit/serving/test_api_server.py` style.
- Config: launcher builds the right env and argv per entry (pure function, no processes started).
- Device acceptance (after the user clears section 7): start the fleet, one request per model through the
  gateway, two different models concurrently, page shows all ready. Evidence under `docs/verification/`.

## 9. Left to the builder

Module layout (e.g. `difflet/serving/gateway/`), CLI wiring in `difflet/cli/main.py`, exact error codes'
wording, page styling, launcher flag names, how long the launcher waits per model.

## 10. Open questions for the user

1. ~~Shared host~~ — answered 2026-11-20: take the devices now, ignore other sessions (section 7).
2. ~~Weights~~ — no objection to the default: sequential start from EBS.
3. The shape list per model (after the inventory step) — the user will set it.

## 11. Review log

An independent reviewer read the first draft as its builder (2026-11-20). Corrections taken: a loading backend
refuses connections rather than answering 503 (page status now uses pid files); the backends must not inherit the
gateway's API key; no gateway read timeout (video queue timeout is 24 h); `model` stripped from the list query;
the bench CPU idle check would refuse the fleet's own entries; generated videos live under `cache_dir`, so caches
move to NVMe; `HF_HUB_OFFLINE=1`; Wan 2.1 vs 2.2 settled; the HY4 lane blocks on any `/dev/neuron*` holder.
