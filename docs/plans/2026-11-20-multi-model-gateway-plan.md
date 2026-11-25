# Multi-model gateway Implementation Plan

> **BLOCKED (2026-11-20):** only MiniMax-H3 can run `difflet serve` on this host's native neuron backend; the
> other five need native serving first (design doc §2, "Finding"). Tasks 2–6 (gateway code) are unaffected;
> Tasks 1 and 7 wait for the user's choice of how to proceed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Six `difflet serve` backends, one per device 0–5, behind one `difflet gateway` endpoint that routes by the request's `model` and serves a page listing each model's shapes and status.

**Architecture:** One `fleet.yaml` drives three things: `difflet gateway up/down/status` (start/stop pinned, detached `difflet serve` processes), `difflet gateway serve` (a FastAPI reverse proxy using `httpx.AsyncClient`), and the page at `/`. The backends and the HTTP protocol are unchanged; the gateway only forwards bytes.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx 0.28, PyYAML, pytest + `fastapi.testclient.TestClient` + `httpx.MockTransport`.

**Spec:** `docs/plans/2026-11-20-multi-model-gateway-design.md` (read it first; section numbers below refer to it).

## Global Constraints

- No behavior change to `difflet serve` or the engine (spec §4). The only edit outside the new package is a new public helper in `difflet/serving/model_registry.py`, the CLI wiring, `pyproject.toml`, and docs.
- Gateway-originated errors use `DiffletServingError(status, code, message).to_payload()`; backend responses pass through with their status and body unchanged (spec §4).
- No read timeout on any forwarded request; connect timeout 5 s (spec §6.4).
- Backends bind `127.0.0.1`, port from the config; the gateway binds the config's `gateway.host:gateway.port` (spec §6.1).
- Backend env always has `DIFFLET_API_KEY=""`, `HF_HUB_OFFLINE=1`, `DIFFLET_BACKEND=neuron`, `NEURON_NUMA_PIN=0`, `NEURON_RT_VISIBLE_CORES=<device's cores>` (spec §6.3).
- `httpx>=0.28` declared in `pyproject.toml` `dependencies` (already in `requirements-neuron-native.lock`).
- Unit tests must pass on a CPU-only box with no Neuron SDK: monkeypatch `topology`, `device_holders`, `numa_cpus`, `neuronx_cc_dir`; never call `neuron-ls`. Before pushing, run them once with `/opt/aws/neuron/bin` removed from `PATH`.
- Test command (from the worktree root, on idle CPUs — check `mpstat -P ALL 2 1` first):
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0 DIFFLET_BACKEND=cpu PYTHONPATH=. taskset -c <idle cpus> /home/ubuntu/venvs/native-fp8/bin/python -m pytest <path> -q -p no:cacheprovider`
- Device runs: `export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH`; the user cleared devices 0–5 for this fleet regardless of other sessions (spec §7), but a device whose cores another process holds is still refused.
- One commit per task, pushed right after: `git push origin HEAD:feat/multi-model-gateway`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A video request carrying keyframe files (MiniMax-H3 `first_frame`/`last_frame`) must reach the backend byte-identical — test in Task 5 (`test_multipart_with_file_forwarded_byte_identical`).
2. A client that disconnects while a long chat/sync-video request is still waiting for its first byte must cause the upstream request to be cancelled — test in Task 5 (`test_client_disconnect_cancels_upstream`).
3. A video id that no backend knows, while one video backend is down, must answer `503` (not a misleading `404`) — test in Task 5 (`test_video_lookup_with_a_backend_down_is_503`).
4. `model` with a trailing slash (`Qwen/Qwen-Image/`) must route like the bare id, as the backends normalize it — test in Task 2 (`test_entry_lookup_ignores_trailing_slash`).
5. A backend that answers `/ready` 200 with a different model id (wrong config entry) must show as `down`, not `ready` — test in Task 4 (`test_ready_for_another_model_is_down`).

---

### Task 1: Device fit check and compiled-shape inventory (no code)

> Finalized after the read-only cache inventory (see "Task 1 inputs" at the end of this plan).

**Files:**
- Create: `docs/verification/2026-11-20-multi-model-gateway-fit.md` (evidence)
- Create: `examples/serving/fleet.yaml` (the real values found here; Task 2 validates its schema)

- [ ] **Step 1:** For each model in spec §6.1 order (device 0..5), pick the compiled shape listed in "Task 1 inputs", copy its compiled artifacts into `/mnt/nvme/difflet-fleet/cache/<slug>/` (layout per "Task 1 inputs").
- [ ] **Step 2:** Check device d idle (`fuser /dev/neuron<d>`, holders' `NEURON_RT_VISIBLE_CORES`), then start that model alone:
  `cd /mnt/nvme/difflet-fleet/runs/<slug> && DIFFLET_API_KEY= HF_HUB_OFFLINE=1 DIFFLET_BACKEND=neuron NEURON_NUMA_PIN=0 NEURON_RT_VISIBLE_CORES=<4d>-<4d+3> taskset -c 48-95,144-191 /home/ubuntu/venvs/native-fp8/bin/python -m difflet.cli.main serve --model-id <id> --host 127.0.0.1 --port <8100+d> --cache-dir /mnt/nvme/difflet-fleet/cache/<slug> --tp-degree 4 <shape flags>`
  Expected: `/ready` 200 without any compile line in the log; record wall time to ready.
- [ ] **Step 3:** One request per model with `curl` (image: README "Generate an image"; video: `/v1/videos/sync`), default steps. Expected: 200 and a valid PNG/MP4. Record latency.
- [ ] **Step 4:** Leave it running only if the next model is on another device (they are); after all six are up, stop them all (`kill -TERM`).
- [ ] **Step 5:** A model that does not fit on one device, or compiles at startup: record the error, drop it from `fleet.yaml`, and report it to the user — do not change its layout.
- [ ] **Step 6:** Write `fleet.yaml` with the six (or fewer) entries and the evidence doc (command, time to ready, request latency, output md5 per model). Commit both; push.

### Task 2: Fleet config

**Files:**
- Create: `difflet/serving/gateway/__init__.py` (empty), `difflet/serving/gateway/fleet.py`
- Modify: `difflet/serving/model_registry.py` (add `serving_output_modality` after `model_accepts_keyframes`, line ~203)
- Modify: `pyproject.toml` (`dependencies`: add `"httpx>=0.28",  # difflet gateway reverse proxy`)
- Test: `tests/unit/serving/gateway/__init__.py`, `tests/unit/serving/gateway/test_fleet.py`

**Interfaces:**
- Produces:
  - `model_registry.serving_output_modality(model_id: str) -> str` — `"image"`/`"video"` of the built-in model whose `checkpoint_ids` contain `model_id.rstrip("/")`; `ValueError` if none. Same shape as `model_accepts_keyframes`.
  - `class FleetConfigError(ValueError)`
  - `@dataclass(frozen=True) class FleetEntry: model_id: str; device: int; port: int; cache_dir: Path; serve_args: tuple[str, ...]` with properties `slug -> str` (lower-case, `/` and `.` → `-`: `black-forest-labs-flux-1-dev`), `kind -> str` (via `serving_output_modality`), `shapes -> tuple[tuple[int, int, int | None], ...]` (H, W, F-or-None, from `--shapes HxWxF,...` or `--height/--width[/--num-frames]`), `base_url -> str` (`http://127.0.0.1:<port>`).
  - `@dataclass(frozen=True) class FleetConfig: host: str; port: int; runs_dir: Path; models: tuple[FleetEntry, ...]` with `entry(model_id: str) -> FleetEntry | None` (trailing `/` ignored), `run_dir(e) -> Path` (`runs_dir / e.slug`), `pid_file(e) -> Path` (`run_dir(e) / "serve.pid"`), `log_file(e) -> Path` (`run_dir(e) / "serve.out"`), `video_entries() -> tuple[FleetEntry, ...]`.
  - `load_fleet(path: str | Path) -> FleetConfig` — YAML schema of spec §6.2.

- [ ] **Step 1: Write the failing tests** in `test_fleet.py`, using `tmp_path` YAML files:
  - `test_load_example_fleet` — loads `examples/serving/fleet.yaml` (Task 1); every entry's `kind` is `image` or `video`, devices are `0..n-1`.
  - `test_image_shape_from_height_width` — `serve_args: [--height, "1024", --width, "1024"]` → `shapes == ((1024, 1024, None),)`.
  - `test_video_shapes_from_shapes_flag` — `[--shapes, "320x512x61,320x512x33"]` → `((320, 512, 61), (320, 512, 33))`.
  - `test_duplicate_device_port_or_model_rejected` — each duplicate raises `FleetConfigError` naming the field.
  - `test_gateway_port_collision_rejected`, `test_owned_flag_in_serve_args_rejected` (`--host`, `--port`, `--model-id`, `--cache-dir` → `FleetConfigError`), `test_missing_shape_rejected`, `test_unservable_model_rejected` (`hunyuanvideo-community/HunyuanVideo-1.5-...` id from `difflet/models/hunyuan_video_15/spec.py`).
  - `test_entry_lookup_ignores_trailing_slash` — `config.entry("Qwen/Qwen-Image/") is config.entry("Qwen/Qwen-Image")`.
  - `test_serving_output_modality` — FLUX → `"image"`, `Lightricks/LTX-2` → `"video"`, unknown → `ValueError`.
- [ ] **Step 2:** Run them; expected FAIL (`ModuleNotFoundError: difflet.serving.gateway`).
- [ ] **Step 3:** Implement `serving_output_modality`, `fleet.py`, the `pyproject.toml` line.
- [ ] **Step 4:** Run; expected PASS. Also `python -c "import difflet.serving.gateway.fleet, sys; assert 'torch' not in sys.modules"`.
- [ ] **Step 5:** Commit `feat(gateway): fleet config`; push.

### Task 3: Launcher (`up` / `down`)

**Files:**
- Create: `difflet/serving/gateway/launcher.py`
- Test: `tests/unit/serving/gateway/test_launcher.py`

**Interfaces:**
- Consumes: `FleetConfig`, `FleetEntry` (Task 2); `difflet.bench.host.devices_to_cores`, `device_holders`, `numa_cpus`, `topology`, `neuronx_cc_dir`, `format_cpu_list`.
- Produces:
  - `class LauncherError(RuntimeError)`
  - `backend_env(entry: FleetEntry, base_env: Mapping[str, str], *, visible_cores: str, cc_dir: str | None) -> dict[str, str]` — `base_env` plus the Global Constraints variables; `cc_dir` prepended to `PATH` when not None.
  - `backend_argv(entry: FleetEntry, *, python: str, cpus: str) -> list[str]` — `["taskset", "-c", cpus, python, "-m", "difflet.cli.main", "serve", "--model-id", id, "--host", "127.0.0.1", "--port", str(port), "--cache-dir", str(cache_dir), *serve_args]`.
  - `read_pid(pid_file: Path) -> int | None` — the pid if the file exists and the process is alive, else None.
  - `up(config: FleetConfig, *, wait_s: float = 3600.0) -> None` — entries in config order: skip if `read_pid` alive; refuse (`LauncherError` naming pid/comm/cores) if `device_holders([d])[d]` has a pid not in any fleet pid file; `mkdir -p run_dir`; `Popen(argv, env=..., cwd=run_dir, stdout=stderr=log_file (append), start_new_session=True)`; write pid; wait until a TCP connect to `127.0.0.1:port` succeeds (poll 5 s), else `LauncherError` with the last 40 log lines if the process exits or `wait_s` passes. CPUs: the device's NUMA node (`topology()["devices"][d].numa_node` → `numa_cpus()`).
  - `down(config: FleetConfig, *, grace_s: float = 60.0) -> None` — `os.killpg(pid, SIGTERM)`, wait `grace_s`, then `SIGKILL`; remove the pid file.

- [ ] **Step 1: Write the failing tests** (monkeypatch every `bench.host` call and `subprocess.Popen`; use a socket listener on an ephemeral port for the wait):
  - `test_backend_env_device_2` — `visible_cores="8-11"` → env has `NEURON_RT_VISIBLE_CORES == "8-11"`, `DIFFLET_API_KEY == ""`, `HF_HUB_OFFLINE == "1"`, `DIFFLET_BACKEND == "neuron"`, `NEURON_NUMA_PIN == "0"`; base `DIFFLET_API_KEY="secret"` is overridden.
  - `test_backend_env_prepends_cc_dir` — `cc_dir="/x/bin"`, base PATH `/usr/bin` → `PATH == "/x/bin:/usr/bin"`.
  - `test_backend_argv` — exact list for a FLUX entry, `cpus="48-95,144-191"`.
  - `test_up_refuses_busy_device` — `device_holders` returns a foreign pid → `LauncherError` mentioning it; `Popen` not called.
  - `test_up_ignores_fleet_own_pids` — the holder is a pid from another entry's pid file → launches.
  - `test_up_skips_running_entry`, `test_up_waits_for_port_then_next` (order of Popen calls follows config; second Popen only after the first port answers), `test_up_reports_log_tail_when_backend_exits`.
  - `test_down_terms_then_kills` — process group gets SIGTERM, then SIGKILL after grace; pid file removed.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement `launcher.py`.
- [ ] **Step 4:** Run; expected PASS (also with `/opt/aws/neuron/bin` off `PATH`).
- [ ] **Step 5:** Commit `feat(gateway): launcher for pinned difflet serve backends`; push.

### Task 4: Status and page

**Files:**
- Create: `difflet/serving/gateway/status.py`, `difflet/serving/gateway/page.py`
- Test: `tests/unit/serving/gateway/test_status.py`, `tests/unit/serving/gateway/test_page.py`

**Interfaces:**
- Consumes: Task 2 config; `read_pid` (Task 3).
- Produces:
  - `Status = Literal["ready", "loading", "recovering", "down"]`
  - `async def backend_status(entry: FleetEntry, pid_file: Path, client: httpx.AsyncClient) -> Status` — no live pid → `down`; `GET {base_url}/ready` with a 2 s timeout: connect error → `loading`; 200 whose JSON `model` equals `entry.model_id` → `ready`, other model → `down`; 503 → `recovering`; anything else → `down`.
  - `async def fleet_status(config: FleetConfig, client: httpx.AsyncClient) -> dict[str, Status]` — keyed by `model_id`, all backends probed concurrently.
  - `render_page(config: FleetConfig, statuses: Mapping[str, Status], *, base_url: str) -> str` — full HTML document (spec §6.5): one table row per entry (model, kind, device, shapes as `H×W` or `H×W×F`, status), one example request per model built from its first shape (image: chat completions JSON with `height`/`width` in `extra_body`; video: `/v1/videos` multipart with `size=<W>x<H>` and `num_frames=<F>`), and the line "Each model runs one request at a time; others wait in its queue (HTTP 429 when the queue is full or times out)." All values HTML-escaped.

- [ ] **Step 1: Write the failing tests** (`httpx.MockTransport` for `/ready`; a temp pid file holding `os.getpid()` for "alive", a non-existent pid for "dead"):
  - `test_status_matrix` — (alive, 200 same model) → ready; (alive, connect error) → loading; (alive, 503) → recovering; (dead, *) → down.
  - `test_ready_for_another_model_is_down`.
  - `test_page_lists_exactly_config_shapes` — HV entry `--shapes 320x512x61,320x512x33` → page contains `320×512×61` and `320×512×33` and no other `×` shape for it.
  - `test_page_video_example_uses_width_by_height` — contains `size=512x320` and `num_frames=61`.
  - `test_page_escapes_html`, `test_page_shows_status_and_queue_line`.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement. **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Commit `feat(gateway): backend status and models page`; push.

### Task 5: Gateway app (routing)

**Files:**
- Create: `difflet/serving/gateway/app.py`
- Test: `tests/unit/serving/gateway/test_app.py`

**Interfaces:**
- Consumes: Tasks 2–4; `difflet.serving.auth.ApiKeyAuthenticationMiddleware`; `difflet.serving.errors.DiffletServingError`; `difflet.serving.openai.api_server._run_until_disconnect` (reuse it for disconnect → cancel).
- Produces: `create_gateway_app(config: FleetConfig, *, api_key: str | None = None, client: httpx.AsyncClient | None = None, status_fn: Callable[[], Awaitable[dict[str, Status]]] | None = None) -> FastAPI` — `client` defaults to `httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))` opened/closed in the lifespan; `status_fn` defaults to `fleet_status(config, client)`.
- Routes and error codes exactly as spec §6.4: `model_required` (400), `model_not_found` (404), `wrong_endpoint` (400), `model_unavailable` (503). Request headers dropped before forwarding: `host`, `content-length`, `authorization`, `connection`, `keep-alive`, `transfer-encoding`, `te`, `trailer`, `upgrade`, `proxy-authorization`, `proxy-authenticate`. Responses are streamed (`client.send(..., stream=True)` → `StreamingResponse(resp.aiter_raw(), ...)`, closed in a background task). `/v1/models` is built from config + `status_fn()` with `difflet_kind`, `difflet_device`, `difflet_shapes` (`["1024x1024"]`, `["320x512x61"]`), `difflet_status`.

- [ ] **Step 1: Write the failing tests** — fake backends as one `httpx.MockTransport` handler that records each request and answers per `request.url.port`; app via `TestClient(create_gateway_app(cfg, client=AsyncClient(transport=...), status_fn=...))`:
  - `test_chat_routes_by_model` (FLUX request reaches port 8100 only; response body/status passed through), `test_chat_model_required`, `test_chat_unknown_model_404`, `test_chat_video_model_wrong_endpoint`.
  - `test_multipart_with_file_forwarded_byte_identical` — a form with `model`, `prompt` and a PNG `first_frame` file: the backend sees the same bytes and `Content-Type` (boundary included).
  - `test_videos_sync_streams_bytes`, `test_backend_error_passed_through` (backend `429` body unchanged).
  - `test_authorization_not_forwarded`; `test_api_key_required_on_v1_only` (`/v1/models` 401 without key; `/`, `/health`, `/ready` open).
  - `test_video_id_fan_out` (second video backend owns the id → its JSON), `test_video_content_streams_from_owner`, `test_video_delete_fan_out`, `test_unknown_video_id_404`, `test_video_lookup_with_a_backend_down_is_503`.
  - `test_list_videos_requires_model_and_strips_it` (backend sees `limit`/`after` only).
  - `test_backend_down_503` (connect error on chat), `test_ready_aggregate` (all ready → 200, one loading → 503 with per-model body), `test_models_lists_config`.
  - `test_client_disconnect_cancels_upstream` — call the chat route handler's forwarding coroutine under `_run_until_disconnect` with a fake request whose `receive()` yields `http.disconnect`; the MockTransport handler (async, blocks on an event) observes cancellation.
- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement `app.py`. **Step 4:** Run the whole `tests/unit/serving/gateway` dir; expected PASS.
- [ ] **Step 5:** Commit `feat(gateway): model-routing reverse proxy`; push.

### Task 6: CLI and docs

**Files:**
- Create: `difflet/cli/gateway.py`, `docs/serving/gateway.md`
- Modify: `difflet/cli/main.py` (subparser after `serve`, ~line 915; dispatch beside `elif args.command == "serve":`, ~line 1488), `README.md` (Serving section: 3-line pointer to `docs/serving/gateway.md`)
- Test: `tests/unit/cli/test_cli_gateway.py`

**Interfaces:**
- Consumes: `load_fleet`, `up`, `down`, `fleet_status`, `create_gateway_app`.
- Produces: `difflet gateway {serve,up,down,status} --config PATH [--api-key KEY]`; `difflet/cli/gateway.py: run(args: argparse.Namespace) -> None`. `serve`: `uvicorn.run(create_gateway_app(cfg, api_key=args.api_key or os.environ.get("DIFFLET_API_KEY") or None), host=cfg.host, port=cfg.port, workers=1)`. `status`: prints `model  device  port  status` rows. `LauncherError`/`FleetConfigError` → one line on stderr, exit 1 (as `serve` does with `DiffletServingError`).

- [ ] **Step 1: Write the failing tests:** `test_gateway_parser_actions` (each action parses, `--config` required), `test_gateway_up_calls_launcher` and `test_gateway_serve_builds_app` (monkeypatched `up`/`uvicorn.run`), `test_gateway_config_error_exits_1`.
- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement CLI + `docs/serving/gateway.md` (start/stop, config reference, routes table, status meanings, limits: one request per model, fixed shapes, no backend auto-restart, CPU sharing shortcut). **Step 4:** Run `tests/unit/serving/gateway tests/unit/cli/test_cli_gateway.py tests/unit/serving/test_api_server.py`; PASS.
- [ ] **Step 5:** Commit `feat(cli): difflet gateway serve/up/down/status`; push.

### Task 7: On-device acceptance

**Files:**
- Create: `docs/verification/2026-11-20-multi-model-gateway.md`

- [ ] **Step 1:** Check devices 0–5 (and holders), then `difflet gateway up --config examples/serving/fleet.yaml`. Expected: each backend reaches its port in order, no compile in any `serve.out`.
- [ ] **Step 2:** `difflet gateway serve --config ... &`; `curl :8000/ready` → 200; `curl :8000/` saved as `page.html` (shows all ready and each shape).
- [ ] **Step 3:** Through `:8000`, one request per model (images via chat completions; videos via `/v1/videos` + poll + `/content`). Expected: 200, valid PNG/MP4, md5 equal to Task 1's direct-to-backend output for the same seed.
- [ ] **Step 4:** Two different models at once (FLUX + Wan): both finish; wall time ≈ max of the two, not the sum.
- [ ] **Step 5:** Restart only the gateway: backends keep their pids; `/ready` 200 again within seconds.
- [ ] **Step 6:** Write the evidence doc (commands, times, md5s, page excerpt); commit; push. Leave the fleet running (the user wants the devices held) and report how to stop it: `difflet gateway down --config ...`.

## Task 1 inputs

_To be filled from the read-only cache inventory before the user confirms this plan: per model, the compiled shape, the source cache dir, what to copy into `/mnt/nvme/difflet-fleet/cache/<slug>/`, and whether a hub snapshot is present._
