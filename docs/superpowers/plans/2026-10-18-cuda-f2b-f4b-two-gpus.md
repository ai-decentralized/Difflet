# CUDA F2b/F4b — Serving on two GPUs and the performance loose ends, one 2-GPU pod Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two items the status log carries as open: (1) **serving on two GPUs** (F2b) — `DIFFLET_BACKEND=cuda difflet serve --model-id Qwen/Qwen-Image --dp-degree 2` keeps one resident worker per GPU and runs two concurrent requests in parallel, so the F3b finding ("the second concurrent request waited on the run lock and got 429 after 30 s") becomes two images in one request-time; (2) **F4b performance** — the Wan generate gap vs diffusers (Difflet 50.9 s vs diffusers 42.9 s for 40 steps on the Server Edition, F1b/F9) located with a DiT-level bench on identical inputs, Flux e2e timings recorded next to diffusers', and a second Server-Edition data point for the Qwen DiT bench. Everything runs on one 2-GPU eu pod (the owner's yes, 2026-10-18: one 2-GPU eu pod, created once the Mac side is green, terminated at the end).

**Architecture:** *Serving.* `ServeOptions.dp_degree` (new `--dp-degree`, default 1). On CUDA at `dp_degree=N>1` the factory builds N `ResidentWorkerServingEngine`s — each over a copy of the runtime bundle whose plan's `environment.available_core_ids` is `(i,)` — and wraps them in a new `ReplicatedServingEngine` (`difflet/serving/engines/replicated.py`): `start`/`shutdown` fan out (loads in parallel, as the DP router does), `ready` = every replica ready, `healthy` = every replica healthy, `generate` dispatches to the replica with the fewest pending requests (a new read-only `pending_requests` property on the resident engine). The resident worker pins its GPU: `_apply_worker_runtime_environment` exports `CUDA_VISIBLE_DEVICES` from the allocation's visible core ids when the plan's stages are placed on `cuda` (mapped through a parent's own `CUDA_VISIBLE_DEVICES`, as `difflet.cli.dp.router._cuda_worker_env` does), so each worker's `cuda:0` is its own card. Refusals before any download: `--dp-degree > 1` on any backend but CUDA ("F2b: data-parallel serving replicas are CUDA-only"); on CUDA, `--dp-degree` above `torch.cuda.device_count()`. `dp_degree=1` is byte-for-byte today's path (one engine, no wrapper). *Smoke.* `scripts/cuda_serve_smoke.py` gains `--dp N` (passes `--dp-degree`) and `--concurrent K` (K requests fired at once from threads; per-request and wall times; the images compared with the serial reference by bytes). *Performance.* `scripts/cuda_bench_wan_dit.py`: one expert (the high-noise `transformer`) forward, Difflet `CudaWanTransformerApplication` vs diffusers `WanTransformer3DModel`, identical inputs (latents `(1, 16, 3, 60, 104)`, timestep 999, text `(1, 512, 4096)`), warm-up, median of N with CUDA events, one profiled forward bucketed by `difflet.backends.cuda.bench` — the F4 recipe; plus a loop-overhead probe (the orchestrator's per-step host work: scheduler step, host↔device copies) timed from a short `--steps 4` generate with the DiT replaced by a recorder. `scripts/cuda_e2e_flux_vs_diffusers.py` records `timings.{difflet,diffusers}.{load_s, generate_s, peak_alloc_gb}` as the Wan oracle does.

**Tech Stack:** FastAPI/uvicorn serving stack, `asyncio`, `multiprocessing` spawn; torch 2.11 cu128, diffusers 0.38; pytest on CPU with the fake worker; Yottalabs API through `scripts/yottalabs_pod.py`.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — D5 (one process per GPU), D6 (DP first), D7 (SDPA, no flash-attn), D4 (eager; `torch.compile` a measured option), the F3b entry (serving overhead none; the 429 on the second concurrent request), the F2 entry (F2b list: serving on two GPUs), the F4 entry (the bench recipe) and the F1/F1b/F9 entries (the Wan generate gap, "F4b"); the `difflet-cuda-verify` skill for the pod session; the evidence report sections "serve", "two GPUs", "performance", NOT MEASURED.

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task (`.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`: regressions=0, new_skips=0); `tests/unit/serving` under both `DIFFLET_BACKEND=cpu` and `=cuda` (the suite branches on backend since F3b); black 100; one task, one commit. Branch `feat/cuda-f2b-f4b-two-gpus` on `feat/cuda-f9b-relative-wan-gate` (eed5d3c, PR #93).
- Trainium/TPU serving behaviour unchanged: `dp_degree=1` takes today's code path; the resident engine gains one read-only property and the worker one CUDA-only environment export; `tests/unit/serving` keeps its pass set.
- Pod rule (the skill): one 2-GPU pod in eu-3/eu-2 (Server Edition, ~$5.30/h, no volume, 250 GB container disk, `HF_HOME=~/hf`), created with `scripts/yottalabs_pod.py create --gpus 2 --wait-for-capacity` only after Tasks 1–3 are green and committed; logs under `~/logs/f2b`; the pod is **terminated** at the end (it cannot be paused), status read back; the pod ID goes in the ledger the moment it exists. Never leave it idle: Mac-side fixes longer than ~30 min happen after termination (Task 5 ruling).
- Flux is gated: its cells run only if the owner does the device login while at the phone; otherwise they are recorded NOT MEASURED and the plan is not blocked on them.
- No gate is loosened; every numeric claim comes from a JSON under `artifacts/cuda-release1/f2b/`.

## Review Focus

1. Replica dispatch and lifecycle: two concurrent requests reach two different workers (Task 1 `test_two_concurrent_requests_run_on_two_replicas` with the fake worker and `sleep:` prompts — wall ≈ one request); a third queues on the least-loaded replica; `ready` only when every replica is ready; a dead replica makes `healthy` false and recovery is per replica; `shutdown` tears every replica down.
2. GPU pinning: the worker exports `CUDA_VISIBLE_DEVICES=<core id>` only for a `cuda`-placed plan, mapped through the parent's `CUDA_VISIBLE_DEVICES` when set (Task 1 `test_cuda_worker_pins_its_gpu`, `test_cuda_worker_pins_within_the_parent_s_visible_devices`); the Trainium path exports nothing new (`test_neuron_worker_environment_is_unchanged`).
3. Refusals before any download: `--dp-degree 2` on trainium → 400 `feature_not_supported` naming F2b; on CUDA with one visible device → refused naming the device count (`test_dp_degree_is_refused_before_the_download`).
4. The bench measures what it claims: identical inputs on both sides (shape, dtype, timestep), both experts selectable (`--expert transformer_2`), `cfg` batch 2 selectable; kernel buckets from `difflet.backends.cuda.bench`; the loop-overhead probe counts the host-side time per step without a GPU (Task 3 CPU tests with a fake DiT).
5. The pod numbers: dp 2 concurrent-2 wall vs dp 1 serial two requests; images byte-identical to the dp 1 image for the same seed; Wan bench Difflet vs diffusers forward ms with the buckets; Flux timings if measured.

---

### Task 1: `difflet serve --dp-degree N` on CUDA — replicated resident engines, one GPU each

**Files:**
- Create: `difflet/serving/engines/replicated.py` (`ReplicatedServingEngine`)
- Modify: `difflet/serving/engines/__init__.py` (export), `difflet/serving/engines/resident_worker.py` (`pending_requests` property; `_apply_worker_runtime_environment` CUDA export), `difflet/serving/options.py` (`dp_degree: int = 1`, validated ≥ 1), `difflet/cli/serve.py` (`dp_degree=getattr(args, "dp_degree", 1)`), `difflet/cli/main.py` (`--dp-degree` on the serve parser), `difflet/serving/factory.py` (N engines + wrapper on CUDA; refusal elsewhere), `difflet/serving/orchestrators/qwen_image_cuda.py` (`refuse_unsupported_profile` → also `refuse_unsupported_options(options)`: dp above the device count), `difflet/serving/model_registry.py` (the non-CUDA refusal for `dp_degree > 1`)
- Test: `tests/unit/serving/test_replicated_engine.py` (new), `tests/unit/serving/test_resident_worker_engine.py` (CUDA pinning), `tests/unit/serving/test_qwen_image_cuda_serving.py` (refusal), `tests/unit/serving/test_serve_cli.py` (`--dp-degree` mapping)

- [ ] **Step 1: Write the failing tests** (Review Focus 1–3): the replicated engine over two fake-worker resident engines (`sleep:0.3` prompts fired together complete in < 0.5 s wall; `ready`/`healthy` aggregation; shutdown fans out; dispatch picks the replica with fewer pending); the worker environment test with a `cuda`-placed plan and `available_core_ids=(1,)` sees `CUDA_VISIBLE_DEVICES=1` (and `=5` when the parent has `CUDA_VISIBLE_DEVICES=4,5`); the serve CLI maps `--dp-degree 2`; `build_serving_stack` with `dp_degree=2` on cpu/trainium raises `feature_not_supported` naming F2b before `prepare_runtime`; on CUDA with `torch.cuda.device_count()` patched to 1 it is refused naming the count, with 2 it builds a `ReplicatedServingEngine` of two engines whose plans claim core 0 and core 1.
- [ ] **Step 2: Run** `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/serving -q -p no:cacheprovider -k "replicated or dp_degree or pins"`. Expected: the new tests fail (no module, no option).
- [ ] **Step 3: Implement** as in Architecture; black.
- [ ] **Step 4: Run** `tests/unit/serving` under `DIFFLET_BACKEND=cpu` and `=cuda` (Expected: the pre-existing pass set + the new tests), then the CPU gate.
- [ ] **Step 5: Commit** `serve: --dp-degree N on CUDA — one resident worker per GPU behind a replicated engine (F2b task 1)`.

### Task 2: The serve smoke drives replicas and concurrency

**Files:**
- Modify: `scripts/cuda_serve_smoke.py` (`--dp N`, `--concurrent K`, `--reference PNG` to compare bytes; `serve_timings.json` gains `dp`, `concurrent`, `wall_s`, per-request `request_s`, `identical_to_reference`), `scripts/cuda_serve_smoke.sh` (unchanged interface)
- Test: `tests/unit/scripts/test_cuda_serve_smoke.py` (`serve_command` carries `--dp-degree`; `fire_concurrently` with a fake poster returns K results and a wall; the comparison helper)

- [ ] **Step 1:** failing tests → **Step 2:** run → **Step 3:** implement → **Step 4:** `tests/unit/scripts` + CPU gate → **Step 5: Commit** `serve smoke: --dp and --concurrent, byte comparison with a reference image (F2b task 2)`.

### Task 3: The Wan DiT bench, the loop-overhead probe, Flux e2e timings

**Files:**
- Create: `scripts/cuda_bench_wan_dit.py` (cases `difflet`, `diffusers`, `--expert transformer|transformer_2`, `--cfg-batch 2`, `--loop-probe` → JSON `bench_wan_dit.json` with `timed.median_ms`, `profile` buckets, `attention_kernels`, `loop_probe.{per_step_ms, scheduler_ms, copies_ms}`)
- Modify: `scripts/cuda_e2e_flux_vs_diffusers.py` (timings per side as the Wan oracle records them), `difflet/backends/cuda/bench.py` only if a helper is missing
- Test: `tests/unit/scripts/test_cuda_bench_wan_dit.py` (argument parsing; identical-input construction from a config; the loop probe on a fake DiT counts steps and host time), `tests/unit/scripts/test_cuda_e2e_flux_script.py` (the timings keys)

- [ ] **Step 1:** failing tests → **Step 2:** run → **Step 3:** implement; `--help` on the CPU host → **Step 4:** `tests/unit/scripts` + CPU gate → **Step 5: Commit** `perf: Wan DiT bench vs diffusers with a loop-overhead probe; Flux e2e timings (F4b task 3)`.

### Task 4: The pod session (2 GPUs)

- [ ] **Step 1:** `scripts/yottalabs_pod.py create --name difflet-cuda-f2b --gpus 2 --regions eu-3,eu-2 --disk-gb 250 --wait-for-capacity` (the owner's yes is recorded in the plan header); `wait-running`, `ssh-config --alias difflet-cuda-f2b`, `ship_branch.sh difflet-cuda-f2b feat/cuda-f2b-f4b-two-gpus`, `pod_setup.sh` (detached, poll `SETUP DONE`); pod ID in the ledger at once; `nvidia-smi` on arrival (two cards, `nvidia-smi topo -m`).
- [ ] **Step 2:** downloads to `HF_HOME=~/hf`: Qwen-Image, Wan 2.2 (ungated, ~176 GB); Flux only if the owner is at the phone for the device login.
- [ ] **Step 3 (F2b serving):** `cuda_serve_smoke.sh --dp 1 --requests 2 --steps 50` (reference: ready, two serial requests, the image), then `--dp 2 --concurrent 2 --steps 50 --reference <dp1 png>` (two loads in parallel; two requests in one request-time; bytes identical), then `--dp 2 --concurrent 3` (the third queues; raise `--queue-timeout` above a request); `nvidia-smi` while serving (one worker per card); probes: `--dp-degree 3` refused naming the count. Copy `serve_timings.json`s and PNGs.
- [ ] **Step 4 (F4b):** `cuda_bench_dit.py --model-path <qwen>` (the Server-Edition data point next to F4's f3c numbers); `cuda_bench_wan_dit.py --model-path <wan>` for both experts, `--cfg-batch 2`, `--loop-probe`; read the table: forward ms Difflet vs diffusers and the buckets; the probe's per-step host ms × 40 vs the 8 s gap.
- [ ] **Step 5 (stretch, cheap):** Wan `difflet run --dp 2 --requests` with two requests (seeds 0, 1) vs serial — closes a NOT MEASURED cell; Flux e2e with timings if Flux is present.
- [ ] **Step 6:** copy artifacts to `artifacts/cuda-release1/f2b/`; **terminate** the pod (`delete`, read back gone) unless Task 5's ruling keeps it for a < 30 min fix.

### Task 5: What the measurements name (ruling at the time)

- [ ] If the Wan bench shows Difflet's forward slower than diffusers' by a bucket the profile names (attention kernel, an fp32 upcast, a Megatron layer at tp=1, RoPE rebuilt per call), and the fix is a contained change under `difflet/models/wan/` or `difflet/backends/cuda/`: implement on the Mac with a unit test pinning it, CPU gate, `ship_branch.sh`, re-run the bench and the Wan parity + e2e gates on the pod. If the gap is the host loop (the probe), the fix is device-resident latents for the CUDA orchestrator — larger; record the number and the design as the follow-up, no code. Pod time is the budget: a fix that is not green on the Mac within ~30 min of the measurement → terminate first, fix after, verify on a later 1-GPU pod.

### Task 6: Docs, status log, evidence report, skill catalog

- [ ] Append the F2b/F4b status-log entry (pod, numbers, rulings); the evidence report (serve section: dp 2; two-GPU section; performance section: the Wan bench table, Flux timings, the Server data point; NOT MEASURED updated; ledger lines; pod cost table); the skill's catalog (serving on two GPUs cell, Wan `--dp` cell, bench command) and SKILL.md (≤ 200 lines); `DEVELOPER.md`'s Wan/serve paragraphs; fold the ledger; CPU gate; commit `docs: F2b/F4b status — serving on two GPUs, the Wan DiT bench, Flux timings`.

## Rulings

- R1: replicas are independent resident engines (one process each, dp=1 inside), not the TPU lockstep replica path — a dead replica does not hang the others; the wrapper is additive and `dp_degree=1` is today's path.
- R2: dispatch is least-pending; admission limits stay per replica (`max_running_requests=1`, `max_queued_requests` each), so the server's capacity is N × today's.
- R3: measurement before fixing — the pod session measures everything first; fixes are gated on the table (F4's rule) and on the pod budget.
- R4: inline execution (executing-plans) with one fresh whole-branch reviewer.
