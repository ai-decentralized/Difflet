# CUDA F3c — `difflet serve` for FLUX.1-dev and Wan 2.2 on CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On CUDA, `difflet serve` serves Qwen-Image only (F3b); FLUX.1-dev and Wan 2.2 are refused by name in `difflet.serving.model_registry` ("follow-ups F1b and F5b"), although both generate on CUDA through `difflet run` and pass every gate. Serve them the way F3b serves Qwen-Image: one resident worker per GPU holding `DiffletPipeline.from_pretrained(backend="cuda")`, one opaque `pipeline` stage calling `app.generate()` per request, the HTTP layer and the engines untouched. Flux answers `/v1/chat/completions` with a PNG; Wan answers `/v1/videos` and `/v1/videos/sync` with an MP4 written by the serving layer's `encode_video_tensor`. GPU verification (one pod session) waits for the owner's yes.

**Architecture:** *(1)* The model-independent half of F3b's adapter — the resident one-GPU `RuntimePlan` (slot 0, one `cuda` stage, a profile identity) and the refusals every CUDA profile shares (`--shapes`, `--tp-degree > 1`, `--sp`, `--cp-degree > 1`) — moves to `difflet/serving/orchestrators/cuda_resident.py`; `qwen_image_cuda.py` keeps its names and behaviour on top of it (its tests are the net). *(2)* `difflet/serving/orchestrators/flux_cuda.py`: preparer (refuse, resolve the snapshot, eager bundle), validator (the profile shape, guidance in [0, 20], the T5 prompt within 512 tokens — the Trainium Flux validator's rules), adapter (load with the CUDA CLI's kwargs, `generate(prompt, num_inference_steps, guidance_scale, seed, output_type="pil")` → PNG; a 4-step startup smoke). *(3)* `difflet/serving/orchestrators/wan_cuda.py`: the same for video — the validator takes the profile's (height, width, num_frames) and fps, guidance and `guidance_scale_2` in [0, 20] (Wan 2.2's two experts), a `negative_prompt`, both prompts within UMT5's 512 tokens, and refuses `boundary_ratio`, `flow_shift`, `true_cfg_scale`; the adapter calls `generate(..., num_frames=profile, output_type="pt")` and encodes the `(B, 3, F, H, W)` frames in [-1, 1] to the request's MP4 target. *(4)* `model_registry`: `_CUDA_SERVING_OVERLAYS` gains `flux` and `wan`; the checkpoint is checked against the CUDA metadata (FLUX.1-dev; Wan 2.2 — Wan 2.1 never ran on a GPU and stays refused by name); `--dp-degree > 1` is refused for video on CUDA, since the video FIFO runs one job at a time and a second replica would sit idle. *(5)* `scripts/cuda_serve_smoke.py --model-id` for Flux (chat) and Wan (`/v1/videos/sync`).

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — D5 (one process per GPU), the F3b entry (the CUDA serving shape, its refusals and probes), F2b (`--dp-degree`), F1b / F5b (serve refused for Wan / Flux), F4f (`DIFFLET_CUDA_TORCH_COMPILE=1` reaches the applications through the environment).

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task; black 100; one task, one commit. Branch `feat/cuda-f3c-serve-flux-wan` on `feat/cuda-f9c-flux-seeds` (a28d1ab, PR #101).
- The Trainium / TPU serving paths and their metadata are unchanged; Qwen-Image's CUDA serving behaves exactly as before.
- Refusals name what owns them (a follow-up, or why it does not apply); nothing is dropped silently.

## Review Focus

1. The worker loads each model with exactly the CLI's `from_pretrained` kwargs (except the pinned revision), and a request reaches `generate` with exactly the CLI's arguments.
2. Every request field the CUDA app cannot honour is refused in the server process, before the worker.
3. Wan's MP4 is the frames tensor's own layout and range (`BCTHW`, [-1, 1]) at the profile fps.

---

### Task 1: The shared CUDA resident plan and refusals

**Files:** Create `difflet/serving/orchestrators/cuda_resident.py`; Modify `difflet/serving/orchestrators/qwen_image_cuda.py`; Test `tests/unit/serving/test_cuda_resident.py` (+ the Qwen suite unchanged).

- [x] Step 1: failing tests (the plan's slot, stage and identity, `num_frames` in the identity when given, the shared refusals naming F3 / F6 / F2b). Expected: FAIL (no module).
- [x] Step 2: implement; Qwen's `cuda_runtime_plan` / `refuse_unsupported_profile` call the shared ones. Expected: the new and the Qwen tests pass.
- [x] Step 3: CPU gate. **Step 4: Commit** `serving: the CUDA resident plan and shared refusals out of the Qwen adapter (F3c task 1)`.

### Task 2: FLUX.1-dev served on CUDA

**Files:** Create `difflet/serving/orchestrators/flux_cuda.py`; Modify `difflet/common/registry/flux.py` (`cuda_serving_metadata`); Test `tests/unit/serving/test_flux_cuda_serving.py`.

- [x] Step 1: failing tests (preparer bundle and refusals before the download — shapes, tp, sp, cp, cfg-parallel N/A, TeaCache F5b; validator — shape, guidance range, T5 512; adapter — CLI kwargs, the pinned revision, the 4-step smoke at 3.5, generate → PNG, cancellation, shutdown). Expected: FAIL.
- [x] Step 2: implement. **Step 3:** CPU gate. **Step 4: Commit** `serving: FLUX.1-dev on CUDA, one resident pipeline stage (F3c task 2)`.

### Task 3: Wan 2.2 served on CUDA

**Files:** Create `difflet/serving/orchestrators/wan_cuda.py`; Modify `difflet/common/registry/wan.py` (`cuda_serving_metadata`); Test `tests/unit/serving/test_wan_cuda_serving.py`.

- [x] Step 1: failing tests (preparer and refusals; validator — mp4 only, the profile shape and fps, guidance and guidance_scale_2 ranges, negative_prompt, UMT5 512 for both prompts, boundary_ratio / flow_shift / true_cfg_scale refused; adapter — CLI kwargs incl. num_frames, the smoke writes a valid MP4 at the profile shape, generate passes guidance_scale_2 and negative_prompt only when given, the MP4 is the frames tensor's). Expected: FAIL.
- [x] Step 2: implement. **Step 3:** CPU gate. **Step 4: Commit** `serving: Wan 2.2 on CUDA, MP4 from one resident pipeline stage (F3c task 3)`.

### Task 4: Registry wiring and the CUDA-only refusals

**Files:** Modify `difflet/serving/model_registry.py`; Test `tests/unit/serving/test_model_registry_cuda.py`.

- [x] Step 1: failing tests (Flux and Wan 2.2 route to the CUDA factories with the CUDA defaults; Wan 2.1 refused naming why; `--dp-degree 2` refused for Wan on CUDA, allowed for Flux; other backends keep the Trainium metadata). Expected: FAIL.
- [x] Step 2: implement. **Step 3:** CPU gate. **Step 4: Commit** `serving: route FLUX.1-dev and Wan 2.2 to their CUDA adapters (F3c task 4)`.

### Task 5: The serve smoke for Flux and Wan

**Files:** Modify `scripts/cuda_serve_smoke.py` (`--model-id`; Wan through `/v1/videos/sync`), `.claude/skills/difflet-cuda-verify/references/feature-catalog.md`; Test `tests/unit/scripts/test_cuda_serve_smoke_script.py` (or the existing one).

- [x] Step 1: failing tests (the request bodies per model; Wan's form fields; output naming). **Step 2:** implement. **Step 3:** CPU gate. **Step 4: Commit** `scripts: cuda_serve_smoke.py serves Flux and Wan (F3c task 5)`.

### Task 6: Docs, status log, PR; the pod run (pending the owner's yes)

- [x] The evidence report's NOT MEASURED / support matrix serve rows stay as measured until the pod; `DEVELOPER.md`, the catalog; status-log entry "landed on the CPU"; push; PR stacked on #101; one fresh whole-branch reviewer. Pod (owner's yes, us-east-3 on `difflet-cache` — Flux on the volume; Wan needs its 126 GB download unless the volume is grown): `cuda_serve_smoke.py --model-id black-forest-labs/FLUX.1-dev --requests 2`, then Wan the same, plus the 400 probes.

## Rulings

- R1: Wan 2.1 is not served on CUDA (it never ran on a GPU; NOT MEASURED) — refused by name, not silently accepted.
- R2: `--dp-degree > 1` is refused for Wan on CUDA: `VideoGenerationService` dispatches one job at a time, so a second replica would only hold a GPU. Flux keeps F2b's replicas (the chat path admits concurrent requests).
- R3: `--torch-compile` is not a serve flag; `DIFFLET_CUDA_TORCH_COMPILE=1` reaches the applications through the environment, as for Qwen (F4f).
- R4: inline execution, one fresh whole-branch reviewer.

## Execution ledger

```
Task 1: complete (4b3f500; CPU gate 2888 passed). Qwen's suite unchanged and green on the shared module.
Task 2: complete (05c9d97; CPU gate 2910 passed). The serve generate kwargs equal FluxCudaOrchestrator.generate_kwargs (asserted).
Task 3: complete (532abf8; CPU gate 2939 passed). Real MP4 encode/decode in the tests (imageio-ffmpeg, PyAV on the Mac). The boundary refusal cites the checkpoint's model_index.json, not a number.
Task 4: complete (c019245; CPU gate 2946 passed). Added an in-process /v1/videos/sync test through the CUDA validator and adapter (not in the plan): video is new on the CUDA path; Flux uses Qwen's GPU-verified chat path.
Task 5: complete (4568b34; CPU gate 2952 passed). serve_timings.json keeps "image" for images, adds "video" and "bytes".
Task 6: docs + status log (this commit); push, PR, review; the pod run pending the owner's yes (Wan needs disk: the volume has 115 GiB free, Wan is 117.5 GiB).
Review fix wave: f7c167e (smoke --worker-restart-timeout, clean CUDA refusals, lone --teacache-calibration refused, video-replica reason first).
Pod: f9c3 resume refused (no capacity) -> eu-2 pod f3c (Server, $2.68/h); HF login + Flux download to its disk (Xet stalled; HTTPS); Flux serving PASS: served PNGs byte-identical to difflet run, probes and refusals as designed; terminated 19:30 UTC. Wan serving still NOT MEASURED (needs disk).
```
