# CUDA F1b — fp32 timestep for Wan on CUDA, measured Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop rounding the diffusion timestep to bf16 on the CUDA Wan path (the F1 e2e gap's first suspect: the shared orchestrator casts UniPC's int64 timestep to the model dtype, so 999 → 1000 and 750 → 752, while diffusers' `WanPipeline` passes the exact integer into a sinusoidal projection that computes in fp32), make the parity oracle able to see that rounding, and measure the e2e gain against the recorded 27.5 dB / 0.9970 — without changing Trainium or TPU behaviour.

**Architecture:** The timestep's dtype becomes a per-component property the orchestrator reads, like `dtype` today: `_component_timestep_dtype(component, fallback=model_dtype)` reads an optional `timestep_dtype` attribute; `_batch_timestep` is called with it at both call sites in `_denoise` (the CFG-parallel batch and the per-step batch). Components without the attribute (Trainium stage apps, TPU `_OnDeviceTransformer`) keep the model dtype, byte for byte. `_CudaDiT` declares `timestep_dtype = torch.float32` by default and moves the timestep to the device in *that* dtype; `CudaWanApplication(timestep_dtype="float32"|"model")` lets the oracles A/B the old path. Difflet's Wan modeling already embeds the timestep through diffusers' `get_timestep_embedding`, which multiplies `timesteps.float()` by an fp32 frequency table, so an fp32 input is exact and the projected embedding is still cast to the embedder's bf16 afterwards.

**Tech Stack:** torch 2.11.0+cu128 eager, bf16 weights, diffusers 0.38.0, pytest, black 100.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — follow-up F1b (status-log entries "F1 verdict" and "F1 review fixes", 2026-10-18), decisions D4, D9, D14; the F1 plan `docs/superpowers/plans/2026-10-18-cuda-f1-wan.md` (ledger: Task 3 minor "bf16 timestep", Task 5 rulings).

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every CPU-host command; every task ends with `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` green (regressions=0, new_skips=0) and touched files `black -l 100` clean (legacy non-clean files exempt). One task, one commit. Branch `feat/cuda-wan-f1b-timestep` on top of `feat/cuda-wan-f1` (544f034).
- Trainium and TPU behaviour unchanged: no component outside the CUDA Wan app gains `timestep_dtype`; `tests/unit/models/wan/test_wan_pipeline*.py`, `tests/unit/backends/test_tpu_*.py`, `tests/unit/cli/test_cli_cfg_parallel.py` stay green with no edits to their assertions.
- The import hygiene of `difflet/models/wan/{pipeline,cuda_application}.py` and `difflet/backends/cuda/**` holds (no trainium/tpu/neuron/xla imports).
- Parity gates (D14) unchanged: rel_l1 ≤ 1.15× the diffusers-bf16 control's, same-dtype ≤ 1.2×, cosine ≥ 0.995. The e2e floors (25 dB / 0.99) move only from a measurement (Task 3).
- Pod rule: pod `difflet-cuda-f5b` (ID 502399632095187214) is PAUSED; resuming it needs the owner's yes (`POST https://api.yottalabs.ai/v2/pods/<id>/resume`, key `~/.config/yottalabs/api_key`); the Wan repo (118 GB) goes onto the `difflet-cache` volume after `hf cache rm model/Qwen/Qwen-Image -y` (never the container disk); logs under `~/logs/f1b`; pause through the API at the end and verify `PAUSED`.

## Review Focus

1. The orchestrator passes an fp32 timestep to a component whose `timestep_dtype` is `torch.float32` and the model-dtype timestep to one without the attribute — in the per-step path and in the CFG-parallel path — Task 1 `test_denoise_passes_the_component_timestep_dtype` (fake component records the dtype it received).
2. `_CudaDiT` moves the timestep in its `timestep_dtype`, not the model dtype; `CudaWanApplication(timestep_dtype="model")` restores the bf16 cast; the default is fp32 — Task 1 `test_cuda_dit_passes_fp32_timestep_by_default`, `test_timestep_dtype_model_restores_the_bf16_cast` (spy on the expert's forward records the timestep dtype and value: 999 stays 999.0 in fp32, becomes 1000.0 under "model").
3. The parity oracle's rounding block gives diffusers the exact integer timestep (as the pipeline does) and Difflet both variants — Task 2 pure-part tests on the JSON shape and the gate.
4. The e2e script's variants share one diffusers run and one noise tensor — Task 2 `test_e2e_variants_share_the_reference`.

---

### Task 1: Timestep dtype as a component property; fp32 on the CUDA Wan DiT

**Files:**
- Modify: `difflet/models/wan/pipeline.py` (`_component_timestep_dtype`, both `_batch_timestep` call sites), `difflet/models/wan/cuda_application.py` (`_CudaDiT.timestep_dtype`, `__call__`; `timestep_dtype` kwarg validated to `"float32"` | `"model"`, passed to both experts' `_CudaDiT`)
- Test: `tests/unit/models/wan/test_wan_pipeline_orchestrator.py` (add), `tests/unit/models/wan/test_wan_pipeline.py` (`_component_timestep_dtype` unit), `tests/unit/models/wan/test_wan_cuda_application.py` (add)

**Interfaces:**
- `difflet.models.wan.pipeline._component_timestep_dtype(component, fallback: torch.dtype) -> torch.dtype` — `getattr(component, "timestep_dtype", None)` normalised like `_component_dtype`, else `fallback`.
- `_CudaDiT(app, dtype, device, timestep_dtype=torch.float32)`; attribute `timestep_dtype`; `__call__` uses `timestep.to(self.device, self.timestep_dtype)`.
- `CudaWanApplication(..., timestep_dtype="float32")`; `"model"` → `self.dtype`; anything else → `ValueError`.

- [x] **Step 1: Write the failing tests** (Review Focus 1–2 plus `test_component_timestep_dtype_defaults_to_fallback`). **Step 2: Run** → failures. **Step 3: Implement.** **Step 4:** `pytest tests/unit/models/wan tests/unit/cli/test_cli_cfg_parallel.py tests/unit/backends/test_cuda_import_hygiene.py -q` + CPU gate green. **Step 5: Commit** `wan: timestep dtype is a component property — fp32 on the CUDA DiT, model dtype elsewhere (F1b task 1)`.

### Task 2: Oracles that can see timestep rounding

**Files:**
- Modify: `tests/numerical/cuda_parity_wan_gpu.py` (a `rounding` block per expert at `ROUNDING_TIMESTEP = {"transformer": 999, "transformer_2": 750}`: upstream fp32 reference and bf16 control fed the **int64** timestep exactly as `WanPipeline` does; Difflet fed `float32` (production) and `bfloat16` (the old cast) — JSON `experts.<e>.rounding = {"timestep", "control", "fp32_timestep": {…, "vs_bf16"}, "bf16_timestep": {…, "vs_bf16"}}`; the existing 900/500 `results.sdpa` block stays and now runs with the fp32 timestep too), `tests/numerical/test_cuda_wan_vs_diffusers.py` (gate: the `rounding.fp32_timestep` block passes the three D14 gates; the `bf16_timestep` ratio is recorded in the test's output, not gated), `scripts/cuda_e2e_wan_vs_diffusers.py` (`--timestep-dtypes float32,model`: one diffusers run and one noise tensor, one Difflet run per variant (reload is acceptable), JSON `variants.<name>.{metrics, timings}` with `metrics` = the first variant's for the gate's existing keys; first-frame PNG per variant), `tests/numerical/test_cuda_wan_e2e_vs_diffusers.py` (reads `metrics` as today; prints every variant's PSNR), `tests/unit/scripts/test_cuda_wan_gates.py` and the e2e pure-part tests (JSON shapes, argument parsing, the variant bookkeeping).

- [x] **Step 1: Write the failing tests.** **Step 2: Run.** **Step 3: Implement.** **Step 4:** `pytest tests/unit/scripts tests/numerical -q` (CPU parts) + CPU gate green; both scripts `--help` on the CPU host. **Step 5: Commit** `wan: parity oracle sees bf16 timestep rounding; e2e A/B of the timestep dtype (F1b task 2)`.

### Task 3: GPU measurement, docs, status log

- [x] **Step 1:** owner's yes → resume the pod, `hf cache rm model/Qwen/Qwen-Image -y`, `HF_HOME=/data/hf hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers`, ship the branch as a bundle.
- [x] **Step 2:** run `cuda_parity_wan_gpu.py --expert both` + gate, then `cuda_e2e_wan_vs_diffusers.py --timestep-dtypes float32,model --steps 40 --num-frames 9` + gate (logs under `~/logs/f1b`); record every number.
- [x] **Step 3:** if the fp32 variant beats 27.5 dB by a clear margin, raise the e2e floors to ~5 dB / 0.002 below the new measurement; otherwise leave them and record the hypothesis as refuted. Append the status-log entry (append-only), update the DEVELOPER.md Wan paragraph, copy artifacts to `artifacts/cuda-release1/wan/f1b/`, fold the ledger, CPU gate, commit `docs: F1b status — fp32 timestep on the CUDA Wan path, measured`. Remove Wan from the volume, re-download Qwen, pause the pod through the API, verify `PAUSED`.

## Rulings

- R1: the attribute is read off the component, not configured on the orchestrator, so Trainium/TPU stay untouched by construction.
- R2: the default on CUDA is fp32 because it is what diffusers does; `"model"` exists only for the A/B and the record.

## Execution ledger (folded from the SDD workspace, 2026-10-18)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-18-cuda-f1b-timestep.md

Branch feat/cuda-wan-f1b-timestep on feat/cuda-wan-f1 (544f034). Pre-flight scan: T1 produces `_component_timestep_dtype` + `_CudaDiT.timestep_dtype` + the `timestep_dtype` kwarg; T2 consumes the kwarg in the e2e script and the attribute in nothing else — consistent. T1 self: the two `_batch_timestep` call sites and the TeaCache probe/shadow receive the same `timestep_batch`; on CUDA Wan TeaCache is refused, so no behaviour change there — consistent. T2 self: the gate reads `experts.<e>.rounding.fp32_timestep` and the e2e `metrics` key — both written by the scripts it modifies. Scan clean.

## Tasks
Task 1: minor (deferred): the CUDA default-path test does not assert latents/text still arrive in the model dtype (the orchestrator test does); `timestep_dtype` stays in `self.kwargs`.
Task 1: Ruling: the existing `test_dit_callable_moves_to_the_model_dtype_and_returns_host_tensors` keeps its assertion and now builds `_CudaDiT(timestep_dtype=bf16)` (the "model" path) — the fp32 default has its own test; costs nothing if wrong.
Task 1: complete (commits 7c0dae4..eca6f9d, review clean)
Task 2: minor (deferred): e2e gate picks the first variant by dict order instead of `timestep_dtypes[0]`; a missing `fp32_timestep`/`bf16_timestep` key raises a bare KeyError; second import blocks with noqa in two test files; the old `wan_e2e_difflet_frame0.png` name is gone (docs pass in Task 3).
Task 2: complete (commits eca6f9d..38bff3c, review clean)
Task 3: Ruling: resuming pod difflet-cuda-f5b for the measurement is covered by the owner's "I am ok with the rest" (F1b was described to them as needing the pod); paused again through the API when the run ends.
Task 3: finding: `POST /v2/pods/{id}/resume` returns 200 but the pod goes INITIALIZING → PAUSED within ~10 s (three attempts, 13:31–13:40 UTC); `GET /v2/gpus` shows NVIDIA_RTX_PRO_6000_96G us-east-3 ($0.45/h) status UNAVAILABLE — no capacity to resume into. Other PRO 6000 pools: eu-2 / eu-3 AVAILABLE at $2.65/h (NVMe, container disk up to 500 GB), eu-1 UNAVAILABLE; the difflet-cache volume (id 469225541779435520, mounted at /data via `persistentVolumes`) is us-east-3 only. Also: the pod record says `containerVolumeInGb: 100` — the 118 GB write to /tmp in F1 exceeded the container disk, which is the restart's cause. Waiting on the owner: retry later vs an eu pod without the volume.
Task 3: owner's words "Let's find another node to continue this" → `POST /v2/pods` in eu-3/eu-2: every 1-GPU request (with/without expose, 150–200 GB disk) returned code 12014 "Pod insufficient resources"; a 2-GPU request in eu-3 succeeded → pod `difflet-cuda-f1b`, ID 502601836865405322, 2× RTX PRO 6000, $5.30/h, 150 GB container disk, no volume (the eu pools hand out whole 2-GPU nodes). Created 2026-10-18 ~13:58 UTC. Terminate (DELETE) when the run ends — nothing to keep on it.
Task 3: GPU run on difflet-cuda-f1b (eu-3, 2× RTX PRO 6000 Blackwell **Server Edition**, driver 580.178.04, GPU 0 only; setup 75 s, Wan download 400 s to the 968 GB local disk, loads 5–23 s from page cache). Parity (both experts, F1 block at 900/500 reproduced bit-for-bit: 1.051× / 1.043×): rounding @ 999 — control rel_l1 5.068e-2; **fp32 timestep 1.039× control (same-dtype 0.804×, cosine vs bf16 0.999124)** vs bf16-cast timestep 1.084× (0.873×, 0.998985); rounding @ 750 — fp32 1.018× (0.803×) vs bf16 1.020× (0.813×); gate 11 passed. e2e A/B (one diffusers run, one noise): **timestep_dtype=float32 PSNR 26.30 dB / cosine 0.99609 / mean_abs 0.0274; timestep_dtype=model 27.54 dB / 0.99704 / 0.0238** — identical to the F1 number on the Workstation card (deterministic across cards); Difflet generate 50.9 s / 50.2 s, diffusers 42.9 s on the Server Edition; gate 3 passed.
Task 3: Ruling: the hypothesis is refuted at the pipeline level — exact timesteps make each DiT call closer to diffusers but the 40-step trajectory ends 1.2 dB further away, so the e2e gap vs Qwen/Flux comes from elsewhere; next suspect is the CFG combine (diffusers combines `uncond + s·(cond − uncond)` in bf16 on the device, Difflet in fp32 on the host) and the host-side scheduler step — measurable with the same A/B knob (F1b item 2). The fp32 default stays (it is what the schedule intends and what diffusers feeds its DiT; DiT parity improves at every non-representable step); `timestep_dtype="model"` remains for the record. Floors stay at 25 dB / 0.99 (fp32 variant 1.3 dB above the floor — thin, noted). Costs: if the owner prefers the higher e2e number, one default flip.
Task 3: complete (GPU gates parity 11 passed / e2e 3 passed on difflet-cuda-f1b; artifacts under artifacts/cuda-release1/wan/f1b; pod terminated; docs commit follows)
```
