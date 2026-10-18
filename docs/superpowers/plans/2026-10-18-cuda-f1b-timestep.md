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

- [ ] **Step 1: Write the failing tests** (Review Focus 1–2 plus `test_component_timestep_dtype_defaults_to_fallback`). **Step 2: Run** → failures. **Step 3: Implement.** **Step 4:** `pytest tests/unit/models/wan tests/unit/cli/test_cli_cfg_parallel.py tests/unit/backends/test_cuda_import_hygiene.py -q` + CPU gate green. **Step 5: Commit** `wan: timestep dtype is a component property — fp32 on the CUDA DiT, model dtype elsewhere (F1b task 1)`.

### Task 2: Oracles that can see timestep rounding

**Files:**
- Modify: `tests/numerical/cuda_parity_wan_gpu.py` (a `rounding` block per expert at `ROUNDING_TIMESTEP = {"transformer": 999, "transformer_2": 750}`: upstream fp32 reference and bf16 control fed the **int64** timestep exactly as `WanPipeline` does; Difflet fed `float32` (production) and `bfloat16` (the old cast) — JSON `experts.<e>.rounding = {"timestep", "control", "fp32_timestep": {…, "vs_bf16"}, "bf16_timestep": {…, "vs_bf16"}}`; the existing 900/500 `results.sdpa` block stays and now runs with the fp32 timestep too), `tests/numerical/test_cuda_wan_vs_diffusers.py` (gate: the `rounding.fp32_timestep` block passes the three D14 gates; the `bf16_timestep` ratio is recorded in the test's output, not gated), `scripts/cuda_e2e_wan_vs_diffusers.py` (`--timestep-dtypes float32,model`: one diffusers run and one noise tensor, one Difflet run per variant (reload is acceptable), JSON `variants.<name>.{metrics, timings}` with `metrics` = the first variant's for the gate's existing keys; first-frame PNG per variant), `tests/numerical/test_cuda_wan_e2e_vs_diffusers.py` (reads `metrics` as today; prints every variant's PSNR), `tests/unit/scripts/test_cuda_wan_gates.py` and the e2e pure-part tests (JSON shapes, argument parsing, the variant bookkeeping).

- [ ] **Step 1: Write the failing tests.** **Step 2: Run.** **Step 3: Implement.** **Step 4:** `pytest tests/unit/scripts tests/numerical -q` (CPU parts) + CPU gate green; both scripts `--help` on the CPU host. **Step 5: Commit** `wan: parity oracle sees bf16 timestep rounding; e2e A/B of the timestep dtype (F1b task 2)`.

### Task 3: GPU measurement, docs, status log

- [ ] **Step 1:** owner's yes → resume the pod, `hf cache rm model/Qwen/Qwen-Image -y`, `HF_HOME=/data/hf hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers`, ship the branch as a bundle.
- [ ] **Step 2:** run `cuda_parity_wan_gpu.py --expert both` + gate, then `cuda_e2e_wan_vs_diffusers.py --timestep-dtypes float32,model --steps 40 --num-frames 9` + gate (logs under `~/logs/f1b`); record every number.
- [ ] **Step 3:** if the fp32 variant beats 27.5 dB by a clear margin, raise the e2e floors to ~5 dB / 0.002 below the new measurement; otherwise leave them and record the hypothesis as refuted. Append the status-log entry (append-only), update the DEVELOPER.md Wan paragraph, copy artifacts to `artifacts/cuda-release1/wan/f1b/`, fold the ledger, CPU gate, commit `docs: F1b status — fp32 timestep on the CUDA Wan path, measured`. Remove Wan from the volume, re-download Qwen, pause the pod through the API, verify `PAUSED`.

## Rulings

- R1: the attribute is read off the component, not configured on the orchestrator, so Trainium/TPU stay untouched by construction.
- R2: the default on CUDA is fp32 because it is what diffusers does; `"model"` exists only for the A/B and the record.

## Execution ledger

(appended by the executor, one line per task: numbers, rulings, deviations)
