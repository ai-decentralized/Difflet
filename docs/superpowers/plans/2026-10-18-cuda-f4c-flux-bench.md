# CUDA F4c — The Flux DiT bench and the fix it names Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** F5c measured Flux's 28-step generate at 9.4 s against diffusers' 8.3 s on the Server Edition (13 %), the largest eager gap left. Locate it the way F4b located Wan's: one DiT forward on identical inputs, Difflet's `CudaFluxTransformerApplication` vs diffusers' `FluxTransformer2DModel`, timed and profiled; then, if the buckets name a contained cause, fix it, re-bench and re-run the Flux gates on the same pod.

**Architecture:** `scripts/cuda_bench_flux_dit.py` (the F4b recipe, `_measure` shared with `cuda_bench_wan_dit.py`): seeded packed latents `(1, image_seq_len, in_channels)`, text `(1, 512, joint_attention_dim)`, pooled `(1, pooled_projection_dim)`, timestep `(1,)` at 0.5 (the pipeline's `t / 1000`, both models scale by 1000 inside), guidance `(1,)` at 3.5, in the model dtype; `txt_ids` zeros `(512, 3)` and `img_ids` from the pipeline's `_prepare_latent_image_ids` over the `H/16 × W/16` grid — diffusers takes the ids, Difflet the adapter's `rotary_embedding(txt_ids, img_ids)` built from the same ids. Warm-up, median of N with CUDA events, one profiled forward bucketed by `difflet.backends.cuda.bench`. JSON `bench_flux_dit.json` with `cases.{difflet,diffusers}`.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — the F4 entry (the bench recipe), the F2b/F4b entry (the Wan bench and the fused norm as the template), the F5c entry (the Flux timings); D4 (eager; compile is a measured option), D7 (SDPA).

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task; black 100; one task, one commit. Branch `feat/cuda-f4c-flux-bench` on `feat/cuda-f5c-flux-cells` (dfe0020, PR #96).
- The pod (1 × RTX PRO 6000, eu, no volume, terminated at the end) is created only after Task 1 is green and committed, and only on the owner's yes; FLUX.1-dev is gated: the owner's device login, issued only once they say they are looking at the screen.
- A fix lands only if it is contained (one op or one modeling path), bit-identical or rounding-level, CPU-proven, and re-gated on the pod (parity + e2e); otherwise the measurement is recorded and the design is the follow-up.

## Review Focus

1. Identical inputs on both sides: the same tensor objects for latents, text, pooled, timestep and guidance; the rope on Difflet's side derived from the same `txt_ids`/`img_ids` diffusers receives (Task 1 `test_bench_inputs_match_the_dit_contract`, `test_image_ids_are_the_pipeline_s`).
2. The bucket arithmetic and the verdict wording: what the profile names is a measurement; the cause is a code read, labelled as such (Task 3).

---

### Task 1: The Flux DiT bench

**Files:**
- Create: `scripts/cuda_bench_flux_dit.py`
- Test: `tests/unit/scripts/test_cuda_bench_flux_dit.py` (parser defaults; `bench_inputs` shapes/dtypes from a FLUX.1-dev-shaped `FluxDiTConfig`; `latent_image_ids` equals diffusers' `_prepare_latent_image_ids`)

- [ ] **Step 1:** failing tests → **Step 2:** run → **Step 3:** implement; `--help` on the CPU host → **Step 4:** `tests/unit/scripts` + CPU gate → **Step 5: Commit** `perf: Flux DiT bench vs diffusers on identical inputs (F4c task 1)`.

### Task 2: The pod session

- [ ] create (1 GPU, eu-3/eu-2, 250 GB), ship, setup, the owner's device login, FLUX.1-dev download; `cuda_bench_flux_dit.py` (both cases); read the buckets; artifacts to `artifacts/cuda-release1/flux/f4c/`.

### Task 3: The fix the buckets name (ruling at the time), docs

- [ ] If contained: implement on the Mac with a bit-identity / rounding test, CPU gate, ship, re-bench, `run_gates.sh flux` (parity + e2e); else record. Terminate the pod. Status-log entry, evidence report (Performance section, ledger, pod cost, NOT MEASURED), catalog; fold the ledger; commit.

## Rulings

- R1: inline execution, one fresh whole-branch reviewer.
- R2: the bench is one forward of the whole DiT (19 double + 38 single blocks); per-block attribution comes from the kernel buckets, not from timing blocks separately.
