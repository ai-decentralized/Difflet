# CUDA F9c — the Flux e2e over seeds, its noise floor, and the gate over seeds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** F4f measured the Flux e2e noise floor (diffusers vs itself with bf16 gemm reductions flipped) at 30.92 / 36.62 / 37.61 dB over seeds 0–2, in three separate single-seed runs, and restated the Flux e2e gate relative to that floor — one seed at a time. A 6.7 dB spread over three seeds says one seed is not a measurement. Do for Flux what F9 / F9b did for Wan: one script run over many seeds (one load per side), per-seed and aggregate metrics in one JSON, and the gate over seeds — gate A on the mean, gate B on every seed against the floor's worst, the loose floor and "the oracle moved" as before. Then `run_gates.sh flux` runs the e2e with the floor over three seeds (Flux generates in ~8 s, so the floor costs about a minute), so the gate runner judges a correct build green instead of failing seed 0's 35 dB floor. The pod run (more seeds, eager and `--torch-compile`) waits for the owner's yes.

**Architecture:** *(1)* The seed helpers F9 wrote for Wan (`parse_seeds`, `run_seeds`, `spread`, `aggregate_metrics`) move to `scripts/cuda_e2e_vs_diffusers.py` (the shared metrics module, as `image_metrics` / `_save` already are); the Wan script imports them. `cuda_e2e_flux_vs_diffusers.py` gains `--seeds` (overrides `--seed`): `run_difflet` loads once and per seed draws the packed noise from `torch.Generator(device).manual_seed(seed)` (as today) and generates; `run_diffusers` loads once and per seed runs exact and, with `--noise-floor`, the flag-flipped run (the flag restored in `finally` every seed). The JSON adds `seeds`, `per_seed.<seed>.{metrics, timings.{difflet, diffusers}, noise_floor, images}` and `aggregate.{difflet, noise_floor}.{psnr_db, cosine}.{min, mean, max}`; `seed`, `metrics`, `timings`, `images`, `noise_floor` stay the first seed's (pre-F9c readers, the committed artifacts), and the first seed's PNGs are also written under the pre-F9c names. *(2)* `tests/numerical/test_cuda_flux_e2e_vs_diffusers.py` reads per-seed entries (`per_seed`, else the one top-level seed): the absolute test judges every seed (35 dB / 0.998 without a floor, the loose 20 dB / 0.98 with one); the relative test runs F9b's gates A and B over two or more seeds and keeps F4f's single-seed statement for one; a new record test checks the first-seed top-level keys and the aggregate against the per-seed numbers and prints them. *(3)* `run_gates.sh flux`: the e2e step runs `--seeds 0,1,2 --noise-floor`.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — D14 (relative gates), the F9 / F9b entries (the Wan multi-seed shape and its gates A/B/C), the F4f entry (the Flux noise floor, ruling R-gate; next: "the Flux noise floor over more seeds and a `--seeds` run for the Flux script").

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task; black 100; one task, one commit. Branch `feat/cuda-f9c-flux-seeds` on `feat/cuda-f4f-compile-wan-flux` (a135366, PR #100).
- Back-compat: a pre-F9c Flux JSON (no `per_seed`) is gated exactly as F4f gates it; the committed artifacts keep their verdicts (the Mac harness re-gates them).
- Nothing in `difflet/` changes; Wan's script behaviour and JSON are unchanged (its tests are the net).

## Review Focus

1. One load per side over seeds, the same noise on both sides per seed, the flag restored after every perturbed run (fakes in `tests/unit/scripts/test_cuda_e2e_flux_script.py`).
2. The gate over seeds: gate A on the mean, gate B on the floor's worst seed, a floor on some seeds only is malformed, the single-seed statement unchanged, every committed artifact's verdict unchanged.
3. The pre-F9c keys are the first seed's; the aggregate matches the per-seed numbers.

---

### Task 1: `--seeds` for the Flux e2e script

**Files:**
- Modify: `scripts/cuda_e2e_vs_diffusers.py` (the seed helpers), `scripts/cuda_e2e_wan_vs_diffusers.py` (import them), `scripts/cuda_e2e_flux_vs_diffusers.py` (`--seeds`, per-seed runs, the payload)
- Test: `tests/unit/scripts/test_cuda_e2e_flux_script.py`

- [x] **Step 1:** failing tests — the helpers are one definition (`flux_e2e.parse_seeds is wan_e2e.parse_seeds`); `--seeds` parses and overrides `--seed`; `run_difflet` loads once and generates each seed from its seeded draw; `run_diffusers` loads once, runs exact (+ perturbed) per seed, restores the flag; `main` over seeds 0,1,2 with `--noise-floor` writes `per_seed`, `aggregate`, first-seed top-level keys, per-seed PNGs and the legacy names; one seed without the floor is the F4f run plus `per_seed` / `aggregate`.
  Expected: FAIL (`AttributeError: parse_seeds`, `TypeError` on the new signatures).
- [x] **Step 2:** implement. Expected: the file's tests pass; Wan's script tests unchanged and green.
- [x] **Step 3:** CPU gate. Expected: `regressions=0`.
- [x] **Step 4: Commit** `flux e2e: --seeds — one load per side, per-seed and aggregate metrics, the noise floor per seed (F9c task 1)`.

### Task 2: The Flux e2e gate over seeds

**Files:**
- Modify: `tests/numerical/test_cuda_flux_e2e_vs_diffusers.py`
- Test: `tests/unit/scripts/test_cuda_flux_gates.py`

- [x] **Step 1:** failing harness tests — a three-seed payload with a floor: gate A / gate B thresholds printed and judged (the F4f floor seeds 0–2 and Difflet's fused numbers pass: mean 38.30 dB ≥ 35.05 − 1.21, every seed ≥ 30.92 − 1.21); a mean short of gate A fails as "mean PSNR"; one seed under gate B fails naming it; the cosine halves; a floor on some seeds only fails; without a floor the 35 dB floors judge every seed and name the failing one; the first-seed top-level keys and the aggregate are checked; the F4f single-seed tests and the committed-artifact verdicts unchanged; the three committed F4f floor JSONs merged into one three-seed result pass.
  Expected: FAIL (the gate reads the top-level seed only; no record test).
- [x] **Step 2:** implement. Expected: the harness passes.
- [x] **Step 3:** CPU gate. Expected: `regressions=0`.
- [x] **Step 4: Commit** `flux: the e2e gate over seeds — gate A on the mean, gate B on the floor's worst seed, as F9b (F9c task 2)`.

### Task 3: `run_gates.sh flux` runs the e2e over three seeds with the floor

**Files:**
- Modify: `.claude/skills/difflet-cuda-verify/scripts/run_gates.sh`
- Test: `tests/unit/scripts/test_cuda_verify_skill_scripts.py`

- [x] **Step 1:** failing dry-run assertion (`--seeds 0,1,2 --noise-floor` in the flux e2e step). Expected: FAIL.
- [x] **Step 2:** implement. **Step 3:** CPU gate. **Step 4: Commit** `verify skill: run_gates.sh flux judges the e2e over three seeds against the noise floor (F9c task 3)`.

### Task 4: Docs, status log, PR; the pod run (pending the owner's yes)

- [ ] The catalog (the Flux multi-seed command, the gate's form), SKILL's runner line, `DEVELOPER.md`, the evidence report's reproduction lines; status-log entry "landed on the CPU"; push; PR stacked on #100; one fresh whole-branch reviewer. Pod (owner's yes): `cuda_e2e_flux_vs_diffusers.py --seeds 0,…,7 --noise-floor`, eager and `DIFFLET_CUDA_TORCH_COMPILE=1`, then `run_gates.sh flux`. Check first that seeds 1–2 of the eager run reproduce F4f's single-seed artifacts on a Server Edition card (Difflet 39.56 / 43.79 dB, floor 36.62 / 37.61 dB): nothing carries over between seeds in one process (the flag, cuBLAS state, caches).

## Rulings

- R1: the seed helpers live in the shared metrics module (`cuda_e2e_vs_diffusers.py`), one definition for both multi-seed scripts, as `image_metrics` / `_save` are.
- R2: one seed keeps F4f's single-seed relative statement (gates A and B coincide at n = 1); two or more seeds get F9b's A/B form. A single-seed result printed the F4f way keeps the committed F4f harness strings and artifact verdicts.
- R3: no perturbed-run PNG for Flux (F4f wrote none); the floor's metrics are what the gate reads.
- R4: inline execution, one fresh whole-branch reviewer.

## Execution ledger

```
Task 1: complete (be118a4; CPU gate 2870 passed).
Task 2: complete (a59f39b; CPU gate 2880 passed). The record test prints "seed N difflet vs diffusers: ..." (a "seed N: PSNR" line collided with the absolute test's failure messages in the harness).
Task 3: complete (7e29f17; CPU gate 2880 passed). SKILL §2's runner line rewritten in place (200 lines kept).
Task 4: docs + status log (e98fda0); pushed, PR #101 stacked on #100. Whole-branch review: no serious defect (the old and new gates give the same verdict on all 17 committed Flux e2e JSONs; every quoted number recomputed from the artifacts). Fix wave: two stale test counts in the evidence report; the clone of Difflet's noise put under test (the fake steps its latents in place; mutation-checked); the Flux e2e floor cells (support matrix, catalog, template) name the relative gate; "per_seed without seeds" asserted; diffusers' log lines name seed and run; two long lines. Not taken: a shared relative-gate module for Wan and Flux (the gates load under --import-mode=importlib; the copy is F9b's, verbatim). The pod run pending the owner's yes.
```
