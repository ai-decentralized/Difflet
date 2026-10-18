# CUDA F9 — Loose ends on one pod: the Wan e2e noise floor and the three-model CLI matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gates the status log records as NOT MEASURED that one single-GPU pod can close: (1) the **Wan e2e noise floor** — how far diffusers' own `WanPipeline` moves from itself when its timestep is rounded to bf16, over several seeds, so the 26.3 dB Difflet-vs-diffusers number has a scale; (2) the **three-cell CLI matrix** (`scripts/verify_cli_cuda.py` with Qwen-Image, FLUX.1-dev and Wan 2.2 on one pod); and (3) the first real use of the `difflet-cuda-verify` skill (`run_gates.sh wan`) — its exit criterion. Everything is recorded in the status log and the evidence report.

**Architecture:** `scripts/cuda_e2e_wan_vs_diffusers.py` gains `--seeds` (several seeds, one JSON with per-seed and aggregate metrics) and `--noise-floor` (per seed, a second diffusers run with both experts' `forward` wrapped so `timestep` is rounded through bf16 — `t.to(torch.bfloat16).to(t.dtype)` — compared against the exact run; the Difflet variants are compared against the exact run as today); the gate prints the noise floor and gates every seed against the existing floors. `scripts/yottalabs_pod.py` sends an explicit `--disk-gb` even when a volume is mounted (the API accepted both on `difflet-cuda-f5b`; the tool currently drops it). The pod session follows the skill: a us-east-3 pod on the `difflet-cache` volume (Qwen + Flux + the HF token) with a 170 GB container disk for the Wan download, the three cells, the noise floor, `run_gates.sh wan`, then pause.

**Tech Stack:** torch 2.11.0+cu128, diffusers 0.38.0, pytest, bash; Yottalabs API through `scripts/yottalabs_pod.py`.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — the NOT MEASURED items in the F1b, F2 and F8 entries; D14 gates; the `difflet-cuda-verify` skill (`.claude/skills/difflet-cuda-verify/SKILL.md`) for the session itself; the evidence report `docs/verification/cuda-release1-evidence.md` (sections "e2e", "CLI matrix", "NOT MEASURED") to update.

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task (`.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`: regressions=0, new_skips=0); black 100; one task, one commit. Branch `feat/cuda-f9-loose-ends` on `feat/cuda-f8-verify-skill` (69f2d98).
- The e2e floors (25 dB / 0.99 for Wan) do not move in this plan; the noise floor is recorded, not gated. Existing JSON keys (`metrics`, `variants`, `timings`) keep working for the F1b artifacts.
- Pod rule (the skill): the owner said "Ok" to this session on 2026-10-18 — deploying the us-east-3 volume pod when capacity appears is covered; the pod is paused at the end (it has the volume) with the status read back; logs under `~/logs/f9`; large downloads on the container disk (170 GB) or the volume, never beyond either.
- Nothing under `difflet/` changes.

## Review Focus

1. The perturbation touches only the timestep: with `--noise-floor`, the wrapped `forward` receives `timestep` rounded through bf16 and every other argument untouched; without it, diffusers runs exactly as before — Task 1 `test_noise_floor_wrapper_rounds_only_the_timestep` (a fake transformer records its kwargs).
2. Several seeds share one load and produce one JSON: `--seeds 0,1,2` loads Difflet once and diffusers once, draws one noise tensor per seed (from `torch.Generator().manual_seed(seed)` as today), and writes `per_seed[<seed>]` plus `aggregate` (min/mean/max of PSNR and cosine per variant and for the noise floor) — Task 1 `test_seeds_share_loads_and_aggregate`.
3. The gate judges every seed against the floors and prints the noise floor — Task 1 gate tests on fabricated JSONs (all seeds pass; one seed below the floor fails; legacy single-seed JSON still passes).
4. The pod tool sends `containerVolumeInGb` when `--disk-gb` is given explicitly, with or without a volume — Task 1 `test_create_body_keeps_an_explicit_disk_with_a_volume`.

---

### Task 1: Noise-floor A/B over seeds; explicit disk size with a volume

**Files:**
- Modify: `scripts/cuda_e2e_wan_vs_diffusers.py` (`--seeds` comma list overriding `--seed`; `--noise-floor` flag; `run_diffusers` loads once and generates per seed, with an optional `perturb_timestep` wrapper applied to `pipe.transformer` and `pipe.transformer_2`; `run_difflet` loads once and generates per seed per variant; JSON gains `seeds`, `per_seed`, `aggregate`, `noise_floor` (per seed: exact-vs-perturbed metrics) while keeping `metrics`/`variants`/`timings` for the first seed), `tests/numerical/test_cuda_wan_e2e_vs_diffusers.py` (every seed gated; the noise floor printed under `-rA`), `scripts/yottalabs_pod.py` (`build_create_body`: an explicit `disk_gb` is sent even with a volume; the CLI distinguishes "not given" from the default)
- Test: `tests/unit/scripts/test_cuda_e2e_wan_script.py`, `tests/unit/scripts/test_cuda_wan_gates.py`, `tests/unit/scripts/test_yottalabs_pod.py`

- [ ] **Step 1: Write the failing tests** (Review Focus 1–4). **Step 2: Run.** **Step 3: Implement.** **Step 4:** `pytest tests/unit/scripts tests/numerical -q` + CPU gate green; `--help` on the CPU host. **Step 5: Commit** `wan: e2e noise floor over seeds — diffusers vs its bf16-rounded timestep; pod tool keeps an explicit disk size with a volume (F9 task 1)`.

### Task 2: The pod session (the skill's first real use)

- [ ] **Step 1:** the capacity watcher creates `difflet-cuda-f9` (us-east-3, 1 GPU, volume at `/data`, 170 GB container disk); `wait-running`, `ssh-config --alias difflet-cuda-f9`, `ship_branch.sh difflet-cuda-f9 feat/cuda-f9-loose-ends`, `pod_setup.sh --volume` (detached, poll `SETUP DONE`). Record the pod ID in the ledger at once. Check `nvidia-smi` on arrival and again after a few minutes.
- [ ] **Step 2:** Wan onto the container disk: `HF_HOME=/tmp/hf-wan hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers` (≈118 GB; the 170 GB disk holds it), then `ln -s /tmp/hf-wan/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers /data/hf/hub/` so one `HF_HOME=/data/hf` sees all three models (record whether the symlink works through `local_files_only`; if not, run the Wan steps with `HF_HOME=/tmp/hf-wan`).
- [ ] **Step 3:** the three-cell matrix `HF_HOME=/data/hf HF_HUB_OFFLINE=1 DIFFLET_BACKEND=cuda python scripts/verify_cli_cuda.py --seed 0` (Qwen 50, Flux 28, Wan 40 steps; the Wan cell writes an MP4); copy `results.json`.
- [ ] **Step 4:** the noise floor: `scripts/cuda_e2e_wan_vs_diffusers.py --model-path <wan snapshot> --seeds 0,1,2 --noise-floor --timestep-dtypes float32 --steps 40 --num-frames 9 --out ~/logs/f9/cuda_e2e_wan_noise.json --images-dir ~/logs/f9` + the gate; then `run_gates.sh wan <snapshot> --out ~/logs/f9/gates-wan` (the skill's first real use; note every friction point for Task 3).
- [ ] **Step 5:** copy artifacts to `artifacts/cuda-release1/f9/`; `pause` the pod and read back `PAUSED`.

### Task 3: Docs, skill fixes from the first real use, status log

- [ ] Append the F9 status-log entry (numbers, the noise-floor verdict: is 26.3 dB inside, near or far outside diffusers' own seed-to-seed/timestep-rounding spread; the matrix verdict; every friction point); update the evidence report's e2e, CLI-matrix and NOT MEASURED sections and the skill's catalog (the matrix Wan cell and the noise floor leave NOT MEASURED); fix the skill where the session showed a gap; fold the ledger; CPU gate; commit `docs: F9 status — Wan e2e noise floor and the three-model CLI matrix on one pod`.

## Rulings

- R1: the noise floor is a measurement of diffusers against itself; it is recorded next to the Difflet number, never used to move a floor in this plan.
- R2: Wan lives on the container disk (symlinked into the volume's hub) rather than evicting Qwen or Flux from the volume, so all three cells run under one `HF_HOME`.
- R3: the pod is paused, not terminated, at the end (it holds the volume).

## Execution ledger

(appended by the executor, one line per task: numbers, rulings, deviations)
