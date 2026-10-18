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

- [x] **Step 1: Write the failing tests** (Review Focus 1–4). **Step 2: Run.** **Step 3: Implement.** **Step 4:** `pytest tests/unit/scripts tests/numerical -q` + CPU gate green; `--help` on the CPU host. **Step 5: Commit** `wan: e2e noise floor over seeds — diffusers vs its bf16-rounded timestep; pod tool keeps an explicit disk size with a volume (F9 task 1)`.

### Task 2: The pod session (the skill's first real use)

- [x] **Step 1:** the capacity watcher creates `difflet-cuda-f9` (us-east-3, 1 GPU, volume at `/data`, 170 GB container disk); `wait-running`, `ssh-config --alias difflet-cuda-f9`, `ship_branch.sh difflet-cuda-f9 feat/cuda-f9-loose-ends`, `pod_setup.sh --volume` (detached, poll `SETUP DONE`). Record the pod ID in the ledger at once. Check `nvidia-smi` on arrival and again after a few minutes.
- [x] **Step 2:** Wan onto the container disk: `HF_HOME=/tmp/hf-wan hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers` (≈118 GB; the 170 GB disk holds it), then `ln -s /tmp/hf-wan/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers /data/hf/hub/` so one `HF_HOME=/data/hf` sees all three models (record whether the symlink works through `local_files_only`; if not, run the Wan steps with `HF_HOME=/tmp/hf-wan`).
- [x] **Step 3:** the three-cell matrix `HF_HOME=/data/hf HF_HUB_OFFLINE=1 DIFFLET_BACKEND=cuda python scripts/verify_cli_cuda.py --seed 0` (Qwen 50, Flux 28, Wan 40 steps; the Wan cell writes an MP4); copy `results.json`.
- [x] **Step 4:** the noise floor: `scripts/cuda_e2e_wan_vs_diffusers.py --model-path <wan snapshot> --seeds 0,1,2 --noise-floor --timestep-dtypes float32 --steps 40 --num-frames 9 --out ~/logs/f9/cuda_e2e_wan_noise.json --images-dir ~/logs/f9` + the gate; then `run_gates.sh wan <snapshot> --out ~/logs/f9/gates-wan` (the skill's first real use; note every friction point for Task 3).
- [x] **Step 5:** copy artifacts to `artifacts/cuda-release1/f9/`; `pause` the pod and read back `PAUSED`.

### Task 3: Docs, skill fixes from the first real use, status log

- [x] Append the F9 status-log entry (numbers, the noise-floor verdict: is 26.3 dB inside, near or far outside diffusers' own seed-to-seed/timestep-rounding spread; the matrix verdict; every friction point); update the evidence report's e2e, CLI-matrix and NOT MEASURED sections and the skill's catalog (the matrix Wan cell and the noise floor leave NOT MEASURED); fix the skill where the session showed a gap; fold the ledger; CPU gate; commit `docs: F9 status — Wan e2e noise floor and the three-model CLI matrix on one pod`.

## Rulings

- R1: the noise floor is a measurement of diffusers against itself; it is recorded next to the Difflet number, never used to move a floor in this plan.
- R2: Wan lives on the container disk (symlinked into the volume's hub) rather than evicting Qwen or Flux from the volume, so all three cells run under one `HF_HOME`.
- R3: the pod is paused, not terminated, at the end (it holds the volume).

## Execution ledger (folded from the SDD workspace, 2026-10-18)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-18-cuda-f9-loose-ends.md

Branch feat/cuda-f9-loose-ends on feat/cuda-f8-verify-skill (69f2d98). Owner: "Ok" (2026-10-18) to the pod session incl. deploy when capacity appears. Pre-flight scan: T1 produces `--seeds`/`--noise-floor` and the JSON keys T2 runs and T3 reports — consistent; T1 self: legacy JSON keys kept for the F1b artifacts — ok; T2 depends on a us-east-3 pod (watcher running with curl because the tool drops --disk-gb with a volume — T1 fixes the tool). Scan clean.

## Tasks
Task 1: minor (deferred): a seed whose variants lack the gated dtype falls back to seed 0's top-level metrics (the variants test catches it); the noise floor also absorbs run-to-run GPU nondeterminism (an upper bound on the timestep-cast effect — say so in Task 3); per-seed `peak_alloc_gb` is cumulative from one reset.
Task 1: complete (commits f05e24c..2387653, review clean)
Task 2: the curl watcher had a substring bug (grep AVAILABLE matched UNAVAILABLE → a create attempt every 5 min, each refused 12014, no pod created); stopped 18:41 UTC and replaced by `scripts/yottalabs_pod.py create --wait-for-capacity` (us-east-3, 1 GPU, volume, 170 GB disk) — the tool's first live watcher run.
Task 2: owner's "Switch to eu" (2026-10-18 18:44 UTC) → us-east-3 watcher stopped; `yottalabs_pod.py create --gpus 1 --regions eu-3,eu-2 --disk-gb 250` accepted a 1-GPU node → pod `difflet-cuda-f9`, ID 502673395609120916, eu-3, $2.65/h, no volume (terminate at the end). Flux needs the owner's HF device login on this pod.
Task 2: skill friction #1 — `pod_setup.sh` exits 2 unless run with cwd inside the checkout (`bash ~/Difflet/.claude/…/pod_setup.sh` from `~` fails: "not a Difflet checkout"); SKILL.md §2 should show `cd ~/Difflet &&` or the script should cd to its own repo root. Fix in Task 3.
Task 2: run on difflet-cuda-f9 (eu-3, 1× RTX PRO 6000 Server Edition, 250 GB disk, no volume): setup 52 s; Wan 401 s, Qwen 175 s, Flux 109 s (device login by the owner, two codes — the first expired); three-cell matrix all PASS in 191 s (Qwen generate 51.9 s, Flux 16.8 s, Wan 111.5 s); noise floor over seeds 0–2 exit 0 in 518 s, gate passed; `run_gates.sh wan` exit 0 in 595 s (`GATES DONE wan overall_rc=0`) — the skill's first real use; no process left on the card. Skill friction #2: none beyond #1 so far (details after reading the logs).
Task 2: results — matrix 3/3 PASS (Qwen 51.9 s, Flux 16.8 s, Wan 111.5 s generate; 191 s wall); noise floor seeds 0–2: Difflet fp32 vs exact 26.30 / 26.39 / 28.89 dB, diffusers exact vs bf16-rounded-timestep self 28.62 / 26.74 / 23.79 dB (aggregates 26.30–28.89 vs 23.79–28.62) → Difflet is inside diffusers' own perturbation spread; `run_gates.sh wan` overall_rc=0 (595 s). Pod terminated 19:50 UTC (GONE), ~66 min ≈ $2.90. Ruling: floors stay 25 dB / 0.99 (R1); recommended follow-up — state the Wan e2e gate relative to the measured noise floor (its worst seed, 23.8 dB, is below the gate). Friction: #1 pod_setup.sh cwd; #2 the device code expired once (issue it only when the owner is ready); #3 skill text predates the tool's explicit --disk-gb-with-volume. Pod terminated through the tool (delete → GONE).
Task 2: complete
Task 3: implemented in 355c269 (docs, pod_setup.sh resolves its own checkout + test, catalog/runbook/evals updated); whole-branch review (opus): every number matches; Important — the verdict is overstated in six places ("the same range", "not a Difflet defect"; matched by seed Difflet is further than the perturbed run on seeds 0 and 1); minors: unfolded ledger (controller), download times lack the status-log-only label, "seed-to-seed spread" wording, the nondeterminism hedge could cite the bit-identical seed-0 PSNR across three runs, run_gates.sh still REPO=$PWD. Ruling: wording cut back to "inside diffusers' own perturbation spread (overlapping, not identical ranges); need not indicate a defect; the noise floor is an upper bound on the timestep-cast effect" — the gate recommendation stands. One fix dispatch follows.
Final fix wave: a5792e9 + bf1b114 (verdict cut back; "overlaps", not "lies inside" — the implementer caught that 28.89 dB exceeds the spread's top; download times labelled status-log only; run_gates.sh header note; two appended status entries). Scoped re-review: all addressed, no new breakage; cosmetic: three unwrapped long lines in the evidence report.
Task 3: complete (commits 2387653..bf1b114)
Branch complete: 69f2d98..bf1b114, final review clean after one fix wave. R2/R3 not applied (eu pod without a volume; terminated).
```
