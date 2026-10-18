# CUDA F9b — The Wan e2e gate stated relative to the measured noise floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** F9 measured the Wan e2e metric's own noise floor — diffusers' `WanPipeline` against itself with its timestep rounded through bf16 — at 28.62 / 26.74 / 23.79 dB over seeds 0–2, next to Difflet vs diffusers at 26.30 / 26.39 / 28.89 dB. The gate still holds every seed to an absolute 25 dB / 0.99, which the floor's own worst seed (23.79 dB) fails: a correct build can fail on seed luck. This plan restates the Wan e2e gate the way D14 states the parity gates — relative to what the reference costs itself — whenever the result JSON carries the noise floor, and keeps the absolute floors for results without it (legacy JSONs, `run_gates.sh wan`'s one-seed e2e). No GPU run: the F9 artifact JSONs are the measurement; the gate is re-run on them on the Mac.

**Architecture:** the gate (`tests/numerical/test_cuda_wan_e2e_vs_diffusers.py`) gets D14's three-gate shape for the e2e metric. With a noise floor on every seed: (A) **the mean over seeds** of Difflet's PSNR is at least the floor's mean PSNR minus `20·log10(1.15)` = 1.21 dB (a 1.15× RMSE ratio, D14's parity allowance, in dB; the mean of dB values is the geometric mean of RMSE ratios), and the mean cosine deviation `1 − cosine` is at most 1.15× the floor's; (B) **every seed** is at least the floor's *worst* seed minus 1.21 dB, and its cosine deviation at most 1.15× the floor's worst — the catastrophe catch, stated against the floor's spread rather than seed-matched because one half-ulp perturbation moves diffusers by 5 dB between seeds; (C) **a loose absolute floor** per seed, 20 dB / 0.98, so a floor that collapsed cannot pass a bad build. Without a noise floor the absolute 25 dB / 0.99 per-seed floors judge exactly as before, and the relative test is SKIPPED with the reason printed. The script changes only its docstring (the floor is no longer "never gated"). The Mac gate tests (`tests/unit/scripts/test_cuda_wan_gates.py`) pin every branch on fabricated JSONs and re-gate the committed F9 artifacts.

**Tech Stack:** pytest, stdlib `math`; no torch on the gate path.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — D14 (relative gates: ≤ 1.15× the control's deviation, a loose absolute floor), the F9 status-log entries (the measurement, ruling R1 and its recommended follow-up), the "F9 review fixes (2)" verdict wording; the evidence report `docs/verification/cuda-release1-evidence.md` (e2e section, "Noise floor (F9)", NOT MEASURED, ledger) and the `difflet-cuda-verify` skill (catalog §e2e, SKILL §5 line 5, failure triage) to update.

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task (`.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`: regressions=0, new_skips=0); black 100; one task, one commit. Branch `feat/cuda-f9b-relative-wan-gate` on `feat/cuda-f9-loose-ends` (e103424, PR #91).
- Nothing under `difflet/` changes; no GPU run; no pod. The F9 JSONs are not edited.
- The absolute floors 25 dB / 0.99 are not loosened for results without a noise floor; the relative gate is a different statement, used only when the floor was measured in the same run. Qwen's and Flux's gates are untouched.
- The committed artifacts keep their verdicts: `wan/f1b/cuda_e2e_wan.json` and `f9/gates-wan/cuda_e2e_wan.json` pass (relative gate skipped), `f9/cuda_e2e_wan_noise.json` passes every test, `wan/cuda_e2e_wan.json` (F1) fails only the variants test as before.

## Review Focus

1. The arithmetic: `MARGIN_DB = 20·log10(MAX_DEVIATION_RATIO)` with `MAX_DEVIATION_RATIO = 1.15` (1.21 dB); cosine deviation is `1 − cosine`; gate A on means over the seeds, gate B against the floor's min PSNR / max deviation, gate C the loose absolute floor — Task 1 fabricated-JSON tests at the boundaries (`test_e2e_relative_gate_*`).
2. The mode switch: a noise floor on every seed → A+B+C judge and the 25 / 0.99 floors are not applied; no floor (legacy, or `--seeds` without `--noise-floor`) → 25 / 0.99 per seed as before and the relative test is SKIPPED with a reason; a floor on some seeds only is a malformed result and fails (the recorded test already requires all-or-none).
3. The committed F9 noise JSON passes the relative gate with the headroom the status-log entry states (mean 27.20 vs 25.17 dB required; worst seed 26.30 vs 22.58 dB required); the Mac test `test_committed_e2e_artifacts_keep_their_verdicts` covers both F9 JSONs.
4. Every doc that said "recorded, never gated" / "floors stay 25 dB / 0.99 (R1)" / "recommended follow-up" now states the relative gate, with the F9 numbers unchanged (Task 2).

---

### Task 1: The relative gate in the Wan e2e test

**Files:**
- Modify: `tests/numerical/test_cuda_wan_e2e_vs_diffusers.py` (docstring; constants `MAX_DEVIATION_RATIO = 1.15`, `MARGIN_DB`, `LOOSE_PSNR_DB = 20.0`, `LOOSE_COSINE = 0.98`; `_noise_floors(payload)` → per-seed floor metrics or `None`; `test_video_is_the_same_clip_as_diffusers` applies the loose floors per seed when a floor is recorded (printing that 25 / 0.99 are superseded), the 25 / 0.99 floors otherwise; new `test_video_is_as_close_to_diffusers_as_diffusers_is_to_itself` with gates A and B, skipped without a floor, each miss named with the numbers), `scripts/cuda_e2e_wan_vs_diffusers.py` (docstring only: the floor scales the gate since F9b)
- Test: `tests/unit/scripts/test_cuda_wan_gates.py` (new `_outcomes` helper that tolerates the relative test's SKIPPED; `test_e2e_relative_gate_passes_the_f9_shape`, `..._fails_when_the_mean_is_more_than_1_15x_from_diffusers_own_floor`, `..._fails_when_one_seed_is_below_the_floor_s_worst_seed_minus_the_margin`, `..._holds_the_loose_absolute_floor_even_when_the_noise_floor_collapses`, `..._cosine_deviation_is_judged_the_same_way`, `..._is_skipped_without_a_noise_floor`; the 25 / 0.99 tests updated: `test_e2e_noise_floor_is_recorded_not_gated` becomes the collapsed-floor case; both F9 JSONs added to `test_committed_e2e_artifacts_keep_their_verdicts`)

- [x] **Step 1: Write the failing Mac tests** (Review Focus 1–3). Boundary values as written, adjusted in the test where noted: floors F9_FLOOR = {0: (28.62, 0.9977), 1: (26.74, 0.99575), 2: (23.79, 0.99255)} (the F9 numbers) with Difflet at the F9 numbers → PASSED; Difflet all three at 25.0 dB (mean 25.0 < 26.38 − 1.21 = 25.17) → gate A FAILED naming "mean PSNR 25.00 dB below 25.17"; seeds (29.0, 29.0, 22.5) (the mean must clear gate A) with floor min 23.79 → gate B FAILED naming "seed 2: PSNR 22.50 dB below 22.58"; floor collapsed to (18, 19, 20 dB) → the relative test FAILED as "the oracle moved" (review fix; first written as PASSED) and Difflet at (19.5, 19.5, 19.5) FAILED on the loose 20 dB floor; cosine: Difflet (0.997, 0.997, 0.993) vs floor cosines (0.999, 0.998, 0.997): mean deviation 0.00433 > 1.3225 × 0.002 → FAILED naming "mean cosine deviation" (seeds 0 and 1 clear gate B's per-seed bound, seed 2 does not); no floor → the relative test SKIPPED and the 25 / 0.99 floors judge (existing tests).
- [x] **Step 2: Run** `DIFFLET_BACKEND=cpu .venv/bin/python -m pytest tests/unit/scripts/test_cuda_wan_gates.py -q`. Expected: the new tests fail (the relative test does not exist; the collapsed-floor case passes the old absolute test wrongly).
- [x] **Step 3: Implement** the gate as in Architecture; black.
- [x] **Step 4: Run** the Mac tests, then `DIFFLET_RUN_CUDA_NUMERICAL=1 DIFFLET_CUDA_WAN_E2E_RESULT=artifacts/cuda-release1/f9/cuda_e2e_wan_noise.json .venv/bin/python -m pytest tests/numerical/test_cuda_wan_e2e_vs_diffusers.py -rA -q` (Expected: 5 passed, the relative lines printed with the required thresholds), the same on `f9/gates-wan/cuda_e2e_wan.json` and `wan/f1b/cuda_e2e_wan.json` (Expected: 4 passed, 1 skipped), then the CPU gate (regressions=0, new_skips=0).
- [x] **Step 5: Commit** `wan: e2e gate stated relative to the measured noise floor — 1.15x (1.21 dB) on the mean over seeds, the floor's worst seed per seed, a loose 20 dB / 0.98 absolute floor (F9b task 1)`.

### Task 2: Docs and the status log

**Files:**
- Modify: `docs/plans/2026-10-16-cuda-backend-support.md` (append the F9b status-log entry: the gate's three statements with the F9 numbers and the headroom; ruling R1 of F9 superseded for results that carry the floor), `docs/verification/cuda-release1-evidence.md` (e2e floors paragraph; the "Noise floor (F9)" verdict's last sentence; the inspect commands' expected counts; NOT MEASURED's "recommended follow-up"; ledger line 28), `.claude/skills/difflet-cuda-verify/references/feature-catalog.md` (§e2e floors; the F9 paragraph's last sentence; the support-matrix cell), `.claude/skills/difflet-cuda-verify/SKILL.md` (§5 line 5), `.claude/skills/difflet-cuda-verify/references/failure-triage.md` (the noise-floor bullet), `DEVELOPER.md` (the Wan paragraph's last sentence on the floors), this plan (fold the ledger).

- [x] **Step 1:** Edit the docs; every F9 number unchanged; `grep -rn "never gated\|recorded, not gated\|recommended follow-up" docs .claude DEVELOPER.md` finds only historical status-log entries.
- [x] **Step 2:** CPU gate green; `pytest tests/unit/scripts -q`.
- [x] **Step 3: Commit** `docs: F9b — the Wan e2e gate relative to the noise floor; status log, evidence report, skill`.

## Rulings

- R1 (supersedes F9's R1 for results that carry the floor): the noise floor is used to *state* the gate, not to move the 25 dB / 0.99 floors; those still judge any result without a floor, so no existing verdict loosens.
- R2: gate B compares every seed with the floor's worst seed, not seed-matched: the per-seed floor varies by 5 dB under a half-ulp perturbation, so a matched comparison would be noise-driven (seed 0 would fail at 2.3 dB behind its own floor while seed 2 is 5 dB ahead of its own).
- R3: inline execution (executing-plans) with one fresh whole-branch reviewer: two tasks, no GPU, no new behaviour under `difflet/`.

## Execution ledger (folded from the SDD workspace, 2026-10-18)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-18-cuda-f9b-relative-wan-gate.md

Branch feat/cuda-f9b-relative-wan-gate on feat/cuda-f9-loose-ends (e103424). Owner picked "Relative Wan gate" from the three follow-ups offered after F9. Inline execution (R3).
Task 1: ruling R2 — gate B against the floor's worst seed, not seed-matched (seed 0 is 2.3 dB behind its own floor, seed 2 is 5.1 dB ahead of its own; a matched gate would be noise-driven). The Mac outcome parser read only the -rA summary, where a SKIPPED line carries no test id — it now reads the -v progress lines. SKILL.md went to 201 lines with the §5 note; compacted back to 200 (the skill test holds it there).
Task 1: complete (commit 4625a69; CPU gate imports=ok unit=[2801 passed, 176 skipped] regressions=0 new_skips=0)
Task 2: complete (docs + status-log entry, this commit; tests/unit/scripts 176 passed; CPU gate imports=ok unit=[2801 passed, 176 skipped] regressions=0 new_skips=0)
```

Whole-branch review (fresh reviewer, e103424..c361df1): needs fixes, none critical. Important: (1) 1 − cosine is quadratic in the error, so 1.15× on it is ~1.07× linear (0.6 dB) — the binding half; fixed by allowing 1.15² = 1.3225× on 1 − cosine (allowed A 0.00617, B 0.00986); (2) the floor's mean at n = 3 carries ~1.4 dB standard error, so the 2.0 dB headroom is ~one SE of the difference — stated in the docstring and docs. Minor: stale F1 inspect count; no sanity check on the floor itself → the relative test fails as "the oracle moved" when the floor is not finite or its worst seed is under the loose floor; the per-seed `_gated_metrics` fallback to the first seed's metrics → asserts; the outcome regex now tolerates `Class::test` and a space in the path; plan Step 1 text aligned with the tests. Fix wave: one commit; `tests/unit/scripts` 178 passed; CPU gate imports=ok unit=[2803 passed, 176 skipped] regressions=0 new_skips=0.
