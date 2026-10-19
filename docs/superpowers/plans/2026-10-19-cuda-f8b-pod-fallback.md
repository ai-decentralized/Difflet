# CUDA F8b — the pod tool waits for us-east-3, then falls back; eu pods download over HTTPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The owner's pod policy (2026-10-19): us-east-3 with the `difflet-cache` volume first ($0.45/h, no download, no login — the only region where a pod volume lives), eu-2 / eu-3 when it is full. Today that fallback is a hand decision. (1) `scripts/yottalabs_pod.py create --prefer-first-for SECONDS`: while the first `--regions` entry has no card, keep polling it for up to that long before taking a later region; the volume rides only on the volume region's body, an eu fallback gets a container disk (unchanged). (2) `pod_setup.sh --no-xet`: `HF_HUB_DISABLE_XET=1` for the session and `~/.bashrc` — on pod f3c (eu-2, 10-19) the Xet download of FLUX.1-dev stalled at 0 MB/s and only plain HTTPS finished it; Xet stays the default (us-east-3 pulled ~1 GB/s with it).

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — F8 (the pod tool, the verify skill), the F3c Flux verdict (the stalled download, the no-capacity resume).

## Global Constraints

- CPU gate green after each task; black 100; one task, one commit. Branch `feat/cuda-f8b-pod-fallback` on `feat/cuda-f3c-serve-flux-wan` (7472112, PR #102).
- Default behaviour of both tools unchanged.

### Task 1: `create --prefer-first-for`

**Files:** Modify `scripts/yottalabs_pod.py`; Test `tests/unit/scripts/test_yottalabs_pod.py`.

- [x] Step 1: failing tests (the first region comes back inside the window → created there with the volume via v1; it never does → after the window the next region via v2 with a container disk; 0 = today's immediate fallback; the window bounds the sleeps). **Step 2:** implement. **Step 3:** CPU gate. **Step 4: Commit** `pod tool: create --prefer-first-for waits for the first region before falling back (F8b task 1)`.

### Task 2: `pod_setup.sh --no-xet`

**Files:** Modify `.claude/skills/difflet-cuda-verify/scripts/pod_setup.sh`; Test `tests/unit/scripts/test_cuda_verify_skill_scripts.py`.

- [x] Step 1: failing tests (flags combine in any order; `--no-xet` is in `--help`; unknown flags still exit 2). **Step 2:** implement. **Step 3:** CPU gate. **Step 4: Commit** `verify skill: pod_setup.sh --no-xet downloads over plain HTTPS (F8b task 2)`.

### Task 3: Docs, status log, PR

- [x] SKILL §2 / the runbook / gotchas name the policy and both flags; status-log entry; push; PR stacked on #102; one fresh reviewer.

## Execution ledger

```
Task 1: complete (569a559; CPU gate 3001 passed).
Task 2: complete (eab3505; CPU gate 3002 passed).
Task 3: docs + status log (this commit); push, PR, review.
```
