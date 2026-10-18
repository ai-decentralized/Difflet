# CUDA F8 — `difflet-cuda-verify` skill, pod tooling and the evidence report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn what Release 1 and follow-ups F1–F5, F3b, F1b and F2 actually ran on Yottalabs pods into (1) a CUDA sibling of `.claude/skills/difflet-device-verify/` that a future session can follow without this history — pod lifecycle through the Yottalabs API, host setup with or without the `difflet-cache` volume, the per-model GPU gate sequence, triage of the failures we hit — (2) a `docs/verification/cuda-release1-evidence.md` report that lets a reviewer audit every CUDA claim from `artifacts/cuda-release1/` and the status log, and (3) pointers from the existing CUDA skill and DEVELOPER.md.

**Architecture:** A stdlib-only pod tool `scripts/yottalabs_pod.py` (urllib, the key read from `~/.config/yottalabs/api_key`, never printed) wraps the five API calls the sessions used (`GET /v2/gpus`, `POST /v2/pods`, `GET /v2/pods/{id}`, `POST …/pause|resume`, `DELETE /v2/pods/{id}`) with the capacity watcher, the RUNNING wait and the SSH-endpoint read as subcommands; its pure parts (request bodies, pool selection, endpoint parsing, polling decisions) are unit-tested against a fake transport. The skill's shell scripts are thin wrappers (`pod_setup.sh` with/without volume, `ship_branch.sh`, `run_gates.sh <model>`), each `bash -n`-checked by a test. The skill text and references are written from the status log and the ledgers, with every number labelled as the reference campaign's. The evidence report is built from the artifacts' JSONs (numbers transcribed, paths cited) and the status log; `.gitignore` gains the `docs/verification/*.md` exception the spec implies.

**Tech Stack:** Python 3.11 stdlib (`urllib.request`, `json`, `argparse`), bash, pytest, black 100, Markdown.

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — follow-up **F8 — `difflet-cuda-verify` skill and docs** ("CUDA sibling of `.claude/skills/difflet-device-verify/`, written from what F1–F4 actually ran; `docs/verification/` evidence report"); the Status log entries 2026-10-16 → 2026-10-18 are the source of every fact; the Trainium skill is the shape to mirror (`SKILL.md` sections 1–6, `references/{feature-catalog,failure-triage,gotchas,evidence-doc-template}.md`, `evals/evals.json`, `scripts/`). The pod-lifecycle rule in `.claude/skills/difflet-cuda-backend/SKILL.md` is binding: pause/terminate is pre-authorised at the end of a session, deploying needs the owner's yes.

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every command; every task ends with `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` green (regressions=0, new_skips=0) and touched Python files `black -l 100` clean. One task, one commit. Branch `feat/cuda-f8-verify-skill` on top of `feat/cuda-f2-second-gpu` (8c06acb).
- No network calls in tests; the pod tool takes an injectable transport. The API key is read from the file at call time and never logged, printed or included in error messages; `--dry-run` prints the request without sending it.
- Every fact in the skill and the report comes from the status log, the plan ledgers, `artifacts/cuda-release1/**` or code the writer read; numbers from the reference campaign are labelled as such (date, pod, card edition) and never presented as expectations for the next run. Nothing is invented; "not measured" is written where nothing was measured.
- No GPU run in F8; the scripts are exercised only by `bash -n`, `--help` and the fake transport. The first real use is the next pod session (recorded as the exit criterion for the skill).
- Nothing under `difflet/` changes.

## Review Focus

1. The pod tool never leaks the key: `--dry-run` output and every error path contain no key material — Task 1 `test_dry_run_and_errors_never_print_the_key`.
2. The capacity watcher creates exactly one pod and stops: with a fake transport that reports UNAVAILABLE twice then AVAILABLE, `create --wait-for-capacity` sends one `POST /v2/pods` and returns its id — Task 1 `test_wait_for_capacity_creates_once`.
3. Region preference and volume: `us-east-3` wins when available and gets `persistentVolumes` with the `difflet-cache` id; an eu region gets `containerVolumeInGb` and no volume — Task 1 `test_create_body_per_region`.
4. The skill's numbers are traceable: every number in `references/feature-catalog.md` and the evidence report appears in a status-log entry or an artifact JSON — Task 3's reviewer check (the report cites the artifact path next to each table).
5. The report's support matrix uses PASS / LIMIT / N/A / NOT MEASURED exactly as the Trainium template defines them, and marks the CLI matrix's Wan cell and `--dp × --cfg-parallel` as NOT MEASURED.

---

### Task 1: `scripts/yottalabs_pod.py` and the skill's shell wrappers

**Files:**
- Create: `scripts/yottalabs_pod.py`, `.claude/skills/difflet-cuda-verify/scripts/pod_setup.sh`, `.claude/skills/difflet-cuda-verify/scripts/ship_branch.sh`, `.claude/skills/difflet-cuda-verify/scripts/run_gates.sh`
- Test: `tests/unit/scripts/test_yottalabs_pod.py` (create), `tests/unit/scripts/test_cuda_verify_skill_scripts.py` (create: `bash -n` every script, `--help` of the pod tool)

**Interfaces:**
- `scripts/yottalabs_pod.py` subcommands: `gpus [--type NVIDIA_RTX_PRO_6000_96G]` (pool table: region, status, price, max count, disk limit); `create --name N --gpus 1|2 [--regions us-east-3,eu-3,eu-2] [--disk-gb 150] [--volume-id ID --mount /data] [--wait-for-capacity --interval 300 --max-wait 21600] [--dry-run]` (prints the pod id; with `--wait-for-capacity` polls `gpus` and creates in the first available preferred region; `us-east-3` adds the volume when `--volume-id` is given, other regions never do); `status ID`; `wait-running ID [--timeout 900]` (prints `host port` of the SSH expose when RUNNING); `pause ID` / `resume ID` (poll until the status leaves RUNNING / reaches RUNNING, report `INITIALIZING → PAUSED` as "no capacity"); `delete ID` (poll until gone); `ssh-config ID --alias NAME` (prints the `Host` block for `~/.ssh/config`). All read the key from `~/.config/yottalabs/api_key` (override `YOTTALABS_API_KEY_FILE`), use `urllib.request`, and take `transport=` for tests. Exit codes: 0 ok, 1 API error, 2 no capacity / timeout.
- Pure functions tested: `build_create_body(name, gpus, region, disk_gb, volume_id, mount, image)`, `pick_region(pools, preferred)`, `ssh_endpoint(pod_json) -> (host, port) | None`, `next_poll_decision(status, target)`.
- `pod_setup.sh [--volume]`: the F1b/F2 recipe (uv, venv, torch cu128, `-e .[test,cuda]`, hf cli, imageio-ffmpeg, env check); with `--volume` sets `HF_HOME=/data/hf` and chowns `/data` once, else `HF_HOME=$HOME/hf`. `ship_branch.sh <alias> <branch>`: bundle, scp, fetch, checkout on the pod. `run_gates.sh <qwen|flux|wan> <snapshot> [--out ~/logs/<name>]`: the per-model sequence (smoke → parity → gate → e2e → gate; Wan adds the rounding block and the e2e A/B flags) with the exact env, mirroring the F3/F5/F1/F1b runbooks.

- [x] **Step 1: Write the failing tests** (Review Focus 1–3; polling decisions; `bash -n`; `--help`). **Step 2: Run** → ImportError/failures. **Step 3: Implement.** **Step 4:** `pytest tests/unit/scripts -q` + CPU gate green. **Step 5: Commit** `scripts: yottalabs_pod — pod lifecycle through the API, and the CUDA verify skill's setup/ship/gate wrappers (F8 task 1)`.

### Task 2: The `difflet-cuda-verify` skill

**Files:**
- Create: `.claude/skills/difflet-cuda-verify/SKILL.md`, `references/feature-catalog.md`, `references/pod-runbook.md`, `references/failure-triage.md`, `references/gotchas.md`, `references/evidence-doc-template.md`, `evals/evals.json`

**Content (from the status log 2026-10-16 → 2026-10-18 and the F3/F4/F5/F3b/F1/F1b/F2 plan ledgers; label every number with its date/pod/card edition):**
- `SKILL.md` (≤ 200 lines, same six sections as the Trainium skill, frontmatter `name`/`description` tuned to CUDA): scope the session (models Qwen-Image / FLUX.1-dev / Wan 2.2; gates smoke, DiT parity, e2e, CLI matrix, serve smoke, two-GPU DP / CFG-parallel); the pod lifecycle through `scripts/yottalabs_pod.py` (deploy needs the owner's yes; pause pre-authorised at the end; a pod without a volume can only be terminated; capacity watcher); host setup; the gate sequence per model; checkpointing (artifacts under `artifacts/cuda-release1/<feature>/`, status-log entry append-only); triage; the report.
- `feature-catalog.md`: each gate's script, env, inputs, D14 thresholds and the reference numbers per model (Qwen F3/F4, Flux F5, Wan F1/F1b, serve F3b, two GPUs F2), with the card edition and the 2.3× Workstation/Server per-step gap called out.
- `pod-runbook.md`: Yottalabs facts — pools and prices seen, `GET /v2/gpus` status, eu pools sell 2-GPU nodes, us-east-3 can be full for hours, pause releases the card and resume needs capacity, container disk limit (`containerVolumeInGb`, the 118 GB overrun restart), the `difflet-cache` volume (id, region, CephFS 0.23 GB/s cold, what it holds), HF device-login recipe for gated repos, SSH alias pattern, `setsid nohup` for long runs, logs under `~/logs` not `/tmp`.
- `failure-triage.md`: spurious causes seen (device-cgroup loss on us-central-7, container restart on disk overrun, capacity refusals, the `.venv/bin/pip` absence under uv, the HF code expiry) before calling a Difflet bug; the bug protocol as on Trainium.
- `gotchas.md`, `evidence-doc-template.md` (CUDA tables), `evals/evals.json` (two evals: plan a session without a GPU; write the evidence skeleton from the artifacts).

- [x] **Step 1:** write the files. **Step 2:** a test in `tests/unit/scripts/test_cuda_verify_skill_scripts.py` asserts the skill's files exist, the frontmatter parses, and every `scripts/*.sh` / `scripts/yottalabs_pod.py` the SKILL.md names exists. **Step 3:** CPU gate green. **Step 4: Commit** `skill: difflet-cuda-verify — pod lifecycle, host setup, per-model gates and triage, from the Release 1 → F2 campaigns (F8 task 2)`.

### Task 3: The evidence report, pointers, status log

**Files:**
- Create: `docs/verification/cuda-release1-evidence.md`
- Modify: `.gitignore` (`!docs/verification/` and `!docs/verification/*.md`, in the style of the plans exception), `DEVELOPER.md` (a "Verifying the CUDA backend" pointer paragraph near the CUDA section), `.claude/skills/difflet-cuda-backend/SKILL.md` (the "Which installed skills" table gains `difflet-cuda-verify`; the pod-lifecycle rule points at `scripts/yottalabs_pod.py` for the API calls), `docs/plans/2026-10-16-cuda-backend-support.md` (append the F8 entry; mark the F8 roadmap line done)

**Report content:** header (branches, PRs #84–#89, pods and card editions, dates); per-feature sections with a table model × gate → numbers → artifact path → status-log entry, for: smoke, DiT parity (D14 gates), e2e (floors), CLI matrix, serve (F3b), performance (F4), two GPUs (F2), the F1b A/B; "How to inspect manually" per section (the exact script invocations); the rulings/bugs ledger (one line each with the commit); the support matrix (PASS / LIMIT / N/A / NOT MEASURED) as model × feature and as one block per feature; what is NOT MEASURED (CLI matrix Wan cell, dp×cfg, Flux under dp, Wan 2.1, noise floor); pod cost table.

- [x] **Step 1:** write the report and the pointers. **Step 2:** `git check-ignore docs/verification/cuda-release1-evidence.md` must exit 1; CPU gate green. **Step 3: Commit** `docs: CUDA Release 1 → F2 evidence report, verify-skill pointers, F8 status (F8 task 3)`.

## Rulings

- R1: the pod tool is stdlib-only and lives in `scripts/` (not inside the skill) so the CLI and future automation can import it; the skill's scripts call it.
- R2: `docs/verification/*.md` becomes tracked by a `.gitignore` exception rather than `git add -f`, so the report cannot be lost the way the TPU plan once was.
- R3: F8 ships without a GPU run; the skill's first real use is its exit criterion and is recorded as such.

## Execution ledger (folded from the SDD workspace, 2026-10-18)

```
# SDD ledger — plan: docs/superpowers/plans/2026-10-18-cuda-f8-verify-skill.md

Branch feat/cuda-f8-verify-skill on feat/cuda-f2-second-gpu (8c06acb). Pre-flight scan: T1 produces the pod tool and wrappers T2 names and T3 points at — consistent; T2 self: the file-existence test lives in T1's test module (T2 extends it) — ok; T3 self: the gitignore exception must precede the report add — ok. Scan clean.

## Tasks
Task 1: Ruling: `pause` waits for PAUSED, a `list` subcommand exists, the watcher retries a failed GET but never the create POST — implementer's departures, all kept. Live read-only check by the controller: `gpus`, `status`, `list`, `create --dry-run` work against the API and print no key.
Task 1: minor (deferred): the watcher retries every ApiError including a revoked key (retry only transient codes); one network error during `wait-running`/`delete`/`pause`/`resume` exits 1 after the request was sent (retry transient GETs in `poll`); the ssh-config block lacks `IdentitiesOnly yes` / `ServerAliveInterval 30`; `run_gates.sh` drops `CUDA_VISIBLE_DEVICES=0`, `/usr/bin/time -v` and `nvidia-smi topo -m` from the runbooks; no test asserts the flags `run_gates.sh` passes exist in the target scripts' argparse.
Task 1: complete (commits 40b7902..40a24fa, review clean)
Task 2: review (opus): numbers accurate and labelled, flags real; Needs fixes — five Important text items (unpinned volume create region, pause-then-delete block, matrix/serve commands without HF_HOME, NVML row deploying without the owner's yes, three unsupported runbook facts) + minors. Ruling: pausing a volume pod and terminating a pod that cannot be paused are the pre-authorised end-of-session actions; terminating a volume pod still needs the owner's word — Task 3 reconciles the backend skill's "DELETE only when the owner says terminate" line with this. Fix round 1 dispatched.
Task 2: fix round 1/5 (11 addressed, 0 open; commits 2cb5b9a..fc222f0). minor (deferred): SKILL.md at exactly 200 lines; two prose lines at 101/103 chars; gotchas.md and evals.json keep a looser "pausing/terminating pre-authorised" paraphrase; the serve-probe block dropped the `--tp-degree 2` refusal probe (the catalog keeps it).
Task 2: complete (commits 40a24fa..fc222f0, review clean after one fix round)
Task 3: review (opus): ~60 numbers all match their artifacts/entries; Needs fixes — two cited F2 `.log` artifacts are gitignored (ruling: track them with `!artifacts/cuda-release1/**/*.log`, R2 spirit); a `~` inside double quotes in the Wan parity inspect command; minors (F2 prompts added to the F8 entry from the controller's runbook, "38 dB" in three entries, 5.3 %, bench `--out`, `$SNAP`/`$H` defined, DEVELOPER run_gates usage, backend-skill heading/step 3/resume wording, two ledger citations, one computed label). Fix round 1 dispatched.
Task 3: fix round 1/5 (11 addressed, 0 open; commits 7e8df43..ad607d5 — the two F2 logs tracked via `!artifacts/cuda-release1/**/*.log`, 39 tracked artifacts, `$HOME` in on-pod commands, F2 prompts in the F8 entry, lifecycle wording in the backend skill).
Task 3: complete (commits fc222f0..ad607d5, review clean after one fix round)
Whole-branch review (8c06acb..ad607d5, opus): flags, flows and numbers agree across files; fresh-session dry run reached the right commands; With fixes — Important: SKILL.md two-GPU block sets no HF_HOME/HF_HUB_OFFLINE/DIFFLET_BACKEND; the lifecycle rule still loose in failure-triage NVML row, the tool's --help, evals.json, gotchas; the paused f5b holding the volume is never mentioned (a second pod on the volume: not measured); run_gates.sh early exits write no `GATES DONE` so the polling loop can hang. Minors: deploy-yes order, status vocabulary/NOT MEASURED count (5 vs 6), fixture public IP, eval 0 remedy (remove Flux, not Qwen), attached setup run, DEVELOPER.md line break. One fix dispatch follows.
Final fix wave: 38f93f8 (HF_HOME/offline/backend exported in the two-GPU block; one lifecycle rule in failure-triage, the tool's --help, evals, gotchas; f5b and the volume named with the not-measured claims; `GATES DONE … overall_rc=2` on every early exit + tests; runbook deploy order; catalog vocabulary + serve LIMIT + serving-on-two-GPUs row; six NOT MEASURED; fixture IP; eval 0 remedy; detached setup with a `SETUP DONE rc=` trap; DEVELOPER.md line break; ssh-config IdentitiesOnly/ServerAliveInterval; an appended "F8 review fixes" entry). Scoped re-review: all twelve addressed, no new breakage; observation: the setup poll ends on any rc and would retry forever if ssh itself fails.
Branch complete: 8c06acb..38f93f8, final review clean after one fix wave.
```
