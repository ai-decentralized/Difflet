---
name: difflet-cuda-verify
description: >-
  Run a GPU verification session for Difflet's CUDA backend on a Yottalabs RTX PRO 6000 pod and
  leave auditable evidence. Use this whenever the user wants to verify, re-verify, benchmark or
  "prove on the GPU" Qwen-Image, FLUX.1-dev or Wan 2.2 on CUDA — smoke (`difflet run`), DiT
  parity vs diffusers, e2e vs diffusers, the CUDA CLI matrix, the `difflet serve` smoke, two-GPU
  `--dp` / Wan `--cfg-parallel` — including requests phrased as "run the GPU gates", "deploy a
  pod and check this branch", "does Wan still pass on CUDA", "resume the pod", "get a 2-GPU
  node", "re-run parity after this change". Also use it to plan a session without a GPU, to
  triage a failed gate (pod infrastructure vs a Difflet bug), and to write the evidence report
  from artifacts/cuda-release1/. It carries the pod lifecycle through scripts/yottalabs_pod.py
  (deploying needs the owner's yes; pausing or terminating at the end is pre-authorised).
---

# Difflet CUDA verification sessions

A session turns "this branch should work on CUDA" into "it ran on an RTX PRO 6000; here are the
JSONs, the images, the timings and the exact commands" — or into a root-caused bug with its own
commit. Distilled from Release 1 → F2 (2026-10-16 → 10-18, eight pods; the status log at the end
of `docs/plans/2026-10-16-cuda-backend-support.md`). Every number in this skill is the reference
campaign's, labelled with date, pod and card edition: what a pass looked like, never a fact
about the next run.

Read `references/feature-catalog.md` for each gate's script, inputs, thresholds and reference
numbers; `references/pod-runbook.md` before creating a pod; `references/failure-triage.md`
before calling a failure a Difflet bug; `references/evidence-doc-template.md` for the report;
`references/gotchas.md` when something fails for a reason that "can't be right". Load
`.claude/skills/difflet-cuda-backend/SKILL.md` as well: its decisions D1–D19, its CPU gate and
its pod-lifecycle rule bind this skill.

## 1. Scope the session with the owner

Pin down only what the request leaves open:

- **Models**: `qwen` = `Qwen/Qwen-Image`, `flux` = `black-forest-labs/FLUX.1-dev` (gated repo,
  needs the owner's Hugging Face login), `wan` = `Wan-AI/Wan2.2-T2V-A14B-Diffusers`. Wan 2.1 is
  routed to the Wan path but has never run on a GPU: NOT MEASURED, not a regression check.
- **Gates**, per model in this order: smoke → DiT parity → parity gate → e2e → e2e gate
  (`run_gates.sh` runs all five); then the CLI matrix (`scripts/verify_cli_cuda.py`, every cell
  at once — it has no cell filter), the serve smoke (Qwen-Image only), two GPUs (`--dp 2`, Wan
  `--cfg-parallel`; a 2-GPU pod). `scripts/cuda_bench_dit.py` only when the change is about
  speed, and only compared on the same card edition.
- **Branch** under test (CPU gate green first) and the evidence home:
  `artifacts/cuda-release1/<feature>/` plus an append-only status-log entry.
- **Pod**: 1 GPU in us-east-3 with the `difflet-cache` volume (cheapest; the pool can be full for
  hours), or a 2-GPU eu node (whole nodes only, no volume, terminate at the end). Disk decides
  what fits — Qwen + Flux live on the volume, Wan (118 GB) does not fit beside them; see the
  runbook's storage table before promising three models on one pod.

State the plan back: the expected outcome of every gate (the catalog's snapshot), the commands,
the pod and its price per hour as `gpus` shows it now, the disk plan. Then ask for the deploy
yes. A planning-only session ends here with the plan written out; nothing is created.

## 2. Get a pod and prepare the host

**Deploying (create, resume) needs the owner's explicit yes. Pausing or terminating at the end of
the session is pre-authorised — do it without asking.** Touch only pods created by the owner's
login whose name starts with `difflet-`. Everything goes through `scripts/yottalabs_pod.py`
(stdlib; key file `~/.config/yottalabs/api_key`, mode 600, never printed; exit 0 ok, 1 API error
or FAILED pod, 2 no capacity or timeout):

```bash
python3 scripts/yottalabs_pod.py gpus                  # pools: status AVAILABLE | UNAVAILABLE
python3 scripts/yottalabs_pod.py create --name difflet-<topic> --gpus 1 \
    --volume-id 469225541779435520 --mount /data --dry-run        # show it; on the owner's yes:
POD=$(python3 scripts/yottalabs_pod.py create --name difflet-<topic> --gpus 1 \
    --volume-id 469225541779435520 --mount /data --wait-for-capacity)
python3 scripts/yottalabs_pod.py wait-running "$POD"               # prints "host port"
python3 scripts/yottalabs_pod.py ssh-config "$POD" --alias difflet-cuda-<topic> >> ~/.ssh/config
```

Write the pod id into the session notes the moment it is printed. `--wait-for-capacity` polls
every 300 s and creates exactly once (F2's pod came from a watcher after every PRO 6000 pool had
been full for an hour). For two GPUs: `--gpus 2 --regions eu-3,eu-2 --disk-gb 150`, no volume.
`resume ID` exits 2 when the pod falls back INITIALIZING → PAUSED: no capacity, not a bug.

Then ship the branch and build the host (the F1b/F2 recipe):

```bash
S=.claude/skills/difflet-cuda-verify/scripts
REPO=$PWD bash $S/ship_branch.sh difflet-cuda-<topic> <branch>    # laptop → ~/Difflet on the pod
ssh difflet-cuda-<topic> nvidia-smi        # now, and again a few minutes later (NVML loss, triage)
ssh difflet-cuda-<topic> "cd ~/Difflet && bash $S/pod_setup.sh --volume"   # no volume: drop flag
```

`pod_setup.sh` builds the uv venv (torch 2.11.0 cu128, `-e .[test,cuda]`, the `hf` CLI,
imageio-ffmpeg), sets `HF_HOME` (`/data/hf` with `--volume`, else `~/hf`) and writes the
`scripts/cuda_env_check.sh` report to `~/logs/env-check-<ts>.log` — the status entry's host line:
GPU name **with its edition**, driver, torch. Weights:
`HF_HOME=/data/hf ~/Difflet/.venv/bin/hf download <repo>` (check `df -h` first). FLUX.1-dev
needs the owner's device-code login once per `HF_HOME` (`scripts/hf_device_login.py`; recipe in
the runbook) — ask for it while the owner is awake, at the start.

## 3. Run the gates, detached and watched

Nothing long runs inside an open SSH session. Start it detached, then poll from the laptop:

```bash
ssh difflet-cuda-<topic> "cd ~/Difflet && setsid nohup bash $S/run_gates.sh wan auto \
    --out ~/logs/<topic>-wan > ~/logs/<topic>-wan.out 2>&1 < /dev/null &"
until ssh difflet-cuda-<topic> 'grep -q "GATES DONE" ~/logs/<topic>-wan/gates.txt'; do
  sleep 60; done                                     # run this loop in the background
ssh difflet-cuda-<topic> 'cat ~/logs/<topic>-wan/gates.txt'
```

`run_gates.sh <qwen|flux|wan> <snapshot|auto>` runs smoke → parity → parity-gate → e2e →
e2e-gate with `DIFFLET_BACKEND=cuda HF_HUB_OFFLINE=1`, one log per step under `--out`, one
`<step> exit: <rc> in <s>s` line each in `gates.txt`, and `GATES DONE <model> overall_rc=<n>`
last. A failed step does not stop the rest. `--dry-run` prints every command (use it in the
plan). Look at the smoke image/video **by eye** before trusting any number: it must be
prompt-faithful (every reference smoke was a cat in a sunlit garden).

After the per-model gates, as the scope says (each detached the same way):

- **CLI matrix**: `DIFFLET_BACKEND=cuda .venv/bin/python scripts/verify_cli_cuda.py --seed 0`
  — download → compile → timed generate per cell; all cells' weights must be in `HF_HOME`.
- **Serve** (Qwen-Image): `scripts/cuda_serve_smoke.sh --output ~/logs/qwen-serve.png --steps 50
  --requests 2`, then the probes in the catalog (two 400s, the concurrency 429, the
  `--tp-degree 2` refusal before download).
- **Two GPUs**: the serial reference with `CUDA_VISIBLE_DEVICES=0`, then `--cfg-parallel` (Wan)
  or `--dp 2 --requests requests.jsonl` (Qwen-Image); compare `.pt` / PNG outputs to the
  references (bit-identical in F2) and sample `nvidia-smi` on both cards during the run.

Record per run: load seconds, generate seconds, per-step seconds, peak GPU memory, the card
edition. A load from the network volume is a cold read (≈ 0.23 GB/s); say so next to the number.

## 4. Checkpoint, then release the pod

The pod can vanish (capacity, a container restart, the owner's stop), so nothing waits for the
end. After each model's gates:

1. Copy the evidence off the pod: `scp -r difflet-cuda-<topic>:~/logs/<topic>-wan <scratch>/`.
   Curate into `artifacts/cuda-release1/<feature>/`: result JSONs, `gates.txt`, smoke timings,
   PNGs / first frames / the short MP4. Never commit weights, compile caches or the parity
   `--reference` tensors.
2. Append a status-log entry to `docs/plans/2026-10-16-cuda-backend-support.md` (format in
   `difflet-cuda-backend`; append-only — corrections are new entries): pod name, id, region,
   price, **card edition**, driver, checkout, every number with its artifact path, rulings, next.
3. Commit on the session branch (artifacts + entry together).

At the end of the session, without asking:

```bash
python3 scripts/yottalabs_pod.py pause "$POD"     # volume pod; API code 12005 = no volume:
python3 scripts/yottalabs_pod.py delete "$POD"    # a pod without a volume can only be terminated
python3 scripts/yottalabs_pod.py status "$POD"    # read back PAUSED (or GONE after delete)
```

"I sent pause" is not done; a status read that says `PAUSED` / gone is. The final status goes in
the status entry and in the **first line** of the report. Never leave a pod running "for the next
session": the venv build and a volume read take minutes; idle hours bill.

## 5. When a gate fails: triage, then the bug protocol

Most first failures on a pod were not Difflet bugs. Check, in order (details and the fix for
each in `references/failure-triage.md`): the card (`nvidia-smi` "Failed to initialize NVML" —
device-cgroup loss), the container (PID 1 young, `/tmp` empty — a disk overrun restarted it),
disk and `HF_HOME` (a partial snapshot, the wrong cache), capacity (12014, INITIALIZING → PAUSED),
the environment (`.venv/bin/pip` absent under uv, no `/usr/bin/time`, no `nsys`), an expired HF
device code, a cold volume read mistaken for a slowdown, a per-step number compared across card
editions. Only then is it a Difflet bug:

1. Root-cause it with evidence (the traceback, the tensor shapes, the exact code line; a control
   run that isolates one factor, as F1b's timestep A/B).
2. Fix it on the Mac with a unit test that pins the repro; CPU gate green.
3. **One commit per bug**, message with *Problem* (symptom, pod, date, log) and *Solution*.
4. Ship the fix (`ship_branch.sh`) and re-run the exact gate that failed on the same pod.
5. A gate threshold is never loosened to pass. Floors move only from a measurement and in their
   own commit (F3/F5 tightened e2e from 25 dB / 0.99 to 35 dB / 0.998 with 5–6 dB headroom; F1
   kept Wan at 25 / 0.99 because 27.5 dB left 2.5 dB).
6. A hypothesis that one A/B does not support is written as "not supported by a single-seed
   A/B", not "refuted" (the F1b correction) — and the next suspect is measured, not assumed.

## 6. Report

Three deliverables, all from this session's evidence:

- **The status-log entry** (step 4) — the audit trail; first line names the pod's final status.
- **The evidence doc** `docs/verification/<topic>-evidence.md` (template in
  `references/evidence-doc-template.md`): per gate a table model → numbers → artifact path →
  status-log entry, a *How to inspect manually* recipe each, the rulings/bugs ledger with
  commits, the pod cost table. `git check-ignore -q <doc>` must fail (exit 1) before you commit
  — an ignored doc is a lost doc.
- **The support matrix**: PASS / LIMIT / N/A / NOT MEASURED, model × feature and one block per
  feature. NOT MEASURED is the honest gap list (on 2026-10-18: the CLI matrix's Wan cell,
  `--dp` × `--cfg-parallel`, Flux and Wan under `--dp`, Wan 2.1, the Wan e2e noise floor).

Be exact about what was and wasn't verified, and which card produced each number: Qwen-Image
ran 0.77 s/step on the Workstation Edition (us-east-3) and 0.33 s/step on the Server Edition
with the same code (F3b entry, 2.3×). Every path, number and explanation must come from this
session's JSONs, logs, or code you read; the numbers in these references describe other pods.
Label inferences ("likely the page cache — not measured") as inferences.
