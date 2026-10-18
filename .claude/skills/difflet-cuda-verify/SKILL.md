---
name: difflet-cuda-verify
description: >-
  Run a GPU verification session for Difflet's CUDA backend on a Yottalabs RTX PRO 6000 pod and
  leave auditable evidence. Use this whenever the user wants to verify, re-verify, benchmark or
  "prove on the GPU" Qwen-Image, FLUX.1-dev or Wan 2.2 on CUDA — smoke (`difflet run`), DiT parity
  vs diffusers, e2e vs diffusers, the CUDA CLI matrix, the `difflet serve` smoke, two-GPU `--dp` /
  Wan `--cfg-parallel` — including requests phrased as "run the GPU gates", "deploy a pod and check
  this branch", "does Wan still pass on CUDA", "resume the pod", "get a 2-GPU node", "re-run parity
  after this change". Also use it to plan a session without a GPU, to triage a failed gate (pod
  infrastructure vs a Difflet bug), and to write the evidence report from artifacts/cuda-release1/.
  It carries the pod lifecycle through scripts/yottalabs_pod.py (deploying needs the owner's yes; at
  the end, pausing a volume pod or terminating one that cannot be paused is pre-authorised).
---

# Difflet CUDA verification sessions

A session turns "this branch should work on CUDA" into "it ran on an RTX PRO 6000; here are the
JSONs, the images, the timings and the exact commands" — or into a root-caused bug with its own
commit. Distilled from Release 1 → F9 (2026-10-16 → 10-18, nine pods; the status log of
`docs/plans/2026-10-16-cuda-backend-support.md`). Every number here is that campaign's, labelled
with date, pod and card edition: what a pass looked like, never a fact about the next run.
References: `feature-catalog.md` (gates, thresholds, numbers), `pod-runbook.md` (before creating
a pod), `failure-triage.md` (before calling a bug), `evidence-doc-template.md`, `gotchas.md`.
`.claude/skills/difflet-cuda-backend/SKILL.md` (D1–D19, the CPU gate, the pod rule) binds too.

## 1. Scope the session with the owner

Pin down only what the request leaves open:

- **Models**: `qwen` = `Qwen/Qwen-Image`, `flux` = `black-forest-labs/FLUX.1-dev` (gated: the
  owner's HF login), `wan` = `Wan-AI/Wan2.2-T2V-A14B-Diffusers`. Wan 2.1 never ran on a GPU.
- **Gates**, per model in this order: smoke → DiT parity → parity gate → e2e → e2e gate
  (`run_gates.sh` runs all five); then the CLI matrix (`scripts/verify_cli_cuda.py`, every cell
  at once — it has no cell filter), the serve smoke (Qwen-Image only), two GPUs (`--dp 2`, Wan
  `--cfg-parallel`; a 2-GPU pod). `scripts/cuda_bench_dit.py` only when the change is about
  speed, and only compared on the same card edition.
- **Branch** (CPU gate green first); evidence in `artifacts/cuda-release1/<feature>/`.
- **Pod**: 1 GPU in us-east-3 with the `difflet-cache` volume (cheapest; the pool can be full for
  hours), or an eu pod without a volume (terminate at the end; 1 GPU accepted in eu-3, F9). The
  volume is held by `difflet-cuda-f5b` (ID 502399632095187214, PAUSED): resume it (owner's yes)
  or create a new us-east-3 pod on it — a second pod on the volume while f5b holds it, and the
  volume outliving f5b, are not measured; if `create` rejects the volume, ask the owner. Disk:
  Qwen + Flux live on the volume, Wan (118 GB) not beside both; F9's 250 GB eu disk held all three.

State the plan back — expected outcome per gate (the catalog's snapshot), commands, pod and its
price as `gpus` shows it now, disk plan — and ask for the deploy yes. Planning-only stops here.

## 2. Get a pod and prepare the host

**Deploying (create, resume) needs the owner's explicit yes. At the end of the session, pausing a
volume pod — or terminating a pod that cannot be paused (no volume) — is pre-authorised: do it
without asking. Terminating a volume pod needs the owner's word "terminate".** Touch only pods
created by the owner's login and named `difflet-…`. All calls go through `scripts/yottalabs_pod.py`
(key `~/.config/yottalabs/api_key`, mode 600, never printed; exit 0 ok, 1 API error, 2 no
capacity or timeout):

```bash
python3 scripts/yottalabs_pod.py gpus                  # pools: status AVAILABLE | UNAVAILABLE
python3 scripts/yottalabs_pod.py create --name difflet-<topic> --gpus 1 --regions us-east-3 \
    --volume-id 469225541779435520 --mount /data --dry-run        # show it; on the owner's yes:
POD=$(python3 scripts/yottalabs_pod.py create --name difflet-<topic> --gpus 1 \
    --regions us-east-3 --volume-id 469225541779435520 --mount /data --wait-for-capacity)
python3 scripts/yottalabs_pod.py wait-running "$POD"               # prints "host port"
python3 scripts/yottalabs_pod.py ssh-config "$POD" --alias difflet-cuda-<topic> >> ~/.ssh/config
```

Pin `--regions us-east-3` whenever `--volume-id` is given: the volume mounts only there; an eu
fallback would be a volume-less pod at $2.65/GPU/h where `pod_setup.sh --volume` fails. Note the
pod id the moment it prints. Use an alias new to `~/.ssh/config` (ssh takes the first matching
`Host`; delete a stale block). `--wait-for-capacity` polls every 300 s and creates once (F9 ran
it live). `--disk-gb` is sent beside a volume too (F9). No volume: `--gpus 1 --regions eu-3,eu-2
--disk-gb 250` (F9) or `--gpus 2 … --disk-gb 150` (F1b, F2). `resume ID` exits 2 when the pod
falls back INITIALIZING → PAUSED: no capacity, not a bug.

Then ship the branch and build the host (the F1b/F2 recipe):

```bash
S=.claude/skills/difflet-cuda-verify/scripts
REPO=$PWD bash $S/ship_branch.sh difflet-cuda-<topic> <branch>    # laptop → ~/Difflet on the pod
ssh difflet-cuda-<topic> nvidia-smi        # now, and again a few minutes later (NVML loss, triage)
ssh difflet-cuda-<topic> "mkdir -p ~/logs && setsid nohup bash ~/Difflet/$S/pod_setup.sh \
    --volume > ~/logs/setup.log 2>&1 < /dev/null &"      # any cwd; no volume: drop --volume
until ssh difflet-cuda-<topic> 'grep -q "SETUP DONE" ~/logs/setup.log'; do sleep 30; done
```

`pod_setup.sh` builds the uv venv (torch 2.11.0 cu128, `-e .[test,cuda]`, the `hf` CLI,
imageio-ffmpeg), sets `HF_HOME` (`/data/hf` with `--volume`, else `~/hf`), writes the
`scripts/cuda_env_check.sh` report to `~/logs/env-check-<ts>.log` — the status entry's host
line: GPU name **with its edition**, driver, torch — and ends `SETUP DONE rc=<n>` (0 = ok).
Weights: `HF_HOME=/data/hf ~/Difflet/.venv/bin/hf download <repo>` (check `df -h` first). FLUX.1-dev
needs the owner's device-code login once per `HF_HOME` (`scripts/hf_device_login.py`; recipe in
the runbook): ask at the start; run the script only once the owner is at the phone (300 s codes).

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
last. A failed step does not stop the rest. `--dry-run` prints every command (planning: pass an
explicit snapshot path; `auto` needs a cached one). Look at the smoke output **by eye** before
trusting a number: it must be prompt-faithful (every reference smoke was a cat in a garden).
Its Wan e2e is one seed without the noise floor; the multi-seed run with it is in the catalog.

Then, as scoped (detached the same way), with `HF_HOME` set in every command — `H=/data/hf`, or
`H=~/hf` without the volume (a non-interactive shell may skip `~/.bashrc`; a fallback cache on
the container disk is the F1 overrun):

```bash
HF_HOME=$H DIFFLET_BACKEND=cuda .venv/bin/python scripts/verify_cli_cuda.py --seed 0  # CLI matrix
HF_HOME=$H scripts/cuda_serve_smoke.sh --output ~/logs/qwen-serve.png --steps 50 --requests 2
# probes, against a running `HF_HOME=$H DIFFLET_BACKEND=cuda .venv/bin/difflet serve
#   --model-id Qwen/Qwen-Image` (port 8091); F3b's answers on the right
U=http://127.0.0.1:8091/v1/chat/completions
post() { curl -s -o ~/logs/probe-$1.json -w "$1 %{http_code}\n" -d "{\"model\": \
  \"Qwen/Qwen-Image\", \"messages\": [{\"role\": \"user\", \"content\": \"a cat\"}]$2}" \
  -H 'Content-Type: application/json' $U; }
post g ', "extra_body": {"guidance_scale": 1.0}'   # 400 feature_not_supported
post h ', "extra_body": {"height": 768}'           # 400 profile_mismatch
post c1 & post c2; wait          # one 200, one 429 queue_timeout after the 30 s default
```

Two GPUs (F2): a serial reference on one card, then both; compare `.pt` tensors / PNGs to the
references (bit-identical in F2), `nvidia-smi` sampled on both cards meanwhile:

```bash
export HF_HOME=$H HF_HUB_OFFLINE=1 DIFFLET_BACKEND=cuda      # H=~/hf on the eu pod (no volume)
P="a cat walking through a sunlit garden, photorealistic"
W="--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --steps 40 --num-frames 9 --seed 0"
CUDA_VISIBLE_DEVICES=0 .venv/bin/difflet run $W --prompt "$P" --output ~/logs/<name>/wan_serial.pt
.venv/bin/difflet run $W --prompt "$P" --cfg-parallel --output ~/logs/<name>/wan_cfg.pt  # 2 ranks
for s in 0 1; do printf '{"prompt": "a cat in a sunlit garden", "output": "%s", "seed": %d}\n' \
  "$HOME/logs/<name>/q$s.png" $s; done > ~/logs/<name>/requests.jsonl
.venv/bin/difflet run --model-id Qwen/Qwen-Image --dp 2 --requests ~/logs/<name>/requests.jsonl
```

Record per run: load, generate and per-step seconds, peak GPU memory, the card edition; a load
from the network volume is a cold read (≈ 0.23 GB/s) — say so next to the number.

## 4. Checkpoint, then release the pod

The pod can vanish (capacity, a container restart, the owner's stop), so nothing waits for the
end. After each model's gates:

1. Copy the evidence off the pod: `scp -r difflet-cuda-<topic>:~/logs/<topic>-wan <scratch>/`.
   Curate into `artifacts/cuda-release1/<feature>/`: result JSONs, `gates.txt`, smoke timings,
   PNGs / first frames / the short MP4 — never weights, compile caches or `--reference` tensors.
2. Append a status-log entry to `docs/plans/2026-10-16-cuda-backend-support.md` (format in
   `difflet-cuda-backend`; append-only — corrections are new entries): pod name, id, region,
   price, **card edition**, driver, checkout, every number with its artifact path, rulings, next.
3. Commit on the session branch (artifacts + entry together).

At the end of the session, without asking — one of the two, never both:

```bash
python3 scripts/yottalabs_pod.py pause "$POD"    # a pod WITH the volume → read back PAUSED
python3 scripts/yottalabs_pod.py delete "$POD"   # a pod WITHOUT one (pause → 12005) → gone
python3 scripts/yottalabs_pod.py status "$POD"   # the read-back
```

"I sent pause" is not done; a read-back of `PAUSED` / gone is — it goes in the status entry and
the report's **first line**. Never keep a pod running "for the next session": idle hours bill.

## 5. When a gate fails: triage, then the bug protocol

Most first failures on a pod were not Difflet bugs. Rule out, in order (`failure-triage.md`):
NVML loss, a container restart after a disk overrun, a partial snapshot or wrong `HF_HOME`,
capacity (12014, INITIALIZING → PAUSED), the image (no pip under uv, no `/usr/bin/time`/`nsys`),
an expired HF code, a cold volume read, a per-step number from another card edition. Then:

1. Root-cause with evidence (traceback, shapes, the code line; a one-factor control run, as
   F1b's timestep A/B). 2. Fix on the Mac with a unit test pinning the repro; CPU gate green.
3. **One commit per bug** (*Problem*: symptom, pod, date, log; *Solution*). 4. `ship_branch.sh`
   the fix and re-run the exact gate that failed on the same pod.
5. Never loosen a gate to pass; floors move only from a measurement, in their own commit (F3/F5:
   25 → 35 dB with 5–6 dB headroom; F1 kept Wan at 25 / 0.99; F9b: Wan relative to its noise floor).
6. One unsupportive A/B reads "not supported by a single-seed A/B", not "refuted" (F1b).

## 6. Report

From this session's evidence: **the status-log entry** (step 4); **the evidence doc**
`docs/verification/<topic>-evidence.md` (the template: per gate model → numbers → artifact path →
entry, *How to inspect manually*, ledger with commits, pod costs; `git check-ignore -q <doc>`
must exit 1 before you commit — an ignored doc is a lost doc); **the support matrix**: PASS /
LIMIT / N/A / NOT MEASURED, model × feature and one block per feature; NOT MEASURED is the gap
list (after F9, four: Wan `--dp` × `--cfg-parallel`, Flux and Wan under `--dp`, Wan 2.1, serving
on two GPUs). Be exact about what was and wasn't verified, and which card edition produced each
number (Qwen 0.77 s/step Workstation vs 0.33 Server, F3b). Every path, number and explanation
comes from this session's JSONs, logs, or code you read — these references describe other pods.
Label inferences ("likely the page cache — not measured") as inferences.
