# Gotchas (each cost real time on a reference pod, 2026-10-16 → 10-18)

## Pods and billing
- A pod bills by the hour whether or not anything runs. On 2026-10-17 a session finished, then
  waited for approval to pause, and the pod billed all night. At the end, pausing a volume pod,
  or terminating a pod that cannot be paused (12005, no volume), is pre-authorised: do it, read
  the status back, report it on the first line. Terminating a volume pod needs the owner's
  word "terminate"; deploying (`create`, `resume`) needs the owner's yes.
- A pod without a network volume cannot be paused (code 12005); only `delete` stops it.
- The eu PRO 6000 pools sell whole 2-GPU nodes ($5.30/h on 2026-10-18): a 1-GPU job there costs
  two cards. The us-east-3 pool ($0.45/GPU/h) can be full for hours, and a paused pod there
  then cannot resume (INITIALIZING → PAUSED).
- The console lists pods created by other logins under the same organization. Touch only pods
  with a `difflet-` name created by the owner's login.
- Prices differ per region and edition; quote what `yottalabs_pod.py gpus` shows today.

## Card editions
- RTX PRO 6000 Blackwell **Workstation** Edition (us-east-3) ran Qwen-Image at 0.77 s/step,
  the **Server** Edition (us-central-7, eu-3) at 0.33 s/step — same driver family, torch and
  code (F3b). Name the edition next to every timing, and compare timings on one edition only.
- Numerics did not move with the edition: Wan's bf16-timestep e2e was 27.54 dB on both (F1b).

## Disk
- `containerVolumeInGb` (the pod record) is the container's write limit, not what `df` shows:
  `/tmp` reported 8.6 TB free on a 100 GB record. Overrunning it restarted the container and
  wiped `/tmp` (F1); `/home` and `/data` survived.
- Wan 2.2's diffusers repo is 126 GB of fp32 shards (118 GB on disk), not the ~70 GB the bf16
  weights suggest; it does not fit on the 200 GB volume beside Qwen-Image and FLUX.1-dev.
- `verify_cli_cuda.py` has no cell filter: it needs every model's weights at once.
- The volume mounts root-owned; `sudo chown user:user /data` once (`pod_setup.sh --volume`).
- Cold volume reads (≈ 0.23 GB/s) make first loads look like regressions (Wan 703 s, Qwen
  `verify_cli` generate 276 s); a second load in the same session is warm.

## Shell and SSH
- `pkill -f <pattern>` inside `ssh '…'` kills the SSH session itself (the pattern matches its
  shell's command line). Kill by PID.
- Long runs: `setsid nohup … < /dev/null &` and a background `until ssh … grep -q DONE …; do
  sleep 60; done` poll from the laptop. An open SSH session is not supervision.
- Logs under `~/logs`. Several repo scripts still write `/tmp/logs/…` by default (the smoke
  scripts, the gates' default JSON paths); pass `--out` / `CUDA_SMOKE_LOGDIR` or copy
  them off before anything can restart the container.
- `run_gates.sh … auto` picks the first snapshot of that model under `HF_HOME`; pass the
  snapshot path when two revisions are cached.

## Environment on the pod
- uv venv: no `.venv/bin/pip` → `uv pip install --python .venv/bin/python …`.
- No `/usr/bin/time`, no `nsys` in the image (`ncu` present, unused).
- Torch for Blackwell (sm_120) needs a CUDA 12.8+ wheel: torch 2.11.0 from the cu128 index.
- Set `DIFFLET_BACKEND` explicitly: `cuda` on the pod (the gate scripts do), `cpu` on the
  laptop — with no accelerator the registry falls back to Trainium.
- `HF_HUB_OFFLINE=1` after the download keeps a run from re-resolving `main` mid-session.
- Gated repos: the token lives in `HF_HOME`. On a container disk it dies with the pod (f3c,
  terminated 2026-10-17 mid-download); on the volume it stays (only f5b has read it; another
  pod reading it is not measured).

## Evidence
- Each numerical gate SKIPs, not fails, when its JSON is missing or the opt-in
  `DIFFLET_RUN_CUDA_NUMERICAL=1` is unset. Read the pytest summary for `skipped`.
- Comparing two-GPU outputs: compare `.pt` tensors (F2's videos), not MP4s — the codec adds
  noise. Pickled `.pt` files can differ at the byte level while the tensors are identical.
- The F1b artifact PNG names differ from the script's output names (mapping in the F1b
  correction entry); cite what is in `artifacts/`, check with `ls`.
- `docs/` is gitignored with exceptions; `git check-ignore -q <path>` must fail (exit 1) before
  you rely on a commit containing a doc.
