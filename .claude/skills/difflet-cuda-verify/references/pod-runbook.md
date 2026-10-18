# Pod runbook — Yottalabs facts and the exact steps

What the reference sessions learned about Yottalabs pods (2026-10-16 → 10-18). Prices,
capacity and limits are what the API showed on those dates; read `gpus` again before quoting
any of them. The binding rule: deploying (create, resume) needs the owner's yes; pause or
terminate at the end of a session is pre-authorised; only pods created by the owner's login
with a `difflet-` name are ours (the console lists other logins' pods beside them).

## The API and the key

- Base `https://api.yottalabs.ai/v2`, header `X-Api-Key`. The key is an Access Key (Console →
  Settings → Access Keys) the owner creates once, stored at `~/.config/yottalabs/api_key`, mode
  600. Never print it, never commit it, never pass it on a command line. If the file is missing,
  ask for it at the **start** of the session, while the owner is awake.
- `scripts/yottalabs_pod.py` wraps every call the sessions used (`GET /v2/gpus`,
  `POST /v2/pods`, `GET /v2/pods/{id}`, `GET /v2/pods`, `POST …/pause|resume`,
  `DELETE /v2/pods/{id}`). Exit 0 ok, 1 API error or FAILED pod, 2 no capacity or timeout.
  `create --dry-run` prints the request body and sends nothing.

## Pools seen for `NVIDIA_RTX_PRO_6000_96G` (2026-10-18)

| Region | $/GPU/h | Card | Container disk limit | Notes |
|---|---|---|---|---|
| us-east-3 | 0.45 | Workstation Edition | 178 GB | the only pool where `difflet-cache` mounts |
| eu-3 | 2.65 | Server Edition | 500 GB | sells whole 2-GPU nodes ($5.30/h) |
| eu-2 | 2.65 | Server Edition | 500 GB | sells whole 2-GPU nodes |
| eu-1 | 2.55 | not used | — | seen UNAVAILABLE |
| us-central-7 | 1.35/h per pod (10-16, 10-17) | Server Edition | — | F3/F4 pods; device-cgroup incident |

- `GET /v2/gpus` (`yottalabs_pod.py gpus`) shows each pool's `status AVAILABLE | UNAVAILABLE`.
  The flag can lag: a create into an AVAILABLE pool can still return code **12014** ("Pod
  insufficient resources"); the capacity watcher keeps waiting on 12014.
- In eu-3 / eu-2 every 1-GPU request (with and without an SSH expose, 150–200 GB disk) returned
  12014; a 2-GPU request succeeded. A 1-GPU job there means renting both cards [F1b ledger].
- All PRO 6000 pools were full for about an hour on 2026-10-18; F2's pod was created by
  `create --wait-for-capacity` (one GET per 300 s round, one POST when a pool opens) [F2].
- The us-east-3 pool can be full for hours: a paused pod there may not resume on demand.

## Pause, resume, terminate

- **Pausing releases the card and stops GPU billing**; the container disk is lost, `/data`
  (the volume) and the pod record stay. `pause` polls until `PAUSED`.
- **A pod without a network volume cannot be paused** (API code **12005**): terminate it
  (`delete`, which polls until the pod is gone). F1b's and F2's eu pods were terminated.
- **Resume needs free capacity**: on 2026-10-18 three `POST …/resume` calls to the paused
  `difflet-cuda-f5b` returned 200, then the pod went INITIALIZING → PAUSED within ~10 s each
  time; `gpus` showed us-east-3 UNAVAILABLE. `resume` exits 2 on that pattern. It is not a bug.
- Cost reference: `difflet-cuda-f2` (2× Server Edition, eu-3, $5.30/h) ran 29 min, ~$2.60.

## Storage

| Where | Size / speed (reference) | Survives | Use for |
|---|---|---|---|
| `difflet-cache` volume at `/data` | 200 GB CephFS, $0.0698/GB/month; cold reads ≈ 0.23 GB/s | pause, terminate, new pods (us-east-3 only) | HF cache + HF token (`HF_HOME=/data/hf`) |
| container disk (`containerVolumeInGb`) | 100 GB on f5b; 150 GB on the eu pods | nothing — lost on pause/terminate, wiped by a restart | venv, eu-pod downloads |
| `/home` | — | the F1 container restart (pod still alive) | `~/Difflet`, `~/logs` |

- The volume: id **469225541779435520**, us-east-3, mounted by `persistentVolumes`
  (`create --volume-id 469225541779435520 --mount /data`). It mounts **root-owned** the first
  time: `sudo chown user:user /data` once (`pod_setup.sh --volume` does it).
- What it holds (2026-10-18, after F1): Qwen-Image (57.7 GB) and FLUX.1-dev (33.7 GB), 116 GB
  free, plus the HF token. Wan 2.2 (118 GB on disk) does **not** fit beside both. To run Wan
  there: `HF_HOME=/data/hf .venv/bin/hf cache rm model/Qwen/Qwen-Image -y`, download Wan
  (~1 GB/s: 118 GB in 108–109 s), and swap back after (the F1 session did exactly that).
- Cold reads from the volume dominate first loads: Qwen 54 GB ≈ 240 s, the Flux DiT 24 GB ≈
  200 s, Wan's load 703 s, a 57 GB expert ≈ 4 min [F5, F1]. Warm loads match local disk.
- **The container disk limit is a hard write limit.** Writing Wan's 118 GB to `/tmp` on f5b
  (record `containerVolumeInGb` 100) was followed by a container restart that wiped `/tmp`
  (logs, the download); `/home` and `/data` survived [F1]. Large downloads go on the volume, or
  on a pod created with a bigger `--disk-gb`. Keep logs under `~/logs`, never `/tmp`.
- `df` does not show that limit: f5b's `/tmp` showed 8.6 TB free under a 100 GB record [F1
  plan], the F1b pod a 968 GB local disk under 150 GB [F1b ledger]. Read the pod record's
  `containerVolumeInGb` (`yottalabs_pod.py status ID`) and budget against it.
- Growing the volume (≥ 350 GB would hold all three models) is the owner's call [F1 plan].

## SSH and long runs

- The account key `~/.ssh/yottalabs_difflet_cuda_qwen.pem` works on every pod.
  `yottalabs_pod.py ssh-config ID --alias difflet-cuda-<name> >> ~/.ssh/config` writes the Host
  block from the pod's SSH expose (proxy host and port); before the endpoint exists it exits 2.
- Ship code with `ship_branch.sh <alias> <branch>` (a full-history git bundle; the pod needs
  no GitHub access).
- Right after the first SSH: `nvidia-smi`, and again a few minutes later, before building
  anything (two us-central-7 pods lost the card within minutes of boot, 2026-10-17).
- Long runs: `setsid nohup bash -c "…" > ~/logs/<name>-wrapper.log 2>&1 < /dev/null &`, then a
  background poll from the laptop:
  `until ssh <alias> 'grep -q "DONE" ~/logs/<name>/gates.txt'; do sleep 60; done`.
  Never hold an SSH session open for a run; a dropped connection must not cost the run.
- Pass `HF_HOME` explicitly in every non-interactive ssh command (as the F1b runbook did), e.g.
  `ssh <alias> 'HF_HOME=/data/hf ~/Difflet/.venv/bin/hf download Qwen/Qwen-Image'`.

## Gated repos: the Hugging Face device-code login

FLUX.1-dev is gated (the licence is accepted on the owner's HF account). The token lives in
`HF_HOME`; on the volume it persists across pods, on a container disk it dies with the pod. When
`hf download black-forest-labs/FLUX.1-dev` says 401/gated:

```bash
scp .claude/skills/difflet-cuda-verify/scripts/hf_device_login.py <alias>:~/hf_device_login.py
ssh <alias> 'cd ~ && HF_HOME=/data/hf setsid nohup ~/Difflet/.venv/bin/python \
    hf_device_login.py > ~/logs/hf-login.log 2>&1 < /dev/null &'
ssh <alias> 'cat ~/logs/hf-login.log'     # OPEN: <url>   CODE: <code> expires_in_s: ...
```

Give the owner the `OPEN:` URL and the `CODE:`; they open it and enter the code within 300 s.
`LOGIN_OK user: …` in the log means the token is saved under `HF_HOME`. If the code expired,
start the script again (a fresh code). **Never `pkill -f hf_device_login.py` from inside an SSH
command**: the pattern matches the SSH session's own shell command line and kills it. List with
`pgrep -af hf_device_login`, take the `python … hf_device_login.py` line's PID, `kill <pid>`.

## Pod image facts (`yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`)

- On the Release 1 pod the image's own torch was a nightly (2.12.0.dev20260311+cu128); every
  session used a fresh uv venv with torch 2.11.0+cu128 instead.
- A uv venv has **no `.venv/bin/pip`**: `uv pip install --python .venv/bin/python …`.
- **No `/usr/bin/time`** (wall times come from the scripts' own timers and log timestamps);
  **no `nsys`** (F4 used `torch.profiler`; `ncu` exists but was unused).
- The eu pods had driver 580.178.04, the us pods 580.126.09.

## A session, start to finish

1. Owner's yes → `gpus`, `create … --dry-run`, `create` (watcher if the pool is full); id into
   the notes; `wait-running`; `ssh-config >> ~/.ssh/config`.
2. `nvidia-smi` twice; `ship_branch.sh`; `pod_setup.sh [--volume]`; downloads (disk checked).
3. Gates detached (`run_gates.sh`, then the matrix / serve / two-GPU runs); poll for `DONE`.
4. Copy evidence off (`scp -r <alias>:~/logs/<name> …`), curate into
   `artifacts/cuda-release1/<feature>/`, status-log entry, commit.
5. `pause` (volume) or `delete` (no volume); `status` read back; the final status in the
   entry and in the first line of the report.
