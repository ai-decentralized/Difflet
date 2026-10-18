# Failure triage and the bug protocol

A session that mislabels pod noise as a Difflet bug wastes a fix cycle and the reader's trust;
one that writes off a real regression as noise ships it. Work through the spurious causes first,
in this order — each one happened on a reference pod (2026-10-16 → 10-18).

## Spurious causes (not Difflet bugs)

| Symptom | Cause | What to do |
|---|---|---|
| `nvidia-smi`: "Failed to initialize NVML: Unknown Error"; `open(/dev/nvidiactl)` EPERM even as root; the kernel driver still lists the card | the container lost its NVIDIA device-cgroup rules host-side (two us-central-7 pods, minutes after boot, 2026-10-17) | Nothing on the pod fixes it. Pause it (volume pod) or terminate it (no volume); terminating a volume pod needs the owner's "terminate"; a replacement pod needs the owner's yes (creating one is a deploy); `nvidia-smi` right after SSH and again a few minutes later before building. Record the pod ids. |
| Run dir, logs or the HF download under `/tmp` gone; PID 1 young; `/home` and `/data` intact | the container restarted after a write past `containerVolumeInGb` (118 GB to `/tmp` on a 100 GB record, F1) | Not a bug. Re-download onto the volume or a bigger container disk; logs under `~/logs`. Copy artifacts off the pod after every gate. |
| `create` → code 12014 "Pod insufficient resources", or `create --wait-for-capacity` waiting for hours | pool full (all PRO 6000 pools for ~1 h on 2026-10-18); eu pools refuse every 1-GPU request | Wait with the watcher, or ask the owner for another region / a 2-GPU node. |
| `resume` returns 200, the pod goes INITIALIZING → PAUSED in seconds (`resume` exits 2) | no capacity in the paused pod's pool (us-east-3, F1b) | Not a pod fault. Retry later, or (owner's yes) a new pod elsewhere without the volume. |
| `pause` → API code 12005 | the pod has no network volume | `delete` instead; that is the only way to stop its billing. |
| `.venv/bin/pip: No such file or directory` | uv venvs ship no pip | `uv pip install --python .venv/bin/python …`. |
| `/usr/bin/time: not found`; `nsys: not found` | not in the pod image | Use the scripts' own timers / log timestamps; `torch.profiler` for timelines (F4). |
| Device login: no `LOGIN_OK` in the log, the token never appears | the HF device code expired (300 s) before the owner entered it (F9's first code, 2026-10-18) | Start `hf_device_login.py` again for a fresh code once the owner says they are at the phone; send the URL the moment it prints. |
| SSH session dies right after a `pkill -f …` | the pattern matched the SSH command's own shell | Kill by PID (`pgrep -af`), never `pkill -f` inside `ssh '…'`. |
| A gate's pytest SKIPs ("no recorded … result", "set DIFFLET_RUN_CUDA_NUMERICAL=1") | the result-JSON env var points elsewhere, or the opt-in is unset | A SKIP is not a PASS. Point the env var at the JSON (`run_gates.sh` does) and re-run the gate. |
| `HF_HUB_OFFLINE` errors / a snapshot without `transformer/` or Wan without `transformer_2/` | partial download or the wrong `HF_HOME` | Re-run `hf download` into the right `HF_HOME`; Difflet refuses a Wan 2.2 snapshot without `transformer_2/` by design. |
| First load or a first parity pass many times slower than the reference (Qwen `verify_cli` 276 s vs 38 s; Flux diffusers bf16 949 s; Wan load 703 s) | cold reads from the CephFS volume (≈ 0.23 GB/s) | Not a regression. Say "cold volume read" next to the number; compare warm loads only. |
| Per-step time ~2.3× the reference | a different card edition (Workstation vs Server, F3b) | Not a regression until compared on the same edition (`scripts/cuda_bench_dit.py`). |
| A second request on `difflet serve` gets 429 `queue_timeout` after 30 s | the default `--queue-timeout` is shorter than one 50-step request (F3b) | Documented behaviour; raise `--queue-timeout` for a queueing client. |
| `RuntimeWarning` from `runpy` in each torchrun rank | `difflet.cli.main` imported before `-m` runs it (F2) | Cosmetic, F2b. |

Rule of thumb: a failure that appears with **no** Difflet traceback (a vanished process, a
missing file, an HTTP error from the API) is infrastructure until proven otherwise; one with a
Difflet traceback, a shape mismatch or a gate miss on a healthy card is a candidate bug.

## When a numerical gate misses

- Re-read the JSON before anything else: which ratio, which case (`sdpa`, `masked`, `rounding`),
  which expert. Quote the numbers.
- Check that the control is sane (the control's own cosine and rel_l1 against fp32 near the
  reference values in the feature catalog). A broken control means the oracle side moved
  (diffusers version, snapshot revision), not Difflet.
- A gate threshold is never loosened to make a run pass. If a floor is wrong, move it from a
  measurement, in its own commit, with the headroom stated.
- One seed and one prompt are one sample. Before blaming a cause for an e2e gap, measure the
  metric's own noise floor (diffusers vs diffusers with a perturbation, several seeds) — the
  F1b A/B moved Wan's PSNR by 1.2 dB, and its "refuted" had to be corrected to "not supported by
  a single-seed A/B". F9 measured Wan's: diffusers vs its bf16-rounded timestep 23.8–28.6 dB over
  seeds 0–2, Difflet vs diffusers 26.3–28.9 dB — overlapping it, so the e2e number alone points to
  no Difflet defect (n = 3: it cannot show there is none). Since F9b the Wan gate is stated
  relative to that floor when the result carries it (the mean over seeds within 1.21 dB of the
  floor's mean, every seed within 1.21 dB of the floor's worst seed, cosine deviation within
  1.15×, a loose 20 dB / 0.98 floor): a relative-gate failure names the gate (A: the mean, B: a
  seed) and the floor numbers it was judged against — read those before blaming Difflet.

## The bug protocol (when it really is a bug)

1. **Root-cause with evidence.** The traceback, the tensor shapes and dtypes, the exact code
   line; a control run that changes one factor (F1b: the timestep dtype as an A/B knob,
   `--timestep-dtypes float32,model`). Read the code path end to end.
2. **Fix on the Mac with a unit test** pinning the repro; the CPU gate
   (`.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh`) green before shipping.
3. **One commit per bug.** Message shape:
   ```
   fix(cuda): one-line symptom

   Problem: what failed, on which pod/card/date, the log path, and the root cause.
   Solution: what changed and why it is safe (other backends, cache keys, gates).
   ```
   Never fold two bugs into one commit; never fold a floor change or docs into a fix.
4. **Re-verify on the pod**: `ship_branch.sh`, then the exact gate that failed, then the rest of
   that model's gates (a fix can move other numbers).
5. **Other backends.** A change in shared code (`difflet/models/**`, the orchestrators,
   `_torch_megatron`) must leave Trainium and TPU unchanged: their unit tests green, the change
   scoped by a per-component property where possible (F1b's `timestep_dtype`). If Trainium
   behaviour could move, say that a trn2 re-run is pending (D13).
6. **Record**: the status-log entry names the commit, the before/after numbers and the gate.
   A wrong claim in an earlier entry is corrected by a new entry, never by editing the old one.
7. **Constraints that are real** (a refused flag, a shape a kernel cannot take) get a fail-fast
   `NotImplementedError` / `ValueError` naming the follow-up (F1b, F2b, F5b, F6) before any
   load — users get a clear message in seconds, not after a 4-minute cold read.
