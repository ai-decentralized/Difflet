# bench4m: Qwen-Image native / AoT quality benchmark scripts

The scripts behind the Qwen-Image quality numbers: native (`--backend neuron`) TP4 at **32.88 dB
PSNR / 0.985 SSIM / 0.9957 latent cosine** against a CPU fp32 diffusers reference from the same
initial noise, and the AoT (`--backend trainium`) before/after runs of PR #134.

They were copied in unchanged. The only edits replace hard-coded host paths with the settings
below. Comments and docstrings still name the original host paths; treat those as history.

## Files

| File | What it does |
|---|---|
| `bench4m.py` | Harness. Runs one `difflet generate` / `compile` (`run`), a list of runs (`cell`), or prints the command and env without touching the device (`dry`). It waits for `/dev/neuron0` to be idle, records the HBM peak with neuron-monitor, and writes `<run>/summary.json` plus `summary.jsonl`. A run named `q*` is a quality run: it dumps `initial.pt` / `final.pt` latents and a lossless output. |
| `probe/sitecustomize.py` | Loaded into every run via PYTHONPATH. It records timings and `difflet.__file__`; `DV_INJECT_INITIAL=<initial.pt>` makes the run start from that noise. |
| `nmon.json`, `check_*.py` | neuron-monitor config and output checkers used by the harness. |
| `job_device.sh` | Native Qwen-Image device runs, before (`nfq_before/*`) and after (`nfq_after/*`) the text-mask fix, under a shared `flock`. |
| `job_cpu_ref.sh`, `cpu_ref_qwen.py`, `cpu_ref_common.py`, `cpu_only.py` | One CPU fp32 diffusers reference from a given `initial.pt` (one at a time; needs 112 GiB MemAvailable and 20 GB free disk). |
| `job_cpu_ref_queue_qwen.sh` | Queues the two Qwen references: `ref_aot_noise_50` (from `aot_main_tp4/q1`) and `ref_native_noise_50` (from `native_tp4_hashcheck/q1`). |
| `metrics.sh`, `quality_metrics.py` | Scores a candidate against a reference: PSNR/SSIM on frames, rel-L2/cosine on final latents. |
| `job_pr134.sh` | AoT TP4 before/after for PR #134: compile, gen0-2 and q1 for Qwen-Image, then the same for HunyuanVideo 1.0. |
| `job_qwen_native_noise.sh`, `score_native_noise.sh` | AoT Qwen-Image started from the native run's noise (`qn` runs), scored against `ref_native_noise_50` and against the bf16 floor. |
| `analyze_pr134.sh`, `frames_map.py` | Scores the PR #134 runs against `ref_aot_noise_50`. |
| `brief.py`, `step_stats.py`, `run_summary.py` | Summary printers for metrics JSONs, `steps.jsonl` and run dirs. |

## Settings

| Variable | Used by | Meaning |
|---|---|---|
| `BENCH4M_DIR` | all (required) | Data root. Run dirs are `<model>/<cell>/<run>/`; compiled roots are `roots/<model>_<cell>/`. |
| `BENCH4M_NATIVE_VENV` | bench4m.py, metrics.sh (required) | venv with torch-neuronx 2.14 (native backend). |
| `BENCH4M_AOT_VENV` | bench4m.py, CPU refs, AoT jobs, scoring (required) | venv with NxD / torch_xla / torch 2.9.1 (AoT backend). It also runs the CPU references. |
| `BENCH4M_NATIVE_WT`, `BENCH4M_AOT_WT` | bench4m.py | Difflet checkout each backend imports (default: this checkout). |
| `HF_HOME` | bench4m.py, queue | HF cache root (default `~/.cache/huggingface`). Qwen-Image is pinned to revision `75e0b4be04f60ec59a75f475837eced720f823b6`. |
| `QWEN_SNAPSHOT` | job_cpu_ref_queue_qwen.sh | Snapshot dir override. |
| `NFQ_AFTER_WT` (required), `NFQ_BEFORE_WT` | job_device.sh | Checkouts for `nfq_after` / `nfq_before` (before defaults to `$BENCH4M_DIR/nf_qwen_text/src_head`). |
| `NEURON_DEVICE_LOCK` | job_device.sh | Lock file shared by device users (default `~/neuron-device.lock`). |
| `PR134_AFTER_WT`, `PR134_BEFORE_WT` | job_pr134.sh, job_qwen_native_noise.sh (required) | AoT checkouts for `after_tp4` / `before_tp4`. |
| `NATIVE_NOISE` | job_qwen_native_noise.sh (required) | The native q1 run's `initial.pt`. |
| `BENCH4M_REF_DIR` | score_native_noise.sh, analyze_pr134.sh | Root holding `qwen_image/cpu_ref/` (default `BENCH4M_DIR`). |

The host also needs:

- `/usr/bin/python3`, which runs the harness (stdlib only);
- `/opt/aws/neuron/bin/neuron-monitor`;
- `flock`;
- passwordless `sudo`, but only for `cold` runs, which drop the page cache.

## Reproduce the native 32.88 dB

Run from the repo root on a trn2 host. The first generate into a new root compiles it.

```bash
export BENCH4M_DIR=/data/bench4m BENCH4M_NATIVE_VENV=/path/to/native-venv BENCH4M_AOT_VENV=/path/to/aot-venv

# 1. Device: native TP4, 1024x1024, 50 steps, seed 42, cfg 4.0, the fox prompt.
#    This writes qwen_image/nfq_after/q1/{initial.pt,final.pt,out.png}. Native noise is
#    deterministic: every native run draws the same initial.pt (sha256 95a4306b...).
NFQ_AFTER_WT=$PWD bash benchmark/bench4m/job_device.sh nfq_after/q1

# 2. CPU fp32 diffusers reference from that noise (about 1 h 10 min, peak RSS about 85 GiB).
#    Running it as a systemd --user unit keeps it alive after the shell exits.
systemd-run --user --unit=cpuref_qwen --collect -- bash "$PWD/benchmark/bench4m/job_cpu_ref.sh" \
  qwen_image ref_native_noise_50 -- \
  --model-path ~/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6 \
  --initial $BENCH4M_DIR/qwen_image/nfq_after/q1/initial.pt --steps 50

# 3. Score. Paths are relative to BENCH4M_DIR; this writes nf_qwen_text/metrics/<label>.json.
bash benchmark/bench4m/metrics.sh nfq_after_q1_vs_fp32 \
  qwen_image/cpu_ref/ref_native_noise_50/final.pt qwen_image/nfq_after/q1/final.pt \
  qwen_image/cpu_ref/ref_native_noise_50/out.pt   qwen_image/nfq_after/q1/out.png
# -> nfq_after_q1_vs_fp32.json cos 0.9957 rel 0.0949 {'psnr_mean': 32.881, 'ssim_mean': 0.985}
```

To see a run's exact command and env without using the device, run `bench4m.py dry` with that run's arguments, for example:

```bash
/usr/bin/python3 benchmark/bench4m/bench4m.py dry --model qwen_image --cell nfq_after --run q1 \
  --mode generate --venv native -- --backend neuron --tp-degree 4 --vae-dtype bf16
```

## AoT runs

```bash
export PR134_AFTER_WT=/path/to/after PR134_BEFORE_WT=/path/to/before
bash benchmark/bench4m/job_pr134.sh                       # compile + gen + q1, after and before
NATIVE_NOISE=$BENCH4M_DIR/qwen_image/nfq_after/q1/initial.pt \
  bash benchmark/bench4m/job_qwen_native_noise.sh          # AoT qn runs from the native noise
bash benchmark/bench4m/score_native_noise.sh               # vs ref_native_noise_50 (CPU only)
```

For `score_native_noise.sh`, also make the bf16 reference first, with the same `job_cpu_ref.sh` command as step 2:

- name it `ref_native_noise_50_bf16`;
- add `--dtype bfloat16 --vae-dtype float32`.

`score_native_noise.sh` also reads `quality_part1/qwen_native_vs_ref.json`, the native score from the four-model campaign. That file is listed last, so if it is missing, `brief.py` prints every other line and then fails on that path.

`analyze_pr134.sh` scores the q1 runs against `ref_aot_noise_50`. That reference comes from `job_cpu_ref_queue_qwen.sh`, which reads `aot_main_tp4/q1/initial.pt`.
