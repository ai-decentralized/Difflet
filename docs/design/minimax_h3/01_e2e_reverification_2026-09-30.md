# MiniMax-H3 — end-to-end re-verification on a fresh host (2026-09-30)

Re-run of the smallest shape verified during bring-up (`00_trainium_bringup_handoff.md`
§1, 256×448×124), on a **fresh** `trn2.3xlarge` with no weights or compile
caches, from the H3 commits rebased onto `main@0f9ef0f`. Purpose: prove the
merge candidate still works end to end before it lands on `main`.

**Verdict: PASS.** `difflet generate` produced a 5.2 s MP4 with a video
stream and a stereo audio stream, and the content matches the prompt.

## 1. Environment

| Item | Value |
| --- | --- |
| Host | `trn2.3xlarge`: 4 logical NeuronCores at LNC=2, 96 GB HBM, 124 GiB RAM, and a 64 GiB swapfile added for this run |
| Kernel / driver / runtime | 6.17.0-1019-aws / aws-neuronx-dkms 2.30.2.0 / aws-neuronx-runtime-lib 2.34.10.0 |
| Python env | `<repo>/.venv` built by `scripts/setup_env.sh` from `requirements-neuron.lock`. This DLAMI no longer ships `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` |
| Toolchain | torch 2.9.1, torch-neuronx 2.9.0.2.15, neuronx-cc 2.26.6360.0, NxD 0.19.28492, NxDI 0.10.18399, diffusers 0.38.0, transformers 4.57.6 — the same versions the bring-up used |
| Weights | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe`, registry download patterns, 135 GB on disk |
| Code | `main@0f9ef0f` + the six `minimax-h3` commits (clean rebase, no conflicts) |

Newer toolchain versions on the Neuron index at the time of the run:
neuronx-cc 2.27.5334.0 and libneuronxla 3.0.5356. torch-neuronx, NxD and NxDI
are already at the newest release for torch 2.9. The DLAMI's newer stack
(torch 2.11 + `torch-neuronx-lite`, transformers 5.x) ships without NxDI or
torch_xla, both of which Difflet depends on. So moving to it is a port, not a
version bump. A/B-ing neuronx-cc 2.27 alone is an open follow-up (§6).

## 2. Command

```bash
difflet generate --model-id MiniMaxAI/MiniMax-H3 --tp-degree 4 \
  --height 256 --width 448 --num-frames 124 --seed 42 \
  --prompt "A red fox trots through a snowy forest at dawn, its breath visible in the cold air, soft crunching footsteps in the snow" \
  --output fox_256x448x124.mp4 --work-dir <W> --keep-work-dir
```

This runs the production entry point: four sequential stage subprocesses,
NEURON_RT_NUM_CORES 4/4/1/1, and 30 steps (the default). The exact job script
is `artifacts/verification-2026-09-30-minimax-h3/job_e2e.sh`.

## 3. Results

| Stage | Compile | e2e wall (from work-dir file mtimes) |
| --- | --- | --- |
| `text` (Qwen3-VL L50, TP4) | ≈ 23 min, inferred from artifact mtimes 16:45 → 17:08 (log overwritten, see §5) | ≈ 8.7 min: mostly loading the 62 GiB presharded weights |
| `generate` (33B DiT, TP4) | 1713 s (HLO hit the Neuron compile cache left by the interrupted first attempt; build 808 s, then presharding) | ≈ 9.4 min: weight load 517 s, the rest is 30 denoising steps |
| `video_vae` (TP1, fp32) | 221 s | ≈ 2.1 min |
| `audio_vae` + AV mux (TP1, fp32) | 1145 s | ≈ 1 min |
| **Total `difflet generate`** | — | **1297 s**, rc=0 |

Output `fox_256x448x124.mp4` (sha256 `4f81f7c2…bba375`, 284 KB):

- video: h264, 448×256, 124 frames at 24 fps, 5.17 s
- audio: AAC stereo, 32 kHz, 5.15 s; mean −58.0 dB, max −34.4 dB. Not silent
  (digital silence is about −91 dB). Quiet, which is consistent with
  "soft footsteps".
- content (`frames_grid.png`, frames 0/40/80/123): a red fox walking through a
  snowy forest in low dawn light, with visible breath vapour in frame 0. It is
  temporally coherent, with no corruption or tiling seams.

## 4. Unit tests

H3-scoped suites on the rebased branch pass: 59 passed
(`tests/unit/models/minimax_h3`, `tests/unit/cli/test_orchestrator_minimax_h3.py`,
`tests/unit/registry/test_minimax_h3_registration.py`, `tests/unit/test_registry.py`).
The full `tests/unit` suite was not run to completion for this verification
(see §5.3).

## 5. Operational notes from this run

1. The first compile attempt was killed from outside, not by a failure: the
   harness task that parented it was reaped, and `setsid` did not shield the
   process tree. Running the job as a transient **systemd service**
   (`sudo systemd-run --unit=… --uid=ubuntu -p MemoryMax=118G -p MemorySwapMax=55G`)
   decouples it completely. Use that for any job longer than a few minutes.
   The killed attempt left a stale `model.hlo_module.pb.lock` in
   `/var/tmp/neuron-compile-cache`, which was removed before the retry
   (handoff §6.2).
2. `difflet compile` recompiles every stage unconditionally, text included
   (about 23 min). After the interruption, only the missing stages were
   rebuilt, with `python -m difflet.cli.stage … --stage-mode compile`
   (same arguments as the orchestrator).
3. Running `pytest -n 16 tests/unit` on a Trainium host is not device-safe.
   Some tests import torch_xla, which initializes the Neuron runtime, and
   workers then fight over cores (`nrt_allocate_neuron_cores … cores busy`).
   One worker held the device until it was killed. Run the unit suite
   serially, or with the device free.
4. Host memory was never the constraint at this shape. Peak observed during
   presharding was about 66 GB used, and swap was untouched.

## 6. Follow-ups

- A/B neuronx-cc 2.27.5334 in a separate venv (DiT compile, output parity,
  per-step latency, HBM report). It may also move the 768×1344 HBM verdict
  (handoff §4).
- Log per-step denoise latency in the `generate` stage. Only the stage wall
  time is recoverable today.
- Handoff §9 items remain open. This run does not change them.

## 7. Evidence

`artifacts/verification-2026-09-30-minimax-h3/`: the MP4, `frames_grid.png`,
the per-stage compile logs, `e2e_generate.log`, `stage_results.txt` (rc and wall
time per step) and `job_e2e.sh`.
