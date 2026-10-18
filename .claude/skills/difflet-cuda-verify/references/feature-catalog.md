# Feature catalog — each gate's script, inputs, thresholds and reference numbers

Every number below is from the reference campaign (Release 1 → F2, 2026-10-16 → 10-18) and
carries its date, pod and card edition. They show what a pass looked like on that card; they are
not expectations for the next run. Sources: the status log at the end of
`docs/plans/2026-10-16-cuda-backend-support.md` (entry date in brackets) and the JSONs under
`artifacts/cuda-release1/`. "Not measured" means no session ran it.

## Card editions — read before comparing any time

| Pod | Region | Card | $/h | Used for |
|---|---|---|---|---|
| `difflet-cuda-qwen` | us-central-7 | 1× RTX PRO 6000 Blackwell **Server Edition** | 1.35 | Release 1 [10-16] |
| `difflet-cuda-f3c` | us-central-7 | 1× **Server Edition** | 1.35 | F3, F4 [10-17] |
| `difflet-cuda-f5b` | us-east-3 | 1× **Workstation Edition**, `difflet-cache` volume | 0.45 | F5, F3b, F1 [10-18] |
| `difflet-cuda-f1b` | eu-3 | 2× **Server Edition** (GPU 0 used) | 5.30 | F1b [10-18] |
| `difflet-cuda-f2` | eu-3 | 2× **Server Edition**, PCIe `PHB`, no NVLink | 5.30 | F2 [10-18] |

Same driver family, torch 2.11.0+cu128 and code path everywhere, yet Qwen-Image's 50-step
`difflet run` took **0.77 s/step on the Workstation Edition** (f5b) and **0.33 s/step on the
Server Edition** (f3c, after F4's trim): a **2.3× per-step gap** from the card/pod [F3b, 10-18].
Never put per-step numbers from two editions in one comparison without naming both. Parity and
e2e metrics did match across editions: Wan's `model`-timestep e2e gave 27.54 dB on both [F1b].

## Support snapshot (2026-10-18; the default expectation per gate)

| Gate / feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B |
|---|---|---|---|
| smoke (`difflet run`, by eye) | PASS [F3] | PASS [F5] | PASS [F1] |
| DiT parity (D14, three gates) | PASS incl. padded prompt [F3, F4] | PASS [F5] | PASS both experts + rounding block [F1, F1b] |
| e2e vs diffusers | PASS, floors 35 dB / 0.998 [F3] | PASS, floors 35 dB / 0.998 [F5] | PASS, floors 25 dB / 0.99 [F1, F1b] |
| CLI matrix cell (`verify_cli_cuda.py`) | PASS [F3, F5] | PASS [F5] | NOT MEASURED (smoke stands in) [F1] |
| `difflet serve` | PASS [F3b] | refused by name (F5b) | refused by name (F1b) |
| `--torch-compile` | PASS, opt-in [F4] | refused (F5b) | refused (F1b) |
| `--teacache-cadence N` | PASS [F4] | refused (F5b) | refused (F1b) |
| `--dp 2 --requests` | PASS, byte-identical [F2] | NOT MEASURED (reachable) | NOT MEASURED (reachable) |
| `--cfg-parallel` (2 GPUs) | N/A (refused: single pass) | N/A (refused: guidance-distilled; true-CFG is the owner's call) | PASS, bit-identical [F2] |
| `--dp` × `--cfg-parallel` | N/A (refuses `--cfg-parallel`) | N/A (refuses `--cfg-parallel`) | NOT MEASURED (needs 4 cards) |
| CP / ulysses (`--cp-degree > 1`) | refused (F2b) | refused (F2b) | refused (F2b) |
| TP across PCIe, `--sp` | refused (F6) | refused (F6) | refused (F6) |

Anything landing differently in a new session is a finding — a regression, an upgrade, or a
NOT MEASURED cell finally measured. Say which.

## Env every GPU gate shares

`DIFFLET_BACKEND=cuda`, `PYTHONPATH=<repo>`, the venv's `bin/` first on `PATH`, `HF_HOME` where
the snapshot lives, `HF_HUB_OFFLINE=1` after the download. `run_gates.sh` sets all of them.
The numerical gates skip unless `DIFFLET_RUN_CUDA_NUMERICAL=1`, and read their JSON from an env
var (defaults under `/tmp/logs/`):

| Model | parity script → JSON env | e2e script → JSON env |
|---|---|---|
| Qwen | `tests/numerical/cuda_parity_gpu.py` → `DIFFLET_CUDA_PARITY_RESULT` | `scripts/cuda_e2e_vs_diffusers.py` → `DIFFLET_CUDA_E2E_RESULT` |
| Flux | `tests/numerical/cuda_parity_flux_gpu.py` → `DIFFLET_CUDA_FLUX_PARITY_RESULT` | `scripts/cuda_e2e_flux_vs_diffusers.py` → `DIFFLET_CUDA_FLUX_E2E_RESULT` |
| Wan | `tests/numerical/cuda_parity_wan_gpu.py` → `DIFFLET_CUDA_WAN_PARITY_RESULT` | `scripts/cuda_e2e_wan_vs_diffusers.py` → `DIFFLET_CUDA_WAN_E2E_RESULT` |

Gates: `tests/numerical/test_cuda_vs_diffusers.py`, `test_cuda_flux_vs_diffusers.py`,
`test_cuda_wan_vs_diffusers.py` (parity); `test_cuda_e2e_vs_diffusers.py`,
`test_cuda_flux_e2e_vs_diffusers.py`, `test_cuda_wan_e2e_vs_diffusers.py` (e2e).

## The DiT parity gate (D14)

One GPU script runs three forwards on identical seeded inputs — diffusers fp32 (reference),
diffusers bf16 (the **control**), Difflet bf16 through the CUDA ops — fp32 first and freed
before the bf16 passes. The gate is relative to what bf16 alone costs upstream (an absolute
0.999 cosine is out of reach for bf16 itself), constants in each `test_cuda_*_vs_diffusers.py`:

1. Difflet's rel_l1 vs fp32 ≤ **1.15×** the control's rel_l1 vs fp32.
2. Difflet's rel_l1 vs the bf16 control ("same-dtype distance") ≤ **1.2×** the control's
   rel_l1 vs fp32 — Difflet sits inside the noise bf16 creates.
3. Difflet's cosine vs the bf16 control ≥ **0.995** (a break, not rounding).

Qwen adds a padded-prompt case (16 valid of 1024 text tokens, `results.masked` /
`control_masked`); Wan runs per expert plus a `rounding` block at timesteps 999 / 750 (bf16
holds them as 1000 / 752; diffusers is fed the int64 timestep as `WanPipeline` does).

## The e2e gate

Prompt in, image or video out, from the **same initial noise** through Difflet and stock
diffusers (`QwenImagePipeline`, `FluxPipeline`, `WanPipeline`), one side loaded at a time;
pixel metrics in [0, 1]. Floors: Qwen and Flux **PSNR ≥ 35 dB, cosine ≥ 0.998** (tightened
from 25 / 0.99 after the first measurement, F3 and F5); Wan **≥ 25 dB, ≥ 0.99** (kept: the
measurement left 2.5 dB, F1). The Wan gate judges the `float32` timestep variant by name.

## Qwen-Image (`Qwen/Qwen-Image`, public)

Snapshot 54 GB (57.7 GB in the volume's cache listing), downloaded in ~90 s on us-central-7
[10-16, 10-17]. Defaults: 1024×1024, 50 steps, guidance 4.0 fixed (`guidance_embeds=False`, so
`--guidance-scale` is refused). Smoke `scripts/cuda_smoke.sh --steps 50 --seed 0` (CLI) or
`scripts/cuda_smoke.py --steps 50 --local-files-only` (phase timings, peak memory).

| Measurement | Value | Pod, edition [entry] |
|---|---|---|
| 50 steps, no mask (Release 1) | load 10.3 s, 20.5 s (0.41 s/step), peak 57.95 GB alloc | qwen, Server [10-16] |
| 50 steps, prompt mask, pre-trim | 26.1 s (0.52 s/step); `difflet run` 38 s wall | f3c, Server [F3, 10-17] |
| 50 steps, trimmed padding (F4) | 16.5 s (0.33 s/step) | f3c, Server [F4] |
| 50 steps, `--torch-compile` | 12.4 s (0.25 s/step), load 15.7 s incl. compiles | f3c, Server [F4] |
| 50 steps, `--teacache-cadence 2` / 3 | 10.0 s at 38.50 dB / 12.3 s at 38.46 dB vs the full run | f3c, Server [F4] |
| diffusers `QwenImagePipeline`, 50 steps | 15.2 s (0.30 s/step) | f3c, Server [F3] |
| 50 steps, `cuda_smoke.py` | load 47.2 s, 38.5 s (0.77 s/step), peak 57.95 GB | f5b, Workstation [F3b] |
| parity sdpa (all-ones) | rel_l1 0.994× control, same-dtype 0.175× | f3c, Server [F3] |
| parity masked (16 tokens) | 0.990× / 0.167×, cosine vs fp32 0.999724 | f3c, Server [F3] |
| parity after trim / compiled | masked 0.992× / 0.180×; compiled 0.888× / 0.459× (sdpa) | f3c, Server [F4] |
| e2e | PSNR 41.40 dB, cosine 0.99972 (trim 41.20; compiled 40.21) | f3c, Server [F3, F4] |
| `verify_cli_cuda.py` cell | PASS, generate 38.5 s (local disk) | f3c, Server [F3] |
| `verify_cli_cuda.py` cell | PASS, generate 276.0 s (cold volume read + Workstation card) | f5b, Workstation [F5] |

Artifacts: `artifacts/cuda-release1/cuda_parity_qwen_image.json`, `cuda_e2e_qwen_image.json`,
`smoke_timings.json`, `verify_cli_cuda_results.json`, `f4/*.json`, the PNGs.

## FLUX.1-dev (`black-forest-labs/FLUX.1-dev`, gated)

Snapshot 33 GB without the duplicate single-file weights (33.7 GB in the volume listing),
downloaded in 31 s to the volume [F5, 10-18]; the owner's HF login is needed once per
`HF_HOME`. Defaults: 1024×1024, 28 steps, distilled guidance 3.5. Smoke
`scripts/cuda_smoke_flux.sh --steps 28 --seed 0`. All rows: pod f5b, **Workstation Edition**.

| Measurement | Value [entry] |
|---|---|
| smoke | load 36.5 s (CLIP + T5 + DiT + VAE), 28 steps 23.2 s [F5] |
| parity | control rel_l1 2.747e-2; Difflet 0.987×, same-dtype 0.398×, cosine vs fp32 0.999575 [F5] |
| parity loads | fp32 reference 90 s; diffusers bf16 949 s (cold volume); Difflet 23 s [F5] |
| e2e | PSNR 40.35 dB, cosine 0.99980; Difflet 23.6 s, diffusers 21.8 s [F5] |
| `verify_cli_cuda.py` cell | PASS, generate 56.5 s [F5] |
| DiT step | ≈ 0.84 s Difflet vs 0.78 s diffusers [F5] |

Artifacts: `artifacts/cuda-release1/flux/`.

## Wan 2.2 T2V-A14B (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`, public)

The repo is **126 GB of fp32 shards, 118 GB on disk** (two 57 GB experts, UMT5-XXL 11 GB, VAE),
cast to bf16 at load; both experts resident on CUDA. Downloaded in 108–109 s (us-east-3, to the
container disk and to the volume) and in 400 s (eu-3 local disk) [F1, F1b]. Defaults: 480×832,
9 frames, 40 steps, guidance 4.0 above the boundary / 3.0 below. Smoke
`scripts/cuda_smoke_wan.sh --steps 40 --num-frames 9 --seed 0` (writes an MP4 and a timings
JSON; peak memory from an `nvidia-smi` sampler). Parity inputs: B=1, latents (1, 16, 3, 60, 104),
text (1, 512, 4096), timesteps 900 (`transformer`) / 500 (`transformer_2`).

| Measurement | Value | Pod, edition [entry] |
|---|---|---|
| smoke | load 703.1 s (cold volume), 55.1 s (1.38 s/step), peak 73,035 MiB | f5b, Workstation [F1] |
| parity `transformer` | 1.051× control, same-dtype 0.870× (control cosine 0.99887) | f5b, Workstation [F1] |
| parity `transformer_2` | 1.043×, same-dtype 0.819× | f5b, Workstation [F1] |
| rounding @ 999 | fp32 timestep 1.039× (0.804×) vs bf16 cast 1.084× (0.873×) | f1b, Server [F1b] |
| rounding @ 750 | 1.018× (0.803×) vs 1.020× (0.813×) | f1b, Server [F1b] |
| e2e, bf16 timestep (`model`) | PSNR 27.54 dB, cosine 0.99704 | f5b Workstation [F1] and f1b Server [F1b] |
| e2e, fp32 timestep (`float32`, default) | PSNR 26.30 dB, cosine 0.99609 | f1b, Server [F1b] |
| e2e generate | Difflet 55.3 s / diffusers 44.6 s | f5b, Workstation [F1] |
| e2e generate | Difflet 50.9 s / 50.2 s, diffusers 42.9 s | f1b, Server [F1b] |
| serial `difflet run` | load 55.9 s, generate 50.9 s (1.27 s/step) | f2, Server [F2] |

The Wan e2e PSNR sits far below Qwen's and Flux's; its cause is open (F1b correction: measure
the e2e metric's own noise floor across seeds first — not measured). The 10 s generate gap to
diffusers is unmeasured. Artifacts: `artifacts/cuda-release1/wan/`, `wan/f1b/`.

## CLI matrix — `scripts/verify_cli_cuda.py`

Cells Qwen-Image 50 steps, FLUX.1-dev 28, Wan 2.2 40 (each download → compile → timed generate
as `DIFFLET_BACKEND=cuda` subprocesses; `results.json` under `--run-root`). It has **no cell
filter**, so all three models must be in `HF_HOME` — 209 GB together, more than the 200 GB
volume holds. The Wan cell has never run (NOT MEASURED); it needs a pod whose disk holds all
three (an eu node's container disk goes up to 500 GB) or the cell filter listed under F1b.

## Serve — `scripts/cuda_serve_smoke.sh` (Qwen-Image only) [F3b, f5b, Workstation]

`--output ~/logs/qwen-serve.png --steps 50 --requests 2`: ready in 56.0 s (load + the 4-step
startup smoke), request 1 38.6 s, request 2 39.2 s (no load: the worker is resident), 60,893 MiB
while serving. Probes against the running server: `extra_body.guidance_scale=1.0` → 400
`feature_not_supported`; `extra_body.height=768` → 400 `profile_mismatch`; two concurrent
requests → the second gets 429 `queue_timeout` after 30.0 s (the default `--queue-timeout` is
shorter than one request on this card); `difflet serve --model-id Qwen/Qwen-Image
--tp-degree 2` → refused naming F6 before any download, exit 1. Artifacts:
`artifacts/cuda-release1/f3b/`.

## Performance — `scripts/cuda_bench_dit.py` [F4, f3c, Server]

Qwen-Image DiT forward, median of 10 (warm-up untimed, a timed run with profilers off, one
`torch.profiler` run; the pod image has no `nsys`): baseline Difflet 516 ms ≈ diffusers on the
same padded input 490 ms; diffusers on its real 16-token input 300 ms; Difflet trimmed 323 ms;
`--torch-compile` 243 ms. Artifacts: `artifacts/cuda-release1/f4/bench_dit_*.json`. Compare
only on one card edition (F3b's open follow-up: characterise Workstation vs Server).

## Two GPUs [F2, f2, 2× Server Edition over PCIe]

NCCL smoke 11 s. **Wan `--cfg-parallel`** (`difflet run` re-execs under `torchrun --standalone
--nproc_per_node=2`): generate 26.1 s (0.65 s/step) vs 50.9 s serial (1.95×), video
bit-identical (max_abs 0), peak 68,074 / 67,519 MiB. **Qwen-Image `--dp 2 --requests`** (JSONL,
one request per line with `prompt`, `output`, `seed`): four 50-step requests 58 s wall vs 190 s
as four serial runs and 79 s as a `--dp 1` batch; all six compared images byte-identical. Not
run: Wan `--dp` × `--cfg-parallel`, Flux or Wan under `--dp`, serving on two GPUs (F2b). Artifacts:
`artifacts/cuda-release1/f2/`.
