# Feature catalog — each gate's script, inputs, thresholds and reference numbers

Every number below is from the reference campaign (Release 1 → F9, 2026-10-16 → 10-18) and
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
| `difflet-cuda-f9` | eu-3 | 1× **Server Edition**, no volume, 250 GB container disk | 2.65 | F9 [10-18] |

Same driver family, torch 2.11.0+cu128 and code path everywhere, yet Qwen-Image's 50-step
`difflet run` took **0.77 s/step on the Workstation Edition** (f5b) and **0.33 s/step on the
Server Edition** (f3c, after F4's trim): a **2.3× per-step gap** from the card/pod [F3b, 10-18].
Never put per-step numbers from two editions in one comparison without naming both. Parity and
e2e metrics did match across editions: Wan's `model`-timestep e2e gave 27.54 dB on both [F1b],
and again on f9 (Server) [F9].

## Support snapshot (2026-10-18; the default expectation per gate)

| Gate / feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B |
|---|---|---|---|
| smoke (`difflet run`, by eye) | PASS [F3] | PASS [F5] | PASS [F1] |
| DiT parity (D14, three gates) | PASS incl. padded prompt [F3, F4] | PASS [F5] | PASS both experts + rounding block [F1, F1b] |
| e2e vs diffusers | PASS, floors 35 dB / 0.998 [F3] | PASS, floors 35 dB / 0.998 [F5] | PASS, floors 25 dB / 0.99 [F1, F1b]; seeds 0–2 26.30–28.89 dB, overlapping diffusers' own noise floor 23.79–28.62 dB [F9]; gate relative to the floor when recorded, 2.0 dB headroom on the mean [F9b] |
| CLI matrix cell (`verify_cli_cuda.py`) | PASS [F3, F5, F9] | PASS [F5, F9] | PASS [F9] |
| `difflet serve` | PASS; LIMIT on concurrency on one GPU (429 `queue_timeout` after the 30 s default) [F3b]; `--dp-degree 2` PASS, two concurrent requests in one request-time [F2b] | N/A · refused by name, F5b | N/A · refused by name, F1b |
| `--torch-compile` | PASS, opt-in [F4] | N/A · refused, F5b | N/A · refused, F1b |
| `--teacache-cadence N` | PASS, fixed cadence only [F4] | N/A · refused, F5b | N/A · refused, F1b |
| `--dp 2 --requests` | PASS, byte-identical [F2] | NOT MEASURED (reachable) | PASS, bit-identical [F2b] |
| `--cfg-parallel` (2 GPUs) | N/A · single pass | N/A · guidance-distilled; true-CFG is the owner's call | PASS, bit-identical [F2] |
| `--dp` × `--cfg-parallel` | N/A · refuses `--cfg-parallel` | N/A · refuses `--cfg-parallel` | NOT MEASURED (needs 4 cards) |
| serving on two GPUs (`--dp-degree 2`) | PASS [F2b] | N/A · serve refused | N/A · serve refused |
| CP / ulysses (`--cp-degree > 1`) | N/A · refused, F2b | N/A · refused, F2b | N/A · refused, F2b |
| TP across PCIe, `--sp` | N/A · refused, F6 | N/A · refused, F6 | N/A · refused, F6 |

Statuses as in the evidence report: PASS (ran end to end in the cited entry), LIMIT (works with a
documented constraint), N/A (excluded by design, or refused by name pending the named
follow-up), NOT MEASURED (not refused, not run).

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
measurement left 2.5 dB, F1). The Wan gate judges the `float32` timestep variant by name, and
every seed of a multi-seed JSON. Wan's e2e over seeds with the metric's own **noise floor** (F9:
per seed, diffusers again with both experts' timestep rounded through bf16, exact vs perturbed
recorded; `run_gates.sh`'s e2e is one seed without it). When the result carries the floor on
every seed the Wan gate is stated **relative to it** (F9b, as D14 states the parity gates): the
mean PSNR over seeds ≥ the floor's mean − 1.21 dB (a 1.15× deviation ratio in dB) and the mean
cosine deviation ≤ 1.15² = 1.3225× the floor's (the same 1.15× on the linear error); every seed
≥ the floor's worst seed − 1.21 dB; a loose 20 dB / 0.98 absolute floor, which a floor must itself
clear ("the oracle moved" otherwise). The 25 dB / 0.99 floors then step aside; without a floor
they judge every seed and the relative test is skipped with that reason. F9's JSON: mean
27.20 dB against 25.17 required (2.0 dB, about one standard error of the difference at n = 3),
worst seed 26.30 against 22.58.

```bash
HF_HOME=$H HF_HUB_OFFLINE=1 DIFFLET_BACKEND=cuda PYTHONPATH=$PWD .venv/bin/python \
  scripts/cuda_e2e_wan_vs_diffusers.py --model-path "$SNAP" --seeds 0,1,2 --noise-floor \
  --timestep-dtypes float32 --steps 40 --num-frames 9 --out ~/logs/<name>/cuda_e2e_wan_noise.json \
  --images-dir ~/logs/<name>     # F9: 518 s; then the gate with DIFFLET_CUDA_WAN_E2E_RESULT=… -rA
```

## Qwen-Image (`Qwen/Qwen-Image`, public)

Snapshot 54 GB (57.7 GB in the volume's cache listing), downloaded in ~90 s on us-central-7
[10-16, 10-17] and in 175 s to an eu-3 container disk [F9]. Defaults: 1024×1024, 50 steps,
guidance 4.0 fixed (`guidance_embeds=False`, so `--guidance-scale` is refused). Smoke
`scripts/cuda_smoke.sh --steps 50 --seed 0` (CLI) or `scripts/cuda_smoke.py --steps 50
--local-files-only` (phase timings, peak memory).

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
| `verify_cli_cuda.py` cell | PASS, compile 3.8 s, generate 51.9 s (container disk) | f9, Server [F9] |

Artifacts: `artifacts/cuda-release1/cuda_parity_qwen_image.json`, `cuda_e2e_qwen_image.json`,
`smoke_timings.json`, `verify_cli_cuda_results.json`, `f4/*.json`, the PNGs.

## FLUX.1-dev (`black-forest-labs/FLUX.1-dev`, gated)

Snapshot 33 GB without the duplicate single-file weights (33.7 GB in the volume listing),
downloaded in 31 s to the volume [F5, 10-18] and in 109 s to an eu-3 container disk [F9]; the
owner's HF login is needed once per `HF_HOME`. Defaults: 1024×1024, 28 steps, distilled guidance
3.5. Smoke `scripts/cuda_smoke_flux.sh --steps 28 --seed 0`. Rows: pod f5b, **Workstation
Edition**, except the F9 row (f9, **Server Edition**).

| Measurement | Value [entry] |
|---|---|
| smoke | load 36.5 s (CLIP + T5 + DiT + VAE), 28 steps 23.2 s [F5] |
| parity | control rel_l1 2.747e-2; Difflet 0.987×, same-dtype 0.398×, cosine vs fp32 0.999575 [F5] |
| parity loads | fp32 reference 90 s; diffusers bf16 949 s (cold volume); Difflet 23 s [F5] |
| e2e | PSNR 40.35 dB, cosine 0.99980; Difflet 23.6 s, diffusers 21.8 s [F5] |
| `verify_cli_cuda.py` cell | PASS, generate 56.5 s [F5] |
| `verify_cli_cuda.py` cell, f9 Server | PASS, compile 2.1 s, generate 16.8 s [F9] |
| DiT step | ≈ 0.84 s Difflet vs 0.78 s diffusers [F5] |

Artifacts: `artifacts/cuda-release1/flux/`.

## Wan 2.2 T2V-A14B (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`, public)

The repo is **126 GB of fp32 shards, 118 GB on disk** (two 57 GB experts, UMT5-XXL 11 GB, VAE),
cast to bf16 at load; both experts resident on CUDA. Downloaded in 108–109 s (us-east-3, to the
container disk and to the volume) and in 400 s / 401 s (eu-3 local disk) [F1, F1b, F9].
Defaults: 480×832, 9 frames, 40 steps, guidance 4.0 above the boundary / 3.0 below. Smoke
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
| e2e `float32`, seeds 0 / 1 / 2 | PSNR 26.30 / 26.39 / 28.89 dB, cosine 0.99609 / 0.99541 / 0.99775 | f9, Server [F9] |
| noise floor: diffusers vs its bf16-rounded timestep, seeds 0 / 1 / 2 | PSNR 28.62 / 26.74 / 23.79 dB, cosine 0.99770 / 0.99575 / 0.99255 | f9, Server [F9] |
| e2e aggregate, min / mean / max | Difflet 26.30 / 27.20 / 28.89 dB; noise floor 23.79 / 26.38 / 28.62 dB | f9, Server [F9] |
| e2e timings (noise-floor run) | Difflet load 56.0 s, generate 50.9 s, peak 68.44 GB; diffusers 48.3 s / 43.2 s / 68.56 GB | f9, Server [F9] |
| `run_gates.sh wan` | exit 0 in 595 s: smoke 104 s (load 47.7 s, 51.0 s, 1.27 s/step, 73,035 MiB), parity 189 s, e2e 302 s (`float32` 26.30 dB, `model` 27.54 dB) | f9, Server [F9] |
| `verify_cli_cuda.py` cell | PASS, compile 2.1 s, generate 111.5 s (40 steps, MP4) | f9, Server [F9] |

The Wan e2e PSNR sits far below Qwen's and Flux's. F9 measured the metric's own noise floor:
diffusers against itself with its timestep rounded through bf16 (at most half a bf16 ulp, e.g. 999
→ 1000) spans 23.79–28.62 dB over three seeds; Difflet vs diffusers (26.30–28.89 dB) overlaps
that spread, so the gap to Qwen/Flux need not indicate a Difflet defect (n = 3: the ranges are
not the same, and it cannot show there is none). The floor is an upper bound on the
timestep-cast effect (it also absorbs run-to-run nondeterminism of the perturbed run; the exact
Difflet and diffusers runs are deterministic, seed 0's PSNR is identical across three runs). The
floors stayed 25 dB / 0.99 in F9 (R1); because the floor's worst seed (23.8 dB) is below 25 dB,
so seed luck could fail a correct build, F9b states the Wan gate relative to the measured noise
floor whenever the result carries it (the floors paragraph above). The 8 s generate gap to
diffusers was the DiT, not the loop (F4b: `scripts/cuda_bench_wan_dit.py`, expert forward
604.5 vs 526.4 ms with identical gemm and attention; the host loop 1.35 ms/step): the q/k
RMSNorm's seven fp32 passes. Since 83d244a it is one op (`difflet.ops.rms_norm_fp32`, fused on
CUDA): 536.2 vs 527.9 ms, generate 44.5 vs 42.8 s, gates green, e2e 27.91 dB (seed 0). Artifacts:
`artifacts/cuda-release1/wan/`, `wan/f1b/`, `f9/`
(`cuda_e2e_wan_noise.json`, per-seed first frames, `gates-wan/`), `f2b/bench/`, `f2b/gates-wan/`.

## CLI matrix — `scripts/verify_cli_cuda.py`

Cells Qwen-Image 50 steps, FLUX.1-dev 28, Wan 2.2 40 (each download → compile → timed generate
as `DIFFLET_BACKEND=cuda` subprocesses; `results.json` under `--run-root`). It has **no cell
filter**, so all three models must be in `HF_HOME` — 209 GB together, more than the 200 GB
volume holds. **All three cells PASS** on f9 (Server, eu-3, the three models on a 250 GB
container disk, `--seed 0`), 191 s wall; compile / generate: Qwen-Image 3.8 s / 51.9 s,
FLUX.1-dev 2.1 s / 16.8 s, Wan 2.2 2.1 s / 111.5 s [F9; `artifacts/cuda-release1/f9/`
`verify_cli_results.json`].

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
as four serial runs and 79 s as a `--dp 1` batch; all six compared images byte-identical.
**Wan `--dp 2 --requests`** (F2b, f2b): two 40-step requests 76 s wall vs 120 s serial, both
videos bit-identical; replicas generating at once take 51.2 s each vs 44.5 s alone (unmeasured).
**Serving on two GPUs** (F2b, f2b): `difflet serve --dp-degree 2` → `--concurrent 2` in 16.7 s
wall, `--concurrent 3` in 33.3 s, identical bytes; `--dp-degree 3` refused naming the GPU count
(`scripts/cuda_serve_smoke.py --dp N --concurrent K --reference PNG`). Not run: Wan `--dp` ×
`--cfg-parallel`, Flux under `--dp`. Artifacts: `artifacts/cuda-release1/f2/`, `f2b/serve/`,
`f2b/wan_dp/`.
