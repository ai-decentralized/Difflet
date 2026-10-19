# CUDA Release 1 → F9 — evidence report

Sessions: 2026-10-16 → 2026-10-18 (Release 1, F3, F4, F5, F3b, F1, F1b, F2, F9), nine Yottalabs
pods.
Branches and PRs (github.com/ai-decentralized/Difflet, stacked in this order, all open on
2026-10-18): `plan/cuda-backend-support` → #84 (Release 1 + F3 + F4), `feat/cuda-flux-f5` → #85,
`feat/cuda-serve-f3b` → #86, `feat/cuda-wan-f1` → #87, `feat/cuda-wan-f1b-timestep` → #88,
`feat/cuda-f2-second-gpu` → #89; this report lands on `feat/cuda-f8-verify-skill` (F8, #90),
which follows #89, and its F9 update on `feat/cuda-f9-loose-ends`, which follows #90.
Status-log entries: 2026-10-16 → 2026-10-18 in `docs/plans/2026-10-16-cuda-backend-support.md`
(cited below as *[date — entry]*).

This document lets a reviewer audit every CUDA claim of those sessions without the session
transcripts. Every number is transcribed from a JSON or log under `artifacts/cuda-release1/` (55
tracked files after F9) or from a status-log entry; the source sits next to each table, and
"status-log only" marks a number no artifact holds. Each section ends with **How to inspect
manually**. The GPU commands there are the ones `.claude/skills/difflet-cuda-verify/` runs
(its `scripts/run_gates.sh` mirrors the F3 / F5 / F1 / F1b runbooks). The numerical gates
re-run on a laptop from the committed JSONs, with no GPU. Nothing in F8 itself ran on a GPU; F9
ran on pod f9 (the Wan noise floor, the three-cell CLI matrix and `run_gates.sh wan`).

Statuses: **PASS** (ran end to end in the cited entry) / **LIMIT** (works with a documented
constraint) / **N/A** (excluded by design, or refused by name pending the named follow-up) /
**NOT MEASURED** (not refused, not run).

On-pod commands below run in `~/Difflet` on the pod after `pod_setup.sh`, with these defined once
(the environment `run_gates.sh` sets, and the snapshot its `auto` resolves):

```bash
H=/data/hf     # HF_HOME: the difflet-cache volume; H=$HOME/hf on a pod without it (pod_setup.sh)
M=models--Qwen--Qwen-Image   # or models--black-forest-labs--FLUX.1-dev,
                             #    models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
SNAP=$(ls -d "$H/hub/$M/snapshots/"*/ | head -1)
export HF_HOME=$H DIFFLET_BACKEND=cuda HF_HUB_OFFLINE=1 PYTHONPATH=$PWD PATH=$PWD/.venv/bin:$PATH
```

## Pods and cards

Every timing below names one of these rows. **Card editions matter:** the same code, driver
family and torch ran Qwen-Image at **0.77 s/step on the Workstation Edition** (f5b) and
**0.33 s/step on the Server Edition** (f3c, after F4's trim), a 2.3× per-step gap *[2026-10-18 —
F3b GPU verdict]*. Per-step numbers from different editions are never compared in one column
here without both being named.

| Pod | ID | Region | Card (edition) × n | Driver / torch | $/h | Created → released (final status) |
|---|---|---|---|---|---|---|
| `difflet-cuda-qwen` | 502004828165112280 | us-central-7 | RTX PRO 6000 Blackwell **Server** × 1 | 580.126.09 / 2.11.0+cu128 | 1.35 | 2026-10-16 → not recorded |
| `difflet-cuda-f3` | 502029337639260908 | us-central-7 | GPU-494f49ea (edition not recorded) × 1 | — | not recorded | 2026-10-17, 00:06–00:50 UTC window → terminated, nothing ran (NVML loss) |
| `difflet-cuda-f3b` | 502034396129263957 | us-central-7 | GPU-494f49ea (edition not recorded) × 1 | — | not recorded | same window → terminated, nothing ran (NVML loss) |
| `difflet-cuda-f3c` | 502176701700227982 | us-central-7 | **Server** × 1 (GPU-a91d562a) | 580.126.09 / 2.11.0+cu128 | 1.35 | 2026-10-17 09:52 UTC → terminated 2026-10-17 at the owner's request (time not recorded) |
| `difflet-cuda-f5` | 502393470423532085 | not recorded | not recorded, no volume | — | not recorded | 2026-10-18, setup only → terminated |
| `difflet-cuda-f5b` | 502399632095187214 | us-east-3 | **Workstation** × 1 (GPU-134e198a), `difflet-cache` volume | 580.126.09 / 2.11.0+cu128 | 0.45 | 2026-10-18 (time not recorded) → **PAUSED** 12:21 UTC (read back) |
| `difflet-cuda-f1b` | 502601836865405322 | eu-3 | **Server** × 2 (GPU 0 used) | 580.178.04 / 2.11.0+cu128 | 5.30 | 2026-10-18 13:58 → 14:24 UTC (**terminated**, read back gone) |
| `difflet-cuda-f2` | 502630011196547564 | eu-3 | **Server** × 2, PCIe `PHB`, no NVLink | 580.178.04 / 2.11.0+cu128 | 5.30 | 2026-10-18 15:52 → 16:21 UTC (**terminated**, read back gone) |
| `difflet-cuda-f9` | 502673395609120916 | eu-3 | **Server** × 1, no volume, 250 GB container disk | 580.178.04 / 2.11.0+cu128 | 2.65 | 2026-10-18 18:44 → 19:50 UTC (**terminated**, read back gone) |
| `difflet-cuda-f9c` | 502996083439452993 | eu-3 | **Server** × 1, no volume, 150 GB disk | 580.178.04 / 2.11.0+cu128 | 2.68 | 2026-10-19 16:06 → 16:20 UTC (**terminated**: the HF code expired; nothing ran) |
| `difflet-cuda-f9c2` | 502999627274010989 | us-east-3 | **Workstation** × 1, the volume requested but not attached (v2 API) | 580.126.09 / — | 0.45 | 2026-10-19 16:20 → 16:23 UTC (**terminated**; nothing ran) |
| `difflet-cuda-f9c3` | 503002432390967883 | us-east-3 | **Workstation** × 1, `difflet-cache` at `/data` (v1 API) | 580.126.09 / 2.11.0+cu128 | 0.45 | 2026-10-19 16:32 → **PAUSED** 16:53 UTC (read back) |

Network volume `difflet-cache`: id 469225541779435520, us-east-3, 200 GB CephFS,
$0.0698/GB/month, mounted at `/data` (HF cache + token, `HF_HOME=/data/hf`); cold reads ≈ 0.23
GB/s *[2026-10-18 — F5 Task 5 verdict]*. Sources: the entries named in each row, plus the F1b
execution-plan ledger for f1b's creation time and f5b's failed resumes; f9's driver and card
from `f9/gates-wan/gates.txt`, its torch from `f9/cuda_e2e_wan_noise.json`, its times from the
F9 entry.

## Result (final)

- **Smoke** (`difflet run`, by eye): PASS for Qwen-Image *[10-16, 10-17]*, FLUX.1-dev *[10-18 F5]*,
  Wan 2.2 T2V-A14B *[10-18 F1; again in `run_gates.sh wan`, 10-18 F9]*. Every image or first
  frame is the prompt's cat in a garden.
- **DiT parity (D14)**: PASS on all three gates for Qwen (all-ones and padded prompt; eager,
  trimmed, compiled), Flux, and both Wan experts plus the 999 / 750 rounding block.
- **e2e vs diffusers**: PASS. Qwen 41.40 dB and Flux 40.35 dB against 35 dB / 0.998 floors. Wan
  26.30 dB (fp32 timestep, gated) and 27.54 dB (bf16 cast) against 25 dB / 0.99. Over seeds 0–2
  (F9) Wan gives 26.30–28.89 dB, which overlaps diffusers' own perturbation spread (diffusers
  vs its bf16-rounded timestep: 23.79–28.62 dB), so the gap to Qwen and Flux need not indicate
  a Difflet defect (three seeds cannot show there is none).
- **CLI matrix**: PASS for all three cells. Qwen-Image three times, FLUX.1-dev twice, and Wan
  2.2 once: F9 ran the three-cell matrix on one pod (generate 51.9 s / 16.8 s / 111.5 s).
- **Serve (F3b, Qwen-Image)**: PASS. Ready in 56.0 s, two requests of 38.6 s and 39.2 s, both
  400 probes and the F6 refusal as designed. Concurrency is a LIMIT: 429 after the 30 s default.
- **Performance (F4)**: the Qwen DiT forward went from 515.7 ms to 323.5 ms (trim) and to 242.8 ms
  (`--torch-compile`); diffusers takes 299.6 ms. Gates still pass with both.
- **Two GPUs (F2)**: PASS. Wan `--cfg-parallel` ran 1.95× faster and bit-identical. Qwen
  `--dp 2 --requests` took 58 s wall against 190 s serial, byte-identical.
- **F1b A/B**: the fp32 timestep improves DiT parity at the rounding steps. The e2e PSNR moves
  −1.2 dB, which a single-seed A/B does not support either way.
- **Ledger**: 27 lines below. **Gaps**: 4 NOT MEASURED items. **Cost**: ~$7.80 for the three eu
  pods (~$2.60 recorded for f2, ~$2.30 computed for f1b, ~$2.90 computed for f9); the other pods'
  hours are not recorded.

## Smoke (`difflet run`, prompt-faithful by eye)

| Model / run | Steps / shape | Load | Generate (s/step) | Peak GPU memory | Pod (edition) | Artifact | Entry |
|---|---|---|---|---|---|---|---|
| Qwen-Image, Release 1 (no mask) | 4 / 1024² | 8.6 s | 1.9 s (0.47) | 57.95 GB alloc / 58.87 reserved | qwen (Server) | `smoke_timings.json` `runs[0]` | 10-16 Release 1 |
| Qwen-Image, Release 1 (no mask) | 50 / 1024² | 10.3 s | 20.5 s (0.41) | 57.95 / 58.87 GB | qwen (Server) | `runs[1]`, `qwen_50steps.png` | 10-16 Release 1 |
| Qwen-Image, F3 (padding mask) | 50 / 1024² | 8.8 s | 26.1 s (0.52) | 57.95 / 58.87 GB | f3c (Server) | `runs[2]`, `e2e_difflet_qwen_50steps.png` | 10-17 F3 |
| Qwen-Image, F3 CLI `cuda_smoke.sh` | 50 / 1024² | 9.3 s | image 26.5 s after load; 38 s wall | — | f3c (Server) | status-log only | 10-17 F3 |
| Qwen-Image, F4 trimmed prompt | 50 / 1024² | 8.4 s | 16.5 s (0.33) | 57.95 / 58.83 GB | f3c (Server) | `runs[3]` | 10-17 F4 |
| Qwen-Image, `--torch-compile` | 50 / 1024² | 15.7 s (both compiles) | 12.4 s (0.25) | 57.95 / 58.6 GB | f3c (Server) | `runs[4]` | 10-17 F4 |
| Qwen-Image, `--teacache-cadence 2` | 50 / 1024² | 10.2 s | 10.0 s (0.20) | 57.95 / 58.83 GB | f3c (Server) | `runs[5]` | 10-17 F4 |
| Qwen-Image, `cuda_smoke.py` after serving | 50 / 1024² | 47.2 s | 38.5 s (**0.77**) | 57.95 GB alloc | f5b (**Workstation**) | status-log only | 10-18 F3b |
| Qwen-Image, 4 serial `difflet run` | 50 / 1024² | 27.1 / 25.3 / 44.0 / 17.6 s | 16.9 / 16.8 / 16.8 / 17.3 s | — | f2 (Server) | `f2/timings.txt` | 10-18 F2 |
| FLUX.1-dev | 28 / 1024² | 36.5 s (CLIP + T5 + DiT + VAE) | 23.2 s (0.83, computed) | not recorded | f5b (**Workstation**) | status-log only; image = `flux/e2e_difflet_flux_28steps.png` | 10-18 F5 |
| Wan 2.2 T2V-A14B | 40 / 480×832 × 9 frames | 703.1 s (**cold volume**) | 55.1 s (1.38) | 73,035 MiB (`nvidia-smi`) | f5b (**Workstation**) | `wan/smoke_timings.json`, `wan/wan_40steps_9frames.mp4`, `wan/wan_40steps_frame0.png` | 10-18 F1 |
| Wan 2.2, serial on one card | 40 / 480×832 × 9 | 55.9 s | 50.9 s (1.27) | — | f2 (Server) | `f2/timings.txt` | 10-18 F2 |
| Wan 2.2, `run_gates.sh wan` smoke | 40 / 480×832 × 9 | 47.7 s | 51.0 s (1.27) | 73,035 MiB (`nvidia-smi`) | f9 (Server) | `f9/gates-wan/cuda-smoke-wan-20261018-193636.json` | 10-18 F9 |
| Wan 2.2, `run_gates.sh wan` smoke after the fused q/k norm (83d244a) | 40 / 480×832 × 9 | 16.7 s (page cache) | 48.5 s (1.21) | 72,491 MiB (`nvidia-smi`) | f2b (Server) | `f2b/gates-wan/cuda-smoke-wan-20261018-220419.json` | 10-18 F2b/F4b |
| Wan 2.2, `--torch-compile` smoke (F4f) | 40 / 480×832 × 9 | 82.6 s (incl. both experts' compiles) | **33.3 s (0.83)** | 73,259 MiB | f4f (Server) | `f4f/gates-wan-compiled/cuda-smoke-wan-20261019-132557.json` | 10-19 F4f |

Footnotes. All paths are under `artifacts/cuda-release1/`. Release 1's 0.41 s/step became
0.52 s/step in F3 (+27 %): the padding mask made the flash kernel ineligible. F4's trim
brought flash back (0.33 s/step). The Qwen 0.77 s/step row is the Workstation card, not a
regression (2.3×, *[10-18 F3b]*). Wan's 703.1 s load is a cold read of fp32 shards from the
network volume, cast to bf16 at load. TeaCache cadence 2 lands at PSNR 38.50 dB against the full
eager run, and cadence 3 at 12.3 s and 38.46 dB *[10-17 F4]*; the JSON note rounds the latter to
38.5 dB. The Flux per-step figure is 23.2 s / 28, computed here; the F5 entry's DiT-step figure
is ≈ 0.84 s (computed here: 23.6 s / 28, from the e2e run).

**What the images show.** `qwen_50steps.png` is a ginger-and-white cat in a sunlit garden
(Release 1, DiT attending over the padding). `e2e_difflet_qwen_50steps.png` is a tabby-and-white
cat on a garden path (F3, masked; bit-identical to the CLI image *[10-17 F3]*).
`flux/e2e_difflet_flux_28steps.png` is a ginger-and-white cat on a garden path (bit-identical to
the Flux smoke image *[10-18 F5]*). `wan/wan_40steps_frame0.png` is a ginger cat in a sunlit
garden. Its MP4's last frame fills with petal-like blobs, and diffusers' output under identical
settings shows the same *[10-18 F1]*.

**How to inspect manually.**

```bash
python3 -m json.tool artifacts/cuda-release1/smoke_timings.json      # runs[].{load_s,generate_s,per_step_s,peak_alloc_gb,note}
python3 -m json.tool artifacts/cuda-release1/wan/smoke_timings.json  # load_s, generate_s, peak_gpu_mem_mib
cat artifacts/cuda-release1/f2/timings.txt                           # "[difflet] loaded in" / "wrote … in" lines
open artifacts/cuda-release1/qwen_50steps.png artifacts/cuda-release1/wan/wan_40steps_9frames.mp4
# on a pod (the skill's per-model sequence; smoke is its first step):
bash .claude/skills/difflet-cuda-verify/scripts/run_gates.sh qwen auto --out $HOME/logs/<name>
scripts/cuda_smoke.sh --prompt "a photorealistic cat sitting in a sunlit garden" \
    --output $HOME/logs/qwen.png --steps 50 --seed 0
DIFFLET_BACKEND=cuda python scripts/cuda_smoke.py --steps 50 --local-files-only  # phases, peak memory
scripts/cuda_smoke_flux.sh --prompt "a photorealistic cat sitting in a sunlit garden" \
    --output $HOME/logs/flux.png --steps 28 --seed 0
scripts/cuda_smoke_wan.sh --prompt "a cat walking through a sunlit garden, photorealistic" \
    --output $HOME/logs/wan.mp4 --steps 40 --num-frames 9 --seed 0
```

## DiT parity vs diffusers (D14)

One GPU script runs three forwards on identical seeded inputs: diffusers fp32 (the reference),
diffusers bf16 (the **control**) and Difflet bf16. The gates are relative to the control. Gate 1:
Difflet rel_l1 vs fp32 ≤ **1.15×** the control's. Gate 2: Difflet's rel_l1 vs the bf16 control
("same-dtype") ≤ **1.2×** the control's rel_l1 vs fp32. Gate 3: cosine vs the bf16 control ≥
**0.995** (constants in `tests/numerical/test_cuda_*_vs_diffusers.py`).

| Model / expert / case | Control rel_l1 vs fp32 | Difflet rel_l1 (× control) | Same-dtype rel_l1 (× control) | Cosine vs bf16 | Gate | Pod (edition) | Artifact | Entry |
|---|---|---|---|---|---|---|---|---|
| Qwen, all-ones mask, Release 1 | 1.709e-2 | 1.703e-2 (0.997×) | 3.26e-3 (0.19×) | 0.999991 | PASS | qwen (Server) | status-log only (JSON overwritten by F3, ruling) | 10-16 Release 1 |
| Qwen, all-ones mask (`sdpa`), F3 | 1.709e-2 | 1.698e-2 (0.994×) | 2.996e-3 (0.175×) | 0.999992 | PASS (4 passed) | f3c (Server) | `cuda_parity_qwen_image.json` | 10-17 F3 |
| Qwen, padded prompt (`masked`, 16 of 1024), F3 | 2.087e-2 | 2.067e-2 (0.990×) | 3.483e-3 (0.167×) | 0.999990 | PASS | f3c (Server) | `cuda_parity_qwen_image.json` | 10-17 F3 |
| Qwen `sdpa`, trimmed (F4) | 1.709e-2 | 1.703e-2 (0.997×) | 3.256e-3 (0.190×) | 0.999991 | PASS | f3c (Server) | `f4/parity_trimmed.json` | 10-17 F4 |
| Qwen `masked`, trimmed (F4) | 2.087e-2 | 2.070e-2 (0.992×) | 3.751e-3 (0.180×) | 0.999989 | PASS | f3c (Server) | `f4/parity_trimmed.json` | 10-17 F4 |
| Qwen `sdpa`, `--torch-compile` | 1.709e-2 | 1.517e-2 (0.888×) | 7.837e-3 (0.459×) | 0.999965 | PASS | f3c (Server) | `f4/parity_compiled.json` | 10-17 F4 |
| Qwen `masked`, `--torch-compile` | 2.087e-2 | 1.882e-2 (0.902×) | 8.244e-3 (0.395×) | 0.999961 | PASS | f3c (Server) | `f4/parity_compiled.json` | 10-17 F4 |
| FLUX.1-dev | 2.747e-2 | 2.710e-2 (0.987×) | 1.094e-2 (0.398×) | 0.999927 | PASS (3 passed) | f5b (Workstation) | `flux/cuda_parity_flux.json` | 10-18 F5 |
| Wan `transformer`, t = 900 | 4.731e-2 | 4.970e-2 (1.051×) | 4.115e-2 (0.870×) | 0.999142 | PASS (5 passed) | f5b (Workstation) | `wan/cuda_parity_wan.json` | 10-18 F1 |
| Wan `transformer_2`, t = 500 | 2.303e-2 | 2.402e-2 (1.043×) | 1.887e-2 (0.819×) | 0.999806 | PASS | f5b (Workstation) | `wan/cuda_parity_wan.json` | 10-18 F1 |
| Wan both experts, re-run | identical | identical (bit-for-bit) | identical | identical | PASS (11 passed) | f1b (Server) | `wan/f1b/cuda_parity_wan.json` | 10-18 F1b |
| Wan both experts + rounding block, `run_gates.sh wan` | identical | identical | identical | identical | PASS (11 passed) | f9 (Server) | `f9/gates-wan/cuda_parity_wan.json` (byte-identical to the F1b JSON, `cmp`) | 10-18 F9 |

Inputs: Qwen at 1024×1024 with 1024 text tokens (16 valid in the padded case). Flux at
1024×1024 with 512 text tokens and guidance embeds. Wan with latents (1, 16, 3, 60, 104) and text
(1, 512, 4096), one expert on the GPU at a time. The control's cosine vs fp32 is 0.999856 (Qwen
all-ones), 0.999721 (Qwen padded), 0.999557 (Flux), 0.998869 (Wan `transformer`) and 0.999726
(`transformer_2`). `--torch-compile` moves the same-dtype distance up (0.459×) while sitting
closer to fp32 (0.888×); the F4 ledger attributes this to inductor's fp32 accumulation, and it is
not a defect. The Wan rounding block is in the F1b A/B section.

**How to inspect manually.** The gates read the JSON, so they re-run on a laptop with no GPU.
All of these were re-run on the Mac (CPU) on 2026-10-18 for this report:

```bash
A=artifacts/cuda-release1; export DIFFLET_BACKEND=cpu DIFFLET_RUN_CUDA_NUMERICAL=1
DIFFLET_CUDA_PARITY_RESULT=$A/cuda_parity_qwen_image.json pytest -q tests/numerical/test_cuda_vs_diffusers.py  # 4 passed
DIFFLET_CUDA_PARITY_RESULT=$A/f4/parity_trimmed.json pytest -q tests/numerical/test_cuda_vs_diffusers.py      # 4 passed
DIFFLET_CUDA_PARITY_RESULT=$A/f4/parity_compiled.json pytest -q tests/numerical/test_cuda_vs_diffusers.py     # 4 passed
DIFFLET_CUDA_FLUX_PARITY_RESULT=$A/flux/cuda_parity_flux.json pytest -q tests/numerical/test_cuda_flux_vs_diffusers.py  # 3 passed
DIFFLET_CUDA_WAN_PARITY_RESULT=$A/wan/f1b/cuda_parity_wan.json pytest -q tests/numerical/test_cuda_wan_vs_diffusers.py  # 11 passed
DIFFLET_CUDA_WAN_PARITY_RESULT=$A/f9/gates-wan/cuda_parity_wan.json pytest -q tests/numerical/test_cuda_wan_vs_diffusers.py  # 11 passed
cmp $A/wan/f1b/cuda_parity_wan.json $A/f9/gates-wan/cuda_parity_wan.json && echo identical   # F9 = F1b
python3 -m json.tool $A/cuda_parity_qwen_image.json  # control.rel_l1, results.<sdpa|masked>.rel_l1, .vs_bf16.{rel_l1,cosine}
python3 -m json.tool $A/wan/f1b/cuda_parity_wan.json # experts.<e>.{control,results.sdpa,rounding.{fp32_timestep,bf16_timestep}}
```

The F1 Wan JSON (`wan/cuda_parity_wan.json`) gives "6 failed, 5 passed" under today's gate. This
is by design, not a regression: the three rounding-block tests, each parametrised over both
experts (6 failures), say
`"[transformer] no rounding block in …: the oracle predates F1b task 2; re-run
cuda_parity_wan_gpu.py"`. The F1 numbers it holds are reproduced bit-for-bit in the F1b JSON.
On a pod (`run_gates.sh` runs these with `HF_HUB_OFFLINE=1`):

```bash
python tests/numerical/cuda_parity_gpu.py --model-path "$SNAP" --out $HOME/logs/x/cuda_parity_qwen_image.json \
    --reference $HOME/logs/x/cuda_parity_reference.pt
python tests/numerical/cuda_parity_flux_gpu.py --model-path "$SNAP" --out $HOME/logs/x/cuda_parity_flux.json \
    --reference $HOME/logs/x/cuda_parity_flux_reference.pt
python tests/numerical/cuda_parity_wan_gpu.py --model-path "$SNAP" --expert both \
    --out $HOME/logs/x/cuda_parity_wan.json --reference "$HOME/logs/x/cuda_parity_wan_reference_{expert}.pt" \
    --cache-dir $HOME/logs/x/cuda_parity_wan_cache
```

## e2e vs diffusers

The prompt goes in and an image or video comes out, from the same initial noise, through Difflet
and through stock diffusers (`QwenImagePipeline`, `FluxPipeline`, `WanPipeline`), one side loaded
at a time. Pixel metrics are in [0, 1]. Floors (in `tests/numerical/test_cuda_*e2e_vs_diffusers.py`):
Qwen and Flux PSNR ≥ **35 dB**, cosine ≥ **0.998**, tightened from 25 / 0.99 after the first
measurement (F3, F5); Flux's gate is relative to a recorded noise floor since F4f, over seeds
since F9c (the F4f subsection), the 35 dB / 0.998 floors judging results without one. Wan stays at ≥ **25 dB**, ≥ **0.99** (kept in F1; the gate judges the
`float32` variant by name since F1b, and every seed of a multi-seed JSON since F9; the floors
did not move in F9, ruling R1). Since F9b, when the result carries the noise floor on every seed
(`--seeds … --noise-floor`), the Wan gate is stated **relative to it**, the way D14 states the
parity gates: (A) the mean PSNR over seeds is at least the floor's mean minus 20·log10(1.15) =
1.21 dB (the 1.15× deviation ratio in dB) and the mean cosine deviation 1 − cosine at most
1.15² = 1.3225× the floor's (1 − cosine is quadratic in the error, ≈ ε²/2, so the square is the
same 1.15× on the linear error); (B) every seed is at least the floor's worst seed minus 1.21 dB,
its cosine deviation at most 1.3225× the floor's worst; (C) a loose absolute floor of 20 dB /
0.98 on every seed. A floor whose own worst seed is under the loose floor, or not finite, fails
the relative test as "the oracle moved" instead of scaling the gate down to it. The 25 dB / 0.99
floors then step aside (the floor's own worst seed, 23.79 dB, is under
them, so seed luck could fail a correct build); without a floor they judge every seed as before
and the relative test is skipped with that reason (`run_gates.sh wan`'s one-seed e2e, the F1b
JSON). On the F9 measurement: gate A 27.20 dB against 25.17 required (floor mean 26.38 − 1.21),
cosine deviation 0.00358 against 0.00617 allowed; gate B 22.58 dB required (floor min 23.79 −
1.21), worst Difflet seed 26.30; 0.00986 allowed, worst 0.00459. With n = 3 the floor's mean
carries a standard error of about 1.4 dB (sample std 2.43 dB), so the 2.0 dB headroom on gate A
is about one standard error of the difference: the relative gate reduces the seed-luck risk, it
does not remove it; more seeds narrow it.

| Model / variant | Steps / shape | PSNR | Cosine | mean / max abs | Floors | Difflet / diffusers generate | Pod (edition) | Artifact | Entry |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image (mask) | 50 / 1024² | 41.40 dB | 0.99972 | 0.0028 / 0.275 | 35 / 0.998 | — / 15.2 s (diffusers load 6.1 s) | f3c (Server) | `cuda_e2e_qwen_image.json`, `e2e_{difflet,diffusers}_qwen_50steps.png` | 10-17 F3 |
| Qwen-Image, trimmed | 50 / 1024² | 41.20 dB | 0.99971 | 0.0031 / 0.258 | 35 / 0.998 | — | f3c (Server) | `f4/e2e_trimmed.json` | 10-17 F4 |
| Qwen-Image, `--torch-compile` | 50 / 1024² | 40.21 dB | 0.99963 | 0.0039 / 0.266 | 35 / 0.998 | — | f3c (Server) | `f4/e2e_compiled.json` | 10-17 F4 |
| FLUX.1-dev (guidance 3.5), Server Edition, timings recorded | 28 / 1024² | 40.35 dB | 0.99980 | 0.0034 / 0.625 | 35 / 0.998 | 9.4 s / 8.3 s (loads 8.3 / 3.0 s; peak 36.33 / 36.32 GB) | flux (Server) | `flux/f5c/cuda_e2e_flux.json`, `flux/f5c/flux_e2e_{difflet,diffusers}.png` | 10-18 F5c |
| FLUX.1-dev, `run_gates.sh flux` on the F4d code | 28 / 1024² | 40.35 dB | 0.99980 | — | 35 / 0.998 | 9.1 s / 8.3 s | f4d (Server) | `f4d/gates-flux/cuda_e2e_flux.json` | 10-19 F4c/F4d |
| FLUX.1-dev, `run_gates.sh flux` after the fused q/k norm (F4e) | 28 / 1024² | **41.58 dB** | 0.99985 | — | 35 / 0.998 | **8.8 s** / 8.3 s | f4e (Server) | `f4e/gates-flux/cuda_e2e_flux.json` | 10-19 F4e |
| FLUX.1-dev, fused projection, seeds 0 / 1 / 2 with the noise floor (F4f) | 28 / 1024² | 31.56 / 39.56 / 43.79 dB (floor 30.92 / 36.62 / 37.61) | 0.99851 / 0.99976 / 0.99991 | — | relative to the floor | 8.8 s / 8.3 s | f4f (Server) | `f4f/floor/e2e_floor_seed{0,1,2}.json` | 10-19 F4f |
| FLUX.1-dev, `--torch-compile`, seeds 0 / 1 / 2 | 28 / 1024² | 31.41 / 39.79 / 42.53 dB | — | — | 35 / 0.998 (seed 0 FAIL, no floor recorded) | **7.1 s** / 8.3 s | f4f (Server) | `f4f/gates-flux-compiled/cuda_e2e_flux.json`, `f4f/ab/e2e_compiled_seed{1,2}.json` | 10-19 F4f |
| Wan 2.2, `--torch-compile`: `float32` / `model`, seed 0 | 40 / 480×832 × 9 | 27.49 / 27.79 dB | 0.99702 / 0.99721 | — | 25 / 0.99 | **32.8 / 32.5 s** / 43.2 s | f4f (Server) | `f4f/gates-wan-compiled/cuda_e2e_wan.json` | 10-19 F4f |
| Qwen-Image, `run_gates.sh qwen` on the F4d code | 50 / 1024² | 41.20 dB | 0.99971 | — | 35 / 0.998 | 16.9 s / 15.2 s | f4d (Server) | `f4d/gates-qwen/cuda_e2e_qwen_image.json` | 10-19 F4c/F4d |
| Wan 2.2, `run_gates.sh wan` on the F4d code: `float32` / `model` | 40 / 480×832 × 9 | 27.91 / 27.51 dB | 0.99730 / 0.99702 | — | 25 / 0.99 | 48.9 (the FLUX.1-dev download ran on the host meanwhile — contended) / 44.6 s / 43.1 s | f4d (Server) | `f4d/gates-wan/cuda_e2e_wan.json` | 10-19 F4c/F4d |
| FLUX.1-dev (guidance 3.5) | 28 / 1024² | 40.35 dB | 0.99980 | 0.0034 / 0.625 | 35 / 0.998 | 23.6 s / 21.8 s (status-log only) | f5b (Workstation) | `flux/cuda_e2e_flux.json`, `flux/e2e_{difflet,diffusers}_flux_28steps.png` | 10-18 F5 |
| Wan 2.2, bf16 timestep (`model`), F1 | 40 / 480×832 × 9 | 27.54 dB | 0.99704 | 0.0238 / 1.0 | 25 / 0.99 | 55.3 s / 44.6 s (loads 417.6 / 452.3 s; peak 68.44 / 68.56 GB) | f5b (Workstation) | `wan/cuda_e2e_wan.json`, `wan/e2e_{difflet,diffusers}_wan_40steps_frame0.png` | 10-18 F1 |
| Wan 2.2, fp32 timestep (`float32`, default, gated) | 40 / 480×832 × 9 | 26.30 dB | 0.99609 | 0.0274 / 1.0 | 25 / 0.99 | 50.9 s / 42.9 s | f1b (Server) | `wan/f1b/cuda_e2e_wan.json` | 10-18 F1b |
| Wan 2.2, `float32` / `model`, seed 0, after the fused q/k norm (83d244a) | 40 / 480×832 × 9 | 27.91 / 27.51 dB | 0.99730 / 0.99702 | — | 25 / 0.99 (no floor in the result) | 44.6 / 44.5 s / 42.8 s | f2b (Server) | `f2b/gates-wan/cuda_e2e_wan.json` | 10-18 F2b/F4b |
| Wan 2.2, `float32`, seeds 0 / 1 / 2 | 40 / 480×832 × 9 | 26.30 / 26.39 / 28.89 dB | 0.99609 / 0.99541 / 0.99775 | 0.0274 / 0.0280 / 0.0223; max 1.0 / 0.988 / 0.920 | 25 / 0.99, every seed | 50.9 / 49.3 / 49.4 s / 43.2 / 42.9 / 42.9 s (loads 56.0 / 48.3 s; peak 68.44 / 68.56 GB) | f9 (Server) | `f9/cuda_e2e_wan_noise.json` `per_seed`, `timings` | 10-18 F9 |
| Wan 2.2, `run_gates.sh wan`: `float32` / `model`, seed 0 | 40 / 480×832 × 9 | 26.30 / 27.54 dB | 0.99609 / 0.99704 | 0.0274 / 0.0238; max 1.0 | 25 / 0.99 | 50.6 / 50.2 s / 43.1 s (loads 49.8 / 48.9 / 52.6 s) | f9 (Server) | `f9/gates-wan/cuda_e2e_wan.json` `variants` | 10-18 F9 |

The difference between the two Qwen e2e Difflet images and the diffusers images is not visible
by eye: they are the same cat. The Wan first frames are the same calico cat on the same garden
path. Two facts support "deterministic across editions". First, the `model` variant gives 27.54 dB
on both the Workstation (F1) and Server (F1b) cards. Second, checked for this report: the F1 and
F1b first frames are byte-identical (`wan/e2e_difflet_wan_40steps_frame0.png` =
`wan/f1b/e2e_difflet_bf16_timestep_frame0.png`, `wan/e2e_diffusers_wan_40steps_frame0.png` =
`wan/f1b/e2e_diffusers_frame0.png`, same MD5). The Wan PSNR sits far below Qwen's and Flux's;
F9 measured why it need not mean a Difflet defect (next subsection). F9 reproduced both F1b
numbers to the last digit (26.303416… dB `float32`, 27.541036… dB `model`), and its seed-0 first
frames are byte-identical to F1b's (`f9/e2e_difflet_fp32_seed0_frame0.png` =
`wan/f1b/e2e_difflet_fp32_timestep_frame0.png`, `f9/e2e_diffusers_seed0_frame0.png` =
`wan/f1b/e2e_diffusers_frame0.png`, same MD5).

### Noise floor (F9): how far diffusers moves from itself

`scripts/cuda_e2e_wan_vs_diffusers.py --seeds 0,1,2 --noise-floor --timestep-dtypes float32
--steps 40 --num-frames 9` on pod f9 (**Server**), checkout 2387653, prompt "a photorealistic cat
walking through a sunlit garden", guidance 4.0 / 3.0, 480×832 × 9 frames; 518 s, gate passed
(status-log only for the wall time) *[10-18 F9]*. Per seed it draws one noise tensor, runs Difflet
(`float32` timestep) and diffusers' `WanPipeline` exactly, then diffusers again with both
experts' `forward` receiving the timestep rounded through bf16 (`t.to(bfloat16).to(t.dtype)`, at
most half a bf16 ulp, e.g. 999 → 1000) and nothing else changed. Difflet and the perturbed run
are both compared against the exact diffusers video.

| Seed | Difflet `float32` vs diffusers exact: PSNR / cosine / mean_abs | Noise floor, diffusers exact vs bf16-rounded timestep: PSNR / cosine / mean_abs | First frames (`f9/`) |
|---|---|---|---|
| 0 | 26.30 dB / 0.99609 / 0.0274 | 28.62 dB / 0.99770 / 0.0203 | `e2e_difflet_fp32_seed0_frame0.png`, `e2e_diffusers_seed0_frame0.png`, `e2e_diffusers_bf16ts_seed0_frame0.png` |
| 1 | 26.39 dB / 0.99541 / 0.0280 | 26.74 dB / 0.99575 / 0.0259 | the same three names, `seed1` |
| 2 | 28.89 dB / 0.99775 / 0.0223 | 23.79 dB / 0.99255 / 0.0365 | the same three names, `seed2` |
| min / mean / max | 26.30 / 27.20 / 28.89 dB; cosine 0.99541 / 0.99642 / 0.99775 | 23.79 / 26.38 / 28.62 dB; cosine 0.99255 / 0.99533 / 0.99770 | |

Source: `f9/cuda_e2e_wan_noise.json` (`per_seed.<s>.variants.float32.metrics`,
`per_seed.<s>.noise_floor.metrics`, `aggregate`); the same lines print under the gate's `-rA`
(`f9/summary_extract.txt`). Timings: Difflet load 56.0 s, generate 50.9 / 49.3 / 49.4 s, peak
68.44 GB; diffusers exact load 48.3 s, generate 43.2 / 42.9 / 42.9 s; perturbed 42.8 / 42.9 /
42.9 s; peak 68.56 GB. The artifact PNG names drop the script's `wan_` prefix and say `fp32`
for `float32` (`e2e_difflet_fp32_seed2_frame0.png` is the script's
`wan_e2e_difflet_float32_seed2_frame0.png`).

**Verdict.** Difflet vs diffusers (26.30–28.89 dB over seeds 0–2) overlaps diffusers' own
perturbation spread (23.79–28.62 dB with its timestep rounded through bf16), so the gap to Qwen
and Flux (40–41 dB) need not indicate a Difflet defect. The noise floor is an upper bound on the
timestep-cast effect (it also absorbs any run-to-run nondeterminism of the perturbed run; the
exact Difflet and diffusers runs are deterministic — seed 0's whole-video PSNR
26.303416187998177 is identical to the last digit across the F1b, F9 noise and F9 gates-wan runs
on two pods). The ranges are not the same (26.30–28.89 vs 23.79–28.62 dB), and
matched by seed Difflet is further from diffusers than the perturbed run on seeds 0 and 1, so
with n = 3 this shows the gap need not indicate a defect; it cannot show there is none. By eye, seed 2's exact and
perturbed first frames (23.8 dB) are the same tabby cat in the same pose with different fur and
foliage texture (the perturbed one also loses the dark collar, which Difflet's frame keeps).
Ruling R1 (F9): the floors (25 dB / 0.99) are unchanged; the floor's own worst seed (23.79 dB)
is below the 25 dB gate, so seed luck could fail a correct build. F9b acted on that: with the
floor in the result the gate is stated relative to it (the e2e floors paragraph above); this JSON
passes it with 2.0 dB of headroom on the mean (about one standard error of the difference at
n = 3) and 3.7 dB on the worst seed.

**How to inspect manually.**

```bash
A=artifacts/cuda-release1; export DIFFLET_BACKEND=cpu DIFFLET_RUN_CUDA_NUMERICAL=1
DIFFLET_CUDA_E2E_RESULT=$A/cuda_e2e_qwen_image.json pytest -q tests/numerical/test_cuda_e2e_vs_diffusers.py     # 2 passed
DIFFLET_CUDA_E2E_RESULT=$A/f4/e2e_trimmed.json pytest -q tests/numerical/test_cuda_e2e_vs_diffusers.py         # 2 passed
DIFFLET_CUDA_E2E_RESULT=$A/f4/e2e_compiled.json pytest -q tests/numerical/test_cuda_e2e_vs_diffusers.py        # 2 passed
DIFFLET_CUDA_FLUX_E2E_RESULT=$A/flux/cuda_e2e_flux.json pytest -q tests/numerical/test_cuda_flux_e2e_vs_diffusers.py  # 3 passed, 1 skipped (no noise floor: 35 dB / 0.998 judged)
DIFFLET_CUDA_WAN_E2E_RESULT=$A/wan/f1b/cuda_e2e_wan.json pytest -q tests/numerical/test_cuda_wan_e2e_vs_diffusers.py  # 4 passed, 1 skipped (no noise floor: 25 dB / 0.99 judged)
DIFFLET_CUDA_WAN_E2E_RESULT=$A/f9/gates-wan/cuda_e2e_wan.json pytest -q tests/numerical/test_cuda_wan_e2e_vs_diffusers.py  # 4 passed, 1 skipped
# the noise floor: the relative gate (F9b) prints its thresholds, the floor and the aggregates (-rA)
DIFFLET_CUDA_WAN_E2E_RESULT=$A/f9/cuda_e2e_wan_noise.json pytest -q -rA tests/numerical/test_cuda_wan_e2e_vs_diffusers.py  # 5 passed
python3 -c "import json; d=json.load(open('artifacts/cuda-release1/f9/cuda_e2e_wan_noise.json'))
for s, v in d['per_seed'].items(): print(s, round(v['variants']['float32']['metrics']['psnr_db'], 2), round(v['noise_floor']['metrics']['psnr_db'], 2))
print(d['aggregate'])"
open $A/f9/e2e_diffusers_seed2_frame0.png $A/f9/e2e_diffusers_bf16ts_seed2_frame0.png $A/f9/e2e_difflet_fp32_seed2_frame0.png
open $A/e2e_difflet_qwen_50steps.png $A/e2e_diffusers_qwen_50steps.png   # the same picture, side by side
```

The F1 Wan e2e JSON fails today's `test_every_timestep_dtype_variant_is_recorded` with `"no
variants: the result predates F1b task 2; re-run the e2e script"` (1 failed, 3 passed, 1 skipped
since F9b added the relative test; 1 failed, 3 passed in F9; 1 failed, 2 passed before). This is
by design, as in the parity section.
On a pod:

```bash
python scripts/cuda_e2e_vs_diffusers.py --model-path "$SNAP" --out $HOME/logs/x/cuda_e2e_qwen_image.json --images-dir $HOME/logs/x
python scripts/cuda_e2e_flux_vs_diffusers.py --model-path "$SNAP" --out $HOME/logs/x/cuda_e2e_flux.json --images-dir $HOME/logs/x
python scripts/cuda_e2e_wan_vs_diffusers.py --model-path "$SNAP" --timestep-dtypes float32,model --steps 40 \
    --num-frames 9 --out $HOME/logs/x/cuda_e2e_wan.json --images-dir $HOME/logs/x --compile-cache-dir $HOME/logs/x/cache
# the F9 noise floor over seeds (one load per side; 518 s on f9)
python scripts/cuda_e2e_wan_vs_diffusers.py --model-path "$SNAP" --seeds 0,1,2 --noise-floor \
    --timestep-dtypes float32 --steps 40 --num-frames 9 --out $HOME/logs/x/cuda_e2e_wan_noise.json --images-dir $HOME/logs/x
DIFFLET_RUN_CUDA_NUMERICAL=1 DIFFLET_CUDA_WAN_E2E_RESULT=$HOME/logs/x/cuda_e2e_wan_noise.json \
    python -m pytest -q -rA tests/numerical/test_cuda_wan_e2e_vs_diffusers.py
```

## CLI matrix (`scripts/verify_cli_cuda.py`)

Each cell runs download → compile → timed generate as `DIFFLET_BACKEND=cuda` subprocesses. The
script has no cell filter, so every model must be in `HF_HOME`.

| Cell | Outcome | Download / compile / generate s | Pod (edition) | Artifact | Entry |
|---|---|---|---|---|---|
| Qwen/Qwen-Image tp1, 50 steps | PASS | 0.21 / 1.52 / 38.47 | f3c (Server, local disk) | `verify_cli_cuda_results.json` | 10-17 F3 |
| Qwen/Qwen-Image tp1, default steps | PASS | 0.26 / 2.52 / 276.01 | f5b (Workstation, cold volume read) | `flux/verify_cli_results.json` | 10-18 F5 |
| black-forest-labs/FLUX.1-dev tp1, default steps | PASS | 0.27 / 3.37 / 56.47 | f5b (Workstation) | `flux/verify_cli_results.json` | 10-18 F5 |
| Qwen/Qwen-Image tp1, 50 steps, `--seed 0` | PASS | 0.21 / 3.77 / 51.88 | f9 (Server, container disk) | `f9/verify_cli_results.json` | 10-18 F9 |
| black-forest-labs/FLUX.1-dev tp1, 28 steps | PASS | 0.21 / 2.07 / 16.79 | f9 (Server, container disk) | `f9/verify_cli_results.json` | 10-18 F9 |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers tp1, 40 steps (MP4) | **PASS** | 0.21 / 2.12 / 111.52 | f9 (Server, container disk) | `f9/verify_cli_results.json` | 10-18 F9 |

The 276 s Qwen generate combines a cold read of 54 GB from the network volume (≈ 240 s at
≈ 0.23 GB/s) with the slower Workstation card *[10-18 F5, F3b]*. In the F5 run
`generate_steps` is `null`, so each cell used its default (`CELL_STEPS`: Qwen 50, Flux 28).
**F9 ran all three cells in one invocation** on pod f9 (Server, eu-3, no volume): the three
models (Wan 401 s, Qwen 175 s, Flux 109 s to download; status-log only, in no artifact) fit its
250 GB container disk under `HF_HOME=~/hf`, and the matrix passed in 191 s wall (status-log only;
the JSON holds per-step durations). `generate_steps` is `null` there too; each cell's `generate_steps` field reads 50 /
28 / 40. Each generate is a fresh `difflet generate` subprocess, load included; the Qwen
51.88 s against f3c's 38.47 s on the same edition was not investigated. The Wan cell's MP4 was
not copied off the pod.

**How to inspect manually.**

```bash
for f in flux/verify_cli_results.json f9/verify_cli_results.json; do
python3 -c "import json, sys; d=json.load(open(sys.argv[1])); print(d['device'])
[print(c['model_id'], c['outcome'], {s: (v['status'], round(v['duration_s'], 2)) for s, v in c['steps'].items()})
 for c in d['cells']]" artifacts/cuda-release1/$f; done
# on a pod, every cell at once (all models must be cached; F9: 191 s on the Server Edition):
HF_HOME=$H DIFFLET_BACKEND=cuda .venv/bin/python scripts/verify_cli_cuda.py --seed 0
```

## Serving (`difflet serve`, F3b — Qwen-Image only)

Pod f5b (**Workstation**), checkout 69624a7, `HF_HOME=/data/hf`, offline *[10-18 F3b GPU verdict]*.

| Check | Result (verbatim message / code) | Time | Artifact |
|---|---|---|---|
| ready (load + 4-step startup smoke) | "Qwen-Image CUDA generation smoke passed" | 56.0 s | `f3b/serve_timings.json` `ready_s` |
| request 1 (50 steps) | 200, the sunlit-garden cat | 38.6 s | `serve_timings.json` `requests[0]`, `f3b/qwen_serve_50steps.png` |
| request 2 (resident worker, no load) | 200 | 39.2 s | `serve_timings.json` `requests[1]` |
| GPU memory while serving | 60,893 MiB (`nvidia-smi`) | | status-log only |
| `extra_body.guidance_scale=1.0` | 400 `feature_not_supported` ("guidance_scale has no effect on Qwen-Image (guidance_embeds=False) … omit it") | | status-log only |
| `extra_body.height=768` | 400 `profile_mismatch` ("request shape 768x1024 is not in the serving profile's compiled shape set ['1024x1024']") | | status-log only |
| two concurrent requests | first 200 in 39.1 s; second **429 `queue_timeout`** ("request waited too long for the resident worker") | 30.0 s | status-log only |
| `difflet serve --model-id Qwen/Qwen-Image --tp-degree 2` | `400 feature_not_supported: [qwen_image/cuda serve] --tp-degree > 1 (tensor parallel across PCIe) is not supported on the CUDA backend in Release 1 (plan follow-up F6)`, exit 1, before any download | | status-log only |
| serving overhead | none. `cuda_smoke.py --steps 50` on the same pod: load 47.2 s, generate 38.5 s | | status-log only |

The engine's default `queue_timeout` (30 s, `difflet/serving/engines/resident_worker.py`) is
shorter than one 50-step request on this card. A deployment that serializes requests must raise
it above the per-request time. This was recorded, not changed: a LIMIT.

**How to inspect manually.**

```bash
python3 -m json.tool artifacts/cuda-release1/f3b/serve_timings.json   # ready_s, requests[].request_s
open artifacts/cuda-release1/f3b/qwen_serve_50steps.png
# on a pod:
HF_HOME=$H scripts/cuda_serve_smoke.sh --output $HOME/logs/qwen-serve.png --steps 50 --requests 2
# probes against a running `HF_HOME=$H DIFFLET_BACKEND=cuda .venv/bin/difflet serve --model-id Qwen/Qwen-Image`
# (port 8091): the exact curl helper is in .claude/skills/difflet-cuda-verify/SKILL.md §3
```

### Serving on two GPUs (F2b, `--dp-degree N`)

Pod f2b (2 × **Server**, PCIe `PHB`), checkout 3d8b026+ (branch `feat/cuda-f2b-f4b-two-gpus`),
`HF_HOME=~/hf`, offline *[10-18 F2b/F4b]*. `difflet serve --dp-degree N` keeps N complete
resident workers, one GPU each (the worker exports `CUDA_VISIBLE_DEVICES` from its plan's core
id), behind one `ReplicatedServingEngine` that routes each request to the replica with the fewest
pending requests; at `--dp-degree 1` the engine path is F3b's, and the worker now pins its GPU
(`CUDA_VISIBLE_DEVICES=0`) where F3b's inherited the parent's environment. 50 steps, 1024², seed 0, the same
prompt; every image compared by bytes with the `--dp 1` run's first image.

| Run (`scripts/cuda_serve_smoke.py`) | ready | requests | wall | identical to the dp 1 image | Artifact |
|---|---|---|---|---|---|
| `--dp 1 --requests 2` (reference) | 44.0 s | 16.7 / 16.7 s (serial) | — | reference (md5 `c74b4d32…`) | `f2b/serve/serve_dp1/serve_timings.json`, `qwen_50steps.png` |
| `--dp 2 --concurrent 2` | 30.0 s (two loads in parallel; likely the page cache after the dp 1 run — not measured) | 16.7 / 16.7 s | **16.7 s** | both | `f2b/serve/serve_dp2/serve_timings.json` |
| `--dp 2 --concurrent 3 --queue-timeout 120` | 14.0 s (third start of the same snapshot; page cache — not measured) | 16.7 / 16.7 / 33.3 s | 33.3 s (the third queued behind a running one) | all three | `f2b/serve/serve_dp2c3/serve_timings.json` |
| GPU memory while serving (42 s into the 3-request run) | | one worker per card, 60,893 MiB each (pids 3088 / 3089), GPU 0 at 98 % serving the third | | | `f2b/serve_dp2c3_nvsmi.log` |
| `difflet serve --model-id Qwen/Qwen-Image --dp-degree 3` | `400 feature_not_supported: --dp-degree 3 exceeds the 2 visible GPU(s) (CUDA_VISIBLE_DEVICES); one resident replica needs one GPU`, exit 1, before any download | | | | `f2b/serve/serve_dp3_refusal.log` |
| `DIFFLET_BACKEND=cpu difflet serve … --dp-degree 2` | `400 feature_not_supported: --dp-degree 2: data-parallel serving replicas (one resident worker per GPU) exist on the CUDA backend only (plan follow-up F2b) …` | | | | status-log only |

The F3b LIMIT (the second concurrent request got 429 after the 30 s default) is lifted by one
replica per GPU: two concurrent requests complete in one request-time; a third still waits for a
free replica, so `--queue-timeout` must exceed a request when clients queue beyond N.

**How to inspect manually.**

```bash
for d in serve_dp1 serve_dp2 serve_dp2c3; do python3 -m json.tool artifacts/cuda-release1/f2b/serve/$d/serve_timings.json; done
cat artifacts/cuda-release1/f2b/serve_dp2c3_nvsmi.log
# on a 2-GPU pod (after a --dp 1 run that wrote $HOME/logs/x/serve_dp1/qwen.png):
HF_HOME=$H scripts/cuda_serve_smoke.sh --output $HOME/logs/x/serve_dp2/qwen.png --steps 50 \
    --dp 2 --concurrent 2 --reference $HOME/logs/x/serve_dp1/qwen.png
```

## Performance (F4, `scripts/cuda_bench_dit.py`) — Server Edition only

Qwen-Image DiT forward, bf16, B = 1, 1024×1024, 1024 text tokens of which 16 are valid. Median
of 10 timed iterations after 3 warm-ups, profilers off; then one `torch.profiler` run for the
kernel categories (the pod image has no `nsys`). Pod f3c (**Server**) *[10-17 F4]*.

| Case | Median ms (of 10) | attention / gemm / elementwise / copy ms | Kernel total ms | Attention kernel | Artifact |
|---|---|---|---|---|---|
| Difflet, padded + mask (baseline) | 515.7 | 177.2 / 236.0 / 65.0 / 24.8 | 506.6 | mem-efficient `fmha_cutlassF` | `f4/bench_dit_baseline.json` `difflet_masked` |
| Difflet, all-ones mask | 516.0 | 177.3 / 236.2 / 65.1 / 24.8 | 506.9 | mem-efficient | `bench_dit_baseline.json` `difflet_allones` |
| diffusers, same padded input | 489.6 | 177.2 / 236.3 / 44.8 / 20.7 | 482.4 | mem-efficient | `bench_dit_baseline.json` `diffusers_masked` |
| diffusers, its real 16-token input | 299.6 | 44.1 / 190.8 / 37.3 / 17.7 | 292.7 | flash | `bench_dit_baseline.json` `diffusers_trimmed` |
| Difflet, trimmed prompt (F4 task 2) | 323.5 | 44.2 / 190.8 / 54.8 / 22.1 | 314.6 | flash | `f4/bench_dit_trimmed.json` |
| Difflet, trimmed + `--torch-compile` | 242.8 | 44.7 / 180.5 / 12.1 / 0.0 | 240.1 | flash | `f4/bench_dit_compiled.json` (first forward 31.1 s) |

Before the trim, Difflet was 5.3 % slower than upstream on identical inputs (515.7 vs 489.6 ms), so
the whole gap was the fixed-length prompt. Pipeline level, 50 steps, same pod: 26.1 → 16.5 s
(trim) → 12.4 s (compile); diffusers' pipeline takes 15.2 s. These tables compare only within
the Server Edition. Characterising Workstation vs Server is the open F4b follow-up.

**How to inspect manually.**

```bash
python3 -c "import json
for f in ['baseline', 'trimmed', 'compiled']:
    d = json.load(open(f'artifacts/cuda-release1/f4/bench_dit_{f}.json'))
    for c, v in d['cases'].items(): print(f, c, round(v['timed']['median_ms'], 1), v['profile']['by_category_ms'])"
# on a pod (Server Edition to compare against these rows):
python scripts/cuda_bench_dit.py --model-path "$SNAP" --no-trim \
    --out $HOME/logs/x/bench_dit_baseline.json
python scripts/cuda_bench_dit.py --model-path "$SNAP" --cases difflet_masked \
    --out $HOME/logs/x/bench_dit_trimmed.json
python scripts/cuda_bench_dit.py --model-path "$SNAP" --cases difflet_masked --torch-compile \
    --out $HOME/logs/x/bench_dit_compiled.json
```

### Second Server-Edition data point (F4b, pod f2b)

`cuda_bench_dit.py` on pod f2b (another Server Edition card, eu-2): Difflet trimmed 322.7 ms
(f3c: 323.5), diffusers padded 489.7 (489.6), diffusers trimmed 299.7 (299.6); Difflet with an
all-ones mask and the trim on 402.5 ms (no f3c counterpart). The same edition reproduces the
F4 table to the millisecond; the Workstation edition was not on this pod, so the
Workstation-vs-Server characterisation stays unmeasured (`f2b/bench/bench_dit_qwen.json`).

### Wan DiT bench and the fused q/k norm (F4b, `scripts/cuda_bench_wan_dit.py`)

One Wan 2.2 expert forward on identical seeded inputs (latents `(1, 16, 3, 60, 104)`, timestep
999 in fp32, text `(1, 512, 4096)`, bf16), Difflet's `CudaWanTransformerApplication` vs diffusers'
`WanTransformer3DModel`, median of 10 after 3 warm-ups, then one profiled forward. Pod f2b
(**Server**) *[10-18 F2b/F4b]*.

| Case | Difflet ms | diffusers ms | ratio | gemm / attention / elementwise / copy / norm (Difflet vs diffusers) | Artifact |
|---|---|---|---|---|---|
| `transformer`, before the fix (d8ca0de) | 604.5 | 526.4 | 1.148× | 286.8 vs 286.9 / 61.6 vs 61.2 / **145.4 vs 94.5** / **95.1 vs 61.3** / 11.9 vs 19.6 | `f2b/bench/bench_wan_dit_transformer.json` |
| `transformer_2`, before | 604.5 | 526.8 | 1.148× | 286.0 vs 286.5 / 61.5 vs 61.2 / 145.4 vs 94.5 / 95.1 vs 61.3 / 11.9 vs 19.6 | `bench_wan_dit_transformer_2.json` |
| `transformer`, `--cfg-batch 2`, before | 1305.7 | 1094.8 | 1.193× | — | `bench_wan_dit_transformer_cfg2.json` |
| `transformer`, after the fix (83d244a) | **536.2** | 527.9 | **1.016×** | 287.4 vs 288.8 / 61.0 vs 61.2 / 101.3 vs 94.6 / 63.8 vs 61.3 / 19.3 vs 19.6 | `bench_wan_dit_transformer_fused.json` |

Both sides run the same gemm and the same flash-attention kernel; the whole gap was
elementwise and copy work, which the code read traced to `WanAttention._global_rms_norm`: the
across-heads q/k RMSNorm at tp=1 made seven fp32 passes (two fp32 copies, pow, sum, mul, affine,
cast) per call, three large calls per block, forty blocks. The fix (commit 83d244a) is a new op,
`difflet.ops.rms_norm_fp32`: the CPU, TPU and Trainium backends export the old math verbatim
(bit for bit, unit-tested), the CUDA backend `torch.nn.functional.rms_norm` (one fused kernel,
the same fp32 math up to the reduction order). The loop-overhead probe (`--loop-probe`: the real
`WanOrchestrator` loop with probe experts) puts the host loop at 1.35 ms/step — 54 ms of a
40-step run, 80 DiT calls (26 high-noise, 54 low-noise) — so the orchestrator was never the gap
(a second probe read 184 ms/step while the other card's gate run was loading weights: host
contention, not the loop). Pipeline level, 40 steps, seed 0, same pod: Difflet generate
50.3 s (old code) → **44.5 s**, diffusers 42.8 s — the F1b/F9 gap of 8.1 s is 1.8 s. The fix is a
rounding-level change: old vs new code on seed 0 differ by PSNR 28.45 dB (mean_abs 0.041,
`f2b/wan_dp/compare_wan_dp.json`), the noise floor's own range; the Wan gates on the new code
(the e2e table and the parity section's F2b rows) pass with the F1b parity numbers to the
third digit and e2e 27.91 dB.

**How to inspect manually.**

```bash
python3 -c "import json
for f in ['transformer', 'transformer_2', 'transformer_cfg2', 'transformer_fused']:
    d = json.load(open(f'artifacts/cuda-release1/f2b/bench/bench_wan_dit_{f}.json'))
    for c, v in d['cases'].items(): print(f, c, round(v['timed']['median_ms'], 1), {k: round(x, 1) for k, x in v['profile']['by_category_ms'].items()})
    print('  loop probe:', d.get('loop_probe'))"
# on a pod:
python scripts/cuda_bench_wan_dit.py --model-path "$WAN" --expert transformer --loop-probe --out $HOME/logs/x/bench_wan_dit_transformer.json
python scripts/cuda_bench_wan_dit.py --model-path "$WAN" --expert transformer_2 --out $HOME/logs/x/bench_wan_dit_transformer_2.json
```

### Flux DiT bench (F4c, `scripts/cuda_bench_flux_dit.py`) and the attention layout copies (F4d)

Pod f4d (1 × **Server**, eu-3, 38 min) *[10-19 F4c/F4d]*. One FLUX.1-dev DiT forward on
identical seeded inputs (packed latents `(1, 4096, 64)`, text `(1, 512, 4096)`, pooled `(1,
768)`, `t / 1000` = 0.5, guidance 3.5, the rope from the pipeline's ids), median of 10 after 3
warm-ups, one profiled forward.

| Case | Difflet ms | diffusers ms | ratio | gemm / attention / elementwise / copy / norm (Difflet vs diffusers) | Artifact |
|---|---|---|---|---|---|
| FLUX.1-dev DiT forward | **311.9** | 290.5 | 1.073× | 183.0 vs 173.1 / 47.1 vs 47.2 / **53.7 vs 42.7** / **21.6 vs 14.1** / **1.5 vs 10.0** | `f4d/bench/bench_flux_dit.json` |
| FLUX.1-dev DiT forward, after the fused q/k norm (F4e, pod f4e) | **303.0** | 292.0 | **1.038×** | 184.2 vs 174.4 / 47.0 vs 47.3 / 39.3 vs 42.8 / 20.5 vs 14.1 / 8.3 vs 10.0 | `f4e/bench/bench_flux_dit_f4e.json` |

Attention is identical (the same flash kernel). The 21 ms gap has two parts, both read from
the kernel buckets: (1) **unfused norms** — diffusers spends 10.0 ms in
`vectorized_layer_norm_kernel`, Difflet 1.5 ms in norm kernels and pays it back as +11 ms of
elementwise and +7.5 ms of casts (`direct_copy_kernel` 16.8 vs 10.1 ms): the Flux modeling's
`NeuronAdaLayerNormZero` / `LayerNorm` run as elementwise passes rather than a fused kernel,
the Wan q/k-norm pattern again (F4b); (2) **+10 ms of gemm from different kernel shapes** —
Difflet's largest gemm kernel runs 167.9 ms where diffusers splits the same work across two
(99.4 + 62.1 ms), the parallel-layer projections presenting different `(M, N, K)` to cuBLAS.
Per 28-step run the forward gap is 0.6 s of the measured 0.8–1.1 s. Neither part is a
30-minute fix; the fused-norm half is the next contained change (F4e), the gemm half needs a
look at the projection layout. The Flux gates on this pod pass (parity `sdpa` rel_l1 0.987×
the control, same-dtype 0.398×, cosine 0.99993; e2e 40.35 dB / 0.99980, generate 9.1 s vs
diffusers 8.3 s; `f4d/gates-flux/`).

**F4e, measured (pod f4e, 28 min):** the CUDA backend's `RMSNorm` as one fused `F.rms_norm`
(commit `5a92098`, Flux's q/k norms) takes the forward from 311.9 to **303.0 ms** against
292.0: the norm bucket is fused (1.5 → 8.3 ms), the elementwise excess is gone (53.7 → 39.3,
now under diffusers' 42.8), casts 21.6 → 20.5. What remains (+11 ms) is the gemm half (184.2
vs 174.4) and +6 ms of casts. Pipeline level: generate **8.8 s vs 8.3 s** (from 9.4 / 9.1), and
the Flux e2e match improved to **41.58 dB / 0.99985** (from 40.35, the fused norm being
diffusers' own numerics); parity `sdpa` 0.990× the control (0.987×), same-dtype 0.396×. The
Qwen and Wan gates on the same build are unchanged (Qwen 41.20 dB, parity 0.997× / 0.992×;
Wan 27.91 dB, 1.051× / 1.043×) — Qwen's q/k norm is diffusers' own class and Wan's was fused
in F4b (`f4e/gates-{qwen,wan,flux}/`).

**F4d, measured: no effect.** The `attention_heads` op (the models' `(B, H, S, D)` views
straight to SDPA, no fold copies) changed nothing: Qwen `difflet_masked` 322.7 ms (322.7
before), Wan `transformer` 535.1 ms (536.2), the copy buckets identical (22.1 / 63.8 ms). The
fold copies are three 48 MB tensors per attention — tens of microseconds each, not the 8 ms
the F4b entry estimated (an arithmetic error, now corrected in the status log). The op is
reverted on the branch; the session's gates (all three models, F4d code) pass and are kept as
evidence (`f4d/gates-{qwen,wan,flux}/`).

**How to inspect manually.**

```bash
python3 -c "import json
for f in ['bench_flux_dit', 'bench_dit_qwen_f4d', 'bench_wan_dit_transformer_f4d']:
    d = json.load(open(f'artifacts/cuda-release1/f4d/bench/{f}.json'))
    for c, v in d['cases'].items(): print(f, c, round(v['timed']['median_ms'], 1), {k: round(x, 1) for k, x in v['profile']['by_category_ms'].items()})"
# on a pod:
python scripts/cuda_bench_flux_dit.py --model-path "$FLUX" --out $HOME/logs/x/bench_flux_dit.json
```

### `--torch-compile` for Wan and Flux, the fused Flux projection, the Flux noise floor (F4f)

Pod f4f (1 × **Server**, eu-2, 56 min), branch `feat/cuda-f4f-compile-wan-flux` *[10-19 F4f]*.
`--torch-compile` (D4's measured, opt-in option; per-block inductor, `dynamic=False` for these
static shapes) now reaches Wan's experts and Flux's DiT; the Flux single block's `proj_out`
runs as diffusers' one gemm over `cat([attn, mlp])` at tp=1 (`fuse_proj_out`, load-time).

| Run | eager | compiled | diffusers | gates |
|---|---|---|---|---|
| Wan 2.2, 40 steps, generate (`run_gates.sh wan` e2e, seed 0) | 45.0 s (27.91 dB) | **32.8 s** (27.49 dB) | 43.2 s | both PASS; compiled parity rounding block 0.947× / 0.928× the control (`f4f/gates-wan-compiled/`) |
| Wan 2.2 smoke, 40 steps | 46.1 s (1.15 s/step) | 33.3 s (0.83 s/step; load 82.6 s incl. the compiles; peak 73,259 MiB) | — | `f4f/gates-wan-compiled/cuda-smoke-wan-*.json` |
| Flux DiT forward (fused projection) | **299.6 ms** (gemm 177.3 / attention 47.1 / elementwise 43.0 / copy 20.5 / norm 8.3) | **241.3 ms** (172.8 / 50.7 / 10.9 / 0.0 / 3.8) | 291.4 ms (173.6 / 47.3 / 42.8 / 14.2 / 10.0) | `f4f/bench/bench_flux_dit_f4f{,_compiled}.json` |
| Flux, 28 steps, generate | 8.8 s | **7.1 s** | 8.3 s | parity 0.987× / 0.958× the control, PASS; e2e below |
| Qwen-Image (regression net) | 41.20 dB, parity 0.997× | — | — | PASS, unchanged (`f4f/gates-qwen/`) |

Eager Flux with the fused projection is 1.028× diffusers per forward (F4e: 1.038×; the gemm
bucket 177.3 vs 173.6); compiled Flux is 0.83× diffusers per forward (0.86× per run) and
compiled Wan 0.76× per run.

**The Flux e2e gate and the noise floor.** On seed 0 both the fused eager run (31.56 dB) and the
compiled run (31.41 dB) failed the single-seed 35 dB floor that F5/F5c passed at 40.35–41.58 dB.
An A/B with the projection kept split (`DIFFLET_CUDA_FLUX_FUSE_PROJ_OUT=0`) read 41.58 dB on
seed 0 but 39.71 / 40.99 dB on seeds 1 / 2 against 39.56 / 43.79 dB fused: not a systematic
loss. The metric's own floor, measured as F9 did for Wan (`--noise-floor`: diffusers against
itself with `allow_bf16_reduced_precision_reduction` flipped, a rounding-level perturbation of
every bf16 gemm), reads **30.92 / 36.62 / 37.61 dB** over seeds 0–2: seed 0 is a chaotic seed
for diffusers too, and Difflet fused (31.56 / 39.56 / 43.79) and compiled (31.41 / 39.79 /
42.53) sit at or above the floor on every seed. The Flux gate is therefore stated relative to a
recorded floor (as F9b's Wan gate: PSNR ≥ floor − 1.21 dB, cosine deviation ≤ 1.3225× the
floor's, a loose 20 dB / 0.98 absolute floor), the 35 dB / 0.998 floors judging results
without one; the three floor runs pass it (`f4f/floor/e2e_floor_seed{0,1,2}.json`), the
seed-0 runs without a floor fail the absolute floor as recorded (`f4f/gates-flux{,-compiled}/`).
Three seeds is a small sample: the floor's spread (6.7 dB) says Flux's single-seed e2e PSNR
carries several dB of noise, not that any build is within it. Since F9c (on the CPU; the pod
run is pending) the script runs many seeds in one load per side (`--seeds … --noise-floor`) and
the gate judges them as F9b judges Wan — the mean over seeds within 1.21 dB of the floor's mean
(gate A), every seed within 1.21 dB of the floor's worst seed (gate B); the three floor runs
above, merged into one result, pass it (mean 38.30 dB against 33.84 required, worst seed
31.56 against 29.71; `tests/unit/scripts/test_cuda_flux_gates.py`).

**How to inspect manually.**

```bash
python3 -c "import json
for s in (0, 1, 2):
    d = json.load(open(f'artifacts/cuda-release1/f4f/floor/e2e_floor_seed{s}.json'))
    print(s, round(d['metrics']['psnr_db'], 2), 'floor', round(d['noise_floor']['metrics']['psnr_db'], 2), d['noise_floor']['perturbation'])"
export DIFFLET_BACKEND=cpu DIFFLET_RUN_CUDA_NUMERICAL=1
DIFFLET_CUDA_FLUX_E2E_RESULT=artifacts/cuda-release1/f4f/floor/e2e_floor_seed0.json pytest -q -rA tests/numerical/test_cuda_flux_e2e_vs_diffusers.py   # 4 passed: the relative gate prints its thresholds
# on a pod: DIFFLET_CUDA_TORCH_COMPILE=1 run_gates.sh wan|flux; cuda_e2e_flux_vs_diffusers.py --seed N --noise-floor
# since F9c: cuda_e2e_flux_vs_diffusers.py --seeds 0,1,2,... --noise-floor (one load per side; the gate over seeds)
```

### The Flux e2e over eight seeds and the gate over seeds (F9c)

Pod `difflet-cuda-f9c3` (us-east-3, **Workstation** × 1, the `difflet-cache` volume: no
download, no login), checkout `f643e46`: `cuda_e2e_flux_vs_diffusers.py --seeds 0,…,7
--noise-floor`, eager, then with `DIFFLET_CUDA_TORCH_COMPILE=1`, then `run_gates.sh flux` and its
new three-seed e2e step *[10-19 F9c verdict]*. PSNR / cosine against diffusers' image from the
same noise, and the floor (diffusers against itself with bf16 gemm reductions flipped):

| seed | Difflet eager | Difflet `--torch-compile` | noise floor |
|---|---|---|---|
| 0 | 31.56 dB / 0.99851 | 42.27 dB / 0.99987 | 30.92 dB / 0.99827 |
| 1 | 39.56 dB / 0.99976 | 41.31 dB / 0.99984 | 36.62 dB / 0.99952 |
| 2 | 43.79 dB / 0.99991 | 40.52 dB / 0.99981 | 37.61 dB / 0.99964 |
| 3 | 41.27 dB / 0.99985 | 43.01 dB / 0.99990 | 31.90 dB / 0.99868 |
| 4 | 35.61 dB / 0.99931 | 36.88 dB / 0.99950 | 33.05 dB / 0.99876 |
| 5 | 44.14 dB / 0.99993 | 39.76 dB / 0.99980 | 39.44 dB / 0.99979 |
| 6 | 33.46 dB / 0.99899 | 34.24 dB / 0.99915 | 35.14 dB / 0.99931 |
| 7 | 24.89 dB / 0.99276 | 28.50 dB / 0.99684 | 24.24 dB / 0.99159 |
| min / mean / max | 24.89 / 36.79 / 44.14 dB | 28.50 / 38.31 / 43.01 dB | 24.24 / 33.61 / 39.44 dB |

**Gates.** Gate A (the mean): eager 36.79 dB against 32.40 required (the floor's mean 33.61 −
1.21 dB; 4.4 dB headroom, about 2.6 standard errors of the floor's mean at n = 8, whose seeds
spread with a standard deviation of 4.8 dB), compiled 38.31 dB (5.9 dB); the mean cosine
deviation 0.00137 / 0.00066 against 0.00239 allowed. Gate B (every seed): at least 23.02 dB
(the floor's worst seed 24.24 − 1.21) and a deviation of at most 0.01113; the worst Difflet
seed is 24.89 dB eager, 28.50 compiled. Both pass (`f9c/eager/gate.log`, `f9c/compiled/gate.log`).
`run_gates.sh flux` (eager): `GATES DONE flux overall_rc=0` — smoke 44 s (a cat in a sunlit
garden), parity 107 s at 0.987× the control and 0.387× same-dtype (F4f's numbers), e2e 149 s
over seeds 0–2 with gate A 38.30 dB against 33.84 (`f9c/gates-flux/`).

**Reproduction.** Eager seeds 0–2, and their floors, equal F4f's three single-seed runs to the
printed digits — on a Workstation card where F4f's ran on a Server card, and with all seeds in
one process: nothing carries over between seeds. `--torch-compile` is not reproducible run to
run (max-autotune picks kernels by timing): its seed 0 reads 42.27 dB here and 31.41 dB on F4f.

**Observations.** (1) Seed 7 is a chaotic seed for diffusers itself (24.24 dB from its own
rounding noise); with eight seeds the floor's worst seed fell from 30.92 to 24.24 dB, and gate
B's threshold from 29.71 to 23.02 dB — with more seeds gate B becomes the backstop and gate A
the judge. (2) Matched seed by seed, Difflet is above its own floor on 7 of 8 seeds in both
modes; eager seed 6 is 1.68 dB under it (33.46 vs 35.14; compiled 0.90). The gate compares with
the floor's spread, not seed by seed (F9b's ruling: the perturbation moves diffusers by several
dB between seeds), so this is inside the gate's design; a per-seed test at 1.21 dB would fail
eager seed 6. **Timings (Workstation):** generate eager 8.7–9.1 s against diffusers 8.6–8.7 s,
compiled 7.5–7.8 s; Difflet's first load 154.9 s (a cold read from the volume, ≈ 0.22 GB/s), the
compiled load 77.8 s (warm page cache plus compiles); peak 36.33 / 36.32 GB.

**Getting the pod.** The v2 create API attaches R2 volumes only (12016) and drops an unknown
volume key without an error (`difflet-cuda-f9c2` came up without `/data`); v1's create with
`persistentVolumes: [{id, volumeType: "Ceph", mountMode: "Data", mountPath}]` attached
`difflet-cache` (`f643e46`, `2497d32`; `yottalabs_pod.py create --volume-id` now sends it).

```bash
export DIFFLET_BACKEND=cpu DIFFLET_RUN_CUDA_NUMERICAL=1 A=artifacts/cuda-release1/f9c
DIFFLET_CUDA_FLUX_E2E_RESULT=$A/eager/cuda_e2e_flux_noise.json pytest -q -rA tests/numerical/test_cuda_flux_e2e_vs_diffusers.py     # 4 passed; gate A/B lines, per-seed table
DIFFLET_CUDA_FLUX_E2E_RESULT=$A/compiled/cuda_e2e_flux_noise.json pytest -q -rA tests/numerical/test_cuda_flux_e2e_vs_diffusers.py  # 4 passed
DIFFLET_CUDA_FLUX_E2E_RESULT=$A/gates-flux/cuda_e2e_flux.json pytest -q -rA tests/numerical/test_cuda_flux_e2e_vs_diffusers.py      # 4 passed
cat $A/gates-flux/gates.txt; open $A/eager/flux_e2e_{difflet,diffusers}_seed7.png   # the chaotic seed, by eye
```

## Two GPUs (F2)

Pod f2: 2 × **Server** over PCIe `PHB`, no NVLink, checkout aef79dd *[10-18 F2 verdict]*. NCCL
smoke (all_gather + broadcast_object_list over both cards) took 11 s (status-log only). The log
shows `NCCL version 2.28.9+cuda12.9` and both ranks report `gathered [0.0, 1.0] box {'req': 7}`.

| Run | Load | Generate | Speed-up | Identity vs single-GPU | Peak MiB per card | Artifact |
|---|---|---|---|---|---|---|
| Wan 2.2 serial, one card | 55.9 s | 50.9 s (1.27 s/step) | 1× | reference | — | `f2/timings.txt` |
| Wan 2.2 `--cfg-parallel` (torchrun, 2 ranks) | 22.8 s per rank (likely the page cache — not measured) | 26.1 s (0.65 s/step) | **1.95×** | max_abs 0, mean_abs 0, PSNR ∞ (`.pt` bytes differ: pickling) | 68,074 / 67,519 | `f2/compare_wan.json`, `f2/wan_cfg_nvsmi.log` |
| Qwen-Image, 4 serial `difflet run` (seeds 0–3) | 17.6–44.0 s each | 16.8–17.3 s each | — | references | — | `f2/timings.txt`; 190 s total status-log only |
| Qwen-Image `--dp 1 --requests` (one card, batch) | 8.6 s (once) | 16.7–16.8 s per image | 79 s wall (status-log only) | requests 0 and 3 byte-identical (`identical_bytes: true`); 1 and 2 not compared | — | `f2/compare_qwen.json` `[4]`, `[5]` |
| Qwen-Image `--dp 2 --requests` (router `worker 0: GPUs 0`, `worker 1: GPUs 1`) | 21.4 / 21.5 s in parallel | 16.7–16.8 s per image | **58 s wall** vs 190 s serial (3.3×, computed) and 79 s `--dp 1` (1.36×, computed) | all four byte-identical | — | `f2/compare_qwen.json` `[0]`–`[3]`, `f2/qwen_dp2_request0_50steps.png` |
| Wan 2.2, 2 serial `difflet run` (seeds 0, 1), pod f2b, checkout 83d244a | 12.9 s (seed 0; seed 1 not recorded) | 44.5 / 44.6 s | — | references | — | `f2b/wan_dp/compare_wan_dp.json` `serial_generate_s`, `serial_seed0.log` *[10-18 F2b/F4b]* |
| Wan 2.2 `--dp 2 --requests` (two 40-step requests, seeds 0, 1), pod f2b | 20.6 / 20.8 s in parallel | 51.2 / 51.2 s (slower than 44.5 s alone: two replicas at once — contention not measured) | **76 s wall** vs 120 s serial (1.58×, computed) | both videos bit-identical (max_abs 0) | — | `f2b/wan_dp/compare_wan_dp.json` `comparisons[0..1]`, `dp2.log` |
| FLUX.1-dev, 2 serial `difflet run` (28 steps, seeds 0, 1), pod flux (1 × Server) | 4.4 / 4.2 s | 9.4 / 9.4 s | wall 17 + 16 s | references (md5 `34f56881…`, `696f82bb…`) | — | `flux/f5c/dp/serial_seed0.log`, `md5sums.txt` *[10-18 F5c]* |
| FLUX.1-dev `--dp 1 --requests` (one card, batch), pod flux | 5.0 s (once) | 9.4 / 9.2 s | **28 s wall** vs 33 s serial | both PNGs byte-identical to the serial ones (md5) | — | `flux/f5c/dp/dp1.log`, `md5sums.txt`, `requests.jsonl` |

The extra ~555 MiB on GPU 0 is likely the parent's CUDA context; it was not measured separately.
After each run no process was left on either card. The `cosine` field in both comparison JSONs is
not meaningful: it reads 0.999496–1.000116 on pairs with max_abs 0 (identical pixels), so the
ad-hoc comparison helper, which is not in the repo, computed it in low precision. The identity
evidence is `max_abs`, `mean_abs` and `identical_bytes`. The four F2 request prompts are in no
artifact; the controller recorded them from the session runbook in the F8 status-log entry
*[2026-10-18 — F8]*: seed 0 "A red fox in fresh snow, golden hour, photorealistic", seed 1 "A
neon-lit Tokyo alley in the rain at night", seed 2 "A bowl of ramen on a wooden table, studio
lighting", seed 3 "A lighthouse on a cliff during a storm, oil painting" (50 steps, 1024²).
`f2/qwen_dp2_request0_50steps.png` (seed 0) shows a red fox sitting in snow in golden light:
prompt-faithful by eye.

**How to inspect manually.**

```bash
python3 -m json.tool artifacts/cuda-release1/f2/compare_wan.json    # max_abs 0.0, mean_abs 0.0
python3 -m json.tool artifacts/cuda-release1/f2/compare_qwen.json   # six pairs, identical_bytes true
grep -E "dp-router|loaded in|wrote" artifacts/cuda-release1/f2/timings.txt
cat artifacts/cuda-release1/f2/wan_cfg_nvsmi.log                    # index, memory.used, util — both cards busy
cat artifacts/cuda-release1/f2/nccl_smoke.log
# on a 2-GPU pod: the exact serial / --cfg-parallel / --dp 2 commands are in
# .claude/skills/difflet-cuda-verify/SKILL.md §3 ("Two GPUs (F2)")
```

## A/B: the Wan timestep dtype (F1b)

Both variants share one diffusers run, one noise tensor and one Difflet run each. Pod f1b
(**Server**, GPU 0), checkout 38bff3c *[10-18 F1b; F1b correction (review)]*.

| Measurement | fp32 timestep (`float32`, default) | bf16-cast timestep (`model`) | Control | Artifact |
|---|---|---|---|---|
| DiT `transformer` @ 999 (bf16 holds 1000): rel_l1 vs fp32 (× control) | 5.264e-2 (1.039×) | 5.494e-2 (1.084×) | 5.068e-2 | `wan/f1b/cuda_parity_wan.json` `experts.transformer.rounding` |
| same, same-dtype (× control) / cosine vs bf16 | 4.076e-2 (0.804×) / 0.999124 | 4.426e-2 (0.873×) / 0.998985 | | same |
| DiT `transformer_2` @ 750 (bf16 holds 752): rel_l1 (× control) | 2.449e-2 (1.018×) | 2.453e-2 (1.020×) | 2.406e-2 | `experts.transformer_2.rounding` |
| same, same-dtype (× control) / cosine vs bf16 | 1.933e-2 (0.803×) / 0.999795 | 1.956e-2 (0.813×) / 0.999788 | | same |
| e2e PSNR / cosine / mean_abs | 26.30 dB / 0.99609 / 0.0274 | 27.54 dB / 0.99704 / 0.0238 | | `wan/f1b/cuda_e2e_wan.json` `variants` |
| Difflet load / generate | 22.7 s / 50.9 s | 10.3 s / 50.2 s | diffusers 26.1 s / 42.9 s | same, `timings` |
| first frame | `wan/f1b/e2e_difflet_fp32_timestep_frame0.png` | `wan/f1b/e2e_difflet_bf16_timestep_frame0.png` | `wan/f1b/e2e_diffusers_frame0.png` | all the same calico cat |

Verdict as corrected by review. Exact timesteps bring every DiT call closer to diffusers, yet the
40-step trajectory ends 1.2 dB further away. The single-seed A/B does not support the timestep
as the cause of the e2e gap, and does not exclude it either (F9's noise floor, in the e2e
section, puts the whole 1.2 dB move inside the range the noise floor spans over seeds 0–2). The fp32 default
stays (ruling). The
artifact PNG names differ from the script's: `e2e_difflet_fp32_timestep_frame0.png` is the
script's `wan_e2e_difflet_float32_frame0.png`, and `e2e_difflet_bf16_timestep_frame0.png` is
`wan_e2e_difflet_model_frame0.png`.

**How to inspect manually.** The gate re-runs above (`wan/f1b/*.json`: 11 + 3 passed). On a pod:
`bash .claude/skills/difflet-cuda-verify/scripts/run_gates.sh wan auto --out $HOME/logs/<name>`
runs the rounding block and `--timestep-dtypes float32,model` as its parity and e2e steps.

## Rulings and bugs ledger

One line each: commit, symptom or decision, what changed. Commits are on the stacked branches
above. Spurious infrastructure events are kept and labelled.

1. `4d1d637` — SDPA chose the math kernel for the folded (B·H, S, D) layout. Attention now lifts
   3-D inputs to 4-D (flash, 1.2 ms per call). `cpu-gate.sh` `-rs` → `-rfEs` (failures were
   invisible) *[10-16 Phase 0]*.
2. `4d1d637` — the image's nightly torch was not used: released torch 2.11.0+cu128 pinned in
   `requirements-cuda.lock` *[10-16 Phase 0]*.
3. `f07fe25` — the orchestrator's `_timesteps` cannot drive Qwen's `use_dynamic_shifting`
   scheduler. `generate` builds linear sigmas + `dynamic_shift_mu`, guidance 4.0 *[10-16 Release 1]*.
4. `5d5007d` — tp/cp/dp > 1 failed at the first collective after loading. Now refused before any
   load; a no-op `guidance_scale` is refused *[10-16 Release 1 exit]*.
5. `40f6ce8` — cut from D2: `difflet run` not wired for CUDA in Release 1 → F3 *[10-16 Release 1 exit]*.
6. `e9fa3c0` — Difflet padded prompts to 1024 tokens and the shared modeling dropped the mask, so
   the DiT attended over ~1000 padding tokens. The CUDA DiT applies the mask; Trainium/TPU keep
   dropping it (ruling) *[10-16 F3 started]*.
7. `7e37581` — `difflet run` refused out-of-scope flags only after the 58 GB download and dropped
   `--guidance-scale`. Both are now refused up front *[10-16 F3 review]*.
8. `69624a7` — F3b review: the serving suite failed 97× under `DIFFLET_BACKEND=cuda` (conftest pins
   cpu). The worker re-resolved `refs/main` (now the pinned commit), a test leaked `NEURON_RT_*`,
   and the smoke script lacked S3 URLs and an API key *[10-16 F3b]*.
9. `abbcace` — Qwen e2e floors moved 25 dB / 0.99 → 35 dB / 0.998 from the measurement. The
   Release 1 parity JSON was overwritten by F3's superset (ruling) *[10-17 F3 GPU verdict]*.
10. status-log only — spurious: pods `difflet-cuda-f3` / `-f3b` lost the GPU within minutes of boot
    (NVML "Unknown Error", `/dev/nvidiactl` EPERM). Both were terminated unused; the third pod
    kept the card with a 30 s watchdog *[10-17 F3]*.
11. `a53821a` — trim the padded prompt before the DiT: exact for the image, flash back,
    516 → 323 ms *[10-17 F4]*.
12. `37c67c1` — `--torch-compile` opt-in (D4 unchanged). The e2e and bench scripts reset dynamo on
    teardown: compiled blocks pinned 41 GB and diffusers OOM'd once (R4) *[10-17 F4]*.
13. `cc04479` — F4 review: the profile summary double-counted op rows (artifacts regenerated),
    `dynamic=False` recompiled per prompt length (now `dynamic=True`), and `--torch-compile` was
    silently dropped off CUDA (now refused) *[10-17 F4]*.
14. `0e8befd` — F4 rulings: no `nsys` → `torch.profiler` (R1); FP8 → F4b (R2); no `flash-attn`
    (R3); CUDA graphs → F4b (R5); 30 dB as the usable TeaCache floor (R6) *[10-17 F4]*.
15. `37d67ba`, `90d13ad` — Flux split: `global_rank.rank` added only by the Trainium converter.
    Final review: `--teacache-calibration` refused on CUDA, and a no-clone converter path so the
    checkpoint is not held twice in host RAM *[10-17 F5]*.
16. `89e3fcd` — Flux e2e floors moved 25 / 0.99 → 35 / 0.998 from the measurement. The volume
    reads cold at ≈ 0.23 GB/s; `/data` mounts root-owned (chown once) *[10-18 F5 Task 5]*.
17. `ba37ddd` — skill rule: pausing the pod at the end of a GPU session is pre-authorised, API
    first (owner's pod left billing overnight on 10-17) *[10-18 F1 header]*.
18. status-log only — spurious: the F1 container restarted after 118 GB was written to `/tmp`,
    likely the 100 GB `containerVolumeInGb` record. Large downloads now go on the volume *[10-18
    F1; the cause: 10-18 F1b]*.
19. `544f034` — a Wan 2.2 snapshot without `transformer_2/` ran silently on one expert (now
    `FileNotFoundError`). Batch > 1 is refused naming F1b, and the parity gate requires both
    experts *[10-18 F1 review fixes]*.
20. `eca6f9d`, `38bff3c` — timestep dtype is a per-component property (fp32 on the CUDA DiT,
    model dtype on Trainium/TPU). The parity oracle gained the rounding block; e2e gained
    `--timestep-dtypes` *[10-18 F1b]*.
21. `f1413f0` — F1b review: the "Difflet combines CFG in fp32 on the host" claim was wrong (both
    sides combine in bf16). "Refuted" became "not supported by a single-seed A/B". The e2e gate
    judges `float32` by name, and `timestep_dtype=None` means the default *[10-18 F1b correction]*.
22. status-log only — capacity: `difflet-cuda-f5b` resume went INITIALIZING → PAUSED three times
    (us-east-3 full) *[10-18 F1b]*. eu 1-GPU requests returned code 12014 ("Pod insufficient
    resources"; `docs/superpowers/plans/2026-10-18-cuda-f1b-timestep.md` ledger), so a 2-GPU node
    was rented, and a pod without a volume can only be terminated *[10-18 F1b]*.
23. `aabbb24`, `aef79dd` — a failed batch request inside a torchrun world fails that worker fast.
    F2 review: the torchrun child dies with its parent (`PR_SET_PDEATHSIG`), `--torch-compile` is
    forwarded to DP workers, cfg is refused without a world-2 group, and `OMP_NUM_THREADS` is split
    per rank *[10-18 F2]*.
24. status-log only — observations not changed: the 30 s serving `queue_timeout` is shorter than a
    request (F3b); a `runpy` `RuntimeWarning` in each torchrun rank (F2b);
    `has_compiled_artifacts` is False in every fresh process, so each rank re-writes its manifest
    (F2).
25. `2387653` — F9 task 1: `cuda_e2e_wan_vs_diffusers.py` gains `--seeds` (one load per side,
    per-seed and aggregate metrics) and `--noise-floor` (diffusers again with both experts'
    timestep rounded through bf16); the gate judges every seed and prints the floor. The pod
    tool sends an explicit `--disk-gb` beside a volume (it used to drop it) *[10-18 F9]*.
26. the F9 docs commit — skill fixes from its first real use: `pod_setup.sh` resolves its checkout
    from its own path (run from `~` it exited 2, "not a Difflet checkout"); the device login is
    started only once the owner is at the phone (a 300 s code expired); the runbook records
    `--disk-gb` with a volume and the measured one-GPU eu path. Ruling R1: the e2e floors stay
    25 dB / 0.99 next to the measured noise floor *[10-18 F9]*.
27. status-log only — F9 plan deviations: the us-east-3 pool stayed full, so on the owner's
    "Switch to eu" the session ran on a one-GPU eu-3 pod without a volume (rulings R2, Wan
    symlinked into the volume's hub, and R3, pause at the end, did not apply); the pod was
    terminated, read back gone *[10-18 F9]*.
28. the F9b commit — the Wan e2e gate stated relative to the measured noise floor when the result
    carries it (mean over seeds within 1.21 dB of the floor's mean, every seed within 1.21 dB of
    the floor's worst seed, cosine deviation within 1.15² = 1.3225×, a loose 20 dB / 0.98
    absolute floor; a floor under the loose floor fails as "the oracle moved"); the 25 dB / 0.99
    floors judge results without a floor as before. No GPU run: the F9 JSONs re-gated on the
    Mac *[10-18 F9b]*.

29. `3d8b026` — F2b task 1: `difflet serve --dp-degree N` on CUDA, N resident workers one GPU
    each behind `ReplicatedServingEngine`; the worker pins `CUDA_VISIBLE_DEVICES` from its plan;
    refused off CUDA and above the GPU count before any download *[10-18 F2b/F4b]*.
30. `9ace97a`, `d8ca0de` — the serve smoke's `--dp` / `--concurrent` / `--reference`;
    `scripts/cuda_bench_wan_dit.py` with the loop-overhead probe; Flux e2e timings recorded
    *[10-18 F2b/F4b]*.
31. `83d244a` — F4b task 5: `difflet.ops.rms_norm_fp32`, the Wan q/k norm at tp=1 as one op,
    fused on CUDA (the reference math elsewhere, bit for bit): the expert forward 604.5 → 536.2 ms
    against diffusers' 527.9, the 40-step generate 50.3 → 44.5 s against 42.8 *[10-18 F2b/F4b]*.
32. status-log only — the DP router's claim markers under `~/.cache/difflet/work/dp` survive a
    run, so a second `--dp N --requests` over the same file reports `done=N` in seconds without
    generating; cleared by hand on f2b (F2b observation); Wan replicas generating at once take
    51.2 s each vs 44.5 s alone (unmeasured cause) *[10-18 F2b/F4b]*.
33. `c8794f6` — F2c: `write_manifest` starts a `--requests` run from an empty queue (stale
    `req_*` manifests and claim/done/failed markers removed first), so the item-32 footgun is
    closed; proven on the CPU with the router run twice over one work dir *[10-18 F2c]*.
34. status-log only — F5c (pod flux, 1 × Server, 22 min): Flux e2e timings recorded (Difflet
    9.4 s vs diffusers 8.3 s, 28 steps; PSNR 40.35 dB again) and Flux `--dp 1 --requests`
    byte-identical to serial; `--dp 2` for Flux still needs a 2-GPU pod *[10-18 F5c]*.
35. `a29d971` — F4c: `scripts/cuda_bench_flux_dit.py`; measured on pod f4d: Difflet 311.9 vs
    diffusers 290.5 ms, the gap in unfused norms (+11 elementwise, +7.5 casts, −8.5 norm) and
    gemm kernel shapes (+10 ms); the Flux gates pass *[10-19 F4c/F4d]*.
36. `7f44984` then reverted — F4d: the `attention_heads` op measured as no effect (Qwen 322.7,
    Wan 535.1 ms, copy buckets unchanged); the F4b estimate of 8 ms for the fold copies was an
    arithmetic error (tens of µs). Reverted on the branch; the three models' gates on the
    F4d code are kept as evidence *[10-19 F4c/F4d]*.
37. `5a92098` — F4e: the CUDA backend's `RMSNorm` / `CustomRMSNorm` is one fused `F.rms_norm`
    (Flux's q/k norms; found with a CPU aten op-count diff): Flux DiT 311.9 → 303.0 ms vs
    diffusers 292.0, generate 8.8 vs 8.3 s, e2e 40.35 → 41.58 dB; Qwen and Wan unchanged; all
    three models' gates pass on pod f4e *[10-19 F4e]*.
38. `d0cb725`, `4d7c8f3` — F4f: `--torch-compile` for Wan's experts and Flux's DiT (per-block
    inductor, `dynamic=False`); the Flux single block's `proj_out` as diffusers' one gemm at
    tp=1 (`fuse_proj_out`, load-time; `DIFFLET_CUDA_FLUX_FUSE_PROJ_OUT=0` keeps the split).
    Measured on pod f4f: compiled Wan 32.8 s / 40 steps vs diffusers 43.2, compiled Flux
    7.1 s / 28 steps vs 8.3 (DiT 241.3 vs 291.4 ms), eager fused Flux 299.6 vs 291.4 ms; all
    parity gates pass *[10-19 F4f]*.
39. `34a343b` — the Wan and Flux parity oracles and DiT benches honour
    `DIFFLET_CUDA_TORCH_COMPILE` (they built the applications directly, so a "compiled" parity
    run would have judged eager blocks) *[10-19 F4f]*.
40. `49a397c`, `9f1dc24` — Flux e2e `--noise-floor` (diffusers vs itself with
    `allow_bf16_reduced_precision_reduction` flipped) and the Flux e2e gate stated relative to a
    recorded floor (F9b's form): the floor reads 30.92 / 36.62 / 37.61 dB over seeds 0–2, so the
    single-seed 35 dB floor fails diffusers' own noise on seed 0, where the fused and compiled
    runs read 31.5 dB (39.6–43.8 dB on seeds 1–2) *[10-19 F4f]*.
41. `be118a4`, `a59f39b`, `7e29f17` — the Flux e2e over seeds (F9c, on the CPU): `--seeds`
    with one load per side and per-seed / aggregate metrics, the gate over seeds (gates A and
    B as F9b's), `run_gates.sh flux` on seeds 0–2 with the noise floor; the F4f floor runs
    merged into one result pass it *[10-19 F9c]*.
42. `f643e46`, `2497d32` — the pod tool's volume path: v2's create dropped `persistentVolumes`
    silently (a pod without `/data`) and answers 12016 for a network volume with its own
    `volumes` field; a volume pod now goes to v1's create (`persistentVolumes`, Ceph, by id),
    which attached `difflet-cache` *[10-19 F9c verdict]*.
43. F9c measured on `difflet-cuda-f9c3` (Workstation): seeds 0–7 with the floor, eager and
    compiled, pass gates A and B (36.79 / 38.31 dB against 32.40); `run_gates.sh flux` green
    end to end; eager seeds 0–2 reproduce F4f's to the digit *[10-19 F9c verdict]*.

## Support matrix

Model × feature, from the entries above. Wan 2.1 (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) is routed
to the CUDA path but never ran: it is NOT MEASURED in every row and has no column.

| Feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B |
|---|---|---|---|
| smoke (`difflet run`, by eye) | PASS [10-16, 10-17 F3] | PASS [10-18 F5] | PASS [10-18 F1] |
| DiT parity (D14, three gates) | PASS incl. padded prompt [F3, F4] | PASS [F5] | PASS both experts + rounding block [F1, F1b] |
| e2e vs diffusers | PASS, 35 dB / 0.998 [F3] | PASS, 35 dB / 0.998 [F5]; relative to the noise floor when recorded [F4f], over seeds [F9c] | PASS, 25 dB / 0.99 [F1, F1b]; seeds 0–2 overlap diffusers' own noise floor [F9]; gate relative to the floor, 2.0 dB headroom on the mean [F9b] |
| CLI matrix cell | PASS [F3, F5, F9] | PASS [F5, F9] | PASS [F9] |
| `difflet serve` | PASS; LIMIT on concurrency (429 after 30 s) [F3b] | N/A · refused by name, F5b | N/A · refused by name, F1b |
| `--torch-compile` | PASS, opt-in [F4] | PASS, 0.83× diffusers per forward [F4f] | PASS, 0.76× diffusers per run [F4f] |
| `--teacache-cadence N` | PASS, fixed cadence only [F4] | N/A · refused, F5b | N/A · refused, F1b |
| `--dp 2 --requests` | PASS, byte-identical [F2] | `--dp 1` batch PASS, byte-identical [F5c]; `--dp 2` NOT MEASURED (needs 2 GPUs) | PASS, bit-identical [F2b] |
| `--cfg-parallel` (2 GPUs) | N/A · single pass | N/A · guidance-distilled; true-CFG is the owner's call | PASS, bit-identical [F2] |
| `--dp` × `--cfg-parallel` | N/A · refuses `--cfg-parallel` | N/A · refuses `--cfg-parallel` | NOT MEASURED (needs 4 cards) |
| serving on two GPUs (`--dp-degree 2`) | PASS, two requests in one request-time [F2b] | N/A · serve refused | N/A · serve refused |
| CP / ulysses (`--cp-degree > 1`) | N/A · refused, F2b | N/A · refused, F2b | N/A · refused, F2b |
| TP across PCIe, `--sp` | N/A · refused, F6 | N/A · refused, F6 | N/A · refused, F6 |

### Smoke   `difflet run` on one card
Prompt in, image or video out, through the CLI's one-process CUDA orchestrator.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 0.33 s/step trimmed (Server), 0.77 (Workstation) | PASS · 28 steps in 23.2 s (Workstation) | PASS · 1.38 s/step, 73,035 MiB (Workstation) |

### DiT parity   `tests/numerical/cuda_parity_*_gpu.py` + gate
Difflet bf16 against diffusers fp32 and bf16 on identical inputs; gates 1.15× / 1.2× / 0.995.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 0.994× / 0.175× (padded 0.990× / 0.167×) | PASS · 0.987× / 0.398× | PASS · 1.051× / 0.870× and 1.043× / 0.819× |

### e2e vs diffusers   `scripts/cuda_e2e_*vs_diffusers.py` + gate
Same initial noise through both pipelines; pixel PSNR / cosine.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 41.40 dB / 0.99972 | PASS · 40.35 dB / 0.99980; seeds 0–7 against the noise floor, gate A 36.79 dB eager / 38.31 compiled vs 32.40 required (F9c, Workstation) | PASS · 26.30 dB / 0.99609 (1.3 dB above the floor); seeds 0–2 26.30–28.89 dB vs diffusers' own noise floor 23.79–28.62 dB (F9); relative gate mean 27.20 dB vs 25.17 required (F9b) |

### CLI matrix   `scripts/verify_cli_cuda.py`
Download → compile → timed generate per cell, `results.json`.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 38.47 s (Server), 276.01 s (Workstation, cold volume), 51.88 s (Server, F9) | PASS · 56.47 s (Workstation), 16.79 s (Server, F9) | PASS · 111.52 s (Server, F9; all three models on a 250 GB container disk) |

### Serving   `difflet serve`
One resident worker holds the pipeline; OpenAI-compatible chat endpoint.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · ready 56.0 s, 38.6 / 39.2 s per request; LIMIT · 429 `queue_timeout` for a concurrent request at the 30 s default on one GPU; `--dp-degree 2` PASS · two concurrent requests in one request-time (16.7 s wall), identical bytes [F2b] | N/A · refused by name, F5b | N/A · refused by name, F1b |

### `--torch-compile`
Per-block inductor compile of the DiT, opt-in (D4: eager by default).
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 242.8 ms forward, 12.4 s / 50 steps, parity and e2e (40.21 dB) pass | PASS · 241.3 ms forward, 7.1 s / 28 steps vs diffusers 8.3; parity 0.958×; e2e at the noise floor on seed 0, 39.8–42.5 dB on seeds 1–2 [F4f] | PASS · 32.8 s / 40 steps vs diffusers 43.2; parity 0.947× / 0.928×; e2e 27.49 dB [F4f] |

### TeaCache   `--teacache-cadence N`
Probe-free fixed cadence on the CUDA loop; calibrated and online-delta modes refused (F4b).
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · cadence 2: 10.0 s at 38.50 dB vs the full run; cadence 3: 12.3 s at 38.46 dB | N/A · refused (incl. `--teacache-calibration`), F5b | N/A · refused, F1b |

### Data parallel   `--dp N --requests file.jsonl`
One single-GPU replica per card through the request router.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| PASS · 4 requests in 58 s on 2 cards vs 190 s serial, byte-identical | PASS · `--dp 1` batch, 2 requests in 28 s vs 33 s serial, byte-identical [F5c]; `--dp 2` NOT MEASURED (needs 2 GPUs) | PASS · 2 requests in 76 s vs 120 s serial, bit-identical [F2b] |

### CFG-parallel   `--cfg-parallel` (2 GPUs, torchrun)
Runs the cond and uncond branches on two cards, one all-gather per step.
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · single pass | N/A · guidance-distilled | PASS · 1.95×, bit-identical (F2) |

### `--dp` × `--cfg-parallel`
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · refuses `--cfg-parallel` | N/A · refuses `--cfg-parallel` | NOT MEASURED · needs 4 cards; allowed by construction (F2 R3) |

### Context parallel / ulysses, tensor parallel, sequence parallel
| Qwen-Image | FLUX.1-dev | Wan 2.2 |
|---|---|---|
| N/A · CP refused (F2b); TP > 1 and `--sp` refused (F6) | same | same |

## NOT MEASURED

The honest gap list for the next session, none of it refused and none of it run:

1. **Wan `--dp` × `--cfg-parallel`.** It needs four cards; it is allowed by construction and
   unverified (F2 R3, F2b).
2. **FLUX.1-dev with two replicas (`--dp 2`)** on CUDA: reachable, needs a 2-GPU pod. The
   one-card `--dp 1 --requests` batch was measured on pod flux (byte-identical to serial, Two
   GPUs section) *[10-18 F5c]*. Wan under `--dp` was measured on pod f2b *[10-18 F2b/F4b]*.
3. **Wan 2.1** (`Wan-AI/Wan2.1-T2V-14B-Diffusers`). It is routed to the CUDA orchestrator and has
   never run on a GPU. Diffusers' Wan 2.1 defaults (50 steps, guidance 5.0) differ from 40 / 4.0
   *[10-18 F1 review fixes]*.
4. ~~Flux e2e timings~~ — measured on pod flux *[10-18 F5c]* (e2e table): Difflet 9.4 s vs
   diffusers 8.3 s for 28 steps on the Server Edition; the 1.1 s gap is a new F4b-class item
   (no Flux DiT bench exists yet).

F9 measured and removed two items: the CLI matrix's Wan cell (PASS, CLI matrix section) and the
Wan e2e noise floor (e2e section) *[10-18 F9]*. F2b/F4b measured and removed two more: serving on
two GPUs (Serving section) and Wan under `--dp` (Two GPUs section) *[10-18 F2b/F4b]*.

Also unmeasured, and recorded where they arose: Workstation vs Server characterisation with
`cuda_bench_dit.py` (F4b: a second Server card reproduced the f3c table to the millisecond, the
Workstation card was not on the pod); Wan's Difflet vs diffusers generate gap — **measured and
closed** in F4b (50.9 vs 42.9 s Server → 44.5 vs 42.8 s after the fused q/k norm; the
Performance section); why two Wan replicas generating at once take 51.2 s each against 44.5 s
alone (F2b); Flux's generate gap vs diffusers — located by the F4c bench, halved by F4e and
closed to 1.028× per forward by F4f's fused projection (8.8 vs 8.3 s eager; 7.1 s compiled);
F2 `--dp 1` requests 1 and 2 against their
references; the origin of GPU 0's extra ~555 MiB under `--cfg-parallel`; the Trainium
byte-identity of the Flux split (D13, needs a trn2); the split of the Wan noise floor into the
timestep effect and run-to-run GPU nondeterminism (F9). F9's recommended follow-up — state the
Wan e2e gate relative to the measured noise floor, whose worst seed (23.79 dB) is below the
25 dB gate — is done (F9b, ledger line 28, the e2e section).

## Pod cost

Hours and prices as the status log and ledgers record them; "~" marks a figure computed here
from recorded times and prices.

| Pod | Hours | $/h | Cost | Final status (read back) |
|---|---|---|---|---|
| `difflet-cuda-qwen` | not recorded (Release 1, 2026-10-16) | 1.35 | not recorded | not recorded |
| `difflet-cuda-f3`, `difflet-cuda-f3b` | within one 00:06–00:50 UTC window (2026-10-17) | not recorded | not recorded | terminated |
| `difflet-cuda-f3c` | from 09:52 UTC 2026-10-17; end not recorded | 1.35 | not recorded | terminated (owner's request) |
| `difflet-cuda-f5` | not recorded (setup only) | not recorded | not recorded | terminated |
| `difflet-cuda-f5b` | creation time not recorded → paused 12:21 UTC 2026-10-18 | 0.45 | not recorded | PAUSED |
| `difflet-cuda-f1b` | 13:58 → 14:24 UTC (~26 min) | 5.30 | ~$2.30 | terminated, gone |
| `difflet-cuda-f2` | 15:52 → 16:21 UTC (~29 min) | 5.30 | ~$2.60 (recorded) | terminated, gone |
| `difflet-cuda-f9` | 18:44 → 19:50 UTC (~66 min) | 2.65 | ~$2.90 (computed) | terminated, gone |
| `difflet-cuda-f2b` (2 GPUs, eu-2) | 21:27 → 22:11 UTC (the status-log entry's 17:27 → 18:11 are EDT; ~44 min) | 5.30 | ~$3.90 (computed) | terminated, gone |
| `difflet-cuda-flux` (1 GPU, eu-3) | 23:03 → 23:25 UTC (~22 min) | 2.65 | ~$1.00 (computed) | terminated, gone |
| `difflet-cuda-f5b` resume attempt | 23:59 UTC 2026-10-18 → INITIALIZING until terminated 16:01 UTC 2026-10-19 on the owner's word; `pause` refused (12005) | 0.45 if billed | unknown (the API shows no billing; ≤ ~$7) | terminated, gone |
| `difflet-cuda-f4d` (1 GPU, eu-3) | 09:46 → 10:24 UTC 2026-10-19 (~38 min) | 2.65 | ~$1.70 (computed) | terminated, gone |
| `difflet-cuda-f4e` (1 GPU, eu-2) | 11:52 → 12:20 UTC 2026-10-19 (~28 min) | 2.65 | ~$1.25 (computed) | terminated, gone |
| `difflet-cuda-f4f` (1 GPU, eu-2) | 12:56 → 13:52 UTC 2026-10-19 (~56 min) | 2.65 | ~$2.50 (computed) | terminated, gone |
| `difflet-cuda-f9c` (1 GPU, eu-3) | 16:06 → 16:20 UTC 2026-10-19 (~14 min, nothing ran) | 2.68 | ~$0.62 (computed) | terminated, gone |
| `difflet-cuda-f9c2` (1 GPU, us-east-3) | 16:20 → 16:23 UTC (~3 min, nothing ran) | 0.45 | ~$0.02 (computed) | terminated, gone |
| `difflet-cuda-f9c3` (1 GPU, us-east-3, the volume) | 16:32 → 16:53 UTC (~21 min) | 0.45 | ~$0.16 (computed) | PAUSED |
| `difflet-cache` volume | ongoing | $0.0698/GB/month × 200 GB | ~$13.96/month | attached to f9c3 (paused) |

A paused pod releases its card and stops GPU billing; the container disk is lost, while the volume
and the pod record stay (`.claude/skills/difflet-cuda-verify/references/pod-runbook.md`).

## Corrections and observations found while writing this report (2026-10-18, F8)

The earlier text and the status-log entries stay as written. These are new observations:

- Three entries compare Wan's e2e PSNR with Qwen's 38 dB: F1 ("well below Qwen's 38 dB"), F1
  review fixes ("27.5 dB vs Qwen's 38 and Flux's 40") and F1b ("vs Qwen (38 dB)"). The Qwen e2e
  JSONs read 41.40 dB (F3), 41.20 dB (trim) and 40.21 dB (compiled). The 38 is likely the
  TeaCache cadence-2 PSNR (38.50 dB, measured against the full Difflet run, not against
  diffusers). The conclusion (Wan sits far below) is unchanged.
- The `cosine` values in `f2/compare_{wan,qwen}.json` are not meaningful (see Two GPUs). The
  identity claims rest on `max_abs` 0 and `identical_bytes`.
- The F1 Wan JSONs fail today's gates on the F1b-only tests, with the gates' own "predates F1b
  task 2" messages. This is expected: the F1b JSONs carry the same F1 numbers and pass.
- The F2 request prompts are in no artifact; the F8 status-log entry records them from the
  session runbook. The request-0 PNG (a red fox in snow, golden light) is prompt-faithful by eye.

## Observations from F9 (2026-10-18)

- `f9/gates-wan/cuda_parity_wan.json` is byte-identical to `wan/f1b/cuda_parity_wan.json`
  (`cmp`), and the F9 seed-0 e2e numbers and first frames match F1b's exactly: the same code on
  two Server Edition pods gave the same bits. (That the file was produced on f9 rests on
  `gates.txt` — `parity exit: 0 in 189s` at checkout 2387653 — and the F9 entry.)
- `run_gates.sh wan`, the skill's first real use, exited 0 in 595 s (smoke 104 s, parity 189 s,
  e2e 302 s; `GATES DONE wan overall_rc=0`, `f9/gates-wan/gates.txt`), with no GPU process left
  afterwards. Its e2e step runs one seed without `--noise-floor`, so part of it repeats the
  noise-floor run; kept as is.
