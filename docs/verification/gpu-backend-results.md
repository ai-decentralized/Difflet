# GPU backend (CUDA): verification results

What was run on GPUs for Qwen-Image, FLUX.1-dev and Wan 2.2 T2V-A14B on the CUDA backend, what
works with a documented limit, what is refused by design, and what was never run. Hardware:
NVIDIA RTX PRO 6000 Blackwell (Server and Workstation editions, 96 GB, PCIe, no NVLink), torch
2.11.0+cu128; Flux with two replicas on 2 × A100 80 GB.

Statuses: **PASS** (ran end to end) / **LIMIT** (works with a documented constraint) / **N/A**
(refused by design) / **NOT MEASURED** (not refused, not run).

## Support matrix

| Feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B |
|---|---|---|---|
| `difflet run` | PASS | PASS | PASS |
| DiT parity vs diffusers | PASS (all-ones and padded prompt) | PASS | PASS (both experts, rounding block) |
| e2e vs diffusers | PASS | PASS, gate relative to a recorded noise floor over seeds | PASS, gate relative to a recorded noise floor over seeds |
| CLI matrix (download, compile, generate) | PASS | PASS | PASS |
| `difflet serve` | PASS; LIMIT: one request at a time (429 after the 30 s queue timeout) | PASS, PNG byte-identical to `difflet run` | PASS, frames bit-identical to `difflet run` in-process; the MP4 writer is not byte-deterministic |
| `--torch-compile` | PASS | PASS | PASS |
| `--teacache-cadence N` | PASS | N/A | N/A |
| `--dp 2 --requests` | PASS, byte-identical | PASS on 2 × A100, byte-identical; NOT MEASURED on two PRO 6000s | PASS, bit-identical |
| `serve --dp-degree 2` | PASS | PASS on 2 × A100 | N/A: the video service runs one job at a time |
| `--cfg-parallel` (2 GPUs) | N/A: one pass per step | N/A: guidance-distilled | PASS, bit-identical |
| `--dp` × `--cfg-parallel` | N/A | N/A | NOT MEASURED (needs 4 GPUs) |
| CP, TP, `--sp` | N/A: refused | N/A: refused | N/A: refused |

Wan 2.1 (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) is routed by the CLI but has never run on a GPU and
is not served.

## Numbers

**Quality against upstream diffusers** (same initial noise through both pipelines; pixel PSNR /
cosine):

- Qwen-Image: 41.40 dB / 0.99972 (floors 35 dB / 0.998); 40.21 dB compiled.
- FLUX.1-dev: 40.35 dB / 0.99980 on seed 0 with the split `proj_out` (the default). Over seeds
  0–7 the gate is relative to diffusers' own noise floor (diffusers against itself with its bf16
  gemm reductions perturbed spans 30.9–37.6 dB over three seeds).
- Wan 2.2: 26.30 dB / 0.99609 with the fp32 timestep; over seeds 0–2, 26.30–28.89 dB against
  diffusers' own noise floor of 23.79–28.62 dB, so the gate is relative to that floor.
- DiT parity (rel_l1 against the fp32 reference, relative to diffusers' bf16 control; gates
  1.15× / 1.2× / 0.995): Qwen 0.994×, Flux 0.987×, Wan 1.051× / 1.043× per expert.

**Speed** (RTX PRO 6000 Server Edition unless noted):

| | Qwen-Image (50 steps, 1024²) | FLUX.1-dev (28 steps, 1024²) | Wan 2.2 (40 steps, 480×832×9) |
|---|---|---|---|
| DiT forward, eager / compiled / diffusers | 323 / 243 / 300 ms | 303 / 241 / 292 ms | 536 / — / 528 ms |
| Whole run, eager / compiled / diffusers | 16.5 / 12.4 / 15.2 s | 9.4 / 7.1 / 8.3 s | 44.5 / 32.8 / 42.8 s |
| Peak memory | 58 GB | — | 71.3 GiB (Workstation) |

The Qwen-Image prompt trim took its DiT forward from 516 ms to 323 ms; the fused across-heads
q/k RMSNorm took Wan's from 614 ms to 536 ms. Two GPUs: Wan `--cfg-parallel` 26.1 s against
50.9 s on one GPU (1.95×); Qwen `--dp 2` ran four requests in 58 s against 190 s serially.

## Original HunyuanVideo

Experimental BF16 support passed 12 full-length accuracy cases and 24 timed-video checks
at 320×512×61, 50 steps, guidance 6, seeds 0–2 on an RTX PRO 6000 Blackwell 96 GB.
Public Python, CLI and resident HTTP checks passed in eager and compiled modes.
Six matched eager pairs measured 65.626 s versus 67.300 s for the unmodified Omni
Diffusers adapter on a controlled common runtime. The modified compiled control is
supplementary; an official compiled comparison remains unavailable. See the
[HunyuanVideo CUDA guide](hunyuanvideo-cuda.md) for methodology and limits.

## Remaining qualification

1. Wan `--dp` × `--cfg-parallel` (needs four GPUs; allowed by construction).
2. FLUX.1-dev with two replicas on two RTX PRO 6000s (measured on 2 × A100 instead).
3. Wan 2.1.
4. The fused Flux `proj_out` (`fuse_proj_out`, off by default): about 1% faster per DiT forward,
   but seed 0's e2e PSNR moved from 41.58 dB to 31.56 dB in eager mode (42.27 dB compiled). A
   bf16 block-level comparison against diffusers' `proj_out` is the next check.
5. The Trainium byte-identity of the Flux modeling split on a trn2 host was checked separately
   (HLO and a FLUX.1-dev image identical before and after); TPU has no compile/load lifecycle to
   run.
