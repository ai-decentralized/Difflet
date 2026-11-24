# GPU backend (CUDA): verification results

What was run on GPUs for Qwen-Image, FLUX.1-dev, Wan 2.2 T2V-A14B and Wan 2.1 T2V-14B on the
CUDA backend, what works with a documented limit, what is refused by design, and what was never
run. Hardware: NVIDIA RTX PRO 6000 Blackwell (Server and Workstation editions, 96 GB, PCIe, no
NVLink), torch 2.11.0+cu128; Flux with two replicas on 2 × A100 80 GB. Wan 2.1's six-seed accuracy
against diffusers and its speed are from one A100 80 GB (SXM4); its DiT parity, its Python, CLI,
staged-CLI, `--dp 2`, `--cfg-parallel` and HTTP cells, its fused-against-unfused comparison and
Wan 2.2's checks with the fused eager blocks are from H200s. The `--dp 2` table for
HunyuanVideo (original), HunyuanVideo 1.5 and LTX-2 is from two H200 141 GB, and for MiniMax-H3
from two B200 180 GB.

Statuses: **PASS** (ran end to end) / **LIMIT** (works with a documented constraint) / **N/A**
(refused by design) / **NOT MEASURED** (not refused, not run).

## Support matrix

| Feature | Qwen-Image | FLUX.1-dev | Wan 2.2 T2V-A14B | Wan 2.1 T2V-14B |
|---|---|---|---|---|
| `difflet run` | PASS | PASS | PASS | PASS |
| DiT parity vs diffusers | PASS (all-ones and padded prompt) | PASS | PASS (both experts, rounding block) | PASS (rounding block) |
| e2e vs diffusers | PASS | PASS, gate relative to a recorded noise floor over seeds | PASS, gate relative to a recorded noise floor over seeds | PASS at 9 and 81 frames, gate relative to a recorded noise floor |
| CLI matrix (download, compile, generate) | PASS | PASS | PASS | PASS, tensor-equal to `difflet run` |
| `difflet serve` | PASS; LIMIT: one request at a time (429 after the 30 s queue timeout) | PASS, PNG byte-identical to `difflet run` | PASS, frames bit-identical to `difflet run` in-process; the MP4 writer is not byte-deterministic | PASS, served frames equal the Python path's (MP4 compared through the same encoder) |
| `--torch-compile` | PASS | PASS | PASS | PASS |
| `--teacache-cadence N` | PASS | N/A | N/A | N/A |
| `--dp 2 --requests` | PASS, byte-identical | PASS on 2 × A100, byte-identical; NOT MEASURED on two PRO 6000s | PASS, bit-identical | PASS, bit-identical |
| `serve --dp-degree 2` | PASS | PASS on 2 × A100 | N/A: the video service runs one job at a time | N/A: the video service runs one job at a time |
| `--cfg-parallel` (2 GPUs) | N/A: one pass per step | N/A: guidance-distilled | PASS, bit-identical | PASS, bit-identical |
| `--dp` × `--cfg-parallel` | N/A | N/A | NOT MEASURED (needs 4 GPUs) | NOT MEASURED |
| CP, TP, `--sp` | N/A: refused | N/A: refused | N/A: refused | N/A: refused |

Wan 2.1 (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) runs Wan 2.2's application with its one transformer
and Wan 2.2's spec defaults (40 steps, guidance 4.0); every Wan 2.1 number here uses its model
card's 50 steps at guidance 5.0. On two H200s, Python, `difflet run` and `difflet run --dp 2`
gave tensor-equal frames for one request, and `difflet serve` the same frames in its MP4.

## Numbers

**Quality against upstream diffusers** (same initial noise through both pipelines; pixel PSNR /
cosine):

- Qwen-Image: 41.40 dB / 0.99972 (floors 35 dB / 0.998); 40.21 dB compiled.
- FLUX.1-dev: 40.35 dB / 0.99980 on seed 0 with the split `proj_out` (the default). Over seeds
  0–7 the gate is relative to diffusers' own noise floor (diffusers against itself with its bf16
  gemm reductions perturbed spans 30.9–37.6 dB over three seeds).
- Wan 2.2: 26.30 dB / 0.99609 with the fp32 timestep; over seeds 0–2, 26.30–28.89 dB against
  diffusers' own noise floor of 23.79–28.62 dB, so the gate is relative to that floor. With the
  fused eager blocks on one H200 (same prompt, steps and seeds): 28.80, 28.65 and 22.97 dB
  against a floor of 27.58, 27.05 and 23.37 dB; every gate passes.
- Wan 2.1 on one A100, 480×832×9: over seeds 0–5, 23.16–27.70 dB (mean 25.68) eager and 19.22–27.88 dB
  (mean 24.65) compiled, against diffusers' own noise floor of 14.45–27.15 dB (mean 23.14).
  480×832×81 with the model card's negative prompt, seed 42: 19.95 dB / 0.9756, against a floor of
  16.15 dB / 0.9414. At 81 frames, and on some 9-frame seeds, a timestep rounded through bf16
  moves diffusers' own clip below 20 dB, so the comparison is against that floor.
- Wan 2.1 eager with and without the fused blocks, on one H200, 480×832×9, PSNR against diffusers
  (the unfused run is the modeling's own forward):

  | Seed | 0 | 1 | 2 | 3 | 4 | 5 | Mean |
  |---|---|---|---|---|---|---|---|
  | Fused blocks | 31.62 | 23.81 | 23.92 | 22.14 | 25.65 | 22.51 | 24.94 dB |
  | Modeling's ops | 30.14 | 19.36 | 23.54 | 21.42 | 24.05 | 22.63 | 23.52 dB |
  | diffusers vs itself | 26.36 | 24.34 | 28.74 | 20.65 | 27.27 | 14.16 | 23.58 dB |
- DiT parity (rel_l1 against the fp32 reference, relative to diffusers' bf16 control; gates
  1.15× / 1.2× / 0.995): Qwen 0.994×, Flux 0.987×, Wan 1.051× / 1.043× per expert (1.052× /
  1.044× with the fused eager blocks on an H200), Wan 2.1 1.007× on an A100 and 1.005× on an H200
  (1.003× / 1.002× at the bf16-rounded timestep).

**Speed** (RTX PRO 6000 Server Edition unless noted):

| | Qwen-Image (50 steps, 1024²) | FLUX.1-dev (28 steps, 1024²) | Wan 2.2 (40 steps, 480×832×9) |
|---|---|---|---|
| DiT forward, eager / compiled / diffusers | 323 / 243 / 300 ms | 303 / 241 / 292 ms | 536 / — / 528 ms |
| Whole run, eager / compiled / diffusers | 16.5 / 12.4 / 15.2 s | 9.4 / 7.1 / 8.3 s | 44.5 / 32.8 / 42.8 s |
| Peak memory | 58 GB | — | 71.3 GiB (Workstation) |

Wan 2.1 (50 steps, guidance 5.0, one A100 80 GB, each run alone): at 480×832×9 a warm run takes
64.4 s eager, 63.8 s compiled and 81.3 s in diffusers; at 480×832×81, 788 s eager, 786 s compiled
and 921 s in diffusers. Eager runs each block's LayerNorms, adaLN modulation, gated residuals and
q/k norm with rotary as fused Triton kernels: a DiT forward at 480×832×9 takes 627 ms, against
761 ms with the modeling's unfused ops. Peak allocated memory is 41.7 GiB at 9 frames and 41.9 GiB
at 81, within 0.1 GiB of diffusers. Compiling adds 20–80 s to a load, depending on the shape and
the compile cache. On H200s at 480×832×9, a request takes 27.4 s on one card and 20.4 s with
`--cfg-parallel` on two. Wan 2.2 eager with the fused blocks on one H200 (40 steps, 480×832×9)
takes 19.3 s per warm request against 26.5 s in diffusers, peak 68.5 GiB against 68.6 GiB.

The Qwen-Image prompt trim took its DiT forward from 516 ms to 323 ms; the fused across-heads
q/k RMSNorm took Wan's from 614 ms to 536 ms. Two GPUs: Wan `--cfg-parallel` 26.1 s against
50.9 s on one GPU (1.95×); Qwen `--dp 2` ran four requests in 58 s against 190 s serially.

## Original HunyuanVideo

BF16 support passed 12 full-length accuracy cases and 24 timed-video checks
at 320×512×61, 50 steps, guidance 6, seeds 0–2 on an RTX PRO 6000 Blackwell 96 GB.
Public Python, CLI and resident HTTP checks passed in eager and compiled modes.
Six matched eager pairs measured 65.626 s versus 67.300 s for the unmodified Omni
Diffusers adapter on a controlled common runtime. The modified compiled control is
supplementary; an official compiled comparison remains unavailable. See the
[HunyuanVideo CUDA guide](hunyuanvideo-cuda.md) for methodology and limits.

## Data parallelism for HunyuanVideo 1.0/1.5, LTX-2 and MiniMax-H3

`difflet run --dp 2 --dp-schedule round_robin --requests` ran four requests on two replicas,
against each card running the same four requests alone (BF16, torch 2.11.0+cu128). Every
`--dp 2` output was `torch.equal` to the same request on the same card (MiniMax-H3: video and
audio), and the two cards' one-card outputs were equal. Generation time is the four requests
on one card against the slower of the two replicas.

| Model | GPUs | Shape, steps | One card | `--dp 2` | Speedup |
|---|---|---|---|---|---|
| HunyuanVideo (original) | 2 × H200 141 GB | 320×512×61, 50 | 204.4 s | 103.4 s | 1.98× |
| HunyuanVideo 1.5 | 2 × H200 141 GB | 480×832×33, 50 | 151.8 s | 76.4 s | 1.99× |
| LTX-2 | 2 × H200 141 GB | 512×768×121, 40 | 159.5 s | 81.6 s | 1.95× |
| MiniMax-H3 | 2 × B200 180 GB | 256×448×124, 50 | 45.0 s | 23.8 s | 1.89× |

`serve --dp-degree` is refused for video models, because the video service runs one job at a
time.

## Remaining qualification

1. Wan `--dp` × `--cfg-parallel` (needs four GPUs; allowed by construction).
2. FLUX.1-dev with two replicas on two RTX PRO 6000s (measured on 2 × A100 instead).
3. The fused Flux `proj_out` (`fuse_proj_out`, off by default): about 1% faster per DiT forward,
   but seed 0's e2e PSNR moved from 41.58 dB to 31.56 dB in eager mode (42.27 dB compiled). A
   bf16 block-level comparison against diffusers' `proj_out` is the next check.
4. The Trainium byte-identity of the Flux modeling split on a trn2 host was checked separately
   (HLO and a FLUX.1-dev image identical before and after); TPU has no compile/load lifecycle to
   run.

## LTX-2 CUDA

Single-GPU BF16 audiovisual generation is qualified in eager and compiled
modes against pinned native Omni, using an explicit native-compatible generation profile.
The four-way B200 comparison measures Difflet eager/compiled and native eager/corrected-private
compiled on reduced and larger workloads. Upstream default native compilation failed quality
and is not a qualified performance baseline. Larger warm medians are 20.795/15.026 seconds
for Difflet and 20.445/15.204 seconds for native. See the [LTX-2 guide](ltx2-cuda.md) for matched
settings, cold/compile cost, memory, limitations and frontend versus model coverage.
