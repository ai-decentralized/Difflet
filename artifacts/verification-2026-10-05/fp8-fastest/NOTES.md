# Fastest fp8 scheme on Wan 2.1 (tp4) — working notes, 2026-10-05

Goal (user, 2026-10-05 06:45 UTC): explore on my own, find the fastest fp8 scheme, report for review.
Baseline: bf16 573.0 ms / XLA fp8 static 586.9 ms per DiT step (480x832x9, 20 steps, tp4).

## Diagnosis (instruction-level traces of 2026-10-03, rank 0, one forward; `trace_analysis/`)

Instruction-level analysis of `/home/ubuntu/ptq-profiles/wan21_{bf16,fp8static}_2_26_rank0.ntff`
(neuron-explorer view → parquet, duckdb; scripts in the job tmp `trace_analysis/a2..a7.py`):

- **79 % of fp8 matmul FLOPs run single-row (bf16 speed).** fp8 dots tiled `[390,1,1]` (4680 tokens =
  12 x 390) take the same 328-379 ns per instruction as bf16. Only one pair of projection layers
  (likely attention out, K=1280/rank) is tiled with 512-wide double-row tiles `[512,1,2]` and gets the
  real 2x (33.5 MFLOP / 379 ns vs 16.8 for bf16), saving ~15 ms of tensor-engine time.
- **Quantize / dequantize chain = ~89 ms of new engine work**: scalar +63 ms (new fp32->fp32 ACTIVATE
  dequant 48.5 ms, bf16->fp32 ×1/scale +12.7), vector new fp32->fp8 clamp+cast 29.6 ms.
- **Tensor engine starved by it**: tensor idle +32 ms (205 vs 173), idle waiting on vector +25 ms;
  vector-waiting-on-scalar 224 vs 50 ms. The span difference is +22 ms (555 vs 533 ms).
- Transposes on the tensor engine: ~107 ms in both models (+11 ms fp8).
- Weight DMA: GpSimd DMA-trigger time 39.7 vs 87.6 ms (half the weight bytes) — the main fp8 win today.

## Compiler research (neuronx-cc 2.26 / 2.27, read-only)

- Double-row fp8 is ON by default (`VectorizeMatMult` / `MMDoubleRowVectorizer`); the only knob is
  `--tensorizer-options=--internal-disable-double-row-gen3`. No enable flag / env var exists.
- Hardware rule (libwalrus): DoubleRow needs the contraction dim packed K/2 x 2 and size % 16 == 0.
- NxD Inference gets fp8 speed from nkilib kernels (`mlp[lnc]`, `rmsnorm_quant`) with explicit
  `perf_mode=double_row`, launched with an LNC grid (`kernel[logical_nc_config]`).

## Screen (2 real Wan blocks, tp4, static scales, 10 forwards, CPU job running alongside)

| config | bf16 ms | fp8 ms | note |
|---|---:|---:|---|
| -O1 (shipped) | 37.71 | 35.49 | bf16 read 35.7 earlier on an idle host |
| -O1, double-row disabled | — | 38.80 | double-row does fire at -O1 (worth 3.3 ms here) |
| -O2 | 45.48 | 43.13 | slower for both |
| -O3 | 45.46 | 43.73 | slower for both |

Pending in the queue: --vectorize-strided-dma, fused bias, bf16-domain quantize, VC2 (attention_cte[2]),
NKI linear kernel with an LNC [2] grid, nkilib fused MLP (FFN) with/without grid, row padding 512/256.

## Fidelity

CPU bf16 vs device bf16 final latents (same prompt, seed, shape): **14.05 dB SNR, cosine 0.980** —
the 20-step trajectory amplifies any numerical difference to ~13-14 dB, so the "13 dB" fp8 arms
(NKI, dynamic, weight-only) are at the same divergence as a different bf16 implementation.
CPU W8A8 arms running (`../nki-fp8/cpu_e2e/`).
