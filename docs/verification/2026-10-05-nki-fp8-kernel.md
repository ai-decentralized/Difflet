# NKI FP8 linear kernel on Wan 2.1 — metrics, findings, suggestions (2026-10-05)

Branch `quantization`. Scope: tp4 + quantization only. Evidence:
`artifacts/verification-2026-10-05/nki-fp8/` (v1 = first kernel, v2 = DMA-transpose kernel,
`cpu_e2e/` = CPU fidelity reference). Tech report (for colleagues):
`artifacts/verification-2026-10-04/fp8-trainium-report.html`, published at
https://claude.ai/artifact/Y2SBUKXUjYAUqarjW3p3ux.

## TL;DR

- **Do not ship the NKI kernel.** In the real Wan 2.1 model it is slower than both bf16 and the
  shipped XLA fp8 static path: 686.8 ms (static) and 747.7 ms (per-token) per DiT step vs
  586.9 ms (XLA fp8 static) and 573 ms (bf16).
- **Keep the shipped path (XLA fp8 static scales).** It is still the fastest fp8 arm and the
  most faithful one measured on the device.
- **New fidelity finding:** the NKI kernel *with the same static scales* lands at the same
  13 dB latent SNR / 24.5–24.8 dB PSNR as the XLA dynamic law and the old weight-only run.
  The "dynamic law drifts" conclusion of 2026-10-03 therefore needs revisiting: see
  [Fidelity](#fidelity-the-13-db-vs-25-db-question) and the CPU reference below.

## What was built

`difflet/backends/trainium/nki_kernels/fp8_linear.py`, one NKI kernel per FP8 linear, routed
from `dynamic_fp8_linear` when `DIFFLET_FP8_NKI=static|token` is set at trace time
(experiment switch, off by default):

1. fp8 weight block (≤ 2048 output columns, all of K) resident in SBUF;
2. per 128-token tile: load x already K-major via `dma_transpose` (v2; v1 loaded it
   token-major and transposed 128×128 blocks on the tensor engine);
3. quantize in SBUF: static = multiply by the calibrated `1/input_scale`, clamp ±240, cast;
   per-token = row absmax on the vector engine, per-row reciprocal broadcast, cast;
4. fp8 double-row matmul, fp32 accumulation in PSUM (≤ 512 columns per bank);
5. the PSUM → SBUF eviction applies `input_scale × weight_scale` (one scalar, or the per-token
   column) in the same instruction, then a DMA writes bf16 rows.

This is the GPU "fused epilogue" structure. The prototype still transposes the fp8 weight with
XLA on every call (`layer.weight.t().contiguous()`), and the bias is added by the caller.

## Metrics

### Correctness

| check | result |
|---|---|
| CPU simulator (`scripts/ptq_fp8_kernel_sim.py`), both modes, incl. tail tiles | 55.6 dB vs CPU reference, cosine ≥ 0.99999 |
| device, single layer, Wan tp4 shapes (`scripts/ptq_fp8_kernel_bench.py`) | 64 dB vs CPU reference (same as XLA fp8) |
| device, real model: 2 real Wan blocks, real text, tp4 (device vs CPU fp8) | cosine 0.99998, all checks pass (v1 and v2, both modes) |

### Speed

Single layer, isolated (`kernel_bench.json`): the XLA fp8 static linear is already as fast as or
faster than bf16 at every Wan tp4 shape (q/k/v 0.70 vs 0.70 ms, FFN-in 1.37 vs 1.46,
attn-out 0.74 vs 0.81, FFN-out 1.25 vs 1.60). NKI kernel calls in a standalone harness carry a
~50 ms fixed overhead (nkilib's `qkv` STATIC kernel too), so standalone kernel timings are not
meaningful; only in-graph numbers count.

In-graph, 2 real Wan blocks, tp4, forward ms (mean of 3):

| arm | forward | vs bf16 |
|---|---:|---:|
| bf16 | 35.7 | — |
| XLA fp8 static (shipped) | 35.6 | −0.3 % |
| NKI v1 (tensor-engine transpose), static / token | 45.8 / 45.7 | +28 % |
| NKI v2 (DMA transpose), static | 41.7 | +17 % |
| NKI v2 (DMA transpose), per-token | 44.4 | +24 % |

Full Wan 2.1 T2V-14B, tp4, 480×832×9, 20 steps, seed 42, host VAE (`v2/ab_*`, `v1/ab_static`):

| arm | compile s | DiT step ms (n=19) | latent SNR vs bf16 | render vs bf16 PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| bf16 | — | 573 | — | — |
| fp8 weight-only (2026-10-01, retired mode) | 355 | 563.0 | — | 24.6 / 0.878 / 0.122 |
| XLA fp8 dynamic law (2026-10-03) | — | 607.2 | 13.2 | 24.5 / 0.88 / 0.12 |
| **XLA fp8 static (shipped)** | — | **586.9** | **24.9** | **33.7 / 0.953 / 0.035** |
| NKI v2 static | 218 | 686.8 | 13.3 | 24.8 / 0.887 / 0.115 |
| NKI v2 per-token | 227 | 747.7 | 12.8 | 24.5 / 0.880 / 0.121 |
| NKI v1 static (generate only, cached artifact) | — | 763.8 | 13.3 | 24.8 / 0.887 / 0.115 |

## Findings

### Why the kernel is slower than the XLA path (speed)

1. **XLA's fp8 linear was never the bottleneck.** In isolation it already matches or beats
   bf16; the 14 ms/step the shipped fp8 path loses to bf16 is graph-level (the scalar-engine
   quantize chain feeding every dot; profile in `2026-10-03/fp8-dyn-step/NOTES.md`). A
   per-linear kernel can only win if it is faster than an already-good matmul, and a first
   hand-written kernel is not.
2. **The custom-call boundary costs fusion.** An NKI kernel is opaque to neuronx-cc: the
   compiler can no longer overlap the linear with the neighbouring norm / modulation / RoPE /
   attention ops, schedule across it, or keep its output in SBUF for the next op. Every kernel
   call reads x from HBM and writes y to HBM. With ~6 linears per block × 40 blocks per step,
   that boundary cost is paid ~240 times per step.
3. **Prototype overheads still in the graph:** the per-call XLA transpose of the fp8 weight
   (`w.t().contiguous()` on [N, K] → [K, N], every linear, every step) and the bias add outside
   the kernel. v1 → v2 (DMA transpose instead of tensor-engine transposes) cut the 2-block
   probe from 45.8 to 41.7 ms; the rest is items 2–3 plus untuned tiling.
4. **Per-token is costlier than static** (+61 ms/step at full depth): per-row absmax on the
   vector engine, a reciprocal, a broadcast transpose of the per-row scales and a tensor-tensor
   multiply per K tile, versus one scalar multiply for static.

### Fidelity: the 13 dB vs 25 dB question

Five fp8 arms share the same fp8 weights (per-tensor absmax / 240). Four of them land at the
same end-to-end fidelity (~13 dB latent SNR, ~24.5–24.8 dB PSNR): XLA dynamic, NKI static,
NKI per-token, and the old weight-only run (which quantizes no activations at all). Only XLA
static is at 24.9 dB / 33.7 dB.

- NKI static uses **the same calibrated static scales** as XLA static, so the 2026-10-03
  explanation ("the amax-derived dynamic scale accumulates error over the loop") cannot be
  the whole story: a static-scale arm drifts just the same.
- A W8A8 arm cannot be more faithful than weight-only with the same weights, yet XLA static is
  9 dB PSNR better than weight-only. Either the weight-only run (2026-10-01, older target set
  and checkpoint) is not comparable, or the XLA static graph is effectively not quantizing part
  of what the other arms quantize.
- Every single-forward check (CPU vs device, 1 step at full depth, 2 real blocks) passes for
  every arm, so whatever separates them only shows over 20 steps.

CPU reference (`scripts/ptq_fp8_cpu_e2e.py`, real 20-step loop, CPU fp8 simulation with the
same checkpoint law; `cpu_e2e/summary.txt`):

_Running at the time of this commit (≈45 min per arm on the 12 host vCPUs); results land in `cpu_e2e/summary.txt` and here._



## Suggestions

1. **Ship XLA fp8 static (as today); keep `DIFFLET_FP8_NKI` as an off-by-default experiment.**
2. **Resolve the fidelity question before claiming 33.7 dB externally.** The CPU reference above is the arbiter: whichever device arm matches it is the correct one.
3. If the kernel work continues, in order of expected gain:
   - store the fp8 weight pre-transposed `[K, N]` in the quantized checkpoint (removes the
     per-call XLA transpose);
   - fuse more than one linear per call (q/k/v as one kernel with three outputs; FFN-in + GELU
     + FFN-out as one kernel) so the boundary cost is paid once per sub-block instead of per
     linear;
   - fuse the bias (column-parallel layers) and the following residual / gate where possible;
   - tune tiling (`_N_BLOCK`, 512-column PSUM banks, double-buffered x loads).
   A fused FFN kernel is the only variant with a realistic chance of beating the XLA graph,
   because it removes an HBM round trip the compiler cannot remove.
4. Per-token scales are not worth their cost at Wan's activation statistics: no fidelity gain
   over static on the device, +61 ms/step.
