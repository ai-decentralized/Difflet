# Why the fp8-dynamic DiT step is slower than bf16 — HLO evidence (neuronx-cc 2.26)

Wan 2.1 14B transformer, tp4, 480×832×9 (4680 latent tokens, hidden 5120, FFN 13824 → 3456 per
rank, 10 heads × 128 per rank). Modules from `/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47`:
`MODULE_f1d9baa9d8bbf1aadd37+a0010e47` (fp8 dynamic, fixed layers) and
`MODULE_f79148cf7ed21271b9e2+f934ae74` (bf16). Histograms over tensors with ≥ 1 M elements:
`hlo_ops_fp8_dyn_2_26.txt`, `hlo_ops_bf16_2_26.txt` (tool `hlo_ops.py`).

## What the fp8 graph does around every one of the 320 quantized dots

Input side, per linear, on the full activation (`[1, 4680, 5120]` = 24 M elements), all in **F32**:

| op | count (fp8) | count (bf16) | note |
|---|---:|---:|---|
| convert → F32 `[1,4680,5120]` | 601 | 281 | bf16 activation up-cast before the quantize math |
| abs F32 | 200 | 0 | `x.abs().amax()` materialises the abs tensor |
| divide F32 | 200 | 0 | `x / scale` (a divide, not a multiply by the reciprocal) |
| clamp F32 | 200 | 0 | clamp to ±240 (redundant: the scale already bounds the range) |
| convert → F8E4M3FN | 200 | 0 | the quantize |
| broadcast F32 `[1,4680,5120]` | 1282 | 442 | scalar scales broadcast to full tensors |

Output side: every dot is typed `F8E4M3FN × F8E4M3FN → F8E4M3FN`, then convert → F32,
multiply by `input_scale × weight_scale` (another broadcast), convert → BF16.

Totals (elements written on large tensors, per forward):

| op class | fp8 dynamic | bf16 | delta |
|---|---:|---:|---:|
| broadcast F32 | 41.1 G | 13.7 G | +27.3 G |
| convert → F32 | 19.7 G | 9.1 G | +10.6 G |
| multiply F32 | 19.1 G | 10.1 G | +9.0 G |
| convert → BF16 | 11.0 G | 6.5 G | +4.5 G |
| abs + divide + clamp + convert→F8 | 24.5 G | 0 | +24.5 G |
| dot | 4.5 G (fp8-typed) | 4.5 G (bf16) | — |

≈ **76 G extra element-writes per forward**, nearly all F32 (4 B): on the order of 300 GB of
HBM traffic if nothing fuses, which at trn2's bandwidth is the same order as the measured
**+84 ms per step** (657 vs 573 ms). The fp8 dots themselves are not the problem; the
un-fused, fp32, memory-bound quantize / dequantize passes around them are.

## Levers (cheapest first)

1. **Do the quantize math in bf16, not fp32**: the input is already bf16; converting to F32
   doubles the bytes of every pass. `amax` in bf16 is exact enough for a per-tensor scale.
2. **Drop the `abs` pass**: `amax = max(x.amax(), -x.amin())` — two reductions (no full-size
   write) instead of abs (full write) + one reduction.
3. **Multiply by the reciprocal scale** instead of dividing (scalar reciprocal once).
4. **Drop the clamp**: with `scale = amax / 240` the scaled tensor is bounded by 240 by
   construction (240 is representable); keep a clamp only if a non-infinite `clamp_bound` is set.
5. **Dequantize in bf16**: convert the dot output straight to bf16 and multiply by the combined
   scalar `input_scale × weight_scale` once (one pass, half the bytes of the F32 path).
6. **Fold the dequantize into the consumer** (bias add / GELU / residual) — compiler-dependent.
7. **NKI W8A8 kernel**: quantize in SBUF, fp8 matmul with fp32 accumulate, scale on the way out —
   no HBM round trips at all. The real fix; sized separately.

Levers 1–5 are a change to `dynamic_fp8_linear` / `quantize_activation_per_tensor` (and the CPU
reference, which must keep matching the device bit-for-bit in value) and can be measured with
the tiny probe plus one Wan 2.1 A/B. Whether neuronx-cc 2.27 lowers the fp8 dots or these
passes better is the spike running alongside (`../cc227/`).
