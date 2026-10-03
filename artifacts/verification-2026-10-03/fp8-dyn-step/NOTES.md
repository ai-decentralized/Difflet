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

## neuronx-cc 2.27 does not change the picture (spike, `../cc227/`)

Same Wan 2.1 A/B recompiled with neuronx-cc 2.27.5334 (venv clone with `nki 0.6.0` and
`islpy==2026.1`; islpy 2026.2 makes 2.27 fail with `NCC_ISMP902 is_subset()`), host VAE,
one generate per arm:

| compiler | bf16 DiT step (ms, median) | fp8-dynamic DiT step | ratio | fp8 vs bf16 quality |
|---|---:|---:|---:|---|
| 2.26.6360 | 573.0 | 656.6 | 1.146× | PSNR 24.6–24.9 dB, SSIM 0.88 |
| 2.27.5334 | 568.7 | 648.2 | 1.140× | PSNR 24.9 dB, SSIM 0.885, LPIPS 0.117 |

Both arms gain about 1 % from the newer compiler; the fp8 graph is lowered the same way
(400 `F8E4M3FN × F8E4M3FN → F8E4M3FN` dots, 2083 `BF16 → F32` converts vs 1683 in bf16,
`hlo_ops_fp8_dyn_2_27.txt` identical to the 2.26 histogram). Transformer compile 309 s (bf16)
/ 319 s (fp8). The compiler version is not the lever; the quantize / dequantize traffic is.

## Device profile confirms it (neuron-explorer, one forward of the 2.26 NEFFs, tp4, rank 0)

`profile/bf16/summary_full.txt`, `profile/fp8_dyn/summary_full.txt` (captures:
`neuron-explorer capture -n model.neff -r 4 --collectives-worker-count 4 --num-exec 3
--profile-nth-exec 3`; the 3.6 GB `.ntff` traces stay on the host under the same dirs).

| metric (per forward) | bf16 | fp8 dynamic | delta |
|---|---:|---:|---:|
| total active time | 517 ms | 576 ms | **+59 ms** |
| tensor engine active | 358 ms | 322 ms | **−36 ms** (the fp8 dots are faster) |
| vector engine active | 202 ms | 268 ms | **+66 ms** |
| scalar engine active | 191 ms | 234 ms | **+43 ms** |
| gpsimd engine active | 93 ms | 36 ms | −57 ms |
| HBM read | 69.7 GB | 40.5 GB | −29 GB (fp8 weights) |
| HBM write | 15.7 GB | 31.7 GB | **+16 GB** (fp32 intermediates) |
| spill save / reload | 11.1 / 13.4 GB | 28.0 / 27.3 GB | **×2.5 / ×2.0** |
| vector / scalar / activate instructions | 758 k / 529 k / 233 k | 994 k / 674 k / 317 k | +31 % / +27 % / +36 % |
| tensor-engine instructions | 6.39 M | 6.17 M | −3 % |

So the matmuls already win from fp8 (−36 ms); the step loses because the un-fused fp32
quantize / dequantize passes land on the vector and scalar engines and spill through HBM.
If that overhead went to zero the fp8 step would be ~7 % *faster* than bf16 at this shape
(and more once the dot is the only cost, i.e. with an NKI kernel that quantizes in SBUF).

## Lean law, round 1 (`lean/`): 656.6 → 607.7 ms

Commit `4f3f585`: absmax from min/max reductions (no abs tensor), scale with a 2^-7 margin,
multiply by the bf16 reciprocal, no clamp, one combined dequant multiply. Measured on the
device with neuronx-cc 2.26 (same prompt / seed / shape, host VAE, `ab/`):

| arm | DiT step ms (median) | vs bf16 (573.0) | quality vs bf16 host-VAE render |
|---|---:|---:|---|
| fp8 dynamic, original law | 656.6 | 1.146× | PSNR 24.6–24.9 dB, SSIM 0.88, LPIPS 0.12 |
| fp8 dynamic, lean round 1 | **607.7** | **1.061×** | PSNR 24.53 dB, SSIM 0.879, LPIPS 0.124 (`compare_lean_fp8_vs_bf16_hostvae.json`); vs the old fp8 render 31.4 dB |

Tiny probe unchanged (device fp8 vs CPU fp8 cosine 0.999982). Transformer compile 479 s.
HLO (`hlo_ops_MODULE_97fda…`): abs / divide / clamp gone (−18 G writes), broadcasts 41 → 24 G,
but the bf16 multiply came back as `convert→F32, multiply F32, convert→BF16, convert→F8E4M3FN`
(`hlo_chain.py`): XLA keeps bf16 arithmetic in fp32 with a round trip, so the next variant
multiplies in fp32 and casts straight to fp8 (one convert fewer per linear).

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
