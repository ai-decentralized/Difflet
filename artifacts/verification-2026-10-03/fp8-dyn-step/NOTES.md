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

## Lean law, round 2 (`lean_round2/`): fp32 multiply form — 607.2 ms, no change

Quantizing as `convert→F32, multiply, convert→F8` (instead of the bf16 multiply that XLA
lowered with an extra bf16 round trip; new module `MODULE_a71171d7…`, compile 321 s,
schema 4 to bust the stage key — the runner's `--force` does not recompile an existing
stage artifact) measures 607.2 ms median — identical to round 1: the compiler already fused
that round trip. Numerics: device fp8 vs CPU fp8 cosine 0.999985 / 45.3 dB (tiny probe).
This form is kept (fewer HLO ops, same speed).

What is left between 607 and 573 ms is the per-linear quantize (two reductions + one fp32
pass + the fp8 cast) and dequantize (fp8→fp32, scale, →bf16) traffic that the compiler does
not fuse into the matmul. Structural levers from here: quantize a shared input once for
q/k/v (the three projections re-quantize the same tensor), and the NKI W8A8 kernel.

## Round 3 (`lean_round3/`): quantize a shared input once for q/k/v — 639.3 ms, reverted

Commit `3cefaf1` quantized the attention input once and fed the fp8 tensor plus its scale to
the q / k / v projections (and `proj_mlp` in the fused single blocks), saving two of the
three quantize passes per attention block. Measured 639.3 ms median (n=19, min 638.9), a
**32 ms regression** over 607.2 ms, so `48f8eb9` reverts it (back to schema 4). Tiny probe
still passed (cosine 0.99997), so it was not a numerics bug.

Why sharing is slower: with one quantize per linear the compiler fuses the
`convert→multiply→convert` chain into each matmul's operand load and never materialises the
fp8 activation. A shared fp8 tensor with three consumers must be written to HBM and read back
three times (new modules `MODULE_e889ec0d…` / `MODULE_f9f4db2d…`, 68 op kinds each). Lesson:
the elementwise quantize / dequantize is already fused away; what is left of the 34 ms gap is
the two absmax reductions per linear that must finish before the scale is known (640 full
reads of the activation per forward). That is what static, calibrated scales remove
(`static/`).

## Static calibrated activation scales (`static/`): 607.2 → 586.9 ms, and a quality surprise

Commit `946cb89`: `--quant-calibration <json>` stores a per-layer `input_scale`
(= calibrated input absmax × 1.25 / 240) in the fp8 checkpoint and the device quantizes as
`(x.f32 × 1/input_scale).clamp(±240).to(f8)` — no reductions. Calibration
(`scripts/ptq_calibrate_activations.py`, real CPU loop, real UMT5 prompt, 20 steps, 46 min on
12 cores): 400 layers; per-step absmax spread median ×1.58, p90 ×2.9; only `attn2.to_out.0`
swings widely (up to ×52, `calibration_spread.txt`).

| arm (same fp8-tensor weights) | DiT step ms (median, n=19) | vs bf16 573.0 | latent SNR vs bf16 | host-VAE PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| dynamic law (round 2, `lean_round2`) | 607.2 | 1.060× | 13.2 dB (cos 0.9758, `ab-fixed` 10-01) | 24.5–24.9 dB / 0.88 / 0.12 |
| **static scales** (`static/ab`, module `MODULE_caa9e95d…`) | **586.9** | **1.024×** | **24.9 dB** (cos 0.9984) | **33.7 dB / 0.953 / 0.035** |
| same static checkpoint, dynamic law forced (`static/forced_dynamic`, `DIFFLET_FP8_IGNORE_INPUT_SCALE=1`) | 607.7 | 1.061× | 12.8 dB (cos 0.9735) | — |

The speed gain is the two absmax reductions per linear (20 ms). The quality gap is **not** a
static-scale advantage in exact arithmetic — it is a defect of the dynamic path end to end:

- the forced-dynamic run proves the checkpoint, weights, calibration and compiler cache are
  not the cause (same checkpoint, only the scale source differs);
- the HLO of the dynamic module is exactly the intended math (`hlo_scale.py`: bf16 max / −min
  reductions over all dims, ÷240 × 1.0078, clamp_min 8.1e-6, reciprocal, broadcast multiply,
  fp8 cast — no clamp, as designed);
- the tiny Wan probe agrees with the CPU dynamic reference at cosine 0.99997 with plain inputs
  and 0.9994–0.99997 with heavy-tailed inputs (`static/outlier_probe`); the HunyuanVideo
  probe with the REAL first 2+2 blocks and real text at production shape agrees at 0.99995
  (`../hv-fp8/real_probe`);
- on CPU, on the real activations of steps 0 / 10 / 19 (`act_quant_error_cpu.json`,
  `act_quant_error_cpu_summary.txt`), dynamic and static are numerically equivalent per layer:
  median linear-output SNR 32.2 vs 32.0 dB (weight-only 35.0), activation SNR 31.5 dB both,
  subnormal share 0.2 % both, no clipping in the static law.

So a single forward of the dynamic law is right on the device and equal to static on CPU, yet
20 steps of it on the device (tp4) drift to 13 dB while static stays at 25 dB. Hypotheses
measured and eliminated:

- step-to-step coherence of the fp8 grid (`DIFFLET_FP8_DYN_POW2=1`: power-of-two scales, the
  grid only moves across binades, `static/pow2_dynamic`): latent SNR 13.8 dB, PSNR 25.4 dB —
  no better than plain dynamic (and 652 ms: the scalar log2/pow is not free);
- TP4 (`static/real_probe/probe_tp4_tiny_out.log`): the tiny model with 1 % × 200 outliers at
  `--tp-degree 4` still agrees with the CPU dynamic reference (cosine 0.9996);
- non-finite activation elements (`DIFFLET_FP8_SANITIZE=1`, HunyuanVideo): no change.

One denoise step at full depth on the device (`static/onestep`): static vs forced-dynamic
latents agree at 27.3 dB SNR (cosine 0.99908) — consistent with two independent ~30 dB
quantization errors, i.e. after ONE step the dynamic prediction is as good as the static one.
The 12 dB gap only exists after 20 steps: whatever the mechanism, it is an accumulation effect
of the amax-derived scale through the sampling loop, not a wrong forward. Practical
conclusion: ship static scales, keep dynamic as the uncalibrated fallback and say so.

### Where the static step's remaining 14 ms over bf16 go (`profile/fp8_static`)

Same capture as the first two profiles (rank 0, exec 3 of 3):

| metric (per forward) | bf16 | fp8 dynamic | **fp8 static** |
|---|---:|---:|---:|
| total active time | 517 ms | 576 ms | **539 ms** |
| tensor engine active | 358 ms | 322 ms | 349 ms |
| vector engine active | 202 ms | 268 ms | 211 ms |
| scalar engine active | 191 ms | 234 ms | **254 ms** |
| gpsimd engine active | 93 ms | 36 ms | 45 ms |
| HBM read / write | 69.7 / 15.7 GB | 40.5 / 31.7 GB | **27.5 / 14.6 GB** |
| spill save / reload | 11.1 / 13.4 GB | 28.0 / 27.3 GB | **10.1 / 9.2 GB** |
| vector / scalar instructions | 758 k / 529 k | 994 k / 674 k | 887 k / 713 k |

Static removed the memory traffic the dynamic path added (reads below bf16: fp8 weights;
writes and spills back at bf16 level) and the vector engine is back at bf16 level. What is
left is the **scalar engine (+63 ms over bf16)** — the per-element quantize chain
(convert → multiply → clamp → fp8 cast) feeding every dot — and it is 20 ms more than the
dynamic law's scalar time, which has no clamp. Experiment `static/clamp`:

| run | result |
|---|---|
| tiny probe, static scales, gain 1 | device static == CPU static, cosine 0.99999 |
| tiny probe, static, inputs ×8 past the calibration, clamp on | device == CPU (both clamp), cosine 0.99837 |
| same, clamp off (`DIFFLET_FP8_STATIC_NO_CLAMP=1`) | device output NaN: the fp8 cast of a value above 240 is **NaN, not a saturate** |
| Wan 2.1 production static arm, clamp off | **586.4 ms** (clamp on: 586.9) and the render collapsed to 8.1 dB PSNR |

So the clamp is mandatory and free; the scalar-engine cost is the convert chain itself
(bf16 → f32, multiply, cast). The remaining levers for it are structural (fold the
per-layer 1/input_scale into the preceding modulation so the layer input arrives pre-scaled
and only the cast remains, or an NKI kernel that quantizes in SBUF) and are out of scope for
this pass: static scales close the gap to 1.024× bf16 with better fidelity than the dynamic
law, and that is the path the fan-out uses.

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
