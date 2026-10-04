# FP8 PTQ with static activation scales — all-model device comparison (2026-10-03)

trn2.3xlarge (1 device, 4 cores), tp4, neuronx-cc 2.26, branch `quantization`. This is the
result of the 2026-10-03 directive: find every DiT-step optimization for the FP8 path, commit
each verified one, then fan the final path out to every model with on-device verification.

## What changed today (one commit per verified step)

| commit | problem | why it was slow / wrong | what changed | Wan 2.1 DiT step |
|---|---|---|---|---:|
| `c93c83f` | two FP8 modes to maintain, weight-only not what FastVideo does | — | weight-only mode removed; FP8 PTQ is always W8A8 | — |
| `4f3f585` lean law round 1 | fp8-dynamic step 656.6 ms vs bf16 573.0 | the per-linear quantize ran abs / divide / clamp passes and two dequant multiplies in f32 over the full activation (profile: vector +66 ms, scalar +43 ms, HBM writes ×2, spills ×2.5) | absmax from amax / −amin reductions, scale with a 2⁻⁷ margin, one reciprocal multiply, no clamp, one combined dequant multiply | 656.6 → 607.7 ms |
| `48fc8b0` round 2 | the bf16 multiply compiled with an extra bf16 round trip | XLA keeps bf16 arithmetic in f32 | quantize in f32 form (fewer HLO ops, same speed) | 607.2 ms |
| `48f8eb9` round 3 (reverted) | share one quantize across q / k / v | a shared fp8 tensor with three consumers is materialised in HBM; per-linear quantize is fused into the dot's operand load | reverted | 639.3 → back to 607.2 ms |
| `946cb89` **static calibrated scales** | the last 34 ms were the two absmax reductions per linear (640 full reads of the activation per forward) that must finish before the scale is known | a dynamic per-tensor scale cannot be fused into the matmul operand producer | per-layer `input_scale` calibrated offline on the real CPU denoise loop (`--quant-calibration`); the device quantizes with a constant: multiply, clamp, cast — no reductions | **607.2 → 586.9 ms (1.024× bf16)** |

Also verified and recorded (not step optimizations): the HunyuanVideo orchestrator honours
`--host-vae` (`d8252c8`: fp8 arms no longer recompile the VAE decoder for an hour), the
dynamic law's end-to-end fidelity defect and the mandatory clamp (below), neuronx-cc 2.27 is
not a lever (1 % either arm).

## The quality finding: static scales are also more faithful than dynamic ones

Same fp8 weights, Wan 2.1 480×832×9, 20 steps, seed 42 (`artifacts/verification-2026-10-03/fp8-dyn-step/static/`):

| arm | DiT step ms | latent SNR vs bf16 | host-VAE render vs bf16 (PSNR / SSIM / LPIPS) |
|---|---:|---:|---|
| dynamic law | 607.2 | 13.2 dB | 24.5 dB / 0.88 / 0.12 |
| **static scales** | **586.9** | **24.9 dB** | **33.7 dB / 0.953 / 0.035** |
| static checkpoint through the dynamic law | 607.7 | 12.8 dB | — |

The dynamic law is right per forward (device == CPU reference at cosine ≥ 0.9994 in every
probe: plain, heavy-tailed, real weights + real text, tp1 and tp4; CPU per-layer error
identical to static; one full-depth step agrees with static at 27 dB) and drifts over the
20-step loop; power-of-two scales, non-finite inputs and TP were ruled out. The static
path's clamp is mandatory (the device's fp8 cast of a value above 240 is NaN) and free
(586.4 ms without it). Details and HLO / profile evidence: `fp8-dyn-step/NOTES.md`.

## All-model comparison (bf16 vs fp8 static scales)

Benchmark harness (`benchmark/trn2/<slug>{,_fp8_static}.json`, `scripts/ptq_model_verify.sh`
with `ARMS="_fp8_static"`): compile wall, true-cold and warm e2e (device VAE where the model
compiles one), in-process DiT step, and the fp8 render compared to the bf16 render of the same
prompt / seed / shape. Calibration = `scripts/ptq_calibrate_activations.py` on the real CPU
loop at the benchmark shape (per-model loops in `scripts/calib_models/`). All fp8-static
rows and the Qwen-Image / LTX-2 bf16 rows were measured on an idle host on 2026-10-03/04.

| model (shape, steps) | arm | compile s | cold e2e s | warm e2e s | DiT step ms | fp8 / bf16 step | render vs bf16 (PSNR / SSIM / LPIPS) | status |
|---|---|---:|---:|---:|---:|---:|---|---|
| Wan 2.1 14B (480×832×9, 20) | bf16 | 7879 ¹ | 413.9 | 85.7 | 573.0 ² | — | — | ok |
| | fp8 dynamic | 505 | 304.3 | 84.2 | 607.2 ² | 1.060× | 33.0 / 0.888 / 0.128 | ok, drifts |
| | **fp8 static** | **478** | **298.0** | **82.5** | **586.9** ² | **1.024×** | **36.7 / 0.929 / 0.076** | **ok** |
| Wan 2.2 A14B (480×832×9, 20) | bf16 | 6581 ¹ | 404.9 | 81.5 | 572.6 | — | — | ok |
| | **fp8 static** | **661** | **300.1** | **78.3** | 597.2 | 1.043× | **35.2 / 0.919 / 0.099** | **ok** |
| FLUX.1-dev 12B (1024², 28) | bf16 | 1038 | 306.7 | 39.9 | 277 ³ | — | — | ok |
| | **fp8 static** | 1349 | **243.1** | **38.6** | **263** ³ | **0.95×** | 29.2 / 0.974 / 0.051 | **ok** |
| Qwen-Image 20B (1024², 20) | bf16 | 1254 ⁴ | 481.8 | 66.5 | 415.3 | — | — | ok |
| | fp8 dynamic (10-02) | 416 | 402.4 | 62.9 | — | — | 36.4 / 0.990 / 0.017 | ok |
| | **fp8 static** | **721** | **400.5** | **64.5** | **361.3** | **0.870×** | **33.3 / 0.988 / 0.026** | **ok** |
| LTX-2 19B (480×704×49, 20) | bf16 | 1839 ⁴ | 823.6 | 60.4 | 457.9 | — | — | ok |
| | **fp8 static** | **834** | **631.9** | **55.4** | **440.3** | **0.962×** | **33.9 / 0.936 / 0.080** | **ok** |
| HunyuanVideo 13B (320×512×61, 20) | bf16 | 4632 ¹ | 596.1 | 113.0 | 869.7 | — | — | ok |
| | fp8 dynamic ⁶ | — | — | — | 774.5 ⁵ | 0.89× | 28.9 / 0.907 / 0.087 ⁵ | ok |
| | **fp8 static** ⁶ | 4144 ¹ | **530.4** | **111.6** | **724.4** | **0.833×** | **26.8 / 0.892 / 0.102** ⁵ | **ok** |

¹ dominated by the VAE decoder compile (Wan 6100 s). ² Wan 2.1 A/B numbers (`fp8-dyn-step`),
same harness for the three arms. ³ FLUX logs carry no per-step timer line; the number is the
28-step denoise-loop rate (bf16 3.61 it/s, static 3.80 it/s). ⁴ full bf16 build from 10-02 (the
10-04 bf16 re-run hit the compile cache). ⁵ host-VAE A/B runs (`hv-fixed/`), compared with the bf16 host-VAE render of the same seed; the DiT step
does not depend on the decode placement. ⁶ with the double blocks' text q / k projections kept bf16 (section below).

What the table shows (all six models pass):

- **DiT step**: fp8 static is faster than bf16 on the image / joint-attention models
  (HunyuanVideo −17 %, Qwen-Image −13 %, FLUX −5 %, LTX-2 −4 %) and 2–4 % slower on Wan, where the scalar-engine
  quantize chain costs a little more than the fp8 dots save (profile in `fp8-dyn-step/NOTES.md`).
- **Cold e2e**: −11 % to −28 % everywhere (fp8 transformer weights are half the bytes):
  Wan 2.1 414 → 298 s, Wan 2.2 405 → 300 s, FLUX 307 → 243 s, Qwen-Image 482 → 401 s,
  LTX-2 824 → 632 s, HunyuanVideo 596 → 530 s.
- **Warm e2e**: at or below bf16 on every model (−1 % to −8 %).
- **Compile**: the fp8 arms reuse the bf16 VAE / text-encoder artifacts and build only the
  transformer (Wan 478–661 s against 6600–7900 s for a full bf16 build); FLUX's components
  share one stage, so its fp8 build (1349 s) is longer than its bf16 one; HunyuanVideo's fp8
  build (4144 s) still recompiles its device VAE (use `--host-vae` to skip it, `d8252c8`).
- **Quality**: 33–37 dB PSNR against the bf16 render for Wan 2.1 / 2.2, Qwen-Image and LTX-2;
  FLUX 29.2 dB with SSIM 0.974 / LPIPS 0.05 (a single sharp 1024² image, where per-pixel PSNR
  is strict); HunyuanVideo 26.8 dB / SSIM 0.892 (static) and 28.9 dB / 0.907 (dynamic), same scene and
  motion, finer texture differences (its 10-02 weight-only arm, bf16 matmuls, measured 31.1 dB). Frame grids and compare JSONs under `artifacts/verification-2026-10-02/ptq-all/<model>/`.

## HunyuanVideo: the tp4 failure and its fix

Before the fix, HunyuanVideo fp8 rendered garbage at tp4 with both activation laws (PSNR 6–9 dB),
while the CPU reference was healthy. Bisected on the device with the real weights and real text at
the production shape (`artifacts/verification-2026-10-04/hv-nan/`):

| probe (tp4 unless noted) | device fp8 vs CPU fp8 |
|---|---|
| real 2+2 blocks, tp1 | cosine 0.99995 |
| real 2+2 blocks, tp4 | all NaN |
| real full model, dynamic law clamped | finite but wrong (cosine 0.96, −7 dB) |
| 1+1 blocks, only image q/k/v, either attention output, image or text FFN, any single-block layer | cosine 0.99990–0.99996 |
| 1+1 blocks, only `add_q_proj` or only `add_k_proj` | all NaN |
| 1+1 blocks, only `add_v_proj` | cosine 0.99990 |
| text q/k/v fp8, dynamic law clamped | all NaN (so not an fp8 overflow) |
| text q/k/v fp8, text padded 256 → 512 or 1024 rows | cosine 0.99999 |

So the device graph mis-executes the fp8 text-stream q / k projection, which feeds the per-head
RMSNorm (`norm_added_q` / `_k`), at exactly 256 text rows under tp4. FLUX's text stream is 512 rows,
so it never hit this. Fix (`ec483f4`): HunyuanVideo's target set is FLUX's minus the double
blocks' `add_q_proj` / `add_k_proj` (bf16). The text stream is 256 of about 10.5 k tokens, so the cost is
negligible. After the fix: the real full model at tp4 matches the CPU at cosine 0.99981; the 20-step
render is 26.8 dB (static) / 28.9 dB (dynamic) against bf16, the same scene and motion
(`hv-fixed/frames_bf16_vs_static.png`); DiT step 0.83× bf16.
