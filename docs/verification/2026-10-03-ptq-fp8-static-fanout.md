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
| HunyuanVideo 13B (320×512×61, 20) | bf16 | 4632 ¹ | 596 | 113.0 | 832.0 ⁵ | — | — | ok |
| | fp8 dynamic | — | — | — | 760.8 ⁵ | 0.914× | 5.8 dB: garbage | **FAIL** |
| | fp8 static | — | — | — | 728.4 ⁵ | 0.875× | 9.3 / 0.156 / 0.889: garbage | **FAIL (open, below)** |

¹ dominated by the VAE decoder compile (Wan 6100 s). ² Wan 2.1 A/B numbers (`fp8-dyn-step`),
same harness for the three arms. ³ FLUX logs carry no per-step timer line; the number is the
28-step denoise-loop rate (bf16 3.61 it/s, static 3.80 it/s). ⁴ full bf16 build from 10-02 (the
10-04 bf16 re-run hit the compile cache). ⁵ host-VAE A/B runs (`hv-fp8/`); the DiT step does
not depend on the decode placement.

What the table shows (five of six models pass):

- **DiT step**: fp8 static is faster than bf16 on the image / joint-attention models
  (Qwen-Image −13 %, FLUX −5 %, LTX-2 −4 %) and 2–4 % slower on Wan, where the scalar-engine
  quantize chain costs a little more than the fp8 dots save (profile in `fp8-dyn-step/NOTES.md`).
- **Cold e2e**: −17 % to −28 % everywhere (fp8 transformer weights are half the bytes):
  Wan 2.1 414 → 298 s, Wan 2.2 405 → 300 s, FLUX 307 → 243 s, Qwen-Image 482 → 401 s,
  LTX-2 824 → 632 s.
- **Warm e2e**: at or below bf16 on every model (−1 % to −8 %).
- **Compile**: the fp8 arms reuse the bf16 VAE / text-encoder artifacts and build only the
  transformer (Wan 478–661 s against 6600–7900 s for a full bf16 build); FLUX's components
  share one stage, so its fp8 build (1349 s) is longer than its bf16 one.
- **Quality**: 33–37 dB PSNR against the bf16 render for Wan 2.1 / 2.2, Qwen-Image and LTX-2;
  FLUX 29.2 dB with SSIM 0.974 / LPIPS 0.05 (a single sharp 1024² image, where per-pixel PSNR
  is strict). Frame grids and compare JSONs under `artifacts/verification-2026-10-02/ptq-all/<model>/`.

## HunyuanVideo: open

The fp8 × fp8 path renders garbage at production depth with both activation laws, while
weight-only (bf16 matmuls on dequantized fp8 weights, 10-02) rendered at 31 dB.

**Full-model probe (real weights, real text, production shape, tp4, one forward;
`hv-fp8/real_probe/report_tp4_d20s40.json`)**: the device bf16 arm matches the CPU at cosine
0.99996; the CPU fp8-dynamic reference is healthy (31.6 dB against CPU bf16); the **device
fp8-dynamic output is entirely NaN** (655 360 of 655 360 elements). The shallow probe (2
double + 2 single blocks) matched at 0.9999, so a single forward goes non-finite somewhere
in the deep stack on the device only. The leading hypothesis: the lean dynamic law has no
clamp (it relies on `scale = absmax/240 × (1+2⁻⁷)` bounding the scaled tensor), and the
device's fp8 cast of a value above 240 is NaN (proven today on the static path); if the
on-device absmax reduction under-reads at HunyuanVideo's joint sequence length (10 496 tokens
× 3072), values exceed 240 and turn into NaN. The static path clamps, which explains why it
is not NaN but still wrong if the same reduction / cast issue sits elsewhere. Next: re-add the
clamp to the dynamic law behind a switch and re-run this probe; probe the reduction length.
