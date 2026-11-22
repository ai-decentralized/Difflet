# MXFP4 weights on the linear layers: weight-movement profile and CPU quality estimate (Trainium3)

**Status:** phases 0 and 1 of the MXFP4 evaluation complete (2026-11-22). No FP4 kernel was
written and nothing ran on a device with FP4 operands: phase 1 profiles the existing MXFP8 path,
phase 0 is a CPU simulation on one prompt and one seed. Host: trn3.48xlarge, Neuron SDK
torch_neuronx 2.14.3, nki 0.7.0, neuronx-cc 2.28. Branch `feat/mxfp4-linear` (main `465c9446`
plus `scripts/mxfp4/`; the capture tooling `scripts/profile_dit/` is borrowed from
`feat/mxfp8-sage-attention-rebased` and not on this branch).

**Short answer.** MXFP4 can only replace the *weights* of the linear layers (activations stay
MXFP8; attention has no weights to replace). On Qwen-Image and FLUX.1-dev at tp4, 1024x1024:

| | Qwen-Image | FLUX.1-dev |
|---|---|---|
| step time FP4 could save with the kernel as it is, ceiling (section 2) | 5.2 ms of 233.7 ms = 2.2% | 5.7 ms of 176.2 ms = 3.2% |
| the same if only the fitted byte-dependent part of each transfer shrinks | 1.9 ms = 0.8% | 2.8 ms = 1.6% |
| of that, the part overlapping idle TensorE inside a linear call | 0.15 ms = 0.06% | 0.65 ms = 0.4% |
| HBM per rank FP4 would free, every target in FP4 (estimate, section 3) | 1.45 GiB of 8.79 GiB | about 0.8 GiB of 15.6 GiB (peak core) |
| image against CPU fp32, 50 steps, MXFP8 (CPU simulation, section 4) | 38.76 dB | 25.28 dB (no verdict: chaotic at 50 steps) |
| the same with every target's weight in FP4 | 20.85 dB (`ocp` law), 24.41 dB (`mse` law) | 20.49 dB (within 1 dB of the bf16 floor, 21.47 dB) |

With the present kernel, FP4 on every target buys at most 2-3% of the step; the three rows are
read off an MXFP8 profile and none is a forecast. On the one prompt and seed simulated,
every-target FP4 costs 14-18 dB of fidelity to the fp32 image on Qwen-Image (the FP4 images are
clean and show the same scene; pose, bench and background drift). Mixed sets (FP4 on the
text-stream layers, with or without one image-stream group) land between: 25.2 dB with `ocp`
(text stream), 27.5 and 30.9 dB with `mse` (text stream; text stream + image MLP-up), each on one
image, and their speed ceiling is half the table's or less (section 2). One seed does not
separate differences of a few dB between FP4 arms from scatter.

Terms: tp4 = tensor parallelism over 4 ranks; a rank is one logical core of two physical cores
(LNC=2). TensorE = the core's matrix engine; HBM = device memory. Text stream and image stream =
the two token streams of a dual-stream block (text targets: `add_q/k/v_proj` and `txt_mlp` or
`ff_context`). MLP-up and MLP-down = the MLP's first and second linear. `ocp` and `mse` = the two
FP4 scale laws of section 4.1.

## 1. What the SDK allows

From the SDK source, as read by the research run of 2026-11-22 and re-checked by a second
reader:

- `nisa.nc_matmul_mx` takes `float4_e2m1fn_x4` for either operand; `nisa.quantize_mx` writes FP8
  only. So an FP4 operand must be quantized on the host: weights can be FP4, activations cannot,
  and the attention products (Q K^T, P V), which have no weight operand, cannot use FP4.
- FP4 (`float4_e2m1fn`): magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 and a sign; four codes in one
  uint16. Block membership is FP8's: one E8M0 scale per 8 partitions x 4 packed values, so
  `difflet/quant/mx.py`'s `swizzle_perm` and `BLOCK = 32` carry over.
- nkilib's offline FP4 law (`quantize_to_mx`): scale byte `e_max - 2`, the block maximum in
  [4, 8), values in (6, 8) saturate at 6, round to nearest even. The kernel only dequantizes
  with the byte it is given, so another exponent per block is legal as far as the SDK source
  goes (an inference: no nkilib kernel ships an FP4 weight with a shifted exponent).
- An FP4 weight changes neither the number of `nc_matmul_mx` instructions nor the tile shapes
  (simulator run; nkilib's sizing ignores the weight's element width). Whether TensorE runs an
  FP4 operand at FP8's rate was not established from the SDK or on a device. FP4 halves the
  weight *data* bytes only: on the kernel path the scales are a `[K/4, N]` byte tensor, 0.25
  byte per weight, moved whole, so weight traffic goes from 1.25 to 0.75 bytes per weight.
- Not verified on a device: that neuronx-cc compiles an FP4 operand; the hardware's nibble
  order; that the hardware accepts a block exponent other than nkilib's (the `mse` law of
  section 4); and whether nkilib's `matmul_mxfp8` takes an FP4 right-hand side unmodified (the
  CPU simulator accepted one where K per rank is a multiple of 512, which is the simulator
  agreeing with itself, not a hardware check).

## 2. Phase 1: how much of the step is weight movement

Step baselines on main, `difflet bench run --metrics dit_step`, MXFP8, tp4, 1024x1024, 50 steps,
seed 42, one DiT forward per step (Qwen-Image with `--true-cfg-scale 1.0`, the setting of every
stored reference):

| model | device, CPUs | DiT step median | n | min / p90 / max |
|---|---|---|---|---|
| Qwen-Image | 2, 48-63 | 233.749 ms | 147 | 232.136 / 235.017 / 494.649 ms (one slow step) |
| FLUX.1-dev | 1, 64-79 | 176.157 ms | 147 | 174.355 / 177.179 / 178.938 ms |

FLUX's `dit_step` carries the bench flag `unchecked: no denoise span` (its loop runs inside
diffusers' `pipe()`; the samples are intervals between step callbacks, host scheduler included).
It agrees with the morning's device-8 run (176.241 ms, n=196).

Block profiles: `scripts/profile_dit/capture.sh --key <model> --arm mxfp8`, run from this
checkout: a 1- or 2-step generate under `neuron-explorer inspect`, then each block NEFF replayed
with four workers and DGE notifications, execution 4 of rank 0 exported to parquet. Both lanes
ended `DONE` (three block captures). Every profile number below comes from that one execution of
one rank.

| block | NEFF sha256 | executions per step | replay | in the model (median) |
|---|---|---|---|---|
| Qwen-Image block | `5e23e168...` | 60 | 4.093 ms (4.006 ms in the other profiled execution) | 3.701 ms |
| FLUX dual block | `455a29dd...` | 19 | 3.432 ms | 3.248 ms |
| FLUX single block | `63a32664...` | 38 | 2.805 ms | 2.689 ms |

The Qwen-Image NEFF is not the one profiled on 2026-11-13 (`781649e0...`): main's attention
changed since (#164). With the same script that capture gives `U` and `W` within 6% of the
values below (`U` 0.1843 against 0.1735 ms, `W` 0.5901 against 0.5919 ms per block); its fitted
and idle-overlap rows are 29% and 26% higher (2.47 and 0.19 ms per step): the fit moves that much
between captures.

`scripts/mxfp4/weight_dma.py <blk> --execs-per-step N --step-ms X --out <json>` reads the
parquet. Per physical core, mean of the two, by interval union: `U` = time any transfer of the
weight data (`w_data`) is in flight; `W` = the MX linear calls' windows (`W_own`: the calls' own
instructions inside them); `I_win` = the part of `U` inside `W` during which TensorE is idle.
With the kernel as it is (same tiles, block sizes and number of transfers) FP4 halves the bytes
of `U`'s transfers and nothing else, so:

- ceiling = `0.5 * U`: transfer time proportional to bytes, all of it on the critical path;
- fitted = `r * U`: least squares `duration = a + s * bytes` over the transfers, then
  `r = s * sum(bytes) / (2 * sum(durations))`. The fit is loose: R^2 0.04-0.49 by core, and `r`
  is 0.24 and 0.13 on the two cores of the Qwen-Image capture;
- idle overlap = the fitted saving of only the transfer time under which TensorE is idle inside
  the calls. An overlap, not a measured wait.

Per step (block value x executions), as ms and share of the step median above:

| | Qwen-Image | FLUX dual | FLUX single | FLUX total |
|---|---|---|---|---|
| MX linear calls' windows, `W` | 35.52 ms, 15.2% | 16.74 ms | 33.86 ms | 50.60 ms, 28.7% |
| of `W`, the calls' own instructions, `W_own` | 34.46 ms, 14.7% | 10.27 ms | 24.53 ms | 34.79 ms, 19.8% |
| weight data in flight, `U` | 10.41 ms, 4.45% | 3.51 ms | 7.89 ms | 11.39 ms, 6.47% |
| weight scales in flight | 3.97 ms | 1.58 ms | 3.40 ms | 4.99 ms |
| weight data per core and execution | 51.9 MB | 51.9 MB | 56.6 MB | |
| FP4 ceiling, `0.5 U` | 5.20 ms, 2.23% | 1.75 ms | 3.94 ms | 5.70 ms, 3.23% |
| FP4 ceiling, text-stream targets only | 2.46 ms, 1.05% | 0.78 ms | none | 0.78 ms, 0.44% |
| FP4 fitted, `r U` (`r`) | 1.91 ms, 0.82% (0.18) | 0.64 ms (0.18) | 2.18 ms (0.28) | 2.82 ms, 1.60% |
| FP4 idle overlap | 0.15 ms, 0.06% | 0.13 ms | 0.52 ms | 0.65 ms, 0.37% |

Reading:

- These are overlaps on an MXFP8 profile, not a measured gain. The replay runs 4-11% slower than
  the block inside the model, so the per-step values are slightly high. The values are the mean
  of the two physical cores; with the busier core the FLUX ceiling is 6.1 ms (3.5%). On FLUX the
  compiler interleaves other work into the call windows, so `W` is wider than the linear layers'
  own work and the idle overlap includes idle time that belongs to that other work.
- The ceiling is one for the unchanged kernel. A kernel re-tuned for the narrower weight (larger
  blocks, fewer activation re-loads), the utilisation throttle and TensorE's rate on an FP4
  operand are outside `U` and were not examined.
- Text-stream FP4 on Qwen-Image with image MLP-up added has a ceiling of 3.17 ms (1.36%).
- An A/B with an FP4 kernel on the device step is what could confirm a gain. The step time has a
  standard deviation of 0.83 ms over the 147 steps of the FLUX bench run (0.85 ms on Qwen-Image
  without its one 495 ms step), and the medians of the three 50-step requests span 0.67 ms
  (FLUX) and 0.73 ms (Qwen-Image): enough to resolve the ceiling and the fitted value, not the
  idle overlap.
- The scale row is untouched by FP4 and is nearly as large as the FP4 ceiling. It might be
  reduced without FP4 and without any change in arithmetic: the kernel moves the scales as a
  `[K/4, N]` byte tensor in which 4 rows of 32 hold values, and nkilib's `matmul_mxfp8` has an
  `enable_scale_packing` option (read in the source, not tried). Compact scales would also free
  0.22 byte per weight for MXFP8 as it is, about 0.63 GiB per rank on Qwen-Image.

## 3. Memory estimate

A quantized target holds 1.28125 bytes per weight resident on the kernel path (packed data 1.0,
the kernel's scale layout 0.25, stored scales 0.03125; 1.25 where q/k/v are fused into one call,
as on Qwen-Image); FP4 data in the same layout saves 0.5. Default targets: Qwen-Image 12.457 B
of 20.430 B DiT parameters, FLUX 8.248 B of 11.901 B. At tp4 a rank holds a quarter of them:

| | targets in FP4 | saved per rank | peak per rank today (MXFP8) |
|---|---|---|---|
| Qwen-Image, every target | 12.457 B | 1.45 GiB | 8.79 GiB |
| Qwen-Image, text-stream targets | 6.229 B (50%) | 0.73 GiB | |
| FLUX, every target | 8.248 B | 0.96 GiB on the DiT-only ranks, about 0.82 GiB on the peak core | 15.59 GiB (peak core), 8.92 GiB (DiT-only ranks) |
| FLUX, text-stream targets | 1.972 B (24%) | 0.23 GiB | |

Estimates, not measurements. Assumed: the same target set, layout and sharding, and that the
whole weight saving reaches the peak. The last held for bf16 to MXFP8 on Qwen-Image (2.06 GiB
measured against 2.09 predicted) and on FLUX's DiT-only ranks (1.34 against 1.38), but on FLUX's
peak core, which also holds other components, 85% of it did: hence about 0.82 GiB there. The
peaks are the bench `memory` metric (`peak_gib_per_core`) of the morning's device-8 quant A/B
runs (`/mnt/nvme/difflet-bench/results-validation/trn3/{qwen_image,flux}@quant_ab.mxfp8/`), not
of the runs in section 2, which measured `dit_step` only. A core has 36 GiB of HBM, so memory is
not binding at tp4 for either model.

## 4. Phase 0: CPU quality estimate

### 4.1 Method

`scripts/mxfp4/run_arm.py` runs one arm through `scripts/fp32_ref/<model>.py`'s own `main`
(diffusers' pipeline in fp32 on the CPU from the device's initial noise: "a cat sitting on a
bench", seed 42, 50 steps, 1024x1024, no true CFG) and swaps the DiT's target linears first:

- `mxfp8`: `difflet.quant.fake_linear.FakeQuantMXLinear` on the model's default targets at tp4:
  the kernel's quantization law (both operands block-quantized, fp32 accumulation), without the
  device's bf16 rounding of the kernel's input and output;
- `fp4-ocp`, `fp4-mse`: the same layers and the same MXFP8 activation law, with the dequantized
  weight replaced by the FP4 law's (`scripts/mxfp4/fp4.py`) on every target or on the targets
  `--fp4-targets` names. `ocp` is nkilib's law, bit-equal to `quantize_to_mx(float4_e2m1fn_x4)`
  on the kernel layout (the script's self-check); `mse` takes, per block, the better of that
  exponent and the next one.

An FP4 arm therefore differs from the `mxfp8` arm in the weight format of the named layers and
in nothing else. Scores are `scripts/compare_difflet_runs.py` against the `fp32` arm of the same
harness. Each arm: 8 threads on 8 cores, 86 s per step in fp32 and 108-144 s quantized.

Weight error of the laws on nine real layers (rel L2 against the fp32 weight): MXFP8
0.0265-0.0266, FP4 `ocp` 0.115-0.119, FP4 `mse` 0.112-0.115 (3% under `ocp`).

### 4.2 Controls

| check | PSNR | latent rel L2 |
|---|---|---|
| Qwen-Image `fp32` arm vs the stored 96-thread reference (`fp8-bench/fp32_ref/qwen_image`) | 77.71 dB | 3.5e-5 |
| FLUX `fp32` arm vs the stored reference | 77.27 dB | 4.2e-5 |
| Qwen-Image `mxfp8` arm vs the device MXFP8 run (`qwen_mx_hm0_tp4`) | 35.44 dB | 0.075 |

The harness reproduces the stored references. The CPU `mxfp8` arm is as far from the device's
MXFP8 image as the device's bf16 image is from fp32 (36.34 dB): the arm has no bf16 arithmetic,
so its scores are not device scores. For Qwen-Image the device measured 36.34 dB (bf16) and
37.36 dB (MXFP8) against the same fp32 reference; the CPU `mxfp8` arm reads 38.76 dB.

### 4.3 50 steps

| model | arm | targets in FP4 | PSNR vs fp32 | SSIM | LPIPS | latent cosine | PSNR vs the device's bf16 image |
|---|---|---|---|---|---|---|---|
| Qwen-Image | `mxfp8` | 0 of 600 | 38.76 dB | 0.9928 | 0.0097 | 0.99856 | 33.74 dB |
| Qwen-Image | `fp4-mse`, text stream + image MLP-up | 360 | 30.88 dB | 0.9804 | 0.0296 | 0.99364 | 31.46 dB |
| Qwen-Image | `fp4-mse`, text stream | 300 | 27.49 dB | 0.9676 | 0.0520 | 0.98967 | 28.22 dB |
| Qwen-Image | `fp4-ocp`, text stream | 300 | 25.17 dB | 0.9574 | 0.0662 | 0.98442 | 25.61 dB |
| Qwen-Image | `fp4-mse`, every target | 600 | 24.41 dB | 0.9473 | 0.0849 | 0.97606 | 24.54 dB |
| Qwen-Image | `fp4-ocp`, every target | 600 | 20.85 dB | 0.9216 | 0.1317 | 0.95021 | 20.83 dB |
| FLUX.1-dev | `mxfp8` | 0 of 418 | 25.28 dB | 0.9391 | 0.0473 | 0.98515 | |
| FLUX.1-dev | `fp4-ocp`, every target | 418 | 20.49 dB | 0.8652 | 0.1658 | 0.94706 | |

- Every FP4 image is a clean, plausible image of the prompt (no noise or artifacts); what drops
  is fidelity to the fp32 image.
- One image per arm, so no scatter was measured. The Qwen-Image mixed arms are not monotonic:
  360 FP4 layers score 3.4 dB above a 300-layer subset of them, with the same law. That is
  either scatter or an interaction between the groups; one seed cannot tell. The 3.6 dB between
  `ocp` and `mse` on every target is of the same size, for laws whose weight errors differ by
  3%. For scale, the MXFP8 gate run (2026-10-31, section 6h) found 4-5 dB between seeds of one
  prompt for one fixed recipe. The 14-18 dB of the every-target arms is outside that range; the
  7.9-13.6 dB of the mixed arms is not clearly outside it.
- The gate accepted for MXFP8 on Qwen-Image is on another scale: the device image against the
  device's bf16 image of the same seed, mean PSNR over 4 prompts x 3 seeds at least 30 dB and
  every image at least 27 dB (the device's MXFP8 scored 38.26 dB on this prompt and seed). No
  arm here passes or fails it: these are CPU simulations of one image. Against that bf16 image
  (last column) the every-target arms read 20.8 and 24.5 dB, under both levels; the mixed arms
  read 25.6, 28.2 and 31.5 dB and need the gate's seeds and prompts to be judged.
- FLUX gives no 50-step verdict. Its trajectory is chaotic at 50 steps: the CPU bf16 pipeline is
  21.47 dB from fp32, and the device scores 23.36 dB (bf16) and 21.30 dB (MXFP8) against the
  same reference. The FP4 arm's 20.49 dB is within 1 dB of those levels and its image is as far
  from the `mxfp8` arm's image (20.99 dB) as from fp32. For FLUX the evidence is the single
  forward of section 4.4.

### 4.4 One forward: per-group sensitivity

`scripts/mxfp4/sensitivity.py` replays the fp32 arm's DiT inputs at steps 0, 12, 25, 37, 49 and
measures the DiT output's rel L2 against the fp32 output (`ocp` law; the unquantized replay is
exactly 0). "+ FP4 on g" is MXFP8 on every target with group g's weights in FP4. Mean of the
first four steps, and the last step:

| Qwen-Image | steps 0-37 | step 49 |
|---|---|---|
| every target MXFP8 | 1.42% | 9.33% |
| + FP4 on text MLP-down (60 linears) | 1.46% | 9.35% |
| + FP4 on text q/k/v (180) | 1.53% | 9.44% |
| + FP4 on text MLP-up (60) | 1.58% | 9.39% |
| + FP4 on image MLP-up (60) | 2.27% | 13.44% |
| + FP4 on image MLP-down (60) | 2.86% | 13.28% |
| + FP4 on image q/k/v (180) | 3.01% | 13.14% |
| every target FP4 | 3.45% | 18.76% |

| FLUX.1-dev | steps 0-37 | step 49 |
|---|---|---|
| every target MXFP8 | 1.60% | 5.89% |
| + FP4 on dual text MLP-up (19) | 1.63% | 5.90% |
| + FP4 on dual text q/k/v (57) | 1.72% | 5.89% |
| + FP4 on dual text MLP-down (19) | 1.88% | 5.92% |
| + FP4 on dual image MLP-down (19) | 2.06% | 6.74% |
| + FP4 on single `proj_out_mlp` (38) | 2.57% | 7.72% |
| + FP4 on single `proj_out_attn` (38) | 2.77% | 7.03% |
| + FP4 on dual image q/k/v (57) | 3.16% | 6.96% |
| + FP4 on single MLP-up (38) | 3.21% | 8.42% |
| + FP4 on single q/k/v (114) | 3.26% | 7.34% |
| + FP4 on dual image MLP-up (19) | 4.21% | 7.87% |
| every target FP4 | 5.78% | 12.04% |

On the mean of steps 0-37 the text-stream groups add 0.03-0.16 points on Qwen-Image (0.03-0.28
on FLUX) and every image-stream or single-stream group costs 1.3-2.6 times the MXFP8 error
(1.1-1.4 times at step 49). The mean hides step 0, where Qwen-Image's text MLP-up adds 0.46
points (1.88% against 1.42%) and FLUX's dual text MLP-down 0.98. This ranking did **not** predict
the 50-step images: with the three text-stream groups together in FP4 (`mse`) the forward's
error rises by 0.09-0.34 points (4.5), and the image loses 11.3 dB (`mse`) or 13.6 dB (`ocp`).

### 4.5 A diagnostic: weight error is correlated across steps, activation error is not

`scripts/mxfp4/error_coherence.py` replays the same Qwen-Image captures with the quantization
split into its two parts, and reports the cosine between the output-error vectors of every pair
of captured steps:

| configuration | rel L2, steps 0-37 | step 49 | mean cosine, 10 pairs | without step 49, 6 pairs |
|---|---|---|---|---|
| MXFP8 activations, fp32 weights | 1.16-1.34% | 8.82% | -0.002 | -0.002 |
| MXFP8 weights, fp32 activations | 0.63-0.76% | 3.23% | 0.261 | 0.360 |
| FP4 weights (`ocp`), fp32 activations | 2.68-3.73% | 16.83% | 0.351 | 0.482 |
| `mxfp8` arm (both) | 1.35-1.47% | 9.33% | 0.057 | 0.084 |
| `fp4-ocp` arm (both) | 2.91-3.87% | 18.76% | 0.301 | 0.416 |
| mixed arm, text stream FP4 (`mse`) | 1.48-1.76% | 9.56% | 0.070 | 0.106 |
| FP4 (`mse`) weights on the text stream alone, all else fp32 | 0.61-1.32% | 2.16% | 0.116 | 0.178 |

What the replays show: in one forward MXFP8's error is mostly the activation part, which is
unrelated between the captured steps; a quantized weight is the same perturbation of the model
at every step, and its error is correlated between steps (cosine up to 0.60 for MXFP8 and 0.72
for FP4, between steps 12 apart); FP4's weight error is 3.5-5.4 times MXFP8's weight error (the
laws' weight errors differ by 4.4 times). The parts add in quadrature within a forward
(`sqrt(1.16^2 + 0.76^2)` = 1.39 against 1.42 at step 0), which is the only cross-check of the
three split rows. Every measured pair is 12 or more steps apart; nearer steps (40% of the pairs
of a 50-step run) were not captured.

What they do not show: that this explains the 50-step images. With `N` = 50 steps of equal
weight and one cosine `c` for every pair, an error of size `e` per step would accumulate as
`e * sqrt(N) * sqrt(1 + (N - 1) c)`. From the arms' own rows that gives 13.9 dB between
`fp4-ocp` and `mxfp8` (measured: 15.5 dB in final-latent rel L2, 17.9 dB in PSNR) but 1.8 dB
between the text-stream arm and `mxfp8` (measured 8.6 dB / 11.3 dB). The estimate is not
validated, its partial agreement on the first may be chance, and by the same formula MXFP8's own
weight part would accumulate to twice its activation part, so it does not show MXFP8 is benign
because its error is activation error. No 50-step arm with weight-only or activation-only
quantization was run. Qwen-Image only; five steps of one trajectory.

## 5. Not done

- No FP4 operand on a device: no compile, no kernel, no step time, no HBM measurement.
- One prompt and one seed per arm. What would settle the mixed sets is the gate's own protocol
  (4 prompts x 3 seeds against bf16, with an MXFP8 control per image). The CPU harness cannot
  run it as written: `scripts/fp32_ref/<model>.py` fixes the prompt and the seed as module
  constants behind a noise md5, and `run_arm.py` has no `--seed` or `--prompt`.
- No `ocp` arm for text stream + image MLP-up; no FLUX mixed arm; no coherence run on FLUX.
- The profile covers one shape per model (tp4, 1024x1024), one execution of rank 0. Shapes with
  fewer rows against the same weights were not captured; there the weight share should be
  larger.
- The other models were not run. The 2026-11-13 captures put weight-data transfer time at 4.3%
  (LTX-2) and 1.3% (Wan 2.1, 81 frames) of the block, an FP4 ceiling of about 2.2% and 0.7%.

## 6. Reproduce

```bash
PY=~/venvs/native-fp8/bin/python; WT=<checkout>; cd $WT
# scripts/profile_dit is not on this branch: take it from the attention branch
git archive origin/feat/mxfp8-sage-attention-rebased scripts/profile_dit | tar -x

# phase 1: step baseline, capture, analysis (devices 1-15: a lane on device 0 reads busy while
# any other lane runs)
env DIFFLET_BENCH_ROOT=<dir> DIFFLET_COMPILE_CACHE=$HOME/difflet-cache-fp8 \
  HF_HUB_CACHE=$HOME/.cache/huggingface/hub $PY -m difflet.cli.main bench run \
  --recipe qwen_image_mxfp8 --metrics dit_step --devices 2 --cpus 48-63 \
  --results-dir <dir>/results -- --true-cfg-scale 1.0
env NEURON_NUMA_PIN=0 HF_HUB_CACHE=$HOME/.cache/huggingface/hub scripts/profile_dit/capture.sh \
  --key qwen_image --arm mxfp8 --device 2 --cpus 48-59 --out <prof>
$PY scripts/mxfp4/weight_dma.py <prof>/qwen_image/mxfp8/blk1 --execs-per-step 60 \
  --step-ms 233.749 --out qwen_blk1.json
# FLUX: DIFFLET_COMPILE_CACHE=$HOME/difflet-cache-video, --recipe flux_mxfp8, no pass-through;
# capture --key flux; weight_dma.py on blk1 (--execs-per-step 19) and blk2 (38), --step-ms 176.157

# phase 0: every process on its own 8 cores
P="env TORCH_DEVICE_BACKEND_AUTOLOAD=0 DIFFLET_BACKEND=cpu OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  NEURON_PLATFORM_TARGET_OVERRIDE=trn3 PYTHONPATH=$WT $PY -B"
taskset -c 48-55 $P scripts/mxfp4/fp4.py                       # the law's self-check
taskset -c 48-55 $P scripts/mxfp4/run_arm.py --model qwen_image --arm fp32 \
  --capture-steps 0,12,25,37,49 --out-dir <out>/qwen_image/fp32
taskset -c 56-63 $P scripts/mxfp4/run_arm.py --model qwen_image --arm fp4-ocp \
  --out-dir <out>/qwen_image/fp4-ocp                           # also mxfp8, fp4-mse
taskset -c 64-71 $P scripts/mxfp4/run_arm.py --model qwen_image --arm fp4-mse --fp4-targets \
  'transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj);transformer_blocks\.\d+\.txt_mlp\.net\.0\.proj;transformer_blocks\.\d+\.txt_mlp\.net\.2' \
  --out-dir <out>/qwen_image/mixed-txt-mse
taskset -c 48-55 $P scripts/mxfp4/sensitivity.py --model qwen_image \
  --capture-dir <out>/qwen_image/fp32 --rows replay,all,mxfp8+fp4 --out <out>/qwen_image/sens.json
taskset -c 56-63 $P scripts/mxfp4/error_coherence.py --model qwen_image \
  --capture-dir <out>/qwen_image/fp32 --out <out>/qwen_image/coh.json
$P scripts/compare_difflet_runs.py --ref-latents <out>/qwen_image/fp32/latents.pt \
  --ref-image <out>/qwen_image/fp32/out.png --cand-latents <out>/qwen_image/fp4-ocp/latents.pt \
  --cand-image <out>/qwen_image/fp4-ocp/out.png --out score.json
```

Outputs of this run: `/mnt/nvme/difflet-fp4/` (bench results, captures, analysis) and
`/mnt/nvme/mxfp4-phase0/` (arms, scores, sensitivity), instance storage; the scores, tables and
parquet copies are also under `/home/ubuntu/mxfp4-research-2026-11-22/`.
