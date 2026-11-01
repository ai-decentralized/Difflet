# Qwen-Image FP8 on the neuron (TorchNeuron) backend, Trainium3 — evidence

**Status:** in progress (2026-10-31). Host `trn3.48xlarge` (16 devices, LNC=2, 144 GB HBM per
device), Neuron SDK 2.33.0: torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631,
nki 0.7.0; venv `~/venvs/native-fp8` (the `/opt` TorchNeuron venv plus diffusers 0.38.0,
transformers 4.57.6). Branch `worktree-fp8-native` off main `40d7828`.

Recipe: W8A8 `float8_e4m3fn` (max 448), per-tensor weight scales, static calibrated
activation scales (margin 1.25), attention q/k/v/out and MLP up/down quantized, everything
else bf16 (`difflet/quant/targets.py`). Compute paths: `dequant` (fp8 weights, bf16 matmul)
and `nki` (the `nc_matmul_mx` kernel, `difflet/backends/neuron/ops_impl/fp8_kernels.py`).
Design and plan: the status page (artifact "Difflet FP8 on Trainium3").

Raw evidence (not in git): `/home/ubuntu/fp8-bench/` (per run: `*.log`, `*.png`,
`work_*/latents.pt`), `/tmp/logs/fp8_*.log` (supervised job logs),
`/home/ubuntu/difflet-cache-fp8/` (NEFF cache, quantized checkpoint).

## 1. Baseline: native bf16, tp4, 1024x1024, 50 steps, seed 42

`difflet compile|generate --model-id Qwen/Qwen-Image --backend neuron --tp-degree 4
--height 1024 --width 1024 --cache-dir /home/ubuntu/difflet-cache-fp8` (prompt "a cat sitting
on a bench"); `/home/ubuntu/fp8-bench/qwen_bf16_tp4/`.

| metric | value | source |
|---|---|---|
| compile (cold NEFF cache) | 2 min 01 s wall; 60 blocks; load+warmup 48.1 s | `compile` job log |
| DiT step | **336 ms** = (stage 48.0 s at 50 steps − 39.6 s at 25 steps) / 25; stage − load = 302 ms | `gen50a.log`, `gen25.log` |
| warm e2e (CLI wall) | 87.8 / 91.4 s (50 steps) | `gen50a.log`, `gen50b.log` |
| weights per rank | 10.22 GB (bf16) | weight-store line |
| determinism | gen50a.png == gen50b.png (md5 `38314d0d0a92…`), latents bit-identical | `compare_difflet_runs.py` |

The first run of the native backend on Trainium3; nothing needed changing.

## 2. CPU arbiter: bf16 vs fake-quant static FP8 (no device)

`scripts/qwen_image_fp8_cpu_reference.py --out /home/ubuntu/fp8-bench/cpu_arbiter --steps 20`
(diffusers `QwenImagePipeline` on 96 host threads, true_cfg 1.0, 1024x1024, seed 42; 4
prompts, the first 2 calibrate, 2 held out). bf16 on the CPU: 12–15 s/step.

Smoke run (512x512, 2 steps, 2 prompts; `cpu_arbiter_smoke/`): PSNR 38.65 / 33.81 dB.

Full run (1024x1024, 20 steps, seed 42; `cpu_arbiter/summary.md`, LPIPS from `lpips.json`),
fake-quant static FP8 against bf16, both on the CPU:

| # | prompt | role | PSNR dB | SSIM | LPIPS | latent cos |
|---|---|---|---:|---:|---:|---:|
| 0 | a cat sitting on a bench | calib | 33.99 | 0.981 | 0.026 | 0.9956 |
| 1 | watercolor lighthouse at dawn | calib | 37.72 | 0.989 | 0.017 | 0.9993 |
| 2 | brass typewriter, studio photo | held out | 32.34 | 0.975 | 0.034 | 0.9956 |
| 3 | astronaut on a bicycle, neon city, night | held out | **28.64** | 0.971 | **0.060** | 0.9908 |

Against the proposed gate (PSNR ≥ 30, SSIM ≥ 0.97, LPIPS ≤ 0.05 on every prompt) the fourth
prompt misses on PSNR and LPIPS; SSIM passes everywhere. The two images are the same scene
(same astronaut, bicycle, car, signs) with small differences in the crowd and the hand. The
bf16-vs-fp32 noise floor of this pipeline was not measured. Calibration: 720 linears; largest activation
amax 17,024 (`transformer_blocks.33.txt_mlp.net.2`), 15,936 (`…45.attn.to_add_out`); q/k/v
of one attention share one amax (they consume one tensor).

## 3. Kernel: `tests/manual/check_neuron_fp8_kernel.py`, one core, 5120 rows

rel = relative L2 error vs the plain-torch reference of the same arithmetic (fp8 operands,
fp32 accumulation, one output scale), eager and under `torch.compile(backend="neuron")`.

| shape (K → N) | tiles | rel | fp8 compiled | bf16 `F.linear` | ratio |
|---|---|---:|---:|---:|---:|
| tp4 q/k/v 3072 → 768 | 6+0 | 8.5e-4 | 0.278 ms | 0.251 ms | 0.90x |
| tp4 to_out 768 → 3072 | 1+64 | 8.5e-4 | 0.226 ms | 0.208 ms | 0.92x |
| tp4 MLP up 3072 → 3072 | 6+0 | 8.5e-4 | 0.383 ms | 0.727 ms | 1.90x |
| tp4 MLP down 3072 → 3072 | 6+0 | 8.5e-4 | 0.380 ms | 0.729 ms | 1.92x |
| tp1 q/k/v 3072 → 3072 | 6+0 | 8.5e-4 | 0.394 ms | 0.749 ms | 1.90x |
| tp1 MLP up 3072 → 12288 (chunk 6144) | 6+0 | 8.5e-4 | 0.903 ms | 2.706 ms | 3.00x |
| tp1 MLP down 12288 → 3072 (chunk 1536) | 24+0 | 8.5e-4 | 1.216 ms | 2.908 ms | 2.39x |

The small-N projections lose to the fixed cost of quantizing and packing the activation in
the graph (0.134 ms at 5120x3072). Sharing one `input_scale` tensor across q/k/v makes the
three quantize subgraphs identical and the compiler folds them: three 3072→768 projections
0.555 → 0.295 ms (bf16 0.565 ms), `prepare_fp8_kernels_` does this (commit `5cbd196`).

## 4. Device V1: `--quant fp8 --quant-compute dequant`, tp4

Quantized checkpoint written by `compile` in under a minute: 40 GB bf16 → 26 GB
(`/home/ubuntu/difflet-cache-fp8/quantized/Qwen--Qwen-Image__75e0b4be/transformer/fp8-311e308cb8465d3e`).
Compile 2 min 27 s wall, 60 blocks, load+warmup 59.1 s, 6.82 GB per rank.

Generate (`/home/ubuntu/fp8-bench/qwen_fp8_dequant_tp4/`, same prompt, seed and 50 steps as
the baseline):

| metric | value | source |
|---|---|---|
| DiT step | 344 ms = (stage 51.9 s at 50 steps − 43.3 s at 25 steps) / 25 (bf16: 336 ms the same way); the fp8 → bf16 weight conversion runs in every call | `gen50b.log`, `gen25.log` |
| warm e2e | 99–112 s (the first run's 111.9 s includes a slower load) | `gen50*.log` |
| determinism | gen50a.png == gen50b.png (md5 `231c08307bf2…`) | |
| vs bf16 baseline, latents | cosine 0.9938, rel L2 0.112 | `vs_bf16_gen50a.json` |
| vs bf16 baseline, image | **PSNR 30.17 dB, SSIM 0.977, LPIPS 0.037** | `vs_bf16_gen50a.json` |

Weight-only FP8 (bf16 activations) on the device; the picture is the same cat on the same
bench. For scale: the bf16 run at 25 steps against itself at 50 steps is 22.7 dB / LPIPS 0.18.

## 5. Static FP8 on the kernel path: unusable in the block graph (resolved by the pivot)

`--quant fp8 --quant-compute nki`, tp4: compile 6 min 41 s (the NKI kernel traced per block
graph: 720 call sites; warm-up 349.6 s, 191 s on every later launch without
`DIFFLET_NKI_TRACE_CACHE=1`), then **178 s per step** (`qwen_fp8_nki_tp4_3steps/a.log`:
step median 178,107 ms, n=2). The ranks sat at 100 % CPU; no new NEFFs were compiled.

Bisected on one core (`/home/ubuntu/.claude/jobs/238a7774/tmp/probe_*.py`):

| graph | time |
|---|---:|
| fp8 kernel alone (quantize+pack in the graph from a graph input) | 0.42–0.53 ms |
| `x * s` → quantize+pack → kernel | **984 ms** (compile 25 s) |
| `matmul(x)` → quantize+pack → kernel | 986 ms |
| kernel → gelu / residual / another kernel | 0.5–0.8 ms (fine) |
| `layer_norm(x) * s` → quantize+pack → kernel | 0.9 ms (fine) |
| any barrier (sum, fp32 round trip, mean) between producer and quantize | ~1007 ms (no help) |
| `x * s` → trivial NKI copy kernel → quantize+pack → kernel | 0.61 ms (fine) |
| trivial copy kernel fed by `x*s`, gelu, matmul | 0.26–1.9 ms (additive, fine) |

So the pathology is neuronx-cc 2.28's handling of the torch-side quantize/transpose/pack chain
when its input is a compiler-internal tensor; a kernel that takes the activation directly is
unaffected. On tp1 shapes the same chain hits `[NCC_IINL902] InferNonlocalTensors error: tuple
index out of range` (q/k/v and MLP-up quantized alone in a 1-block graph; to_out compiles but
runs 77 ms per block, MLP-down 3,185 ms per block against 16.5 ms in bf16).

Resolution: the activation quantization moves into the kernel. nkilib's `matmul_mxfp8`
(MXFP8, block-32 E8M0 scales computed in the kernel) takes the bf16 activation and a
pre-quantized weight; `difflet/quant/mx.py` reproduces its block membership and law in torch
(bit-exact against nkilib's numpy reference) so the CPU reference equals the device. One core,
5120 rows (`probe_mx_parity.py`):

| shape (K → N) | mx | bf16 | ratio | rel vs CPU MX reference |
|---|---:|---:|---:|---:|
| 3072 → 3072 | 0.393 ms | 0.780 ms | 1.98x | 1.67e-3 |
| 3072 → 768 (tp4 q/k/v) | 0.309 ms | 0.312 ms | 1.01x | 1.68e-3 |
| 768 → 3072 (tp4 to_out) | 0.220 ms | 0.243 ms | 1.11x | 1.68e-3 |
| 1024 rows, 3072 → 12288 | 0.296 ms | 0.653 ms | 2.21x | 1.68e-3 |
| 1024 rows, 12288 → 3072 | 0.321 ms | 0.721 ms | 2.24x | 1.68e-3 |

With an in-graph producer (`x * s`): 0.49 ms; `matmul → mx → gelu`: additive (2.15 ms). The
small-N projections pay the in-kernel activation quantize once per call; fusing q/k/v into one
call (N = 2304) is the next win (`probe_qkv_share.py`: one fp8 call 0.30 ms vs three bf16
0.57 ms).

## 6. mxfp8 on the device (`--quant mxfp8 --quant-compute nki`), tp4, 1024x1024

`/home/ubuntu/fp8-bench/qwen_mx_nki_tp4/`, `DIFFLET_NKI_TRACE_CACHE=1`, no calibration file.

| metric | value | source |
|---|---|---|
| quantized checkpoint | 26 GB (`quantized/…/transformer/mxfp8-545621be9e816402`), written by `compile` | `compile.log` |
| compile (cold NEFF cache) | 5 min 55 s wall; 60 blocks; load+warmup 264 s | `compile.log` |
| generate start-up (warm) | load 57–66 s, of which `eval` 43–49 s = `prepare_fp8_kernels_` building the kernel layout on the device; after commit `d5e8e4c` (layout on the host) load 36.6 s, `eval` 22.8 s (`qwen_mx_nki_tp4_fused/gen3.log`); warm-up forward 5 s | `gen*.log` |
| **DiT step median** | **267 ms** (n=49, max 271; the same on all four runs) | `gen50a.log` `request:` line |
| warm e2e (CLI wall) | 2 min 25 s at 50 steps (the start-up above dominates) | `gen50a.log` |
| determinism | gen50a.png == gen50b.png (md5 `271c9fcde17a…`) | |
| vs bf16 baseline, latents (50 steps, same seed) | cosine 0.9909, rel L2 0.135 | `vs_bf16_gen50a.json` |
| vs bf16 baseline, image | **PSNR 29.17 dB, SSIM 0.973, LPIPS 0.041** | `vs_bf16_gen50a.json` |

The bf16 step by the same `request:` timing (`qwen_bf16_tp4_timed/gen50a.log`, `gen50b.log`):
**298.8 / 299.2 ms** (n=49). So mxfp8 is **0.89x** per step, between the proposed 0.80x
"ship" and 0.90x "no-go" marks. (The 50-vs-25-step difference method overstated the bf16 step
at 336 ms because it included per-launch costs.) bf16's start-up is 16 s of load against
mxfp8's 57–66 s (the 45 s kernel-layout preparation on the device; 36.6 s after `d5e8e4c`).

### 6a. Fusing q/k/v into one kernel call: no step-time change

Commit `5c92643` concatenates each stream's three projections (N = 3 x 768 at tp4) into one
`matmul_mxfp8` call, so the in-kernel activation quantize runs once instead of three times
(`FusedFP8Projections`, built after the load; the attention processor calls
`attn.fused_qkv` / `attn.fused_add_qkv`). On the device the step median is unchanged:
**267.7 ms** at 50 steps (`qwen_mx_nki_tp4_fused/gen50a.log`, n=49, max 271; 266.9 at 3
steps) against 267.7 ms unfused; the tp1 2-block probe is 13.6 vs 13.4 ms per block. Quality
vs the bf16 baseline is the same as unfused: PSNR 29.28 dB, SSIM 0.974, LPIPS 0.039, latent
cosine 0.9912 (`vs_bf16_gen50a.json`); the image differs from the unfused run's bits (the
kernel's N tiling changes the accumulation order) and is deterministic across runs (gen50a ==
gen50b, md5 `6f79968f…`; gen50b step median 268.3 ms). The compile dropped from 5 min 55 s to 2 min 19 s (60
blocks, load+warmup 85.7 s), mostly the host-side layout. The fusion stays (it is correct and
costs nothing) but the single-layer microbenchmark's 0.30-vs-0.57 ms did not carry into the
block graph: the projections are not where the block's time goes.

### 6b. Where the block's time goes (tp1 2-block probe, `probe_block_step.py`)

The first subset runs of this probe were invalid: `--targets` was applied only on the static
branch, so every "subset" quantized all 12 layers (all read 13.4 ms per block against bf16
16.4 ms). Corrected runs (one core, 2 blocks, 10 iterations, `blockstep_mx7_*.log`):

| quantized layers (per block) | ms per block | vs bf16 16.4 |
|---|---|---|
| none (bf16) | 16.4 | — |
| MLP up only (2) | 15.8 | −0.6 |
| MLP up + down (4) | 15.2 | −1.2 |
| q/k/v + add_q/k/v (6) | 15.6 | −0.8 |
| q/k/v + to_out + to_add_out (8) | **17.7** | **+1.3** |
| all 12 | 13.4–13.6 | −2.8 to −3.0 |

The savings are not additive (MLP −1.2 and attention +1.3 do not sum to −2.8), and
quantizing the two output projections on top of q/k/v costs 2.1 ms: the compiler's schedule
for the whole block changes with each substitution, so single-layer microbenchmarks (section
5: 2.0–2.2x on the MLP shapes) do not predict the block. A bf16 block with a trivial NKI copy kernel
inserted before `to_q`, `to_out`, `to_add_out` or MLP-down stays at 16.4–16.9 ms (19.8 ms
before `to_q`), so the presence of a custom op by itself does not change the compiler's
schedule. The elementwise work at the per-rank shapes is not small: standalone compiled
`gelu(tanh)` on 5120 x 3072 bf16 is 0.34 ms, a modulation 0.16 ms, a residual add 0.16 ms, a
layer norm 0.23 ms (one core); a 3072 x 3072 bf16 matmul on the same rows is 0.74 ms.

### 6c. CPU arbiter for mxfp8 (fake-quant vs bf16, 20 steps, 1024x1024, seed 42)

`scripts/qwen_image_fp8_cpu_reference.py --format mxfp8 --tp 4`, no calibration;
`/home/ubuntu/fp8-bench/cpu_arbiter_mx/metrics.json`. The same four prompts as section 2
(all four are "held out" here: MX needs no calibration set).

| prompt | PSNR | SSIM | latent cos | static fp8 (section 2) |
|---|---|---|---|---|
| cat on a bench | 33.27 dB | 0.979 | 0.9948 | 33.99 dB |
| watercolor lighthouse | 40.12 dB | 0.992 | 0.9997 | 37.72 dB |
| brass typewriter | **28.74 dB** | 0.958 | 0.9919 | 32.34 dB |
| astronaut, neon city | **27.57 dB** | 0.948 | 0.9852 | 28.64 dB |

Gate (min PSNR >= 30 dB): **fail**, two of four below; static fp8 failed one. Reading: the
E8M0 (power-of-two) block scales are coarser than a per-tensor fp32 scale on prompts the
static calibration covers, so mxfp8 is 1–2 dB behind there, and ahead where static's single
scale saturates; neither recipe clears 30 dB on every prompt at 20 steps. The device's 50-step
29.2 dB (section 6) sits inside this distribution. A 20-step comparison penalises both: the
fake-quant and bf16 trajectories diverge further per step at the larger step size.

### 6d. Layer-set variants on the device (tp4, 1024x1024, 50 steps, seed 42, vs bf16)

`DIFFLET_QUANT_TARGETS` (commit `a39bffb`) replaces the layer set; each variant gets its own
quantized checkpoint and NEFFs. `/home/ubuntu/fp8-bench/qwen_mx_<variant>_tp4/`.

| variant (quantized layers) | step median | vs bf16 299 ms | PSNR | SSIM | LPIPS | latent cos |
|---|---|---|---|---|---|---|
| all 12 (q/k/v, add_q/k/v, to_out, to_add_out, MLP up/down x2) | 267.7 ms | 0.89x | 29.28 dB | 0.974 | 0.039 | 0.9912 |
| **noout**: all but `to_out.0` / `to_add_out` (10) | **260.2 ms** | **0.87x** | **31.59 dB** | **0.987** | **0.017** | 0.9963 |
| noout minus `add_q/k/v` (image q/k/v + MLPs, 8) | 256.6 ms | 0.86x | 30.23 dB | 0.982 | 0.036 | 0.9944 |
| MLP up/down only (4) | 273.8 ms | 0.92x | 28.96 dB | 0.976 | 0.048 | 0.9916 |

Keeping the two attention output projections in bf16 is faster *and* better: the tp1 probe
(6b) already showed quantizing them costs block time, and they sit right before the residual
add, where their error lands unattenuated. The noout variant clears all three quality gates
(PSNR >= 30, SSIM >= 0.97, LPIPS <= 0.05). Two readings from the other rows: step time is
not monotonic in the number of quantized layers (MLP-only is slower than all 12), so the
compiler's block schedule, not the kernel count, sets it; and the single-prompt quality
metrics move by about 1.5 dB between layer sets that should order the other way (8 layers
30.2 dB vs 10 layers 31.6 dB), so differences of that size are noise, not signal. **noout is
the default layer set** (commit below); `DIFFLET_QUANT_TARGETS` restores any other.

### 6d-ii. The default layer set on four prompts (tp4, 1024x1024, 50 steps, seed 42, vs bf16)

`/home/ubuntu/fp8-bench/multiprompt/` (the cat prompt is 6d's noout row). Every prompt clears
PSNR >= 30, SSIM >= 0.97, LPIPS <= 0.05; the step is 260 ms on each against bf16's 298–299.

| prompt | PSNR | SSIM | LPIPS | latent cos | mxfp8 step | bf16 step |
|---|---|---|---|---|---|---|
| a cat sitting on a bench | 31.59 dB | 0.987 | 0.017 | 0.9963 | 260.2 ms | 299 ms |
| watercolor lighthouse at dawn | 42.61 dB | 0.996 | 0.005 | 0.9997 | 260.5 ms | 299.0 ms |
| vintage brass typewriter | 31.22 dB | 0.976 | 0.035 | 0.9947 | 260.0 ms | 298.4 ms |
| astronaut, neon city at night | 33.94 dB | 0.985 | 0.033 | 0.9970 | 260.1 ms | — |

### 6e. Profiler: where the block's time goes (neuron-explorer 2.33, one logical core)

Captures of the cached one-block NEFFs (tp1 shapes; `neuron-profile` is a stub, the tool is
`neuron-explorer capture --num-exec=3 --profile-nth-exec=3`, then `view --output-format
summary-text|json`); scripts and the 2 GB JSON profiles under
`/home/ubuntu/.claude/jobs/238a7774/tmp/profile/`. Device time per block (the probe's wall time
is ~1.5 ms more in every variant: launch glue outside the NEFF):

| | bf16 | mxfp8, all 12 (fused q/k/v) | mxfp8, attention only |
|---|---|---|---|
| device time | 14.87 ms | 12.03 ms | 16.12 ms |
| TensorE busy | 13.80 ms (93%) | 8.41 ms (70%) | 12.03 ms (75%) |
| VectorE / ScalarE / GpSimd busy | 7.97 / 4.48 / 3.06 | 9.56 / 2.95 / 4.76 | 9.25 / 5.22 / 3.02 |
| SBUF write bytes; spill reloads | 11.8 GB; 1.5 GB | 36.1 GB; 12.9 GB | 20.8 GB; 5.1 GB |
| utilisation-throttled time | 2.6 ms | 6.0 ms | 6.1 ms |

The bf16 block: the 12 quantizable linears are **8.0 ms** of TensorE time (img MLP up/down
2.06 + 2.09, q/k/v 0.58 + 0.57 + 0.53, txt MLP 0.53 + 0.51, to_out 0.51, add_q/k/v ~0.17 each,
to_add_out 0.13); the attention kernel is 4.0 ms and softmax-bound on VectorE (FP8 does not
touch it); modulation linears 0.2 ms; norms, rope, GELU and residuals ~2.5 ms, mostly
overlapped with the matmuls (TensorE 91–98% busy in every phase).

The mxfp8 block: the eight `matmul_mxfp8` calls span **5.2 ms** (1.55x the bf16 dots, not the
microbenchmark's 2x) with TensorE busy only 55% of that span: the kernel is bound by GpSimd
`DMA_GATHER_TRANSPOSE` loads of the activation tiles (3.4 ms, absent in bf16) and VectorE
`QUANTIZE_MX` plus copies (2.7 ms); when `MATMUL_MX` runs it is fast (~390 TFLOPS active
against bf16's 117). The custom call is opaque to the compiler: q/k RMSNorm, rope and the
bias add that overlapped the bf16 q/k/v dots now wait for the kernel and run with TensorE
21% busy for 1.6 ms (~1.3 ms lost). Hardware utilisation throttling is active 6.0 of 12.0 ms
against 2.6 of 14.9 in bf16.

Attention-only is slower for two reasons the table shows: the same post-kernel idle window
(+1.3 ms), and the untouched bf16 img MLP compiling worse once custom calls split the graph:
4.57 → 6.1 ms (up and down no longer interleaved, GELU serialised on ScalarE with the down
projection, 11.4k ACTIVATE instructions against 6.8k, semaphore waits 1.73 vs 0.17 ms).

What would move the step further, in order: (i) let the kernel overlap its neighbours (q/k
norm + rope + bias in a kernel epilogue, or an un-fused q/k/v the compiler can pipeline);
(ii) a kernel load path without the gather-transpose (activations handed over pre-laid-out);
(iii) the throttle. None is in this change.

### 6f. Kernel load-path options (2026-11-01): 260 -> 239 ms, 0.80x

The user chose to keep chasing the 0.80x gate. One-core sweeps of nkilib `matmul_mxfp8`'s
load/layout options at the tp4 per-rank shapes, inside a compiled graph with a producer
(`probe_mx_options.py`, `/home/ubuntu/fp8-bench/mx_options*.log`; bf16 `F.linear` with the
same producer for scale):

| shape (M x K -> N) | bf16 | default | `fast_dma_transpose` | PE swizzle on x^T (`K_BY_F`) |
|---|---|---|---|---|
| 4096 x 3072 -> 2304 (fused q/k/v) | 0.564 ms | 0.569 | 0.568 | **0.400** |
| 1024 x 3072 -> 2304 (text q/k/v) | 0.203 | 0.128 | 0.111 | 0.103 (bm4 / shard-M: 0.092) |
| 4096 x 3072 -> 3072 (MLP) | 0.655 | 0.339 | **0.282** (bk6 / lm2 / shard-M 0.266) | 0.324 |
| 1024 x 3072 -> 3072 (text MLP) | 0.244 | 0.150 | **0.117** | 0.140 |

Not usable: `quant_scheme="1x32"` (wrong numbers, cosine 0.003), `disable_dma_transpose`
(unsupported for this loader), non-`mnk` loop orders (unsupported), `TILES_IN_BLOCK_M=8` at
1024 rows (LNC2 M-sharding). Numbers were identical across all usable variants.

**On the device**, `fast_dma_transpose=True` alone (commit above; `qwen_mx_fastT_tp4/`):
step median **239.1 / 239.0 ms** at tp4, 1024x1024, 50 steps = **0.80x of bf16's 299**;
PSNR 31.73 dB, SSIM 0.987, LPIPS 0.017 vs bf16; gen50a == gen50b. The PE-swizzle variant for
the q/k/v widths (N <= 2304, the activation handed over transposed) is the next run.

## 7. FLUX.1-dev in mxfp8 (`--quant mxfp8`), tp4, 1024x1024, 28 steps, seed 42

Commit `23c731d` wires FLUX (targets: the dual-stream blocks' q/k/v for both streams and both
MLPs, the single-stream blocks' q/k/v and `proj_mlp`; `to_out.0`, `to_add_out` and the single
blocks' `proj_out` stay bf16). The native backend's first FLUX run on trn3; the bf16 compile
(VAE 71 s, DiT 33 s) and both arms ran clean. `/home/ubuntu/fp8-bench/flux_tp4/`.

| | bf16 | mxfp8 (default FLUX targets, 18 per dual+single pair) |
|---|---|---|
| DiT step median (n=27) | 222.6 / 222.1 ms | 220.1 / 218.5 ms (**0.99x**) |
| compile | VAE 71 s + DiT 33 s + T5 9 s | quantized checkpoint written; DiT load 60 s |
| generate load (warm) | 15.5 s (DiT 6.7) | 25 s (DiT 16) |
| determinism | gena == genb (md5 `8b217058…`) | gena == genb (md5 `0175c118…`) |
| image vs bf16 | — | **PSNR 31.72 dB, SSIM 0.972, LPIPS 0.023** |

Quality passes the gates; the step does not move. Two differences from Qwen-Image explain
part of it: the single-stream blocks' largest matmul, `proj_out` (15360 → 3072, the fused
attention + MLP output), is kept bf16 (the module reads it as two column ranges of the
checkpoint tensor, which the quantized writer would have to slice with its block scales), so
only ~58% of a single block's linear FLOPs are quantized against ~86% of a Qwen block's; and
the profiler's finding (6e) that a bf16 matmul left between custom calls compiles worse
applies to `proj_out` in every single block. Layer-set variants (`DIFFLET_QUANT_TARGETS`,
`/home/ubuntu/fp8-bench/flux_<variant>_tp4/`, one 28-step run each, vs `bf16_gena.png`):

| quantized layers | step median | vs bf16 222 ms | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| default FLUX set (dual q/k/v + MLPs, single q/k/v + proj_mlp) | 219–220 ms | 0.99x | 31.72 dB | 0.972 | 0.023 |
| dual blocks only (their q/k/v + both MLPs) | 211.3 ms | 0.95x | 25.06 dB | 0.935 | 0.040 |
| single blocks only (q/k/v + proj_mlp) | 230.0 ms | **1.04x** | 35.49 dB | 0.980 | 0.013 |
| dual MLPs only | 216.0 ms | 0.97x | 24.74 dB | 0.931 | 0.046 |

Quantizing the single-stream blocks makes the step *slower* (230 against 222 ms): their
`proj_out` stays a bf16 dot between the q/k/v and `proj_mlp` custom calls, the configuration
the profiler showed compiling worst (6e). The dual blocks alone give 5%. The single-prompt
quality numbers swing by 6–10 dB between layer sets in no monotonic order (fewer quantized
layers, lower PSNR), so at one seed they rank nothing; the default set's 31.7 dB and the
Qwen multi-prompt table (6d-ii) are the quality evidence that counts.

**FLUX conclusion**: the wiring is correct and deterministic and the default set passes the
quality gates, but at tp4 / 1024x1024 there is no speed-up to ship (0.99x). What would change
it is quantizing the single blocks' `proj_out` (the writer slicing block scales along K for
the `proj_out_attn` / `proj_out_mlp` column ranges) so no bf16 dot is left between custom
calls — and, as for Qwen, a kernel that overlaps its neighbours. FLUX keeps its default target
set; `--quant mxfp8` on FLUX is available and documented as memory-only for now.

## Bug ledger

1. **Static-FP8 kernel path pathological inside block graphs** (section 5): compiler
   behaviour, worked around by moving the quantize into the kernel (mxfp8). A minimal repro is
   `probe_variants.py` case B; not filed with AWS (user decision).
2. **NKI kernel trace repeated per launch**: without `DIFFLET_NKI_TRACE_CACHE=1` every generate
   re-traced the kernel at 720 call sites (191 s). Fixed in `d5e8e4c`: the Qwen orchestrator
   sets it for the ranks when `--quant`'s compute is the kernel (an explicit value is kept).
3. **mxfp8 dequant mode dequantized inside the block graph**: the application prepared layers
   after the load only on the kernel path, so `--quant-compute dequant` traced the block-scale
   dequantization (a permutation gather) into every block: 6.9 s per block on one core.
   Fixed in `596cffb` (prepare every quantized layer; the dequant path dequantizes once).
4. **`prepare_fp8_kernels_` 45 s on the device**: `kernel_layout` eager on the device is a
   chain of small launches per layer. Fixed in `d5e8e4c` (layout on the host, moved once).
