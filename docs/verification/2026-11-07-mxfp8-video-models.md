# MXFP8 on the video DiTs — device evidence (Trainium3)

**Status:** LTX-2 device matrix complete (2026-11-07); CPU fp32 reference pending (section G). Host `trn3.48xlarge` (16 devices, LNC=2 is the host
default, 192 vCPU), Neuron SDK: torch_neuronx 2.14.3, nkilib 0.7.0 (`matmul_mxfp8`
experimental). Branch `feat/mxfp8-video-models` on `feat/fp8-native-trn3` (`03b43ba`).
Plan: `docs/plans/2026-11-07-mxfp8-video-models.md`. Every run: tp4, 50 steps, seed 42,
two generates per arm (determinism = identical md5). The quick quality signal is mxfp8 against
the device's bf16 clip; the final PSNR/SSIM reference is diffusers on the CPU in fp32 at the
same config, run once for every model at the end (section G).

Device runs use the `native-fp8` venv with `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin`
on PATH (`neuronx-cc`), `PYTHONPATH=<checkout> python -m difflet.cli.main ...`, and their own
compile cache `/home/ubuntu/difflet-cache-video`; outputs under `/home/ubuntu/fp8-bench/`.

## 0. The shared layer (phase A): Qwen-Image stays bit-identical

Commits `657c8b3..60a63c0` move the per-model MXFP8 wiring into the application base, one
resolver, one driver-side preparer and one HBM helper, and tighten two checks (K multiple of 128
named on the writer and in the row layer; a dead target pattern named). Device check on
Qwen-Image, tp4, 1024x1024, 50 steps, seed 42, "a cat sitting on a bench", `--quant mxfp8`
(`/home/ubuntu/fp8-bench/refactor_check/`): image md5 `7737239be780a7877cd8b2ebe32bfb2f`,
equal to the pre-refactor reference `qwen_mx_hm0_tp4/gen50a.png`; DiT step median 239.2 ms
(n=49) against 238.8 ms before; 60 compiled blocks, no fallbacks.

## 1. Kernel shape probe (one core, `fp8-bench/ltx2_probe/probe_mx_shapes.py`)

nkilib `matmul_mxfp8` through `FP8Linear(compute="nki")` inside a compiled region with a
producer, random weights, against the exact CPU MX reference (`mx.fake_quantize_mx` on the
activation, `mx.dequantize_mx` on the weight), at LTX-2's per-rank shapes at tp4:

| layer (per rank) | M x K -> N | rel L2 | cosine | result |
|---|---|---|---|---|
| attn1.to_q (video, column) | 6144 x 4096 -> 1024 | 2.33e-3 | 1.0000 | ok |
| ff.net.0.proj (video, column) | 6144 x 4096 -> 4096 | 2.33e-3 | 1.0000 | ok |
| ff.net.2 (video, row, K = 16384/4) | 6144 x 4096 -> 4096 | 2.33e-3 | 1.0000 | ok |
| audio_to_video_attn.to_q, video_to_audio_attn.to_k (video -> 2048) | 6144 x 4096 -> 512 | 2.33e-3 | 1.0000 | ok |
| attn2.to_k (text, 1024 tokens) | 1024 x 4096 -> 1024 | 2.34e-3 | 0.99997 | ok |
| attn1.to_q under CP2 | 3072 x 4096 -> 1024 | 2.33e-3 | 1.0000 | ok |
| audio_attn1.to_q, audio_ff.net.0.proj, audio_ff.net.2, audio_attn1.to_out.0 (audio, 126 tokens) | 126 x {2048, 512} -> ... | — | — | **refused**: `NCC_INKI016 ... F dim must be divisible by 8 for quantization, got F=126` |

The relative error is the same at every shape, so it is the kernel's bf16 output, not a
shape effect (section 1a). The kernel requires the activation's row count to be a multiple of
8: LTX-2's audio stream (126 tokens at 512x768x121, 51 at 480x704x49), HunyuanVideo-1.5
(49,290 image and 1,985 text tokens) and Wan under CP2 (2,340) all violate it, so `mx_linear`
pads the rows to a multiple of 8 and slices the output (plan task B7). With that padding
(`probe_mx_shapes_padded.json`, first run against a scratch copy carrying the fix, re-run
from the committed fix `100f150` with the criterion below) the audio shapes pass like the
others:

| layer (per rank) | M x K -> N | rel L2 | cosine | output |
|---|---|---|---|---|
| audio_attn1.to_q | 126 x 2048 -> 512 | 2.37e-3 | 0.999999 | [126, 512] |
| audio_ff.net.0.proj | 126 x 2048 -> 2048 | 2.36e-3 | 0.999987 | [126, 2048] |
| audio_ff.net.2 (row, K = 8192/4) | 126 x 2048 -> 2048 | 2.36e-3 | 0.999987 | [126, 2048] |

The default LTX-2 target set (video and text streams) never needed the pad.

The probe's `pass` flag is rel L2 < 3e-3 and cosine > 0.9999 at the output's expected shape
(the two bf16 roundings of section 1a measure 2.3e-3). The first two runs
(`probe_mx_shapes.json`, `probe_mx_shapes_padded.json` of 08:02) were judged at the plan's
provisional rel L2 < 2e-3 and so record `"pass": false` at the values above; the criterion
was widened after section 1a explained the 2.3e-3, and `probe_mx_shapes_padded.json` was
re-run under it (the video-shape JSON keeps its pre-fix refusal rows as the evidence for them).

**LNC2 M-sharding (`probe_mx_shapes_lnc2.json`, the B7 review):** with a pre-quantized weight
the kernel splits the activation's rows across the two physical cores of a logical core
(`resolve_lnc2_sharding` -> SHARD_M for M > 128, `M // 2` rows each) and asserts the
multiple-of-8 rule on the full M only, so a padded count whose half is not a multiple of 8
was unverified: every per-core count the device had seen (512, 1536, 2048, 3072) was a
multiple of 128, no partial last tile either. The same probe on a free device (core 2) at the
HunyuanVideo-1.5 text and Wan+CP2 counts through `mx_linear`'s pad, plus a per-core count that
is a multiple of 8 with a partial tile:

| M (per core) | K -> N | rel L2 | cosine | output |
|---|---|---|---|---|
| 1040 (520: 4 tiles + 8 rows) | 4096 -> 1024 | 2.34e-3 | 0.999973 | [1040, 1024] |
| 1985 -> 1992 (996: 7 tiles + 100 rows) | 4096 -> 1024 | 2.33e-3 | 0.999976 | [1985, 1024] |
| 2340 -> 2344 (1172: 9 tiles + 20 rows) | 4096 -> 1024 | 2.33e-3 | 0.999980 | [2340, 1024] |

Same error as every other shape, correct shapes: the per-core half needs no alignment, so
`M_MULTIPLE` stays 8 (16 would have padded the same models to 2,000 and 2,352 for nothing).

### 1a. Output rounding (`probe_rounding.py`, 1024 x 4096 -> 1024)

The device output against the fp32 CPU MX reference rounded four ways:

| reference | rel L2 | elements equal |
|---|---|---|
| fp32, no rounding, + bias | 2.34e-3 | 0.003% |
| **bf16 round-to-nearest matmul, bias added in bf16** | **8.5e-4** | **90.3%** |
| bf16 truncated matmul, bias in bf16 | 4.1e-3 | 49% |
| bf16 RNE matmul, bias added in fp32 | 1.8e-3 | 1% |

The kernel rounds its fp32 accumulator to bf16 to nearest, and `mx_linear` adds the bias in
bf16 after it; the remaining 10% of elements differ by one bf16 ulp (fp32 accumulation order
before the rounding). The 2.33e-3 of section 1 is these two bf16 roundings. Nothing to fix.

## 2. LTX-2 bf16 on trn3 (first native run), tp4, 512x768x121, 50 steps, seed 42

`/home/ubuntu/fp8-bench/ltx2_tp4/`, prompt "a cat walking through a garden", Gemma-3 on the
device (the driver's default; weight store hit, 5.88 GB per rank), host decode.

| | value |
|---|---|
| compile (`difflet compile`) | 96 compiled blocks, no fallbacks, load + warm-up 302.6 s |
| generate stage (two runs) | 132.6 s / 128.2 s, of which load + warm-up 56.7 / 55.3 s |
| determinism | mp4 md5 `0702ffd2bfaa79f922dbf5d33f4dcd6b` and latents md5 `2805c388b4ec6172989be1e0440c3e33` on both runs |
| DiT step median | **1283.9 ms** (n=49, max 1315.8; request 71.9 s, 8.9 s to the first step's end incl. the prompt encode), from a third run with the stage's timing hook (`bf16_gen50c_timed2.log`, same mp4 md5) |

Getting the step median exposed a bug (task B8): the stage's timing hook replaced
`scheduler.step` with a plain closure over the video scheduler's bound method;
`LTX2Orchestrator` deep-copies that scheduler into its audio scheduler, `copy` keeps functions
by reference, so the audio step advanced the video scheduler's index too and the run died at
`sigmas[51]`. A `types.MethodType` wrapper that marks only the original instance fixes it
(`copy` rebinds methods to the copy); verified on a scratch copy before the commit.

Note: a checkout at another path recompiles every block (warm-up 214 s against 55 s from the
cache): the NEFF cache key depends on the source path. Qwen-Image's md5 matched across two
paths, so it costs a compile per checkout, nothing else.
| frames | 121 x 512 x 768, a cat on a garden path (`bf16_frame_{1,2,3}.png`) |

## 3. LTX-2 mxfp8 (`--quant mxfp8`, default target set), tp4, 512x768x121, 50 steps, seed 42

Same prompt, cache and host as section 2; the run from a snapshot of `fc8b94c` carrying the
B7/B8 patches (`/home/ubuntu/.claude/jobs/4e438750/tmp/scratch-fc8b94c`, identical to the
later commits), `mx_compile.log`, `mx_gen50{a,b}.log`.

| | bf16 | mxfp8 (video + text linears; audio stream and output projections bf16) |
|---|---|---|
| quantized DiT directory | — | 24 GB (`quantized/Lightricks--LTX-2__64f65d09/transformer/mxfp8-825b78f48b70b6c8`), bf16 source 37.8 GB |
| compile | 96 blocks, 302.6 s | 96 blocks, 138.9 s load + warm-up, no fallbacks |
| generate load + warm-up | 55–57 s | 61.5 s |
| request (encode + 50 steps) | 71.9 s | 56.2 s |
| **DiT step median** | **1283.9 ms** (n=49, max 1315.8) | **966.4 / 968.2 ms** (n=49, max 991.9 / 970.4) = **0.75x** |
| determinism | mp4 `0702ffd2…`, latents `2805c388…` on both runs | mp4 `c1057ed5…`, latents `80316ff7…` on both runs |
| video latents vs bf16 | — | cosine 0.9909, rel L2 0.134 |
| audio latents vs bf16 (not quantized) | — | cosine 0.9998, rel L2 0.019 |
| clip vs bf16 (121 frames, per-frame mean) | — | PSNR 22.47 dB (worst frame 20.85, #92), SSIM 0.830, LPIPS 0.078 |

The two clips show the same scene and the same cat (`bf16_frame_2.png`, `mx_frame_2.png`,
frame 60); the difference is in the fine foliage texture and, on the worst frame (#92,
`frame92_bf16_vs_mx.png`, mean |diff| 16/255, p99 77), the cat's pose by a few pixels: a
different sample along a diverged trajectory, which 50 steps amplify from any small
perturbation. Whether 22 dB is fp8 damage or the video's own sensitivity is
answered by the bf16-vs-bf16 noise floor (section 3a) and, finally, by the CPU fp32 reference
(section G).

### 3a. bf16 noise floor: tp2 against tp4

A bf16 run at tp2 (`bf16_tp2_gen50a.log`; step 2379.4 ms on two cores) differs from the tp4
run only in the TP all-reduce order. Against the tp4 bf16 clip:

| | bf16 tp2 vs bf16 tp4 (noise floor) | mxfp8 tp4 vs bf16 tp4 |
|---|---|---|
| video latents | cosine 0.9975, rel L2 0.071 | cosine 0.9909, rel L2 0.134 |
| audio latents | cosine 0.9999, rel L2 0.014 | cosine 0.9998, rel L2 0.019 |
| clip PSNR (mean / worst frame) | 25.9 dB / 23.6 | 22.5 dB / 20.8 |
| SSIM / LPIPS | 0.909 / 0.040 | 0.830 / 0.078 |

mxfp8 sits 3.5 dB below the bf16 noise floor at twice its latent distance: distinguishable from
reduction-order noise, by about the margin Qwen-Image showed (32–38 dB against a 39–46 dB floor).
The CPU fp32 reference (section G) is the gate.

### 3b. Audio-stream variant (`DIFFLET_QUANT_TARGETS` = the default six patterns + the six audio ones)

`mx_audio_compile.log`, `mx_audio_gen50a.log`; quantized directory `mxfp8-8121daf377ecfe89`
(12 patterns, every linear of the block except the output projections; the audio rows are
padded 126 -> 128 by B7). One run each:

| | default set (video + text) | + audio stream |
|---|---|---|
| DiT step median | 966.4 ms | 972.3 ms (max 1270.5) |
| clip vs bf16 | 22.47 dB / 0.830 / 0.078 | 22.70 dB / 0.835 / 0.076 |
| video latents vs bf16 | cosine 0.9909 | cosine 0.9918 |
| audio latents vs bf16 | cosine 0.9998, rel 0.019 | cosine 0.9997, rel 0.030 |

No speed to gain (the audio stream is 126 of 7,294 tokens), the video quality is the same
within the single-seed noise, and the audio latents' own error doubles. The default stays
video + text; the audio linears remain an override, now runnable.

### 3c. Context parallelism: tp4 x CP2 (gather_kv), mxfp8, default set

`mx_cp2_compile.log` (8 ranks, 96 blocks, 106.6 s warm-up, no fallbacks), `mx_cp2_gen50a.log`:
DiT step median **625.4 ms** (n=49, max 626.9) on eight cores against 966.4 ms on four;
request 42.5 s. The clip (`02eaadeb…`) against the tp4 clips:

| CP2 mxfp8 vs | PSNR (mean / worst) | SSIM | LPIPS |
|---|---|---|---|
| tp4 bf16 | 23.4 dB / 21.3 | 0.855 | 0.068 |
| tp4 mxfp8 | 24.7 dB / 22.2 | 0.879 | 0.052 |

CP2 moves the mxfp8 output by about the bf16 reduction-order floor (section 3a): the
quantized layers behave under gather_kv as the bf16 ones do.

### 3d. Data parallelism: `--dp 2` (two tp4 replicas), mxfp8, two requests

`mx_dp2.log`, `dp2_requests.jsonl` (the cat prompt at seed 42 and a second prompt). The
router placed worker 0 on cores 0-3 and worker 1 on cores 4-7; both workers printed
`[ltx_2] MXFP8 transformer (nki): .../mxfp8-825b78f48b70b6c8` (the `--quant` flags reach the
workers, `difflet/cli/dp/router.py`), both ran at the single-launch speed (step medians 968.9
and 970.3 ms, no interference), and worker 0's clip is bit-identical to the single-launch
mxfp8 clip (`dp2_cat.mp4` md5 `c1057ed5…` = `mx_gen50a.mp4`).

### 3e. Summary for LTX-2

| layout | DiT step | vs bf16 tp4 clip |
|---|---|---|
| bf16 tp4 | 1283.9 ms | — (noise floor tp2: 25.9 dB / 0.909 / 0.040) |
| **mxfp8 tp4 (default set)** | **966.4 ms = 0.75x** | 22.5 dB / 0.830 / 0.078 |
| mxfp8 tp4 + audio stream | 972.3 ms | 22.7 dB / 0.835 / 0.076 |
| mxfp8 tp4 x CP2 | 625.4 ms | 23.4 dB / 0.855 / 0.068 |
| mxfp8 dp2 (each replica) | 968.9 / 970.3 ms | bit-identical to tp4 mxfp8 |

Speed passes the "recommended" line (<= 0.90x) by a wide margin; quality is judged against the
CPU fp32 reference in section G.
