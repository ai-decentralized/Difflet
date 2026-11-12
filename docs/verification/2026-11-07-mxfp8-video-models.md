# MXFP8 on the video DiTs — device evidence (Trainium3)

**Status:** LTX-2, Wan 2.1, Wan 2.2-A14B and FLUX.1-dev (revisited) complete (2026-11-07); HunyuanVideo (section 7) and phase H (section 8) complete (2026-11-08): Wan's self-attention q/k/v run in the half-split rotary layout only under an override, since with both Wan 2.2 experts they failed the default rule (8i), so Wan's default is the cross-attention q/k/v and the FFN again (`90c648f`, 0.89x at 480x832x9 on the final tree), and bf16 Wan is bit-reproducible again (H5, `c7c2346`); HunyuanVideo-1.5 (section 9) complete (2026-11-08), its default the user's decision (taken at 1.007x and the same peak HBM, with phase D's D4 rotary; F7 reverted D4 (`dd3eed2`), and on that tree the step is 1.011x, 9b; issue #145); the CPU fp32 reference (phase G, section 10) complete (2026-11-08): MXFP8 minus bf16 in PSNR against diffusers fp32, Qwen-Image +1.0 dB, LTX-2 +0.3, HunyuanVideo -0.2, FLUX -2.1 (one image), Wan 2.1 -2.4, Wan 2.2 -2.9, HunyuanVideo-1.5 -7.5, which therefore has no MXFP8 default (the user's decision, 2026-11-09: `--quant mxfp8` needs `DIFFLET_QUANT_TARGETS`, 10e). The official-settings rerun (section 12, 2026-11-09, a second trn3): Wan 2.1 and Wan 2.2 at 480x832x81 with CFG 0.98x DiT step, HunyuanVideo-1.5 at 480x848x121 1.01x; Wan's faster compile and cold start are the NKI trace cache `--quant` turns on (a Wan 2.1 bf16 control with it: MXFP8 1.21x compile, 1.01x cold); against bf16 at another TP degree Wan 2.1 and HunyuanVideo-1.5 fail the device screen, Wan 2.2 passes. Host `trn3.48xlarge` (16 devices, LNC=2 is the host
default, 192 vCPU), Neuron SDK: torch_neuronx 2.14.3, nkilib 0.7.0 (`matmul_mxfp8`
experimental). Branch `feat/mxfp8-video-models` on `feat/fp8-native-trn3` (`03b43ba`).
Plan: `docs/plans/2026-11-07-mxfp8-video-models.md`. Every device run: 50 steps, seed 42, and,
unless noted, tp4 and two generates per arm (determinism = identical md5); HunyuanVideo-1.5
(section 9) runs tp2 x CFG2. The quick quality signal is mxfp8 against
the device's bf16 clip; the final PSNR/SSIM reference is diffusers on the CPU in fp32 at the
same config, run once for every model at the end (section 10).

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
(section 10: at 121 frames both clips are far from fp32, bf16 16.55 dB and mxfp8 16.85, mxfp8
the closer).

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
The CPU fp32 reference (section 10) puts both clips far from fp32 at this length (bf16 16.55 dB,
mxfp8 16.85): mxfp8 is no further from it than bf16.

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

Speed passes the "recommended" line (<= 0.90x) by a wide margin; against the CPU fp32 reference
(section 10) mxfp8 is no worse than bf16 (16.85 against 16.55 dB, both far from fp32 at 121
frames).
## 4. Wan 2.1-T2V-14B on trn3, tp4, 480x832, 50 steps, seed 42

`/home/ubuntu/fp8-bench/wan21_tp4/`, cache `/home/ubuntu/difflet-cache-video`, prompt "a cat
walking through a garden", umT5 on the device (weight store 2.84 GB per rank), device VAE. The
hub DiT is fp32 (57.2 GB); `difflet compile` makes the bf16 copy (28 GB, 11.8 s on this host)
and every arm loads it.

### 4a. bf16

| | 9 frames (4,680 tokens) | 81 frames (32,760 tokens) |
|---|---|---|
| compile | 64 compiled blocks, no fallbacks, load + warm-up 249.4 s | 64 compiled blocks, no fallbacks, load + warm-up 781.3 s |
| generate (warm) | load + warm-up 40.8 s, stage 73.6 s, request 32.8 s | load + warm-up 447.0 s (the warm-up forward 420.4 s), stage 713.8 s, request 266.9 s |
| DiT step median | **437.6 ms** (n=49, max 441.9) | **5065.0 ms** (n=49, max 5080.6; request 266.9 s) |
| determinism | mp4 `450d04ef…` and latents `bf61d6d7…` on both runs (437.6 ms both) | mp4 `efe453e2…` (one run) |

The 81-frame step is 11.6x the 9-frame one for 7x the tokens: at the realistic shape the
block is attention-bound (the plan's estimate: linear share 85% at 9 frames, 47% at 81), so the
MXFP8 linears can move the 81-frame step by far less than the 9-frame one.

**bf16 noise floor at 9 frames** (`bf16_tp2_f9_gen50a.log`, tp2 bf16 at 772.2 ms/step
against the tp4 clip, only the all-reduce order differs): latents cosine 0.99990 / rel L2
0.014, clip PSNR 37.0 dB (worst frame 36.4), SSIM 0.960, LPIPS 0.020 — a far tighter floor
than LTX-2's 25.9 dB: nine frames of Wan barely diverge, so the mxfp8 gap is read against 37 dB.

### 4b. mxfp8 (`--quant mxfp8`, default set: attention q/k/v and FFN; `to_out` bf16)

The driver quantizes from the bf16 copy (`[wan] bf16 copy of transformer: ... (0.0 s)` — already
made — then `[wan] MXFP8 transformer (nki): .../quantized/Wan-AI--Wan2.1-T2V-14B-Diffusers__38ec498c/transformer/mxfp8-21684cd3a2b409c4`):
the quantized DiT is **16 GB** for 28 GB of bf16 (57 GB of fp32 on the hub); each rank's DiT
bytes 4.60 GB instead of 7.50; compile 64 blocks, no fallbacks, load + warm-up 115.5 s.

| | 9 frames | 81 frames |
|---|---|---|
| DiT step median | **1271.9 ms — 2.9x slower than bf16** (first run) | — |
| vs bf16 | **wrong**: PSNR 20.2 dB, SSIM 0.724, LPIPS 0.285; latents cosine 0.917, rel L2 0.41; the cat is a smeared blob (`f9_frame4_bf16_vs_mx.png`) | — |

Not fp8 noise: a correctness and speed failure of the kernel path at Wan's shapes, which are
the first with a row-parallel K per rank that is not a multiple of 512 (`ffn.net_out`:
13824/4 = 3456 = 6 x 512 + 384) and an M that is not a multiple of 128 (4680). Diagnosed in
4c with a one-core probe at exactly these shapes.

### 4c. Kernel probe at Wan's per-rank shapes (`probe_wan_shapes.py`, one core)

Each shape through `FP8Linear(compute="nki")` in a compiled region against the exact CPU MX
reference, and timed (20 calls, one sync) against a compiled bf16 `nn.Linear`:

| layer (per rank, tp4) | M x K -> N | rel L2 | cosine | MX | bf16 | ratio |
|---|---|---|---|---|---|---|
| attn1.to_q (column) | 4680 x 5120 -> 1280 | 2.33e-3 | 1.0000 | 0.358 ms | 0.660 ms | 0.54 |
| ffn.net_in (column) | 4680 x 5120 -> 3456 | 2.32e-3 | 1.0000 | 0.895 | 1.498 | 0.60 |
| **ffn.net_out (row, K = 13824/4 = 3456)** | 4680 x 3456 -> 5120 | **0.408** | **0.919** | 1.382 | 1.907 | 0.72 |
| attn2.to_k (text) | 512 x 5120 -> 1280 | 2.32e-3 | 1.0000 | 0.188 | 0.193 | 0.97 |
| attn1.to_q, M padded to 4736 | 4736 x 5120 -> 1280 | 2.33e-3 | 1.0000 | 0.379 | 0.671 | 0.56 |
| **net_out with K padded to 3584** | 4680 x 3584 -> 5120 | 2.33e-3 | 1.0000 | 1.466 | 2.051 | 0.71 |
| LTX-2 attn1.to_q (control) | 6144 x 4096 -> 1024 | 2.33e-3 | 1.0000 | 0.279 | 0.563 | 0.50 |
| LTX-2 ff.net.2 (control) | 6144 x 4096 -> 4096 | 2.33e-3 | 1.0000 | 1.303 | 2.322 | 0.56 |

**The quality bug is the K remainder 384.** nkilib's own golden (`matmul_mxfp8_torch._swizzle`)
says the legacy DGT loader decomposes a K remainder into 256 + 128 sub-tiles while the
fast-DMA loader (`fast_dma_transpose=True`, what difflet calls) takes the whole remainder as one
partial tile, and "the two layouts agree for every 32-aligned remainder except 384 … producing
a ~100% relative error". `difflet/quant/mx._tile_sizes` implemented the legacy decomposition;
Qwen-Image, FLUX and LTX-2 never had a remainder (K per rank 3072, 4096, 1024, 512, 2048).
Wan's `ffn.net_out` is 13824/4 = 6 x 512 + 384. Fixed by making `_tile_sizes` the fast-DMA law
(section 4d) and bumping the quantized-checkpoint schema.

**The 2.9x step is not the kernels' own time**: at Wan's shapes every MX layer is faster than
its bf16 twin (0.54–0.72x per layer; an M of 4680 costs nothing against a padded 4736), so
the loss sits in how the compiled block schedules around the custom calls. Re-measured after
the layout fix in 4d.

### 4d. After the layout fix (`c813a47`, schema 3): quality right, step unchanged

`mx_f9_compile_v3.log` (new directory `mxfp8-e17b42dc8e5dc336`, 64 blocks, 102 s warm-up),
`mx3_f9_gen50a.log`:

| | bf16 | mxfp8, schema 2 (wrong layout) | **mxfp8, schema 3** |
|---|---|---|---|
| DiT step median | 437.6 ms | 1271.9 ms | **1271.9 ms (2.9x)** |
| latents vs bf16 | floor: cos 0.9999 / rel 0.014 | cos 0.917 / rel 0.41 | **cos 0.9984 / rel 0.057** |
| clip vs bf16 (PSNR mean / worst, SSIM, LPIPS) | floor: 37.0 / 36.4, 0.960, 0.020 | 20.2 / 18.2, 0.724, 0.285 | **34.0 / 32.5, 0.949, 0.031** |

The layout fix restores the numbers to the usual distance below the bf16 floor (3 dB, as
Qwen-Image and LTX-2); the cat is a cat again (`f9_frame4_bf16_vs_mx3.png`). The step time
did not move, as the per-layer probe predicted: it is not the kernels. Bisected in 4e.

### 4e. Step-time bisection by target set (`DIFFLET_QUANT_TARGETS`)

One compile + generate per set at 480x832x9, tp4 (`mx_attn_f9_*`, `mx_ffn_f9_*`):

| quantized layers | DiT step median | vs bf16 437.6 ms |
|---|---|---|
| default set (attn1/attn2 q/k/v + FFN up/down) | 1271.9 ms | 2.91x |
| attention q/k/v only (attn1 + attn2) | 1322.1 ms | 3.02x |
| **FFN only (`net_in`, `net_out`)** | **407.8 ms** | **0.93x** |

The FFN linears alone give a real gain (the row layer with the K remainder included); the
attention q/k/v custom calls cost ~22 ms per block, while the same kernels in isolation run
in 0.36 ms (section 4c). The difference from Qwen-Image and LTX-2, where the same q/k/v
layers gain: Wan's token count, 4680 = 36 x 128 + 72, is not a multiple of 128 (Qwen 4096,
LTX-2 6144 are), so the transposes that bring q/k/v into the attention kernel's layout, which
the compiler folds into its own matmul's output for bf16, have to be done on a custom-op
output of unaligned height. Test: `M_MULTIPLE = 128` (the kernel's rows padded to 128 and
sliced back, giving the compiler an aligned, freshly written tensor) with the attention-only
set — section 4f.

### 4f. Rows padded to 128 (attention-only set): no change

`M_MULTIPLE = 128` on a scratch copy, attention-only set: 1329.8 ms (against 1322.1 with the
8-row pad). The kernel's output alignment is not it. Finer bisection in 4g (attn1 alone, attn2
alone, attn1.to_q alone).

### 4g. Finer bisection

| quantized layers | DiT step median | vs bf16 |
|---|---|---|
| attn2 (cross-attention) q/k/v only | 425.1 ms | 0.97x |
| attn1 (self-attention) q/k/v only | 1353.7 ms | 3.09x |
| attn1 `to_q` only | **2409.0 ms** | **5.50x** |

Self-attention's projections carry the whole cost, cross-attention's none, and one quantized
`to_q` per block costs more than all three. What self-attention's q and k go through that
cross-attention's do not is RoPE (`modeling_wan.py`: `apply_rotary_emb` only when
`not self.is_cross_attention`); the global qk-norm is applied in both. The one-block tp1 probe
(4h) isolates it with `to_k` / `to_v` alone and a RoPE-as-identity timing variant.

### 4h. One block on one core (`probe_wan_block.py`, tp1, num_layers=1)

A 1-block checkpoint carved from the bf16 copy (block 0, the embedders, the head), difflet's
own transformer application at tp1 with the example inputs (480x832x9), the forward timed
over 10 calls (the whole 1-block model, so embedders and head included):

| variant (old RoPE op) | forward | vs bf16 |
|---|---|---|
| bf16 | 133.9 ms | — |
| mx FFN (`net_in`, `net_out`) | 130.9 ms | 0.98x |
| mx attn1 q/k/v | 138.0 ms | 1.03x |
| mx attn2 q/k/v | 141.9 ms | 1.06x |
| **mx attn1 `to_q` alone** | **240.0 ms** | **1.79x** |

The lone-`to_q` pathology reproduces without TP; quantizing all three of attn1's projections
does not (at tp4 it did, 3x), so two effects are in play: one around a single custom-op
output feeding RoPE, and one that needs TP (the global qk-norm all-reduces its sum of squares
right after `to_q` / `to_k` at tp > 1).

**Superseded by section 8:** the tp4 block profile puts the 5x in the rotary op's element-wise
DMAs on q/k from the MXFP8 kernel, not in a TP collective.

**The RoPE formulation.** Wan's rotary op (`difflet/backends/cpu/ops_impl/embeddings.py`,
re-used by the neuron backend) is the interleaved-pair one: `unflatten(-1, (-1, 2)).unbind(-1)`
and `out[..., 0::2] = ...`, `out[..., 1::2] = ...` — stride-2 slices of the activation and
strided in-place writes. For a bf16 matmul the compiler lays the output out for that access;
for an opaque custom op it cannot. LTX-2's RoPE takes contiguous halves (the "split" type),
which is why LTX-2 never showed this; FLUX and HunyuanVideo use the interleaved form. The
neuron backend now has its own `apply_rotary_emb`: the pair swap is a matmul with a constant
permutation matrix (`x @ P`), the rotation two products and one add per element —
bit-identical to the CPU op (`tests/unit/backends/test_neuron_rotary.py`), and the Wan
compile-mode test still traces one graph per block class.

| variant (new RoPE op) | forward | vs new bf16 | vs old-op bf16 |
|---|---|---|---|
| **bf16** | **49.8 ms** | — | **0.37x** |
| mx attn1+attn2 q/k/v | 38.2 ms | 0.77x | 0.29x |
| mx FFN | 42.5 ms | 0.85x | |
| mx attn1 q/k/v | 40.1 ms | 0.81x | |
| mx attn1 `to_q` alone | 45.8 ms | 0.92x | (was 1.79x) |
| mx attn1 `to_k` alone | 45.9 ms | 0.92x | |
| mx attn1 `to_v` alone | 48.8 ms | 0.98x | |
| mx attn2 q/k/v | 48.3 ms | 0.97x | |

The strided RoPE was costing the bf16 block itself 2.7x on this compiler; with the pair swap
as a matmul every quantized subset is faster than bf16 and the lone-`to_q` pathology is gone.
The bf16 Wan reference is therefore re-measured with the new op at tp4 (4i): the op is
bit-identical, so the clip must keep its md5 while the step drops.

Who else: HunyuanVideo calls the same backend op (`difflet.ops.apply_rotary_emb`), so its
bf16 and mxfp8 steps change with this commit; FLUX's TP attention (`models/flux/tp_sharding.py`)
and HunyuanVideo-1.5 call diffusers' own `apply_rotary_emb`, the same interleaved
`reshape(..., 2).unbind(-1)` / `stack(...).flatten(3)` pattern — the likely reason FLUX's
mxfp8 gained nothing at tp4 (0.99x) beyond its unquantized `proj_out`; a neuron-friendly
rotary at those two call sites is now part of phase D.

### 4i. tp4 with the new rotary op: the one-block probe does not predict tp4

`rope_bf16_f9_*`, `rope_mx_f9_*` (scratch copy = `c813a47` + the rotary op, later `fcb0dca`):

| | old op | new op |
|---|---|---|
| bf16 step | 437.6 ms | 431.8 ms |
| bf16 clip md5 | `450d04ef…` | `d1d89955…` — the device compiler fuses multiply-adds per graph, so bit-identity holds on the CPU, not across NEFFs; the two bf16 clips are 36.3 dB / SSIM 0.953 / latents cosine 0.99993 apart, inside the tp2-vs-tp4 floor |
| mxfp8 step (default set) | 1271.9 ms | **2262.9 / 2264.7 ms** (deterministic, `76ed6e4e…`) |
| mxfp8 clip vs its bf16 | 34.0 dB / 0.949 / 0.031, cos 0.9984 | 32.3 dB / 0.938 / 0.042, cos 0.9973 |

bf16 moves by 1% (the strided RoPE was never the bf16 bottleneck at tp4; the one-block tp1
forward includes the embedders and head at full width, which is where its 2.7x came from),
and mxfp8 is slower still. The one-core probe is therefore not a model of the tp4 block, and
what remains is TP-specific: the collectives that follow the custom calls (the qk-norm's
all-reduce of the sum of squares, the scatter of its weight). A device profile of the real
tp4 run (`neuron-explorer inspect`) is the next instrument (4k).

**Superseded by section 8:** not in the collectives: the replayed tp4 block spends the time in
the rotary op's 2-byte DMAs on the kernel's q/k (section 8b).

### 4k. System profile of the real tp4 run (`neuron-explorer inspect`, 2 steps, rank 0)

`inspect_{mx,bf16}/`, exported with `neuron-explorer view --output-format json` and aggregated
per NEFF (`analyze_profile.py`). Each run has 665 executions of 60 NEFFs on core 0; the one
that matters is the Wan block (240 executions: 40 blocks x warm-up + steps, CFG batched):

| | bf16 | mxfp8 (default set, new rotary op) |
|---|---|---|
| block NEFF, device time per execution (median) | **10.50 ms** | **56.27 ms** (5.4x) |
| all collectives (`cc_running`, 2240 each) | 126.7 ms | 163.8 ms |
| executions of every other NEFF | identical | identical |

40 blocks x 56.3 ms = 2.25 s, the measured step. The time is inside the compiled block's own
device code: no host gaps, no collective stalls. What the block's engines do is the device
profile in 4l.

**Superseded by section 8:** the engine-level profile (8b) has 49.31 ms of ACT and 51.67 ms of
hardware DMA in the 56.89 ms block, in the rotary op.

### 4l. Device profile of the mxfp8 block NEFF: not obtained

`NEURON_RT_INSPECT_DEVICE_PROFILE=1` under `inspect` (1 step, every rank; with
`NEURON_RT_INSPECT_EVENT_FILTER_NC=0` the other ranks refuse "NC 0 not visible") records a
`device_profile_model_494866922837647.json` reference on each block execution but writes no
device profile; `inspect` does dump every NEFF it ran (the block is the 5.3 MB
`neff_494866922837647_vnc_0.neff` under `devprof_mx/`). An engine-level view would mean
replaying that tp4 NEFF, which holds the qk-norm and row-parallel collectives, on four cores
with `neuron-explorer capture`'s multi-worker options: left as the next instrument for the
open item below, not done here.

**Superseded by section 8:** obtained in phase H by exactly that replay (recipe in 8a, tables in
8b).

### 4m. What ships for Wan: the layers that gain at tp4

The measurements say which projections MXFP8 can take today on Wan at tp4: the FFN (0.93x)
and cross-attention's q/k/v (0.97x) gain; self-attention's q/k/v cost the block 5x in device
time, a compiler effect around those custom-op outputs that the one-core probe does not
reproduce (its block runs faster quantized). So Wan's default target set is FFN +
cross-attention q/k/v, measured as a set below; self-attention q/k/v stay bf16 and are the
open item (the device profile of 4l, or a kernel that writes q/k in the layout the rotary and
attention kernels read).

**Superseded by section 8:** the cause is the rotary's element-wise DMAs on the kernel's q/k; in
the half-split layout H2 (`f0ad1c6`) made the self-attention q/k/v part of the default again
(0.77x at 9 frames, 0.94x at 81), and H6 (`90c648f`) took them out again after the two-expert
Wan 2.2 measurement failed the rule (8i, 8j): this section's set is the default once more, and
the half-split layout serves overrides.

`DIFFLET_QUANT_TARGETS` = the three patterns, then committed as the default (`917d087`,
same pattern order, so the same quantized directory `mxfp8-6d28df7cc78386b5`, 19 GB, and the
same NEFFs); `mx_ffnx_f9_*`, against the bf16 run with the same rotary op (`rope_bf16_f9_*`):

| | bf16 | **mxfp8, default set** |
|---|---|---|
| DiT step median | 431.8 ms | **384.8 / 385.1 ms = 0.89x** |
| determinism | | mp4 `fd29d953…` on both runs |
| per-rank DiT weights | 7.50 GB | 5.36 GB |
| latents vs bf16 | (floor: cos 0.9999) | cos 0.9977, rel L2 0.068 |
| clip vs bf16 | (floor: 37.0 dB / 0.960 / 0.020) | 33.0 dB (worst 31.5) / 0.944 / 0.036 |

**CP2 (gather_kv), default set** (`c6_cp2_f9_*`, 8 ranks, 2340 rows per rank: `mx_linear`'s
8-row padding on the device): 64 blocks, no fallbacks; DiT step **251.1 ms** on eight cores;
the clip is 35.6 dB / 0.950 / 0.029 from the tp4 mxfp8 clip (inside the 37 dB reduction-order
floor) and 32.0 dB / 0.936 / 0.044 from tp4 bf16.

**81 frames (32,760 tokens), tp4, default set** (`c6_bf16_f81_*`, `c6_mx_f81_*`, both with the
new rotary op; the mxfp8 compile took 732 s of load + warm-up):

| | bf16 | mxfp8 |
|---|---|---|
| DiT step median | 5051.3 ms | 4955.1 ms = **0.98x** |
| request (encode + 50 steps) | 253.9 s | 249.1 s |
| latents vs bf16 | | cos 0.978, rel L2 0.208 |
| clip vs bf16 (81 frames) | floor (tp2 vs tp4 bf16, `c6_bf16_tp2_f81_*`, 11.7 s/step on two cores): 37.2 dB / 0.952 / 0.026, latents cos 0.9988 | 28.8 dB (worst 27.8) / 0.853 / 0.096 |

At the realistic shape the block is attention-bound (the plan's estimate: linear share 47%
at 81 frames against 85% at 9, and the default set leaves the self-attention projections in
bf16), so the MXFP8 linears move the step by 2%: by the plan's rule ("recommended" is judged at
the long shape), Wan 2.1's `--quant mxfp8` is **memory-only** at 480x832x81 (per-rank DiT
weights 5.36 GB against 7.50) and 0.89x only for short clips.

The 81-frame clip sits 8.4 dB under the bf16 floor (4 dB at 9 frames): the gap grows with the
clip. Frames 10/40/75 side by side (`f81_bf16_bf16tp2_mx.png`: bf16 tp4, bf16 tp2, mxfp8) show
the same scene, cats and jar in all three, clean, with the mxfp8 camera framing drifted by a
few pixels: a different sample of the same trajectory family, which PSNR scores as error. The
CPU fp32 reference (section 10) ran 9 frames only (there the default is 2.36 dB further from fp32
than bf16), so this 81-frame clip has none.

**DP2** (`c6_dp2.log`, `--dp 2`, two requests at 9 frames): the router put the workers on
cores 0-3 and 4-7, both printed `[wan] MXFP8 transformer (nki): .../mxfp8-6d28df7cc78386b5`,
both ran 385.7 / 385.8 ms per step side by side, and worker 0's clip is bit-identical to the
single launch (`fd29d953…`).

### 4n. Summary for Wan 2.1 (tp4, 50 steps, seed 42)

| | 9 frames | 81 frames |
|---|---|---|
| bf16 step (new rotary op) | 431.8 ms | 5051.3 ms |
| **mxfp8 step, default set** (cross-attention q/k/v + FFN) | **384.8 ms = 0.89x** | **4955.1 ms = 0.98x** |
| mxfp8 vs bf16 clip | 33.0 dB / 0.944 / 0.036 (floor 37.0) | 28.8 dB / 0.853 / 0.096 (floor 37.2) |
| CP2 / DP2 | 251.1 ms on 8 cores / forwards and reproduces | — |
| per-rank DiT weights | 7.50 -> 5.36 GB | |

Bugs found and fixed on the way: the K-remainder layout (`c813a47`), the driver tests' 27 GB
copies (`33504a7`), the rotary op's strided form (`fcb0dca` — its commit message credits it
with Wan's 3x step; section 4i shows that held only on one core, and the tp4 cause stayed open,
resolved by the default set of `917d087`).

**Superseded by section 8:** the tp4 cause is the rotary's element-wise DMAs on the kernel's
q/k. The default set of this section (cross-attention q/k/v + FFN) is the shipped default again
since H6 (`90c648f`, 8j). On the final tree its 9-frame step is 393.1 ms against 440.0 in bf16
(0.89x, 8k), against 4m's 384.8 ms: H5 put the bf16 self-attention back on the stride-2 rotary
(8h), which both arms pay.

## 5. Wan 2.2-T2V-A14B (two experts) on trn3, tp4, 480x832, 50 steps, seed 42

`/home/ubuntu/fp8-bench/wan22_tp4/`, `--dual-expert`, same prompt and cache as section 4; the
code is `917d087` (Wan's default set then, and again since H6, 8j). The hub DiTs are two fp32 experts (57.2 GB
each); `compile` made both bf16 copies (12 s each on this host) and, with `--quant`, one quantized
directory per expert from them with one spec (`[wan] MXFP8 transformer (nki): .../transformer/mxfp8-372c52a68c440756`,
`[wan] MXFP8 transformer_2 (nki): .../transformer_2/mxfp8-03f6d63f26aab3cf`, 19 GB each);
104 compiled blocks, no fallbacks.

| 9 frames | bf16 | mxfp8 (default set, both experts) |
|---|---|---|
| DiT step median | 431.6 ms | **385.2 / 385.1 ms = 0.89x** |
| determinism | | mp4 `775c8a22…` on both runs |
| latents vs bf16 | | cos 0.9984, rel L2 0.057 |
| clip vs bf16 | | 35.6 dB (worst 34.6) / 0.958 / 0.029 |
| per-rank DiT weights, both experts resident | 15.0 GB | 10.7 GB |

| 81 frames | bf16 | mxfp8 (default set, both experts) |
|---|---|---|
| compile (load + warm-up) | 439.8 s | 131.0 s |
| DiT step median | 5052.6 ms | 4953.2 ms = **0.98x** |
| request | 253.9 s | 248.8 s |
| latents vs bf16 | | cos 0.869, rel L2 0.514 |
| clip vs bf16 | floor (bf16 tp8 vs tp4; tp2 cannot hold both experts at 81 frames, `NRT ... neuron::alloc::lazy`): 28.7 dB / 0.878 / 0.067, latents cos 0.982 | 20.0 dB (worst 18.7) / 0.593 / 0.252 |

The 81-frame clip pair (`f81_bf16_vs_mx.png`, frames 5/40/75) shows two clean videos of the
same scene (the same tabby cat, the same planter boxes and composition) with the cat placed
differently and different foreground details: a different sample, not damage. Wan 2.2 hands the
trajectory from the high-noise to the low-noise expert mid-way, which plausibly amplifies a
small perturbation into a different layout. The bf16 floor confirms the sensitivity: a change of
TP degree alone (reduction order) puts two bf16 clips 28.7 dB apart at 81 frames, against 37
dB for Wan 2.1. mxfp8 is still well beyond it and **fails the device screen** (PSNR >= 24.7,
SSIM >= 0.78, LPIPS <= 0.13 at this floor); like Wan 2.1 at 81 frames it has no CPU fp32
reference: phase G ran 9 frames (section 10, where the default is 2.93 dB further from fp32 than
bf16), so the 81-frame screen failure stays unresolved. The default set is not changed
on this (the 9-frame run passes, and no subset was measured to do better at 81 frames).

### 5a. Summary for Wan 2.2-A14B

| | 9 frames | 81 frames |
|---|---|---|
| bf16 step | 431.6 ms | 5052.6 ms |
| **mxfp8 step** | **385.2 ms = 0.89x** | **4953.2 ms = 0.98x** |
| vs bf16 | 35.6 dB / 0.958 / 0.029 (passes) | 20.0 dB / 0.593 / 0.252 (floor 28.7 / 0.878; fails the screen) |
| vs CPU fp32 (section 10, the final tree's reruns of 8k) | mxfp8 33.46 dB / 0.949 where bf16 is 36.39 / 0.962: 2.93 dB further | no fp32 reference |
| per-rank DiT weights, both experts | 15.0 -> 10.7 GB | |

**Superseded by section 8:** H2's default (the self-attention q/k/v added in the half-split
layout) measured 0.77x at 9 frames and 0.94x at 81 on the high-noise expert alone (8e); with
both experts (8i) it ran 0.76x and 0.93x but moved the 12 clips 1.44 dB / SSIM 0.034 further
from bf16, past the rule, so H6 (`90c648f`) made this section's `917d087` set the default again
(8j). On the final tree the two-expert default runs 391.8 / 391.0 ms against 439.6 / 439.4 in
bf16 at 9 frames (0.89x, 8k). This section's numbers are the shipped set's, measured before H5.

## 6. FLUX.1-dev revisited (phase D): tp4, 1024x1024, 50 steps

`/home/ubuntu/fp8-bench/flux_d_tp4/`; the pre-D tree is `c2537cd` (`BASE`), the phase-D tree a
snapshot of `d1af3e1` (D1–D3; D4 touches HunyuanVideo-1.5 only). Prompt "a cat sitting on a
bench", seed 42 unless stated; two generates per arm, every pair bit-identical.

### 6a. The arms

| arm | tree | set | step medians | md5 | vs A1 (PSNR / SSIM / LPIPS) |
|---|---|---|---|---|---|
| A0 bf16 | pre-D | — | 221.4 / 221.6 ms | `67da9e98` | |
| M0 mxfp8 | pre-D | pre-D default | 195.0 / 195.0 ms (0.88x) | `676ecf5e` | 22.0 / 0.900 / 0.114 (vs A0) |
| A1 bf16 | D | — (new rotary) | 221.6 / 225.0 ms | `67da9e98` = A0 | — |
| F2 bf16 tp2 | D | — | 391.0 / 390.9 ms | `27a9ecd5` | 29.4 / 0.963 / 0.021 (the floor) |
| B0 mxfp8 | D | pre-D set, halves bf16 by name | 190.9 / 190.6 ms (0.86x) | `373dc396` | 22.2 / 0.902 / 0.109 |
| **B1 mxfp8** | D | **+ `proj_out_attn` / `proj_out_mlp`** | **182.9 / 183.3 ms (0.82x)** | `7b6a23dc` | 22.6 / 0.908 / 0.098 |

- **The old "0.99x" is history.** Section 7 of the Qwen evidence measured FLUX on 2026-10-31,
  before the kernel's `fast_dma_transpose` load path (2026-11-01); the pre-D tree today gives
  0.88x with the same targets. At 28 steps that tree's bf16 image is bit-identical to October's
  (`8b217058`), and its mxfp8 image is 34.7 dB / 0.978 / 0.016 from bf16 (October: 31.7 dB).
- **D3 (FLUX's rotary through the neuron op) stays:** A1's image is bit-identical to A0's and
  its step within 1% (223.3 against 221.5 ms, mean of the two runs).
- **The halves' kernel shapes** (one core, `probe_mx_shapes_flux.py`): 4608 x 768 -> 3072
  (K = 512 + 256), 4608 x 3072 -> 3072 and 4608 x 384 -> 3072 (tp8's single 384 tile) all
  pass (rel L2 2.34–2.36e-3, the bf16 rounding of section 1a).

### 6b. Quality at 50 steps is trajectory drift, judged on twelve images

On one image every mxfp8 arm is about 22 dB from bf16, and bf16 at another TP degree is only
29.4 dB from itself: at 50 steps the trajectory turns the cat's head (`A1_F2_M0_B1.png`), a
different valid image. The plan's single-image screen (30 dB / 0.97 / 0.05) therefore fails
every set, bf16's own floor included, and its bisection would rank noise; the twelve-image
confirmation (4 prompts x 3 seeds, the Qwen 6h prompts, one batch per arm) ran instead, for B1,
B0, bf16 and the tp2 floor (`gate/`, `score.py`):

| 12 images, 50 steps | step median | PSNR vs bf16, mean (min) | floor-relative gate per image |
|---|---|---|---|
| floor: bf16 tp2 vs tp4 | 390.8 ms | 29.3 (19.5) dB | — |
| B0 | 191.1 ms | 24.2 (16.9) dB | 5 / 12 |
| **B1** | **183.4 ms** | **24.7 (16.6) dB** | **6 / 12** |
| bf16 tp4 | 222.3 ms | — | — |

The worst image (typewriter, seed 7: B1 16.6 dB / 0.715, B0 16.9 / 0.723) is a clean typewriter
placed differently with a different invented label, as is the tp2 bf16 one (`p2_s7_strip.png`).
No set can pass a 30 dB mean here (the floor's mean is 29.3), so the quality verdict at this
step count was left to the CPU fp32 reference (section 10: on its one image B1 is 21.30 dB from
fp32 and bf16 23.36, a 2.06 dB gap that one image of this sampler cannot resolve). Between the two candidates the screen does
decide: B1 drifts no more than B0 and is 4% faster, so **FLUX's default becomes B1's set**,
the single-stream `proj_out` halves included: **0.82x** at tp4, 1024x1024, 50 steps.

The default commit (`3d7dd6c`) was checked on the device: `compile --quant mxfp8` with no
override resolves B1's directory (`mxfp8-809df193b011ba1f`, 15 GB for 23.8 GB of bf16), and a
generate reproduces B1's image bit for bit (`7b6a23dc`) at 183.1 ms per step.

## 7. HunyuanVideo on trn3, tp4, 320x512x61 and 544x960x129, 50 steps, seed 42

`/home/ubuntu/fp8-bench/hv1_tp4/`, `hunyuanvideo-community/HunyuanVideo` at revision
`e8c2aaa66fe3742a32c11a6766aecbf07c56e773` (a bf16 hub DiT, so the quantized directory is written
straight from the snapshot, no bf16 copy), prompt "a cinematic shot of a red fox running through
a snowy forest", guidance 6.0, cache `/home/ubuntu/difflet-cache-video`, decode in-process on
rank 0. Plan: `docs/plans/2026-11-07-mxfp8-hunyuan-video.md` (phase E). Three source snapshots:
`src` (`d19b0b2`: the bf16 arms at S), `src-e` (`f61e833`: the mxfp8 and leave-one-out arms, the
profiles, the layouts and bf16 at the long shape) and `src-e2` (`fc38fb6`, the committed default:
mxfp8 at the long shape). E6 (`fd0d68f`, the single-stream `proj_out` as its halves) landed before
E5's device runs, so E5's arms measured the final table, G7 included, and E6's own arms (6b-6d)
were not needed (7d). Every compile and generate below logged 60 compiled blocks and no
fallbacks; the step medians are over n=49 steps.

### 7a. bf16, the noise floor and the screen (E5 steps 1-3)

The snapshot's DiT is 24 GB (`du -L` of `transformer/`; step 1's download ran earlier).

| arm | step median (gen50a / gen50b) | determinism | vs bf16 tp4 |
|---|---|---|---|
| bf16 tp4 (`src`, compile load + warm-up 182.8 s) | **660.0 / 659.0 ms** (request 52.67 / 47.34 s) | mp4 `75e6ba8d…`, latents `d24571b8…` on both runs | - |
| bf16 tp2, the floor (one generate, 173.2 s warm-up) | 1230.9 ms on two cores | mp4 `946c3baf…` | PSNR 36.64 dB (worst 33.71), SSIM 0.957, LPIPS 0.031; latents cosine 0.99998, rel L2 0.0068 |

The decode after the last step takes 13.79 s. The thresholds of steps 4-8 resolve to:

- `J` = max(the largest gen50a/gen50b gap among the compared arms, 1.7 ms for V15; 1% of bf16's
  659.5 ms, 6.6 ms) = **6.6 ms**. An arm's step is the mean of its two medians.
- `F` = 36.64 dB / 0.957 / 0.031.
- `Q`, the floor-relative screen of the plan's Decisions block (the same for phases D, E and F):
  PSNR >= min(30, 36.64 - 4) = **30 dB**, SSIM >= min(0.97, 0.957 - 0.10) = **0.857**,
  LPIPS <= max(0.05, 2 x 0.031) = **0.062**. Step 5's own text still carries an older margin form
  (F - 4.1 dB, F - 0.017, F + 0.017, which is 30 dB / 0.940 / 0.05 here); under it the default
  below (V15, SSIM 0.938, LPIPS 0.0501) would miss SSIM by 0.002 and LPIPS by 0.0001.

### 7b. The default set by leave-one-out (E5 steps 4-5)

The groups are the seven patterns of `HUNYUAN_VIDEO_TARGETS` as they stood at `f61e833`, in table
order (G1-G7, 440 linears; the committed table of `fc38fb6` keeps G2-G4, G6 and G7, 260 linears,
and moves G1 and G5 to `HUNYUAN_VIDEO_KEPT_BF16`), with `B = transformer_blocks\.\d+` (20
dual-stream blocks) and `S = single_transformer_blocks\.\d+` (40 single-stream blocks):

| group | pattern | layers |
|---|---|---|
| G1 | `B\.attn\.(to_q\|to_k\|to_v)` | dual-stream image q/k/v |
| G2 | `B\.attn\.(add_q_proj\|add_k_proj\|add_v_proj)` | dual-stream text q/k/v (256 rows) |
| G3 | `B\.ff\.net\.(0\.proj\|2)` | dual-stream image MLP |
| G4 | `B\.ff_context\.net\.(0\.proj\|2)` | dual-stream text MLP (256 rows) |
| G5 | `S\.attn\.(to_q\|to_k\|to_v)` | single-stream joint q/k/v |
| G6 | `S\.proj_mlp` | single-stream MLP up |
| G7 | `S\.proj_out_(attn\|mlp)` | single-stream `proj_out`, as its attention and MLP halves |

`mx` quantizes all seven. Each arm `Vij` drops groups i and j through `DIFFLET_QUANT_TARGETS`
(table order kept), on its compile and both generates; every generate's MXFP8 line named its own
compile's directory (`loo_<arm>.info`). V15, V24 and V7 are the step's required arms; V1 to V6,
which the step requires only when `mx` is not faster than bf16 (here it is), ran as well, so every
group has its own arm. Ratio = the arm's step against bf16's 659.5 ms; quality is gen50a against
bf16 gen50a:

| arm | quantized groups | step medians | ratio | mp4 md5, a = b | PSNR / SSIM / LPIPS | latents cos | clears `Q` |
|---|---|---|---|---|---|---|---|
| bf16 | - | 660.0 / 659.0 ms | 1.00x | `75e6ba8d…` yes | - | - | - |
| mx | G1-G7 (440 linears, 17 GB) | 606.0 / 605.6 ms | 0.92x | `75a4016c…` yes | 21.19 / 0.830 / 0.170 | 0.98370 | no (all three) |
| **V15** | **G2-G4, G6, G7 (260 linears, 18 GB)** | **634.9 / 636.6 ms** | **0.96x** | `7d54b413…` yes | **31.09 / 0.938 / 0.050** | 0.99894 | **yes** |
| V24 | G1, G3, G5-G7 | 605.6 / 606.0 ms | 0.92x | `98432224…` yes | 27.64 / 0.912 / 0.085 | 0.99702 | no (PSNR, LPIPS) |
| V7 | G1-G6 | 667.9 / 668.2 ms | 1.01x | `b8a4cfbe…` yes | 27.59 / 0.912 / 0.084 | 0.99686 | no (PSNR, LPIPS) |
| V1 | G2-G7 | 623.9 / 625.1 ms | 0.95x | `f1e49514…` yes | 29.77 / 0.931 / 0.059 | 0.99837 | no (PSNR) |
| V2 | G1, G3-G7 | 603.1 / 603.3 ms | 0.91x | `82c74997…` yes | 20.95 / 0.826 / 0.175 | 0.98248 | no (all three) |
| V3 | G1, G2, G4-G7 | 662.3 / 662.5 ms | 1.00x | `27d87cb9…` yes | 26.71 / 0.902 / 0.097 | 0.99596 | no (PSNR, LPIPS) |
| V4 | G1-G3, G5-G7 | 612.2 / 613.8 ms | 0.93x | `fc400270…` yes | 28.24 / 0.919 / 0.075 | 0.99756 | no (PSNR, LPIPS) |
| V5 | G1-G4, G6, G7 | 619.5 / 620.4 ms | 0.94x | `85da2876…` yes | 29.97 / 0.931 / 0.060 | 0.99855 | no (PSNR) |
| V6 | G1-G5, G7 | 690.3 / 689.2 ms | 1.05x | `b2e8932c…` yes | 28.48 / 0.920 / 0.075 | 0.99759 | no (PSNR, LPIPS) |

The latents files match within every pair too. The two directory sizes are those of `mx` and
the default; the other arms' directories were deleted after the decision without a recorded
size.

- **Quality.** With every group the clip is 21.19 dB from bf16, 15 dB beyond the floor. No single
  group carries the loss: dropping G1, G5, G4 or G6 alone recovers 7 to 9 dB (29.77, 29.97,
  28.24, 28.48), G3 and G7 5.5 and 6.4 dB, G2 nothing (20.95). Only V15, without both image-token
  q/k/v groups (G1, G5), clears the screen.
- **Speed.** Without G6 or G7 the step is slower than bf16's (689.8 and 668.1 ms) and without G3
  level with it (662.4 ms): each leaves a bf16 matmul between MX kernels in its block (not
  profiled further). V15 buys its quality with 30 ms per step against `mx` (635.8 against 605.8).
- **Rule.** The candidates are the sets that clear `Q`: V15 alone (`mx` fails all three gates).
  V15 is faster than bf16 by more than `J` (635.8 < 659.5 - 6.6), so branch 2 makes it the
  default; there is no tie to break.

The default was committed as `fc38fb6`: G1 and G5 move to `HUNYUAN_VIDEO_KEPT_BF16` with this
measurement in its comment, and the table equals V15's override pattern for pattern, so the commit
resolves V15's directory (`mxfp8-46e0952c1c5a9f74`) and NEFFs with no override (the long-shape
compile of 7f, from `fc38fb6`, printed that directory).

### 7c. Block device time (E5 step 5's profile)

The plan's 3-step `inspect` system profile gives no engine view, so the profile used the capture
recipe of the Wan phase H work (`wan21_tp4/replay/capture.sh`), as step 5 allows: each arm's
NEFFs dumped from a 1-step generate under `neuron-explorer inspect` with
`NEURON_RT_INSPECT_DEVICE_PROFILE=1` (cores 8-11), rank 0's two block NEFFs replayed with their
collectives as four workers (`neuron-explorer capture --collectives-worker-count=4
--collectives-workers-per-node=4 ... --num-exec=4 --profile-nth-exec=4`), and the fourth
execution read with `view --output-format summary-json` (`run_prof.sh`, `run_prof2.sh`):

| block NEFF, rank 0, device time per execution | bf16 | mx (G1-G7) | V15 (default) |
|---|---|---|---|
| dual-stream (13 collectives) | 11.45 ms | 13.02 ms | **11.06 / 11.29 ms** (two replays) |
| single-stream (6 collectives) | 10.31 ms | 9.56 ms | **9.65 ms** |

The V15 dual-stream NEFF was replayed twice, by `run_prof.sh` (`prof_V15/blk2`) and by
`run_prof2.sh` (`wan21_tp4/replay/hv1_V15_first`): the same 6,569,387-byte NEFF (md5
`10ec59b4…`), 11.06 and 11.29 ms. The other cells are one replay each, so differences below about
0.25 ms are within replay noise. G1 is the dual block's cost: with the image q/k/v quantized the
block is 1.57 ms slower than in bf16; without them it is 11.06 or 11.29 ms against 11.45, level
with bf16 within that noise. In the single block G5's 0.09 ms is inside the noise as well, so it
stays bf16 for quality alone. The isolated replays do not add up to the step: 20 x 13.02 +
40 x 9.56 = 643 ms predicts `mx`, measured 605.8, and they put V15's blocks ahead of `mx`'s while
its step is 30 ms slower. They rank the blocks, not the step.

### 7d. The single-stream `proj_out` halves (E6) and the kernel probe at K = 768

E6 (`fd0d68f`) put the single-stream `proj_out` in the table as its attention and MLP halves
(G7; phase D's split writer, each half quantized per rank K range). At tp4 `proj_out_attn`'s K
per rank is 768 = 512 + one partial tile of 256, HV1's first K remainder, so step 6a probed it on
one core before any arm quantized it (`probe_hv1_shapes.py`, M = 10,496, the joint rows of a single
block at 320x512x61):

| layer (per rank, tp4) | M x K -> N | rel L2 | MX | bf16 | ratio |
|---|---|---|---|---|---|
| `proj_out_attn` (row, K = 3072/4 = 512 + 256) | 10496 x 768 -> 3072 | 2.35e-3 | 1.50 ms | 1.48 ms | 1.02 |
| `proj_out_mlp` (row, K = 12288/4) | 10496 x 3072 -> 3072 | 2.34e-3 | 1.81 ms | 2.74 ms | 0.66 |
| `proj_mlp` (column, control) | 10496 x 3072 -> 3072 | 2.34e-3 | 1.90 ms | 2.77 ms | 0.69 |

All three pass at rel L2 < 3e-3, the two bf16 roundings of section 1a (FLUX's own 768-wide halves:
2.34-2.36e-3, section 6a). The probe's fp32 cosine printed 1.004 at every shape, an artifact of
fp32 accumulation over 32M output elements, so rel L2 is the criterion here. In the pipeline G7
pays although `proj_out_attn` alone does not: without it (V7) the step is 668.1 ms against
`mx`'s 605.8.

**State of `proj_out`:** quantized, as its halves, in the default (G7). E6 step 6's arms 6b-6d
were for the order E5-then-E6; here E5's leave-one-out already holds E6's arm (V7: slower and
outside the screen), and the rule kept G7.

### 7e. Layouts at S with the default (E5 step 8)

From `src-e` (`f61e833`) with V15's override, the committed default's spec (the same directory
`mxfp8-46e0952c1c5a9f74`, `run_layouts.sh`); quality is gen50a against bf16 tp4 gen50a and against
the tp4 default's gen50a:

| layout | cores | step medians | mp4 md5, a = b | vs bf16 tp4 (PSNR / SSIM / LPIPS) | vs the tp4 default |
|---|---|---|---|---|---|
| tp4 (7b) | 4 | 634.9 / 636.6 ms | `7d54b413…` yes | 31.09 / 0.938 / 0.050 | - |
| CP2 gather_kv (`--tp-degree 4 --cp-degree 2 --cp-mode gather_kv`) | 8 | **402.6 / 401.1 ms** | `2e0fe77a…` yes | 31.82 / 0.942 / 0.044 | 31.63 / 0.940 / 0.050 |
| SP (`--tp-degree 4 --sp`) | 4 | **1263.8 / 1264.2 ms** | `bce5eb03…` yes | 31.63 / 0.940 / 0.047 | 33.80 / 0.948 / 0.041 |
| bf16 SP (context, one generate) | 4 | 659.4 ms | `b846ea8d…` | | |
| DP2 (`--dp 2`, two tp4 replicas) | 8 | 636.1 / 635.9 ms, one per worker | `dp2_fox.mp4` = `7d54b413…` | | bit-identical |

- **CP2** runs on eight cores at 401.9 ms against 635.8 on four (compile load + warm-up 111.0 s).
- **SP** reproduces itself bit for bit, and against bf16 its clip is as close as the tp4
  default's (31.63 against 31.09 dB). Against the tp4 default it is 33.80 dB, 2.8 dB beyond the
  bf16 tp2-vs-tp4 floor (36.64): Review Focus 3 expected at most the floor; recorded, not
  explained. **MXFP8 under SP is 1.92x bf16 under SP** (1264.0 against 659.4 ms; bf16 SP is level
  with bf16 tp4). The same capture recipe under SP (`profsp_bf16`, `profsp_V15`):

  | block NEFF under SP, rank 0 | bf16 SP | default SP |
  |---|---|---|
  | dual-stream: device time | 11.50 ms | 42.53 ms |
  | of which ACT (scalar) active / hardware dynamic DMA active | 3.07 / 3.92 ms | 32.87 / 34.95 ms |
  | the second block NEFF: device time | 10.22 ms | 29.48 ms |
  | of which vector engine active | 6.76 ms | 26.79 ms |

  In the dual block, HLO `%add.330`, the rotary (`x*cos + swap(x)*sin`,
  `difflet/backends/neuron/ops_impl/embeddings.py`), takes 47.8 ms of scalar-engine
  `DMA_DIRECT2D` (summed over its DMAs), the signature Wan's self-attention q/k showed before the
  half-split layout. In the default the q/k it rotates come from the bf16 `to_q` / `to_k` (G1
  stays bf16), not from the MXFP8 kernel, so what makes the compiler move them element by element
  under SP is open. The second NEFF's extra time is vector/gpsimd `TENSOR_TENSOR` work. Not fixed
  here: a user decision.
- **DP2:** the router put the workers on cores 0-3 and 4-7, both printed
  `[hunyuan_video] MXFP8 transformer (nki): .../mxfp8-46e0952c1c5a9f74`, each ran within `J` of
  the tp4 default (636.1 and 635.9 against 635.8 ms), and worker 0's clip is bit-identical to the
  single launch's. **DP router bug found on the way** (pre-existing, on `main`):
  `replica_core_ranges` (`difflet/cli/dp/router.py`) counts cores from 0 and ignores an inherited
  `NEURON_RT_VISIBLE_CORES`, so an earlier DP2 attempt given cores 20-27 took 0-7 and collided
  with another job (`failed_collision/`). Not fixed on this branch.

### 7f. The long shape (E5 steps 6-7)

**720x1280x129 does not run at tp4 on this host.**

| attempt | result |
|---|---|
| bf16, tp4 | HBM OOM at load; the log is lost (the CP2 fallback's run overwrote it) |
| bf16, tp4 x CP2 gather_kv (`bf16_L_cp2_attempt1_compile.log`) | "Failed to allocate DEVICE memory (179884288 bytes)" for the static rings of `qSPSpillReload` at load, and allgather watchdog timeouts after 1800 s on the ranks; stopped |

The compile of that shape ran on the host: at one `ps` sample (22:30) the four compiler backends
(`walrus_driver`) held 117 to 149 GB of RSS each, about 550 GB together. Its peak RSS was not
recorded with `time -v`, which step 6 asked for. The first fallback in the plan's order,
**544x960x129 at tp4**, runs and is the realistic shape:

| 544x960x129, tp4 | bf16 (`src-e`) | **mxfp8, default (`fc38fb6`)** |
|---|---|---|
| compile (load + warm-up) | 2397.0 s, 60 blocks, no fallbacks | 2490.3 s, 60 blocks, no fallbacks, `mxfp8-46e0952c1c5a9f74` |
| DiT step median | 18493.6 / 18493.4 ms | **17967.6 / 17969.6 ms = 0.97x** |
| request (encode + 50 steps + decode) | 1016.69 / 1002.56 s | 976.76 / 977.24 s |
| decode (after the last step) | 76.98 s | 77.06 / 76.99 s |
| determinism | mp4 `258cd73e…`, latents `2330bb46…` on both runs | mp4 `00c19154…`, latents `a313d8aa…` on both runs |
| vs bf16 (129 frames) | - | PSNR 26.46 dB (worst 20.07), SSIM 0.907, LPIPS 0.128; latents cosine 0.98981, rel L2 0.143 |

The step gains 524.9 ms, beyond `J` at this shape (1% of bf16, 184.9 ms): 0.97x where the plan
expected about 1.0x.

**The floor at this shape does not fit.** Step 7's `bf16_tp2_L` (bf16, `--tp-degree 2`, one
generate) ran twice. The first attempt (cores 0-1, started 05:31) died in the 06:14 full-disk
window ("No space left on device" from the compiler), and its ranks lingered on core 0 until the
wrapper exited. The rerun on device 8 (cores 32-33) ran out of HBM at 07:19 ("Failed to allocate
DEVICE memory (415236096 bytes)") and was stopped cleanly by the OOM watcher (exit `oom`). As step
7 prescribes when tp2 does not fit, the floor quoted is step 3's at 320x512x61: 36.64 dB / 0.957 /
0.031. Against S's thresholds (30 dB, SSIM 0.857, LPIPS 0.062) the 544x960x129 clip (26.46 dB /
0.907 / 0.128) misses PSNR and LPIPS, but a floor from another shape cannot resolve the screen at
this one, so by the Decisions block it was left to the CPU fp32 reference, as Wan 2.1 at 81
frames was. Phase G referenced 320x512x61 only (section 10), so this shape stays unresolved.

### 7g. Sizes

| | bf16 | mxfp8 default (V15) | mxfp8, every group (`mx`) |
|---|---|---|---|
| DiT on disk | 24 GB (`du -L` of the snapshot's `transformer/`) | **18 GB** (`mxfp8-46e0952c1c5a9f74`) | 17 GB (`mxfp8-bab954df5f0f081e`, deleted) |
| per-rank DiT weights at tp4 (weight-store line) | 6.54 GB | 4.85 GB | 4.43 GB |

### 7h. Summary for HunyuanVideo

| layout | DiT step | vs bf16 tp4 clip (PSNR / SSIM / LPIPS) | latents rel L2 / cosine |
|---|---|---|---|
| bf16 tp4, 320x512x61 | 660.0 ms | - (floor tp2: 36.64 / 0.957 / 0.031) | (floor: 0.0068 / 0.99998) |
| **mxfp8 tp4, default** | **634.9 ms = 0.96x** | 31.09 / 0.938 / 0.050 (clears the screen) | 0.046 / 0.99894 |
| mxfp8 tp4, every group | 606.0 ms = 0.92x | 21.19 / 0.830 / 0.170 (fails it) | 0.180 / 0.98370 |
| mxfp8 tp4 x CP2 | 402.6 ms on 8 cores | 31.82 / 0.942 / 0.044 | 0.040 / 0.99920 |
| mxfp8 tp4 SP | 1263.8 ms (1.92x bf16 SP) | 31.63 / 0.940 / 0.047 | 0.042 / 0.99912 |
| mxfp8 dp2 (each replica) | 636.1 / 635.9 ms | bit-identical to tp4 mxfp8 | |
| bf16 tp4, 544x960x129 | 18493.6 ms | - (no floor at this shape: bf16 tp2 runs out of HBM; S's 36.64 / 0.957 / 0.031 quoted) | |
| **mxfp8 tp4, 544x960x129, default** | **17967.6 ms = 0.97x** | 26.46 / 0.907 / 0.128 | 0.143 / 0.98981 |

Speed: below bf16 at both shapes (0.96x and 0.97x), with the DiT at 18 GB for 24 GB of bf16.
Quality: the default clears the device screen at 320x512x61, and against the CPU fp32 reference
there (section 10) it is 20.59 dB where bf16 is 20.79, both far from fp32 and the default 0.20 dB
the further: no worse than bf16 within one clip. 544x960x129 has no fp32 reference. Open items:

- **MXFP8 under SP is 1.92x bf16 SP**: the rotary's element-wise DMAs in the dual block, as Wan's
  self-attention showed, here on q/k from the bf16 image projections (cause open, not fixed).
  README and `docs/cli-staged-commands.md` say not to combine `--quant mxfp8` with `--sp` (SP
  with MXFP8 is unmeasured on the other models).
- **The DP router ignores an inherited `NEURON_RT_VISIBLE_CORES`** (`replica_core_ranges`),
  pre-existing on `main`.
- 720x1280x129 does not load at tp4 or tp4 x CP2 on this host (HBM), and bf16 at tp2 does not fit
  at 544x960x129 (HBM OOM, 7f), so the long shape has no floor of its own.

## 8. Wan self-attention q/k/v: the rotary layout (phase H)

Plan: `docs/plans/2026-11-08-mxfp8-wan-half-split-rope.md` (`9c86f0a`). Code: H1 `73f1722` (the
half-split layout in the Wan model), H2 `f0ad1c6` with the review fix `65b4ec5` (the
application's rule and the default set), and `1a8382f` (the TPU sequence-parallel override
rotates in the module's layout). CPU gate at `65b4ec5`: 2058 passed, 111 skipped, 0 failed. Then
H5 `c7c2346` (bf16 Wan's interleaved rotary back on the stride-2 law, 8h) and H6 `90c648f` with
the review fix `19b0aa9` (Wan's default back to the `917d087` set, 8j). **What ships:** the
`917d087` set (cross-attention q/k/v + FFN); H2's set (8a-8g) failed the rule with both Wan 2.2
experts (8i) and is an override now, the half-split layout with it. Host,
cache and conventions as in section 4: trn3.48xlarge, LNC=2, tp4, 480x832, 50 steps, seed 42,
"a cat walking through a garden" unless noted. Raw files: `/home/ubuntu/fp8-bench/wan21_tp4/replay/`
(diagnosis and prototype) and `/home/ubuntu/fp8-bench/wan_h3/` (H3, H5, the two-expert runs and
the final-tree reruns).

### 8a. The capture recipe

The device profile 4l could not get: dump the NEFFs of a real tp4 run, then replay the block
NEFF alone with its collectives.

1. **Dump**: `NEURON_RT_VISIBLE_CORES=8-11 NEURON_RT_INSPECT_DEVICE_PROFILE=1 neuron-explorer
   inspect -o <dir> <generate ... --steps 1>`, without `NEURON_RT_INSPECT_EVENT_FILTER_NC` (the
   non-zero cores refuse it: "Requested capture for NC 0, but it is not visible"). Rank 0's
   largest NEFF is the block.
2. **Replay**: `NEURON_RT_VISIBLE_CORES=8-11 neuron-explorer capture -n <neff>
   --collectives-worker-count=4 --collectives-workers-per-node=4 --collectives-worker-start-id=0
   --collectives-profile-id=0 --num-exec=4 --profile-nth-exec=4`.
3. **Read**: `neuron-explorer view -n <neff> -s profile_rank_0_exec_4.ntff --output-format
   summary-json` (prints to stdout; it takes no `--output-file`), or `--output-format parquet
   --output-file <dir>` for the instruction tables.

`wan21_tp4/replay/capture.sh` wraps steps 2-3; HunyuanVideo's block profiles (7c, 7e) used it.

### 8b. The profile

| per block, rank 0 | bf16 | default `917d087` (attn2 + FFN) | all q/k/v + FFN, interleaved | all q/k/v + FFN, half-split |
|---|---|---|---|---|
| total exec | 10.67 ms | 9.30 ms | **56.89 ms** | **8.05 ms** |
| TensorE active | 9.21 ms | 7.32 ms | 6.58 ms | 5.74 ms |
| ACT (scalar) active | 3.12 ms | 2.25 ms | **49.31 ms** | 1.99 ms |
| hardware dynamic DMA active | 4.59 ms | 4.23 ms | **51.67 ms** | 3.94 ms |
| hardware dynamic DMA bytes | 2.34 GB | 2.16 GB | 2.31 GB | - |
| ACT instructions | 12.3k | 11.1k | 15.7k | - |

The replays agree with 4k's pipeline medians (10.50 and 56.27 ms). The slow block moves no more
bytes than bf16 and does less TensorE work; its ACT queue and its DMAs are busy for 49 to 52 ms.
The instruction tables put that in the rotary op (`difflet/backends/neuron/ops_impl/embeddings.py:39`,
`x*cos + (x@P)*sin`; HLO `%add.237` / `%add.246`):

| the rotary op, per block | q/k from bf16 matmuls (bf16, default `917d087`) | q/k from the MXFP8 kernel (interleaved) |
|---|---|---|
| scalar-queue DMAs | 240 | 1480 |
| element per DMA (`src_elem_size`) | 256 B, whole head rows | 2 B, single bf16 elements (pattern `[2,262144][64,128]`) |
| DMA time, summed | 0.04 ms | 87.8 ms |
| op time, summed | 0.92 ms | 91.2 ms |

The DMAs overlap, so their summed 87.8 ms exceeds the 56.89 ms block. The op's HBM write in the
slow arm is 11.98 MB, one rank's q (4680 tokens x 10 heads x 128 x 2 B).

### 8c. Cause and fix

**Cause.** The q/k that come out of the MXFP8 kernel land in a layout the interleaved pair swap
(pairs `(2i, 2i+1)`) can only reach element by element, so the compiler moves them in 2-byte DMAs
on the scalar queue; from a bf16 matmul it lays q/k out for the swap and moves whole rows. This
is the effect 4h saw around a single custom-op output feeding RoPE; at tp4 its time is in the
rotary op itself, not in a collective.

**Fix (H1, H2).** When any self-attention `to_q` or `to_k` is quantized, the Wan application
reorders every self-attention head's q/k output channels at load from (0, 1, 2, 3, ...) to (0, 2,
4, ..., 1, 3, 5, ...), with their MX block scales, the bias and the q/k RMSNorm weights, and logs
`[wan] self-attention q/k reordered for the half-split rotary (40 attentions): their q/k come from
the MXFP8 kernel`. The rotary then uses the half-split law: tables `cat(c, c)` / `cat(-s, s)`, the
rotation `x * cos + cat(x[d/2:], x[:d/2]) * sin` on contiguous halves. q and k get the same
permutation, so `q . k` is unchanged up to summation order, and v and the output projection are
untouched. bf16 q/k keep the interleaved law: the half split costs bf16 12% (483.6 against
431.9 ms per step, block 11.36 against 10.67 ms). On the CPU toys the reordered half-split model
matches the interleaved one to 2.4e-7 / 3.0e-7 (max relative error); the control (reordered
weights, interleaved rotary) is 2-3e-2 off. H2 put the self-attention q/k/v in the default set;
since H6 (8j) the default leaves them bf16, and the layout applies to a `DIFFLET_QUANT_TARGETS`
override that quantizes a self-attention q or k.

### 8d. Same-session A/B (the prototype, 480x832x9, every NEFF cached)

A scratch copy carrying the plan (`wan21_tp4/replay/make_hs_scratch.py`), one session:

| arm | step median | block (device) | vs bf16 | clip vs bf16 (PSNR / SSIM) |
|---|---|---|---|---|
| bf16 (interleaved) | 431.9 ms (432.3 rerun) | 10.67 ms | 1.00x | the rerun: 37.24 dB / 0.961 (not bit-identical) |
| bf16, half-split | 483.6 ms | 11.36 ms | 1.12x | 36.35 / 0.953 (vs 4i's bf16 clip, another session) |
| default `917d087` (attn2 + FFN) | 387.3 ms | 9.30 ms | 0.90x | (4m: 33.0 / 0.944) |
| all q/k/v + FFN, interleaved | (4i: 2262.9 ms) | 56.89 ms | (5.2x) | (4i: 32.3 / 0.938) |
| **all q/k/v + FFN, half-split** | **330.7 ms** | **8.05 ms** | **0.77x** | 34.23 / 0.946, latents cos 0.99877 (vs 4i's bf16 clip, another session) |

The step medians are this session's. Only the bf16 rerun's quality is against this session's bf16
(`ab_bf16.mp4`); the two half-split arms' quality was measured in the earlier `run_hs.sh` session
against 4i's bf16 clip (`rope_bf16_f9_gen50a.mp4`, `d1d89955…`). Their clips are the same bytes in
both sessions (`da16a5d3…`, `34fb1ddc…`), but bf16 is not reproducible run to run (8g), so the
reference differs from the rerun's.

The half-split arm's clip (`da16a5d3…`) is the one H2's committed default reproduces in 8e, in
another session.

### 8e. H3: H2's default as committed (`1a8382f`), one session per model

`wan_h3/h3.sh` on a snapshot of `1a8382f`: per model and frame count, bf16, the old default (the
`917d087` set through `DIFFLET_QUANT_TARGETS`, the same directories as before) and the new default
(no override), each compiled first; two generates at 9 frames, one at 81; compares against the
same session's bf16 gen50a. "New default" here and in 8f is H2's set, which H6 reverted (8j).

| step median | bf16 | old default (`917d087` set) | **new default** |
|---|---|---|---|
| Wan 2.1, 9 frames | 433.4 / 433.9 ms | 385.5 / 387.1 ms (0.89x) | **331.6 / 336.4 ms (0.77x)** |
| Wan 2.1, 81 frames | 5052.9 ms | 4956.0 ms (0.98x) | **4731.2 ms (0.94x)** |
| Wan 2.2, 9 frames | 432.8 / 432.9 ms | 385.6 / 387.3 ms (0.89x) | **331.3 / 334.2 ms (0.77x)** |
| Wan 2.2, 81 frames | 5046.2 ms | 4950.2 ms (0.98x) | **4731.7 ms (0.94x)** |

- **Determinism.** Both mxfp8 defaults reproduce across their two 9-frame generates: Wan 2.1 new
  `da16a5d3…` (the prototype's clip, 8d) and old `fd29d953…` (4m's clip), Wan 2.2 new
  `c75290d6…` and old `f4fb1bc6…`, latents equal too. bf16 does not (8g).
- **Wan 2.2 ran one expert.** `h3.sh` (and the 12-clip batches of 8f) pass no `--dual-expert`, so
  Wan 2.2 ran its high-noise expert for every step, the CLI's default: one MXFP8 directory
  (`transformer/mxfp8-0a076ca2174a047e` for the new default), no `transformer_2` line, 64
  compiled blocks (section 5's two-expert runs had 104). The two-expert runs came later, in 8i.
- **Weights.** The new default's per-rank DiT weights are 4.60 GB (weight-store line) against
  7.50 in bf16 and 5.36 with the old default.

Single-clip quality against the same session's bf16 (gen50a; at 9 frames also against gen50b):

| PSNR / SSIM vs bf16 | old default | new default | new - old |
|---|---|---|---|
| Wan 2.1, 9 frames | 32.57 / 0.939 | 34.63 / 0.952 | +2.06 dB / +0.013 |
| Wan 2.1, 9 frames, vs bf16 run b | 33.71 / 0.947 | 33.67 / 0.943 | -0.04 dB / -0.004 |
| Wan 2.1, 81 frames | 28.83 / 0.853 | 28.12 / 0.831 | -0.71 dB / **-0.022** |
| Wan 2.2, 9 frames | 34.12 / 0.946 | 32.84 / 0.940 | **-1.28 dB** / -0.006 |
| Wan 2.2, 9 frames, vs bf16 run b | 34.18 / 0.947 | 32.98 / 0.941 | **-1.20 dB** / -0.006 |
| Wan 2.2, 81 frames | 23.31 / 0.798 | 28.87 / 0.894 | +5.56 dB / +0.096 |
| bf16 run a vs run b, 9 frames | 36.01 / 0.952 (Wan 2.1), 36.42 / 0.955 (Wan 2.2) | | |

### 8f. The rule on twelve clips (a deviation, as FLUX's 6b)

H3 step 5 ships the new default when it is faster than the `917d087` default by more than `J` at
9 frames (334.0 against 386.3 ms for Wan 2.1, 332.8 against 386.5 for Wan 2.2, with `J` = 4.8 ms,
the new default's own run-to-run gap: it is) and, at 9 and 81 frames for both models, its PSNR is
not below the old default's by more than 1 dB and its SSIM not by more than 0.01. On the single
clips two cells fail (Wan 2.1 at 81 frames, SSIM -0.022; Wan 2.2 at 9 frames, -1.28 / -1.20 dB)
while the other two gain +2.06 and +5.56 dB, and Wan 2.1's 9-frame gain shrinks to -0.04 dB when
the reference is bf16's second run. Two bf16 runs of one tree are only 36.0 / 36.4 dB apart, so
these cells are noise-dominated and the rule would rank noise. As FLUX's D5 did, the decision moved to the mean over 4 prompts x 3 seeds at 9
frames: "a cat walking through a garden", "a red fox running through a snowy forest, cinematic",
"ocean waves crashing against a rocky shore at sunset", "a busy city street at night with neon
signs and people walking, cinematic", seeds 42, 7 and 1234; each arm one batch launch of the 12
requests (`run_multi.sh`, from a scratch copy `src-multi` whose DP router draws its cores from
`NEURON_RT_VISIBLE_CORES`, because of the router bug of 7e):

| 12 clips, 9 frames, mean vs bf16 | old default | new default | new - old | clips where new is closer |
|---|---|---|---|---|
| Wan 2.1 | 32.11 dB / 0.934 / LPIPS 0.042 (worst clip 28.42) | 31.65 / 0.931 / 0.045 (28.15) | -0.46 dB / -0.003 | 3 / 12 |
| Wan 2.2 (high-noise expert) | 30.77 / 0.907 / 0.092 (19.88) | 30.35 / 0.904 / 0.095 (19.95) | -0.41 dB / -0.003 | 4 / 12 |

Per-clip table: `wan_h3/multi/score.json`. On these means the rule passed (-0.46 and -0.41 dB
against -1 dB; -0.003 against -0.01), and H2's `targets.py` was kept at the time. The trade was
plain: on the twelve clips H2's set sits 0.4 to 0.5 dB further from bf16 than the old one and
closer on only 3 and 4 of 12, for 0.77x against 0.89x per step. At 81 frames only the single
clips exist (above). **The Wan 2.2 row is one expert's**, so it did not test the shipped
two-expert path: rerun with both experts (8i), the rule fails, and H6 reverted the default (8j).

### 8g. bf16 reproducibility (H3 step 4, Wan 2.1, 9 frames, `wan_h3/repro/`)

| tree | step medians | two runs |
|---|---|---|
| the tip (the neuron rotary op of `fcb0dca`) | 433.3 / 433.1 ms | differ: mp4 `fc0671ba…` / `aa2918aa…`, latents differ too |
| a scratch copy whose neuron rotary is the CPU op (the pre-`fcb0dca` law) | 438.3 / 439.9 ms | identical, mp4 `450d04ef…` = the pre-campaign bf16 clip of 4a |

The H3 sessions' bf16 pairs differ the same way (Wan 2.1 `992565ae…` / `5dcfdd2c…`, 36.01 dB
apart; Wan 2.2 36.42 dB), while HunyuanVideo's (7a) and HunyuanVideo-1.5's bf16 pairs are
bit-identical. So since `fcb0dca` the interleaved bf16 Wan clip is not reproducible run to run;
the CPU op gives it back for 1.4% of the step (439.1 against 433.2 ms, means). A finding of
`fcb0dca`, reported to the user, who chose reproducibility: H5 (8h). The mxfp8 defaults were
unaffected: they reproduce bit for bit (8e).

### 8h. H5: bf16 Wan's interleaved rotary back on the stride-2 law (`c7c2346`)

The user chose bf16 reproducibility over the 1.4% of 8g (2026-11-08). H5 adds
`difflet.ops.apply_rotary_emb_slices`, the interleaved law computed with stride-2 slices; a
backend without one of its own falls back to its `apply_rotary_emb`, and the neuron backend's is
the CPU backend's op (re-exported, the same function object). WanAttention's interleaved
self-attention calls it; the half-split path is unchanged, and HunyuanVideo keeps the neuron
`x @ P` op (its bf16 pairs are reproducible, 7a and 9g, and every HunyuanVideo number was
measured with it). CPU gate at `c7c2346`: 2103 passed, 111 skipped, 0 failed; the full CPU suite
there: 5046 passed, 266 skipped, 1 failed (`test_neuron_hv1_app::test_registry_builds_the_neuron_application`,
pre-existing on `main`), 26 min 40 s.

Device check (plan H5 step 6; `wan_h3/h5/`, `verify_h5.sh` on a snapshot of `c7c2346`, whose
default was still H2's set; Wan 2.1, tp4, 480x832x9):

| arm | step median | mp4 md5 | latents md5 |
|---|---|---|---|
| bf16, gen50a / gen50b | 440.0 / 439.6 ms | `450d04ef…` on both, the pre-campaign bf16 clip of 4a | `bf61d6d7…` on both, those of the CPU-op scratch pair of 8g |
| H2's set (half-split), gen50a | 330.3 ms | `da16a5d3…`, unchanged (8d, 8e) | |

The bf16 pair is bit-identical and equals the clip from before `fcb0dca`, and the half-split
clip did not move. Cost: 440 against 433 ms per bf16 step (1.6%).

### 8i. Wan 2.2 with both experts: H2's set fails the rule

H3's Wan 2.2 rows and its 12-clip test had run one expert (8e). Rerun with `--dual-expert`
(`run_wan22d.sh`: `h3d.sh` on the H3 snapshot of `1a8382f`, then the same 12 requests per arm;
`wan_h3/wan22d/`, `wan_h3/multi/wan22d/`), both experts quantized. Each expert's application
reorders its own self-attention, so H2's set logs the reorder line twice (directories
`transformer/mxfp8-0a076ca2174a047e` and `transformer_2/mxfp8-df02091d6d56dbe4`). The first
81-frame attempt and the first 12-clip batches died in the 06:14 full-disk window
(`failed_diskfull/`) and were rerun.

| tp4, 480x832, both experts | bf16 | old default (`917d087` set) | H2's set |
|---|---|---|---|
| 9 frames, step medians (a / b) | 432.7 / 432.2 ms | 390.2 / 389.8 ms (0.90x) | **329.0 / 330.8 ms (0.76x)** |
| 81 frames, step median | 5079.7 ms | 4984.5 ms (0.98x) | **4730.3 ms (0.93x)** |
| 9 frames, mp4 md5 a / b | `a2e53498…` / `84872d01…` (differ: before H5) | `775c8a22…` on both (section 5's clip) | `10010cf9…` on both |
| single clip vs bf16, 9 frames (PSNR / SSIM) | - | 35.24 / 0.957 | 29.78 / 0.918 |
| single clip vs bf16, 81 frames | - | 20.02 / 0.593 | 25.28 / 0.796 |

| 12 clips, 9 frames, mean vs bf16 | old default | H2's set | H2 - old | clips where H2 is closer |
|---|---|---|---|---|
| **Wan 2.2, both experts** (`multi/score_wan22d.json`) | 29.04 dB / 0.895 / LPIPS 0.084 | 27.60 / 0.860 / 0.106 | **-1.44 dB / -0.034** | 3 / 12 |
| Wan 2.1 (8f) | 32.11 / 0.934 / 0.042 | 31.65 / 0.931 / 0.045 | -0.46 dB / -0.003 | 3 / 12 |

The worst clip for H2's set is prompt 1 ("a red fox running through a snowy forest, cinematic"),
seed 1234: 28.50 dB / SSIM 0.924 with the old default, 17.90 dB / 0.562 with H2's set. H3 step
5's rule on the means (not below the old default by more than 1 dB in PSNR or 0.01 in SSIM)
**fails for Wan 2.2 on both thresholds**. The old default's two-expert 9-frame clip at `1a8382f`
is section 5's (`775c8a22…`, from `917d087`) bit for bit.

### 8j. H6: the default goes back to the `917d087` set (`90c648f`)

One target table serves Wan 2.1 and Wan 2.2, and Wan 2.2 failed, so H6 reverted H2's
`targets.py` change for "wan" (Wan 2.1 alone had passed at -0.46 dB). `WAN_TARGETS` is the
`917d087` set again, in its order (`attn2.(to_q|to_k|to_v)`, `ffn.net.0.proj`, `ffn.net.2`), so
the default spec's digest is `917d087`'s and the default resolves the old directories (Wan 2.1
`mxfp8-6d28df7cc78386b5`; Wan 2.2 section 5's `transformer/mxfp8-372c52a68c440756` and
`transformer_2/mxfp8-03f6d63f26aab3cf`, 8k). `WAN_KEPT_BF16` lists `attn1.(to_q|to_k|to_v)`
again, and the table comment keeps phase H's cause and fix beside the two-expert numbers. H1 and
the application rule stay: a `DIFFLET_QUANT_TARGETS` override that quantizes a self-attention q
or k still gets the half-split layout and its reorder line, at 0.76x to 0.77x per 9-frame step.
The review fix `19b0aa9` runs the tp2 Wan test on the shipped default (interleaved) too and makes
the `--quant` help and the Wan application docstring name the bf16 self-attention q/k/v.

### 8k. The final tree's reruns (`90c648f`, `wan_h3/final/`)

`final_reruns.sh` on a snapshot of `90c648f` (H5's stride-2 rotary, H6's default), tp4,
480x832x9, no override, two generates per arm, as references for phase G:

| arm | step medians | vs bf16 | mp4 md5, a = b | latents md5, a = b | quantized directories |
|---|---|---|---|---|---|
| Wan 2.1, default (`wan21_mx`) | **393.1 / 393.5 ms** | **0.89x** (bf16 440.0 ms at `c7c2346`, 8h) | `a48ead02…` yes | `3b49d32c…` yes | `mxfp8-6d28df7cc78386b5` |
| Wan 2.2, both experts, bf16 (`wan22d_bf16`) | 439.6 / 439.4 ms | - | `68419054…` yes | `c1fec867…` yes | - |
| Wan 2.2, both experts, default (`wan22d_mx`) | **391.8 / 391.0 ms** | **0.89x** (of 439.5) | `90335e77…` yes | `96eece69…` yes | `transformer/mxfp8-372c52a68c440756`, `transformer_2/mxfp8-03f6d63f26aab3cf` |

Every pair is bit-identical: H5 makes Wan 2.2's two-expert bf16 reproducible too (at `1a8382f`
its pair differed, 8i). Neither default logs a reorder line (the self-attention runs bf16,
interleaved). Before H5 the same set ran 385.5 / 387.1 ms for Wan 2.1 (8e) and 390.2 / 389.8 ms
for Wan 2.2 with both experts (8i), with `x @ P` on the bf16 self-attention. The 81-frame step
was not rerun on this tree.

### 8l. Summary for Wan (tp4, 480x832, 50 steps, seed 42)

**What ships** (`90c648f`: cross-attention q/k/v + FFN; self-attention q/k/v bf16, interleaved):

| | 9 frames | 81 frames |
|---|---|---|
| Wan 2.1 step | **393.1 ms against 440.0 = 0.89x** (final tree, 8k) | 4956.0 against 5052.9 ms = 0.98x (before H5, 8e) |
| Wan 2.2 step, both experts | **391.8 ms against 439.6 = 0.89x** (final tree, 8k) | 4984.5 against 5079.7 ms = 0.98x (before H5, 8i) |
| vs bf16 (before H5) | 12 clips: 32.11 dB / 0.934 / 0.042 (Wan 2.1), 29.04 / 0.895 / 0.084 (Wan 2.2, both experts) | single clip: 28.83 / 0.853 (Wan 2.1), 20.02 / 0.593 (Wan 2.2, both experts) |
| per-rank DiT weights | 7.50 -> 5.36 GB (Wan 2.1, 4m); both Wan 2.2 experts 15.0 -> 10.7 GB (5) | |

**H2's set as an override** (+ self-attention q/k/v, half-split rotary; measured before H5):

| | 9 frames | 81 frames |
|---|---|---|
| Wan 2.1 step | 331.6 ms = 0.77x (8e) | 4731.2 ms = 0.94x (8e) |
| Wan 2.2 step, both experts | 329.0 ms = 0.76x (8i) | 4730.3 ms = 0.93x (8i) |
| vs bf16 | 12 clips: 31.65 / 0.931 / 0.045 (Wan 2.1), 27.60 / 0.860 / 0.106 (Wan 2.2, both experts) | single clip: 28.12 / 0.831 (Wan 2.1), 25.28 / 0.796 (Wan 2.2, both experts) |
| per-rank DiT weights | 7.50 -> 4.60 GB (Wan 2.1, 8e) | |

The 5x self-attention block of section 4 was the rotary's element-wise DMAs on the kernel's q/k,
and the half-split layout removes it; but with both Wan 2.2 experts the quantized self-attention
q/k/v move the clips 1.44 dB / SSIM 0.034 further from bf16 than the old set does, past the rule,
so they are an override and the default is the `917d087` set again. bf16 Wan is reproducible
again (H5). Against the CPU fp32 reference at 9 frames (section 10, on 8k's final-tree reruns)
the default is 2.36 dB (Wan 2.1) and 2.93 dB (Wan 2.2, both experts) further from fp32 than bf16,
itself 36.08 and 36.39 dB from fp32, about Wan 2.1's 37.0 dB TP floor: a measured cost. Open: the
81-frame quality (no fp32 reference) and step on the final tree (not rerun); and the DP router bug of 7e (it ignores an inherited
`NEURON_RT_VISIBLE_CORES`; 8f's and 8i's batches ran from a scratch copy around it).

## 9. HunyuanVideo-1.5 on trn3, tp2 x CFG2, 480x848x121, 50 steps, seed 42

`/home/ubuntu/fp8-bench/hv15/`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` at
snapshot `286be7ce` (an fp32 hub DiT: `compile` makes the bf16 copy under the cache's
`weights-bf16/`, and every quantized directory is written from it), prompt "a cat walking through
a garden", true CFG 6.0, the driver's default layout tp2 x CFG2 (4 cores), cache
`/home/ubuntu/difflet-cache-video`. Plan: `docs/plans/2026-11-07-mxfp8-hunyuan-video-15.md`
(phase F, Task F6). Three source trees: `d1af3e1` (`PRE_D4`, D4's parent, so diffusers' rotary at
HunyuanVideo-1.5's call site: the arm `bf16_pre`), `src-head` (`41d0685`, D4's neuron rotary
`71f3986` included: the arm `bf16`) and `src-f` (`73f1722`: every other arm, F1-F5 included).
`41d0685` predates F5's memory lines, so the bf16 memory column is `bf16f`, one generate of bf16
from `src-f` whose mp4 and latents equal `bf16`'s, and the 121-frame ratios are against `bf16f`'s
10241 ms (the tree of every MXFP8 arm; `bf16`'s two medians average 10236.5 ms). The gates are
F6's: S, a step median at or below 1.01x bf16's at the same shape and layout; Q, the
floor-relative screen of 9d.

### 9a. Kernel probe at the per-rank shapes (Step 0b, one core)

`probe/probe_mx_shapes_hv15.py`: section 1's probe at HunyuanVideo-1.5's tp2 per-rank shapes, with
a bf16 arm per shape (an `nn.Linear` holding the dequantized weight, compiled and timed the same
way). The first eight shapes are in `probe.log`; Dynamo's recompile limit
(`FailOnRecompileLimitHit` under `fullgraph=True`) stopped that run before the ninth, which ran
alone (`probe_9th.log`). Cosine in float64:

| layer (per rank, tp2) | M x K -> N | MX | bf16 | ratio | rel L2 | cosine |
|---|---|---|---|---|---|---|
| image q/k/v (`attn.to_q/k/v`, column) | 49290 x 2048 -> 1024 | 2.79 ms | 3.21 ms | 0.868 | 2.345e-3 | 0.999997 |
| image MLP up (`ff.net.0.proj`, column) | 49290 x 2048 -> 4096 | 6.21 ms | 7.79 ms | 0.797 | 2.358e-3 | 0.999997 |
| image MLP down (`ff.net.2`, row, K = 8192/2) | 49290 x 4096 -> 2048 | 5.44 ms | 8.15 ms | 0.668 | 2.327e-3 | 0.999997 |
| text q/k/v (`attn.add_q/k/v_proj`, column) | 1985 x 2048 -> 1024 | 0.341 ms | 0.827 ms | 0.413 | 2.347e-3 | 0.999997 |
| text MLP up (`ff_context.net.0.proj`, column) | 1985 x 2048 -> 4096 | 0.462 ms | 1.159 ms | 0.399 | 2.342e-3 | 0.999997 |
| text MLP down (`ff_context.net.2`, row) | 1985 x 4096 -> 2048 | 0.982 ms | 1.540 ms | 0.638 | 2.328e-3 | 0.999997 |
| CP2 image q/k/v | 24645 x 2048 -> 1024 | 1.762 ms | 2.868 ms | 0.614 | 2.346e-3 | 0.999997 |
| CP2 image MLP up | 24645 x 2048 -> 4096 | 3.843 ms | 6.246 ms | 0.615 | 2.343e-3 | 0.999997 |
| CP2 image MLP down | 24645 x 4096 -> 2048 | 3.540 ms | 6.195 ms | 0.572 | 2.325e-3 | 0.999997 |

Every shape passes (rel L2 < 3e-3, the two bf16 roundings of section 1a; cosine > 0.9999; output
`[M, N]`; no kernel assert), and every MX layer is faster than its bf16 twin, 0.40 to 0.87: the
probe predicts no group for Step 7 to drop. The 49,290-, 1,985- and 24,645-row activations run
through B7's 8-row pad; no arm below logged `NCC_INKI016`.

### 9b. Step times

The groups split the 540 linears of the first `HUNYUAN_VIDEO_15_TARGETS` (F1, `ee13e44`: three
patterns covering both streams) into four patterns by stream, as F6 Step 7 defines them
(`g1/targets.txt` to `g4/targets.txt`), with `B = transformer_blocks\.\d+` (54 dual-stream
blocks, no single-stream ones):

| group | pattern | layers |
|---|---|---|
| G1 | `B\.attn\.(to_q\|to_k\|to_v)` | image q/k/v (q/k RMSNorm, rotary, attention) |
| G2 | `B\.attn\.(add_q_proj\|add_k_proj\|add_v_proj)` | text q/k/v (1,985 rows) |
| G3 | `B\.ff\.net\.(0\.proj\|2)` | image MLP |
| G4 | `B\.ff_context\.net\.(0\.proj\|2)` | text MLP (1,985 rows) |

| arm | tree | layout, frames | quantized | step medians (a / b) | ratio | mp4 md5, a = b |
|---|---|---|---|---|---|---|
| `bf16_pre` | `d1af3e1` | tp2 x CFG2, 121 | - (diffusers' rotary) | 9924 / 9924 ms | 0.969x | `d3ef0f1c…` yes |
| `bf16` | `41d0685` | tp2 x CFG2, 121 | - | 10238 / 10235 ms | 1.000x | `a7c1ab92…` yes |
| `bf16f` | `73f1722` | tp2 x CFG2, 121 | - | 10241 ms (one generate) | the base | `a7c1ab92…` (= `bf16`) |
| `bf16_tp4` (the floor) | `73f1722` | tp4 x CFG2 (8 cores), 121 | - | 5135 / 5133 ms | - | `bc4056b1…` yes |
| `bf16_9` | `73f1722` | tp2 x CFG2, 9 | - | 557 / 559 ms | the 9-frame base | `9ab6e81c…` yes |
| `mx_9` | `73f1722` | tp2 x CFG2, 9 | G1-G4 | 1528 / 1519 ms | 2.74x / 2.72x | `95966326…` yes |
| `mx` | `73f1722` | tp2 x CFG2, 121 | G1-G4 | 21551 ms (one generate) | **2.10x** | - |
| `mx_final` | `73f1722` | tp2 x CFG2, 121 | G1, G3 | **10314 / 10313 ms** | **1.007x** | `daa1bcaa…` yes |
| `bf16_cp2` | `73f1722` | tp2 x CFG2 x CP2 (8 cores), 121 | - | 5687 / 5682 ms | the CP2 base | `5f269452…` yes |
| `mx_cp2` | `73f1722` | tp2 x CFG2 x CP2 (8 cores), 121 | G1, G3 | 10198 / 9539 ms | **1.79x / 1.68x** | `f6fb7081…` yes |

The 9-frame ratios are gen a against gen a and gen b against gen b, as are CP2's.

**D4 against its own rule.** Phase D's plan (`docs/plans/2026-11-07-mxfp8-proj-out-rope.md`,
"After phase D") keeps D4 only if `bf16`'s step is at most 1.01x `bf16_pre`'s and either the
latents are bit-identical or the clip PSNR against `bf16_pre` is at least the floor's. Here
`bf16` is 1.031x `bf16_pre` (10236.5 against 9924 ms), the latents differ, and the clip is
15.44 dB from `bf16_pre` against the floor's 15.90 (9d): D4 fails both conditions. It was not
reverted before Step 4, as that rule asks, so every arm from `bf16` to `mx_cp2` carries it.

**D4 reverted (Task F7, `dd3eed2`).** `nod4/` (`run_nod4.sh`, `run_nod4_121.sh`) ran a scratch
of `8514606` with `_joint_qkv` back on diffusers' `apply_rotary_emb(..., sequence_dim=1)`, the
committed default (no `DIFFLET_QUANT_TARGETS`, directory `mxfp8-afcbee695501d1d2`), one compile
and one generate per arm. Its executable code is the final tree's (`dd3eed2`'s `difflet/`
differs from the scratch's only in docstrings and a comment):

| arm | frames | quantized | step median | against | mp4 md5 |
|---|---|---|---|---|---|
| `nod4/bf16` | 9 | - | 557 ms | `bf16_9` 557 ms | `9ab6e81c…` (= `bf16_9`) |
| `nod4/mx` | 9 | G1, G3 | 518 ms | 0.930x `bf16_9` (`u3`, with D4: 511 ms) | `7a9ddae7…` (`u3`: `222a660d…`) |
| `nod4/mx121` | 121 | G1, G3 | 10032 ms | **1.011x** `bf16_pre`'s 9924 (`mx_final`, with D4: 10314) | `8da829b2…` |

Without D4 the default is 2.7% faster at 121 frames (10032 against 10314 ms), so D4 goes for it
too. bf16 at 121 frames was not rerun on the final tree; `bf16_pre` (`d1af3e1`) carries the same
rotary at this call site, so it is the base. Against it the default is 1.011x, over S (at or
below 1.01x) by 0.1%, and 108 ms above it where J is 99 ms (1% of 9924; `bf16_pre`'s two runs
are equal). `nod4/mx121`'s F5 lines: DiT weights 6.97 GiB resident (6.68 estimated), peak HBM
9.69 + 8.45 = 18.14 GiB, 0.48 GiB more allocator than `mx_final`; bf16's peak without D4 was
not measured (`d1af3e1` predates F5's lines); its clip against `bf16_pre` is in 9d.
`nod4/mx121` ran once, so its determinism is not measured. F7 Step 5, a 9-frame generate of the
default from the final tree (`0f75d11`, `hv15/f7/`), reproduced `nod4/mx`'s mp4 `7a9ddae7…` and
latents `aaadc89f…` (`f7/md5.txt`), so the measured no-D4 runs are the final tree's (its 644 ms
median ran under nice beside two CPU fp32 references: not a timing). The user's decision (9c) was
taken with the D4 numbers.

### 9c. Bisection (Step 7) and the default

`mx_9` failed S at 9 frames (2.74x) and `mx` at 121 (2.10x), so Step 7 ran at 9 frames against
`bf16_9` (gen a, 557 ms): one compile and one generate per arm, its groups in
`DIFFLET_QUANT_TARGETS` on both (`run_g.sh`; the directories in `g.summary`):

| arm | quantized groups | step median | vs bf16 | S |
|---|---|---|---|---|
| `g1` | G1 | 546 ms | 0.980x | passes |
| `g2` | G2 | 554 ms | 0.995x | passes |
| `g3` | G3 | 528 ms | 0.948x | passes |
| `g4` | G4 | 556 ms | 0.998x | passes |
| `mx_9` (the union `U` of the passing groups, Step 7(d)'s `u1`) | G1-G4 | 1528 ms | 2.743x | fails |
| `u2` (without G4, the highest `g` ratio) | G1, G2, G3 | 1514 ms | 2.718x | fails |
| **`u3`** (without G2, the next) | **G1, G3** | **511 ms** | **0.917x** | **passes** |

- **Outcome.** The slowdowns do not add up. Each group alone is level with bf16 or faster, G1 +
  G3 runs 0.917x, and adding the text q/k/v (G2) to it takes the step to 1514 ms. Every set
  measured at 9 frames that quantizes both the image q/k/v (G1) and the text q/k/v (G2) is 2.7x
  bf16's (`mx` at 121 frames: 2.10x), and every set without one of them is at speed: the
  slowdown needs the image and text q/k/v quantized together. G1 + G2 alone was not measured.
- **Not the kernels.** The probe (9a) has every one of these layers faster than bf16 (text q/k/v
  0.413, image q/k/v 0.868), so the cost is in the compiled block around the kernel calls. Not
  profiled for this model (issue #145 holds the hypothesis and the next steps, 9h).
- **`S` = G1 + G3**, the image stream's q/k/v and MLP.

**`mx_final` (Step 7(e)):** G1 + G3 at 121 frames under the same override (`run_final.sh`):
10314 / 10313 ms against `bf16f`'s 10241 (**1.007x**, passes S, with D4; the final tree's
1.011x is in 9b), bit-identical (`daa1bcaa…`), directory `mxfp8-afcbee695501d1d2`, the `u3`
directory (the spec carries no shape). Its clip fails Q on SSIM (9d), so by the plan Step 7
stopped there and reported.

**The decision (user, 2026-11-08, option A).** G1 + G3 becomes the default despite the nominal
SSIM miss, before phase G had measured its quality (since measured: at 9 frames 7.47 dB further
from fp32 than bf16, 10c; on 2026-11-09 the user decided that HunyuanVideo-1.5 has no default, 10e). Committed as `9f6aedc`: `HUNYUAN_VIDEO_15_TARGETS` holds G1
and G3 in G order, each written to render exactly the string `mx_final` carried (270 linears,
2.49B of the 8.33B parameters); G2 and G4 move to `HUNYUAN_VIDEO_15_KEPT_BF16`, and the table
comment records the measurement and that the plan never makes a set failing the screen the
default, this one being the user's decision. CPU gate at `c7c2346` (which carries `9f6aedc`):
2103 passed, 111 skipped, 0 failed.

**The default-directory check (Step 7(f)).** From a snapshot of `c7c2346` with no
`DIFFLET_QUANT_TARGETS`, `compile --quant mxfp8` at the default shape and layout exits 0, and its
MXFP8 line names `mxfp8-afcbee695501d1d2`, `mx_final`'s directory (`default_check.log`; DiT
weights per rank 6.97 GiB resident, 6.68 estimated at load).

**Step 7(g).** The `g1`-`g4` and `u2` directories (`mxfp8-f18733aab4ff51f0`,
`mxfp8-0c8429257f6add31`, `mxfp8-8a37195572ac193b`, `mxfp8-075f68c64f6b4be3`,
`mxfp8-7acb5d079338888a`) are deleted; the model's quantized `transformer/` holds
`mxfp8-afcbee695501d1d2` (the default's) and `mxfp8-3495b42101abbf8d` (G1-G4, `mx_9`'s, 12 GB by
`du` at Step 5), kept because the disk was not tight (282 GB free).

### 9d. Quality against bf16 (121 frames unless noted)

`compare_difflet_runs.py`, gen a against the reference's gen a (`bf16` unless noted):

| pair | PSNR (mean / worst frame) | SSIM | LPIPS | latents cosine | Q |
|---|---|---|---|---|---|
| `bf16_tp4` vs `bf16`: the floor | 15.90 / 15.31 dB | 0.312 | 0.363 | 0.911 | - |
| `bf16f` (D4's rotary) vs `bf16_pre` | 15.44 / 14.68 dB | 0.298 | 0.384 | 0.897 | - |
| `mx_9` vs `bf16_9` (9 frames) | 17.19 / 16.61 dB | 0.427 | 0.231 | 0.955 | not gated (no 9-frame floor) |
| `mx` (G1-G4) | 13.26 / 12.65 dB | 0.2117 | 0.483 | 0.818 | fails SSIM |
| `mx_final` (G1, G3, with D4) | 12.76 / 12.24 dB | 0.190 | 0.510 | 0.774 | fails SSIM |
| **`nod4/mx121` (G1, G3, the final tree) vs `bf16_pre`** | **12.65 / 12.20 dB** | **0.183** | **0.504** | **0.776** | **fails SSIM** |

Q at this floor (`gate.json`): PSNR >= min(30, 15.90 - 4) = **11.90 dB**, SSIM >= min(0.97,
0.312 - 0.10) = **0.212** (0.2125 unrounded), LPIPS <= max(0.05, 2 x 0.363) = **0.727**.
`mx_final` clears PSNR and LPIPS and misses SSIM (0.190 against 0.212); `mx` misses it by less
(0.2117 against 0.2125). On the final tree (9b) the default misses SSIM too (0.183 against
`bf16_pre`, `nod4/mx121_vs_bf16_pre.json`), screened at the same floor: no bf16 floor pair was
run without D4.

The floor is what makes these numbers unreadable: two bf16 runs that differ only in their TP
degree are 15.90 dB / SSIM 0.312 apart, and D4's rotary, bit-identical on the CPU, moves the
device clip about as far (15.44 dB), as the neuron rotary op moved Wan's bf16 clip in 4i. On
`montage_bf16_tp_rotary_mx.png` the three bf16 clips share their composition, and the MXFP8 clip
on it, `mx` (G1-G4; `mx_final`'s was not put on it), is a coherent clip of the prompt in a
different garden: the 121-frame, 50-step trajectory is
chaotic, single-clip metrics cannot judge it, and the screen degenerates at this floor (11.90 dB
/ 0.212 / 0.727). Against the CPU fp32 reference, which phase G ran at 9 frames (section 10), the
default is 7.47 dB / SSIM 0.399 further from fp32 than bf16; 121 frames has no fp32 reference.

### 9e. Memory (rank 0, the transformer stage's F5 lines)

| rank 0, 121 frames | `bf16f`, tp2 x CFG2 | **`mx_final`, tp2 x CFG2** | `bf16_cp2` | `mx_cp2` |
|---|---|---|---|---|
| DiT weights per rank, resident (estimated at load) | 7.80 (7.80) GiB | **6.97 (6.68) GiB** | 7.80 (7.80) GiB | 6.97 (6.68) GiB |
| peak HBM, allocator + runtime | 10.04 + 7.52 = 17.56 GiB | **9.21 + 8.45 = 17.66 GiB** | 9.60 + 4.40 = 14.00 GiB | 8.77 + 4.57 = 13.34 GiB |
| of the runtime: scratchpad / DMA rings | 1.75 / 5.34 GiB | 2.69 / 5.34 GiB | 1.31 / 2.80 GiB | 1.50 / 2.80 GiB |
| per-core HBM, free at the peak | 36.0 GiB, 18.44 free | 36.0 GiB, 18.35 free | 36.0 GiB, 22.00 free | 36.0 GiB, 22.66 free |

- **At tp2 x CFG2 the peak is the same.** The DiT weights shrink by 0.83 GiB per rank and the
  runtime's scratchpad grows by 0.94 GiB: 17.66 against 17.56 GiB. Under CP2 the scratchpad grows
  by 0.19 GiB and the peak drops by 0.66 GiB (13.34 against 14.00).
- The estimate at load omits the kernel's `[K/4, N]` scale layout (the plan's memory arithmetic):
  0.29 GiB per rank here. CP replicates the weights, so `mx_cp2` holds `mx_final`'s bytes.
- With every group (`mx`, G1-G4) the weights are 6.14 GiB resident (5.56 estimated; the plan
  expected about 6.1) and the peak 8.37 + 8.95 = 17.32 GiB, with a 3.19 GiB scratchpad.
- On disk, the G1-G4 directory is 12 GB; the sizes of the G1 + G3 directory and of the bf16
  copy were not recorded. neuron-monitor's per-core maximum over the `transformer` samples
  (`resources.jsonl` of the sampled a-runs) is not read out here.

### 9f. CP2 (Step 8)

`--tp-degree 2 --cfg-parallel --cp-degree 2` (8 cores, Ulysses; 24,645 image rows per rank,
padded to 24,648), `run_cp2_best.sh`. `mx_cp2` ran before `9f6aedc`, with
`DIFFLET_QUANT_TARGETS` equal to the default's spec (G1 + G3), so its MXFP8 line names the
default's directory `mxfp8-afcbee695501d1d2`:

| | `bf16_cp2` | `mx_cp2` (G1, G3) |
|---|---|---|
| step medians (a / b) | 5687 / 5682 ms | 10198 / 9539 ms |
| ratio (a / a, b / b) | - | **1.79x / 1.68x** |
| determinism | mp4 `5f269452…`, latents `38ba1d1f…` on both runs | mp4 `f6fb7081…`, latents `e2233d44…` on both runs |

Under CP2 (with D4) the default is 1.7-1.8x slower than bf16, where at tp2 x CFG2 it is level
(1.007x) and
the probe put the CP2 shapes' kernels at 0.57-0.62x bf16's. `mx_cp2`'s two medians differ by 7%
with identical outputs: the host was busy (workflow test runs, other compiles). The plan's
comparisons of `mx_cp2` against `bf16`, `mx_final` and `bf16_cp2` were not run.

### 9g. Determinism

Every arm with two generates reproduced its mp4 and its latents:

| arm | mp4 md5 | latents md5 |
|---|---|---|
| `bf16_pre` | `d3ef0f1c…` | `ff7ee4e5…` |
| `bf16` | `a7c1ab92…` | `6d9b0fdc…` |
| `bf16_tp4` | `bc4056b1…` | `c42f683e…` |
| `bf16_9` | `9ab6e81c…` | `1dbecba1…` |
| `mx_9` | `95966326…` | `33c1476a…` |
| `mx_final` | `daa1bcaa…` | `f6b46fa5…` |
| `bf16_cp2` | `5f269452…` | `38ba1d1f…` |
| `mx_cp2` | `f6fb7081…` | `e2233d44…` |

`bf16f`'s one generate from `73f1722` equals `bf16`'s from `41d0685` in both. Unlike Wan before
H5 (8g), HunyuanVideo-1.5's bf16 is reproducible run to run.

### 9h. Summary for HunyuanVideo-1.5

| tp2 x CFG2 unless noted, 480x848x121 | DiT step | vs bf16 | DiT weights per rank | peak HBM, rank 0 |
|---|---|---|---|---|
| **bf16, the final tree's rotary (`bf16_pre`)** | **9924 ms** | - | 7.80 GiB | not measured |
| **mxfp8, default, the final tree (D4 reverted, `nod4/mx121`)** | **10032 ms** | **1.011x** | **6.97 GiB** | **18.14 GiB** |
| bf16, with D4 | 10241 ms | - | 7.80 GiB | 17.56 GiB |
| mxfp8, every group (G1-G4), with D4 | 21551 ms | 2.10x | 6.14 GiB | 17.32 GiB |
| mxfp8, default (G1 + G3, `9f6aedc`), with D4 | 10314 ms | 1.007x | 6.97 GiB | 17.66 GiB |
| bf16, tp2 x CFG2 x CP2 (8 cores), with D4 | 5687 ms | - | 7.80 GiB | 14.00 GiB |
| mxfp8 default, tp2 x CFG2 x CP2, with D4 | 10198 / 9539 ms | 1.79x / 1.68x | 6.97 GiB | 13.34 GiB |

Today the default buys HunyuanVideo-1.5 0.83 GiB of resident DiT weights per rank. At tp2 x
CFG2 it buys nothing else: with D4 the step was level with bf16 (1.007x) at the same peak HBM
(17.66 against 17.56 GiB); on the final tree, without D4, it is 1.011x `bf16_pre`'s, just over
S, and peaks at 18.14 GiB (bf16's peak without D4 not measured). Under CP2 the peak drops
0.66 GiB (13.34 against 14.00 GiB), at 1.7-1.8x the step. With every group the weights shrink by
1.66 GiB and the peak by 0.24 GiB (17.32 against 17.56 GiB), at 2.1x the step. Quality:
12.65 dB / SSIM 0.183 from `bf16_pre` on the final tree (with D4 12.76 dB / SSIM 0.190 from
bf16) against a chaotic floor of 15.90 / 0.312; against the CPU fp32 reference at 9 frames
(section 10) the default is 17.41 dB / SSIM 0.446 where bf16 is 24.89 / 0.846, 7.47 dB further,
and the image MLP alone 4.69 dB further. The default is the user's decision (option A,
2026-11-08), taken with the D4 numbers and before phase G; on 2026-11-09, with phase G's
numbers, the user withdrew it: HunyuanVideo-1.5 has no MXFP8 default (10e). Issue
[#145](https://github.com/ai-decentralized/Difflet/issues/145) tracks the problem ("quantizing
image and text q/k/v together makes the DiT step 2.7x slower; the best set gives neither speed
nor peak-HBM savings") with its next steps: profile the block NEFF with the recipe of 8a, explain
the scratchpad growth, judge quality on the CPU fp32 reference (done at 9 frames, section 10; not
yet posted to the issue) or on many samples. D4 failed phase D's keep rule and F7 reverted it
(`dd3eed2`, 9b).

## 10. CPU fp32 reference (phase G)

Plan: `docs/plans/2026-11-08-cpu-fp32-reference.md` (Tasks G1-G11). Scripts: `scripts/fp32_ref/`
(`2e83328`: the shared helper and the scorer; `3821d3e`: one reference script per model;
`0f75d11`: the scorer's HunyuanVideo-1.5 MXFP8 file). The per-model research (device settings with
their sources, device files, latents layouts, noise checks) is
`/home/ubuntu/fp8-bench/phase_g_research.json`. Raw files: `/home/ubuntu/fp8-bench/fp32_ref/<key>/`
(`latents.pt`, `out.png` or `out.mp4` with the lossless frames `out.pt`, `run.json`, `run.log`
under `/usr/bin/time -v`, `score.json`, `score.log`) and the runner's log `fp32_ref/progress.txt`.
Each reference is diffusers in fp32 on the host CPU at the settings of the device runs it scores,
from their initial noise: only the dtype and the device differ. MXFP8 is read against bf16's own
distance from fp32.

### 10a. What ran

- **Software and math.** The device runs' venv (`native-fp8`: diffusers 0.38.0, torch
  2.14.0+cu130), every component loaded in fp32 (the bf16 hub checkpoints upcast exactly; the fp32
  DiTs of Wan and HunyuanVideo-1.5 as stored), torch's CPU capability `AVX512` (in every `run.json`
  but Wan's, which do not record it). The host has AMX-BF16, so each script
  first calls `require_fp32_math`, which refuses a oneDNN fp32 precision other than `none` or
  `ieee` and a default fpmath mode other than strict. `TORCH_DEVICE_BACKEND_AUTOLOAD=0`,
  `DIFFLET_BACKEND=cpu` and `HF_HUB_OFFLINE=1`: torch_neuronx is never imported and no Neuron
  device is touched.
- **Settings.** Each model's device settings from its research object (table below): snapshot,
  prompt, negative prompt, guidance, shape, 50 steps, seed 42, the snapshot's scheduler, text
  length, fps. Where the device run differs from a diffusers default the script passes the
  device's value (LTX-2's guidance 3.5 and 50 steps, for instance, against diffusers' 4.0 and 40).
  The decoded output is written as the device wrote it (PNG; MP4 through the device's writer at
  its fps, plus the lossless frames), the two HunyuanVideo decoders tile as the device decoders do
  (256 pixels and 16 frames; 128 pixels), and `latents.pt` holds the final latents in the
  device's layout.
- **Noise.** The device's initial noise, bit for bit. FLUX, Qwen-Image, LTX-2 (video and audio)
  and HunyuanVideo-1.5 draw it in bf16 on a CPU generator, which is diffusers' fp32 seed-42 draw
  rounded to bf16 (FLUX's: rel L2 1.7e-3 from the unrounded draw; the research checked the law for
  each of the four), so
  the reference injects that bf16 draw upcast, which is exact. Wan 2.1, Wan 2.2 and HunyuanVideo
  draw it in fp32 exactly as diffusers does. FLUX, LTX-2, HunyuanVideo and HunyuanVideo-1.5
  assert the md5 of what they inject against the research's (LTX-2 also asserts `torch.equal`
  against difflet's own draw); Qwen-Image and the two Wan scripts assert `torch.equal` against
  difflet's own draw (the orchestrator's `prepare_latents`) and record its md5 in `run.json`
  without checking it.
- **Shapes.** The short ones: the device shape for FLUX, Qwen-Image and LTX-2, HunyuanVideo's
  320x512x61, Wan 2.1 and Wan 2.2 at 9 frames, HunyuanVideo-1.5 at 9 frames (its default is 121).
  The long shapes take 5-16 h each on this CPU and are out of the plan's scope (10f).

| model (`key`) | snapshot | shape | prompt | guidance | noise injected (shape, md5) |
|---|---|---|---|---|---|
| FLUX.1-dev (`flux`) | `3de623fc` | 1024x1024 | "a cat sitting on a bench" | embedded 3.5, no true CFG; T5 at 512 tokens | bf16 draw (1, 16, 128, 128), packed (1, 4096, 64), `be1ad0a6…` |
| Qwen-Image (`qwen_image`) | `75e0b4be` | 1024x1024 | "a cat sitting on a bench" | none (no guidance embedder, no true CFG) | bf16 draw (1, 1, 16, 128, 128), packed (1, 4096, 64), `be1ad0a6…` (FLUX's bytes) |
| LTX-2 (`ltx_2`) | `64f65d09` | 512x768x121, 24 fps | "a cat walking through a garden" | true CFG 3.5 on video and audio, negative ""; Gemma-3 at 1024 tokens | bf16 draws: video (1, 128, 16, 16, 24) `d4d72668…`, audio (1, 8, 126, 16) `acdb384e…` |
| Wan 2.1 (`wan21`) | `38ec498c` | 480x832x9, 16 fps | "a cat walking through a garden" | 1.0 (no CFG), flow shift 3.0; umT5 at 512 tokens | fp32 draw (1, 16, 3, 60, 104), `2f6a7d06…` |
| Wan 2.2-A14B (`wan22`) | `5be7df96` | 480x832x9, 16 fps | "a cat walking through a garden" | 1.0 on both experts, boundary 0.875: 16 high-noise and 34 low-noise forwards | fp32 draw, `2f6a7d06…` (Wan 2.1's) |
| HunyuanVideo (`hunyuan_video`) | `e8c2aaa6` | 320x512x61, 24 fps | "a cinematic shot of a red fox running through a snowy forest" | embedded 6.0, no true CFG; Llama at 256 tokens | fp32 draw (1, 16, 16, 40, 64), `fa10d605…` |
| HunyuanVideo-1.5 (`hunyuan_video_15`) | `286be7ce` | 480x848x9, 24 fps | "a cat walking through a garden" | true CFG 6.0 (the snapshot's guider), negative "" | bf16 draw (1, 32, 3, 30, 53), `aad07dc0…` |

**The runner** (`/home/ubuntu/fp8-bench/fp32_ref/run_queue.sh`): two workers pull the model keys
from one queue, fastest first (the plan's order), worker 0 on NUMA node 0 (`taskset -c
0-47,96-143`) and worker 1 on node 1 (`taskset -c 48-95,144-191`), 96 threads each; each reference
runs under `/usr/bin/time -v` and is scored by `score.py` as soon as it exits. The source is a
snapshot of `3821d3e` (`fp32_ref/src`, `src.sha`) with one file patched: `score.py`'s
HunyuanVideo-1.5 MXFP8 entry names `hv15/nod4/mx_gena.mp4` instead of `hv15/u3/gena.mp4` (10b),
the change the repo took in `0f75d11` (`diff -rq` against `3821d3e`'s `scripts/` and `difflet/`:
that file only; against `0f75d11`'s `score.py`: its comment only).

| model | worker | started | done | wall (`time -v`) | peak RSS | step median |
|---|---|---|---|---|---|---|
| FLUX.1-dev | 0 | 09:55:40 | 10:38:05 | 42:24 | 67.99 GiB | 50.72 s (n=49, steps 2-50) |
| Qwen-Image | 1 | 09:55:42 | 10:38:42 | 42:59 | 110.25 GiB | 47.41 s (n=49, steps 2-50) |
| Wan 2.1 | 0 | 10:38:20 | 11:23:01 | 44:41 | 80.20 GiB | 51.49 s (n=50) |
| HunyuanVideo-1.5 | 1 | 10:38:58 | 11:35:06 | 56:08 | 63.60 GiB | 60.72 s (n=50) |
| Wan 2.2-A14B | 0 | 11:23:26 | 12:13:19 | 49:53 | 137.35 GiB | 51.64 s (n=50; the 17th, the low-noise expert's first, 434.2 s) |
| HunyuanVideo | 1 | 11:35:32 | 12:50:49 | 1:15:17 | 84.10 GiB | 86.39 s (n=50) |
| LTX-2 | 0 | 12:13:44 | 14:08:08 | 1:54:23 | 172.43 GiB | 135.68 s (n=50) |

Every reference exited 0. Peak RSS is `time -v`'s maximum resident set size (in KiB) converted to
GiB. Scoring took 15 to 267 s per model (LTX-2's 121 frames the longest); the last row was written
at 14:12:35. The step times are this host's with two references at once and say nothing about the
device.

### 10b. The device files scored

`score.py`'s `DEVICE_OUTPUTS`, gen a of each arm (paths under `/home/ubuntu/fp8-bench/`):

| model | bf16 | MXFP8, the shipped default | latents |
|---|---|---|---|
| FLUX.1-dev | `flux_d_tp4/A1_gena.png` `67da9e98…` (6a) | `flux_d_tp4/B1_gena.png` `7b6a23dc…` (= `3d7dd6c`'s default check, 6b) | none: tp4 decodes in process, so the image only |
| Qwen-Image | `qwen_bf16_tp4_ref2/gen50a.png` `5478687f…` (the bf16 reference rebased after the text-mask change `c9f4009`) | `qwen_mx_hm0_tp4/gen50a.png` `7737239b…` (reproduced after phase A, section 0) | both |
| LTX-2 | `ltx2_tp4/bf16_gen50a.mp4` `0702ffd2…` (2) | `ltx2_tp4/mx_gen50a.mp4` `c1057ed5…` (3) | both, video and audio |
| Wan 2.1 | `wan_h3/h5/bf16_gen50a.mp4` `450d04ef…` (`c7c2346`, 8h) | `wan_h3/final/wan21_mx/gen50a.mp4` `a48ead02…` (`90c648f`, 8k) | both |
| Wan 2.2-A14B, both experts | `wan_h3/final/wan22d_bf16/gen50a.mp4` `68419054…` (`90c648f`, 8k) | `wan_h3/final/wan22d_mx/gen50a.mp4` `90335e77…` (`90c648f`, 8k) | both |
| HunyuanVideo | `hv1_tp4/bf16_gen50a.mp4` `75e6ba8d…` (7a) | `hv1_tp4/loo_V15_gen50a.mp4` `7d54b413…` (V15 = `fc38fb6`'s default, 7b) | both |
| HunyuanVideo-1.5, 9 frames | `hv15/bf16_9/gena.mp4` `9ab6e81c…` (= `nod4/bf16`, 9b) | `hv15/nod4/mx_gena.mp4` `7a9ddae7…` (the default without D4, 9b) | both |

- **Wan's final-tree reruns.** The research found no run of the shipped Wan defaults on the final
  tree: Wan 2.1's MXFP8 default had run at `1a8382f`, where the bf16 self-attention used the
  `x @ P` rotary, and both Wan 2.2 arms had run at `1a8382f`, before H5. So the Wan files are
  8k's reruns from `90c648f`; Wan 2.1's bf16 clip is H5's (`c7c2346`), whose `difflet/` matches the
  final tree's except H6's `targets.py`, which bf16 does not read.
- **HunyuanVideo-1.5's no-D4 default.** The default ships on the tree without D4 (F7, `dd3eed2`),
  so its file is `nod4/mx`'s (`7a9ddae7…`), not `u3`'s (`222a660d…`: the same set and directory
  with D4's rotary, another clip). F7 Step 5 reproduced it from the final tree (`0f75d11`,
  `hv15/f7/`): mp4 `7a9ddae7…` and latents `aaadc89f…`, both `nod4/mx`'s. The bf16 clip is one on
  both trees (`nod4/bf16` = `bf16_9`).
- The other files are the shipped defaults' by the research's code check (nothing on the model's
  output path changed after the run): FLUX's since `3d7dd6c`, Qwen-Image's since phase A, LTX-2's
  since `fc8b94c`, HunyuanVideo's since `fc38fb6`.

### 10c. Scores

`compare_difflet_runs.py` through `score.py`, the reference first: PSNR (mean over frames, the
worst frame in parentheses), SSIM and LPIPS on the decoded output, both sides through the same
encoder; cosine on the final latents (`score.json`). The last column is the device's own bf16
floor, bf16 against bf16 at another TP degree, from the sections named:

| model, shape | bf16 vs fp32: PSNR (worst) / SSIM / LPIPS / latents cos | MXFP8 vs fp32 | MXFP8 - bf16: PSNR / SSIM | device bf16 floor |
|---|---|---|---|---|
| Qwen-Image, 1024x1024 | 36.34 / 0.990 / 0.014 / 0.9973 | 37.36 / 0.992 / 0.009 / 0.9983 | **+1.02 dB** / +0.002 | 39-46 dB (`2026-10-31-qwen-image-fp8-native-trn3.md`) |
| LTX-2, 512x768x121 | 16.55 (15.13) / 0.472 / 0.195 / video 0.9352, audio 0.9843 | 16.85 (15.54) / 0.485 / 0.190 / video 0.9408, audio 0.9835 | **+0.29 dB** / +0.014 | 25.9 dB / 0.909 (3a) |
| HunyuanVideo, 320x512x61 | 20.79 (17.56) / 0.846 / 0.160 / 0.9819 | 20.59 (17.42) / 0.837 / 0.168 / 0.9809 | **-0.20 dB** / -0.008 | 36.64 dB / 0.957 (7a) |
| FLUX.1-dev, 1024x1024 (image only) | 23.36 / 0.917 / 0.075 / - | 21.30 / 0.885 / 0.139 / - | **-2.06 dB** / -0.032 | 29.4 dB on this image, 29.3 mean over 12 (6a, 6b) |
| Wan 2.1, 480x832x9 | 36.08 (34.73) / 0.952 / 0.026 / 0.9998 | 33.72 (32.13) / 0.947 / 0.033 / 0.9982 | **-2.36 dB** / -0.005 | 37.0 dB / 0.960 (4a) |
| Wan 2.2-A14B, both experts, 480x832x9 | 36.39 (35.89) / 0.962 / 0.021 / 0.9993 | 33.46 (30.97) / 0.949 / 0.036 / 0.9970 | **-2.93 dB** / -0.013 | not measured with both experts |
| HunyuanVideo-1.5, 480x848x9 | 24.89 (24.63) / 0.846 / 0.085 / 0.9946 | 17.41 (16.60) / 0.446 / 0.237 / 0.9565 | **-7.47 dB** / -0.399 | none at 9 frames (9d) |

### 10d. Reading

No numeric gate: the campaign plan's phase G line fixes the relative gate "from Qwen's pair
first", and Qwen-Image's MXFP8 image is the closer of its two (+1.02 dB), which sets no margin.
Each model is read by MXFP8's distance from fp32 against bf16's own, on one sample, with the
device's bf16 floor as the scale one sample can resolve:

- **Qwen-Image: no worse than bf16.** MXFP8 is 1.02 dB closer to fp32 than bf16 (SSIM 0.992
  against 0.990, latents cosine 0.9983 against 0.9973). The two device images are 38.26 dB apart
  (`qwen_mx_hm0_tp4/vs_bf16_ref2.json`), closer to each other than either is to fp32.
- **LTX-2: no worse than bf16; both far from fp32 at 121 frames.** bf16 is 16.55 dB / SSIM 0.472
  from fp32, 9.4 dB beyond its own TP floor (25.9 dB): over 121 frames and 50 steps the bf16
  trajectory leaves fp32's too. MXFP8 lands 0.29 dB closer (video latents cosine 0.9408 against
  0.9352; the audio stream, bf16 in both arms, 0.9835 against 0.9843), and the two device clips
  are 22.47 dB apart (3).
- **HunyuanVideo: no worse than bf16 within one clip; both far from fp32.** bf16 is 20.79 dB from
  fp32 against a 36.64 dB TP floor; MXFP8 is 0.20 dB / SSIM 0.008 further (latents cosine 0.9809
  against 0.9819), and the two device clips are 31.09 dB apart (7b). 544x960x129 has no fp32
  reference.
- **FLUX.1-dev: not resolved, one image of a chaotic sampler.** MXFP8 is 2.06 dB / SSIM 0.032
  further from fp32 than bf16 on the cat (21.30 against 23.36 dB). At 50 steps FLUX's composition
  drifts: two bf16 runs at different TP degrees are 29.3 dB apart on average over 12 images (19.5
  at the worst) and MXFP8 is 24.7 dB from bf16 on average (6b), so one image cannot separate 2 dB.
  The 12-image set was not referenced (one image took 42 min here), and the score is the image's
  alone (the tp4 runs kept no latents).
- **Wan 2.1 and Wan 2.2: a measured cost of 2.4 and 2.9 dB at 9 frames.** At 9 frames bf16 follows
  fp32 to about its own reduction-order noise (36.08 dB against Wan 2.1's 37.0 dB TP floor; Wan 2.2
  36.39 dB), so one clip resolves the quantization: the default is 2.36 dB (Wan 2.1) and 2.93 dB
  (Wan 2.2, both experts) further from fp32, at SSIM 0.947 / 0.949 and LPIPS 0.033 / 0.036 (bf16
  0.952 / 0.962 and 0.026 / 0.021). That is in line with the device screen's gap at 9 frames (4m,
  before H5: 33.0 dB from bf16 against the 37.0 dB floor). 81 frames has no fp32 reference, so
  the 81-frame clips stay with the device screen (4m, 5).
- **HunyuanVideo-1.5: a large measured cost at 9 frames; no default since (10e).** The
  then default (G1 + G3) is 7.47 dB / SSIM 0.399 further from fp32 than bf16 (17.41 dB
  / 0.446 against 24.89 / 0.846; latents cosine 0.9565 against 0.9946), the largest gap of the
  seven. 121 frames has no fp32 reference. 10e measures the two groups alone.

The pixel metrics include the decoder, which both device arms of a model share; the latents
cosine measures the DiT alone.

### 10e. HunyuanVideo-1.5: the single groups against fp32

After 10c, the default's two groups were measured alone on the final tree:
`hv15/run_g_final.sh` from the snapshot `hv15/f7/src` (by `diff -rq`, its `difflet/` equals
`0f75d11`'s), `DIFFLET_QUANT_TARGETS` the group's pattern of 9b on compile and generate, tp2 x
CFG2, 480x848x9, one generate each (mp4 `96edcac0…` for G3, `585aafde…` for G1), scored with
`compare_difflet_runs.py` against the same fp32 reference (`final_g3/vs_fp32.json`,
`final_g1/vs_fp32.json`):

| set | directory | step median (vs `bf16_9`'s 557 ms) | DiT weights per rank, resident | vs fp32: PSNR (worst) / SSIM / LPIPS / latents cos | MXFP8 - bf16: PSNR / SSIM |
|---|---|---|---|---|---|
| bf16 (`bf16_9`) | - | 557 ms | 7.80 GiB | 24.89 (24.63) / 0.846 / 0.085 / 0.9946 | - |
| **G3, the image MLP only** (`final_g3`) | `mxfp8-8a37195572ac193b` | 527 ms (0.946x) | 7.20 GiB | 20.20 (19.15) / 0.657 / 0.175 / 0.9796 | **-4.69 dB** / -0.189 |
| G1, the image q/k/v only (`final_g1`) | `mxfp8-f18733aab4ff51f0` | 532 ms (0.955x) | 7.58 GiB | 18.10 (17.34) / 0.505 / 0.213 / 0.9616 | -6.79 dB / -0.341 |
| G1 + G3, the committed default (`nod4/mx`) | `mxfp8-afcbee695501d1d2` | 518 ms (0.930x) | 6.97 GiB | 17.41 (16.60) / 0.446 / 0.237 / 0.9565 | -7.47 dB / -0.399 |

- Every MXFP8 set costs HunyuanVideo-1.5 4.7 to 7.5 dB against fp32 beyond bf16's distance. The
  image q/k/v cost the most (HunyuanVideo's default likewise leaves its image-token q/k/v bf16,
  7b); the image MLP alone costs the least, still more than either Wan model's default (2.4 and
  2.9 dB).
- The two single-group generates ran concurrently, from 12:55 to 12:57, while LTX-2's reference
  held NUMA node 0, so their step medians are one generate each on a loaded host; the default's
  518 ms ran at 09:18, before phase G. At 9 frames every set is faster than bf16; at 121 frames
  the default is 1.011x `bf16_pre`'s step (9b), and G3 alone was not run at 121 frames.

**The decision (user, 2026-11-09): option 2.** The measured options were:

1. **The image MLP alone (G3) as the default:** 4.69 dB beyond bf16's distance from fp32 instead
   of 7.47, 0.946x at 9 frames, 7.20 GiB of DiT weights per rank (the default 6.97, bf16 7.80);
   its 121-frame step and peak HBM are not measured.
2. **Withdraw HunyuanVideo-1.5 from `--quant`:** at 121 frames the current default buys neither
   speed (1.011x) nor peak HBM (18.14 GiB without D4; with D4 17.66 against bf16's 17.56).

HunyuanVideo-1.5 has no MXFP8 default (plan `docs/plans/2026-11-07-mxfp8-hunyuan-video-15.md`,
Task F8): `--quant mxfp8` without `DIFFLET_QUANT_TARGETS` is refused on the driver before
anything is written (exit 2), the four groups stay reachable as `HUNYUAN_VIDEO_15_GROUPS`, and
G1 + G3 through the override is the former default's spec (on host A, the same quantized
directory, `mxfp8-afcbee695501d1d2`; a copy-built directory's name is per host, 12c). The numbers above stay as measured; issue #145 is closed with this
decision.

### 10f. What phase G does not cover

- **The long shapes:** Wan 2.1 and Wan 2.2 at 480x832x81, HunyuanVideo at 544x960x129,
  HunyuanVideo-1.5 at 480x848x121 (5-16 h per reference on this CPU, out of the plan's scope).
  Their quality stays with the device screens (4m, 5, 7f, 9d).
- **More samples:** one prompt at seed 42 per model. FLUX's 12 images (6b) and Wan's 12 clips (8f,
  8i) have no fp32 references.
- **The CPU's own floor:** every reference ran once, at 96 threads; no second CPU run (at another
  thread count) was made, so the fp32 side's run-to-run spread is unmeasured.
- **FLUX's latents:** the tp4 runs kept none (FLUX has no latents dump hook; a device rerun with
  one would be needed).

## 11. Before and after quantization: compile, e2e cold and warm, DiT, HBM, quality

The user's question (2026-11-08): per model, bf16 against the shipped MXFP8 default for quality, the
DiT, compile time, e2e cold and e2e warm. Every model ran at its phase G shape and settings (10a's
table: snapshot, prompt, guidance, 50 steps, seed 42, LNC=2, tp4; HunyuanVideo-1.5 tp2 x CFG2 at
480x848x9), both arms on one device and one CPU set, one arm after the other. HunyuanVideo-1.5's
MXFP8 arm is G1 + G3, its default when this bench ran; since 2026-11-09 it has none, and the arm
is the override of 10e.

### 11a. What ran

- **Scripts** (`/home/ubuntu/fp8-bench/quant_bench/`): `bench.sh` (the runs), `report.py` (timings
  -> `report.json`, `report.md`), `quality.py` (md5s and MXFP8 vs bf16 -> `quality.json`),
  `hbm_sampler.py` (`neuron-monitor` at 1 s, the lanes' cores only -> `hbm_samples.txt`),
  `stamp.py` (a timestamp on every log line). Logs: `<model>/<arm>/{compile,cold_1,cold_2,warm}.log`
  under `/usr/bin/time -v`, `cache_size.txt`; the runner's `progress.txt`.
- **Source.** `quant_bench/src` is `0f75d11` (`difflet/` equals `d83755a`'s) with one file patched,
  `difflet/cli/dp/router.py` (`src_router.patch`): replica cores come from the inherited
  `NEURON_RT_VISIBLE_CORES`, the bug fixed on `fix/dp-router-visible-cores`. Without it, a bench on
  device 4 would have put its ranks on cores 0-3.
- **Lanes.** Device 0 (cores 0-3, `taskset -c 48-71,144-167`): Wan 2.1, FLUX, HunyuanVideo,
  HunyuanVideo-1.5. Device 4 (cores 16-19, `taskset -c 72-95,168-191`): LTX-2, Wan 2.2,
  Qwen-Image. Each lane has half of NUMA node 1, hyperthread pairs kept together.
- **Per arm.** (1) `difflet compile` into an empty `--cache-dir` on the local NVMe RAID
  (`/mnt/nvme`): NEFFs, HLO and NKI caches, the bf16 copy of an fp32 hub DiT, and the quantized
  directory, all rebuilt. (2) Two `generate` processes, one request each: e2e cold. (3) One
  `generate` process with two identical requests (`--requests`): e2e warm is the second request.
  In run 4 (five models), each arm first read the model's hub files through into the page cache.
  Wan 2.1 and LTX-2 kept their run-3 results, which had one read-through before the run
  (`page_cache_warm.log`), not one per arm.
- **Definitions.**
  - *compile*: `time -v`'s wall time of (1).
  - *e2e cold*: the lower wall time of the two (2).
  - *e2e warm*: the second request's `request:` time, plus the decode the parent runs after the
    DP router's workers, from the saved-file timestamps (Wan, Qwen-Image, LTX-2). HunyuanVideo-1.5's
    staged CLI gives encode + denoise + decode of request 1.
  - *DiT step*: that request's step median.
  - *peak HBM*: the largest per-process device memory on the lane's cores during the warm
    process. Only the 4 DiT rank processes appear in the samples, each with whatever else it loads
    and its runtime. The parent's separate one-core VAE decode process (Wan, Qwen-Image,
    HunyuanVideo-1.5's VAE stage) was not captured.
- **Disk.** `time -v`'s "File system inputs" is 0.0 GB for 54 of the 56 steps counted here (7 x 2 x
  4). The other two are Wan 2.1's and LTX-2's MXFP8 compiles, which read 0.49 and 0.35 GB. Wan 2.1 and LTX-2 read their snapshots from the EBS hub's page cache (run 3); the other five
  read from a copy on the NVMe RAID (`/mnt/nvme/difflet-quant-bench-hub`, run 4).

**Runs discarded** (archived beside the results):
1. **Run 1** (`run1_contended/`, `failed_laneB/`). Another session's server took devices 8-15 at
   14:46, so lane B's arms after LTX-2 bf16 failed with "cores busy". Its GLM checkpoint load also
   slowed HunyuanVideo-1.5's MXFP8 compile: the bf16 copy took 370 s against the bf16 arm's 11 s.
2. **Run 2** (`run2_ebs/`). The caches sat on the EBS root, a 125 MB/s volume at 95% utilisation,
   so compiles waited on I/O. LTX-2 bf16 took 17:45 at 86% CPU against 2:06 at 769% on NVMe, with
   about the same user time (0.67 to 0.75 ks across runs 1-3).
3. **Run 3 for FLUX, Wan 2.2 and HunyuanVideo** (`run3_polluted/`). The other session's two server
   launches (16:32, 16:59 UTC; each read ~600 GB) evicted the hub snapshots from the page cache.
   FLUX bf16's compile read 31 GB from the EBS root, Wan 2.2 bf16's 141 GB.
4. **HunyuanVideo-1.5's first run-4 attempt** (`run4_paused/`). Stopped at the user's request
   during bf16's first cold run, to give the node to the other session; rerun after 19:03 UTC.

### 11b. Performance

bf16 -> MXFP8 (MXFP8/bf16):

| model, shape | compile | e2e cold | e2e warm | DiT step | peak HBM per rank | DiT weights per rank |
|---|---|---|---|---|---|---|
| FLUX.1-dev, 1024x1024 | 147 -> 165 s (1.12x) | 51.7 -> 59.2 s (1.14x) | 11.5 -> 9.6 s (0.84x) | 221 -> 183 ms (0.83x) | 16.7 -> 15.4 GiB (0.92x) | 5.95 -> 3.95 GB |
| Qwen-Image, 1024x1024 | 108 -> 170 s (1.58x) | 71.3 -> 85.1 s (1.19x) | 16.1 -> 13.0 s (0.81x) | 303 -> 240 ms (0.79x) | 10.8 -> 8.8 GiB (0.81x) | 10.22 -> 7.20 GB |
| LTX-2, 512x768x121 | 126 -> 168 s (1.34x) | 155 -> 153 s (0.99x) | 106.3 -> 76.0 s (0.71x; ~0.79x without bf16's 10 s stall, below) | 1290 -> 969 ms (0.75x) | 16.9 -> 14.6 GiB (0.87x) | 9.96 -> 6.94 GB |
| Wan 2.1, 480x832x9 | 213 -> 278 s (1.30x) | 91.5 -> 96.2 s (1.05x) | 23.9 -> 21.5 s (0.90x) | 440 -> 391 ms (0.89x) | 11.9 -> 10.6 GiB (0.89x) | 7.50 -> 5.36 GB |
| Wan 2.2-A14B (2 experts), 480x832x9 | 238 -> 285 s (1.20x) | 100.2 -> 107.1 s (1.07x) | 23.8 -> 21.5 s (0.91x) | 438 -> 392 ms (0.89x) | 20.0 -> 16.7 GiB (0.84x) | 15.0 -> 10.7 GB |
| HunyuanVideo, 320x512x61 | 201 -> 212 s (1.05x) | 106.2 -> 116.4 s (1.10x) | 48.2 -> 46.2 s (0.96x) | 662 -> 636 ms (0.96x) | 13.3 -> 12.2 GiB (0.92x) | 6.54 -> 4.85 GB |
| HunyuanVideo-1.5, 480x848x9 | 397 -> 405 s (1.02x) | 110.2 -> 116.9 s (1.06x) | 41.8 -> 38.9 s (0.93x) | 554 -> 501 ms (0.90x) | 8.70 -> 7.88 GiB (0.91x) | 7.80 -> 6.97 GiB resident |

Further detail behind the table:
- **Both cold runs** (bf16 / MXFP8, s):

  | model | bf16 | MXFP8 |
  |---|---|---|
  | FLUX | 57.2, 51.7 | 73.2, 59.2 |
  | Qwen-Image | 78.4, 71.3 | 85.1, 93.4 |
  | LTX-2 | 162.0, 155.0 | 152.8, 171.3 |
  | Wan 2.1 | 94.4, 91.5 | 96.2, 99.9 |
  | Wan 2.2 | 100.2, 101.4 | 107.1, 108.0 |
  | HunyuanVideo | 110.6, 106.2 | 126.7, 116.4 |
  | HunyuanVideo-1.5 | 122.1, 110.2 | 127.2, 116.9 |

- **Warm parts** (bf16 -> MXFP8):

  | model | request | decode or after the last step |
  |---|---|---|
  | FLUX | 11.47 -> 9.59 s | 0.34 s after the last step, inside the request |
  | Qwen-Image | 15.47 -> 12.31 s | parent decode 0.6 -> 0.7 s |
  | LTX-2 | 82.9 -> 56.1 s | parent decode 23.4 -> 19.9 s |
  | Wan 2.1 | 22.14 -> 19.62 s | parent decode 1.8 -> 1.9 s |
  | Wan 2.2 | 21.98 -> 19.63 s | parent decode 1.8 -> 1.9 s |
  | HunyuanVideo | 48.2 -> 46.2 s | 13.8 s after the last step, inside the request |

  HunyuanVideo-1.5's warm request splits as encode 6.1 -> 5.8 s, denoise 27.7 -> 25.0 s,
  decode 8.0 -> 8.1 s.
- **LTX-2 bf16's second warm request stalled.** It took 82.93 s, with 19.12 s to the first step's
  end, against 72.92 s / 9.61 s for request 1 of the same process. MXFP8's two requests are
  steady (56.23, 56.10 s). The stall (16:34:28-16:35:51) overlaps lane A's Wan 2.1 MXFP8 compile
  (16:33:20-16:37:57). With bf16's request 1 instead, warm is about 96.3 -> 76.0 s (~0.79x).
  The arm was not rerun: by then the node had been handed to the other session.
- **HBM.** HunyuanVideo-1.5's own log line agrees with the sampler within 0.1 GiB: rank 0 peaks at
  8.12 GiB allocator + 0.50 runtime = 8.62 in bf16 and 7.29 + 0.50 = 7.79 in MXFP8 (sampler: 8.70
  and 7.88).
- **DiT weights per rank.** All seven are from this bench's logs: the DiT weight-store lines, and
  HunyuanVideo-1.5's "DiT weights per rank" line. Wan 2.2's are 2 x 7.50 and 2 x 5.36 GB. Wan 2.1,
  Wan 2.2 and HunyuanVideo match sections 4, 5 and 7g.

### 11c. Quality

Three checks, in `quality.json` and `hv15_threads_quality.json`:
- **Determinism.** Each arm's four outputs (both cold runs, both warm requests) have equal md5s,
  for all 14 arms.
- **Provenance.** For six models the md5 equals the device file phase G scored (10b), so 10c's CPU
  fp32 scores are scores of exactly these runs. HunyuanVideo-1.5 is the exception (below).
- **MXFP8 vs bf16 on device.** `compare_difflet_runs.py` on the cold_1 outputs, bf16 first. The
  videos are compared through the .mp4 encoder, as in 10c.

| model | bf16 md5 / MXFP8 md5 (= phase G file?) | device MXFP8 vs bf16: PSNR / SSIM / LPIPS | vs CPU fp32: bf16 -> MXFP8 PSNR / SSIM | MXFP8 - bf16 vs fp32 |
|---|---|---|---|---|
| FLUX.1-dev | `67da9e98` / `7b6a23dc` (yes) | 22.59 / 0.908 / 0.098 | 23.36 / 0.917 -> 21.30 / 0.885 | -2.06 dB (one image) |
| Qwen-Image | `5478687f` / `7737239b` (yes) | 38.26 / 0.992 / 0.008 | 36.34 / 0.990 -> 37.36 / 0.992 | +1.02 dB |
| LTX-2 | `0702ffd2` / `c1057ed5` (yes) | 22.47 / 0.830 / 0.078 | 16.55 / 0.472 -> 16.85 / 0.485 | +0.29 dB |
| Wan 2.1 | `450d04ef` / `a48ead02` (yes) | 33.90 / 0.945 / 0.034 | 36.08 / 0.952 -> 33.72 / 0.947 | -2.36 dB |
| Wan 2.2-A14B | `68419054` / `90335e77` (yes) | 33.97 / 0.950 / 0.036 | 36.39 / 0.962 -> 33.46 / 0.949 | -2.93 dB |
| HunyuanVideo | `75e6ba8d` / `7d54b413` (yes) | 31.09 / 0.938 / 0.050 | 20.79 / 0.846 -> 20.59 / 0.837 | -0.20 dB |
| HunyuanVideo-1.5 | `a6b173bc` / `07bd8c81` (no: phase G's are `9ab6e81c` / `7a9ddae7`) | 17.32 / 0.439 / 0.234 | this bench's clips: 24.53 / 0.842 -> 17.32 / 0.440 | -7.20 dB (phase G's clips: -7.47) |

The Wan 2.1 and Wan 2.2 rows are the first device MXFP8-vs-bf16 readings of the final tree. 8k
reran Wan 2.1's MXFP8 default and both Wan 2.2 arms for phase G without comparing them (Wan 2.1's
bf16 clip is H5's, 8h), and 4m's and 5's 33.0 and 35.6 dB predate H5.
For HunyuanVideo-1.5 at 9 frames, 9d's `mx_9` read 17.19 dB / 0.427.

**HunyuanVideo-1.5's clips depend on the host CPU's thread count.**
- **Where.** The prompt encode runs on the host CPU in bf16
  (`difflet/cli/orchestrators/hunyuan_video_15.py:256-257`, `pipe.encode_prompt(device="cpu",
  dtype=torch.bfloat16)`), and the bf16 CPU matmul's result depends on how many threads split it.
- **Check.** `quant_bench/hv15_encode_threads.py` loads the pipeline without the DiT, encodes the
  bench prompt once per process and prints the md5 of `encode_prompt`'s four outputs:

  | taskset | torch threads | `prompt_embeds` md5 |
  |---|---|---|
  | `48-71,144-167` | 48 | `ae873641` (three runs) |
  | `48-95,144-191` | 96 | `eeea0836` (two runs) |

  The mask and the two byT5 outputs are equal (`77a82e57`, `c5fa6e35`, `bf619eac`). The last run of
  each is in `quant_bench/hv15_encode_threads.log`.
- **Effect.** Run 1 (96 threads, `run1_contended/`) reproduced phase G's two files bit for bit;
  run 4 (48 threads) gives other clips, each deterministic at its thread count. Both arms of the
  bench share the 48-thread embedding, so the comparison holds. Bench bf16 against phase G bf16 is
  28.88 dB / SSIM 0.886.
- **The quality reading does not move.** Scoring the bench's own clips against the CPU fp32
  reference (`fp32_ref/hunyuan_video_15/out.mp4`) gives -7.20 dB for MXFP8 - bf16, against phase
  G's -7.47. Device MXFP8 vs bf16 is 17.32 / 0.439 here against 17.44 / 0.446 on phase G's clips.
- **The other six models reproduced phase G's files at either CPU set**, for two reasons.
  - FLUX (T5, CLIP), LTX-2 (Gemma) and both Wans (umT5) encode on the device.
  - Qwen-Image and HunyuanVideo encode on rank 0's host CPU, but through `HostTextEncoder` with a
    pinned thread count (`encoder_threads: 12`; `difflet/cli/orchestrators/qwen_image.py:127`,
    `hunyuan_video.py:150-151`).
  - HunyuanVideo-1.5's staged encode pins nothing.
- **Consequence for serving.** One HunyuanVideo-1.5 request gives different clips on hosts, or under
  CPU affinities, with different thread counts. Pinning its encode's threads as the other two host
  encoders do would remove that (not done here).

### 11d. Reading

- **DiT and e2e warm.** MXFP8 is faster on every model: the step at 0.75x (LTX-2) to 0.96x
  (HunyuanVideo), the warm request at 0.71x to 0.96x. The warm gain is the denoise gain diluted by
  the fixed parts: text encode, decode, and HunyuanVideo's 13.8 s after the last step. LTX-2's 0.71x
  is better than its 0.75x step only because of bf16's stall (11b); without it, it is about 0.79x.
- **Peak HBM per rank** falls 0.81x to 0.92x. The DiT weights shrink 0.66x to 0.89x, and
  activations, other components and the runtime stay the same.
- **E2e cold is slower with MXFP8** on six models (1.05x to 1.19x; LTX-2 0.99x). The cause is the
  load's `eval` phase. `ApplicationBase._after_load` (`difflet/backends/neuron/core/application_base.py:254`)
  calls `prepare_fp8_kernels_`, which packs every MXFP8 weight for the kernel at each load. It costs
  nothing in bf16 and this much with MXFP8, from the faster cold run's DiT phases:

  | model | `eval` (MXFP8) |
  |---|---|
  | Wan 2.2 | 22.2 s (12.1 + 10.1, one per expert) |
  | Qwen-Image | 19.6 s |
  | LTX-2 | 15.4 s |
  | Wan 2.1 | 10.2 s |
  | FLUX | 9.8 s |
  | HunyuanVideo | 8.0 s |
  | HunyuanVideo-1.5 | 5.8 s |

  One request's denoise saves less than that at these short shapes. LTX-2 is the exception: its
  cold request saves 13.4 s (73.0 -> 59.6 s) and its DiT warmup forward 11.1 s (20.7 -> 9.7 s),
  against 15.4 s of packing. Persisting the packed layout in the quantized directory would remove the cost (not
  done here).
- **Compile is 1.02x to 1.58x longer.** The MXFP8 compile builds the quantized directory (at most
  8-45 s: the time between the bf16-copy line, or the process start, and the quantized-directory
  line, so it includes loading the source). Its warmup also loads the DiT, which pays the same
  `eval` packing. Qwen-Image's +62.7 s holds three parts: the quantization (at most 36.1 s), the
  compile's own packing (22.8 s), and a longer `from_pretrained` (18.9 against 11.2 s). The parts
  overlap, because the quantization bound includes some loading.
  HunyuanVideo-1.5's compile is dominated by parts both arms share.
- **Disk.** The quantized directory adds, per model:

  | FLUX | Qwen-Image | LTX-2 | Wan 2.1 | Wan 2.2 | HunyuanVideo | HunyuanVideo-1.5 |
  |---|---|---|---|---|---|---|
  | 15 GB | 27 GB | 24 GB | 19 GB | 38 GB | 18 GB | 14 GB |

  For the fp32 hub DiTs it adds to the bf16 copy both arms keep: 27 GB for Wan 2.1, 54 GB for
  Wan 2.2, 16 GB for HunyuanVideo-1.5. On a 125 MB/s EBS volume, writing a cache this size bounds
  the compile time (run 2).

## 12. Before and after quantization at the official settings: Wan 81 frames with CFG, HunyuanVideo-1.5 121 frames

The user's request (2026-11-08): section 11's comparison again for Wan 2.1, Wan 2.2-A14B and
HunyuanVideo-1.5 at their publishers' sampling settings. Plan:
`docs/plans/2026-11-08-mxfp8-official-settings-rerun.md` (Tasks 1-7; Task 8, the CPU fp32
references at these shapes, waits for the user). Ran on the second trn3 (trn3.48xlarge, same
Neuron SDK and the same 51 venv pins as host A) on 2026-11-09: the lanes, floors and control from
the checkout at `a300f61`, the determinism gate from `a504169` (the two differ in scripts, tests
and the HunyuanVideo-1.5 driver's `run`, which no step here calls).
Results, logs and floors: `/mnt/nvme/difflet-bench/results/` on that host.

### 12a. What ran

- **Settings (D2).** Prompt "a cat walking through a garden", seed 42, LNC=2, each snapshot
  pinned by `--revision`:

  | key | snapshot | shape | steps | guidance | negative prompt | layout |
  |---|---|---|---|---|---|---|
  | `wan21_official` | Wan2.1-T2V-14B-Diffusers @ `38ec498c` | 480x832x81 | 50 | 5.0 | Wan 2.1 card's | tp4, CFG sequential |
  | `wan22_official` | Wan2.2-T2V-A14B-Diffusers @ `5be7df96`, both experts | 480x832x81 | 40 | 4.0 (high noise) / 3.0 (low noise) | Wan 2.2 card's | tp4, CFG sequential |
  | `hv15_official` | HunyuanVideo-1.5-Diffusers-480p_t2v @ `286be7ce` | 480x848x121 | 50 | true CFG 6.0 (the snapshot's guider) | "" | tp2 x CFG2 |

  HunyuanVideo-1.5 has no MXFP8 default (Task F8), so its MXFP8 arm sets the former default,
  G1 + G3, as `DIFFLET_QUANT_TARGETS` for its compile, cold and warm commands only.
- **Tooling.** `scripts/quant_bench/`: `bench.sh` (per key, bf16 then MXFP8 on one
  lane: compile into an empty cache, two cold `generate` processes, one warm process with two
  requests), `report.py`, `quality.py`, `hbm_sampler.py` (neuron-monitor at 1 s on cores 0-3,
  16-19, 32-35). Definitions as 11a. The first cold run of each arm kept its work dir, so its final
  latents are there for Task 8.
- **Determinism gate (Task 4).** Section 11's Wan 2.1 key at 480x832x9 on device 0: all eight
  outputs have section 11's md5s (bf16 `450d04ef`, MXFP8 `a48ead02`), the caches section 11's
  sizes (38 and 57 GB), and the timings section 11's: compile 196.2 / 229.3 s (section 11: 213 /
  278), e2e cold 84.5 / 86.7 s (91.5 / 96.2), e2e warm 23.8 / 21.5 s (23.9 / 21.5), DiT step
  437.3 / 390.8 ms (440 / 391). This host's numbers compare with the campaign's (its compiles run
  8-18% faster).
- **Lanes (D4), 01:48-04:13.** W1 `wan21_official` on device 0 (`taskset -c 48-71,144-167`, 48
  threads), W2 `wan22_official` on device 4 (`72-95,168-191`), H `hv15_official` on device 8
  (`0-47,96-143`, the whole of NUMA node 0: 96 threads, the count of section 9's clips).
  `neuron-ls` showed every device idle before the lanes, the floors and the control, and on the
  lanes' cores the sampler saw only the lanes' processes. All 24 steps exited 0; `time -v`'s "File system inputs" is 0.0 GB in
  every step (0.01 GB in HunyuanVideo-1.5's bf16 compile).
- **Floors (D5), 04:13-04:41, after the lanes.** One bf16 compile and generate each, with the
  key's own command lines and only the layout changed (`floors/floors.json`,
  `floors/run_floor.sh`): Wan 2.1 and Wan 2.2 at `--tp-degree 8` (cores 0-7, 16-23),
  HunyuanVideo-1.5 at `--tp-degree 4 --cfg-parallel` (cores 32-39).
- **Control (not in the plan), 04:41-05:01.** Wan 2.1's bf16 arm with the NKI trace cache on
  (`DIFFLET_NKI_TRACE_CACHE=1`), one compile and one cold generate on lane W1 alone (12b).

**Runs discarded:** the first determinism gate (`discarded/wan21_shared_caches/`). This host's
`~/.bashrc` exports `DIFFLET_COMPILE_CACHE` and the `TORCH_NEURONX_*` cache locations, which win
over `--cache-dir`, so its compiles reused another cache's NEFFs and bf16 copy (8 KB per
`--cache-dir`, a bf16 copy in 0.00 s, compile 103 / 133 s). Its md5s matched; its timings are
not used. `bench.sh` unsets those variables since `a504169`.

### 12b. Performance

bf16 -> MXFP8 (MXFP8/bf16):

| model, shape | compile | e2e cold | e2e warm | DiT step | peak HBM per rank | DiT weights per rank |
|---|---|---|---|---|---|---|
| Wan 2.1, 480x832x81, CFG 5.0 | 958 -> 652 s (0.68x) | 996.5 -> 660.2 s (0.66x) | 529.7 -> 518.6 s (0.98x) | 10,143 -> 9,958 ms (0.98x) | 23.8 -> 22.1 GiB (0.93x) | 7.50 -> 5.36 GB |
| Wan 2.2-A14B, 480x832x81, CFG 4.0/3.0, 40 steps | 984 -> 738 s (0.75x) | 900.8 -> 571.1 s (0.63x) | 426.6 -> 419.2 s (0.98x) | 10,151 -> 9,963 ms (0.98x) | 31.5 -> 28.2 GiB (0.89x) | 2 x 7.50 -> 2 x 5.36 GB |
| HunyuanVideo-1.5, 480x848x121, CFG 6.0 | 1,352 -> 1,439 s (1.06x) | 767.9 -> 797.2 s (1.04x) | 563.1 -> 569.2 s (1.01x) | 9,916 -> 10,022 ms (1.01x) | 18.5 -> 18.5 GiB (1.00x) | 7.80 -> 6.97 GiB resident |

- **Both cold runs** (s): Wan 2.1 996.5, 1001.1 / 661.9, 660.2; Wan 2.2 903.5, 900.8 / 595.0,
  571.1; HunyuanVideo-1.5 783.4, 767.9 / 815.8, 797.2.
- **Warm parts.** Wan 2.1: request 507.7 -> 498.1 s, the parent's decode 22.0 -> 20.5 s. Wan 2.2:
  406.1 -> 398.6 s, decode 20.5 -> 20.6 s. HunyuanVideo-1.5: encode 8.4 -> 9.0 s, denoise 495.9 ->
  501.1 s, decode 58.8 -> 59.1 s.
- **CFG ran** (Review Focus 2): Wan's step is 10.14 s (Wan 2.1) and 10.15 s (Wan 2.2), twice
  the 5.05 s of the no-CFG 81-frame step (4m, 5).
- **HBM.** The sampler's peaks include each rank's runtime and are taken over the warm process.
  HunyuanVideo-1.5's own rank-0 line (allocator + runtime) agrees there: 10.04 + 7.64 = 17.68 GiB
  in bf16, 9.21 + 8.45 = 17.66 GiB in MXFP8, the allocator 0.83 GiB lower and the runtime 0.81 GiB
  higher (section 9's `mx_final`: 9.21 + 8.45). The MXFP8 arm's first cold run, the first process
  after its compile, peaked higher: 9.70 + 8.45 = 18.15 GiB (sampler 19.06 GiB; `nod4/mx121`: 9.69
  + 8.45); its second cold run and the warm process read 9.21 + 8.45, and bf16 reads 10.04 + 7.64
  in all three.
- **Where Wan's cold and compile gains come from: the NKI trace cache, not the quantization.**
  The runner turns torch-neuronx's NKI trace cache off for the ranks (`difflet/cli/runner.py`),
  and the Wan driver never opts in (the HunyuanVideo and HunyuanVideo-1.5 drivers do), but
  `--quant` does (`difflet/cli/quant.py:156`, for the MX kernel). Without the cache every Wan
  process re-traces its kernels at 81 frames. From the first cold run's start-up line:

  | | bf16 load (of it: DiT warmup forward) | MXFP8 load (of it: packing `eval`, warmup forward) | the request itself |
  |---|---|---|---|
  | Wan 2.1 | 421.9 s (415.0) | 89.4 s (9.9, 72.8) | 509.4 -> 500.9 s |
  | Wan 2.2 | 430.7 s (415.2, plus 5.0 for the second expert) | 118.7 s (10.9 + 9.9 for the two experts, 83.6 + 5.0) | 408.0 -> 401.1 s |

  and the compile's warmup forward is 777.5 -> 423.9 s (Wan 2.1) and 776.5 -> 424.2 s (Wan 2.2).
  The control measures it for Wan 2.1: its bf16 arm with `DIFFLET_NKI_TRACE_CACHE=1` compiles in
  540.5 s and runs cold in 652.6 s (one run; warmup forward 72.3 s), its clip `295dcb67`, the
  default bf16 arm's (the cache changes no output), at a 10,152 ms step (the default arm:
  10,139-10,143 ms). Against that bf16, MXFP8 is 1.21x on compile (652.1 against 540.5 s) and
  1.01x on e2e cold (660.2 against 652.6 s). Wan 2.2 had no control; its warmup split is Wan
  2.1's. HunyuanVideo-1.5's driver opts in for both arms, and there
  MXFP8 is 1.06x and 1.04x.

### 12c. Quality

- **Determinism.** Each arm's four outputs (both cold runs, both warm requests) have one md5:
  Wan 2.1 `295dcb67` / `ede99aa5`, Wan 2.2 `ee61e12e` / `79860c25`, HunyuanVideo-1.5 `d3ef0f1c`
  / `8da829b2` (bf16 / MXFP8). For Review Focus 1: the cold runs carry the negative prompt on the
  command line, the warm requests in their `--requests` lines (the router's
  `work_warm/requests/req_000{0,1}.json` hold it) and also on the warm command line (`bench.sh`
  passes it to every generate), so the equal md5s show the two paths agree but do not isolate the
  request-line field; Task 1's unit tests pin the line's precedence. Nothing here compares a clip
  without the negative prompt.
- **Provenance.** HunyuanVideo-1.5's clips are section 9's, bit for bit: bf16 `d3ef0f1c` is
  `bf16_pre`, MXFP8 `8da829b2` is `nod4/mx121` (Review Focus 4: 96 encode threads). Its quantized
  directory is named `mxfp8-2b29f4dfd5ee3b17` here, not section 9's `mxfp8-afcbee695501d1d2`: the
  bf16 copy's manifest, part of the source fingerprint, records the fp32 source's resolved path
  and mtimes (`difflet/quant/checkpoint.py` `source_fingerprint`), so a copy-built directory's
  name is per host. The spec is the same, and so are the weights, as the clip shows.
- **Against bf16 on the device**, with the floors (bf16 at another TP degree) and the campaign's
  floor-relative screen (PSNR >= min(30, floor - 4), SSIM >= min(0.97, floor - 0.10), LPIPS <=
  max(0.05, 2 x floor)); `quality.py`, PSNR the mean over frames with the worst frame in
  parentheses:

  | model | MXFP8 vs bf16: PSNR / SSIM / LPIPS | floor vs bf16 (floor md5) | screen thresholds | screen |
  |---|---|---|---|---|
  | Wan 2.1 | 15.08 (12.96) / 0.391 / 0.351 | 22.75 (18.31) / 0.784 / 0.098, tp8 (`e542f897`) | 18.75 / 0.684 / 0.197 | **fails** all three |
  | Wan 2.2-A14B | 14.29 (13.27) / 0.398 / 0.275 | 14.83 (13.74) / 0.439 / 0.254, tp8 (`9379b013`) | 10.83 / 0.339 / 0.508 | passes |
  | HunyuanVideo-1.5 | 12.65 (12.20) / 0.183 / 0.504 | 16.94 (15.94) / 0.362 / 0.326, tp4 x CFG2 (`230cd63e`) | 12.94 / 0.262 / 0.653 | **fails** PSNR and SSIM |

  HunyuanVideo-1.5's MXFP8 row is section 9's (12.65 / 0.183 / 0.504; the clips are the same).
  Its floor here is on the tree without D4; section 9's `bf16_tp4` (15.90 / 0.312 / 0.363) was
  on the D4 tree, and the screen failed there on SSIM alone.
- No CPU fp32 reference at these shapes: Task 8, gated (about 30-38 h of both NUMA nodes).

### 12d. Reading

- **At the official settings MXFP8 buys about 2% of DiT step on Wan and costs about 1% on
  HunyuanVideo-1.5.** Wan 2.1 and Wan 2.2 run 0.98x with CFG at 81 frames (0.89x at section 11's
  9 frames), HunyuanVideo-1.5 1.01x at 121 frames (0.90x at 9). The warm request follows the
  step: 0.98x, 0.98x and 1.01x. With CFG both of a step's forwards gain alike, so CFG does not
  change the ratio; the frame count does, as attention, which stays bf16, takes over the step.
- **Peak HBM falls for Wan** (0.93x, 0.89x: the DiT weights shrink 2.1 GB per rank and per
  expert) **and not for HunyuanVideo-1.5** (1.00x by the sampler over the warm process, 17.66
  against 17.68 GiB by its own line there; 18.15 GiB in the MXFP8 arm's first cold run).
- **Wan's faster compile and cold start are the NKI trace cache** (measured on Wan 2.1; Wan 2.2
  shows the same warmup split). Quantization adds work: the packing `eval` at every load (10-11 s
  per expert: 11 s for Wan 2.1, 20 s for Wan 2.2's two) and, in the compile, the quantized
  directory (at most 34 / 53 s) and a longer warmup forward (Wan 2.1: 423.9 against the control's
  369.3 s). With the trace cache bf16 Wan 2.1 compiles in 541 s instead of 958 and starts cold in
  653 s instead of 997 (12b's control), and against it MXFP8 is 1.21x on compile (111.6 s more)
  and 1.01x cold. Opting the Wan driver into the trace cache, as the HunyuanVideo drivers are, is
  a follow-up (not done here).
- **Quality at these settings.** Wan 2.1's MXFP8 clip is much further from bf16 than bf16 at
  another TP degree (15.1 against 22.75 dB) and fails the screen. Wan 2.2's two bf16 runs are
  themselves 14.8 dB apart, so its MXFP8 clip (14.3 dB) passes the screen without that proving it
  close. HunyuanVideo-1.5 fails as in section 9 (there on SSIM alone, here on PSNR too), and the
  CPU fp32 reference at 9 frames already put MXFP8 7.5 dB further than bf16 (10e). Only the CPU fp32 references at these shapes (Task 8)
  can say which Wan arm is closer to the true sample.
