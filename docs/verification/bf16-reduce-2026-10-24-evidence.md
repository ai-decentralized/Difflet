# bf16 tensor-parallel all-reduce — evidence (2026-10-24)

Campaign for the change on branch `worktree-bf16-tp-reduce-qwen-ltx2`: every model except MiniMax-H3 all-reduces its tensor-parallel outputs in bf16 by default (one commit per model), and fp32 stays available through `--tp-reduce-dtype fp32`. Every number below was measured on one host on 2026-10-24 and comes from that day's logs; a value marked *inference* was not measured directly.

Plan: `docs/superpowers/plans/2026-10-24-qwen-ltx2-bf16-tp-reduce.md`. Executor ledger: `.superpowers/sdd/2026-10-24-qwen-ltx2-bf16-tp-reduce/progress.md`.

## Summary

| Model | What changed | DiT step, fp32 → bf16 reduce | All-reduce device time / forward | Output, bf16 vs fp32 |
|---|---|---|---|---|
| **Qwen-Image** | DiT: 240 all-reduces f32 → bf16 | **419.8 → 361.9 ms (−13.8%)** | 84.1 → 42.7 ms | latent cosine 0.99994, PSNR 47.2 dB |
| **LTX-2** | DiT: 384 hidden all-reduces f32 → bf16 (RMS statistics stay f32) | **459.8 → 433.5 ms (−5.7%)** | 92.5 → 52.0 ms | video latent cosine 0.99958, PSNR 38.2 dB |
| Flux | T5 + CLIP encoders (DiT already bf16) | warm e2e 40.7 → 39.9 s | not profiled | PSNR 46.1 dB, SSIM 0.997 |
| Wan 2.2 | umT5 encoder (DiT already bf16) | warm e2e 72.1 → 69.1 s | not profiled | PSNR 38.3 dB, SSIM 0.943 |
| HunyuanVideo | DiT: 84 all-reduces f32 → bf16 | warm e2e 109.8 → 123.2 s (n = 1; open, issue #110) | not profiled | PSNR 36.6 dB, SSIM 0.952 |
| MiniMax-H3 | unchanged (fp32 default kept; flag rejected) | — | — | — |

Against a CPU fp32 reference that starts from the device runs' exact initial noise, Qwen-Image's bf16-reduce output is as far from fp32 as its fp32-reduce output: latent relative L2 error 0.1935 vs 0.1923, PSNR 24.72 vs 24.64 dB. The all-reduce precision is not what separates the device from fp32. LTX-2's bf16-reduce output is slightly further from the CPU fp32 reference than its fp32-reduce output: relative L2 0.1397 vs 0.1305, PSNR 34.14 vs 34.45 dB, one seed. That 0.009 increase is small next to the 0.13 that both runs share.

## Setup

| | |
|---|---|
| Host | AWS trn2.3xlarge: 1 Trainium2 chip, `logical-neuroncore-config: 2` (4 logical NeuronCores), 96 GB device memory, 124 GB host RAM, 12 vCPU. Neuron runtime 2.35.30, driver 2.31.0 |
| Software | venv from `requirements-neuron.lock`: Python 3.12.3, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360, neuronx-distributed 0.19.28492, diffusers 0.38.0 |
| Parallelism | `--backend trainium --tp-degree 4 --cp-degree 1` (NxD ahead-of-time) for every run |
| Qwen-Image | `Qwen/Qwen-Image` @ `75e0b4be`, 1024×1024, 20 steps, guidance 4.0 (the transformer has `guidance_embeds: false`, so one forward per step), seed 42 |
| LTX-2 | `Lightricks/LTX-2` @ `47da56e2`, 480×704×49, 20 steps, guidance 1.0, seed 42 |
| Flux / Wan 2.2 / HunyuanVideo | the `benchmark/trn2/*.md` reproduction configs: 1024² 28 steps g3.5; 480×832×9 20 steps g1.0; 320×512×61 20 steps g6.0 |
| Prompt | "a cinematic shot of a red fox running through a snowy forest" |
| Discipline | one device job at a time, detached jobs, nothing else on the host during timed runs; CPU references only after all device timing |

### How each number is measured

- **DiT step:** wall time between consecutive denoise iterations inside a real `difflet generate`, recorded by `DIFFLET_STEP_TIMES_FILE` (commit `dda1a0e`). Step 0 is excluded and the steps of three warm runs are pooled (n = 57). Trainium DiT calls return host tensors, so each forward is already synchronized.
- **DiT forward:** time inside the DiT call of each step (same records).
- **Host work per step:** step median minus forward median.
- **Warm end-to-end:** wall time of one `difflet generate` process. Each measurement sequence starts with one discarded run that warms the page cache, followed by three timed runs.
- **Page-cold end-to-end:** `sync; echo 3 > /proc/sys/vm/drop_caches` first, n = 1.
- **Device profile:** one DiT forward, captured with `neuron-explorer capture` (4 collective workers, profile id 0, the 4th execution, random inputs). `total_exec_time` and `cc_op_time` are rank 0's.
- **HLO fingerprint:** `all-reduce` ops counted by element type and shape in each compiled `model.hlo_module.pb`.
- **Quality:** latent relative L2 error and cosine (float64), and per-frame PSNR and SSIM (Wang 2004, 11×11 Gaussian window, σ 1.5) on uint8 frames (`scripts/quality_vs_reference.py`). The CPU fp32 reference is diffusers in fp32 on the host CPU, started from the device run's dumped initial latents (`scripts/cpu_fp32_reference.py`). The fp32 and bf16 runs' initial latents were checked byte-identical.

## The change in the compiled graphs

The premise holds in the installed NxD: `RowParallelLinear.__init__` defaults to `reduce_dtype=torch.float32` (`neuronx_distributed/parallel_layers/layers.py:863`), and `_rpl_maybe_reduce_output` casts to that dtype before the all-reduce, only when `reduce_output=True` (`:1029-1043`). `None` falls back to the layer dtype (`:878`).

| Artifact | fp32 reduce (legacy key) | bf16 reduce (new default) |
|---|---|---|
| Qwen-Image DiT | 240 all-reduce, **f32**: 120 × `[1,1024,3072]` + 120 × `[1,4096,3072]` | 240 all-reduce, **bf16**, same shapes |
| Qwen-Image text encoder (NxDI Qwen2.5-VL) | 56, already bf16 | unchanged |
| LTX-2 DiT | 192 × `[1,2310,4096]` + 192 × `[1,51,2048]` **f32**; RMS statistics 240 × `[1,2310,1]` + 240 × `[1,51,1]` + 96 × `[1,1024,1]` f32 | hidden 384 → **bf16**; RMS statistics stay f32 (intended) |
| HunyuanVideo DiT | 40 × `[1,10240,3072]` + 44 × `[1,256,3072]` **f32**; 40 × `[1,10496,3072]` single-stream manual reduce already bf16 | all 124 **bf16** |

## Qwen-Image (TP4)

| Metric | fp32 reduce | bf16 reduce | Δ | June baseline |
|---|---|---|---|---|
| DiT step median / p90 | 419.8 / 421.0 ms | **361.9 / 362.5 ms** | −57.9 ms (−13.8%) | 447.2 ms (neuronx-cc 2.25) |
| DiT step mean, min–max | 419.9, 418.0–423.3 | 361.7, 360.1–365.1 | | |
| DiT forward median | 417.6 ms | 361.1 ms | −56.5 ms | |
| Host work per step | 2.2 ms | 0.7 ms | | |
| Denoising loop (20 steps), 3 runs | 8.76 / 8.61 / 8.76 s | 7.59 / 7.56 / 7.44 s | −1.2 s | |
| Warm e2e, 3 runs | 66.0 / 66.6 / 67.3 s | 60.8 / 60.8 / 60.3 s | | |
| **Warm e2e mean** | **66.6 s** | **60.6 s** | **−6.0 s** | 62.7 s |
| Page-cold e2e | 513.6 s | 498.8 s | | 509 s |
| Phases, first warm run: text encoded / latents ready / image written | 17.2 / 51.0 / 64.6 s | 15.1 / 42.7 / 59.6 s | | |

## LTX-2 (TP4)

| Metric | fp32 reduce | bf16 reduce | Δ | June baseline |
|---|---|---|---|---|
| DiT step median / p90 | 459.8 / 460.6 ms | **433.5 / 434.4 ms** | −26.3 ms (−5.7%) | 437.9 ms (neuronx-cc 2.25) |
| DiT step mean, min–max | 459.7, 458.0–464.1 | 433.6, 432.1–436.9 | | |
| DiT forward median | 459.3 ms | 433.2 ms | | |
| Host work per step | 0.3 ms | 0.3 ms | | |
| Denoising loop, 3 runs | 9.20 / 9.20 / 9.19 s | 8.67 / 8.68 / 8.67 s | −0.5 s | |
| Warm e2e, 3 runs | 58.1 / 54.4 / 55.5 s ¹ | 55.8 / 54.3 / 54.2 s | | |
| **Warm e2e mean** | **56.0 s** | **54.8 s** | **−1.2 s** | 58.5 s |
| Page-cold e2e | 767.7 s | 767.5 s | 0 | 778 s |

¹ fp32 warm1 (426.2 s) is excluded and replaced by warm4. The one discarded warm-up after switching from Qwen to LTX-2 did not refill the page cache, so warm1 re-read about 370 s of checkpoint shards at disk speed (`runs/ltx2/fp32/excluded.txt`).

## Device profile (one DiT forward, rank 0, random inputs)

| Category | Qwen fp32 | Qwen bf16 | LTX-2 fp32 | LTX-2 bf16 |
|---|---|---|---|---|
| **Device time / forward** | 403.7 ms | **344.5 ms** | 432.5 ms | **406.9 ms** |
| Collective ops (all-reduce) | 84.1 ms | **42.7 ms** | 92.5 ms | **52.0 ms** |
| Collective op count | 846 | 488 | 1729 | 1345 |
| Tensor engine (matmul) active | 263.1 ms | 254.8 ms | 268.7 ms | 255.6 ms |
| Vector engine active | 189.4 ms | 176.8 ms | 184.4 ms | 189.5 ms |
| Scalar engine active | 188.1 ms | 171.8 ms | 201.5 ms | 199.7 ms |
| GpSimd engine active | 69.6 ms | 62.6 ms | 97.1 ms | 86.2 ms |
| DMA active | 331.6 ms | 280.2 ms | 329.4 ms | 312.7 ms |

- **Qwen:** bf16 halves all-reduce time (−41.4 ms) and DMA time (−51 ms). The device-time saving, 59.2 ms, matches the 57.9 ms per-step saving measured on real data.
- **LTX-2:** all-reduce time falls by 40.5 ms but device time by only 25.6 ms, which matches the 26.3 ms per-step saving. *Inference:* part of LTX-2's all-reduce time already overlapped other work.
- **Collective op count:** bf16 needs fewer ops for the same all-reduces. *Inference:* larger fp32 payloads are split into more chunks.

## Quality vs CPU fp32

Each comparison uses the first warm run. The device runs and the CPU reference start from byte-identical initial latents.

| Qwen-Image | Latent rel. L2 | Latent cosine | Image PSNR | SSIM |
|---|---|---|---|---|
| Device fp32 reduce vs CPU fp32 | 0.1923 | 0.9814 | 24.64 dB | 0.957 |
| Device bf16 reduce vs CPU fp32 | 0.1935 | 0.9811 | 24.72 dB | 0.957 |
| Device bf16 vs device fp32 | 0.0111 | 0.99994 | 47.16 dB | 0.996 |

![CPU fp32 / device fp32 reduce / device bf16 reduce](qwen_cpu_fp32_bf16.jpg)

| LTX-2 | Video latent rel. L2 | cosine | Frames PSNR mean (worst) | SSIM |
|---|---|---|---|---|
| Device fp32 reduce vs CPU fp32 | 0.1305 | 0.9919 | 34.45 dB (32.95) | 0.941 |
| Device bf16 reduce vs CPU fp32 | 0.1397 | 0.9907 | 34.14 dB (32.53) | 0.939 |
| Device bf16 vs device fp32 | 0.0289 | 0.99958 | 38.17 dB (37.28) | 0.954 |

For Qwen-Image the reduce precision does not change the distance to fp32. For LTX-2, bf16 adds 0.009 to the video-latent relative error against CPU fp32 (0.1305 → 0.1397) and costs 0.3 dB PSNR; the bf16-vs-fp32 device difference itself is 0.029. The CPU references took 2205 s (Qwen, 125 s per step) and 2231 s (LTX-2, 86 s per step).

The LTX-2 audio latents are compared device against device only (bf16 vs fp32: cosine 0.99983, relative L2 0.0199), because the diffusers step callback does not expose them. Qwen's guidance setting cannot skew the reference: its transformer config has `guidance_embeds: false`, so the device ignores guidance 4.0, and the diffusers reference runs `true_cfg_scale=1.0` (no CFG).

## Regression check: Flux, Wan 2.2, HunyuanVideo (Task 11)

| Model | Compile fp32 / bf16 | Warm e2e fp32 / bf16 (n = 1) | bf16 vs fp32 output | Gate (≥ 30 dB) |
|---|---|---|---|---|
| Flux | 1221 s / 221 s ² | 40.7 / 39.9 s | PSNR 46.13 dB, SSIM 0.997 | PASS |
| Wan 2.2 | 6557 s / 304 s ² | 72.1 / 69.1 s | PSNR 38.30 dB (worst 36.81), SSIM 0.943 | PASS |
| HunyuanVideo | 4821 s / 4145 s | 109.8 / **123.2 s** | PSNR 36.64 dB (worst 33.22), SSIM 0.952 | PASS (quality) |

² The bf16 compiles reused graphs that are identical in both dtypes from the neuronx-cc cache (the VAE; Wan 2.2's fp32 compile includes a device VAE build). These are not like-for-like compile times.

HunyuanVideo's single bf16 warm run was 13.4 s slower. Most of the gap is in stages the change does not touch: the Llama encode took 4.7 s longer and the DiT device init 4.9 s longer. A +1.7 s residual in denoise + decode is unresolved. At the user's request, the follow-up (device profiles of both DiT graphs, plus n = 3 warm runs) was not run and is tracked in **https://github.com/ai-decentralized/Difflet/issues/110** with ready-to-run commands.

Two fp32 runs of the same model produce identical output (Wan 2.2 warm-up vs warm1: PSNR ∞), so every bf16-vs-fp32 difference comes from the reduce dtype.

## Compile

DiT build times (the neuronx-cc "Finished building model" line): LTX-2 540.8 s (fp32) vs 546.8 s (bf16); Qwen-Image bf16 357.5 s. The Qwen fp32 DiT was built during a first compile attempt that was lost when its supervisor was stopped. The 493.5 s rerun mostly built the VAE (464.6 s), so there is no like-for-like Qwen fp32 DiT build time. The reduce precision does not change compile time materially (LTX-2: +6 s).

## Bugs found and fixed (one commit each, test that failed first)

| Commit | Bug | Found by |
|---|---|---|
| `8258155` | Serving artifact identities (Qwen DiT stage, HunyuanVideo denoiser, Wan, LTX-2, Flux) ignored the reduce dtype: an upgraded server kept loading fp32 generations, and one identity could hold two precisions | final branch review |
| `d294840` | `qwen_image` was missing from DiffletPipeline's legacy table, so bf16 and fp32 requests shared one key | final branch review |
| `d9069ec` | The DP/batch router did not forward `--tp-reduce-dtype` to workers | final branch review |
| `d196c82` | Flux CLI `generate` pre-check hashed the raw `tp_reduce_dtype` (`fp32`) while compile keyed `float32`, so every Flux generate missed its artifact | device (Flux fp32 warm-up exited in 13 s) |

## Caveats

- One seed and one prompt per model. Every bf16-vs-fp32 difference is consistent with run-to-run determinism (same-dtype runs are identical), but that does not show quality across prompts.
- The CPU reference encodes the prompt in fp32 on the host, while the device encodes in bf16 (Qwen2.5-VL on the device; LTX-2's Gemma runs on the host in both cases). That gap is part of every "vs CPU" number and is the same for both reduce dtypes.
- Device profiles use random inputs, so totals sit below real-data times (Qwen bf16: 344.5 ms profiled vs 361.1 ms real forward); the categories compare like with like.
- Warm end-to-end on this host depends on the page cache. After a model switch, one discarded warm-up was not always enough (see LTX-2 note ¹).
- Task 13 (PyTorch-native Qwen and Wan 2.1 runs) was dropped at the user's request. The fallback native environment (`/home/ubuntu/venvs/native-312`) and checkout (`/home/ubuntu/campaign/native_src`) remain unused on the host.

## Evidence on the host

`/home/ubuntu/campaign/bf16-reduce/`: `runs/<model>/<dtype>/<run>/` (timestamped `generate.log`, `steps.jsonl`, latents, outputs), `runs/<model>/summary.json`, `runs/<model>/quality_*.json`, `runs/<model>/hlo_allreduce.*`, `runs/profile/<label>/` (NTFF and summary JSON), `runs/profile/table.json`, `runs/cpu_ref/`, `timings.txt`, and `jobs/` (every job script used).
