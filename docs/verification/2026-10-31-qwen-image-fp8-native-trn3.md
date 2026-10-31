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

## 5. Device V2–V4: `--quant-compute nki`

_pending_.

## Bug ledger

(none yet)
