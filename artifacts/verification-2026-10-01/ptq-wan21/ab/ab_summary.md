# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'activation': 'dynamic'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | — | 6711.9 (False) | 0 | 412.7 | 0.0 | 573.2 / 573.0 (19) |
| bf16 | — | 6711.9 (False) | 1 | 87.1 | 0.0 | 573.7 / 573.6 (19) |
| fp8 | 32.2 | 590.1 (False) | 0 | 317.4 | 0.0 | 822.1 / 821.9 (19) |
| fp8 | 32.2 | 590.1 (False) | 1 | 94.7 | 0.0 | 822.0 / 821.7 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_vs_bf16_run0 | 36.50 | 0.9269 | 0.0789 | 0.997457 | 2.240e-03 | 22.93 |
| fp8_vs_bf16_run1 | 36.50 | 0.9269 | 0.0789 | 0.997457 | 2.240e-03 | 22.93 |
| bf16_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
| fp8_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
