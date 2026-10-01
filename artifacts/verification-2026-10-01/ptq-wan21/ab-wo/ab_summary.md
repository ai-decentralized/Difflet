# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'activation': 'none'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| fp8 | — | 444.6 (False) | 0 | 312.3 | 0.0 | 732.8 / 732.7 (19) |
| fp8 | — | 444.6 (False) | 1 | 92.4 | 0.0 | 733.0 / 732.9 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
