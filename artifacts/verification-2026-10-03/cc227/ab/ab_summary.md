# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | — | 308.6 (False) | 0 | 394.6 | 0.0 | 568.8 / 568.7 (19) |
| fp8 | — | 319.3 (False) | 0 | 204.9 | 0.0 | 648.3 / 648.2 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_vs_bf16_run0 | 24.93 | 0.8845 | 0.1167 | 0.975817 | 2.104e-02 | 13.20 |
