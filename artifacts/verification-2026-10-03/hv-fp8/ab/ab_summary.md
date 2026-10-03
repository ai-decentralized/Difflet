# FP8 PTQ A/B — hunyuanvideo-community/HunyuanVideo

shape 320x512x61, 20 steps, guidance 6.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': None}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | — | 450.9 (True) | 0 | 967.3 | 128.3 | 831.3 / 832.0 (19) |
| fp8 | 1.3 | 1013.8 (True) | 0 | 841.9 | 128.6 | 760.8 / 760.8 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_vs_bf16_run0 | 5.83 | 0.0110 | 0.9439 | — | — | — |
