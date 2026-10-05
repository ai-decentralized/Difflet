# FP8 PTQ A/B — hunyuanvideo-community/HunyuanVideo

shape 320x512x61, 20 steps, guidance 6.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': None}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| fp8 | 51.9 | 701.9 (True) | 0 | 479.5 | 16.3 | 783.1 / 774.5 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
