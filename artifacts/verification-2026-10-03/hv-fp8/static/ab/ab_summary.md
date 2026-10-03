# FP8 PTQ A/B — hunyuanvideo-community/HunyuanVideo

shape 320x512x61, 20 steps, guidance 6.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': '/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-03/hv-fp8/static/act_calibration_hv.json'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| fp8 | 218.9 | 1044.4 (True) | 0 | 688.0 | 151.8 | 729.6 / 728.4 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
