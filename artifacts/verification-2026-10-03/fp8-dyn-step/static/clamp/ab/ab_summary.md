# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': '/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| fp8 | 1.8 | 445.8 (False) | 0 | 339.5 | 0.0 | 586.1 / 586.4 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
