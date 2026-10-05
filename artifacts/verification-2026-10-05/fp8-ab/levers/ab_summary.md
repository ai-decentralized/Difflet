# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': '/home/ubuntu/Difflet/.claude/worktrees/fp8-step-experiments/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| fp8 | 1.3 | 411.9 (False) | 0 | 342.4 | 0.0 | 473.1 / 473.0 (19) |
| fp8 | 1.3 | 411.9 (False) | 1 | 84.9 | 0.0 | 472.8 / 472.7 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
