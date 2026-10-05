# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': '/home/ubuntu/Difflet/.claude/worktrees/fp8-step-experiments/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | — | 324.6 (False) | 0 | 338.8 | 0.0 | 573.9 / 573.7 (19) |
| bf16 | — | 324.6 (False) | 1 | 111.5 | 0.0 | 573.8 / 573.6 (19) |
| fp8 | 414.8 | 342.1 (False) | 0 | 88.6 | 0.0 | 585.4 / 585.3 (19) |
| fp8 | 414.8 | 342.1 (False) | 1 | 89.2 | 0.0 | 585.1 / 585.0 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_vs_bf16_run0 | 33.69 | 0.9531 | 0.0346 | 0.998393 | 1.415e-03 | 24.93 |
| fp8_vs_bf16_run1 | 33.69 | 0.9531 | 0.0346 | 0.998393 | 1.415e-03 | 24.93 |
| bf16_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
| fp8_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
