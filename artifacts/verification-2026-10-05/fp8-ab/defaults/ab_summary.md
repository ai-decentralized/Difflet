# FP8 PTQ A/B — Wan-AI/Wan2.1-T2V-14B-Diffusers

shape 480x832x9, 20 steps, guidance 1.0, seed 42, tp4; quant {'format': 'fp8', 'weight_granularity': 'tensor', 'calibration': '/home/ubuntu/Difflet/.claude/worktrees/fp8-step-experiments/artifacts/verification-2026-10-03/fp8-dyn-step/static/act_calibration_wan21.json'}

| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | — | 332.2 (False) | 0 | 179.2 | 0.0 | 486.3 / 486.2 (19) |
| bf16 | — | 332.2 (False) | 1 | 89.5 | 0.0 | 486.1 / 485.9 (19) |
| fp8 | 294.0 | 354.6 (False) | 0 | 86.2 | 0.0 | 473.0 / 473.1 (19) |
| fp8 | 294.0 | 354.6 (False) | 1 | 89.3 | 0.0 | 472.9 / 472.8 (19) |

| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| fp8_vs_bf16_run0 | 33.69 | 0.9531 | 0.0346 | 0.998393 | 1.415e-03 | 24.93 |
| fp8_vs_bf16_run1 | 33.69 | 0.9531 | 0.0346 | 0.998393 | 1.415e-03 | 24.93 |
| bf16_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
| fp8_run1_vs_run0_control | inf | 1.0000 | 0.0000 | 1.000000 | 0.000e+00 | inf |
