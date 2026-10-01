import json
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21"
for d in ("ab", "ab-fixed", "ab-wo-fixed"):
    o = json.load(open(f"{base}/{d}/compare_hostvae_fp8_vs_bf16.json"))["output"]
    print(f"{d:12s} host-VAE fp8 vs bf16: psnr {o['psnr_db']:.2f} ssim {o['ssim']:.4f} lpips {o['lpips']:.4f} pixel cosine {o['pixel']['cosine']:.5f}")
