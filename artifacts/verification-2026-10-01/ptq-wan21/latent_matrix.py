"""Pairwise latent cosine / SNR between all arms' run-1 DiT output latents."""
import itertools

import torch

base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21"
arms = {
    "bf16": f"{base}/ab/work_bf16_run1/latents.pt",
    "dyn-pre": f"{base}/ab/work_fp8_run1/latents.pt",
    "wo-pre": f"{base}/ab-wo/work_fp8_run1/latents.pt",
    "dyn-fixed": f"{base}/ab-fixed/work_fp8_run1/latents.pt",
    "wo-fixed": f"{base}/ab-wo-fixed/work_fp8_run1/latents.pt",
}
lat = {k: torch.load(v, map_location="cpu").float().flatten() for k, v in arms.items()}
for k, v in lat.items():
    print(f"{k:10s} mean={v.mean():+.4f} std={v.std():.4f} absmax={v.abs().max():.3f}")
print()
print(f"{'':10s}" + "".join(f"{k:>18s}" for k in arms))
for a in arms:
    row = []
    for b in arms:
        x, y = lat[a], lat[b]
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        snr = 10 * torch.log10(x.pow(2).mean() / ((x - y).pow(2).mean() + 1e-30)).item()
        row.append(f"{cos:.5f}/{snr:5.1f}dB")
    print(f"{a:10s}" + "".join(f"{r:>18s}" for r in row))
