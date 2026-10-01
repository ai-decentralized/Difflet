"""Per-latent-frame statistics of the DiT output latents (before the VAE) for both arms."""
import sys

import torch

base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21/ab"
lat = {}
for arm in ("bf16", "fp8"):
    z = torch.load(f"{base}/work_{arm}_run1/latents.pt", map_location="cpu").float()
    lat[arm] = z
    print(arm, "latents", tuple(z.shape), z.dtype, "nonfinite", int((~torch.isfinite(z)).sum()))
    for f in range(z.shape[2]):
        zf = z[0, :, f]
        print(f"  latent frame {f}: mean={zf.mean():+.4f} std={zf.std():.4f} absmax={zf.abs().max():.3f}")
a, b = lat["bf16"], lat["fp8"]
for f in range(a.shape[2]):
    x, y = a[0, :, f].flatten(), b[0, :, f].flatten()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    snr = 10 * torch.log10(x.pow(2).mean() / (x - y).pow(2).mean()).item()
    print(f"frame {f}: fp8 vs bf16 cosine={cos:.6f} snr={snr:.2f} dB")
