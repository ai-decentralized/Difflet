"""One line per quality_metrics.py JSON: PSNR / SSIM (frames) and rel-L2 / cosine (each latent)."""

import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    parts = []
    f = d.get("frames")
    if f:
        parts.append(f"psnr {f['psnr_mean']:.3f} ssim {f['ssim_mean']:.4f}")
    for name, m in (d.get("latents") or {}).items():
        parts.append(f"{name}: cos {m['cosine']:.5f} rel_l2 {m['rel_l2']:.4f}")
    print(f"{path.rsplit('/', 1)[-1]:32s} " + " | ".join(parts))
