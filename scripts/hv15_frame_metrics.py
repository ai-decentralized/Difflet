#!/usr/bin/env python3
"""Per-frame PSNR (dB) and SSIM between two decoded videos of the same shape, values in [0, 1].

Each input is a ``.pt`` tensor, ``[F, C, H, W]`` or the VAE's ``[1, C, F, H, W]``, or an MP4 (read
as uint8 and scaled to [0, 1]). PSNR uses a peak of 1.0. SSIM is Wang et al. 2004: an 11x11
Gaussian window (sigma 1.5), K1 = 0.01, K2 = 0.03, valid region only, averaged over channels.
Computed in float64. Prints the mean and minimum of each, then one line per frame.

    python scripts/hv15_frame_metrics.py reference.pt candidate.pt [--json metrics.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

_K1, _K2, _WINDOW, _SIGMA = 0.01, 0.03, 11, 1.5


def load_frames(path: str | Path) -> torch.Tensor:
    """A video as a float64 ``[F, C, H, W]`` tensor in [0, 1]."""
    path = Path(path)
    if path.suffix == ".pt":
        x = torch.load(path, map_location="cpu", weights_only=True)
        if x.dim() == 5:  # the VAE's [1, C, F, H, W]
            if x.shape[0] != 1:
                raise ValueError(f"{path}: batch {x.shape[0]}; pass one video")
            x = x[0].permute(1, 0, 2, 3)
        if x.dim() != 4:
            raise ValueError(f"{path}: expected [F, C, H, W] or [1, C, F, H, W], got {tuple(x.shape)}")
        return x.to(torch.float64)
    import imageio.v3 as iio
    import numpy as np

    frames = np.stack([f[..., :3] for f in iio.imiter(path)])  # [F, H, W, 3] uint8
    return torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float64) / 255.0


def _window(channels: int) -> torch.Tensor:
    coords = torch.arange(_WINDOW, dtype=torch.float64) - (_WINDOW - 1) / 2
    g = torch.exp(-(coords**2) / (2 * _SIGMA**2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, _WINDOW, _WINDOW).contiguous()


def psnr(ref: torch.Tensor, cand: torch.Tensor) -> float:
    """One frame pair, peak 1.0; inf when identical."""
    mse = float(((ref - cand) ** 2).mean())
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim(ref: torch.Tensor, cand: torch.Tensor) -> float:
    """One ``[C, H, W]`` frame pair in [0, 1]: the mean SSIM map over the valid region and channels."""
    c = ref.shape[0]
    w = _window(c)
    x, y = ref.unsqueeze(0), cand.unsqueeze(0)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    var_x = F.conv2d(x * x, w, groups=c) - mu_x**2
    var_y = F.conv2d(y * y, w, groups=c) - mu_y**2
    cov = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = _K1**2, _K2**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())


def frame_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    """Per-frame PSNR and SSIM of two ``[F, C, H, W]`` videos in [0, 1], with their mean and min."""
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    for name, x in (("reference", ref), ("candidate", cand)):
        lo, hi = float(x.min()), float(x.max())
        if lo < -1e-6 or hi > 1 + 1e-6:
            raise ValueError(f"{name} values span [{lo:.4f}, {hi:.4f}]; expected [0, 1]")
    ref, cand = ref.to(torch.float64), cand.to(torch.float64)
    psnrs = [psnr(r, c) for r, c in zip(ref, cand)]
    ssims = [ssim(r, c) for r, c in zip(ref, cand)]
    return {"frames": len(psnrs), "psnr_mean": sum(psnrs) / len(psnrs), "psnr_min": min(psnrs),
            "ssim_mean": sum(ssims) / len(ssims), "ssim_min": min(ssims),
            "psnr_per_frame": psnrs, "ssim_per_frame": ssims}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--json", default=None, help="also write the metrics here")
    a = p.parse_args()
    m = frame_metrics(load_frames(a.reference), load_frames(a.candidate))
    print(f"frames {m['frames']}: PSNR mean {m['psnr_mean']:.3f} dB, min {m['psnr_min']:.3f} dB; "
          f"SSIM mean {m['ssim_mean']:.5f}, min {m['ssim_min']:.5f}")
    for i, (pv, sv) in enumerate(zip(m["psnr_per_frame"], m["ssim_per_frame"])):
        print(f"frame {i:3d}: PSNR {pv:.3f} dB  SSIM {sv:.5f}")
    if a.json:
        Path(a.json).write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
