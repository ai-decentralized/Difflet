#!/usr/bin/env python3
"""Compare a candidate run against a reference run: latents and decoded frames.

Each run directory holds the denoise-loop dumps written under
``DIFFLET_DUMP_LATENTS_DIR`` (``final.pt`` for Qwen-Image; ``final_video.pt`` and
``final_audio.pt`` for LTX-2) and the decoded output (``out.png`` or
``out.mp4``). Metrics follow the Wan2.1 backend report:

* latents: relative L2 error ||cand - ref|| / ||ref|| and cosine similarity,
  computed in float64;
* frames: per-frame PSNR (mean and worst) and SSIM (mean) on uint8 RGB.
  SSIM is Wang et al. 2004: 11x11 Gaussian window (sigma 1.5), K1=0.01,
  K2=0.03, valid region only, averaged over channels.

Usage:
    python scripts/quality_vs_reference.py --ref-dir R --cand-dir C --out metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def latent_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    if ref.shape != cand.shape:
        raise ValueError(f"latent shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    r = ref.detach().to(torch.float64).flatten()
    c = cand.detach().to(torch.float64).flatten()
    return {
        "rel_l2": float(torch.linalg.vector_norm(c - r) / torch.linalg.vector_norm(r)),
        "cosine": float(F.cosine_similarity(r, c, dim=0)),
    }


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def _ssim(ref: np.ndarray, cand: np.ndarray) -> float:
    """SSIM of one uint8 HxWx3 frame pair."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    x = torch.from_numpy(ref).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    y = torch.from_numpy(cand).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    w = _gaussian_window().unsqueeze(0).unsqueeze(0)
    mu_x = F.conv2d(x, w)
    mu_y = F.conv2d(y, w)
    var_x = F.conv2d(x * x, w) - mu_x**2
    var_y = F.conv2d(y * y, w) - mu_y**2
    cov = F.conv2d(x * y, w) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim_map.mean())


def _psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def frame_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    """PSNR mean / worst and SSIM mean over uint8 [F, H, W, 3] frame stacks."""
    if ref.shape != cand.shape:
        raise ValueError(f"frame shape mismatch: {ref.shape} vs {cand.shape}")
    psnrs = [_psnr(r, c) for r, c in zip(ref, cand)]
    ssims = [_ssim(r, c) for r, c in zip(ref, cand)]
    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_worst": float(np.min(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "psnr_per_frame": psnrs,
    }


def _read_frames(run_dir: Path) -> np.ndarray | None:
    png = run_dir / "out.png"
    mp4 = run_dir / "out.mp4"
    if png.exists():
        import imageio.v3 as iio

        return np.asarray(iio.imread(png))[None, ..., :3]
    if mp4.exists():
        import imageio.v3 as iio

        return np.stack([frame[..., :3] for frame in iio.imiter(mp4)])
    return None


def compare_runs(ref_dir: Path, cand_dir: Path) -> dict:
    result: dict = {"ref_dir": str(ref_dir), "cand_dir": str(cand_dir), "latents": {}}
    for ref_path in sorted(ref_dir.glob("final*.pt")):
        cand_path = cand_dir / ref_path.name
        if cand_path.exists():
            result["latents"][ref_path.stem] = latent_metrics(
                torch.load(ref_path), torch.load(cand_path)
            )
    ref_frames, cand_frames = _read_frames(ref_dir), _read_frames(cand_dir)
    if ref_frames is not None and cand_frames is not None:
        result["frames"] = frame_metrics(ref_frames, cand_frames)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.ref_dir, args.cand_dir)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    if "frames" in result:
        frames = {k: v for k, v in result["frames"].items() if k != "psnr_per_frame"}
        print(json.dumps(frames, indent=2))


if __name__ == "__main__":
    main()
