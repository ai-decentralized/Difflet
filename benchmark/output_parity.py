"""Pixel-space parity of two generated outputs (same seed): PNG or MP4.

    python -m benchmark.output_parity a.png b.png
    python -m benchmark.output_parity a.mp4 b.mp4

Reports PSNR (dB), mean |diff| (0-255 scale), and SSIM for images when
scikit-image is available; for videos the metrics are averaged over frames.
A TeaCache / attention-impl variant that returns a byte-identical output is a
no-op (the skill's "silent pass" trap), so ``identical`` is reported too.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _load(path: Path) -> np.ndarray:
    """uint8 array: [H, W, 3] for images, [F, H, W, 3] for videos."""
    if path.suffix.lower() == ".mp4":
        import imageio.v3 as iio
        return np.asarray(iio.imread(str(path), plugin="pyav"))
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def _ssim(a: np.ndarray, b: np.ndarray):
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def compare(a_path: str | Path, b_path: str | Path) -> dict:
    a, b = _load(Path(a_path)), _load(Path(b_path))
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}"}
    frames_a = a if a.ndim == 4 else a[None]
    frames_b = b if b.ndim == 4 else b[None]
    psnr = [_psnr(x, y) for x, y in zip(frames_a, frames_b)]
    ssim = [_ssim(x, y) for x, y in zip(frames_a, frames_b)]
    out = {
        "frames": int(frames_a.shape[0]),
        "identical": bool(np.array_equal(a, b)),
        "psnr_db": round(float(np.mean([p for p in psnr if math.isfinite(p)] or [float("inf")])), 2)
                   if not all(math.isinf(p) for p in psnr) else float("inf"),
        "psnr_min_db": round(min(psnr), 2) if not all(math.isinf(p) for p in psnr) else float("inf"),
        "mean_abs_diff": round(float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))), 3),
    }
    if all(s is not None for s in ssim):
        out["ssim"] = round(float(np.mean(ssim)), 4)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    a = p.parse_args()
    print(json.dumps(compare(a.a, a.b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
