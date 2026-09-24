#!/usr/bin/env python3
"""Pixel-space quality of a cached generation against its caching-off reference.

The paper's caching table reports PSNR against the same-seed output with caching
disabled. This computes that, plus SSIM, for either an image or a video; for
video both metrics are averaged over frames, which is what the paper's footnote
describes.

    python scripts/psnr_compare.py \
      --reference cclogs/caching-official-steps/wan/off/rep1.mp4 \
      --candidate cclogs/caching-official-steps/wan/cadence2/rep1.mp4

    # every mode of one model against that model's own caching-off output
    python scripts/psnr_compare.py --sweep cclogs/caching-official-steps/wan

The reference must be the caching-off run at the SAME step count, seed and
prompt: a 50-step reference against a 20-step candidate measures the step count,
not the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".gif"}


def _load_frames(path: Path) -> np.ndarray:
    """Return uint8 frames shaped (T, H, W, C); an image is a single frame."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)[None]
    if suffix in VIDEO_SUFFIXES:
        import imageio.v3 as iio

        frames = iio.imread(path, plugin="pyav")
        frames = np.asarray(frames, dtype=np.uint8)
        if frames.ndim == 3:  # a single-frame video decodes to (H, W, C)
            frames = frames[None]
        return frames[..., :3]
    raise SystemExit(f"unsupported file type: {path}")


def _psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    """PSNR over 8-bit pixels, the paper's definition: 10 log10(255^2 / MSE)."""
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10(255.0**2 / mse))


def _ssim(ref: np.ndarray, cand: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ModuleNotFoundError:
        return None
    return float(
        structural_similarity(ref, cand, channel_axis=-1, data_range=255)
    )


def compare(reference: Path, candidate: Path) -> dict:
    ref = _load_frames(reference)
    cand = _load_frames(candidate)
    if ref.shape != cand.shape:
        raise SystemExit(
            f"shape mismatch: reference {ref.shape} vs candidate {cand.shape}. "
            "Both runs must use the same resolution, frame count and step count."
        )

    # Per frame, then averaged -- averaging the metric rather than pooling the
    # MSE is what the paper's "averaged over frames for video" footnote says.
    psnrs = [_psnr(r, c) for r, c in zip(ref, cand)]
    ssims = [s for s in (_ssim(r, c) for r, c in zip(ref, cand)) if s is not None]

    result = {
        "reference": str(reference),
        "candidate": str(candidate),
        "frames": int(ref.shape[0]),
        "psnr_db": round(float(np.mean(psnrs)), 2),
        "psnr_min_db": round(float(np.min(psnrs)), 2),
    }
    if ssims:
        result["ssim"] = round(float(np.mean(ssims)), 4)
    else:
        result["ssim"] = None
        result["ssim_note"] = "scikit-image not installed; PSNR alone invites reviewer pushback"
    return result


def _sweep(model_dir: Path) -> list[dict]:
    off = model_dir / "off"
    if not off.is_dir():
        raise SystemExit(f"no caching-off reference under {off}")
    references = sorted(p for p in off.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)
    if not references:
        raise SystemExit(f"no generated output in {off}")
    reference = references[0]

    rows = []
    for mode_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != "off"):
        candidates = sorted(
            p for p in mode_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
        )
        if not candidates:
            continue
        row = compare(reference, candidates[0])
        row["mode"] = mode_dir.name
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, help="the caching-off output, same seed and step count")
    p.add_argument("--candidate", type=Path, help="the cached output to score")
    p.add_argument("--sweep", type=Path, help="a model directory holding off/ and the cached modes")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args()

    if args.sweep:
        rows = _sweep(args.sweep)
    elif args.reference and args.candidate:
        rows = [compare(args.reference, args.candidate)]
    else:
        p.error("pass --sweep DIR, or both --reference and --candidate")

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    width = max((len(r.get("mode", "")) for r in rows), default=4)
    print(f"{'mode'.ljust(width)}  {'PSNR (dB)':>10}  {'min':>7}  {'SSIM':>7}  frames")
    for r in rows:
        ssim = f"{r['ssim']:.4f}" if r.get("ssim") is not None else "n/a"
        print(
            f"{r.get('mode', '-').ljust(width)}  {r['psnr_db']:>10.2f}  "
            f"{r['psnr_min_db']:>7.2f}  {ssim:>7}  {r['frames']}"
        )
    if any(r.get("ssim") is None for r in rows):
        print("\nnote: scikit-image not installed, so SSIM was skipped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
