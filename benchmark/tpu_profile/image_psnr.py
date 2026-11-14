"""PSNR / mean abs difference between saved bench images, with their sha256.

    python -m benchmark.tpu_profile.image_psnr A.png B.png [C.png D.png ...]

Pairs are taken in order (A vs B, C vs D, ...). Images are compared as 8-bit
RGB; identical files report ``inf``. Prints each file's sha256 so a reported
number can be tied to the exact images it came from.
"""

from __future__ import annotations

import hashlib
import math
import sys

import numpy as np
from PIL import Image


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def _sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:12]


def compare(a_path: str, b_path: str) -> dict:
    a, b = _load(a_path), _load(b_path)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(((a - b) ** 2).mean())
    return {
        "psnr_db": math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse),
        "mean_abs": float(np.abs(a - b).mean()),
        "max_abs": float(np.abs(a - b).max()),
        "sha_a": _sha(a_path),
        "sha_b": _sha(b_path),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    for a_path, b_path in zip(argv[0::2], argv[1::2]):
        r = compare(a_path, b_path)
        print(f"{a_path} vs {b_path}: PSNR {r['psnr_db']:.2f} dB, mean |d| {r['mean_abs']:.3f}/255, "
              f"max {r['max_abs']:.0f}  (sha256 {r['sha_a']} / {r['sha_b']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
