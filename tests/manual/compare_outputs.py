"""PSNR and SSIM between generated outputs: images (PNG) or decoded video tensors (.pt).

Each ``--pair NAME A B`` compares two outputs of the same prompt and seed; video tensors are
compared frame by frame and averaged. Pixels are scaled to [0, 1] first (PNG: /255; tensors: their
[-1, 1] range mapped to [0, 1] when negative values are present).

    python tests/manual/compare_outputs.py --pair "flux s42 aot-vs-native" a.png b.png --pair ...

SSIM: the standard Gaussian-window form (11x11, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2) on the
luma channel.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn.functional as F


def load(path: str) -> torch.Tensor:
    """[N, 3, H, W] in [0, 1] (N = 1 for an image)."""
    if path.endswith(".pt"):
        t = torch.load(path, map_location="cpu").float()
        while t.dim() > 5:
            t = t[0]
        if t.dim() == 5:  # [B, C, F, H, W] -> frames of the first batch item
            t = t[0].permute(1, 0, 2, 3)
        elif t.dim() == 4 and t.shape[0] == 3 and t.shape[1] != 3:  # [C, F, H, W]
            t = t.permute(1, 0, 2, 3)
        if t.min() < 0:
            t = (t + 1) / 2
        return t.clamp(0, 1)
    from PIL import Image

    img = Image.open(path).convert("RGB")
    x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(img.height, img.width, 3)
    return (x.permute(2, 0, 1).float() / 255.0)[None]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    luma = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    x, y = (a * luma).sum(1, keepdim=True), (b * luma).sum(1, keepdim=True)
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :]) / (g.sum() ** 2)
    w = w.view(1, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x ** 2
    syy = F.conv2d(y * y, w) - mu_y ** 2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "A", "B"), required=True)
    args = p.parse_args()
    for name, pa, pb in args.pair:
        a, b = load(pa), load(pb)
        if a.shape != b.shape:
            print(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
            continue
        per = [(psnr(a[i:i + 1], b[i:i + 1]), ssim(a[i:i + 1], b[i:i + 1])) for i in range(a.shape[0])]
        mp = sum(v[0] for v in per) / len(per)
        ms = sum(v[1] for v in per) / len(per)
        worst = min(v[0] for v in per)
        print(f"{name:<34} PSNR {mp:6.2f} dB (worst frame {worst:6.2f})  SSIM {ms:.4f}  frames={a.shape[0]}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
