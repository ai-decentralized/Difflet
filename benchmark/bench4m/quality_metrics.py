#!/usr/bin/env python3
"""Quality metrics: a device run against a CPU fp32 reference (latents and decoded frames).

Adapted from main's ``scripts/quality_vs_reference.py`` (540e5f0, copy kept as
``quality_vs_reference.main-540e5f0.py``). The metric definitions are unchanged:

* latents: relative L2 ||cand - ref|| / ||ref|| and cosine similarity, both in float64,
  on the packed loop-output latents (``final.pt``; ``final_video.pt`` / ``final_audio.pt``);
* frames: RGB uint8, PSNR per frame (mean and worst, peak 255), SSIM per Wang et al. 2004
  (11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, L 255, valid region only, computed
  per channel and averaged over the three channels), mean and worst over frames.

What changed against main's script (all additive):
* reads lossless ``.pt`` frame tensors as well as PNG (main read PNG / mp4); mp4 is refused,
  H.264 is not lossless;
* float frames are converted to uint8 with ONE rule for every source, the rule both device
  CLIs use for their PNG (torchvision ``save_image`` on AoT; native ``_save_image_pil``):
  ``uint8(clamp(x * 255 + 0.5, 0, 255))``, computed in float32 (bf16 is upcast exactly
  first), i.e. round half up of x*255 for x in [0, 1];
* per-frame SSIM, worst SSIM and per-channel SSIM are reported too;
* explicit file arguments, so a reference dir need not mirror a device dir's names.

Usage:
    python quality_metrics.py --ref-latents R/final.pt --cand-latents C/final.pt \
        --ref-frames R/out.pt --cand-frames C/out.png --out metrics.json
    python quality_metrics.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

UINT8_RULE = "uint8(clamp(float32(x) * 255 + 0.5, 0, 255)); x in [0, 1]"


# ----------------------------------------------------------------------------- latents
def latent_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    if tuple(ref.shape) != tuple(cand.shape):
        raise ValueError(f"latent shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    r = ref.detach().to(torch.float64).flatten()
    c = cand.detach().to(torch.float64).flatten()
    return {
        "rel_l2": float(torch.linalg.vector_norm(c - r) / torch.linalg.vector_norm(r)),
        "cosine": float(F.cosine_similarity(r, c, dim=0)),
        "ref_dtype": str(ref.dtype),
        "cand_dtype": str(cand.dtype),
        "shape": list(ref.shape),
    }


# ----------------------------------------------------------------------------- frames
def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Float frames in [0, 1] -> uint8, the device CLIs' PNG rule (see module doc)."""
    if x.dtype == torch.uint8:
        return x
    return x.detach().to(torch.float32).mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8)


def frames_from_tensor(t: torch.Tensor) -> np.ndarray:
    """Any of (3,H,W), (1,3,H,W), (F,3,H,W), (1,F,3,H,W) -> uint8 [F, H, W, 3]."""
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim == 5:
        if t.shape[0] != 1:
            raise ValueError(f"batch {t.shape[0]} frames tensor: expected batch 1")
        t = t[0]
    if t.ndim != 4 or t.shape[1] != 3:
        raise ValueError(f"frames tensor shape {tuple(t.shape)}: expected (..., F, 3, H, W)")
    return to_uint8(t).permute(0, 2, 3, 1).contiguous().numpy()


def read_frames(path: Path) -> tuple[np.ndarray, dict]:
    """uint8 [F, H, W, 3] plus a record of how it was read."""
    path = Path(path)
    if path.suffix == ".png":
        from PIL import Image

        with Image.open(path) as im:
            mode = im.mode
            arr = np.asarray(im.convert("RGB"))
        return arr[None], {"path": str(path), "format": "png", "png_mode": mode}
    if path.suffix == ".pt":
        t = torch.load(path, map_location="cpu")
        if not torch.is_tensor(t):
            raise ValueError(f"{path}: not a tensor ({type(t).__name__})")
        rec = {
            "path": str(path),
            "format": "pt",
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "min": float(t.float().min()),
            "max": float(t.float().max()),
            "uint8_rule": UINT8_RULE,
        }
        return frames_from_tensor(t), rec
    raise ValueError(f"{path}: only lossless .png / .pt frames are accepted (mp4 is lossy)")


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim_channels(ref: np.ndarray, cand: np.ndarray) -> list[float]:
    """SSIM of one uint8 HxWx3 frame pair, one value per channel (main's formula)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    x = torch.from_numpy(np.array(ref, copy=True)).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    y = torch.from_numpy(np.array(cand, copy=True)).to(torch.float64).permute(2, 0, 1).unsqueeze(1)
    w = _gaussian_window().unsqueeze(0).unsqueeze(0)
    mu_x = F.conv2d(x, w)
    mu_y = F.conv2d(y, w)
    var_x = F.conv2d(x * x, w) - mu_x**2
    var_y = F.conv2d(y * y, w) - mu_y**2
    cov = F.conv2d(x * y, w) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return [float(v) for v in ssim_map.mean(dim=(1, 2, 3))]


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def frame_metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    if ref.shape != cand.shape:
        raise ValueError(f"frame shape mismatch: {ref.shape} vs {cand.shape}")
    psnrs = [psnr(r, c) for r, c in zip(ref, cand)]
    per_ch = [ssim_channels(r, c) for r, c in zip(ref, cand)]
    ssims = [float(np.mean(ch)) for ch in per_ch]
    return {
        "num_frames": int(ref.shape[0]),
        "frame_hw": [int(ref.shape[1]), int(ref.shape[2])],
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_worst": float(np.min(psnrs)),
        "psnr_worst_frame": int(np.argmin(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_worst": float(np.min(ssims)),
        "ssim_worst_frame": int(np.argmin(ssims)),
        "ssim_mean_per_channel_rgb": [float(v) for v in np.mean(np.array(per_ch), axis=0)],
        "psnr_per_frame": psnrs,
        "ssim_per_frame": ssims,
    }


# ----------------------------------------------------------------------------- driver
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(pairs: list[tuple[str, Path, Path]], ref_frames: Path | None, cand_frames: Path | None) -> dict:
    out: dict = {"definitions": __doc__.split("Usage:")[0].strip(), "latents": {}, "inputs": {}}
    for name, rp, cp in pairs:
        out["inputs"][f"{name}.ref"] = {"path": str(rp), "sha256": _sha(rp)}
        out["inputs"][f"{name}.cand"] = {"path": str(cp), "sha256": _sha(cp)}
        out["latents"][name] = latent_metrics(torch.load(rp, map_location="cpu"), torch.load(cp, map_location="cpu"))
    if ref_frames is not None or cand_frames is not None:
        if ref_frames is None or cand_frames is None:
            raise ValueError("--ref-frames and --cand-frames go together")
        rf, rrec = read_frames(ref_frames)
        cf, crec = read_frames(cand_frames)
        rrec["sha256"] = _sha(ref_frames)
        crec["sha256"] = _sha(cand_frames)
        out["inputs"]["frames.ref"] = rrec
        out["inputs"]["frames.cand"] = crec
        out["frames"] = frame_metrics(rf, cf)
    return out


def selftest() -> int:
    """Checks the definitions on synthetic data, and SSIM against an independent scipy build."""
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest] {'ok  ' if cond else 'FAIL'} {name} {detail}")

    a = rng.integers(0, 256, size=(2, 64, 80, 3), dtype=np.uint8)
    m = frame_metrics(a, a.copy())
    check("identical frames: psnr inf, ssim 1", math.isinf(m["psnr_mean"]) and abs(m["ssim_mean"] - 1) < 1e-12)
    b = np.clip(a.astype(int) + rng.integers(-3, 4, size=a.shape), 0, 255).astype(np.uint8)
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2, axis=(1, 2, 3))
    want = [10 * math.log10(255**2 / v) for v in mse]
    got = frame_metrics(a, b)
    check("psnr per frame vs closed form", np.allclose(got["psnr_per_frame"], want, atol=1e-9))
    try:
        from scipy.ndimage import gaussian_filter

        def ssim_scipy(x, y):
            vals = []
            for ch in range(3):
                X = x[..., ch].astype(np.float64)
                Y = y[..., ch].astype(np.float64)
                kw = dict(sigma=1.5, truncate=3.5, mode="constant")
                mx, my = gaussian_filter(X, **kw), gaussian_filter(Y, **kw)
                vx = gaussian_filter(X * X, **kw) - mx * mx
                vy = gaussian_filter(Y * Y, **kw) - my * my
                cxy = gaussian_filter(X * Y, **kw) - mx * my
                c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
                s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
                vals.append(s[5:-5, 5:-5].mean())  # valid region of an 11x11 window
            return float(np.mean(vals))

        smooth = np.stack([gaussian_filter(a[0, ..., c].astype(float), 2) for c in range(3)], -1)
        smooth = np.clip(smooth * 1.0, 0, 255).astype(np.uint8)
        noisy = np.clip(smooth.astype(int) + rng.integers(-20, 21, size=smooth.shape), 0, 255).astype(np.uint8)
        mine = float(np.mean(ssim_channels(smooth, noisy)))
        ref = ssim_scipy(smooth, noisy)
        check("ssim vs independent scipy build", abs(mine - ref) < 1e-9, f"{mine:.12f} vs {ref:.12f}")
    except ImportError:
        print("[selftest] skip scipy cross-check (scipy missing)")
    x = torch.tensor([0.0, 0.5 / 255, 1.49 / 255, 1.5 / 255, 0.5, 1.0, 1.2, -0.1])
    check("uint8 rule", to_uint8(x).tolist() == [0, 1, 1, 2, 128, 255, 255, 0], str(to_uint8(x).tolist()))
    xb = torch.rand(1, 2, 3, 8, 8).to(torch.bfloat16)
    u = frames_from_tensor(xb)
    check("bf16 (1,F,3,H,W) -> uint8 (F,H,W,3)", u.shape == (2, 8, 8, 3) and u.dtype == np.uint8)
    r = torch.randn(1, 50, 16)
    lm = latent_metrics(r, r * 2)
    check("latent cosine of scaled copy = 1, rel_l2 = 1", abs(lm["cosine"] - 1) < 1e-12 and abs(lm["rel_l2"] - 1) < 1e-12)
    lm = latent_metrics(r, -r)
    check("latent cosine of negated copy = -1", abs(lm["cosine"] + 1) < 1e-12)
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--latents", nargs=3, action="append", default=[], metavar=("NAME", "REF", "CAND"),
                   help="a latent pair, repeatable: NAME REF.pt CAND.pt")
    p.add_argument("--ref-latents", type=Path)
    p.add_argument("--cand-latents", type=Path)
    p.add_argument("--ref-frames", type=Path)
    p.add_argument("--cand-frames", type=Path)
    p.add_argument("--label", default=None)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    pairs = [(n, Path(r), Path(c)) for n, r, c in a.latents]
    if a.ref_latents or a.cand_latents:
        pairs.append((a.ref_latents.stem, a.ref_latents, a.cand_latents))
    if not pairs and a.ref_frames is None:
        p.error("nothing to compare")
    res = compare(pairs, a.ref_frames, a.cand_frames)
    res["label"] = a.label
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))
    brief = {"label": a.label, "latents": {k: {kk: v[kk] for kk in ("rel_l2", "cosine")} for k, v in res["latents"].items()}}
    if "frames" in res:
        brief["frames"] = {k: v for k, v in res["frames"].items() if not k.endswith("_per_frame")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
