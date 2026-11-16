"""The ``quality`` metric: a candidate run's dumps against a reference's (design §7, quality).

Compared on what the engine produced before any file encoder touched it: the final latents
(``final*.pt``, relative L2 and cosine in float64) and the pre-encode uint8 frames (``frames.npy``:
PSNR and SSIM per frame, with the mean, the minimum and the worst frame's index). The output file's
hash is recorded but never compared, since an MP4 depends on the encoder build. The formulas are
``scripts/quality_vs_reference.py``'s (SSIM: Wang et al. 2004, 11x11 Gaussian window, sigma 1.5,
K1=0.01, K2=0.03, valid region, averaged over channels; PSNR peak 255), so old and new numbers mean
the same; ``ssim_min`` and the worst-frame indices are new.

torch and numpy are imported inside the functions: ``difflet bench`` itself stays light.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class QualityError(RuntimeError):
    """A reference or candidate dump is missing or does not match: no quality number."""


def latent_metrics(ref, cand) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if ref.shape != cand.shape:
        raise QualityError(f"latent shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}")
    r = ref.detach().to(torch.float64).flatten()
    c = cand.detach().to(torch.float64).flatten()
    norm = torch.linalg.vector_norm(r)
    return {
        "rel_l2": float(torch.linalg.vector_norm(c - r) / norm) if norm > 0 else math.nan,
        "cosine": float(F.cosine_similarity(r, c, dim=0)),
        "bit_identical": bool(torch.equal(ref, cand)),
    }


def _gaussian_window(size: int = 11, sigma: float = 1.5):
    import torch

    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(ref, cand) -> float:
    """SSIM of one uint8 HxWx3 frame pair."""
    import torch
    import torch.nn.functional as F

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
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())


def psnr(ref, cand) -> float:
    import numpy as np

    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


def frame_metrics(ref, cand) -> dict[str, Any]:
    """PSNR and SSIM over uint8 [F, H, W, 3] stacks: mean, min, worst frame, per frame."""
    import numpy as np

    if ref.shape != cand.shape:
        raise QualityError(f"frame shape mismatch: {ref.shape} vs {cand.shape}")
    psnrs = [psnr(r, c) for r, c in zip(ref, cand)]
    ssims = [ssim(r[..., :3], c[..., :3]) for r, c in zip(ref, cand)]
    return {
        "frames": int(ref.shape[0]),
        "psnr_mean": float(np.mean(psnrs)), "psnr_min": float(np.min(psnrs)),
        "psnr_worst_frame": int(np.argmin(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_min": float(np.min(ssims)),
        "ssim_worst_frame": int(np.argmin(ssims)),
        "psnr_per_frame": psnrs, "ssim_per_frame": ssims,
        "identical": bool(np.array_equal(ref, cand)),
    }


def read_frames(dump_dir: Path):
    """``frames.npy`` and its ``frames.json`` from a dump dir; QualityError when absent."""
    import numpy as np

    npy, meta = Path(dump_dir) / "frames.npy", Path(dump_dir) / "frames.json"
    if not npy.exists() or not meta.exists():
        raise QualityError(f"no pre-encode frame dump (frames.npy/frames.json) in {dump_dir}")
    frames = np.load(npy)
    info = json.loads(meta.read_text())
    digest = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
    if info.get("sha256") != digest:
        raise QualityError(f"{npy} does not match the sha256 in {meta}")
    return frames, info


def dump_summary(dump_dir: Path) -> dict[str, Any]:
    """What a candidate's dump holds, with hashes (recorded even without a reference)."""
    out: dict[str, Any] = {"dir": str(dump_dir), "latents": {}, "frames": None}
    for path in sorted(Path(dump_dir).glob("*.pt")):
        out["latents"][path.stem] = {"bytes": path.stat().st_size,
                                     "sha256": _file_sha256(path)}
    meta = Path(dump_dir) / "frames.json"
    if meta.exists():
        out["frames"] = json.loads(meta.read_text())
    return out


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_dumps(ref_dir: Path, cand_dir: Path) -> dict[str, Any]:
    """Every reference ``final*.pt`` and the frame dumps, candidate against reference.

    Missing files raise QualityError: a run without its dumps must not look complete. The two frame
    dumps must use the same quantization rule, or the comparison mixes two roundings.
    """
    import torch

    ref_dir, cand_dir = Path(ref_dir), Path(cand_dir)
    ref_latents = sorted(ref_dir.glob("final*.pt"))
    if not ref_latents:
        raise QualityError(f"no final*.pt latents in the reference {ref_dir}")
    result: dict[str, Any] = {"latents": {}}
    for ref_path in ref_latents:
        cand_path = cand_dir / ref_path.name
        if not cand_path.exists():
            raise QualityError(f"{cand_path} missing (the reference has {ref_path.name})")
        result["latents"][ref_path.stem] = latent_metrics(
            torch.load(ref_path, map_location="cpu"), torch.load(cand_path, map_location="cpu"))
    ref_frames, ref_info = read_frames(ref_dir)
    cand_frames, cand_info = read_frames(cand_dir)
    if ref_info.get("rule") != cand_info.get("rule"):
        raise QualityError(f"frame rules differ: reference {ref_info.get('rule')!r}, candidate "
                           f"{cand_info.get('rule')!r}")
    result["frames"] = frame_metrics(ref_frames, cand_frames)
    result["frames"]["rule"] = cand_info.get("rule")
    return result
