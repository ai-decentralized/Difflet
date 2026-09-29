"""Comparison metrics for quantization studies (pure torch; optional ``lpips``).

Tensor level: MSE, mean/max abs, cosine, relative L2, SNR(dB) — the same set
FastVideo reports per kernel call, plus the latent-level SNR from its MLX
benchmark. Output level: PSNR, SSIM (11×11 Gaussian window, σ=1.5, per frame,
averaged) and LPIPS (``alex`` backbone, the FastVideo default) between a
reference and a test image / video / tensor file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
_TENSOR_SUFFIXES = {".pt", ".pth"}


# ----------------------------------------------------------------- tensor level


def tensor_error_metrics(reference: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """Error of ``test`` against ``reference`` (any matching shapes; fp64 math)."""
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: reference {tuple(reference.shape)} vs test {tuple(test.shape)}")
    ref = reference.detach().to(torch.float64).reshape(-1)
    tst = test.detach().to(torch.float64).reshape(-1)
    diff = tst - ref
    mse = float(diff.pow(2).mean()) if diff.numel() else 0.0
    ref_power = float(ref.pow(2).mean()) if ref.numel() else 0.0
    ref_norm = float(ref.norm())
    cosine = float(F.cosine_similarity(ref.unsqueeze(0), tst.unsqueeze(0)).item()) if ref.numel() else 1.0
    return {
        "mse": mse,
        "mean_abs": float(diff.abs().mean()) if diff.numel() else 0.0,
        "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
        "cosine": cosine,
        "rel_l2": float(diff.norm()) / ref_norm if ref_norm > 0 else (0.0 if mse == 0 else math.inf),
        "snr_db": 10.0 * math.log10(ref_power / mse) if mse > 0 and ref_power > 0 else math.inf,
    }


# ------------------------------------------------------------------ image level


def _as_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected [N, C, H, W] (or [C, H, W]) images, got {tuple(x.shape)}")
    return x.to(torch.float32)


def psnr(a: torch.Tensor, b: torch.Tensor, *, data_range: float = 1.0) -> float:
    """Mean per-image PSNR in dB over ``[N, C, H, W]`` batches; ``inf`` when identical."""
    a, b = _as_nchw(a), _as_nchw(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    mse = (a - b).pow(2).flatten(1).mean(dim=1)
    values = [
        math.inf if m <= 0 else 10.0 * math.log10(data_range * data_range / m)
        for m in mse.tolist()
    ]
    return float(sum(values) / len(values)) if values else math.inf


def _gaussian_window(size: int, sigma: float, channels: int, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).to(dtype)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean per-image SSIM (Wang et al.; Gaussian window, valid region)."""
    a, b = _as_nchw(a), _as_nchw(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    channels = a.shape[1]
    window = _gaussian_window(window_size, sigma, channels, a.dtype)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=channels)

    mu_a, mu_b = blur(a), blur(b)
    sigma_a = blur(a * a) - mu_a * mu_a
    sigma_b = blur(b * b) - mu_b * mu_b
    sigma_ab = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(ssim_map.flatten(1).mean(dim=1).mean())


def lpips_distance(
    a: torch.Tensor, b: torch.Tensor, *, net: str = "alex", batch_size: int = 8
) -> float | None:
    """Mean LPIPS over ``[N, 3, H, W]`` frames in [0, 1]; ``None`` when the
    optional ``lpips`` package is not installed."""
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    a, b = _as_nchw(a), _as_nchw(b)
    model = lpips.LPIPS(net=net, verbose=False).eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, a.shape[0], batch_size):
            xa = a[start : start + batch_size] * 2.0 - 1.0
            xb = b[start : start + batch_size] * 2.0 - 1.0
            values.extend(model(xa, xb).flatten().tolist())
    return float(sum(values) / len(values)) if values else None


# ------------------------------------------------------------------- media I/O


def _normalize_frames(x: torch.Tensor) -> torch.Tensor:
    """Any image/video tensor layout -> ``[N, C, H, W]`` float in [0, 1]."""
    x = x.detach()
    if x.ndim == 5:  # [B, C, T, H, W] (Wan / diffusers video) -> [T, C, H, W]
        if x.shape[0] != 1:
            raise ValueError(f"batched video tensors are not supported: {tuple(x.shape)}")
        x = x[0].permute(1, 0, 2, 3)
    elif x.ndim == 4 and x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    elif x.ndim == 3:
        x = x.unsqueeze(0) if x.shape[0] in (1, 3, 4) else x.permute(2, 0, 1).unsqueeze(0)
    x = x.to(torch.float32)
    if x.numel() == 0:
        return x
    if float(x.min()) < 0.0:
        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    elif float(x.max()) > 1.5:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def load_media(path: str | Path) -> torch.Tensor:
    """Load an image, a video, or a saved tensor as ``[N, 3, H, W]`` in [0, 1]."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            array = np.asarray(img.convert("RGB"))
        return _normalize_frames(torch.from_numpy(array.copy()))
    if suffix in _VIDEO_SUFFIXES:
        import av
        import numpy as np

        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"no video frames decoded from {path}")
        return _normalize_frames(torch.from_numpy(np.stack(frames)))
    if suffix in _TENSOR_SUFFIXES:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, (list, tuple)):
            data = data[0]
        if not torch.is_tensor(data):
            raise ValueError(f"{path} does not hold a tensor")
        return _normalize_frames(data)
    raise ValueError(f"unsupported media file: {path}")


def compare_outputs(
    reference: str | Path, test: str | Path, *, lpips_net: str | None = "alex"
) -> dict[str, Any]:
    """PSNR / SSIM / LPIPS of ``test`` vs ``reference`` (image, video or tensor file)."""
    ref = load_media(reference)
    tst = load_media(test)
    if ref.shape != tst.shape:
        raise ValueError(
            f"reference {tuple(ref.shape)} and test {tuple(tst.shape)} differ in frames/size"
        )
    result: dict[str, Any] = {
        "reference": str(reference),
        "test": str(test),
        "frames": int(ref.shape[0]),
        "height": int(ref.shape[2]),
        "width": int(ref.shape[3]),
        "psnr_db": psnr(ref, tst),
        "ssim": ssim(ref, tst),
        "pixel": tensor_error_metrics(ref, tst),
    }
    result["lpips"] = lpips_distance(ref, tst, net=lpips_net) if lpips_net else None
    result["lpips_net"] = lpips_net if result["lpips"] is not None else None
    return result


def compare_latents(reference: str | Path, test: str | Path) -> dict[str, Any]:
    """Tensor-level error of two saved latent tensors (e.g. Wan ``latents.pt``)."""
    ref = torch.load(reference, map_location="cpu")
    tst = torch.load(test, map_location="cpu")
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    tst = tst[0] if isinstance(tst, (list, tuple)) else tst
    return {
        "reference": str(reference),
        "test": str(test),
        "shape": list(ref.shape),
        **tensor_error_metrics(ref, tst),
    }


__all__ = [
    "compare_latents",
    "compare_outputs",
    "load_media",
    "lpips_distance",
    "psnr",
    "ssim",
    "tensor_error_metrics",
]
