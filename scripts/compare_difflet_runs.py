#!/usr/bin/env python3
"""Compare two ``difflet generate`` runs (e.g. the bf16 and the FP8 DiT from one seed): the
packed latents the generate stage wrote (``<work-dir>/latents.pt``) and the decoded outputs.

Metrics as ``scripts/quality_vs_reference.py`` (whose functions this reuses): latent relative
L2 and cosine in float64; PSNR and SSIM (Wang 2004) on uint8 RGB. LPIPS (AlexNet) when the
``lpips`` package is installed, else reported as null.

A decoded output is a PNG, an ``.mp4`` or a directory of frame PNGs (in name order). A clip is
read frame by frame with ``imageio`` (the library diffusers' ``export_to_video`` writes it
with, over the ``imageio-ffmpeg`` binary); each metric is computed per frame and averaged over
the frames, with ``frames`` and ``min_psnr_frame`` (the worst frame's index) beside them. PSNR
is ``Infinity`` for identical frames, so an identical pair is the determinism signal for a clip
as for an image.

    python scripts/compare_difflet_runs.py \\
        --ref-latents B/work_gen50a/latents.pt --ref-image B/gen50a.png \\
        --cand-latents F/work_gen50a/latents.pt --cand-image F/gen50a.png --out metrics.json
    python scripts/compare_difflet_runs.py --ref-image B/out.mp4 --cand-image F/out.mp4
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


@functools.cache
def _qvr():
    spec = importlib.util.spec_from_file_location("qvr", REPO / "scripts" / "quality_vs_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _is_video(path: Path) -> bool:
    """An .mp4 or a directory of frame PNGs; a PNG is the one-frame case, reported as before."""
    return path.is_dir() or path.suffix.lower() == ".mp4"


def _read_frames(path: Path) -> np.ndarray:
    """A decoded output as uint8 frames [F, H, W, 3]: every frame of an .mp4, the PNGs of a
    directory in name order, or the one image."""
    if path.is_dir():
        pngs = sorted(path.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"no frame PNGs in {path}")
        return np.stack([_image(png) for png in pngs])
    if path.suffix.lower() == ".mp4":
        import imageio.v3 as iio

        return np.stack([frame[..., :3] for frame in iio.imiter(path)])
    return _image(path)[None]


def _lpips_net():
    """The AlexNet LPIPS model, or None when the ``lpips`` package is not installed."""
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    return lpips.LPIPS(net="alex", verbose=False)


def lpips_distance(ref: np.ndarray, cand: np.ndarray) -> float | None:
    """LPIPS (AlexNet) of uint8 [F, H, W, 3] frame stacks, one frame pair at a time (the metric
    is per frame; a clip is not one batch), averaged over the frames; None without the package."""
    net = _lpips_net()
    if net is None:
        return None

    def t(a):
        return torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

    with torch.no_grad():
        return float(np.mean([net(t(r), t(c)).item() for r, c in zip(ref, cand)]))


def compare(a: Path, b: Path) -> dict:
    """PSNR, SSIM and LPIPS between two decoded outputs: an image pair on its one frame; an .mp4
    or frame-directory pair frame by frame, each metric averaged over the frames, with
    ``frames`` and ``min_psnr_frame`` (the worst frame's index). The frame counts must agree."""
    a, b = Path(a), Path(b)
    ref, cand = _read_frames(a), _read_frames(b)
    if len(ref) != len(cand):
        raise ValueError(f"frame count mismatch: {a} has {len(ref)} frames, {b} has {len(cand)} frames")
    frames = _qvr().frame_metrics(ref, cand)
    out = {"psnr": frames["psnr_mean"], "ssim": frames["ssim_mean"], "lpips": lpips_distance(ref, cand),
           "ref": str(a), "cand": str(b)}
    if _is_video(a) or _is_video(b):
        out["frames"] = len(ref)
        out["min_psnr_frame"] = int(np.argmin(frames["psnr_per_frame"]))
    return out


def compare_runs(ref_latents: Path | None, cand_latents: Path | None, ref_output: Path | None,
                 cand_output: Path | None) -> dict:
    """Latent metrics when both latent files are given, decoded-output metrics when both outputs
    are (a run that decodes in process, FLUX at tp4, saves no latents): under ``image`` for an
    image pair, under ``video`` for an .mp4 or frame-directory pair."""
    out: dict = {}
    if ref_latents and cand_latents:
        out.update(ref_latents=str(ref_latents), cand_latents=str(cand_latents))
        ref = torch.load(ref_latents, map_location="cpu", weights_only=True)
        cand = torch.load(cand_latents, map_location="cpu", weights_only=True)
        out["latents"] = _qvr().latent_metrics(ref, cand)
        out["latents"]["bit_identical"] = bool(torch.equal(ref, cand))
    if ref_output and cand_output:
        ref_output, cand_output = Path(ref_output), Path(cand_output)
        key = "video" if _is_video(ref_output) or _is_video(cand_output) else "image"
        out[key] = compare(ref_output, cand_output)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-latents", type=Path, default=None)
    p.add_argument("--cand-latents", type=Path, default=None)
    p.add_argument("--ref-image", type=Path, default=None, help="a PNG, an .mp4 or a directory of frame PNGs")
    p.add_argument("--cand-image", type=Path, default=None, help="the candidate's, of the same kind")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not ((args.ref_latents and args.cand_latents) or (args.ref_image and args.cand_image)):
        p.error("give both latent files, both decoded outputs, or both pairs")
    result = compare_runs(args.ref_latents, args.cand_latents, args.ref_image, args.cand_image)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
