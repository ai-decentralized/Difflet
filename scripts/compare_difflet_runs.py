#!/usr/bin/env python3
"""Compare two ``difflet generate`` runs (e.g. the bf16 and the FP8 DiT from one seed): the
packed latents the generate stage wrote (``<work-dir>/latents.pt``) and the decoded images.

Metrics as ``scripts/quality_vs_reference.py`` (whose functions this reuses): latent relative
L2 and cosine in float64; PSNR and SSIM (Wang 2004) on uint8 RGB. LPIPS (AlexNet) when the
``lpips`` package is installed, else reported as null.

    python scripts/compare_difflet_runs.py \\
        --ref-latents B/work_gen50a/latents.pt --ref-image B/gen50a.png \\
        --cand-latents F/work_gen50a/latents.pt --cand-image F/gen50a.png --out metrics.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _qvr():
    spec = importlib.util.spec_from_file_location("qvr", REPO / "scripts" / "quality_vs_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def lpips_distance(ref: np.ndarray, cand: np.ndarray) -> float | None:
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    net = lpips.LPIPS(net="alex", verbose=False)

    def t(a):
        return torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0

    with torch.no_grad():
        return float(net(t(ref), t(cand)).item())


def compare(ref_latents: Path | None, cand_latents: Path | None, ref_image: Path | None,
            cand_image: Path | None) -> dict:
    """Latent metrics when both latent files are given, image metrics when both images are
    (a run that decodes in process, FLUX at tp4, saves no latents)."""
    qvr = _qvr()
    out: dict = {}
    if ref_latents and cand_latents:
        out.update(ref_latents=str(ref_latents), cand_latents=str(cand_latents))
        ref = torch.load(ref_latents, map_location="cpu", weights_only=True)
        cand = torch.load(cand_latents, map_location="cpu", weights_only=True)
        out["latents"] = qvr.latent_metrics(ref, cand)
        out["latents"]["bit_identical"] = bool(torch.equal(ref, cand))
    if ref_image and cand_image:
        a, b = _image(Path(ref_image)), _image(Path(cand_image))
        frames = qvr.frame_metrics(a[None], b[None])
        out["image"] = {"psnr": frames["psnr_mean"], "ssim": frames["ssim_mean"],
                        "lpips": lpips_distance(a, b), "ref": str(ref_image),
                        "cand": str(cand_image)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-latents", type=Path, default=None)
    p.add_argument("--cand-latents", type=Path, default=None)
    p.add_argument("--ref-image", type=Path, default=None)
    p.add_argument("--cand-image", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not ((args.ref_latents and args.cand_latents) or (args.ref_image and args.cand_image)):
        p.error("give both latent files, both images, or both pairs")
    result = compare(args.ref_latents, args.cand_latents, args.ref_image, args.cand_image)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
