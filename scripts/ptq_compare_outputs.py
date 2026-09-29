#!/usr/bin/env python3
"""Compare a quantized run against its bf16 reference: PSNR / SSIM / LPIPS + latents.

    PYTHONPATH=$PWD python scripts/ptq_compare_outputs.py \\
        --reference out/bf16.mp4 --test out/fp8.mp4 \\
        [--latents-reference work_bf16/latents.pt --latents-test work_fp8/latents.pt] \\
        [--no-lpips] --out artifacts/ptq/compare.json

Inputs may be .png / .mp4 / .pt (Wan frame tensors). LPIPS uses the optional
``lpips`` package (net=alex, FastVideo's default) and is reported as null when
it is not installed. Latent metrics are MSE / cosine / max-abs / rel-L2 / SNR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.quant.metrics import compare_latents, compare_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True, help="bf16 output (.png/.mp4/.pt)")
    p.add_argument("--test", type=Path, required=True, help="quantized output (.png/.mp4/.pt)")
    p.add_argument("--latents-reference", type=Path, default=None)
    p.add_argument("--latents-test", type=Path, default=None)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p


def run(args: argparse.Namespace) -> dict:
    result = {"output": compare_outputs(args.reference, args.test,
                                        lpips_net=None if args.no_lpips else "alex")}
    if args.latents_reference and args.latents_test:
        result["latents"] = compare_latents(args.latents_reference, args.latents_test)
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    out = result["output"]
    lpips = out["lpips"]
    print(f"frames={out['frames']} {out['width']}x{out['height']}  PSNR={out['psnr_db']:.2f} dB  "
          f"SSIM={out['ssim']:.4f}  LPIPS={'n/a (pip install lpips)' if lpips is None else f'{lpips:.4f}'}")
    if "latents" in result:
        lat = result["latents"]
        print(f"latents {lat['shape']}: cosine={lat['cosine']:.6f} mse={lat['mse']:.3e} "
              f"rel_l2={lat['rel_l2']:.4f} snr={lat['snr_db']:.2f} dB")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
