#!/usr/bin/env python3
"""Decode HunyuanVideo latents with the HF AutoencoderKLHunyuanVideo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Parent dir containing vae/")
    parser.add_argument("--latents", required=True, help="Latents .pt file")
    parser.add_argument("--output", default=None, help="Optional decoded video tensor .pt")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--no-tiling", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float32


def main() -> int:
    args = parse_args()
    dtype = _dtype(args.dtype)
    model_dir = Path(args.model_dir)
    latents_path = Path(args.latents)

    print(f"[vae-smoke] model_dir = {model_dir}")
    print(f"[vae-smoke] latents   = {latents_path}")
    latents = torch.load(latents_path, map_location="cpu")
    print(f"[vae-smoke] latents shape = {tuple(latents.shape)}")
    print(f"[vae-smoke] latents dtype  = {latents.dtype}")
    print(
        "[vae-smoke] latents mean/std = "
        f"{latents.float().mean().item():.6e} / {latents.float().std().item():.6e}"
    )

    from diffusers import AutoencoderKLHunyuanVideo

    print("[vae-smoke] load HF VAE ...")
    t0 = time.time()
    vae = AutoencoderKLHunyuanVideo.from_pretrained(model_dir / "vae", torch_dtype=dtype).eval()
    print(f"[vae-smoke] load elapsed = {time.time() - t0:.3f}s")
    print(f"[vae-smoke] scaling_factor = {vae.config.scaling_factor}")
    if not args.no_tiling:
        vae.enable_tiling()
        print("[vae-smoke] tiling enabled")

    print("[vae-smoke] decode ...")
    t1 = time.time()
    with torch.no_grad():
        video = vae.decode(latents.to(dtype=dtype) / vae.config.scaling_factor, return_dict=False)[0]
    print(f"[vae-smoke] decode elapsed = {time.time() - t1:.3f}s")
    print(f"[vae-smoke] video shape = {tuple(video.shape)}")
    print(f"[vae-smoke] video dtype  = {video.dtype}")
    print(f"[vae-smoke] video finite all = {bool(torch.isfinite(video).all())}")
    print(
        "[vae-smoke] video mean/std = "
        f"{video.float().mean().item():.6e} / {video.float().std().item():.6e}"
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(video.cpu(), output)
        print(f"[vae-smoke] saved video -> {output}")
    print("[vae-smoke] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
