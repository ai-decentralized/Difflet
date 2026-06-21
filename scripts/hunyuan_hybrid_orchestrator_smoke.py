#!/usr/bin/env python3
"""Run the HunyuanVideo M3 hybrid orchestrator over a cached DiT bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--compiled-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--save-latents", default=None)
    parser.add_argument("--return-trajectory", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")

    meta = json.loads(Path(args.bundle + ".meta.json").read_text())
    tensors = load_file(args.bundle)
    print(f"[hybrid] bundle = {args.bundle}")
    print(f"[hybrid] prompt = {meta.get('prompt')}")
    print(
        f"[hybrid] shape = {meta['height']}x{meta['width']}x{meta['num_frames']}, "
        f"steps={meta['num_inference_steps']}"
    )

    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
        text_seq_len=meta["text_seq_len"],
    )
    print(f"[hybrid] dit_input_contract = {app.dit_input_contract()}")
    print(f"[hybrid] load(skip_warmup=True) from {args.compiled_dir} ...")
    t0 = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    print(f"[hybrid] load elapsed = {time.time() - t0:.3f}s")

    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    print("[hybrid] orchestrator denoise ...")
    t1 = time.time()
    output = app.pipeline(
        bundle=bundle,
        timesteps=tensors["timesteps"],
        output_type="latent",
        return_trajectory=args.return_trajectory,
    )
    print(f"[hybrid] denoise elapsed = {time.time() - t1:.3f}s")
    latents = output.latents
    print(f"[hybrid] latents shape = {tuple(latents.shape)}")
    print(f"[hybrid] latents dtype = {latents.dtype}")
    print(f"[hybrid] latents finite all = {bool(torch.isfinite(latents).all())}")
    print(
        "[hybrid] latents mean/std (cast fp32) = "
        f"{latents.float().mean().item():.6e} / {latents.float().std().item():.6e}"
    )
    if output.trajectory is not None:
        print(f"[hybrid] trajectory entries = {len(output.trajectory)}")
    if args.save_latents:
        path = Path(args.save_latents)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.cpu(), path)
        print(f"[hybrid] saved latents -> {path}")
    print("[hybrid] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
