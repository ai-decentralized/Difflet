#!/usr/bin/env python
"""On-device HunyuanVideo ring-vs-gather-KV parity: single-mode runner.

Runs ONE cp_mode (gather_kv | ring) of the HunyuanVideo DiT backbone at
tp=2/cp=2 on a fixed-seed input and saves the output latent tensor to ``--out``.
Each mode MUST run in its own process because neuronx_distributed's
parallel_state initializes once per process.
``scripts/hunyuan_ring_parity_smoke.sh`` drives both modes and compares the two
saved tensors.

A reduced ``--layers`` (default 2) keeps the compile small; weights load with
strict=False so only the first ``--layers`` transformer blocks are populated
from the real checkpoint (identical weights across both modes → a fair compare).

Ring kernel constraints for the default shape (height=256, width=256, num_frames=5):
  latent_frames = (5-1)//4 + 1 = 2
  patch_size = 2 (spatial), patch_size_t = 1 (temporal)
  seq = latent_frames × (height/8/patch_size) × (width/8/patch_size)
      = 2 × (256/8/2) × (256/8/2) = 2 × 16 × 16 = 512
  per_rank (cp=2): 256 — divisible by 128 ✓
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(
        description="HunyuanVideo ring/gather-KV parity single-mode runner"
    )
    p.add_argument("--mode", required=True, choices=["gather_kv", "ring"])
    p.add_argument("--out", required=True, help="path to save the output latent tensor (.pt)")
    p.add_argument(
        "--model",
        default=os.environ.get("DIFFLET_HUNYUAN_MODEL", "hunyuanvideo-community/HunyuanVideo"),
    )
    p.add_argument(
        "--tp-degree",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_TP_DEGREE", "2")),
    )
    p.add_argument(
        "--cp-degree",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_CP_DEGREE", "2")),
    )
    p.add_argument(
        "--layers",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_LAYERS", "2")),
    )
    p.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_HEIGHT", "256")),
    )
    p.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_WIDTH", "256")),
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_NUM_FRAMES", "5")),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_SEED", "1234")),
    )
    p.add_argument(
        "--work-dir",
        default=os.environ.get("DIFFLET_HUNYUAN_PARITY_WORKDIR", "/tmp/hunyuan_ring_parity"),
    )
    args = p.parse_args()

    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config
    from difflet.backends.trainium.hunyuan_video.backbone import (
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree * args.cp_degree
    model_dir = resolve_model_path(args.model, local_files_only=True)

    # latent_frames = (num_frames - 1) // 4 + 1
    latent_frames = (args.num_frames - 1) // 4 + 1
    # seq = latent_frames × (H/8/patch_size) × (W/8/patch_size), patch_size=2
    patch_size = 2
    seq = latent_frames * (args.height // 8 // patch_size) * (args.width // 8 // patch_size)
    per_rank = seq // args.cp_degree

    print(
        f"[parity] mode={args.mode} tp={args.tp_degree} cp={args.cp_degree} "
        f"world={world_size} layers={args.layers} "
        f"shape={args.height}x{args.width}x{args.num_frames}"
    )
    print(
        f"[parity] latent_frames={latent_frames} seq={seq} per_rank={per_rank} "
        f"(per_rank % 128 = {per_rank % 128})"
    )
    if args.cp_degree > 1 and args.mode == "ring" and per_rank % 128 != 0:
        raise ValueError(
            f"ring requires per-rank seqlen ({per_rank}) divisible by 128; "
            "adjust --height/--width/--num-frames"
        )

    config = create_hunyuan_video_backbone_config(
        model_path=model_dir,
        world_size=world_size,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        batch_size=1,
        context_parallel_enabled=args.cp_degree > 1,
        cp_mode=args.mode,
    )
    # Reduce depth for a fast compile; strict=False load populates only these
    # blocks from the real checkpoint (same weights for both modes).
    config.num_layers = args.layers
    config.num_single_layers = 0  # skip single-stream blocks for speed

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    transformer_path = os.path.join(model_dir, "transformer")
    app = NeuronHunyuanVideoBackboneApplication(model_path=transformer_path, config=config)
    print(f"[parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed inputs: identical across both processes/modes.
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        [1, config.in_channels, latent_frames, args.height // 8, args.width // 8],
        dtype=torch.bfloat16,
    )
    timestep = torch.ones([1], dtype=torch.bfloat16)
    text_seq_len = int(getattr(config, "text_seq_len", 256))
    encoder_hidden_states = torch.randn(
        [1, text_seq_len, config.text_embed_dim],
        dtype=torch.bfloat16,
    )
    encoder_attention_mask = torch.ones([1, text_seq_len], dtype=torch.int64)
    pooled_projections = torch.randn(
        [1, config.pooled_projection_dim],
        dtype=torch.bfloat16,
    )
    guidance = torch.full([1], 6000.0, dtype=torch.bfloat16)

    print("[parity] running forward")
    with torch.no_grad():
        out = app.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
