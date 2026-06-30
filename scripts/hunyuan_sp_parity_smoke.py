#!/usr/bin/env python
"""On-device HunyuanVideo Megatron-SP vs dense-TP parity: single-mode runner.

Runs ONE mode (dense | sp) of the HunyuanVideo DiT backbone at a fixed tp (no CP)
on a fixed-seed input and saves the output latent tensor to ``--out``. Each mode
MUST run in its own process because neuronx_distributed's parallel_state
initializes once per process. ``scripts/hunyuan_sp_parity_smoke.sh`` drives both
modes and compares the two saved tensors (expect cosine >= 0.999 — SP is
mathematically lossless vs dense TP).

A reduced ``--layers`` / ``--single-layers`` keeps the compile small; weights load
with strict=False so only those blocks are populated from the real checkpoint
(identical weights across both modes → a fair compare).

Megatron-SP shards BOTH the latent and text streams across the tensor-parallel
group, so the per-rank latent sequence (seq // tp) and the text sequence
(text_seq_len // tp) must both be integers. For the default shape
(height=256, width=256, num_frames=5):
  latent_frames = (5-1)//4 + 1 = 2
  patch_size = 2 (spatial), patch_size_t = 1 (temporal)
  seq = latent_frames × (height/8/patch_size) × (width/8/patch_size)
      = 2 × 16 × 16 = 512  → 256/rank at tp=2
  text_seq_len = 256       → 128/rank at tp=2
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(
        description="HunyuanVideo dense/SP parity single-mode runner"
    )
    p.add_argument("--mode", required=True, choices=["dense", "sp"])
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
        "--layers",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_LAYERS", "2")),
    )
    p.add_argument(
        "--single-layers",
        type=int,
        default=int(os.environ.get("DIFFLET_HUNYUAN_SINGLE_LAYERS", "2")),
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
        default=os.environ.get("DIFFLET_HUNYUAN_SP_PARITY_WORKDIR", "/tmp/hunyuan_sp_parity"),
    )
    args = p.parse_args()

    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config
    from difflet.backends.trainium.hunyuan_video.backbone import (
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree  # SP reuses the TP group; no extra axis.
    model_dir = resolve_model_path(args.model, local_files_only=True)

    latent_frames = (args.num_frames - 1) // 4 + 1
    patch_size = 2
    seq = latent_frames * (args.height // 8 // patch_size) * (args.width // 8 // patch_size)
    per_rank = seq // args.tp_degree

    config = create_hunyuan_video_backbone_config(
        model_path=model_dir,
        world_size=world_size,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        batch_size=1,
        sp_enabled=(args.mode == "sp"),
    )
    config.num_layers = args.layers
    config.num_single_layers = args.single_layers

    text_seq_len = int(getattr(config, "text_seq_len", 256))
    print(
        f"[sp-parity] mode={args.mode} tp={args.tp_degree} world={world_size} "
        f"layers={args.layers} single_layers={args.single_layers} "
        f"shape={args.height}x{args.width}x{args.num_frames}"
    )
    print(
        f"[sp-parity] latent_frames={latent_frames} seq={seq} per_rank={per_rank} "
        f"text_seq_len={text_seq_len} (seq % tp={seq % args.tp_degree}, "
        f"text % tp={text_seq_len % args.tp_degree})"
    )
    if args.mode == "sp" and seq % args.tp_degree != 0:
        raise ValueError(
            f"SP requires latent sequence ({seq}) divisible by tp ({args.tp_degree}); "
            "adjust --height/--width/--num-frames"
        )
    if args.mode == "sp" and text_seq_len % args.tp_degree != 0:
        raise ValueError(
            f"SP requires text_seq_len ({text_seq_len}) divisible by tp ({args.tp_degree})"
        )

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    transformer_path = os.path.join(model_dir, "transformer")
    app = NeuronHunyuanVideoBackboneApplication(model_path=transformer_path, config=config)
    print(f"[sp-parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[sp-parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed inputs: identical across both processes/modes.
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        [1, config.in_channels, latent_frames, args.height // 8, args.width // 8],
        dtype=torch.bfloat16,
    )
    timestep = torch.ones([1], dtype=torch.bfloat16)
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

    print("[sp-parity] running forward")
    with torch.no_grad():
        out = app.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
    # The backbone may return a dict / tuple / object wrapping the sample tensor.
    if isinstance(out, dict):
        out = out.get("sample", out.get("hidden_states", next(iter(out.values()))))
    elif isinstance(out, (tuple, list)):
        out = out[0]
    elif hasattr(out, "sample"):
        out = out.sample
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[sp-parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
