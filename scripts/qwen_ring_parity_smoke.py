#!/usr/bin/env python
"""On-device Qwen-Image ring-vs-gather-KV parity: single-mode runner.

Runs ONE cp_mode (gather_kv | ring) of the Qwen-Image DiT backbone at
tp=2/cp=2 on a fixed-seed input and saves the output packed-latent tensor to
``--out``. Each mode MUST run in its own process because neuronx_distributed's
parallel_state initializes once per process.
``scripts/qwen_ring_parity_smoke.sh`` drives both modes and compares the two
saved tensors.

A reduced ``--layers`` (default 2) keeps the compile small; weights load with
strict=False, so only the first ``--layers`` transformer blocks are populated
from the real checkpoint (identical weights across both modes → a fair compare).

Ring kernel constraints for the default shape (height=256, width=256):
  vae_scale_factor = 8  (Qwen-Image VAE stride)
  latent_height    = 2 * (256 // (8 * 2)) = 2 * 16 = 32
  latent_width     = 2 * (256 // (8 * 2)) = 2 * 16 = 32
  packed_height    = 32 // 2 = 16   (patch_size = 2)
  packed_width     = 32 // 2 = 16
  image_seq_len    = 16 * 16 = 256
  per_rank (cp=2)  = 256 / 2 = 128  (128 % 128 == 0 ✓)
  head_dim         = read from model config (typically 128)
  num_heads/rank   = num_attention_heads // tp_degree  (must be whole number)

NOTE: device parity is DEFERRED — Qwen-Image transformer weights are not cached
on the compile box and disk cannot fit them. This script is committed unrun.
Run it on a box with the real weights and Trainium2 hardware.
Generic joint-ring correctness is proven by Task 3 (cosine 0.999972 on trn2
at tp=2/cp=2 for the joint_ring_attention op directly).
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(
        description="Qwen-Image ring/gather-KV parity single-mode runner"
    )
    p.add_argument("--mode", required=True, choices=["gather_kv", "ring"])
    p.add_argument("--out", required=True, help="path to save the output latent tensor (.pt)")
    p.add_argument(
        "--model",
        default=os.environ.get("DIFFLET_QWEN_MODEL", "Qwen/Qwen-Image"),
    )
    p.add_argument(
        "--tp-degree",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_TP_DEGREE", "2")),
    )
    p.add_argument(
        "--cp-degree",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_CP_DEGREE", "2")),
    )
    p.add_argument(
        "--layers",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_LAYERS", "2")),
    )
    p.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_HEIGHT", "256")),
    )
    p.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_WIDTH", "256")),
    )
    p.add_argument(
        "--text-seq-len",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_TEXT_SEQ_LEN", "256")),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("DIFFLET_QWEN_SEED", "1234")),
    )
    p.add_argument(
        "--work-dir",
        default=os.environ.get("DIFFLET_QWEN_PARITY_WORKDIR", "/tmp/qwen_ring_parity"),
    )
    args = p.parse_args()

    from difflet.models.qwen_image.application import create_qwen_image_transformer_config
    from difflet.backends.trainium.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree * args.cp_degree
    model_dir = resolve_model_path(args.model, local_files_only=True)

    # Qwen-Image sequence length: image_seq_len = packed_height * packed_width
    # packed_height = (2*(height//(vae_scale_factor*2))) // patch_size
    #               = (2*(height//16)) // 2 = height // 16
    vae_scale_factor = 8
    patch_size = 2
    packed_height = 2 * (args.height // (vae_scale_factor * patch_size)) // patch_size
    packed_width = 2 * (args.width // (vae_scale_factor * patch_size)) // patch_size
    image_seq_len = packed_height * packed_width
    per_rank = image_seq_len // args.cp_degree

    print(
        f"[parity] mode={args.mode} tp={args.tp_degree} cp={args.cp_degree} "
        f"world={world_size} layers={args.layers} "
        f"shape={args.height}x{args.width}"
    )
    print(
        f"[parity] packed={packed_height}x{packed_width} "
        f"image_seq_len={image_seq_len} per_rank={per_rank} "
        f"(per_rank % 128 = {per_rank % 128})"
    )
    if args.cp_degree > 1 and args.mode == "ring" and per_rank % 128 != 0:
        raise ValueError(
            f"ring requires per-rank seqlen ({per_rank}) divisible by 128; "
            "adjust --height/--width"
        )

    # Build the config the same way NeuronQwenImageApplication does:
    # load_diffusers_config reads the transformer's config.json (num_layers,
    # attention_head_dim, num_attention_heads, in_channels, joint_attention_dim,
    # axes_dims_rope, guidance_embeds, etc.) from the real checkpoint.
    config = create_qwen_image_transformer_config(
        model_path=model_dir,
        world_size=world_size,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        text_seq_len=args.text_seq_len,
        batch_size=1,
        context_parallel_enabled=args.cp_degree > 1,
        cp_mode=args.mode,
    )
    # Reduce depth for a fast compile; strict=False load populates only these
    # blocks from the real checkpoint (same weights for both modes).
    config.num_layers = args.layers

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    transformer_path = os.path.join(model_dir, "transformer")
    app = NeuronQwenImageTransformerApplication(model_path=transformer_path, config=config)
    print(f"[parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed inputs: identical across both processes/modes.
    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        [1, image_seq_len, int(config.in_channels)],
        dtype=torch.bfloat16,
    )
    timestep = torch.ones([1], dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(
        [1, args.text_seq_len, int(config.joint_attention_dim)],
        dtype=torch.bfloat16,
    )
    encoder_hidden_states_mask = torch.ones(
        [1, args.text_seq_len], dtype=torch.bool
    )
    guidance = torch.full([1], 4.0, dtype=torch.bfloat16)

    print("[parity] running forward")
    with torch.no_grad():
        out = app.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            guidance,
        )
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
