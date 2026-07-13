#!/usr/bin/env python
"""On-device Wan ring-vs-gather-KV parity: single-mode runner.

Runs ONE cp_mode (gather_kv | ring) of the Wan DiT backbone at tp=2/cp=2 on a
fixed-seed input and saves the output latent tensor to ``--out``. Each mode MUST
run in its own process because neuronx_distributed's parallel_state initializes
once per process. ``scripts/wan_ring_parity_smoke.sh`` drives both modes and
compares the two saved tensors.

A reduced ``--layers`` (default 2) keeps the compile small; weights load with
strict=False, so only the first ``--layers`` transformer blocks are populated
from the real checkpoint (identical weights across both modes -> a fair compare).

Ring kernel constraints honored by the default shape: per-rank seqlen
(seq // cp_degree) must be a multiple of 128, head_dim <= 128 (Wan = 128), and
num_attention_heads divisible by tp_degree (Wan = 40, tp=2 -> 20/rank).
seq = num_frames(latent) * (height/16) * (width/16).
Default 256x512x1 -> seq = 1*16*32 = 512 -> 256/rank (mult. of 128).
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Wan ring/gather-KV parity single-mode runner")
    p.add_argument("--mode", required=True, choices=["gather_kv", "ring", "ulysses"])
    p.add_argument("--out", required=True, help="path to save the output latent tensor (.pt)")
    p.add_argument("--model", default=os.environ.get("DIFFLET_WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
    p.add_argument("--subfolder", default=os.environ.get("DIFFLET_WAN_TRANSFORMER_SUBFOLDER", "transformer"))
    p.add_argument("--tp-degree", type=int, default=int(os.environ.get("DIFFLET_WAN_TP_DEGREE", "2")))
    p.add_argument("--cp-degree", type=int, default=int(os.environ.get("DIFFLET_WAN_CP_DEGREE", "2")))
    p.add_argument("--layers", type=int, default=int(os.environ.get("DIFFLET_WAN_LAYERS", "2")))
    p.add_argument("--height", type=int, default=int(os.environ.get("DIFFLET_WAN_HEIGHT", "256")))
    p.add_argument("--width", type=int, default=int(os.environ.get("DIFFLET_WAN_WIDTH", "512")))
    p.add_argument("--latent-frames", type=int, default=int(os.environ.get("DIFFLET_WAN_LATENT_FRAMES", "1")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("DIFFLET_WAN_SEED", "1234")))
    p.add_argument("--work-dir", default=os.environ.get("DIFFLET_WAN_PARITY_WORKDIR", "/tmp/wan_ring_parity"))
    args = p.parse_args()

    from difflet.models.wan.application import create_wan_backbone_config
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree * args.cp_degree
    model_dir = resolve_model_path(args.model, local_files_only=True)
    component_dir = os.path.join(model_dir, args.subfolder)
    if not os.path.exists(os.path.join(component_dir, "config.json")):
        raise FileNotFoundError(f"missing {args.subfolder}/config.json under {model_dir}")

    config = create_wan_backbone_config(
        model_path=model_dir,
        world_size=world_size,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.latent_frames,
        batch_size=1,
        subfolder=args.subfolder,
        context_parallel_enabled=args.cp_degree > 1,
        cp_mode=args.mode,
    )
    # Reduce depth for a fast compile; strict=False load populates only these
    # blocks from the real checkpoint (same weights for both modes).
    config.num_layers = args.layers

    seq = args.latent_frames * (args.height // 16) * (args.width // 16)
    per_rank = seq // args.cp_degree
    print(f"[parity] mode={args.mode} tp={args.tp_degree} cp={args.cp_degree} "
          f"world={world_size} layers={args.layers} shape={args.height}x{args.width}x{args.latent_frames}")
    print(f"[parity] seq={seq} per_rank={per_rank} (per_rank % 128 = {per_rank % 128})")
    if args.cp_degree > 1 and args.mode == "ring" and per_rank % 128 != 0:
        raise ValueError(
            f"ring requires per-rank seqlen ({per_rank}) divisible by 128; adjust height/width/latent-frames"
        )
    if args.cp_degree > 1 and args.mode == "ulysses":
        # Ulysses shards heads across cp on top of the TP head shard, so the model's
        # head count must divide by tp*cp. Wan is 40 heads: fine at tp2/cp2 (10/rank).
        heads = int(config.num_attention_heads)
        if heads % (args.tp_degree * args.cp_degree) != 0:
            raise ValueError(
                f"ulysses requires num_attention_heads ({heads}) divisible by "
                f"tp_degree * cp_degree ({args.tp_degree} * {args.cp_degree})"
            )
        print(f"[parity] ulysses heads: {heads} -> {heads // args.tp_degree}/rank (tp) "
              f"-> {heads // (args.tp_degree * args.cp_degree)}/rank in attention (cp)")

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    app = NeuronWanBackboneApplication(model_path=component_dir, config=config)
    print(f"[parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed inputs: identical across both processes/modes.
    torch.manual_seed(args.seed)
    hidden = torch.randn(
        [1, config.in_channels, args.latent_frames, args.height // 8, args.width // 8],
        dtype=torch.bfloat16,
    )
    timestep = torch.randn([1], dtype=torch.bfloat16)
    encoder = torch.randn([1, int(config.text_seq_len), config.text_dim], dtype=torch.bfloat16)

    print("[parity] running forward")
    with torch.no_grad():
        out = app.forward(hidden, timestep, encoder)
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
