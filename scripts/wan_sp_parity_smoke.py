#!/usr/bin/env python
"""On-device Wan Megatron-SP vs dense-TP parity: single-mode runner.

Runs ONE mode (dense | sp) of the Wan DiT backbone at a fixed tp (no CP) on a
fixed-seed input and saves the output latent to ``--out``. Each mode MUST run in
its own process because neuronx_distributed's parallel_state initializes once per
process. ``scripts/wan_sp_parity_smoke.sh`` drives both modes and compares the
two saved tensors (expect cosine >= 0.999 — SP is mathematically lossless vs
dense TP).

Megatron-SP shards the sequence across the tensor-parallel group, so the
per-rank sequence (seq // tp_degree) must be an integer. ``seq = latent_frames *
(height/16) * (width/16)``; the default 256x512x1 at tp=2 gives seq=512 ->
256/rank.
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Wan dense/SP parity single-mode runner")
    p.add_argument("--mode", required=True, choices=["dense", "sp"])
    p.add_argument("--out", required=True, help="path to save the output latent tensor (.pt)")
    p.add_argument("--model", default=os.environ.get("DIFFLET_WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
    p.add_argument("--subfolder", default=os.environ.get("DIFFLET_WAN_TRANSFORMER_SUBFOLDER", "transformer"))
    p.add_argument("--tp-degree", type=int, default=int(os.environ.get("DIFFLET_WAN_TP_DEGREE", "2")))
    p.add_argument("--layers", type=int, default=int(os.environ.get("DIFFLET_WAN_LAYERS", "2")))
    p.add_argument("--height", type=int, default=int(os.environ.get("DIFFLET_WAN_HEIGHT", "256")))
    p.add_argument("--width", type=int, default=int(os.environ.get("DIFFLET_WAN_WIDTH", "512")))
    p.add_argument("--latent-frames", type=int, default=int(os.environ.get("DIFFLET_WAN_LATENT_FRAMES", "1")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("DIFFLET_WAN_SEED", "1234")))
    p.add_argument("--work-dir", default=os.environ.get("DIFFLET_WAN_SP_PARITY_WORKDIR", "/tmp/wan_sp_parity"))
    args = p.parse_args()

    from difflet.models.wan.application import create_wan_backbone_config
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree  # SP reuses the TP group; no extra axis.
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
        sp_enabled=(args.mode == "sp"),
    )
    config.num_layers = args.layers

    seq = args.latent_frames * (args.height // 16) * (args.width // 16)
    per_rank = seq // args.tp_degree
    print(f"[sp-parity] mode={args.mode} tp={args.tp_degree} world={world_size} "
          f"layers={args.layers} shape={args.height}x{args.width}x{args.latent_frames}")
    print(f"[sp-parity] seq={seq} per_rank={per_rank} (seq % tp = {seq % args.tp_degree})")
    if args.mode == "sp" and seq % args.tp_degree != 0:
        raise ValueError(
            f"SP requires sequence ({seq}) divisible by tp ({args.tp_degree}); "
            "adjust height/width/latent-frames"
        )

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    app = NeuronWanBackboneApplication(model_path=component_dir, config=config)
    print(f"[sp-parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[sp-parity] loading weights to device")
    app.load(out_dir)

    torch.manual_seed(args.seed)
    hidden = torch.randn(
        [1, config.in_channels, args.latent_frames, args.height // 8, args.width // 8],
        dtype=torch.bfloat16,
    )
    timestep = torch.randn([1], dtype=torch.bfloat16)
    encoder = torch.randn([1, int(config.text_seq_len), config.text_dim], dtype=torch.bfloat16)

    print("[sp-parity] running forward")
    with torch.no_grad():
        out = app.forward(hidden, timestep, encoder)
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[sp-parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
