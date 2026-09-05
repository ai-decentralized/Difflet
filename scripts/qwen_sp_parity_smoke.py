#!/usr/bin/env python
"""On-device Qwen-Image Megatron-SP vs dense-TP parity: single-mode runner.

Runs ONE mode (dense | sp) of the Qwen-Image DiT transformer at a fixed tp (no
CP) on a fixed-seed input and saves the output to ``--out``. Each mode MUST run
in its own process because neuronx_distributed's parallel_state initializes
once per process. ``scripts/qwen_sp_parity_smoke.sh`` drives both modes and
compares the two saved tensors (expect cosine >= 0.999 — SP is mathematically
lossless vs dense TP).

Megatron-SP shards both residual streams (image AND text) across the
tensor-parallel group, so each per-rank sequence must be an integer:
image seq = (height/16) * (width/16); text seq = --text-seq-len. The default
256x256 @ tp=4 with text-seq-len 256 gives 256/4 = 64 tokens per rank on both
streams.

Uses a reduced layer count for a fast compile; both modes load the same
(subset) checkpoint weights, so the comparison is apples-to-apples.

Note: tp=4 (the whole trn2.3xlarge device under LNC=2) is the default because
this host's Neuron driver build rejects partial multi-core allocations
(`NEURON_RT_NUM_CORES=2` fails with "must request one core, or the whole
device"); 1 and 4 work. See docs/plans/2026-09-02-qwen-sp-completion.md.
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen-Image dense/SP parity single-mode runner")
    p.add_argument("--mode", required=True, choices=["dense", "sp"])
    p.add_argument("--out", required=True, help="path to save the output tensor (.pt)")
    p.add_argument("--model", default=os.environ.get("DIFFLET_QWEN_MODEL", "Qwen/Qwen-Image"))
    p.add_argument("--revision", default=os.environ.get(
        "DIFFLET_QWEN_REVISION", "75e0b4be04f60ec59a75f475837eced720f823b6"))
    p.add_argument("--tp-degree", type=int, default=int(os.environ.get("DIFFLET_QWEN_TP_DEGREE", "4")))
    p.add_argument("--layers", type=int, default=int(os.environ.get("DIFFLET_QWEN_LAYERS", "2")))
    p.add_argument("--height", type=int, default=int(os.environ.get("DIFFLET_QWEN_HEIGHT", "256")))
    p.add_argument("--width", type=int, default=int(os.environ.get("DIFFLET_QWEN_WIDTH", "256")))
    p.add_argument("--text-seq-len", type=int,
                   default=int(os.environ.get("DIFFLET_QWEN_TEXT_SEQ_LEN", "256")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("DIFFLET_QWEN_SEED", "1234")))
    p.add_argument("--work-dir", default=os.environ.get("DIFFLET_QWEN_SP_PARITY_WORKDIR", "/tmp/qwen_sp_parity"))
    args = p.parse_args()

    from difflet.models.qwen_image.application import (
        create_qwen_image_transformer_config,
    )
    from difflet.backends.trainium.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from difflet.pipeline.path_resolver import resolve_model_path

    world_size = args.tp_degree  # SP reuses the TP group; no extra axis.
    model_dir = resolve_model_path(
        args.model, revision=args.revision or None, local_files_only=True)
    component_dir = os.path.join(model_dir, "transformer")
    if not os.path.exists(os.path.join(component_dir, "config.json")):
        raise FileNotFoundError(f"missing transformer/config.json under {model_dir}")

    config = create_qwen_image_transformer_config(
        model_path=model_dir,
        world_size=world_size,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        text_seq_len=args.text_seq_len,
        sp_enabled=(args.mode == "sp"),
    )
    config.num_layers = args.layers

    img_seq = (args.height // 16) * (args.width // 16)
    print(f"[sp-parity] mode={args.mode} tp={args.tp_degree} world={world_size} "
          f"layers={args.layers} shape={args.height}x{args.width} text_seq={args.text_seq_len}")
    print(f"[sp-parity] img_seq={img_seq} txt_seq={args.text_seq_len} "
          f"(img % tp = {img_seq % args.tp_degree}, txt % tp = {args.text_seq_len % args.tp_degree})")
    if args.mode == "sp" and (img_seq % args.tp_degree or args.text_seq_len % args.tp_degree):
        raise ValueError(
            f"SP requires both sequence lengths ({img_seq}, {args.text_seq_len}) "
            f"divisible by tp ({args.tp_degree}); adjust height/width/text-seq-len"
        )

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    app = NeuronQwenImageTransformerApplication(model_path=component_dir, config=config)
    print(f"[sp-parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[sp-parity] loading weights to device")
    app.load(out_dir)

    torch.manual_seed(args.seed)
    hidden = torch.randn([1, config.image_seq_len, int(config.in_channels)], dtype=torch.bfloat16)
    timestep = torch.randn([1], dtype=torch.bfloat16)
    encoder = torch.randn([1, args.text_seq_len, int(config.joint_attention_dim)], dtype=torch.bfloat16)
    mask = torch.ones([1, args.text_seq_len], dtype=torch.bool)
    guidance = torch.randn([1], dtype=torch.bfloat16)

    print("[sp-parity] running forward")
    with torch.no_grad():
        out = app.forward(hidden, timestep, encoder, mask, guidance)
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[sp-parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
