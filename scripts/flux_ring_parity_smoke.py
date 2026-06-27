#!/usr/bin/env python
"""On-device Flux ring-vs-gather-KV parity: single-mode runner.

Runs ONE cp_mode (gather_kv | ring) of the Flux DiT backbone at tp=2/cp=2 on a
fixed-seed input and saves the output tensor to ``--out``. Each mode MUST
run in its own process because neuronx_distributed's parallel_state initializes
once per process. ``scripts/flux_ring_parity_smoke.sh`` drives both modes and
compares the two saved tensors.

A reduced ``--layers`` / ``--single-layers`` (default 2/2) keeps the compile
small; weights load with strict=False, so only the first blocks are populated
from the real checkpoint (identical weights across both modes → a fair compare).

Ring kernel constraints honored by the default shape:
  per-rank joint seqlen ((num_patches + text_len) // cp_degree) must be
  divisible by 128.
  Default 256×256 + text_len=512 at cp=2, vae_scale_factor=8:
    num_patches = 256*256//(2*8)**2 = 256, joint = 256+512 = 768, per_rank = 384 = 3×128 ✓
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Flux ring/gather-KV parity single-mode runner")
    p.add_argument("--mode", required=True, choices=["gather_kv", "ring"])
    p.add_argument("--out", required=True, help="path to save the output tensor (.pt)")
    p.add_argument("--model", default=os.environ.get("DIFFLET_FLUX_MODEL", "black-forest-labs/FLUX.1-dev"))
    p.add_argument("--subfolder", default=os.environ.get("DIFFLET_FLUX_TRANSFORMER_SUBFOLDER", "transformer"))
    p.add_argument("--tp-degree", type=int, default=int(os.environ.get("DIFFLET_FLUX_TP_DEGREE", "2")))
    p.add_argument("--cp-degree", type=int, default=int(os.environ.get("DIFFLET_FLUX_CP_DEGREE", "2")))
    p.add_argument("--layers", type=int, default=int(os.environ.get("DIFFLET_FLUX_LAYERS", "2")))
    p.add_argument("--single-layers", type=int, default=int(os.environ.get("DIFFLET_FLUX_SINGLE_LAYERS", "2")))
    p.add_argument("--height", type=int, default=int(os.environ.get("DIFFLET_FLUX_HEIGHT", "256")))
    p.add_argument("--width", type=int, default=int(os.environ.get("DIFFLET_FLUX_WIDTH", "256")))
    p.add_argument("--text-seq-len", type=int, default=int(os.environ.get("DIFFLET_FLUX_TEXT_SEQ_LEN", "512")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("DIFFLET_FLUX_SEED", "1234")))
    p.add_argument("--work-dir", default=os.environ.get("DIFFLET_FLUX_PARITY_WORKDIR", "/tmp/flux_ring_parity"))
    args = p.parse_args()

    from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig, NeuronFluxBackboneApplication
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.pipeline.path_resolver import resolve_model_path
    from difflet.utils.diffusers_adapter import load_diffusers_config

    world_size = args.tp_degree * args.cp_degree
    model_dir = resolve_model_path(args.model, local_files_only=True)
    component_dir = os.path.join(model_dir, args.subfolder)
    if not os.path.exists(os.path.join(component_dir, "config.json")):
        raise FileNotFoundError(f"missing {args.subfolder}/config.json under {model_dir}")

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        world_size=world_size,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    config = FluxBackboneInferenceConfig(
        context_parallel_enabled=args.cp_degree > 1,
        cp_mode=args.mode,
        neuron_config=neuron_config,
        load_config=load_diffusers_config(component_dir),
        height=args.height,
        width=args.width,
    )
    # Reduce depth for a fast compile; strict=False load populates only these
    # blocks from the real checkpoint (same weights for both modes).
    config.num_layers = args.layers
    config.num_single_layers = args.single_layers

    # Mirror the real create_flux_config path (application.py:161):
    #   setattr(backbone_config, "vae_scale_factor", decoder_config.vae_scale_factor)
    # where vae_scale_factor = 2 ** (len(block_out_channels) - 1) (modeling_vae.py:111).
    # We read the VAE config.json directly to avoid constructing a full
    # VAEDecoderInferenceConfig (which would require a separate NeuronConfig).
    import json as _json
    _vae_cfg_path = os.path.join(model_dir, "vae", "config.json")
    with open(_vae_cfg_path) as _f:
        _vae_cfg = _json.load(_f)
    vae_scale_factor = 2 ** (len(_vae_cfg["block_out_channels"]) - 1)
    config.vae_scale_factor = vae_scale_factor

    # num_patches formula matches ModelWrapperFluxBackbone.input_generator() (modeling_flux.py:1365):
    #   height * width // ((2 * vae_scale_factor) ** 2)
    # The factor of 2 comes from Flux's pack_latents: 2x2 spatial blocks → channels,
    # giving patches at 1/(2 * vae_scale_factor) of original resolution per side.
    num_patches = args.height * args.width // ((2 * vae_scale_factor) ** 2)
    joint_seq = num_patches + args.text_seq_len
    per_rank = joint_seq // args.cp_degree

    print(f"[parity] mode={args.mode} tp={args.tp_degree} cp={args.cp_degree} "
          f"world={world_size} layers={args.layers}/{args.single_layers} "
          f"shape={args.height}x{args.width}")
    print(f"[parity] num_patches={num_patches} text_len={args.text_seq_len} "
          f"joint={joint_seq} per_rank={per_rank} (per_rank % 128 = {per_rank % 128})")
    if args.cp_degree > 1 and args.mode == "ring" and per_rank % 128 != 0:
        raise ValueError(
            f"ring requires per-rank joint seqlen ({per_rank}) divisible by 128; "
            f"adjust height/width/text-seq-len"
        )

    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    app = NeuronFluxBackboneApplication(model_path=component_dir, config=config)
    print(f"[parity] compiling -> {out_dir}")
    app.compile(out_dir, debug=False)
    print("[parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed inputs: identical across both processes/modes.
    torch.manual_seed(args.seed)
    in_channels = config.in_channels
    joint_attention_dim = config.joint_attention_dim
    pooled_projection_dim = config.pooled_projection_dim

    hidden_states = torch.randn([1, num_patches, in_channels], dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn([1, args.text_seq_len, joint_attention_dim], dtype=torch.bfloat16)
    pooled_projections = torch.randn([1, pooled_projection_dim], dtype=torch.bfloat16)
    timestep = torch.randn([1], dtype=torch.bfloat16)
    guidance = torch.randn([1], dtype=torch.bfloat16) if getattr(config, "guidance_embeds", True) else torch.tensor([], dtype=torch.bfloat16)
    # ModelWrapperFluxBackbone.forward() computes image_rotary_emb internally from
    # img_ids + txt_ids via pos_embed. We use zero IDs (same for both modes → fair compare).
    img_ids = torch.zeros([num_patches, 3], dtype=torch.bfloat16)
    txt_ids = torch.zeros([args.text_seq_len, 3], dtype=torch.bfloat16)

    print("[parity] running forward")
    with torch.no_grad():
        out = app.forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
    # out is the noise prediction tensor [B, num_patches, in_channels]
    out = out.detach().to(torch.float32).cpu()
    torch.save(out, args.out)
    print(f"[parity] saved output {tuple(out.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
