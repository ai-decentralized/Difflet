#!/usr/bin/env python3
"""Does decoding the Wan VAE one latent frame at a time equal decoding all at once?

The device VAE compiles the whole temporal loop as one graph, so its instruction
count grows with frames and neuronx-cc rejects it past 3 latent frames at 480x832
(NCC_EBVF030, limit 5,000,000; measured 6.68M at 4 latent frames). The loop in
WanVAEDecoderModel.forward already runs one latent frame per iteration, carrying a
causal feat_cache between them -- the same structure upstream diffusers uses
(AutoencoderKLWan._decode). So a per-frame graph called N times should be an
equivalence, not an approximation.

This checks that claim on CPU before any of it is built for device, and records
the feat_cache slot shapes a per-frame graph would have to declare as inputs and
outputs.

    python scripts/wan_vae_chunked_parity.py --latent-frames 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


def _tiny_config() -> WanVAEDecoderConfig:
    """A small decoder with the real module structure, so this runs on CPU."""
    return WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 4],
        num_res_blocks=1,
        attn_scales=[],
        # The decoder derives its temporal upsampling from this, so keep a stage
        # that actually upsamples in time -- that is the path carrying the "Rep"
        # sentinel, which is the awkward part for a fixed-shape graph.
        temperal_downsample=[False, True],
        dropout=0.0,
        out_channels=3,
    )


def decode_chunked(model: WanVAEDecoderModel, z: torch.Tensor):
    """Frame-by-frame decode, exposing the cache between calls.

    This mirrors WanVAEDecoderModel.forward, except the caller owns feat_cache,
    which is what a per-frame device graph would do: the cache becomes graph
    state instead of a Python list captured inside one traced loop.
    """
    num_frames = z.shape[2]
    feat_cache = model._clear_cache()
    zq = model.post_quant_conv(z)
    chunks = []
    cache_shapes: list[list] = []
    for i in range(num_frames):
        conv_idx = [0]
        chunk = model.decoder(
            zq[:, :, i : i + 1, :, :],
            feat_cache=feat_cache,
            feat_idx=conv_idx,
            first_chunk=i == 0,
        )
        chunks.append(chunk)
        cache_shapes.append(
            [
                None if c is None else ("Rep" if isinstance(c, str) else list(c.shape))
                for c in feat_cache
            ]
        )
    out = torch.clamp(torch.cat(chunks, dim=2), min=-1.0, max=1.0)
    return out, feat_cache, cache_shapes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-frames", type=int, default=6)
    p.add_argument("--height", type=int, default=16, help="latent height")
    p.add_argument("--width", type=int, default=16, help="latent width")
    p.add_argument("--out", type=Path, default=None, help="write the cache-shape record here")
    args = p.parse_args()

    torch.manual_seed(0)
    config = _tiny_config()
    model = WanVAEDecoderModel(config).eval()
    z = torch.randn(1, config.z_dim, args.latent_frames, args.height, args.width)

    with torch.no_grad():
        reference = model(z)
        chunked, feat_cache, cache_shapes = decode_chunked(model, z)

    same_shape = reference.shape == chunked.shape
    max_abs = float((reference - chunked).abs().max()) if same_shape else float("nan")
    bit_identical = bool(torch.equal(reference, chunked)) if same_shape else False

    print(f"latent frames      : {args.latent_frames}")
    print(f"output shape       : {tuple(reference.shape)} vs {tuple(chunked.shape)}")
    print(f"max abs difference : {max_abs:.3e}")
    print(f"bit identical      : {bit_identical}")

    # What a per-frame graph must carry between calls.
    slots = cache_shapes[-1]
    tensor_slots = [(i, s) for i, s in enumerate(slots) if isinstance(s, list)]
    print(f"\ncache slots        : {len(slots)} total, {len(tensor_slots)} hold tensors")
    sentinels = [(i, s) for i, s in enumerate(slots) if not isinstance(s, list)]
    if sentinels:
        print(f"non-tensor slots   : {sentinels}")
    for i, shape in tensor_slots[:8]:
        print(f"  slot {i:<3} shape {shape}")
    if len(tensor_slots) > 8:
        print(f"  ... {len(tensor_slots) - 8} more")

    # Do the slot shapes settle after the first frame? A per-frame graph needs
    # fixed shapes, so a slot that keeps changing would have to be bucketed.
    settled = True
    if len(cache_shapes) > 2:
        for a, b in zip(cache_shapes[1:-1], cache_shapes[2:]):
            if a != b:
                settled = False
                break
    print(f"\nslot shapes settle after frame 1: {settled}")
    print(f"frame 0 slots differ from steady state: {cache_shapes[0] != cache_shapes[-1]}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "latent_frames": args.latent_frames,
                    "bit_identical": bit_identical,
                    "max_abs_difference": max_abs,
                    "cache_shapes_per_frame": cache_shapes,
                    "steady_state_slots": slots,
                    "slot_shapes_settle": settled,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")

    return 0 if bit_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
