#!/usr/bin/env python3
"""Clone HunyuanVideo DiT input bundle with multiple random seeds.

Reuses the text-encoded fields (encoder_hidden_states, encoder_attention_mask,
pooled_projections, guidance, timesteps) from an existing bundle and replaces
``latents_init`` with newly sampled noise from explicit per-output seeds.

This avoids running the HuggingFace HunyuanVideo text encoder (LLaMA-3-8B not
locally cached) while still producing distinct (mod_input_diff, noise_pred_diff)
trajectories for TeaCache calibration sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bundle", required=True, help="Path to existing .safetensors bundle")
    p.add_argument("--output-dir", required=True, help="Directory to write cloned bundles into")
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds for latents_init (one bundle per seed)",
    )
    p.add_argument(
        "--prefix",
        default="cat_walking_4step_seed",
        help="Output filename prefix (default: cat_walking_4step_seed)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    src = Path(args.source_bundle)
    meta_src = Path(str(src) + ".meta.json")
    if not src.exists():
        print(f"missing source bundle: {src}", file=sys.stderr)
        return 1
    if not meta_src.exists():
        print(f"missing source meta: {meta_src}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(src), device="cpu")
    meta = json.loads(meta_src.read_text())
    latent_shape = tensors["latents_init"].shape
    latent_dtype = tensors["latents_init"].dtype
    print(
        f"[clone] source bundle: {src} (latents_init {tuple(latent_shape)} {latent_dtype})",
        flush=True,
    )

    for seed in args.seeds:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
        new_latents = torch.randn(
            *latent_shape, dtype=torch.float32, generator=rng
        ).to(dtype=latent_dtype)

        out_tensors = dict(tensors)
        out_tensors["latents_init"] = new_latents

        out_path = out_dir / f"{args.prefix}{int(seed)}.safetensors"
        save_file(out_tensors, str(out_path))

        out_meta = dict(meta)
        out_meta["seed"] = int(seed)
        out_meta["derived_from"] = str(src)
        out_meta["latents_resampled"] = True
        Path(str(out_path) + ".meta.json").write_text(
            json.dumps(out_meta, indent=2, sort_keys=True) + "\n"
        )
        print(f"[clone] wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
