#!/usr/bin/env python3
"""HunyuanVideo VAE parity: Trainium VAE vs HF CPU AutoencoderKLHunyuanVideo.

Decodes the same latent through both VAEs in one process and reports cosine /
max-abs / mean-abs deltas. Numerical gate for `cclogs/m3-hunyuan/37` §4 item 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
from nova.pipeline.parallel_config import NovaParallelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real",
        help="Parent dir containing vae/ (HF source).",
    )
    parser.add_argument(
        "--compiled-dir",
        default="/tmp/nova_hunyuan_vae_decoder_ws4_smoke",
        help="Directory containing the ws=4 vae_decoder/ Trainium artifact.",
    )
    parser.add_argument(
        "--latents",
        default="/home/ubuntu/nova/.nova-cache/hunyuan_dit_inputs/cat_walking_4step_nova_latents.pt",
        help="Latent tensor .pt produced by the Nova DiT (shape (1,16,16,40,64)).",
    )
    parser.add_argument(
        "--height", type=int, default=320, help="Sample height; must match compiled artifact."
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Sample width; must match compiled artifact."
    )
    parser.add_argument(
        "--frames", type=int, default=61, help="Frame count; must match compiled artifact."
    )
    parser.add_argument(
        "--tp-degree", type=int, default=4, help="World size used at compile (ws=4 by default)."
    )
    parser.add_argument(
        "--metrics", default=None, help="Optional path to dump JSON metrics."
    )
    parser.add_argument(
        "--save-trainium",
        default=None,
        help="Optional path to dump Trainium decoded video tensor (.pt).",
    )
    parser.add_argument(
        "--save-cpu", default=None, help="Optional path to dump HF CPU decoded video tensor (.pt)."
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.999,
        help="Gate threshold for cosine (default 0.999 to match M3 single-step alignment).",
    )
    return parser.parse_args()


def _stats_pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32 = a.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    b32 = b.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    diff = a32 - b32
    cosine = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item()
    return {
        "cosine": float(cosine),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "trainium_mean": float(a32.mean().item()),
        "trainium_std": float(a32.std().item()),
        "cpu_mean": float(b32.mean().item()),
        "cpu_std": float(b32.std().item()),
    }


def main() -> int:
    args = parse_args()
    print(f"[vae-parity] model_dir   = {args.model_dir}")
    print(f"[vae-parity] compiled_dir= {args.compiled_dir}")
    print(f"[vae-parity] latents     = {args.latents}")
    print(f"[vae-parity] shape (h,w,f) = ({args.height}, {args.width}, {args.frames})")
    print(f"[vae-parity] tp_degree   = {args.tp_degree}")

    latents = torch.load(args.latents, map_location="cpu")
    print(f"[vae-parity] latent shape = {tuple(latents.shape)} dtype = {latents.dtype}")
    print(
        "[vae-parity] latent mean/std = "
        f"{latents.float().mean().item():.6e} / {latents.float().std().item():.6e}"
    )

    print("[vae-parity] build NeuronHunyuanVideoApplication (vae-only)...")
    app = NeuronHunyuanVideoApplication(
        model_path=args.model_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree, cp_enabled=False),
        dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.frames},
        enable_transformer=False,
        enable_vae_decoder=True,
    )
    assert app.vae_decoder is not None, "vae_decoder failed to initialize"

    t0 = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    print(f"[vae-parity] trainium load elapsed = {time.time() - t0:.3f}s")

    scaling = float(app.vae_decoder.config.scaling_factor)
    print(f"[vae-parity] scaling_factor = {scaling}")
    pre_scaled = latents.to(dtype=torch.bfloat16) / scaling

    print("[vae-parity] trainium decode ...")
    t1 = time.time()
    trainium_out = app.vae_decoder.decode(pre_scaled, return_dict=False)[0]
    trainium_elapsed = time.time() - t1
    print(f"[vae-parity] trainium decode elapsed = {trainium_elapsed:.3f}s")
    print(f"[vae-parity] trainium video shape = {tuple(trainium_out.shape)} dtype = {trainium_out.dtype}")
    print(f"[vae-parity] trainium finite all = {bool(torch.isfinite(trainium_out).all())}")
    trainium_cpu = trainium_out.detach().to(device="cpu", dtype=torch.float32)
    if args.save_trainium:
        out_path = Path(args.save_trainium)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainium_cpu, out_path)
        print(f"[vae-parity] saved trainium tensor -> {out_path}")

    print("[vae-parity] HF VAE (CPU) load ...")
    from diffusers import AutoencoderKLHunyuanVideo

    t2 = time.time()
    hf_vae = AutoencoderKLHunyuanVideo.from_pretrained(
        Path(args.model_dir) / "vae", torch_dtype=torch.bfloat16
    ).eval()
    hf_vae.enable_tiling()
    print(f"[vae-parity] HF load elapsed = {time.time() - t2:.3f}s")

    print("[vae-parity] HF decode (this is the ~150s CPU step)...")
    t3 = time.time()
    with torch.no_grad():
        hf_out = hf_vae.decode(pre_scaled, return_dict=False)[0]
    hf_elapsed = time.time() - t3
    print(f"[vae-parity] HF decode elapsed = {hf_elapsed:.3f}s")
    print(f"[vae-parity] HF video shape = {tuple(hf_out.shape)} dtype = {hf_out.dtype}")
    print(f"[vae-parity] HF finite all = {bool(torch.isfinite(hf_out).all())}")
    hf_cpu = hf_out.detach().to(device="cpu", dtype=torch.float32)
    if args.save_cpu:
        out_path = Path(args.save_cpu)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(hf_cpu, out_path)
        print(f"[vae-parity] saved HF tensor -> {out_path}")

    if trainium_cpu.shape != hf_cpu.shape:
        raise RuntimeError(
            f"shape mismatch: trainium {tuple(trainium_cpu.shape)} vs HF {tuple(hf_cpu.shape)}"
        )

    stats = _stats_pair(trainium_cpu, hf_cpu)
    stats["trainium_decode_s"] = trainium_elapsed
    stats["hf_decode_s"] = hf_elapsed
    stats["scaling_factor"] = scaling
    stats["latent_shape"] = list(latents.shape)
    stats["output_shape"] = list(trainium_cpu.shape)
    stats["min_cosine_threshold"] = args.min_cosine

    print(f"[vae-parity] cosine     = {stats['cosine']:.10f}")
    print(f"[vae-parity] max_abs    = {stats['max_abs']:.6e}")
    print(f"[vae-parity] mean_abs   = {stats['mean_abs']:.6e}")
    print(f"[vae-parity] trainium mean/std (fp32) = {stats['trainium_mean']:.6e} / {stats['trainium_std']:.6e}")
    print(f"[vae-parity] HF        mean/std (fp32) = {stats['cpu_mean']:.6e} / {stats['cpu_std']:.6e}")

    if args.metrics:
        m_path = Path(args.metrics)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(json.dumps(stats, indent=2))
        print(f"[vae-parity] metrics -> {m_path}")

    pass_gate = stats["cosine"] >= args.min_cosine
    print(f"[vae-parity] gate cosine >= {args.min_cosine}: {'PASS' if pass_gate else 'FAIL'}")
    return 0 if pass_gate else 2


if __name__ == "__main__":
    sys.exit(main())
