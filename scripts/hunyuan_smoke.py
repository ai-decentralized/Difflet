"""HunyuanVideo end-to-end M3 v0 smoke.

Single-process driver for the M3 hybrid path:

  cached DiT input artifact  -->  Difflet Trainium DiT (4 steps)
                               -->  HF VAE decode (CPU, default)
                                    or Difflet Trainium VAE decode (opt-in)
                               -->  (1, 3, T, H, W) bf16 video tensor
                               -->  optional best-effort MP4 export

Exercises the public ``NeuronHunyuanVideoApplication.__call__(bundle=...,
output_type="pt")`` path so the same call shape works when reached via
``DiffletPipeline.from_pretrained``.

Usage:

    scripts/hunyuan_smoke.sh    # default paths from cclog 29 §7.x

or

    python scripts/hunyuan_smoke.py \\
        --source-dir /home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real \\
        --compiled-dir /home/ubuntu/difflet/.difflet-cache/hunyuan_n4_20d40s2r/compiled \\
        --bundle /home/ubuntu/difflet/.difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors \\
        --output /tmp/hunyuan_smoke.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True,
                        help="Parent dir containing transformer/ and vae/ subdirs")
    parser.add_argument("--compiled-dir", required=True,
                        help="Parent dir containing transformer/model.pt + neuron_config.json")
    parser.add_argument("--bundle", required=True,
                        help="Cached DiT inputs safetensors (.meta.json sidecar required)")
    parser.add_argument("--output", default="/tmp/hunyuan_smoke.mp4",
                        help="Output path; .mp4 attempts video export, .pt always saves tensor")
    parser.add_argument("--save-tensor", default=None,
                        help="Explicit .pt path; defaults to <output basename>.pt next to --output")
    parser.add_argument("--save-latents", default=None,
                        help="Optional .pt path for the final latent (pre-VAE)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override; defaults to len(timesteps) from cached artifact")
    parser.add_argument("--fps", type=int, default=15,
                        help="MP4 framerate (HunyuanVideo diffusers default is 15)")
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument(
        "--enable-trainium-vae",
        action="store_true",
        help="Load compiled vae_decoder/ and use Difflet Trainium VAE decode instead of HF CPU VAE.",
    )
    return parser.parse_args()


def _try_export_mp4(video: torch.Tensor, output_path: str, fps: int) -> bool:
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        print("[smoke] diffusers.utils.export_to_video unavailable; mp4 export skipped",
              flush=True)
        return False
    frames = video.detach().to(torch.float32).clamp(-1, 1)
    frames = ((frames + 1.0) / 2.0).clamp(0, 1)
    frames = (frames[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype("uint8")
    try:
        export_to_video(list(frames), output_path, fps=fps)
    except Exception as exc:
        print(f"[smoke] mp4 export failed ({exc}); skipping", flush=True)
        return False
    print(f"[smoke] mp4 -> {output_path}", flush=True)
    return True


def main() -> int:
    args = parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")

    meta = json.loads(Path(args.bundle + ".meta.json").read_text())
    tensors = load_file(args.bundle)
    print(f"[smoke] bundle = {args.bundle}")
    print(f"[smoke] prompt = {meta.get('prompt')}")
    print(f"[smoke] shape = {meta['height']}x{meta['width']}x{meta['num_frames']}, "
          f"text_seq_len={meta['text_seq_len']}, cached_steps={meta['num_inference_steps']}")

    from difflet.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
        text_seq_len=meta["text_seq_len"],
        enable_vae_decoder=args.enable_trainium_vae,
    )
    print(f"[smoke] load(skip_warmup=True) from {args.compiled_dir} ...", flush=True)
    t_load = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    print(f"[smoke] load elapsed = {time.time() - t_load:.3f}s")

    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=tensors["timesteps"][:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )

    num_steps = args.num_inference_steps or int(meta["num_inference_steps"])
    print(f"[smoke] orchestrator output_type='pt' num_inference_steps={num_steps} ...",
          flush=True)
    t_run = time.time()
    output = app(
        bundle=bundle,
        timesteps=tensors["timesteps"],
        num_inference_steps=num_steps,
        output_type="pt",
        return_trajectory=False,
    )
    print(f"[smoke] full pipeline elapsed = {time.time() - t_run:.3f}s")

    frames = output.frames
    latents = output.latents
    print(f"[smoke] video shape = {tuple(frames.shape)} dtype = {frames.dtype}")
    print(f"[smoke] video finite all = {bool(torch.isfinite(frames).all())}")
    print(f"[smoke] video mean/std (fp32 cast) = "
          f"{frames.float().mean().item():.6e} / {frames.float().std().item():.6e}")
    print(f"[smoke] latent shape = {tuple(latents.shape)} dtype = {latents.dtype}")
    print(f"[smoke] latent mean/std (fp32 cast) = "
          f"{latents.float().mean().item():.6e} / {latents.float().std().item():.6e}")

    output_path = Path(args.output)
    tensor_path = Path(args.save_tensor) if args.save_tensor else output_path.with_suffix(".pt")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frames.cpu(), tensor_path)
    print(f"[smoke] video tensor -> {tensor_path}")

    if args.save_latents:
        latents_path = Path(args.save_latents)
        latents_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.cpu(), latents_path)
        print(f"[smoke] latent tensor -> {latents_path}")

    if output_path.suffix == ".mp4":
        mp4_ok = _try_export_mp4(frames, str(output_path), args.fps)
        if not mp4_ok:
            print(f"[smoke] tensor saved at {tensor_path}; mp4 path {output_path} skipped",
                  flush=True)

    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
