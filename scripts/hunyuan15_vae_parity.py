#!/usr/bin/env python3
"""Run a HunyuanVideo 1.5 VAE decoder Trainium parity/runtime gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402

from difflet.backends.trainium.core.config import NeuronConfig  # noqa: E402
from difflet.backends.trainium.hunyuan_video.vae15 import (  # noqa: E402
    HunyuanVideo15VAEDecoderInferenceConfig,
    HunyuanVideo15VAEDecoderModel,
    NeuronHunyuanVideo15VAEDecoderApplication,
)
from difflet.utils.diffusers_adapter import load_diffusers_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--tile-sample-min-height", type=int, default=32)
    parser.add_argument("--tile-sample-min-width", type=int, default=32)
    parser.add_argument("--tile-overlap-factor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latents-in")
    parser.add_argument("--save-trainium")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--metrics-out", default="/tmp/difflet_hunyuan15_vae15_parity_metrics.json")
    parser.add_argument(
        "--reference",
        choices=["difflet", "diffusers"],
        default="difflet",
        help="difflet: this repo's decoder module on the CPU, untiled; diffusers: diffusers' "
        "AutoencoderKLHunyuanVideo15 on the CPU with enable_tiling() at the same tile settings",
    )
    parser.add_argument("--reference-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--untiled-reference",
        action="store_true",
        help="also decode untiled in fp32 with diffusers and report the distance to it (seam check)",
    )
    parser.add_argument(
        "--latent-frames",
        type=int,
        default=None,
        help="with --latents-in: keep the first N latent frames (the causal VAE decodes them alone)",
    )
    parser.add_argument(
        "--divide-scaling-factor",
        action="store_true",
        help="with --latents-in: divide by the VAE's scaling factor first, as the pipelines do",
    )
    return parser


def _make_config(args: argparse.Namespace) -> HunyuanVideo15VAEDecoderInferenceConfig:
    return HunyuanVideo15VAEDecoderInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=1,
            world_size=1,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(args.vae_dir),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        tile_sample_min_height=int(args.tile_sample_min_height),
        tile_sample_min_width=int(args.tile_sample_min_width),
        tile_overlap_factor=float(args.tile_overlap_factor),
    )


def _reference_decode(
    app: NeuronHunyuanVideo15VAEDecoderApplication,
    config: HunyuanVideo15VAEDecoderInferenceConfig,
    latents: torch.Tensor,
) -> torch.Tensor:
    model = HunyuanVideo15VAEDecoderModel(config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(app.checkpoint_loader_fn(), strict=True)
    with torch.no_grad():
        return model(latents)


def _diffusers_decode(
    vae_dir: str,
    latents: torch.Tensor,
    dtype: torch.dtype,
    tiling: dict | None,
) -> torch.Tensor:
    from diffusers import AutoencoderKLHunyuanVideo15

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(vae_dir, torch_dtype=dtype).eval()
    if tiling is not None:
        vae.enable_tiling(**tiling)
    with torch.no_grad():
        return vae.decode(latents.to(dtype), return_dict=False)[0]


def _psnr(a: torch.Tensor, ref: torch.Tensor) -> float:
    """PSNR in dB on frames clamped to [-1, 1] and mapped to [0, 1] (peak 1)."""
    a01 = (a.float().clamp(-1, 1) + 1) / 2
    r01 = (ref.float().clamp(-1, 1) + 1) / 2
    mse = float(((a01 - r01) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main() -> int:
    args = build_parser().parse_args()
    config = _make_config(args)
    app = NeuronHunyuanVideo15VAEDecoderApplication(model_path=args.vae_dir, config=config)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    compile_elapsed = None
    if args.skip_compile:
        config.save(cache_dir)
        if not (cache_dir / "model.pt").exists():
            raise FileNotFoundError(f"Missing compiled VAE artifact: {cache_dir / 'model.pt'}")
    else:
        t_compile = time.perf_counter()
        app.compile(str(cache_dir))
        compile_elapsed = time.perf_counter() - t_compile

    t_load = time.perf_counter()
    app.load(str(cache_dir), skip_warmup=args.skip_warmup)
    load_elapsed = time.perf_counter() - t_load

    if args.latents_in:
        latents = torch.load(args.latents_in, map_location="cpu")
        if isinstance(latents, dict):
            latents = latents.get("latents", latents.get("sample"))
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"--latents-in must contain a tensor, got {type(latents)!r}")
        if args.latent_frames is not None:
            latents = latents[:, :, : int(args.latent_frames)]
        if args.divide_scaling_factor:
            latents = latents.float() / float(config.scaling_factor)
        latents = latents.to(dtype=torch.bfloat16)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        latents = torch.randn(
            [
                1,
                int(config.latent_channels),
                int(config.latent_frames),
                int(config.latent_height),
                int(config.latent_width),
            ],
            generator=generator,
            dtype=torch.bfloat16,
        )
    expected_shape = (
        1,
        int(config.latent_channels),
        int(config.latent_frames),
        int(config.latent_height),
        int(config.latent_width),
    )
    if tuple(latents.shape) != expected_shape:
        raise ValueError(f"Latents have shape {tuple(latents.shape)}, expected {expected_shape}.")

    t_forward = time.perf_counter()
    with torch.no_grad():
        trainium = app.decode(latents, return_dict=False)[0]
    forward_elapsed = time.perf_counter() - t_forward

    metrics = {
        "cache_dir": str(cache_dir),
        "latent_shape": list(latents.shape),
        "trainium_shape": list(trainium.shape),
        "compile_elapsed_s": compile_elapsed,
        "load_elapsed_s": load_elapsed,
        "forward_elapsed_s": forward_elapsed,
        "trainium_mean": float(trainium.float().mean()),
        "trainium_absmax": float(trainium.float().abs().max()),
    }
    if not args.skip_reference:
        t_ref = time.perf_counter()
        if args.reference == "diffusers":
            reference = _diffusers_decode(
                args.vae_dir,
                latents,
                torch.bfloat16 if args.reference_dtype == "bf16" else torch.float32,
                {
                    "tile_sample_min_height": int(args.tile_sample_min_height),
                    "tile_sample_min_width": int(args.tile_sample_min_width),
                    "tile_overlap_factor": float(args.tile_overlap_factor),
                },
            )
        else:
            reference = _reference_decode(app, config, latents)
        ref_elapsed = time.perf_counter() - t_ref
        diff = (trainium.float() - reference.float()).abs()
        metrics.update(
            {
                "reference_elapsed_s": ref_elapsed,
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        trainium.float().flatten(),
                        reference.float().flatten(),
                        dim=0,
                    )
                ),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "reference": args.reference + (f" {args.reference_dtype}" if args.reference == "diffusers" else ""),
                "psnr_db": _psnr(trainium, reference),
                "nan": int(torch.isnan(trainium.float()).sum()),
                "inf": int(torch.isinf(trainium.float()).sum()),
            }
        )
    if args.untiled_reference:
        untiled = _diffusers_decode(args.vae_dir, latents, torch.float32, None)
        metrics["untiled_fp32_psnr_db"] = {"trainium": _psnr(trainium, untiled)}
        if not args.skip_reference:
            metrics["untiled_fp32_psnr_db"]["reference"] = _psnr(reference, untiled)
    if args.save_trainium:
        save_path = Path(args.save_trainium)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainium.detach().cpu(), save_path)
        metrics["save_trainium"] = str(save_path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
