#!/usr/bin/env python3
"""Tiny HunyuanVideo 1.5 Trainium compile/load smoke.

This uses a synthetic one-block HunyuanVideo15Transformer3DModel config and
random diffusers-compatible weights. It validates Nova's 1.5 fixed-boundary
wrapper without downloading the production 8.3B checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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
from safetensors.torch import save_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/tmp/nova_hunyuan15_tiny_model")
    parser.add_argument("--cache-dir", default="/tmp/nova_hunyuan15_tiny_cache")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--text-seq-len", type=int, default=7)
    parser.add_argument("--text-seq-len-2", type=int, default=3)
    parser.add_argument("--image-seq-len", type=int, default=4)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--metrics-out", default=None)
    return parser


def _tiny_config() -> dict[str, Any]:
    return {
        "_class_name": "HunyuanVideo15Transformer3DModel",
        "in_channels": 65,
        "out_channels": 32,
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "num_layers": 1,
        "num_refiner_layers": 1,
        "mlp_ratio": 2.0,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "text_embed_dim": 12,
        "text_embed_2_dim": 10,
        "image_embed_dim": 6,
        "rope_theta": 256.0,
        "rope_axes_dim": [2, 2, 4],
        "target_size": 640,
        "task_type": "t2v",
    }


def _prepare_model_dir(model_dir: Path) -> None:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    config = _tiny_config()
    (transformer_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = HunyuanVideo15Transformer3DModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        num_layers=config["num_layers"],
        num_refiner_layers=config["num_refiner_layers"],
        mlp_ratio=config["mlp_ratio"],
        patch_size=config["patch_size"],
        patch_size_t=config["patch_size_t"],
        qk_norm=config["qk_norm"],
        text_embed_dim=config["text_embed_dim"],
        text_embed_2_dim=config["text_embed_2_dim"],
        image_embed_dim=config["image_embed_dim"],
        rope_theta=config["rope_theta"],
        rope_axes_dim=tuple(config["rope_axes_dim"]),
        target_size=config["target_size"],
        task_type=config["task_type"],
    )
    save_file(model.state_dict(), str(transformer_dir / "diffusion_pytorch_model.safetensors"))


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("NOVA_BACKEND", "trainium")
    os.environ.setdefault("NEURON_RT_NUM_CORES", str(args.tp_degree))
    os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")

    from nova import NovaParallelConfig, NovaPipeline
    from nova.models.hunyuan_video.application import HunyuanVideo15DiTInputBundle

    model_dir = Path(args.model_dir)
    cache_dir = Path(args.cache_dir)
    if args.force_clean:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    _prepare_model_dir(model_dir)

    app_kwargs = {
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
    }
    t0 = time.perf_counter()
    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="hunyuan_video_15",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype="bf16",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=str(cache_dir),
        force_compile=args.force_clean,
        load=args.load,
        skip_warmup=args.skip_warmup,
        application_kwargs=app_kwargs,
    )
    compile_load_elapsed = time.perf_counter() - t0
    metrics: dict[str, Any] = {
        "compiled_path": str(pipe.compiled_path),
        "compile_load_elapsed_s": compile_load_elapsed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "tp_degree": args.tp_degree,
        "loaded": bool(args.load),
    }

    if args.load:
        latent_frames = (args.num_frames - 1) // 4 + 1
        latent_height = args.height // 16
        latent_width = args.width // 16
        bundle = HunyuanVideo15DiTInputBundle(
            hidden_states=torch.randn([1, 65, latent_frames, latent_height, latent_width], dtype=torch.bfloat16),
            timestep=torch.ones([1], dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn([1, args.text_seq_len, 12], dtype=torch.bfloat16),
            encoder_attention_mask=torch.ones([1, args.text_seq_len], dtype=torch.int64),
            timestep_r=torch.ones([1], dtype=torch.bfloat16),
            encoder_hidden_states_2=torch.randn([1, args.text_seq_len_2, 10], dtype=torch.bfloat16),
            encoder_attention_mask_2=torch.ones([1, args.text_seq_len_2], dtype=torch.int64),
            image_embeds=torch.zeros([1, args.image_seq_len, 6], dtype=torch.bfloat16),
        )
        t1 = time.perf_counter()
        with torch.no_grad():
            output = pipe(bundle)
        forward_elapsed = time.perf_counter() - t1
        if isinstance(output, (tuple, list)):
            output = output[0]
        metrics.update(
            {
                "forward_elapsed_s": forward_elapsed,
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "output_mean": float(output.float().mean()),
            }
        )

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-tiny] metrics -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
