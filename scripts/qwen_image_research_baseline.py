#!/usr/bin/env python3
"""Run a Qwen-Image Trainium transformer research baseline.

This intentionally measures the cached-embedding latent loop only. Qwen2.5-VL
prompt encoding and VAE decode stay host-side/out of scope for M4a research
closure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
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
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Parent dir containing transformer/")
    parser.add_argument("--bundle", required=True, help="Cached Qwen DiT inputs safetensors")
    parser.add_argument("--cache-dir", default=".nova-cache/qwen_image_transformer_full")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--text-seq-len", type=int, default=None)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--metrics-out", default="/tmp/nova_qwen_image_research_baseline_metrics.json")
    parser.add_argument("--save-latents", default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_meta(bundle_path: Path) -> dict[str, Any]:
    meta_path = Path(str(bundle_path) + ".meta.json")
    if meta_path.exists():
        return _load_json(meta_path)
    return {}


def _meta_or_arg(meta: dict[str, Any], args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    if name in meta:
        return int(meta[name])
    raise ValueError(f"--{name.replace('_', '-')} is required when bundle meta is missing")


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("sample", "images", "frames", "latents"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if hasattr(value, "sample") and torch.is_tensor(value.sample):
        return value.sample
    raise TypeError(f"could not extract tensor from {type(value)!r}")


def _load_bundle(bundle_path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    required = {
        "latents_init",
        "timesteps",
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "guidance",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise KeyError(f"bundle {bundle_path} is missing tensors: {missing}")
    return {
        "latents": tensors["latents_init"].to(dtype=dtype).contiguous(),
        "timesteps": tensors["timesteps"].contiguous(),
        "encoder_hidden_states": tensors["encoder_hidden_states"].to(dtype=dtype).contiguous(),
        "encoder_hidden_states_mask": tensors["encoder_hidden_states_mask"].to(torch.bool).contiguous(),
        "guidance": tensors["guidance"].to(dtype=dtype).contiguous(),
    }


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    from nova import NovaParallelConfig, NovaPipeline
    from nova.models.qwen_image.application import QwenImageDiTInputBundle

    bundle_path = Path(args.bundle)
    meta = _bundle_meta(bundle_path)
    tensors = _load_bundle(bundle_path, args.dtype)
    height = _meta_or_arg(meta, args, "height")
    width = _meta_or_arg(meta, args, "width")
    text_seq_len = _meta_or_arg(meta, args, "text_seq_len")

    print(
        f"[qwen-research] load shape={height}x{width} text_seq_len={text_seq_len} "
        f"tp={args.tp_degree}",
        flush=True,
    )
    t0 = time.perf_counter()
    pipe = NovaPipeline.from_pretrained(
        args.model_dir,
        model_type="qwen_image",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=height,
        width=width,
        compile_cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=True,
        skip_warmup=args.skip_warmup,
        application_kwargs={"text_seq_len": text_seq_len},
    )
    load_elapsed = time.perf_counter() - t0
    print(f"[qwen-research] load/compile elapsed = {load_elapsed:.3f}s", flush=True)
    print(f"[qwen-research] compiled_path = {pipe.compiled_path}", flush=True)

    latents = tensors["latents"]
    timesteps = tensors["timesteps"]
    scheduler_loop = getattr(pipe.app, "pipeline", None)
    per_step_elapsed: list[float] = []

    t_loop = time.perf_counter()
    with torch.no_grad():
        for step_index, timestep in enumerate(timesteps):
            timestep_batch = timestep.to(dtype=args.dtype).reshape(1).expand(latents.shape[0]).contiguous()
            bundle = QwenImageDiTInputBundle(
                hidden_states=latents.to(dtype=args.dtype).contiguous(),
                timestep=timestep_batch,
                encoder_hidden_states=tensors["encoder_hidden_states"],
                encoder_hidden_states_mask=tensors["encoder_hidden_states_mask"],
                guidance=tensors["guidance"],
            )
            t_step = time.perf_counter()
            noise_pred = _first_tensor(pipe(bundle))
            if scheduler_loop is not None:
                latents = scheduler_loop._scheduler_step(  # noqa: SLF001
                    noise_pred,
                    timestep,
                    latents,
                    int(timesteps.shape[0]),
                )
            else:
                latents = latents - noise_pred.to(dtype=latents.dtype) / float(max(int(timesteps.shape[0]), 1))
            elapsed = time.perf_counter() - t_step
            per_step_elapsed.append(elapsed)
            print(f"[qwen-research] step {step_index} elapsed = {elapsed:.3f}s", flush=True)
    loop_elapsed = time.perf_counter() - t_loop

    final_latents = latents.detach().cpu()
    metrics = {
        "bundle": str(bundle_path),
        "compiled_path": str(pipe.compiled_path),
        "model_dir": str(args.model_dir),
        "height": height,
        "width": width,
        "text_seq_len": text_seq_len,
        "tp_degree": int(args.tp_degree),
        "dtype": str(args.dtype),
        "num_steps": int(timesteps.shape[0]),
        "load_compile_elapsed_s": load_elapsed,
        "loop_elapsed_s": loop_elapsed,
        "per_step_elapsed_s": per_step_elapsed,
        "per_step_mean_s": mean(per_step_elapsed),
        "per_step_pstdev_s": pstdev(per_step_elapsed) if len(per_step_elapsed) > 1 else 0.0,
        "final_latents_shape": list(final_latents.shape),
        "final_latents_mean": float(final_latents.float().mean()),
        "final_latents_std": float(final_latents.float().std()),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[qwen-research] metrics -> {metrics_path}", flush=True)

    if args.save_latents:
        torch.save(final_latents, args.save_latents)
        print(f"[qwen-research] final latents -> {args.save_latents}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
