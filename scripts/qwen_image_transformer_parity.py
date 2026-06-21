#!/usr/bin/env python3
"""Qwen-Image transformer parity: Trainium artifact vs CPU reference.

The default CPU reference is Difflet's trace module, matching the fixed-shape
Trainium boundary. Use ``--reference-mode diffusers`` to compare against the
upstream QwenImageTransformer2DModel path, including its text-mask semantics.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[1]}{os.pathsep}{env.get('PYTHONPATH', '')}"
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
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Parent dir containing transformer/")
    parser.add_argument("--bundle", required=True, help="Cached Qwen DiT inputs safetensors")
    parser.add_argument("--cache-dir", default=".difflet-cache/qwen_image_transformer_parity")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--text-seq-len", type=int, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--reference-mode", choices=("trace", "diffusers"), default="trace")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--save-trainium", default=None)
    parser.add_argument("--save-reference", default=None)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "sample" in output:
            return output["sample"]
        return output[next(iter(output))]
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"cannot extract tensor from output type {type(output)!r}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_meta(bundle_path: Path) -> dict[str, Any]:
    meta_path = Path(str(bundle_path) + ".meta.json")
    if meta_path.exists():
        return _load_json(meta_path)
    return {}


def _meta_or_arg(meta: dict[str, Any], args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name.replace("-", "_"))
    if value is not None:
        return int(value)
    if name in meta:
        return int(meta[name])
    raise ValueError(f"--{name.replace('_', '-')} is required when bundle meta is missing")


def _load_bundle(bundle_path: Path, step_index: int, dtype: torch.dtype):
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
    if step_index < 0 or step_index >= int(tensors["timesteps"].shape[0]):
        raise IndexError(
            f"step-index {step_index} outside timesteps length {tensors['timesteps'].shape[0]}"
        )
    timestep = tensors["timesteps"][step_index : step_index + 1]
    return {
        "hidden_states": tensors["latents_init"].to(dtype=dtype).contiguous(),
        "timestep": timestep.to(dtype=dtype).contiguous(),
        "encoder_hidden_states": tensors["encoder_hidden_states"].to(dtype=dtype).contiguous(),
        "encoder_hidden_states_mask": tensors["encoder_hidden_states_mask"].to(torch.bool).contiguous(),
        "guidance": tensors["guidance"].to(dtype=dtype).contiguous(),
    }


def _iter_safetensor_shards(component_dir: Path) -> list[Path]:
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        index_path = component_dir / index_name
        if index_path.exists():
            index = _load_json(index_path)
            return [component_dir / name for name in sorted(set(index["weight_map"].values()))]
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = component_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"no safetensors shard/index found under {component_dir}")


def _load_trace_state_dict(model: torch.nn.Module, transformer_dir: Path, dtype: torch.dtype) -> None:
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected_all: set[str] = set()
    for shard in _iter_safetensor_shards(transformer_dir):
        print(f"[qwen-parity] reference shard = {shard.name}", flush=True)
        state = load_safetensors_file(str(shard), device="cpu")
        state = {
            (key if key.startswith("transformer.") else f"transformer.{key}"): (
                value.to(dtype) if torch.is_floating_point(value) else value
            )
            for key, value in state.items()
        }
        _, unexpected = model.load_state_dict(state, strict=False)
        loaded.update(state)
        unexpected_all.update(unexpected)
        del state
        gc.collect()
    missing = sorted(expected.difference(loaded))
    unexpected = sorted(unexpected_all)
    allow_unexpected = any(
        os.environ.get(name, "").lower() in {"1", "true", "yes"}
        for name in ("DIFFLET_QWEN_ZERO_BLOCK_ATTN", "DIFFLET_QWEN_ZERO_BLOCK_MLP")
    )
    if missing or (unexpected and not allow_unexpected):
        raise RuntimeError(
            f"trace reference load mismatch: missing={missing[:8]} (n={len(missing)}), "
            f"unexpected={unexpected[:8]} (n={len(unexpected)})"
        )
    if unexpected:
        print(
            f"[qwen-parity] ignored unexpected diagnostic reference keys: n={len(unexpected)}",
            flush=True,
        )


def _run_trainium(args: argparse.Namespace, meta: dict[str, Any], inputs: dict[str, torch.Tensor]):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")

    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.models.qwen_image.application import QwenImageDiTInputBundle

    height = _meta_or_arg(meta, args, "height")
    width = _meta_or_arg(meta, args, "width")
    text_seq_len = _meta_or_arg(meta, args, "text_seq_len")
    print(
        f"[qwen-parity] trainium load/compile shape={height}x{width} "
        f"text_seq_len={text_seq_len} tp={args.tp_degree}",
        flush=True,
    )
    t0 = time.time()
    pipe = DiffletPipeline.from_pretrained(
        args.model_dir,
        model_type="qwen_image",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
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
    print(f"[qwen-parity] trainium load/compile elapsed = {time.time() - t0:.3f}s", flush=True)
    print(f"[qwen-parity] compiled_path = {pipe.compiled_path}", flush=True)
    bundle = QwenImageDiTInputBundle(**inputs)
    t1 = time.time()
    with torch.no_grad():
        output = _extract_tensor(pipe(bundle)).detach().cpu()
    print(f"[qwen-parity] trainium forward elapsed = {time.time() - t1:.3f}s", flush=True)
    return output, pipe


def _run_trace_reference(
    args: argparse.Namespace,
    pipe,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    from difflet.backends.trainium.qwen_image.transformer import _QwenImageTransformerTraceModule

    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    model = _QwenImageTransformerTraceModule(pipe.app.transformer.config).eval()
    _load_trace_state_dict(model, Path(args.model_dir) / "transformer", dtype)
    model.to(device=device, dtype=dtype)
    model_inputs = [
        inputs["hidden_states"].to(device=device, dtype=dtype),
        inputs["timestep"].to(device=device, dtype=dtype),
        inputs["encoder_hidden_states"].to(device=device, dtype=dtype),
        inputs["encoder_hidden_states_mask"].to(device=device),
        inputs["guidance"].to(device=device, dtype=dtype),
    ]
    t0 = time.time()
    with torch.no_grad():
        output = model(*model_inputs).detach().cpu()
    print(f"[qwen-parity] trace reference elapsed = {time.time() - t0:.3f}s", flush=True)
    return output


def _run_diffusers_reference(
    args: argparse.Namespace,
    pipe,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    transformer_dir = Path(args.model_dir) / "transformer"
    model = QwenImageTransformer2DModel.from_pretrained(
        transformer_dir,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()
    model.to(device=device, dtype=dtype)
    cfg = pipe.app.transformer.config
    img_shapes = [[(1, int(cfg.packed_height), int(cfg.packed_width))]]
    guidance = inputs["guidance"].to(device=device, dtype=dtype)
    if not bool(getattr(cfg, "guidance_embeds", False)):
        guidance = None
    t0 = time.time()
    with torch.no_grad():
        output = model(
            hidden_states=inputs["hidden_states"].to(device=device, dtype=dtype),
            timestep=inputs["timestep"].to(device=device, dtype=dtype),
            encoder_hidden_states=inputs["encoder_hidden_states"].to(device=device, dtype=dtype),
            encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"].to(device=device),
            guidance=guidance,
            img_shapes=img_shapes,
            return_dict=False,
        )[0].detach().cpu()
    print(f"[qwen-parity] diffusers reference elapsed = {time.time() - t0:.3f}s", flush=True)
    return output


def main() -> int:
    args = build_parser().parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    bundle_path = Path(args.bundle)
    meta = _bundle_meta(bundle_path)
    inputs = _load_bundle(bundle_path, args.step_index, args.dtype)
    print(f"[qwen-parity] bundle = {bundle_path}", flush=True)
    print(f"[qwen-parity] prompt = {meta.get('prompt')}", flush=True)
    print(f"[qwen-parity] step_index = {args.step_index}", flush=True)

    trainium, pipe = _run_trainium(args, meta, inputs)
    if args.reference_mode == "trace":
        reference = _run_trace_reference(args, pipe, inputs)
    else:
        reference = _run_diffusers_reference(args, pipe, inputs)

    if trainium.shape != reference.shape:
        raise RuntimeError(f"shape mismatch: trainium={tuple(trainium.shape)} reference={tuple(reference.shape)}")

    diff = (trainium.float() - reference.float()).abs()
    metrics = {
        "bundle": str(bundle_path),
        "compiled_path": str(pipe.compiled_path),
        "model_dir": args.model_dir,
        "reference_mode": args.reference_mode,
        "step_index": args.step_index,
        "height": pipe.shape["height"],
        "width": pipe.shape["width"],
        "text_seq_len": int(pipe.app.text_seq_len),
        "tp_degree": args.tp_degree,
        "cosine": _cosine(trainium, reference),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "trainium_mean": float(trainium.float().mean()),
        "trainium_std": float(trainium.float().std()),
        "reference_mean": float(reference.float().mean()),
        "reference_std": float(reference.float().std()),
        "trainium_shape": list(trainium.shape),
        "reference_shape": list(reference.shape),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[qwen-parity] metrics -> {metrics_path}", flush=True)
    if args.save_trainium:
        torch.save(trainium, args.save_trainium)
        print(f"[qwen-parity] trainium tensor -> {args.save_trainium}", flush=True)
    if args.save_reference:
        torch.save(reference, args.save_reference)
        print(f"[qwen-parity] reference tensor -> {args.save_reference}", flush=True)

    pass_gate = metrics["cosine"] >= args.min_cosine
    if args.max_mean_abs is not None:
        pass_gate = pass_gate and metrics["mean_abs"] <= args.max_mean_abs
    print(f"[qwen-parity] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
