#!/usr/bin/env python3
"""One-step real-weight HunyuanVideo alignment: Trainium vs diffusers CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Parent dir containing transformer/ with real HF weights",
    )
    parser.add_argument(
        "--compiled-dir",
        required=True,
        help="Parent dir containing transformer/model.pt + neuron_config.json",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Cached DiT inputs safetensors (.meta.json sidecar required)",
    )
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--reference-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser.parse_args()


def _extract_tensor(output) -> torch.Tensor:
    if isinstance(output, dict):
        if "sample" in output:
            return output["sample"]
        return output[next(iter(output))]
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()


def _load_bundle(path: str, step_index: int):
    meta_path = Path(path + ".meta.json")
    meta = json.loads(meta_path.read_text())
    tensors = load_file(path)
    timestep = tensors["timesteps"][step_index : step_index + 1].clone()
    return meta, tensors, timestep


def run_trainium(args: argparse.Namespace, meta: dict, tensors: dict, timestep: torch.Tensor):
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        shape={"height": meta["height"], "width": meta["width"], "num_frames": meta["num_frames"]},
        text_seq_len=meta["text_seq_len"],
    )
    print(f"[align] trainium contract = {app.dit_input_contract()}")
    print(f"[align] trainium load from {args.compiled_dir} ...")
    t0 = time.time()
    app.load(args.compiled_dir, skip_warmup=True)
    print(f"[align] trainium load elapsed = {time.time() - t0:.3f}s")

    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=timestep,
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    print(f"[align] trainium forward step={args.step_index} timestep={timestep.item()} ...")
    t1 = time.time()
    output = _extract_tensor(app.forward_dit(bundle)).detach().cpu()
    print(f"[align] trainium forward elapsed = {time.time() - t1:.3f}s")
    print(
        "[align] trainium mean/std = "
        f"{output.float().mean().item():.6e} / {output.float().std().item():.6e}"
    )
    return output


def run_diffusers(args: argparse.Namespace, tensors: dict, timestep: torch.Tensor):
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel as DiffusersHunyuanVideoTransformer3DModel,
    )

    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    transformer_dir = Path(args.source_dir) / "transformer"
    print(f"[align] diffusers load from {transformer_dir} dtype={dtype} device={device} ...")
    t0 = time.time()
    model = DiffusersHunyuanVideoTransformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=dtype,
    ).eval()
    model.to(device=device, dtype=dtype)
    print(f"[align] diffusers load elapsed = {time.time() - t0:.3f}s")

    inputs = {
        "hidden_states": tensors["latents_init"].to(device=device, dtype=dtype),
        "timestep": timestep.to(device=device, dtype=dtype),
        "encoder_hidden_states": tensors["encoder_hidden_states"].to(device=device, dtype=dtype),
        "encoder_attention_mask": tensors["encoder_attention_mask"].to(device=device),
        "pooled_projections": tensors["pooled_projections"].to(device=device, dtype=dtype),
        "guidance": tensors["guidance"].to(device=device, dtype=dtype),
    }
    print("[align] diffusers forward ...")
    t1 = time.time()
    with torch.no_grad():
        output = model(**inputs, return_dict=False)[0].detach().cpu()
    print(f"[align] diffusers forward elapsed = {time.time() - t1:.3f}s")
    print(
        "[align] diffusers mean/std = "
        f"{output.float().mean().item():.6e} / {output.float().std().item():.6e}"
    )
    return output


def main() -> int:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    meta, tensors, timestep = _load_bundle(args.bundle, args.step_index)
    print(f"[align] bundle = {args.bundle}")
    print(f"[align] prompt = {meta.get('prompt')}")
    print(
        f"[align] shape = {meta['height']}x{meta['width']}x{meta['num_frames']}, "
        f"text_seq_len={meta['text_seq_len']}, step_index={args.step_index}"
    )

    trainium = run_trainium(args, meta, tensors, timestep)
    reference = run_diffusers(args, tensors, timestep)
    cosine = _cosine(trainium, reference)
    max_abs = (trainium.float() - reference.float()).abs().max().item()
    mean_abs = (trainium.float() - reference.float()).abs().mean().item()
    print(f"[align] cosine = {cosine:.9f}")
    print(f"[align] max_abs = {max_abs:.6e}")
    print(f"[align] mean_abs = {mean_abs:.6e}")
    if cosine < args.cosine_min:
        print(f"[align] FAIL cosine {cosine:.9f} < {args.cosine_min:.9f}")
        return 1
    print(f"[align] PASS cosine {cosine:.9f} >= {args.cosine_min:.9f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
