#!/usr/bin/env python3
"""Compile and compare a masked SDPA toy graph for HunyuanVideo M3."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx


class MaskedSdpaProbe(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--masked-tail", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--work-dir", default="/tmp/nova_hunyuan_sdpa_probe")
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--mean-abs-max", type=float, default=0.005)
    parser.add_argument(
        "--compiler-args",
        default=(
            "--model-type=transformer -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.masked_tail < 0 or args.masked_tail >= args.seq_len:
        raise SystemExit("--masked-tail must be >= 0 and < --seq-len")

    torch.manual_seed(args.seed)
    shape = (args.batch_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    mask = torch.ones((args.batch_heads, 1, args.seq_len), dtype=torch.bool)
    if args.masked_tail:
        mask[:, :, -args.masked_tail :] = False

    model = MaskedSdpaProbe(scale=1.0 / math.sqrt(args.head_dim)).eval()
    with torch.no_grad():
        ref = model(q, k, v, mask).detach().cpu()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    traced = torch_neuronx.trace(
        model,
        (q, k, v, mask),
        compiler_workdir=str(work_dir / "compiler"),
        compiler_args=args.compiler_args,
    )
    compile_elapsed = time.time() - start

    with torch.no_grad():
        start = time.time()
        out = traced(q, k, v, mask).detach().cpu()
        forward_elapsed = time.time() - start

    diff = (ref.float() - out.float()).abs()
    metrics = {
        "shape": list(shape),
        "masked_tail": args.masked_tail,
        "compile_elapsed": compile_elapsed,
        "forward_elapsed": forward_elapsed,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "cosine": float(F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0)),
        "cosine_min": args.cosine_min,
        "mean_abs_max": args.mean_abs_max,
    }
    (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if metrics["cosine"] < args.cosine_min:
        raise RuntimeError(f"cosine too low: {metrics['cosine']} < {args.cosine_min}")
    if metrics["mean_abs"] > args.mean_abs_max:
        raise RuntimeError(f"mean_abs too high: {metrics['mean_abs']} > {args.mean_abs_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
