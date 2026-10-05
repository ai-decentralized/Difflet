#!/usr/bin/env python3
"""Single-core microbenchmark of one FP8 DiT linear on Trainium, at Wan 2.1's tp4 per-rank shapes.

Methods (same x, same fp8 weight, same static input scale):
  bf16        y = x @ W.T in bf16 (XLA)
  xla_static  Difflet's current static path in torch ops: (x / s_in).clamp(+-240).fp8 -> fp8 dot ->
              * (s_in * s_w) -> bf16 (XLA lowers and fuses it)
  nki_static  nkilib qkv (-> qkv_cte) with QuantizationType.STATIC: quantize + clamp in SBUF, fp8
              double-row matmul, scale (and bias) applied on chip, bf16 out; I > 4096 runs as 2 calls

Timing: each method runs REPEAT independent calls inside one graph (outputs summed) so the per-call
host / launch overhead is amortised; the reported time is graph time / REPEAT (median of RUNS).
Correctness: each fp8 method's output vs the CPU reference of the same fp8 math (cosine / SNR).

    NEURON_RT_NUM_CORES=1 python scripts/ptq_fp8_kernel_bench.py --out bench.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FP8_MAX = 240.0
# Wan 2.1 at 480x832x9 (4680 tokens), tp4 per-rank shapes: (name, S, K_in, N_out)
SHAPES = [
    ("attn.to_q/k/v (col)", 4680, 5120, 1280),
    ("ffn.net_in (col)", 4680, 5120, 3456),
    ("attn.to_out (row)", 4680, 1280, 5120),
    ("ffn.net_out (row)", 4680, 3456, 5120),
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=8)
    p.add_argument("--runs", type=int, default=7)
    p.add_argument("--methods", default="bf16,xla_static,nki_static")
    p.add_argument("--shapes", default="all", help="comma list of shape indices or 'all'")
    return p


def snr_db(ref, test):
    import torch
    ref = ref.float().flatten(); test = test.float().flatten()
    err = (test - ref).pow(2).sum().item(); sig = ref.pow(2).sum().item()
    cos = torch.nn.functional.cosine_similarity(ref, test, dim=0).item()
    return cos, (10 * math.log10(sig / err) if err > 0 else float("inf"))


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("NEURON_CC_FLAGS", "")
    if "fp8e4m3fn-as-fp8e4m3" not in os.environ["NEURON_CC_FLAGS"]:
        os.environ["NEURON_CC_FLAGS"] += " --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3'"
    import torch
    import torch_xla.core.xla_model as xm
    from nkilib.core.qkv.qkv import qkv
    from nkilib.core.utils.common_types import QuantizationType

    dev = xm.xla_device()
    methods = args.methods.split(",")
    idx = range(len(SHAPES)) if args.shapes == "all" else [int(i) for i in args.shapes.split(",")]
    results = []
    for si in idx:
        name, S, K, N = SHAPES[si]
        g = torch.Generator().manual_seed(si)
        # heavy-ish tailed activations, roughly DiT-like
        xs = [(torch.randn(1, S, K, generator=g) * (1 + 3 * (torch.rand(1, S, 1, generator=g) > 0.98))).to(torch.bfloat16)
              for _ in range(args.repeat)]
        w = (torch.randn(N, K, generator=g) / math.sqrt(K)).to(torch.bfloat16)
        bias = (torch.randn(N, generator=g) * 0.01).to(torch.bfloat16)
        s_w = (w.float().abs().max() / FP8_MAX).reshape(1)
        w_fp8 = (w.float() / s_w).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        s_in = (max(x.float().abs().max() for x in xs) * 1.25 / FP8_MAX).reshape(1)

        # CPU references of the fp8 math for xs[0]: static (calibrated per-tensor scale) and per-token
        q0 = (xs[0].float() / s_in).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s_in
        ref = (q0 @ (w_fp8.float() * s_w).t() + bias.float()).to(torch.bfloat16)
        s_tok = (xs[0].float().abs().amax(dim=-1, keepdim=True) / FP8_MAX * (1 + 2 ** -7)).clamp_min(1 / (240 * 512))
        qt = (xs[0].float() / s_tok).to(torch.float8_e4m3fn).float() * s_tok
        ref_tok = (qt @ (w_fp8.float() * s_w).t() + bias.float()).to(torch.bfloat16)
        ref_bf16 = (xs[0].float() @ w.float().t() + bias.float()).to(torch.bfloat16)

        xs_d = [x.to(dev) for x in xs]
        w_d = w.to(dev); b_d = bias.to(dev)
        wfp8_d = w_fp8.to(dev); s_in_d = s_in.to(dev); s_w_d = s_w.to(dev)
        # kernel operands: W^T [K, N] fp8, scales [1,3] / [1,1], bias [1, N]
        wT_fp8_d = w_fp8.t().contiguous().to(dev)
        n_heads_total = N // 128

        def op_bf16(x):
            return torch.nn.functional.linear(x, w_d, b_d)

        def op_xla_static(x):
            q = (x.to(torch.float32) * (1.0 / s_in_d)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            y = torch.matmul(q, wfp8_d.t())
            return ((y.to(torch.float32) * (s_in_d * s_w_d)).to(torch.bfloat16) + b_d)

        def _nki_call(x, wT, b, n):
            heads = n // 128
            kv = max(1, (heads - 1) // 2)
            qh = heads - 2 * kv
            return qkv(x, wT, bias=b.reshape(1, -1), quantization_type=QuantizationType.STATIC,
                       qkv_w_scale=s_w_d.reshape(1, 1).expand(1, 3).contiguous(),
                       qkv_in_scale=s_in_d.reshape(1, 1), d_head=128, num_q_heads=qh, num_kv_heads=kv)

        def op_nki_static(x):
            if N <= 4096:
                return _nki_call(x, wT_fp8_d, b_d, N)
            half = N // 2
            y0 = _nki_call(x, wT_fp8_d[:, :half].contiguous(), b_d[:half], half)
            y1 = _nki_call(x, wT_fp8_d[:, half:].contiguous(), b_d[half:], half)
            return torch.cat([y0, y1], dim=-1)

        from difflet.backends.trainium.nki_kernels.fp8_linear import (
            fp8_linear_static_kernel, fp8_linear_token_kernel)

        b2_d = b_d.reshape(1, -1)
        s_in2_d = s_in_d.reshape(1, 1)
        s_w2_d = s_w_d.reshape(1, 1)

        def op_own(mode):
            def run(x):
                kernel = fp8_linear_static_kernel if mode == "static" else fp8_linear_token_kernel
                y = kernel(x.reshape(S, K), wT_fp8_d, s_w2_d, s_in2_d, b2_d)
                return y.reshape(1, S, N)
            return run

        ops = {"bf16": op_bf16, "xla_static": op_xla_static, "nki_static": op_nki_static,
               "own_static": op_own("static"), "own_token": op_own("token")}
        refs = {"own_token": ref_tok}
        row = {"layer": name, "S": S, "K": K, "N": N, "flops_g": round(2 * S * K * N / 1e9, 1)}
        for m in methods:
            fn = ops[m]
            try:
                # correctness (single call)
                y = fn(xs_d[0]); xm.mark_step(); y_cpu = y.cpu()
                cos, snr = snr_db(refs.get(m, ref) if m != "bf16" else ref_bf16, y_cpu)
                cos_b, snr_b = snr_db(ref_bf16, y_cpu)
                # timing: REPEAT calls in one graph
                def graph():
                    acc = None
                    for x in xs_d:
                        out = fn(x)
                        acc = out.float().sum(dim=-1) if acc is None else acc + out.float().sum(dim=-1)
                    return acc
                for _ in range(2):
                    r = graph(); xm.mark_step(); xm.wait_device_ops()
                times = []
                for _ in range(args.runs):
                    t0 = time.perf_counter(); r = graph(); xm.mark_step(); xm.wait_device_ops()
                    times.append((time.perf_counter() - t0) * 1000.0 / args.repeat)
                row[m] = {"ms": round(statistics.median(times), 3), "ms_min": round(min(times), 3),
                          "cos_vs_ref": round(cos, 6), "snr_vs_ref": round(snr, 2),
                          "cos_vs_bf16": round(cos_b, 6), "snr_vs_bf16": round(snr_b, 2)}
            except Exception as exc:  # the failure is the finding
                row[m] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}"}
            print(f"[bench] {name:22s} {m:11s} {row[m]}", flush=True)
        results.append(row)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
