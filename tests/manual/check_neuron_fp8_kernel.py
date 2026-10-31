#!/usr/bin/env python3
"""Device check of the FP8 linear kernel (``ops_impl/fp8_kernels.py``) on one NeuronCore:
numerics against the plain-torch reference and wall time against bf16 ``F.linear``, at every
Qwen-Image shard shape of the configured TP degree, eagerly and under
``torch.compile(backend="neuron", fullgraph=True)`` through the ``FP8Linear`` layer.

    NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=. python tests/manual/check_neuron_fp8_kernel.py --tp 4

Prints one line per shape and exits 1 when any relative error exceeds ``--max-rel``.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl import fp8_kernels as K
from difflet.backends.neuron.ops_impl.fp8_linear import FP8Linear
from difflet.quant import fp8 as law

#: Qwen-Image's linears at 1024x1024 (4096 image + text rows): (name, K, N) at tp=1.
QWEN_SHAPES = [
    ("attn.to_q/k/v (col)", 3072, 3072),
    ("attn.to_out (row)", 3072, 3072),
    ("img_mlp.net.0 (col)", 3072, 12288),
    ("img_mlp.net.2 (row)", 12288, 3072),
]


def shard(name: str, k: int, n: int, tp: int) -> tuple[int, int]:
    return (k // tp, n) if "row" in name else (k, n // tp)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rows", type=int, default=4096 + 1024, help="image + text tokens")
    p.add_argument("--max-rel", type=float, default=5e-3)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    dev = torch.device("neuron")
    torch.manual_seed(0)
    failed = False
    lnc = K.logical_cores()
    print(f"[fp8-kernel] lnc={lnc} rows={args.rows} tp={args.tp}", flush=True)
    for name, k_full_model, n_full_model in QWEN_SHAPES:
        k, n = shard(name, k_full_model, n_full_model, args.tp)
        x = torch.randn(args.rows, k) * 2
        w = torch.randn(n, k) * 0.02
        wq, ws = law.quantize_weight(w)
        xs = law.activation_scale(x.abs().amax().item(), margin=1.25)
        ref = K.reference(x.to(torch.bfloat16), wq, ws, xs, fp8_max=448.0)

        layer = FP8Linear(k, n, bias=False, dtype=torch.bfloat16, compute="nki")
        with torch.no_grad():
            layer.weight.copy_(wq)
            layer.weight_scale.copy_(ws)
            layer.input_scale.copy_(xs)
        layer = layer.to(dev)
        layer.prepare_kernel_(lnc=lnc)
        xb = x.to(torch.bfloat16).to(dev)
        try:
            out = layer(xb)
            torch.neuron.synchronize()
            r_eager = rel(out, ref)
            t_eager = bench(lambda: layer(xb), n=args.iters)
            compiled = torch.compile(layer, backend="neuron", dynamic=False, fullgraph=True)
            out_c = compiled(xb)
            torch.neuron.synchronize()
            r_comp = rel(out_c, ref)
            t_comp = bench(lambda: compiled(xb), n=args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"[fp8-kernel] {name} K={k} N={n}: FAIL {type(exc).__name__}: "
                  f"{str(exc)[:400]}", flush=True)
            failed = True
            continue
        wb = (wq.float() * ws).to(torch.bfloat16).to(dev)
        t_bf16 = bench(lambda: F.linear(xb, wb), n=args.iters)
        ok = max(r_eager, r_comp) <= args.max_rel and torch.isfinite(out_c.float()).all()
        failed |= not ok
        print(f"[fp8-kernel] {name} K={k} N={n} chunk={layer.kernel_n_chunk} "
              f"ktiles={layer.kernel_k_full}+{layer.kernel_k_rem}: rel eager {r_eager:.2e} "
              f"compiled {r_comp:.2e} | fp8 eager {t_eager:.3f} ms, compiled {t_comp:.3f} ms, "
              f"bf16 {t_bf16:.3f} ms ({t_bf16 / t_comp:.2f}x) {'OK' if ok else 'FAIL'}",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
