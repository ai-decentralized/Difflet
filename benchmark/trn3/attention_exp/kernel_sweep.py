"""Neuron flash attention with the softmax exp on the Vector vs the Scalar engine (trn3).

Calls ``difflet.backends.neuron.ops_impl.attention.attention`` itself, once with
``vector_exp=True`` (attention_cte's default) and once with ``vector_exp=False``, at the
per-rank shapes of the models, on one logical core; prints the median kernel time of 9 calls
and each output's relative L2 error against fp64 on 256 random query rows.

Run on a free trn3 device (one LNC=2 logical core is enough):

    NEURON_RT_VISIBLE_CORES=40 DIFFLET_BACKEND=neuron \\
        python -m torch.distributed.run --nproc_per_node 1 benchmark/trn3/attention_exp/kernel_sweep.py
"""
import statistics
import time

import torch

from difflet.backends import get_backend
from difflet.pipeline.parallel_config import DiffletParallelConfig

get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=1))

from difflet.backends.neuron.ops_impl import attention as A  # noqa: E402

D = 128
# (label, batch x heads, query length, key length, valid keys for the bounded path or None)
SHAPES = [
    ("1,024 tokens", 24, 1024, 1024, None),
    ("FLUX 1024^2 (4,608)", 12, 4608, 4608, None),
    ("Wan 480p 9 frames (4,680)", 12, 4680, 4680, None),
    ("Qwen-Image 1328^2 (~7.4k)", 12, 7424, 7424, None),
    ("16,384 tokens", 8, 16384, 16384, None),
    ("Wan 480p 81 frames (32,760)", 5, 32760, 32760, None),
    ("cross-attention 32,760 x 512", 5, 32760, 512, None),
    ("H3 768p TP4xU2, bounded (38,272)", 7, 38272, 38272, 38010),
]


def run(q, k, v, lo, hi, vector_exp):
    kw = dict(scale=D ** -0.5, tp_q=True, tp_k=True, vector_exp=vector_exp)
    if lo is not None:
        kw.update(bound_min=lo, bound_max=hi)
    return A.attention(q, k, v, **kw)


def bench(args, vector_exp):
    with torch.no_grad():
        out = run(*args, vector_exp)
        torch.neuron.synchronize()
        for _ in range(2):
            run(*args, vector_exp)
        torch.neuron.synchronize()
        times = []
        for _ in range(9):
            t = time.perf_counter()
            run(*args, vector_exp)
            torch.neuron.synchronize()
            times.append(time.perf_counter() - t)
    return out.cpu(), 1000 * statistics.median(times)


def main():
    torch.set_num_threads(16)
    dev = torch.device("neuron")
    print("| shape | B*H | Vector exp | Scalar exp | change | rel. error vs fp64 (Vector / Scalar) |")
    print("|---|---|---|---|---|---|")
    for i, (label, bh, sq, sk, live) in enumerate(SHAPES):
        g = torch.Generator().manual_seed(i)
        q = torch.randn(bh, sq, D, generator=g).to(torch.bfloat16)
        k = torch.randn(bh, sk, D, generator=g).to(torch.bfloat16)
        v = torch.randn(bh, sk, D, generator=g).to(torch.bfloat16)
        n_keys = sk if live is None else live
        lo = hi = None
        if live is not None:
            lo = torch.zeros(bh, sq, 1, dtype=torch.int32, device=dev)
            hi = torch.full((bh, sq, 1), live, dtype=torch.int32, device=dev)
        rows = torch.randperm(sq, generator=g)[:256].sort().values
        scores = q[:, rows].double() @ k[:, :n_keys].double().transpose(1, 2) * D ** -0.5
        ref = torch.softmax(scores, -1) @ v[:, :n_keys].double()
        args = (q.to(dev), k.to(dev), v.to(dev), lo, hi)
        res = {}
        for vector_exp in (True, False):
            out, ms = bench(args, vector_exp)
            o = out[:, rows].double()
            res[vector_exp] = (ms, ((o - ref).norm() / ref.norm()).item())
        (tv, ev), (ts, es) = res[True], res[False]
        print(f"| {label} | {bh} | {tv:.2f} ms | {ts:.2f} ms | {100 * (ts - tv) / tv:+.1f}% "
              f"| {ev:.3e} / {es:.3e} |", flush=True)


if __name__ == "__main__":
    main()
