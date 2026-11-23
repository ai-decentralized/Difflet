"""Collective-matmul rings vs plain collectives, compiled, one process per chip.

    python -m benchmark.tpu_profile.cm_ops_parity [--rows 578 --hidden 4096]

For each (kind, arity) in the list below, runs the ring op and its meaning with
plain collectives (gather + linear, linear + reduce-scatter) inside
``torch.compile``, on the same random bf16 input, and prints the relative L2 of
ring against plain on rank 0. Defaults are LTX-2's video shapes at 480x704x49
(2312 padded rows / 4 = 578 per rank, hidden 4096, 1024 heads-dims per rank).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

OPS = (("ag", 1), ("ag", 2), ("ag", 3), ("rs", 1))


def _worker(rank, world, args, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch
        import torch.nn.functional as F

        from difflet.backends.tpu.ops_impl import collective_matmul as cm
        from difflet.backends.tpu.ops_impl import parallel_mesh
        from difflet.ops import (
            gather_from_sequence_parallel_region,
            reduce_scatter_to_sequence_parallel_region,
        )
        from difflet.pipeline.parallel_mesh import MeshSpec

        parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))
        dev = device_runtime.device()
        cm.prepare(OPS)
        g = torch.Generator().manual_seed(100 + rank)  # every rank its own shard / weights
        bf = torch.bfloat16
        hidden, local = args.hidden, args.hidden // world

        def lin(n_out, n_in, bias=True):
            m = torch.nn.Linear(n_in, n_out, bias=bias)
            with torch.no_grad():
                m.weight.copy_(torch.randn(n_out, n_in, generator=g) / n_in ** 0.5)
                if bias:
                    m.bias.copy_(torch.randn(n_out, generator=g) * 0.1)
            return m.to(dev, bf)

        x_shard = torch.randn(1, args.rows, hidden, generator=g).to(dev, bf)
        x_full_local = torch.randn(1, args.rows * world, local, generator=g).to(dev, bf)
        cols = [lin(local, hidden) for _ in range(3)]
        row = lin(hidden, local)
        # A row-parallel bias is replicated: the same on every rank.
        bias = torch.randn(hidden, generator=torch.Generator().manual_seed(5)).to(dev, bf)
        with torch.no_grad():
            row.bias.copy_(bias)

        def ring_ag(n):
            return lambda x: cm.all_gather_matmul(x, cols[:n])

        def plain_ag(n):
            def f(x):
                full = gather_from_sequence_parallel_region(x, dim=1)
                return [F.linear(full, c.weight, c.bias) for c in cols[:n]]
            return f

        def ring_rs(x):
            return [cm.matmul_reduce_scatter(x, row)]

        def plain_rs(x):
            out = reduce_scatter_to_sequence_parallel_region(F.linear(x, row.weight), dim=1)
            return [out + row.bias]

        res = {}
        for kind, n in OPS:
            if kind == "ag":
                ring, plain, x = ring_ag(n), plain_ag(n), x_shard
            else:
                ring, plain, x = ring_rs, plain_rs, x_full_local
            with torch.no_grad():
                a = torch.compile(ring, backend="tpu")(x)
                b = torch.compile(plain, backend="tpu")(x)
            errs = []
            for u, v in zip(a, b):
                u, v = u.float().cpu(), v.float().cpu()
                errs.append(round(float((u - v).norm() / v.norm()), 5))
            res[f"{kind}{n}"] = errs
        # Twice in one compiled graph, different weights (as consecutive blocks do).
        cols2 = [lin(local, hidden) for _ in range(3)]
        row2 = lin(hidden, local)
        with torch.no_grad():
            row2.bias.copy_(bias)
        for kind, n in OPS:
            if kind == "ag":
                def ring2(x, n=n):
                    a = cm.all_gather_matmul(x, cols[:n])
                    b = cm.all_gather_matmul(x * 0.5, cols2[:n])
                    return list(a) + list(b)

                def plain2(x, n=n):
                    full = gather_from_sequence_parallel_region(x, dim=1)
                    full2 = gather_from_sequence_parallel_region(x * 0.5, dim=1)
                    return ([F.linear(full, c.weight, c.bias) for c in cols[:n]]
                            + [F.linear(full2, c.weight, c.bias) for c in cols2[:n]])
                x = x_shard
            else:
                def ring2(x):
                    return [cm.matmul_reduce_scatter(x, row), cm.matmul_reduce_scatter(x * 0.5, row2)]

                def plain2(x):
                    a = reduce_scatter_to_sequence_parallel_region(F.linear(x, row.weight), dim=1)
                    b = reduce_scatter_to_sequence_parallel_region(F.linear(x * 0.5, row2.weight), dim=1)
                    return [a + row.bias, b + row2.bias]
                x = x_full_local
            with torch.no_grad():
                a = torch.compile(ring2, backend="tpu")(x)
                b = torch.compile(plain2, backend="tpu")(x)
            res[f"{kind}{n}_twice"] = [round(float((u.float().cpu() - v.float().cpu()).norm()
                                                  / v.float().cpu().norm()), 5) for u, v in zip(a, b)]
        reply_q.put({"rank": rank, **res})
    except BaseException as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}",
                     "tb": traceback.format_exc()[-2500:]})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=578)
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()
    from difflet.backends.tpu.ops_impl import device_runtime

    world = 4
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, args, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = sorted((reply_q.get(timeout=1200) for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join(timeout=60)
    for r in results:
        print(r, flush=True)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
