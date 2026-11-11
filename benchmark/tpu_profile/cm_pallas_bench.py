"""All-gather + matmul on the chips: JAX ring (dus / concat) vs the Pallas kernel.

    python -m benchmark.tpu_profile.cm_pallas_bench [--iters 50] [--shapes wan_qkv,ltx_ffn]

Plain JAX, one process owning the four chips (no torch_tpu): the JAX rings are
``collective_matmul._op`` with ``torch.tpu.pallas.jax_op`` stubbed out (as in
cm_pure_jax), the kernel is ``collective_matmul_pallas.ag_matmul_body``. For
each shape, prints the time per call and the kernel's difference from the dus
ring (bf16 inputs, fp32 accumulation in both). Step 2 of
docs/plans/2026-11-10-tpu-collective-matmul-pallas.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

#: name -> (rows per device, K, output columns per device of each weight)
SHAPES = {
    "wan_qkv": (1170, 5120, (1280, 1280, 1280)),
    "wan_ffn": (1170, 5120, (3456,)),
    "ltx_qkv": (578, 4096, (1024, 1024, 1024)),
    "ltx_ffn": (578, 4096, (4096,)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--tn", type=int, default=512)
    parser.add_argument("--ablate", action="store_true",
                        help="also time the kernel without comm / without compute, and one-chip XLA einsums")
    parser.add_argument("--trace", default="", help="jax.profiler trace dir for the kernel calls")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    torch.tpu = types.SimpleNamespace(pallas=types.SimpleNamespace(
        jax_op=lambda name, fn, **kw: fn))
    from difflet.backends.tpu.ops_impl import collective_matmul as cm
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kp

    devices = jax.devices()
    n = len(devices)
    mesh = jax.sharding.Mesh(np.array(cm._snake(devices)), ("tp",))
    shard_of_pos = tuple(range(n))  # one process: ring position p holds shard p
    cm._STATE = {"mesh": mesh, "tp": n, "shard_of_pos": shard_of_pos, "ids": None, "ops": {}}

    def put(a, spec, dtype=jnp.bfloat16):
        return jax.device_put(jnp.asarray(a, dtype), NamedSharding(mesh, spec))

    def timed(f, *xs):
        out = f(*xs)
        jax.block_until_ready(out)
        for _ in range(3):
            jax.block_until_ready(f(*xs))
        t = time.perf_counter()
        for _ in range(args.iters):
            out = f(*xs)
        jax.block_until_ready(out)
        return out, (time.perf_counter() - t) / args.iters * 1e3

    print(f"jax {jax.__version__} {devices[0].device_kind} x{n}, iters {args.iters}, tn<={args.tn}",
          flush=True)
    ids = put(np.arange(n), P("tp"), jnp.int32)
    rng = np.random.default_rng(0)
    for name in args.shapes.split(","):
        rows, k, cols = SHAPES[name]
        x = put(rng.standard_normal((1, rows * n, k), dtype=np.float32), P(None, "tp", None))
        ws = [put(rng.standard_normal((c * n, k), dtype=np.float32) / k ** 0.5, P("tp", None))
              for c in cols]
        res = {}
        for mode in ("dus", "concat"):
            os.environ[cm.AG_ASSEMBLE_ENV] = mode
            ring = jax.jit(cm._op("ag", len(ws)))
            out, ms = timed(ring, x, ids, *ws)
            res[mode] = (out if isinstance(out, tuple) else (out,), ms)

        def body(x, ids, *ws):
            owners, nbrs = kp.ring_tables(ids[0], shard_of_pos, n)
            return kp.ag_matmul_body(x, ws, owners, nbrs, n=n, tn=args.tn)

        kern = jax.jit(jax.shard_map(
            body, mesh=mesh, in_specs=(P(None, "tp", None), P("tp")) + (P("tp", None),) * len(ws),
            out_specs=(P(None, None, "tp"),) * len(ws), check_vma=False))
        try:
            res["pallas"] = timed(kern, x, ids, *ws)
            if args.trace:
                with jax.profiler.trace(f"{args.trace}/{name}"):
                    for _ in range(5):
                        jax.block_until_ready(kern(x, ids, *ws))
        except Exception as exc:  # noqa: BLE001 -- report and go on to the next shape
            print(f"{name}: pallas failed: {type(exc).__name__}: {str(exc)[:600]}", flush=True)
            continue
        if args.ablate:
            extra = []
            for ab in ("comm", "compute"):
                def body_ab(x, ids, *ws, ab=ab):
                    owners, nbrs = kp.ring_tables(ids[0], shard_of_pos, n)
                    return kp.ag_matmul_body(x, ws, owners, nbrs, n=n, tn=args.tn, ablate=ab)
                f = jax.jit(jax.shard_map(
                    body_ab, mesh=mesh,
                    in_specs=(P(None, "tp", None), P("tp")) + (P("tp", None),) * len(ws),
                    out_specs=(P(None, None, "tp"),) * len(ws), check_vma=False))
                extra.append(f"no-{ab} {timed(f, x, ids, *ws)[1]:.3f}")
            # The same FLOPs on one chip, plain XLA: the full sequence times each local weight.
            d0 = devices[0]
            xf = jax.device_put(jnp.zeros((1, rows * n, k), jnp.bfloat16), d0)
            wl = [jax.device_put(jnp.zeros((c, k), jnp.bfloat16), d0) for c in cols]
            ein = jax.jit(lambda x, *ws: tuple(jnp.einsum("bsk,nk->bsn", x, w) for w in ws))
            extra.append(f"xla-1chip {timed(ein, xf, *wl)[1]:.3f}")
            print(f"{name:8s} ablation (ms): " + ", ".join(extra), flush=True)
        # v2: weights pre-transposed to (K, N) once, outside the timed call.
        wts = [jax.device_put(w.T, NamedSharding(mesh, P(None, "tp"))) for w in ws]

        def body_kn(x, ids, *ws):
            owners, nbrs = kp.ring_tables(ids[0], shard_of_pos, n)
            return kp.ag_matmul_body(x, ws, owners, nbrs, n=n, tn=args.tn, weights_kn=True)

        kern_kn = jax.jit(jax.shard_map(
            body_kn, mesh=mesh, in_specs=(P(None, "tp", None), P("tp")) + (P(None, "tp"),) * len(ws),
            out_specs=(P(None, None, "tp"),) * len(ws), check_vma=False))
        res["pallas-kn"] = timed(kern_kn, x, ids, *wts)

        # v3: halves resident in VMEM, remote DMA VMEM -> VMEM, pipelines stream weights only.
        def body_res(x, ids, *ws, ab="", manual=False):
            owners, nbrs = kp.ring_tables(ids[0], shard_of_pos, n)
            return kp.ag_matmul_body(x, ws, owners, nbrs, n=n, tn=args.tn, weights_kn=True,
                                     resident=True, manual=manual, ablate=ab)

        def mk(ab="", manual=False):
            return jax.jit(jax.shard_map(
                lambda x, ids, *ws: body_res(x, ids, *ws, ab=ab, manual=manual), mesh=mesh,
                in_specs=(P(None, "tp", None), P("tp")) + (P(None, "tp"),) * len(ws),
                out_specs=(P(None, None, "tp"),) * len(ws), check_vma=False))
        try:
            res["pallas-res"] = timed(mk(), x, ids, *wts)
            if args.ablate:
                print(f"{name:8s} resident ablation (ms): no-comm {timed(mk('comm'), x, ids, *wts)[1]:.3f}",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: pallas-res failed: {type(exc).__name__}: {str(exc)[:800]}", flush=True)
        # v4: one hand-written weight-tile stream over every (hop, weight, tile).
        try:
            res["pallas-man"] = timed(mk(manual=True), x, ids, *wts)
            if args.ablate:
                print(f"{name:8s} manual ablation (ms): no-comm "
                      f"{timed(mk('comm', manual=True), x, ids, *wts)[1]:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: pallas-man failed: {type(exc).__name__}: {str(exc)[:800]}", flush=True)
        ref = [np.asarray(o.astype(jnp.float32)) for o in res["dus"][0]]
        line = [f"{name:8s} rows {rows} K {k} N {cols}:"]
        for mode in [m for m in ("dus", "concat", "pallas", "pallas-kn", "pallas-res", "pallas-man") if m in res]:
            outs, ms = res[mode]
            err = max(float(np.abs(np.asarray(o.astype(jnp.float32)) - r).max() / np.abs(r).max())
                      for o, r in zip(outs, ref))
            line.append(f"{mode} {ms:.3f} ms (max|d|/max {err:.1e})")
        print("  ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
