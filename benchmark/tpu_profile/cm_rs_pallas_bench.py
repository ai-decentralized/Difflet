"""Matmul + reduce-scatter on the chips: the JAX ring vs the Pallas kernel.

    python -m benchmark.tpu_profile.cm_rs_pallas_bench [--iters 50] [--shapes wan_attn_out,...]

Plain JAX, one process owning the four chips (no torch_tpu), as cm_pallas_bench:
both are ``collective_matmul._op("rs", pairs)`` with ``torch.tpu.pallas.jax_op``
stubbed out, built once with ``DIFFLET_TPU_SP_RS_KERNEL=jax`` and once with
``pallas``, so the kernel runs through the same entry as in the models. For each
shape, prints the time per call and the kernel's largest difference from the
ring relative to the ring's largest value (bf16 inputs).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

#: name -> (rows per device, this device's in_features of each pair, out_features)
SHAPES = {
    "wan_attn_out": (1170, (1280,), 5120),
    "wan_ffn_down": (1170, (3456,), 5120),
    # Wan's FFN down with rows a multiple of 8: no relayout of the input before the kernel.
    "wan_ffn_down_aligned": (1168, (3456,), 5120),
    "ltx_attn_out": (578, (1024,), 4096),
    "flux_single": (1152, (768, 3072), 3072),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--ablate", action="store_true",
                        help="also time the kernel without communication / without compute")
    parser.add_argument("--column-tile-limits", default="",
                        help="comma-separated: also time the kernel at these column tile limits")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    # No torch_tpu here: jax_op returns the JAX function itself. is_initialized is
    # asked at interpreter exit.
    torch.tpu = types.SimpleNamespace(is_initialized=lambda: False,
                                      pallas=types.SimpleNamespace(jax_op=lambda name, fn, **kw: fn))
    from difflet.backends.tpu.ops_impl import collective_matmul as cm
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    devices = jax.devices()
    n = len(devices)
    mesh = jax.sharding.Mesh(np.array(cm._snake(devices)), ("tp",))
    cm._STATE = {"mesh": mesh, "tp": n, "shard_of_pos": tuple(range(n)), "ids": None, "ops": {}}

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

    print(f"jax {jax.__version__} {devices[0].device_kind} x{n}, iters {args.iters}", flush=True)
    ids = put(np.arange(n), P("tp"), jnp.int32)
    rng = np.random.default_rng(0)
    for name in args.shapes.split(","):
        rows, local_in_features, out_features = SHAPES[name]
        operands = []
        for k in local_in_features:
            operands.append(put(rng.standard_normal((1, rows * n, k * n), dtype=np.float32),
                                P(None, None, "tp")))
            operands.append(put(rng.standard_normal((out_features, k * n), dtype=np.float32)
                                / (k * n) ** 0.5, P(None, "tp")))
        results = {}
        for mode in ("jax", "pallas"):
            os.environ[cm.SP_RS_KERNEL_ENV] = mode
            try:
                op = jax.jit(cm._op("rs", len(local_in_features)))
                results[mode] = timed(op, *operands, ids)
            except Exception as exc:  # noqa: BLE001 -- report and go on to the next shape
                print(f"{name}: {mode} failed: {type(exc).__name__}: {str(exc)[:800]}", flush=True)
        if len(results) < 2:
            continue
        ring = np.asarray(results["jax"][0].astype(jnp.float32))
        kern = np.asarray(results["pallas"][0].astype(jnp.float32))
        diff = float(np.abs(kern - ring).max() / np.abs(ring).max())
        if args.ablate:
            pairs = len(local_in_features)

            def ablated(ablation, pairs=pairs, column_tile_limit=kernel.DEFAULT_COLUMN_TILE_LIMIT):
                def body(*args):
                    xs, ws, ids = args[0:2 * pairs:2], args[1:2 * pairs:2], args[-1]
                    chunk_shards, neighbors = kernel.reduce_scatter_tables(ids[0], tuple(range(n)), n)
                    return kernel.matmul_reduce_scatter_body(
                        xs, ws, chunk_shards, neighbors, ring_size=n, profile_ablation=ablation,
                        column_tile_limit=column_tile_limit)
                return jax.jit(jax.shard_map(
                    body, mesh=mesh, in_specs=(P(None, None, "tp"), P(None, "tp")) * pairs + (P("tp"),),
                    out_specs=P(None, "tp", None), check_vma=False))

            print(f"{name:13s} ablation (ms): " + ", ".join(
                f"no-{a} {timed(ablated(a), *operands, ids)[1]:.3f}"
                for a in ("communication", "compute")), flush=True)
            for limit in [int(v) for v in args.column_tile_limits.split(",") if v]:
                print(f"{name:13s} column tile <= {limit}: "
                      f"{timed(ablated('', column_tile_limit=limit), *operands, ids)[1]:.3f} ms",
                      flush=True)
        print(f"{name:13s} rows {rows} K/dev {local_in_features} N {out_features}: "
              f"ring {results['jax'][1]:.3f} ms  kernel {results['pallas'][1]:.3f} ms  "
              f"(max|d|/max {diff:.1e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
