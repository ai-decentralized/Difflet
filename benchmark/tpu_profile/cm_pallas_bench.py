"""All-gather + matmul on the chips: JAX ring (dus / concat) vs the Pallas kernel.

    python -m benchmark.tpu_profile.cm_pallas_bench [--iters 50] [--shapes wan_qkv,ltx_ffn]

Plain JAX, one process owning the four chips (no torch_tpu): the JAX rings are
``collective_matmul._op`` with ``torch.tpu.pallas.jax_op`` stubbed out (as in
cm_pure_jax), the kernel is ``collective_matmul_pallas.all_gather_matmul_body``. For
each shape, prints the time per call and the kernel's difference from the dus
ring (bf16 inputs, fp32 accumulation in both); "resident" is the kernel with
``resident_weights=True`` (weights loaded into VMEM once, not once per hop),
where they fit. Step 2 of
docs/plans/2026-11-10-tpu-collective-matmul-pallas.md.

The kernel is timed as the models run it: weights (N, K), column tiles up to
``DEFAULT_COLUMN_TILE_LIMIT``. The plan's op-level version table was taken at
commit 01d027fe with pre-transposed (K, N) weights and tiles up to 512.
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
    # LTX-2's narrow video-stream gathers (2026-11-13 trace: 51-100 TF/s in the
    # kernel): output widths 512 (one / two weights) and 1024 (one weight).
    "ltx_n512": (578, 4096, (512,)),
    "ltx_n512x2": (578, 4096, (512, 512)),
    "ltx_n1024": (578, 4096, (1024,)),
    # FLUX.1-dev at 1024x1024 (512 text + 4096 image tokens, tp=4).
    "flux_txt_qkv": (128, 3072, (768, 768, 768)),
    "flux_txt_ffn": (128, 3072, (3072,)),
    "flux_img_qkv": (1024, 3072, (768, 768, 768)),
    "flux_single": (1152, 3072, (768, 768, 768, 3072)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--column-tile-limit", type=int, default=None,
                        help="the kernel's column tile limit (default: the production value)")
    parser.add_argument("--ablate", action="store_true",
                        help="also time the kernel without comm / without compute, and one-chip XLA einsums")
    parser.add_argument("--trace", default="", help="jax.profiler trace dir for the kernel calls")
    parser.add_argument("--heads", type=int, default=0,
                        help="also run the kernel writing its outputs heads first (head size)")
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

    if args.column_tile_limit is None:
        args.column_tile_limit = kernel.DEFAULT_COLUMN_TILE_LIMIT

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

    print(f"jax {jax.__version__} {devices[0].device_kind} x{n}, iters {args.iters}, column tile <= {args.column_tile_limit}",
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

        def kernel_call(ablation="", resident=False, heads=0):
            head_dims = (heads,) * len(ws) if heads else None

            def body(x, ids, *ws):
                owner_shards, neighbors = kernel.ring_tables(ids[0], shard_of_pos, n)
                return kernel.all_gather_matmul_body(
                    x, ws, owner_shards, neighbors, ring_size=n,
                    column_tile_limit=args.column_tile_limit, profile_ablation=ablation,
                    resident_weights=resident, head_dims=head_dims)
            spec = P(None, "tp", None, None) if heads else P(None, None, "tp")
            return jax.jit(jax.shard_map(
                body, mesh=mesh, in_specs=(P(None, "tp", None), P("tp")) + (P("tp", None),) * len(ws),
                out_specs=(spec,) * len(ws), check_vma=False))

        try:
            res["pallas"] = timed(kernel_call(), x, ids, *ws)
            if kernel.unsupported_reason((1, rows, k), [(c, k) for c in cols], 2,
                                         resident_weights=True) is None:
                res["resident"] = timed(kernel_call(resident=True), x, ids, *ws)
            if args.heads and all(c % args.heads == 0 for c in cols):
                outs, ms = timed(kernel_call(heads=args.heads), x, ids, *ws)
                outs = outs if isinstance(outs, tuple) else (outs,)
                # (B, H, S, d) back to (B, S, H*d), to compare with the others
                res["heads"] = (tuple(o.transpose(0, 2, 1, 3).reshape(o.shape[0], o.shape[2], -1)
                                      for o in outs), ms)
            if args.trace:
                variants = [("", False)] + ([("-resident", True)] if "resident" in res else [])
                for suffix, resident in variants:
                    call = kernel_call(resident=resident)
                    with jax.profiler.trace(f"{args.trace}/{name}{suffix}"):
                        for _ in range(5):
                            jax.block_until_ready(call(x, ids, *ws))
        except Exception as exc:  # noqa: BLE001 -- report and go on to the next shape
            print(f"{name}: pallas failed: {type(exc).__name__}: {str(exc)[:600]}", flush=True)
            continue
        if args.ablate:
            extra = [f"no-{ablation} {timed(kernel_call(ablation), x, ids, *ws)[1]:.3f}"
                     for ablation in ("communication", "compute")]
            # The same FLOPs on one chip, plain XLA: the full sequence times each local weight.
            d0 = devices[0]
            xf = jax.device_put(jnp.zeros((1, rows * n, k), jnp.bfloat16), d0)
            wl = [jax.device_put(jnp.zeros((c, k), jnp.bfloat16), d0) for c in cols]
            ein = jax.jit(lambda x, *ws: tuple(jnp.einsum("bsk,nk->bsn", x, w) for w in ws))
            extra.append(f"xla-1chip {timed(ein, xf, *wl)[1]:.3f}")
            # The same, in the kernel's matmul sizes: one dot per (hop, direction) on
            # rows/2 rows, or per hop on both directions' rows -- is the gap to one big
            # dot the rows per dot? Unrolled static slices, every chunk's output kept
            # (a lax.map version measured its loop's own overhead, 2026-11-13).
            for chunks, label in ((2 * n, "half"), (n, "hop")):
                xc = xf.reshape(chunks, rows * n // chunks, k)
                per = jax.jit(lambda xc, *ws: tuple(
                    jnp.einsum("sk,nk->sn", xc[c], w) for w in ws for c in range(xc.shape[0])))
                extra.append(f"xla-1chip-{label}({rows * n // chunks}x{chunks}) "
                             f"{timed(per, xc, *wl)[1]:.3f}")
            print(f"{name:8s} ablation (ms): " + ", ".join(extra), flush=True)
        ref = [np.asarray(o.astype(jnp.float32)) for o in res["dus"][0]]
        line = [f"{name:8s} rows {rows} K {k} N {cols}:"]
        for mode in [m for m in ("dus", "concat", "pallas", "resident", "heads") if m in res]:
            outs, ms = res[mode]
            err = max(float(np.abs(np.asarray(o.astype(jnp.float32)) - r).max() / np.abs(r).max())
                      for o, r in zip(outs, ref))
            line.append(f"{mode} {ms:.3f} ms (max|d|/max {err:.1e})")
        print("  ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
