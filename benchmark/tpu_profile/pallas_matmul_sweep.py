"""How fast a plain Pallas matmul gets on one v5e chip at the SP ring's shapes.

    python -m benchmark.tpu_profile.pallas_matmul_sweep [--iters 50]

The all-gather + matmul kernel (collective_matmul_pallas) hides its
communication fully, but its per-tile matmuls run 20-45% slower than XLA's
einsum at the same FLOPs. This isolates the matmul: out (M, N) = x (M, K) @
w (K, N), x held in VMEM, w streamed in (tk, tn) tiles, fp32 accumulation over
K, R = 16 distinct weights per call (one call is ~100 us of dispatch on its own),
for M in {the ring half, a multiple of 8 above it, both halves together}
and a few tile shapes, next to XLA. No communication, one chip.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

#: name -> (M candidates, K, N)
SHAPES = {
    "wan": ((585, 592, 1170, 1184), 5120, 1280),
    "ltx": ((289, 296, 578, 592), 4096, 1024),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=16, help="matmuls (distinct weights) per call")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def timed(f, *xs):
        jax.block_until_ready(f(*xs))
        for _ in range(3):
            jax.block_until_ready(f(*xs))
        t = time.perf_counter()
        for _ in range(args.iters):
            out = f(*xs)
        jax.block_until_ready(out)
        return (time.perf_counter() - t) / args.iters / args.reps * 1e3  # per matmul

    def pallas_mm(m, k, n, tn, tk):
        nk = k // tk

        def kernel(x_ref, w_ref, o_ref, acc_ref):
            kk = pl.program_id(2)

            @pl.when(kk == 0)
            def _():
                acc_ref[...] = jnp.zeros_like(acc_ref)

            acc_ref[...] += jnp.dot(x_ref[:, pl.ds(pl.multiple_of(kk * tk, tk), tk)], w_ref[...],
                                    preferred_element_type=jnp.float32)

            @pl.when(kk == nk - 1)
            def _():
                o_ref[...] = acc_ref[...].astype(o_ref.dtype)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((args.reps, m, n), jnp.bfloat16),
            grid=(args.reps, n // tn, nk),
            in_specs=[pl.BlockSpec((m, k), lambda r, j, kk: (0, 0)),  # x resident (index constant)
                      pl.BlockSpec((pl.Squeezed(), tk, tn), lambda r, j, kk: (r, kk, j))],
            out_specs=pl.BlockSpec((pl.Squeezed(), m, tn), lambda r, j, kk: (r, 0, j)),
            scratch_shapes=[pltpu.VMEM((m, tn), jnp.float32)],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", "parallel", "arbitrary"), vmem_limit_bytes=100 * 2**20),
        )

    dev = jax.devices()[0]
    print(f"jax {jax.__version__} {dev.device_kind}, iters {args.iters}", flush=True)
    for name, (ms, k, n) in SHAPES.items():
        w = jax.device_put(jnp.ones((args.reps, k, n), jnp.bfloat16) * 0.01, dev)
        for m in ms:
            x = jax.device_put(jnp.ones((m, k), jnp.bfloat16), dev)
            flops = 2 * m * k * n
            xla = timed(jax.jit(lambda x, w: jnp.einsum("mk,rkn->rmn", x, w)), x, w)
            line = [f"{name} M {m:4d} K {k} N {n}: xla {xla * 1e3:7.1f} us "
                    f"({flops / xla / 1e9:5.1f} TF/s)"]
            for tn, tk in itertools.product((256, 512), (k, 1024, 512)):
                if n % tn or k % tk:
                    continue
                try:
                    t = timed(jax.jit(pallas_mm(m, k, n, tn, tk)), x, w)
                    line.append(f"tn{tn}/tk{tk} {t * 1e3:7.1f}")
                except Exception as exc:  # noqa: BLE001
                    line.append(f"tn{tn}/tk{tk} FAIL({type(exc).__name__})")
            print("  ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
