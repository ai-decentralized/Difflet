"""The collective-matmul rings in plain JAX, without torch_tpu: AG -> RS blocks at a padded length.

    python -m benchmark.tpu_profile.cm_pure_jax [--seq 2312] [--blocks 2]

The SP rings misbehave on torch_tpu at 2312 rows (hang / core halt / wrong
values; 6144 is fine; 2026-11-10). This runs the *same* ring functions --
``collective_matmul._op`` builds them; ``torch.tpu.pallas.jax_op`` is stubbed
to hand back the JAX function itself -- under ``jax.jit`` in one process that
owns all four chips, no torch_tpu, and compares against an fp32 numpy
reference. A block is ``x += RS(AG(x) @ w1^T, w2)``, the ``rsonly`` variant of
cm_padded_repro. Right here and wrong there puts the bug in the torch_tpu
jax_op path; wrong here puts it in the ring code or XLA.
"""

from __future__ import annotations

import argparse
import sys
import types

HID, FF = 4096, 8192


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=2312)
    parser.add_argument("--blocks", type=int, default=2)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    # Stub the torch_tpu entry point: _op() then returns the raw JAX function.
    torch.tpu = types.SimpleNamespace(pallas=types.SimpleNamespace(
        jax_op=lambda name, fn, **kw: fn))
    from difflet.backends.tpu.ops_impl import collective_matmul as cm

    devices = jax.devices()
    tp = len(devices)
    mesh = jax.sharding.Mesh(np.array(cm._snake(devices)), ("tp",))
    # One process: the shard of ring position p is simply shard p.
    cm._STATE = {"mesh": mesh, "tp": tp, "shard_of_pos": tuple(range(tp)), "ids": None, "ops": {}}
    ag = cm._op("ag", 1)
    rs = cm._op("rs", 1)

    seq = args.seq
    if seq % (2 * tp):
        raise SystemExit(f"--seq must divide by 2*tp={2 * tp}")
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, seq, HID), dtype=np.float32)
    w1s = [rng.standard_normal((FF, HID), dtype=np.float32) / HID ** 0.5 for _ in range(args.blocks)]
    w2s = [rng.standard_normal((HID, FF), dtype=np.float32) / FF ** 0.5 for _ in range(args.blocks)]

    def put(a, spec, dtype=jnp.bfloat16):
        return jax.device_put(jnp.asarray(a, dtype), NamedSharding(mesh, spec))

    ids = put(np.arange(tp, dtype=np.int32), P("tp"), jnp.int32)
    x = put(x0, P(None, "tp", None))
    w1 = [put(w, P("tp", None)) for w in w1s]
    w2 = [put(w, P(None, "tp")) for w in w2s]

    @jax.jit
    def model(x, ids, w1, w2):
        for a, b in zip(w1, w2):
            h = ag(x, ids, a)  # (1, S, FF) sharded on FF
            x = x + rs(h, b, ids)  # (1, S, HID) sharded on S
        return x

    print(f"jax {jax.__version__} devices {[d.id for d in mesh.devices.flat]} seq {seq} "
          f"rows/rank {seq // tp} half {seq // tp // 2} blocks {args.blocks}", flush=True)
    got = np.asarray(model(x, ids, w1, w2).astype(jnp.float32))

    ref = x0.astype(np.float64)
    for a, b in zip(w1s, w2s):
        # The device rounds the operands to bf16; so does the reference.
        bf = lambda t: np.asarray(jnp.asarray(t, jnp.bfloat16).astype(jnp.float32), np.float64)  # noqa: E731
        ref = ref + (bf(ref) @ bf(a).T) @ bf(b).T
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    rows = np.linalg.norm(got - ref, axis=-1)[0] / np.maximum(np.linalg.norm(ref, axis=-1)[0], 1e-6)
    worst = np.argsort(rows)[-6:][::-1]
    print(f"RESULT seq {seq} blocks {args.blocks}: rel {err:.5f} finite {bool(np.isfinite(got).all())} "
          f"worst rows {worst.tolist()} {np.round(rows[worst], 3).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
