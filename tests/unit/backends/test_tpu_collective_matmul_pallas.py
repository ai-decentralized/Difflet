"""The Pallas all-gather + matmul ring kernel, in TPU interpret mode on 4 CPU devices.

Checks values against numpy with a non-identity ring position -> shard table,
two weights of different widths, and batch 1 and 2, with Pallas' race detector
on (two communication slots instead of three is caught by it). Needs ``jax``;
skipped without it (it is not a test dependency).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

SCRIPT = textwrap.dedent('''
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax, jax.numpy as jnp, numpy as np
    import jax._src.tpu_info as ti
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental.pallas import tpu as pltpu
    # The pipeline's tiling inference needs a TPU generation: treat the host as a v5e.
    ti.registry["cpu"] = lambda: ti.get_tpu_info_for_chip(ti.ChipVersion.TPU_V5E, 1)
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kp

    n, B, rows, K = 4, int(sys.argv[1]), 16, 256
    kn = "kn" in sys.argv[2]
    resident = "res" in sys.argv[2]
    manual = "man" in sys.argv[2]
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:n]), ("tp",))
    shard_of_pos = (2, 0, 3, 1)  # ring position -> shard, not the identity
    rng = np.random.default_rng(0)
    x_by_shard = [rng.standard_normal((B, rows, K)).astype(np.float32) for _ in range(n)]
    ws = [rng.standard_normal((N * n, K)).astype(np.float32) / K ** 0.5 for N in ((256, 384) if "mixed" in sys.argv[2] else (256, 256) if manual else (256, 384))]
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    x = put(np.concatenate([x_by_shard[shard_of_pos[p]] for p in range(n)], axis=1),
            P(None, "tp", None))
    ids = put(np.arange(n, dtype=np.int32), P("tp"))

    def body(x, ids, *ws):
        owners, nbrs = kp.ring_tables(ids[0], shard_of_pos, n)
        return kp.ag_matmul_body(x, ws, owners, nbrs, n=n, tn=128 if manual else 256,
                                 weights_kn=kn, resident=resident or manual, manual=manual,
                                 interpret=pltpu.InterpretParams(detect_races=True))

    f = jax.jit(jax.shard_map(body, mesh=mesh,
                              in_specs=(P(None, "tp", None), P("tp"))
                              + ((P(None, "tp"),) if kn else (P("tp", None),)) * 2,
                              out_specs=(P(None, None, "tp"),) * 2, check_vma=False))
    outs = f(x, ids, *[put(w.T, P(None, "tp")) if kn else put(w, P("tp", None)) for w in ws])
    full = np.concatenate(x_by_shard, axis=1)  # the sequence in shard order
    for o, w in zip(outs, ws):
        np.testing.assert_allclose(np.asarray(o), full @ w.T, rtol=2e-5, atol=2e-5)
    print("OK", B, sys.argv[2])
''')


@pytest.mark.parametrize("batch,layout", [(1, "nk"), (2, "nk"), (2, "kn"), (1, "kn-res"), (2, "nk-res"),
                                         (1, "kn-man"), (2, "nk-man"),
                                         (2, "nk-man-mixed")])
def test_pallas_ag_matmul_matches_numpy(batch, layout):
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(batch), layout], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert f"OK {batch} {layout}" in proc.stdout
