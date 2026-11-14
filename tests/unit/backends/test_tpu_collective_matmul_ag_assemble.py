"""The all-gather ring's two output assemblies give the same values (CPU, 4 fake devices).

``collective_matmul._op`` builds the JAX ring; ``torch.tpu.pallas.jax_op`` is
stubbed to return it, and the mesh is four XLA host devices. The ring position
-> TP rank table is a non-identity permutation, so the runtime reordering of
``DIFFLET_TPU_SP_AG_ASSEMBLE=concat`` is exercised. Needs ``jax``; skipped
without it (it is not a test dependency).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

SCRIPT = textwrap.dedent('''
    import os, sys, types
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["DIFFLET_TPU_SP_AG_ASSEMBLE"] = sys.argv[1]
    import jax, jax.numpy as jnp, numpy as np, torch
    from jax.sharding import NamedSharding, PartitionSpec as P
    torch.tpu = types.SimpleNamespace(pallas=types.SimpleNamespace(jax_op=lambda name, fn, **kw: fn))
    from difflet.backends.tpu.ops_impl import collective_matmul as cm

    n = 4
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:n]), ("tp",))
    shard_of_pos = (2, 0, 3, 1)  # ring position -> TP rank, not the identity
    cm._STATE = {"mesh": mesh, "tp": n, "shard_of_pos": shard_of_pos, "ids": None, "ops": {}}
    ag = cm._op("ag", 2)

    rows, K, N = 6, 8, 4  # rows per rank (even: two ring halves)
    rng = np.random.default_rng(0)
    # The model's sequence shard of TP rank r sits at ring position pos_of[r].
    x_by_rank = [rng.standard_normal((1, rows, K)).astype(np.float32) for _ in range(n)]
    pos_of = [shard_of_pos.index(r) for r in range(n)]
    x_by_pos = [x_by_rank[shard_of_pos[p]] for p in range(n)]
    w0 = rng.standard_normal((N * n, K)).astype(np.float32)
    w1 = rng.standard_normal((N * n, K)).astype(np.float32)

    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    x = put(np.concatenate(x_by_pos, axis=1), P(None, "tp", None))
    ids = put(np.arange(n, dtype=np.int32), P("tp"))
    o0, o1 = jax.jit(ag)(x, ids, put(w0, P("tp", None)), put(w1, P("tp", None)))

    full = np.concatenate(x_by_rank, axis=1)  # the sequence in TP-rank order
    for o, w in ((o0, w0), (o1, w1)):
        got = np.asarray(o)
        want = full @ w.T  # each device holds its N columns; the global array is (1, S, N*n)
        assert got.shape == want.shape, (got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    print("OK", sys.argv[1])
''')


@pytest.mark.parametrize("mode", ["dus", "concat"])
def test_ag_ring_assembly_matches_numpy(mode):
    proc = subprocess.run([sys.executable, "-c", SCRIPT, mode], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert f"OK {mode}" in proc.stdout
