"""The Pallas matmul + reduce-scatter ring kernel, in TPU interpret mode on 4 CPU devices.

Checks values against numpy with a non-identity ring position -> shard table,
one and two (x, weight) pairs of different widths, and batch 1 and 2, with
Pallas' race detector on: it catches two receive slots instead of three. It
cannot catch a missing ``wait_send``: in interpret mode a remote copy reads its
source when it starts, so overwriting the source while the copy is in flight is
not modelled. Needs ``jax``; skipped without it (it is not a test dependency).
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
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    ring_size, batch, rows, out_features = 4, int(sys.argv[1]), 16, 512
    local_in_features = (256,) if sys.argv[2] == "one" else (256, 128)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:ring_size]), ("tp",))
    shard_of_position = (2, 0, 3, 1)  # ring position -> shard, not the identity
    rng = np.random.default_rng(0)
    sequence = ring_size * rows
    xs = [rng.standard_normal((batch, sequence, k * ring_size)).astype(np.float32)
          for k in local_in_features]
    weights = [rng.standard_normal((out_features, k * ring_size)).astype(np.float32)
               / (k * ring_size) ** 0.5 for k in local_in_features]
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    ids = put(np.arange(ring_size, dtype=np.int32), P("tp"))
    num_pairs = len(xs)

    def body(*args):
        xs, weights, ids = args[:num_pairs], args[num_pairs:2 * num_pairs], args[-1]
        chunk_shards, neighbors = kernel.reduce_scatter_tables(ids[0], shard_of_position, ring_size)
        return kernel.matmul_reduce_scatter_body(
            xs, weights, chunk_shards, neighbors, ring_size=ring_size, column_tile_limit=128,
            interpret=pltpu.InterpretParams(detect_races=True))

    f = jax.jit(jax.shard_map(body, mesh=mesh,
                              in_specs=(P(None, None, "tp"),) * num_pairs
                              + (P(None, "tp"),) * num_pairs + (P("tp"),),
                              out_specs=P(None, "tp", None), check_vma=False))
    output = np.asarray(f(*[put(x, P(None, None, "tp")) for x in xs],
                          *[put(w, P(None, "tp")) for w in weights], ids))
    full = sum(x @ w.T for x, w in zip(xs, weights))  # (batch, sequence, out_features)
    # The device at ring position p holds the rows of shard shard_of_position[p].
    expected = np.concatenate([full[:, s * rows:(s + 1) * rows] for s in shard_of_position], axis=1)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
    print("OK", batch, sys.argv[2])
''')


@pytest.mark.parametrize("batch,pairs", [(1, "one"), (2, "one"), (2, "two")])
def test_pallas_matmul_reduce_scatter_matches_numpy(batch, pairs):
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(batch), pairs], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert f"OK {batch} {pairs}" in proc.stdout


def test_reduce_scatter_unsupported_reason():
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    # Wan attention out (K 5120 / 4) and FFN down (13824 / 4) at 1170 rows per device fit.
    assert kernel.reduce_scatter_unsupported_reason([(1, 4680, 1280)], [(5120, 1280)], 2, 4) is None
    assert kernel.reduce_scatter_unsupported_reason([(1, 4680, 3456)], [(5120, 3456)], 2, 4) is None
    assert "split" in kernel.reduce_scatter_unsupported_reason([(1, 4681, 1280)], [(5120, 1280)], 2, 4)
    assert "odd" in kernel.reduce_scatter_unsupported_reason([(1, 4680, 1280)], [(5121, 1280)], 2, 4)
    assert "no column tile" in kernel.reduce_scatter_unsupported_reason(
        [(1, 4680, 1280)], [(5000, 1280)], 2, 4)
    assert "VMEM" in kernel.reduce_scatter_unsupported_reason(
        [(1, 10496, 768), (1, 10496, 3072)], [(3072, 768), (3072, 3072)], 2, 4)
