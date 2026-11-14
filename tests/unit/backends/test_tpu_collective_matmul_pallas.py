"""The Pallas all-gather + matmul ring kernel, in TPU interpret mode on 4 CPU devices.

Checks values against numpy with a non-identity ring position -> shard table,
two weights of equal or different widths, and batch 1 and 2, with Pallas' race detector
on (two ring slots instead of three is caught by it); streamed and VMEM-resident
weights (``resident_weights``); the first weight's output heads first
(``head_dims``, [B, H, S, 128]) next to a plain one. Needs ``jax``;
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
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    # rows 32: 16-row halves, so heads-first outputs take the plain 4-D path.
    ring_size, batch, rows, in_features = 4, int(sys.argv[1]), 32, 256
    widths = (256, 384) if sys.argv[2] == "mixed" else (256, 256)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:ring_size]), ("tp",))
    shard_of_position = (2, 0, 3, 1)  # ring position -> shard, not the identity
    rng = np.random.default_rng(0)
    x_by_shard = [rng.standard_normal((batch, rows, in_features)).astype(np.float32)
                  for _ in range(ring_size)]
    weights = [rng.standard_normal((width * ring_size, in_features)).astype(np.float32)
               / in_features ** 0.5 for width in widths]
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    x = put(np.concatenate([x_by_shard[shard_of_position[p]] for p in range(ring_size)], axis=1),
            P(None, "tp", None))
    ids = put(np.arange(ring_size, dtype=np.int32), P("tp"))

    def body(x, ids, *weights):
        owner_shards, neighbors = kernel.ring_tables(ids[0], shard_of_position, ring_size)
        return kernel.all_gather_matmul_body(
            x, weights, owner_shards, neighbors, ring_size=ring_size, column_tile_limit=128,
            interpret=pltpu.InterpretParams(detect_races=True),
            resident_weights=sys.argv[3] == "resident", head_dims=head_dims)

    head_dims = (128, 0) if sys.argv[4] == "heads" else None
    out_specs = ((P(None, "tp", None, None) if head_dims else P(None, None, "tp")),
                 P(None, None, "tp"))
    f = jax.jit(jax.shard_map(body, mesh=mesh,
                              in_specs=(P(None, "tp", None), P("tp")) + (P("tp", None),) * 2,
                              out_specs=out_specs, check_vma=False))
    outputs = f(x, ids, *[put(w, P("tp", None)) for w in weights])
    full = np.concatenate(x_by_shard, axis=1)  # the sequence in shard order
    for i, (output, weight) in enumerate(zip(outputs, weights)):
        want = full @ weight.T
        if head_dims and head_dims[i]:  # (B, S, H*d) -> (B, H, S, d), H over all ranks
            d = head_dims[i]
            # rank r owns columns [r*width, (r+1)*width) of the weight, i.e. heads
            # [r*width/d, (r+1)*width/d): the global head order is the column order.
            want = want.reshape(batch, want.shape[1], -1, d).transpose(0, 2, 1, 3)
        np.testing.assert_allclose(np.asarray(output), want, rtol=2e-5, atol=2e-5)
    print("OK", batch, sys.argv[2], sys.argv[3], sys.argv[4])
''')


@pytest.mark.parametrize("output_mode", ["plain", "heads"])
@pytest.mark.parametrize("weights_mode", ["streamed", "resident"])
@pytest.mark.parametrize("batch,layout", [(1, "equal"), (2, "equal"), (2, "mixed")])
def test_pallas_ag_matmul_matches_numpy(batch, layout, weights_mode, output_mode):
    proc = subprocess.run([sys.executable, "-c", SCRIPT, str(batch), layout, weights_mode,
                           output_mode], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert f"OK {batch} {layout} {weights_mode} {output_mode}" in proc.stdout


def test_unsupported_reason():
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    # Wan q/k/v and FLUX / HunyuanVideo single blocks (q|k|v 768 + mlp 3072) fit.
    assert kernel.unsupported_reason((1, 1170, 5120), [(1280, 5120)] * 3, 2) is None
    assert kernel.unsupported_reason((1, 1024, 3072), [(768, 3072)] * 3 + [(3072, 3072)], 2) is None
    assert "odd rows" in kernel.unsupported_reason((1, 1171, 5120), [(1280, 5120)], 2)
    assert "in_features" in kernel.unsupported_reason((1, 1170, 5120), [(1280, 4096)], 2)
    assert "no column tile" in kernel.unsupported_reason((1, 1170, 5120), [(1000, 5120)], 2)
    assert "VMEM" in kernel.unsupported_reason((1, 8192, 8192), [(1024, 8192)], 2)
    # FLUX's text stream keeps its weights resident (q|k|v and the FFN up); a
    # weight set that cannot sit in VMEM whole is refused (the caller streams it).
    assert kernel.unsupported_reason((1, 128, 3072), [(768, 3072)] * 3, 2,
                                     resident_weights=True) is None
    assert kernel.unsupported_reason((1, 128, 3072), [(3072, 3072)], 2,
                                     resident_weights=True) is None
    assert "VMEM" in kernel.unsupported_reason((1, 128, 8192), [(8192, 8192)], 2,
                                               resident_weights=True)
