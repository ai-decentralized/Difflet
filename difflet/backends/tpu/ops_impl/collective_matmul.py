"""Collective matmuls for Megatron sequence parallelism on TPU: the reduce-scatter
fused into a row-parallel matmul, the all-gather into a column-parallel one.

Under tensor parallelism every row-parallel linear ends in an all-reduce that
TorchTPU runs strictly after the matmul: 2.5 ms of a compiled Wan 14B block's
8.8 ms at tp=4 on v5e. Megatron-SP splits that all-reduce into a reduce-scatter
(paired with the row-parallel matmul) and an all-gather (paired with the next
column-parallel matmul); what runs between them -- residual, norm, adaLN -- is
per token, so it runs on this chip's 1/tp of the sequence. Each half here is a
bidirectional ring over the chips' physical 2x2 snake: half of the columns (RS)
or half of the rows (AG) travel clockwise, half counter-clockwise, and each
hop's matmul runs while the next hop is in flight.

Measured on Wan's three row->col pairs per block (v5e-4, compiled, S=4680,
benchmark/tpu_profile/sp_pair_probe.py, 2026-11-03): 6.33 ms -> 4.95 ms per
block, about 55% of the all-reduce hidden. ICI moves 46.7 GB/s per link and
direction (alpha ~2.5 us), so a hop of Wan's 6 MB chunk is ~131 us of transfer.

The rings are JAX programs run through ``torch.tpu.pallas.jax_op`` (every
torch_tpu worker's JAX sees all chips). That is cheap only inside
``torch.compile``: eager re-exports the program on every call (~100x slower),
so :func:`enabled` requires ``DIFFLET_TPU_COMPILE``. Two torch_tpu 0.2.0 traps
are handled here: output partition specs go in a tuple (a bare PartitionSpec
is iterated, giving the output spec ``"tp"`` and a ``KeyError 't'``), and a
compiled region must not receive a view at a nonzero storage offset (the
J5ag ring then hangs before compiling; see sp_pair_probe).

**Shard order.** The model's sequence shard for TP rank ``r`` is rows
``[r*S/tp, (r+1)*S/tp)`` (``scatter_tp_dim``). The ring runs in physical
order, which is not torch rank order (rank 0 may hold TPU_2), so every chunk
index below goes through ``_SHARD_OF_POS``: ring position -> TP rank.
"""

from __future__ import annotations

import os

import torch

SP_ENV = "DIFFLET_TPU_SP"

_STATE: dict | None = None


def requested() -> bool:
    """``DIFFLET_TPU_SP=1``: run Wan sequence-parallel with collective matmuls."""
    return os.environ.get(SP_ENV, "").strip().lower() in ("1", "true", "yes")


def require_compiled(*, default: bool) -> None:
    """Refuse ``DIFFLET_TPU_SP=1`` on a model that will not be compiled.

    ``default`` is the model's ``maybe_compile`` default; ``DIFFLET_TPU_COMPILE``
    overrides it the same way. Through jax_op, eager re-exports every ring on every
    call (~100x slower), so the fused path is compiled-only."""
    from difflet.backends.tpu.ops_impl import device_runtime

    raw = os.environ.get(device_runtime.COMPILE_ENV, "").strip().lower()
    compiled = raw in ("1", "true", "yes") or (default and raw not in ("0", "false", "no"))
    if not compiled:
        raise ValueError(
            f"{SP_ENV}=1 needs the compiled path: set {device_runtime.COMPILE_ENV}=1"
            + ("" if default else " (this model is eager by default)"))


def _snake(devices):
    """Devices in ring order on their physical grid: rows alternate direction,
    so every hop is between neighbours (a diagonal hop waits on two links)."""
    def key(d):
        x, y = d.coords[0], d.coords[1]
        return (y, x if y % 2 == 0 else -x)

    return sorted(devices, key=key)


def prepare() -> None:
    """Build the mesh, the shard table and the jax_op ops, outside any compiled
    region (traced lazily, Dynamo would graph-break on the registration).
    Collective: every TP rank must call it, in step."""
    global _STATE
    if _STATE is not None:
        return
    import jax
    import numpy as np

    from difflet.backends.tpu.ops_impl import device_runtime, parallel_mesh

    tp = parallel_mesh.get_tp_size()
    devices = jax.devices()
    if tp != len(devices):
        raise NotImplementedError(
            f"{SP_ENV}=1 needs the TP group to span every chip JAX sees "
            f"(tp={tp}, {len(devices)} devices)"
        )
    mesh = jax.sharding.Mesh(np.array(_snake(devices)), ("tp",))
    pos = list(mesh.devices.flat).index(jax.local_devices()[0])
    # Which TP rank sits at each ring position: gather every rank's position.
    dev = device_runtime.device()
    mine = torch.tensor([pos], dtype=torch.int32, device=dev)
    groups = parallel_mesh.get_tp_groups()
    positions = device_runtime.all_gather(mine, dim=0, groups=groups).cpu().tolist()
    shard_of_pos = [0] * tp
    for tp_rank, p in enumerate(positions):
        shard_of_pos[p] = tp_rank
    _STATE = {
        "mesh": mesh,
        "tp": tp,
        "shard_of_pos": tuple(shard_of_pos),
        # This chip's ring position, as data: axis_index lowers to partitionId,
        # which jax_op's partitioner rejects.
        "ids": torch.tensor([pos], dtype=torch.int32, device=dev),
        "ops": {},
    }
    # Every op the Wan forwards use, built now: built lazily they would be
    # built inside Dynamo's trace, which cannot run JAX.
    for kind, n_weights in (("rs", 1), ("ag", 1), ("ag", 3)):
        _op(kind, n_weights)


def _op(kind: str, n_weights: int):
    """The jax_op for ``kind`` ('rs' or 'ag') over ``n_weights`` weights, built once."""
    key = (kind, n_weights)
    ops = _STATE["ops"]
    if key in ops:
        return ops[key]
    if torch.compiler.is_compiling():
        raise RuntimeError(f"collective_matmul op {key} was not built by prepare()")

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    mesh, n = _STATE["mesh"], _STATE["tp"]
    shard_table = list(_STATE["shard_of_pos"])
    cw = [(i, (i + 1) % n) for i in range(n)]
    ccw = [(i, (i - 1) % n) for i in range(n)]
    shard_map = jax.shard_map if hasattr(jax, "shard_map") else \
        __import__("jax.experimental.shard_map", fromlist=["shard_map"]).shard_map

    if kind == "rs":
        # x (B, S, K/tp) @ w (N, K/tp)^T, summed over the group, this rank's rows.
        def body(x, w, ids):
            shard_of = jnp.asarray(shard_table, dtype=jnp.int32)
            j = ids[0]
            rows = x.shape[1] // n
            half = w.shape[0] // 2
            wl, wr = w[:half], w[half:]

            def chunk(pos):
                return jax.lax.dynamic_slice_in_dim(x, shard_of[pos % n] * rows, rows, axis=1)

            left = right = None
            for step in range(n):
                pl = jnp.einsum("bsk,nk->bsn", chunk(j - 1 - step), wl)
                pr = jnp.einsum("bsk,nk->bsn", chunk(j + 1 + step), wr)
                # Fence each partial from the add that consumes the received
                # accumulator: fused, the matmul would wait for the hop instead
                # of running under it (88 us idle per hop on attn_out).
                pl, pr = jax.lax.optimization_barrier((pl, pr))
                if left is None:
                    left, right = pl, pr
                else:
                    left = jax.lax.ppermute(left, "tp", cw) + pl
                    right = jax.lax.ppermute(right, "tp", ccw) + pr
            return jnp.concatenate([left, right], axis=2)

        in_specs = (P(None, None, "tp"), P(None, "tp"), P("tp"))
        out_specs = (P(None, "tp", None),)

        def fn(x, w, ids):
            return shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs[0],
                             check_vma=False)(x, w, ids)

        fn.__annotations__ = {"x": jax.Array, "w": jax.Array, "ids": jax.Array,
                              "return": jax.Array}
    else:
        # x (B, S/tp, K) gathered, @ each w_i (N_i/tp, K)^T -> (B, S, N_i/tp).
        def body(x, ids, *ws):
            shard_of = jnp.asarray(shard_table, dtype=jnp.int32)
            j = ids[0]
            rows = x.shape[1]
            hr = rows // 2
            seq = rows * n
            outs = [jnp.zeros((x.shape[0], seq, w.shape[0]), x.dtype) for w in ws]
            top, bot = x[:, :hr], x[:, hr:]
            for step in range(n):
                if step:
                    top = jax.lax.ppermute(top, "tp", cw)
                    bot = jax.lax.ppermute(bot, "tp", ccw)
                # After `step` hops the top half held is ring position j-step's,
                # the bottom half j+step's; each lands at its owner's rows.
                at_top = shard_of[(j - step) % n] * rows
                at_bot = shard_of[(j + step) % n] * rows + hr
                outs = [
                    jax.lax.dynamic_update_slice_in_dim(
                        jax.lax.dynamic_update_slice_in_dim(
                            o, jnp.einsum("bsk,nk->bsn", top, w), at_top, axis=1),
                        jnp.einsum("bsk,nk->bsn", bot, w), at_bot, axis=1)
                    for o, w in zip(outs, ws)
                ]
            # One output is returned bare: a 1-tuple fails jax_op's fake-tensor
            # step under Dynamo (Wan cross-attention, 2026-11-03).
            return tuple(outs) if len(outs) > 1 else outs[0]

        in_specs = (P(None, "tp", None), P("tp")) + (P("tp", None),) * n_weights
        out_specs = (P(None, None, "tp"),) * n_weights

        def ring(*args):
            return shard_map(body, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs if n_weights > 1 else out_specs[0],
                             check_vma=False)(*args)

        # jax_op reads each parameter's annotation, so the arities are spelled
        # out, and the annotations are set as objects: under this module's
        # `from __future__ import annotations` they would be strings naming a
        # `jax` that is imported locally, not at module scope.
        if n_weights == 1:
            def fn(x, ids, w0):
                return ring(x, ids, w0)
        elif n_weights == 3:
            def fn(x, ids, w0, w1, w2):
                return ring(x, ids, w0, w1, w2)
        else:
            raise ValueError(f"all_gather_matmul supports 1 or 3 weights, got {n_weights}")
        names = ["x", "ids"] + [f"w{i}" for i in range(n_weights)]
        fn.__annotations__ = {**{k: jax.Array for k in names},
                              "return": (tuple[(jax.Array,) * n_weights] if n_weights > 1
                                         else jax.Array)}

    op = torch.tpu.pallas.jax_op(
        f"difflet::sp_{kind}_matmul_{n_weights}", fn, mesh=mesh,
        input_partition_specs=in_specs,
        # A tuple of specs, one per output: see the module docstring.
        output_partition_specs=tuple(out_specs),
    )
    ops[key] = op
    return op


def matmul_reduce_scatter(x: torch.Tensor, linear) -> torch.Tensor:
    """``linear(x)`` for a row-parallel ``linear``, reduced over the TP group and
    scattered by sequence: (B, S, K/tp) -> (B, S/tp, N). The bias is added once,
    after the reduction."""
    out = _op("rs", 1)(x, linear.weight, _STATE["ids"])
    if linear.bias is not None:
        out = out + linear.bias.to(out.dtype)
    return out


def all_gather_matmul(x: torch.Tensor, linears) -> list[torch.Tensor]:
    """``[l(gather(x)) for l in linears]`` for column-parallel ``linears`` sharing
    one input: (B, S/tp, K) -> (B, S, N_i/tp) each, the gather overlapped with
    the matmuls."""
    outs = _op("ag", len(linears))(x, _STATE["ids"], *[l.weight for l in linears])
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    return [o + l.bias.to(o.dtype) if l.bias is not None else o
            for o, l in zip(outs, linears)]


def supports(seq_len: int, tp: int) -> bool:
    """The rings split each sequence shard in two (AG), so S must divide by 2*tp."""
    return seq_len % (2 * tp) == 0


__all__ = ["SP_ENV", "all_gather_matmul", "matmul_reduce_scatter", "prepare",
           "require_compiled", "requested", "supports"]
