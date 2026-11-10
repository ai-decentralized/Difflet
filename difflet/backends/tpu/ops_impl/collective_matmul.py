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

**Every rank must compile the same program around the rings.** torch_tpu
compiles one program per process and inlines these rings into it (their
``collective-permute``s all carry ``channel_id=1`` and pair by order). Values
that differ by rank therefore enter the compiled graph as runtime tensors (the
ring position is ``_STATE["ids"]`` for this reason), never as Python constants
the compiler can specialize on. LTX-2 broke this by padding its sequence inside
the graph and slicing each rank's shard at a constant offset: XLA folded the pad
away on some ranks only, the programs diverged, and a chip halted in the first
ring hop (v5e-4, torch_tpu 0.2.0, 2026-11-10; ``benchmark/tpu_profile/
cm_padded_repro.py``). Pad before the graph, or pass the offset as a tensor.

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


#: The ops Wan / Qwen-Image / HunyuanVideo use: (kind, arity).
DEFAULT_OPS = (("rs", 1), ("ag", 1), ("ag", 3))


def prepare(ops=DEFAULT_OPS) -> None:
    """Build the mesh, the shard table and the jax_op ops, outside any compiled
    region (traced lazily, Dynamo would graph-break on the registration).
    Collective: every TP rank must call it, in step. ``ops`` lists the
    ``(kind, arity)`` a model needs; calling again with more builds the rest."""
    global _STATE
    if _STATE is not None:
        for kind, arity in ops:
            _op(kind, arity)
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
    # Every op the model's forwards use, built now: built lazily they would be
    # built inside Dynamo's trace, which cannot run JAX.
    for kind, arity in ops:
        _op(kind, arity)


def _op(kind: str, n_weights: int):
    """The jax_op for ``kind`` built once: 'rs' over ``n_weights`` (input, weight)
    pairs whose products are summed, 'ag' over ``n_weights`` weights sharing one
    gathered input."""
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
        # sum_i x_i (B, S, K_i/tp) @ w_i (N, K_i/tp)^T, summed over the group,
        # this rank's rows. One pair is the plain row-parallel linear; two are a
        # row-parallel linear over a concatenation (FLUX's single-block proj_out
        # over [attn, mlp]) without materialising the concatenation.
        def body(*args):
            *pairs, ids = args
            xs, ws = pairs[0::2], pairs[1::2]
            shard_of = jnp.asarray(shard_table, dtype=jnp.int32)
            j = ids[0]
            rows = xs[0].shape[1] // n
            half = ws[0].shape[0] // 2

            def chunk(x, pos):
                return jax.lax.dynamic_slice_in_dim(x, shard_of[pos % n] * rows, rows, axis=1)

            def partial(pos, sl):
                out = None
                for x, w in zip(xs, ws):
                    term = jnp.einsum("bsk,nk->bsn", chunk(x, pos), w[sl])
                    out = term if out is None else out + term
                return out

            lo, hi = slice(None, half), slice(half, None)
            left = right = None
            for step in range(n):
                pl = partial(j - 1 - step, lo)
                pr = partial(j + 1 + step, hi)
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

        in_specs = (P(None, None, "tp"), P(None, "tp")) * n_weights + (P("tp"),)
        out_specs = (P(None, "tp", None),)

        def reduce(*args):
            return shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_specs[0],
                             check_vma=False)(*args)

        # Arities spelled out for jax_op, as for 'ag' below.
        if n_weights == 1:
            def fn(x, w, ids):
                return reduce(x, w, ids)
            names = ["x", "w", "ids"]
        elif n_weights == 2:
            def fn(x0, w0, x1, w1, ids):
                return reduce(x0, w0, x1, w1, ids)
            names = ["x0", "w0", "x1", "w1", "ids"]
        else:
            raise ValueError(f"matmul_reduce_scatter supports 1 or 2 pairs, got {n_weights}")
        fn.__annotations__ = {**{k: jax.Array for k in names}, "return": jax.Array}
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
        elif n_weights == 2:
            def fn(x, ids, w0, w1):
                return ring(x, ids, w0, w1)
        elif n_weights == 3:
            def fn(x, ids, w0, w1, w2):
                return ring(x, ids, w0, w1, w2)
        elif n_weights == 4:
            def fn(x, ids, w0, w1, w2, w3):
                return ring(x, ids, w0, w1, w2, w3)
        else:
            raise ValueError(f"all_gather_matmul supports 1 to 4 weights, got {n_weights}")
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


def matmul2_reduce_scatter(x0: torch.Tensor, linear0, x1: torch.Tensor, linear1,
                           bias: torch.Tensor | None = None) -> torch.Tensor:
    """``linear0(x0) + linear1(x1)`` for two row-parallel linears with the same
    output, reduced and scattered as one: (B, S, K_i/tp) -> (B, S/tp, N). The
    linears' own biases are ignored; ``bias`` is added once after the reduction."""
    out = _op("rs", 2)(x0, linear0.weight, x1, linear1.weight, _STATE["ids"])
    if bias is not None:
        out = out + bias.to(out.dtype)
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


__all__ = ["DEFAULT_OPS", "SP_ENV", "all_gather_matmul", "matmul2_reduce_scatter",
           "matmul_reduce_scatter", "prepare",
           "require_compiled", "requested", "supports"]
