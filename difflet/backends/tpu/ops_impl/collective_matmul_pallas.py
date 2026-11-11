"""Pallas TPU kernel for the SP all-gather + column-parallel matmul ring.

Step 2 of docs/plans/2026-11-10-tpu-collective-matmul-pallas.md: the same
bidirectional ring as ``collective_matmul``'s JAX ``ag`` op, but each hop's
product tile goes from VMEM straight to its final rows of the output (one HBM
write), with no temporary product and no ``dynamic_update_slice`` / gather
assembly afterwards. The remote copy of the next half overlaps the matmul of
the current one.

``ag_matmul_body`` is the per-device body, called inside ``shard_map`` over a
1-D mesh axis (default ``"tp"``) whose order is the ring order. Rank-dependent
values (the ring position, owners, neighbours) are runtime int32 inputs, never
Python constants -- rule 1 of the plan; every device runs the same program.

Layout per device (x is this device's sequence shard, split in two halves that
travel in opposite directions):

    x     (B, R, K)            R = rows per device, even
    w_i   (N_i, K)             this device's output columns
    out_i (B, n * R, N_i)      viewed as (B, n, 2, R/2, N_i): [owner shard, half]

The ``pallas_call``s are not marked ``has_side_effects``: through torch_tpu's
``jax_op`` the program goes through XLA's SPMD partitioner, which rejects a
side-effecting custom call without a sharding ("Side-effect HLO must have
sharding", 2026-11-11). The outputs are always consumed, so nothing is dropped.

Communication buffers are an extra output (2 directions x 3 slots x the half
shape) in HBM. Three slots make the reuse race-free: a neighbour is at most one
hop behind (each device waits for both neighbours' previous hop before going
on), so the slot written at hop s+1 is never the one a neighbour still reads.
"""

from __future__ import annotations

SLOTS = 3


def ag_matmul_body(x, ws, owners, nbrs, *, n: int, axis: str = "tp", tn: int = 256,
                   vmem_limit_bytes: int = 96 * 2**20, interpret=False, ablate: str = "",
                   weights_kn: bool = False, resident: bool = False,
                   manual: bool = False):
    """Per-device all-gather + matmul with every weight in ``ws``.

    owners: int32 (2, n) -- owners[0, s] / owners[1, s] is the shard (output row
        block) whose top / bottom half this device holds after s hops.
    nbrs:   int32 (2,)   -- mesh index along ``axis`` of the clockwise
        neighbour (receives the top halves) and the counter-clockwise one.
    weights_kn: the weights are given transposed, (K, N_i): the kernel then runs a
        plain (M, K) x (K, N) matmul per tile instead of contracting both
        operands on their last axis.
    resident: keep the halves in VMEM for the whole call -- the local ones are
        copied in once, the remote ones arrive by VMEM-to-VMEM DMA (they never
        touch HBM), and each matmul pipeline streams only weight tiles. Needs
        2 x 3 x B x R/2 x K elements of VMEM (Wan: 36 MB).
    manual: with ``resident``, replace the per-(hop, weight) ``emit_pipeline``s by
        one hand-written double-buffered stream over every (hop, weight, column
        tile): the next weight tile is always in flight, across weights and hops,
        so only the very first tile load is exposed. Output tiles are written back
        asynchronously, two in flight. The tiles of a hop are a runtime loop, so
        compile time does not grow with the tile count (a fully unrolled stream
        took 225 s to compile at Wan's FFN shape). Weights may differ in width;
        one tile width (a multiple of 128) must divide them all.
    ablate: profiling only -- ``"comm"`` skips the remote copies (every hop
        multiplies the local halves; wrong values), ``"compute"`` skips the matmuls.
    Returns a tuple of (B, n * R, N_i), one per weight.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    b, rows, k = x.shape
    if rows % 2:
        raise ValueError(f"rows per device must be even (two ring halves), got {rows}")
    hr = rows // 2
    tns = []
    kd, nd = (0, 1) if weights_kn else (1, 0)  # weight axis of K / of the columns
    cols = []
    for w in ws:
        if w.shape[kd] != k:
            raise ValueError(f"weight {w.shape} does not match K={k}")
        cols.append(w.shape[nd])
        tns.append(_col_tile(w.shape[nd], tn))
    nw = len(ws)
    tm0 = _common_tile(cols, tn)  # the manual stream's one tile width

    def kernel(owners_ref, nbrs_ref, x_ref, *refs):
        w_refs = refs[:nw]
        out_refs = refs[nw:2 * nw]
        buf_ref, send_sem, recv_sem = refs[2 * nw:]

        # Every neighbour must be inside this kernel (its buffers live) before
        # anything is written into them.
        barrier = pltpu.get_barrier_semaphore()
        for d in range(2):
            pl.semaphore_signal(barrier, 1, device_id={axis: nbrs_ref[d]})
        pl.semaphore_wait(barrier, 2)

        def held(d, s):
            # The half travelling in direction d after s hops.
            return x_ref.at[:, d] if s == 0 or ablate == "comm" else buf_ref.at[d, s % SLOTS]

        def matmul(s, w_ref, out_ref, tn_i):
            # Both halves held after s hops against one weight tile: each tile
            # of W is read from HBM once per hop, not once per half.
            own_top, own_bot = owners_ref[0, s], owners_ref[1, s]

            def body(xt_blk, xb_blk, w_blk, ot_blk, ob_blk):
                for x_blk, o_blk in ((xt_blk, ot_blk), (xb_blk, ob_blk)):
                    o_blk[...] = jax.lax.dot_general(
                        x_blk[...], w_blk[...], (((1,), (kd,)), ((), ())),
                        preferred_element_type=jnp.float32).astype(o_blk.dtype)

            x_spec = pl.BlockSpec((pl.Squeezed(), hr, k), lambda bi, j: (bi, 0, 0))
            o_shape = (pl.Squeezed(), pl.Squeezed(), pl.Squeezed(), hr, tn_i)
            w_spec = (pl.BlockSpec((k, tn_i), lambda bi, j: (0, j)) if weights_kn
                      else pl.BlockSpec((tn_i, k), lambda bi, j: (j, 0)))
            pltpu.emit_pipeline(
                body,
                grid=(b, w_ref.shape[nd] // tn_i),
                in_specs=[x_spec, x_spec, w_spec],
                out_specs=[pl.BlockSpec(o_shape, lambda bi, j: (bi, own_top, 0, 0, j)),
                           pl.BlockSpec(o_shape, lambda bi, j: (bi, own_bot, 1, 0, j))],
            )(held(0, s), held(1, s), w_ref, out_ref, out_ref)

        for s in range(n):
            copies = []
            if s + 1 < n and ablate != "comm":
                slot = (s + 1) % SLOTS
                for d in range(2):
                    c = pltpu.make_async_remote_copy(
                        held(d, s), buf_ref.at[d, slot], send_sem.at[d, slot],
                        recv_sem.at[d, slot], device_id={axis: nbrs_ref[d]})
                    c.start()
                    copies.append(c)
            for w_ref, out_ref, tn_i in zip(w_refs, out_refs, tns):
                if ablate != "compute":
                    matmul(s, w_ref, out_ref, tn_i)
            for c in copies:
                c.wait_send()
                c.wait_recv()

    def kernel_resident(owners_ref, nbrs_ref, x_ref, *refs):
        w_refs = refs[:nw]
        out_refs = refs[nw:2 * nw]
        xbuf, load_sem, send_sem, recv_sem = refs[2 * nw:]

        barrier = pltpu.get_barrier_semaphore()
        for d in range(2):
            pl.semaphore_signal(barrier, 1, device_id={axis: nbrs_ref[d]})
        pl.semaphore_wait(barrier, 2)

        loads = [pltpu.make_async_copy(x_ref.at[:, d], xbuf.at[d, 0], load_sem.at[d])
                 for d in range(2)]
        for c in loads:
            c.start()
        for c in loads:
            c.wait()

        def held(d, s):
            return xbuf.at[d, 0 if ablate == "comm" else s % SLOTS]

        def matmul(s, w_ref, out_ref, tn_i):
            own_top, own_bot = owners_ref[0, s], owners_ref[1, s]
            xt, xb = held(0, s), held(1, s)

            def body(w_blk, ot_blk, ob_blk):
                w = w_blk[...]
                for x_r, o_blk in ((xt, ot_blk), (xb, ob_blk)):
                    for bi in range(b):
                        o_blk[bi] = jax.lax.dot_general(
                            x_r[bi], w, (((1,), (kd,)), ((), ())),
                            preferred_element_type=jnp.float32).astype(o_blk.dtype)

            o_shape = (b, pl.Squeezed(), pl.Squeezed(), hr, tn_i)
            w_spec = (pl.BlockSpec((k, tn_i), lambda j: (0, j)) if weights_kn
                      else pl.BlockSpec((tn_i, k), lambda j: (j, 0)))
            pltpu.emit_pipeline(
                body,
                grid=(w_ref.shape[nd] // tn_i,),
                in_specs=[w_spec],
                out_specs=[pl.BlockSpec(o_shape, lambda j: (0, own_top, 0, 0, j)),
                           pl.BlockSpec(o_shape, lambda j: (0, own_bot, 1, 0, j))],
            )(w_ref, out_ref, out_ref)

        for s in range(n):
            copies = []
            if s + 1 < n and ablate != "comm":
                slot = (s + 1) % SLOTS
                for d in range(2):
                    c = pltpu.make_async_remote_copy(
                        held(d, s), xbuf.at[d, slot], send_sem.at[d, slot],
                        recv_sem.at[d, slot], device_id={axis: nbrs_ref[d]})
                    c.start()
                    copies.append(c)
            for w_ref, out_ref, tn_i in zip(w_refs, out_refs, tns):
                if ablate != "compute":
                    matmul(s, w_ref, out_ref, tn_i)
            for c in copies:
                c.wait_send()
                c.wait_recv()

    def kernel_manual(owners_ref, nbrs_ref, x_ref, *refs):
        w_refs = refs[:nw]
        out_refs = refs[nw:2 * nw]
        xbuf, wbuf, obuf, load_sem, send_sem, recv_sem, w_sem, o_sem = refs[2 * nw:]
        tn0 = tm0                    # one column-tile width for every weight
        cum = [0]                    # first tile of each weight within a hop
        for c in cols:
            cum.append(cum[-1] + c // tn0)
        tph = cum[-1]                # tiles per hop
        total = n * tph

        barrier = pltpu.get_barrier_semaphore()
        for d in range(2):
            pl.semaphore_signal(barrier, 1, device_id={axis: nbrs_ref[d]})
        pl.semaphore_wait(barrier, 2)

        def cols_of(j):
            return pl.ds(pl.multiple_of(j * tn0, tn0), tn0)

        def w_src(ii, j):
            return w_refs[ii].at[:, cols_of(j)] if weights_kn else w_refs[ii].at[cols_of(j), :]

        def which(t):
            # Global tile t -> (weight, column tile of that weight). Weights may
            # differ in width (HunyuanVideo / FLUX single blocks: q|k|v|mlp), so
            # each has its own run of tiles within a hop.
            u = jnp.asarray(t, jnp.int32) % tph
            i = sum(((u >= cum[kk]).astype(jnp.int32) for kk in range(1, nw)), jnp.int32(0))
            base = sum((jnp.where(i == kk, cum[kk], 0) for kk in range(1, nw)), jnp.int32(0))
            return i, u - base

        def w_start(t, slot):
            # The weights are separate refs, so one guarded DMA start per weight.
            i, j = which(t)
            for ii in range(nw):
                @pl.when(i == ii)
                def _():
                    pltpu.make_async_copy(w_src(ii, j), wbuf.at[slot], w_sem.at[slot]).start()

        def w_wait(slot):
            pltpu.make_async_copy(w_src(0, 0), wbuf.at[slot], w_sem.at[slot]).wait()

        def o_wait(slot):
            for d in range(2):
                pltpu.make_async_copy(obuf.at[slot, d], out_refs[0].at[:, 0, d, :, cols_of(0)],
                                      o_sem.at[slot, d]).wait()

        w_start(0, 0)  # the first weight tile overlaps the local loads
        loads = [pltpu.make_async_copy(x_ref.at[:, d], xbuf.at[d, 0], load_sem.at[d])
                 for d in range(2)]
        for c in loads:
            c.start()
        for c in loads:
            c.wait()

        def slot_of(s):
            return 0 if ablate == "comm" else s % SLOTS

        copies = []
        for s in range(n):  # hops: static (n = 4), tiles within a hop: a runtime loop
            for c in copies:  # hop s's halves have arrived
                c.wait_send()
                c.wait_recv()
            copies = []
            if s + 1 < n and ablate != "comm":
                for d in range(2):
                    c = pltpu.make_async_remote_copy(
                        xbuf.at[d, slot_of(s)], xbuf.at[d, (s + 1) % SLOTS],
                        send_sem.at[d, (s + 1) % SLOTS], recv_sem.at[d, (s + 1) % SLOTS],
                        device_id={axis: nbrs_ref[d]})
                    c.start()
                    copies.append(c)

            def tile(u, carry, s=s):
                t = s * tph + u
                slot = t % 2

                @pl.when(t + 1 < total)
                def _():
                    w_start(t + 1, 1 - slot)  # always one weight tile in flight

                w_wait(slot)

                @pl.when(t >= 2)
                def _():
                    o_wait(slot)  # this output slot's previous write-back is done

                if ablate != "compute":
                    w = wbuf[slot]
                    for d in range(2):
                        for bi in range(b):
                            obuf[slot, d, bi] = jax.lax.dot_general(
                                xbuf[d, slot_of(s), bi], w, (((1,), (kd,)), ((), ())),
                                preferred_element_type=jnp.float32).astype(obuf.dtype)
                i, j = which(t)
                for ii in range(nw):
                    @pl.when(i == ii)
                    def _():
                        for d in range(2):
                            pltpu.make_async_copy(
                                obuf.at[slot, d],
                                out_refs[ii].at[:, owners_ref[d, s], d, :, cols_of(j)],
                                o_sem.at[slot, d]).start()
                return carry

            jax.lax.fori_loop(0, tph, tile, 0)
        for c in copies:
            c.wait_send()
            c.wait_recv()
        for t in range(max(0, total - 2), total):  # the last write-backs
            o_wait(t % 2)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    if resident and manual:
        if tm0 is None:
            raise ValueError(f"no column tile (a multiple of 128) divides every width {cols}")
        wshape = (k, tm0) if weights_kn else (tm0, k)
        outs = pl.pallas_call(
            kernel_manual,
            out_shape=[jax.ShapeDtypeStruct((b, n, 2, hr, c), x.dtype) for c in cols],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[any_spec] * (1 + nw),
                out_specs=[any_spec] * nw,
                scratch_shapes=[pltpu.VMEM((2, SLOTS, b, hr, k), x.dtype),
                                pltpu.VMEM((2,) + wshape, x.dtype),
                                pltpu.VMEM((2, 2, b, hr, tm0), x.dtype),
                                pltpu.SemaphoreType.DMA((2,)),
                                pltpu.SemaphoreType.DMA((2, SLOTS)),
                                pltpu.SemaphoreType.DMA((2, SLOTS)),
                                pltpu.SemaphoreType.DMA((2,)),
                                pltpu.SemaphoreType.DMA((2, 2))],
            ),
            compiler_params=pltpu.CompilerParams(collective_id=0,
                                                 vmem_limit_bytes=vmem_limit_bytes),
            interpret=interpret,
        )(owners, nbrs, x.reshape(b, 2, hr, k), *ws)
        return tuple(o.reshape(b, n * rows, o.shape[-1]) for o in outs)
    if resident:
        outs = pl.pallas_call(
            kernel_resident,
            out_shape=[jax.ShapeDtypeStruct((b, n, 2, hr, c), x.dtype) for c in cols],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[any_spec] * (1 + nw),
                out_specs=[any_spec] * nw,
                scratch_shapes=[pltpu.VMEM((2, SLOTS, b, hr, k), x.dtype),
                                pltpu.SemaphoreType.DMA((2,)),
                                pltpu.SemaphoreType.DMA((2, SLOTS)),
                                pltpu.SemaphoreType.DMA((2, SLOTS))],
            ),
            compiler_params=pltpu.CompilerParams(collective_id=0,
                                                 vmem_limit_bytes=vmem_limit_bytes),
            interpret=interpret,
        )(owners, nbrs, x.reshape(b, 2, hr, k), *ws)
        return tuple(o.reshape(b, n * rows, o.shape[-1]) for o in outs)

    out_shape = [jax.ShapeDtypeStruct((b, n, 2, hr, c), x.dtype) for c in cols]
    out_shape.append(jax.ShapeDtypeStruct((2, SLOTS, b, hr, k), x.dtype))
    outs = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[any_spec] * (1 + nw),
            out_specs=[any_spec] * (nw + 1),
            scratch_shapes=[pltpu.SemaphoreType.DMA((2, SLOTS)),
                            pltpu.SemaphoreType.DMA((2, SLOTS))],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0,
                                             vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
    )(owners, nbrs, x.reshape(b, 2, hr, k), *ws)
    return tuple(o.reshape(b, n * rows, o.shape[-1]) for o in outs[:nw])


#: VMEM the kernel may plan for (v5e has 128 MiB per core; leave room for Mosaic).
VMEM_BUDGET = 96 * 2**20


def unsupported(x_shape, w_shapes, itemsize: int, tn: int = 512) -> str | None:
    """Why the streamed kernel (``resident=True, manual=True``, weights (N, K))
    cannot take these shapes, or None. Callers fall back to the JAX ring."""
    b, rows, k = x_shape
    if rows % 2:
        return f"odd rows per device ({rows})"
    if any(ws[1] != k for ws in w_shapes):
        return "weight K does not match the input"
    widths = [ws[0] for ws in w_shapes]
    t = _common_tile(widths, tn)
    if t is None:
        return f"no column tile (a multiple of 128) divides every width {sorted(set(widths))}"
    hr = rows // 2
    need = itemsize * (2 * SLOTS * b * hr * k   # resident halves
                       + 2 * t * k              # weight tiles, double-buffered
                       + 2 * 2 * b * hr * t)    # output tiles, double-buffered
    need += 4 * hr * t                          # one fp32 product
    if need > VMEM_BUDGET:
        return f"needs {need / 2**20:.0f} MiB of VMEM (budget {VMEM_BUDGET / 2**20:.0f})"
    return None


def _common_tile(widths, limit: int) -> int | None:
    """The largest multiple of 128 up to ``limit`` that divides every width, or
    None (q|k|v 768 + mlp 3072 per device: 384)."""
    for t in range(limit - limit % 128, 0, -128):
        if all(w % t == 0 for w in widths):
            return t
    return None


def _col_tile(cols: int, limit: int) -> int:
    """The largest multiple of 128 up to ``limit`` that divides ``cols``, else ``cols``
    (Wan's FFN has 3456 columns per device: 384)."""
    for t in range(limit - limit % 128, 0, -128):
        if cols % t == 0:
            return t
    return cols


def ring_tables(j, shard_of_pos, n: int):
    """Runtime owners / neighbours for ring position ``j`` (a traced int32).

    After s hops a device at position j holds position (j - s)'s top half
    (sent clockwise) and (j + s)'s bottom half; ``shard_of_pos`` maps a ring
    position to the shard (output row block) it owns.
    """
    import jax.numpy as jnp

    shard_of = jnp.asarray(shard_of_pos, dtype=jnp.int32)
    s = jnp.arange(n, dtype=jnp.int32)
    owners = jnp.stack([shard_of[(j - s) % n], shard_of[(j + s) % n]])
    nbrs = jnp.stack([(j + 1) % n, (j - 1) % n]).astype(jnp.int32)
    return owners, nbrs
