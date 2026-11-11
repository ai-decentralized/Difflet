"""Pallas TPU kernel for the SP all-gather + column-parallel matmul ring.

The same bidirectional ring as ``collective_matmul``'s JAX ``ag`` op
(docs/plans/2026-11-10-tpu-collective-matmul-pallas.md), as one kernel:

- Each device's sequence shard is split into a top and a bottom half. The top
  halves travel clockwise, the bottom halves counter-clockwise; after
  ``ring_size - 1`` hops every device has seen every half.
- The halves stay in VMEM for the whole call: the local ones are copied in once,
  the remote ones arrive by VMEM-to-VMEM DMA and never touch HBM. The copy of the
  next hop's halves overlaps the current hop's matmuls.
- The weights are streamed through VMEM one column tile at a time, as a single
  double-buffered stream over every (hop, weight, column tile): the next tile is
  always in flight, across weights and hops, so only the first load is exposed.
- Each product tile is written once, asynchronously, at its final rows of the
  output: no temporary product, no ``dynamic_update_slice`` assembly afterwards.

:func:`all_gather_matmul_body` is the per-device body, called inside ``shard_map``
over a 1-D mesh axis (default ``"tp"``) whose order is the ring order.
Rank-dependent values (the ring position, the owner shards, the neighbours) are
runtime int32 inputs, never Python constants -- rule 1 of the plan: every device
must run the same program.

Layout per device:

    x          (batch, rows, in_features)       rows per device, even
    weight_i   (out_features_i, in_features)    this device's output columns
    output_i   (batch, ring_size * rows, out_features_i)
               written as (batch, ring_size, 2, rows / 2, out_features_i):
               [owner shard, half]

Communication slots: the halves have three VMEM slots per direction. A
neighbour is at most one hop behind (each device waits for the previous hop's
copies before starting the next), so the slot written at hop h + 1 is never the
one a neighbour still reads; Pallas' race detector catches two slots.

The ``pallas_call`` is not marked ``has_side_effects``: through torch_tpu's
``jax_op`` the program goes through XLA's SPMD partitioner, which rejects a
side-effecting custom call without a sharding ("Side-effect HLO must have
sharding", 2026-11-11). The outputs are always consumed, so nothing is dropped.

Earlier variants (one ``emit_pipeline`` per hop and weight, with the halves in
HBM or in VMEM; pre-transposed weights) were measured and dropped; they are in
commit 01d027fe and the numbers in the plan.
"""

from __future__ import annotations

#: Ring slots per direction for the travelling halves (see the module docstring).
RING_SLOTS = 3

#: VMEM the kernel may plan for (v5e has 128 MiB per core; leave room for Mosaic).
VMEM_BUDGET_BYTES = 96 * 2**20

#: Upper bound on the column tile width. The end-to-end results in the plan
#: (Wan, FLUX, HunyuanVideo) were measured with this value.
DEFAULT_COLUMN_TILE_LIMIT = 256


def all_gather_matmul_body(x, weights, owner_shards, neighbors, *, ring_size: int,
                           axis: str = "tp",
                           column_tile_limit: int = DEFAULT_COLUMN_TILE_LIMIT,
                           vmem_limit_bytes: int = VMEM_BUDGET_BYTES,
                           interpret=False, profile_ablation: str = ""):
    """Per-device all-gather of ``x`` along the ring + matmul with every weight.

    x:            (batch, rows, in_features), this device's sequence shard.
    weights:      sequence of (out_features_i, in_features); widths may differ, but
                  one column tile width (a multiple of 128, at most
                  ``column_tile_limit``) must divide them all.
    owner_shards: int32 (2, ring_size) -- owner_shards[0, hop] / owner_shards[1, hop]
                  is the shard (output row block) whose top / bottom half this
                  device holds after ``hop`` hops. From :func:`ring_tables`.
    neighbors:    int32 (2,) -- mesh index along ``axis`` of the clockwise neighbour
                  (receives the top halves) and of the counter-clockwise one.
    interpret:    False on the chips; ``pltpu.InterpretParams(...)`` for CPU tests.
    profile_ablation: profiling only, wrong values -- ``"communication"`` skips
                  the remote copies (every hop multiplies the local halves),
                  ``"compute"`` skips the matmuls.

    Returns a tuple of (batch, ring_size * rows, out_features_i), one per weight.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if profile_ablation not in ("", "communication", "compute"):
        raise ValueError(f"profile_ablation must be '', 'communication' or 'compute', "
                         f"got {profile_ablation!r}")
    batch, rows, in_features = x.shape
    if rows % 2:
        raise ValueError(f"rows per device must be even (two ring halves), got {rows}")
    half_rows = rows // 2
    output_widths = []
    for weight in weights:
        if weight.shape[1] != in_features:
            raise ValueError(f"weight {weight.shape} does not match in_features={in_features}")
        output_widths.append(weight.shape[0])
    num_weights = len(weights)
    tile_width = _common_column_tile(output_widths, column_tile_limit)
    if tile_width is None:
        raise ValueError(f"no column tile (a multiple of 128) divides every width {output_widths}")

    # The global tile index runs over every (hop, weight, column tile) in that
    # order; within a hop, weight i owns tiles [first_tile_of_weight[i],
    # first_tile_of_weight[i + 1]).
    first_tile_of_weight = [0]
    for width in output_widths:
        first_tile_of_weight.append(first_tile_of_weight[-1] + width // tile_width)
    tiles_per_hop = first_tile_of_weight[-1]
    total_tiles = ring_size * tiles_per_hop
    skip_communication = profile_ablation == "communication"
    skip_compute = profile_ablation == "compute"

    def kernel(owner_shards_ref, neighbors_ref, x_ref, *refs):
        weight_refs = refs[:num_weights]
        output_refs = refs[num_weights:2 * num_weights]
        (halves, weight_tiles, output_tiles, load_semaphores, send_semaphores,
         receive_semaphores, weight_semaphores, output_semaphores) = refs[2 * num_weights:]

        # Both neighbours must be inside this kernel (their VMEM slots belong to
        # it) before anything is copied into them.
        barrier = pltpu.get_barrier_semaphore()
        for direction in range(2):
            pl.semaphore_signal(barrier, 1, device_id={axis: neighbors_ref[direction]})
        pl.semaphore_wait(barrier, 2)

        def columns_of(column_tile):
            return pl.ds(pl.multiple_of(column_tile * tile_width, tile_width), tile_width)

        def locate_tile(tile):
            """Global tile index -> (weight index, column tile of that weight)."""
            tile_in_hop = jnp.asarray(tile, jnp.int32) % tiles_per_hop
            weight_index = sum(
                ((tile_in_hop >= first_tile_of_weight[i]).astype(jnp.int32)
                 for i in range(1, num_weights)), jnp.int32(0))
            weight_first_tile = sum(
                (jnp.where(weight_index == i, first_tile_of_weight[i], 0)
                 for i in range(1, num_weights)), jnp.int32(0))
            return weight_index, tile_in_hop - weight_first_tile

        def start_weight_tile_load(tile, buffer_slot):
            # The weights are separate refs: one guarded DMA start per weight.
            weight_index, column_tile = locate_tile(tile)
            for i in range(num_weights):
                @pl.when(weight_index == i)
                def _():
                    pltpu.make_async_copy(
                        weight_refs[i].at[columns_of(column_tile), :],
                        weight_tiles.at[buffer_slot], weight_semaphores.at[buffer_slot]).start()

        def wait_weight_tile_load(buffer_slot):
            # A wait only needs a descriptor of the same size: any weight's tile 0.
            pltpu.make_async_copy(weight_refs[0].at[columns_of(0), :],
                                  weight_tiles.at[buffer_slot],
                                  weight_semaphores.at[buffer_slot]).wait()

        def wait_output_write_back(buffer_slot):
            for direction in range(2):
                pltpu.make_async_copy(output_tiles.at[buffer_slot, direction],
                                      output_refs[0].at[:, 0, direction, :, columns_of(0)],
                                      output_semaphores.at[buffer_slot, direction]).wait()

        def ring_slot(hop):
            return 0 if skip_communication else hop % RING_SLOTS

        # The first weight tile load overlaps the local halves' load.
        start_weight_tile_load(0, 0)
        local_loads = [pltpu.make_async_copy(x_ref.at[:, direction], halves.at[direction, 0],
                                             load_semaphores.at[direction])
                       for direction in range(2)]
        for copy in local_loads:
            copy.start()
        for copy in local_loads:
            copy.wait()

        remote_copies = []
        # Hops: a static loop (ring_size = 4). Tiles within a hop: a runtime loop, so
        # compile time does not grow with the tile count (fully unrolled, Wan's FFN
        # took 225 s to compile).
        for hop in range(ring_size):
            for copy in remote_copies:  # this hop's halves have arrived
                copy.wait_send()
                copy.wait_recv()
            remote_copies = []
            if hop + 1 < ring_size and not skip_communication:
                next_slot = (hop + 1) % RING_SLOTS
                for direction in range(2):
                    copy = pltpu.make_async_remote_copy(
                        halves.at[direction, ring_slot(hop)], halves.at[direction, next_slot],
                        send_semaphores.at[direction, next_slot],
                        receive_semaphores.at[direction, next_slot],
                        device_id={axis: neighbors_ref[direction]})
                    copy.start()
                    remote_copies.append(copy)

            def process_tile(tile_in_hop, carry, hop=hop):
                tile = hop * tiles_per_hop + tile_in_hop
                buffer_slot = tile % 2

                @pl.when(tile + 1 < total_tiles)
                def _():
                    start_weight_tile_load(tile + 1, 1 - buffer_slot)  # always one in flight

                wait_weight_tile_load(buffer_slot)

                @pl.when(tile >= 2)
                def _():
                    wait_output_write_back(buffer_slot)  # this slot's previous tile is out

                if not skip_compute:
                    weight_tile = weight_tiles[buffer_slot]
                    for direction in range(2):
                        for batch_index in range(batch):
                            output_tiles[buffer_slot, direction, batch_index] = jax.lax.dot_general(
                                halves[direction, ring_slot(hop), batch_index], weight_tile,
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32).astype(output_tiles.dtype)
                weight_index, column_tile = locate_tile(tile)
                for i in range(num_weights):
                    @pl.when(weight_index == i)
                    def _():
                        for direction in range(2):
                            owner = owner_shards_ref[direction, hop]
                            pltpu.make_async_copy(
                                output_tiles.at[buffer_slot, direction],
                                output_refs[i].at[:, owner, direction, :, columns_of(column_tile)],
                                output_semaphores.at[buffer_slot, direction]).start()
                return carry

            jax.lax.fori_loop(0, tiles_per_hop, process_tile, 0)
        for copy in remote_copies:
            copy.wait_send()
            copy.wait_recv()
        for tile in range(max(0, total_tiles - 2), total_tiles):  # the last write-backs
            wait_output_write_back(tile % 2)

    any_memory = pl.BlockSpec(memory_space=pl.ANY)
    outputs = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((batch, ring_size, 2, half_rows, width), x.dtype)
                   for width in output_widths],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[any_memory] * (1 + num_weights),
            out_specs=[any_memory] * num_weights,
            scratch_shapes=[
                pltpu.VMEM((2, RING_SLOTS, batch, half_rows, in_features), x.dtype),  # halves
                pltpu.VMEM((2, tile_width, in_features), x.dtype),             # weight tiles
                pltpu.VMEM((2, 2, batch, half_rows, tile_width), x.dtype),     # output tiles
                pltpu.SemaphoreType.DMA((2,)),               # local loads, per direction
                pltpu.SemaphoreType.DMA((2, RING_SLOTS)),    # remote sends
                pltpu.SemaphoreType.DMA((2, RING_SLOTS)),    # remote receives
                pltpu.SemaphoreType.DMA((2,)),               # weight tile loads
                pltpu.SemaphoreType.DMA((2, 2)),             # output write-backs
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0,
                                             vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
    )(owner_shards, neighbors, x.reshape(batch, 2, half_rows, in_features), *weights)
    return tuple(output.reshape(batch, ring_size * rows, output.shape[-1])
                 for output in outputs)


def unsupported_reason(x_shape, weight_shapes, itemsize: int,
                       column_tile_limit: int = DEFAULT_COLUMN_TILE_LIMIT) -> str | None:
    """Why :func:`all_gather_matmul_body` cannot take these shapes, or None.
    Callers fall back to the JAX ring."""
    batch, rows, in_features = x_shape
    if rows % 2:
        return f"odd rows per device ({rows})"
    if any(shape[1] != in_features for shape in weight_shapes):
        return "weight in_features does not match the input"
    widths = [shape[0] for shape in weight_shapes]
    tile_width = _common_column_tile(widths, column_tile_limit)
    if tile_width is None:
        return f"no column tile (a multiple of 128) divides every width {sorted(set(widths))}"
    half_rows = rows // 2
    needed = itemsize * (2 * RING_SLOTS * batch * half_rows * in_features  # resident halves
                         + 2 * tile_width * in_features                    # weight tiles
                         + 2 * 2 * batch * half_rows * tile_width)         # output tiles
    needed += 4 * half_rows * tile_width                                   # one fp32 product
    if needed > VMEM_BUDGET_BYTES:
        return (f"needs {needed / 2**20:.0f} MiB of VMEM "
                f"(budget {VMEM_BUDGET_BYTES / 2**20:.0f})")
    return None


def _common_column_tile(widths, limit: int) -> int | None:
    """The largest multiple of 128 up to ``limit`` that divides every width, or None
    (q|k|v 768 + mlp 3072 per device, limit 256: 256)."""
    for tile_width in range(limit - limit % 128, 0, -128):
        if all(width % tile_width == 0 for width in widths):
            return tile_width
    return None


def ring_tables(ring_position, shard_of_position, ring_size: int):
    """Runtime owner shards and neighbours for ``ring_position`` (a traced int32).

    After h hops the device at position p holds position (p - h)'s top half (sent
    clockwise) and position (p + h)'s bottom half; ``shard_of_position`` maps a
    ring position to the shard (output row block) it owns.
    Returns (owner_shards int32 (2, ring_size), neighbors int32 (2,)).
    """
    import jax.numpy as jnp

    shard_of = jnp.asarray(shard_of_position, dtype=jnp.int32)
    hops = jnp.arange(ring_size, dtype=jnp.int32)
    owner_shards = jnp.stack([shard_of[(ring_position - hops) % ring_size],
                              shard_of[(ring_position + hops) % ring_size]])
    neighbors = jnp.stack([(ring_position + 1) % ring_size,
                           (ring_position - 1) % ring_size]).astype(jnp.int32)
    return owner_shards, neighbors
