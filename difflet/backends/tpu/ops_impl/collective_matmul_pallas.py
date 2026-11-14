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
                           interpret=False, profile_ablation: str = "",
                           resident_weights: bool = False, head_dims=None):
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
    resident_weights: load every weight tile into VMEM once (all loads started
                  up front, waited on in the first hop) instead of streaming the
                  weights again in every hop. For few rows, where the streamed
                  weights -- ring_size times their bytes -- cost more than the
                  matmuls (FLUX's 512-token text: 64-row halves at ~63 TFLOP/s,
                  trace 2026-11-12). Needs the weights' bytes in VMEM.
    head_dims:    None, or one int per weight: a nonzero ``d`` writes that weight's
                  output heads first, (batch, width // d, ring_size * rows, d) --
                  each column tile's d-wide slices go to their head's rows -- so an
                  attention reading [B, H, S, D] needs no relayout of it. Written
                  as that plain 4-D array (row offsets aligned to 16 rows): handed
                  over as a 6-D (.., ring, 2, rows/2, d) array reshaped, XLA reduced
                  the 64-row text stream's RMSNorm in another order (1-2 float32
                  ulps; FLUX's output 41 dB off, norm_layout_probe 2026-11-13).

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
    head_dims = tuple(int(d) for d in head_dims) if head_dims else (0,) * num_weights
    if len(head_dims) != num_weights:
        raise ValueError(f"head_dims {head_dims} must name one head size per weight")
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
    for width, head_dim in zip(output_widths, head_dims):
        if head_dim and (width % head_dim or tile_width % head_dim):
            raise ValueError(f"head size {head_dim} must divide the width {width} and "
                             f"the column tile {tile_width}")
    # Heads-first outputs are written into a plain (batch, H, S, d) array at a
    # runtime row offset, which must sit on a whole (16, 128) bf16 tile.
    plain_heads = half_rows % 16 == 0
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
            # A wait needs only a descriptor of the tile's size (heads-first
            # outputs write a tile as several head-wide copies on one semaphore).
            for direction in range(2):
                pltpu.make_async_copy(output_tiles.at[buffer_slot, direction],
                                      output_tiles.at[buffer_slot, direction],
                                      output_semaphores.at[buffer_slot, direction]).wait()

        def ring_slot(hop):
            return 0 if skip_communication else hop % RING_SLOTS

        if resident_weights:  # every tile's load in flight now, each into its own slot
            for weight_index in range(num_weights):
                for column_tile in range(output_widths[weight_index] // tile_width):
                    slot = first_tile_of_weight[weight_index] + column_tile
                    pltpu.make_async_copy(
                        weight_refs[weight_index].at[pl.ds(column_tile * tile_width, tile_width), :],
                        weight_tiles.at[slot], weight_semaphores.at[slot]).start()
        else:  # the first weight tile load overlaps the local halves' load
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
                if resident_weights:
                    weight_slot = tile_in_hop
                    if hop == 0:  # loaded once; later hops reuse the slot
                        wait_weight_tile_load(weight_slot)
                else:
                    weight_slot = buffer_slot

                    @pl.when(tile + 1 < total_tiles)
                    def _():
                        start_weight_tile_load(tile + 1, 1 - buffer_slot)  # always one in flight

                    wait_weight_tile_load(buffer_slot)

                @pl.when(tile >= 2)
                def _():
                    wait_output_write_back(buffer_slot)  # this slot's previous tile is out

                if not skip_compute:
                    weight_tile = weight_tiles[weight_slot]
                    for direction in range(2):
                        for batch_index in range(batch):
                            output_tiles[buffer_slot, direction, batch_index] = jax.lax.dot_general(
                                halves[direction, ring_slot(hop), batch_index], weight_tile,
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32).astype(output_tiles.dtype)
                weight_index, column_tile = locate_tile(tile)
                for i in range(num_weights):
                    @pl.when(weight_index == i)
                    def _(i=i):
                        for direction in range(2):
                            owner = owner_shards_ref[direction, hop]
                            semaphore = output_semaphores.at[buffer_slot, direction]
                            head_dim = head_dims[i]
                            if not head_dim:
                                pltpu.make_async_copy(
                                    output_tiles.at[buffer_slot, direction],
                                    output_refs[i].at[:, owner, direction, :,
                                                      columns_of(column_tile)],
                                    semaphore).start()
                                continue
                            heads_per_tile = tile_width // head_dim
                            row = pl.multiple_of((owner * 2 + direction) * half_rows, 16) \
                                if plain_heads else None
                            for j in range(heads_per_tile):
                                head = column_tile * heads_per_tile + j
                                destination = (
                                    output_refs[i].at[:, head, pl.ds(row, half_rows), :]
                                    if plain_heads else
                                    output_refs[i].at[:, head, owner, direction, :, :])
                                pltpu.make_async_copy(
                                    output_tiles.at[buffer_slot, direction, :, :,
                                                    pl.ds(j * head_dim, head_dim)],
                                    destination, semaphore).start()
                return carry

            jax.lax.fori_loop(0, tiles_per_hop, process_tile, 0)
        for copy in remote_copies:
            copy.wait_send()
            copy.wait_recv()
        for tile in range(max(0, total_tiles - 2), total_tiles):  # the last write-backs
            wait_output_write_back(tile % 2)

    weight_slots = tiles_per_hop if resident_weights else 2
    any_memory = pl.BlockSpec(memory_space=pl.ANY)
    outputs = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct(
            ((batch, width // head_dim, ring_size * rows, head_dim) if plain_heads
             else (batch, width // head_dim, ring_size, 2, half_rows, head_dim)) if head_dim
            else (batch, ring_size, 2, half_rows, width), x.dtype)
            for width, head_dim in zip(output_widths, head_dims)],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[any_memory] * (1 + num_weights),
            out_specs=[any_memory] * num_weights,
            scratch_shapes=[
                pltpu.VMEM((2, RING_SLOTS, batch, half_rows, in_features), x.dtype),  # halves
                pltpu.VMEM((weight_slots, tile_width, in_features), x.dtype),  # weight tiles
                pltpu.VMEM((2, 2, batch, half_rows, tile_width), x.dtype),     # output tiles
                pltpu.SemaphoreType.DMA((2,)),               # local loads, per direction
                pltpu.SemaphoreType.DMA((2, RING_SLOTS)),    # remote sends
                pltpu.SemaphoreType.DMA((2, RING_SLOTS)),    # remote receives
                pltpu.SemaphoreType.DMA((weight_slots,)),    # weight tile loads
                pltpu.SemaphoreType.DMA((2, 2)),             # output write-backs
            ],
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0,
                                             vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
    )(owner_shards, neighbors, x.reshape(batch, 2, half_rows, in_features), *weights)
    return tuple(output.reshape(batch, output.shape[1], ring_size * rows, head_dim) if head_dim
                 else output.reshape(batch, ring_size * rows, output.shape[-1])
                 for output, head_dim in zip(outputs, head_dims))


def unsupported_reason(x_shape, weight_shapes, itemsize: int,
                       column_tile_limit: int = DEFAULT_COLUMN_TILE_LIMIT,
                       resident_weights: bool = False) -> str | None:
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
    needed = all_gather_vmem_bytes(x_shape, widths, itemsize, tile_width, resident_weights)
    if needed > VMEM_BUDGET_BYTES:
        return (f"needs {needed / 2**20:.0f} MiB of VMEM "
                f"(budget {VMEM_BUDGET_BYTES / 2**20:.0f})")
    return None


def all_gather_vmem_bytes(x_shape, widths, itemsize: int, tile_width: int,
                          resident_weights: bool = False) -> int:
    """VMEM :func:`all_gather_matmul_body` needs: its scratch (the travelling halves,
    the weight tiles, the double-buffered output tiles) and one fp32 product."""
    batch, rows, in_features = x_shape
    half_rows = rows // 2
    weight_rows = sum(widths) if resident_weights else 2 * tile_width
    needed = itemsize * (2 * RING_SLOTS * batch * half_rows * in_features  # resident halves
                         + weight_rows * in_features                       # weight tiles
                         + 2 * 2 * batch * half_rows * tile_width)         # output tiles
    return needed + 4 * half_rows * tile_width                             # one fp32 product


def column_tiles(widths, limit: int):
    """Every multiple of 128 up to ``limit`` that divides every width, largest first."""
    return [t for t in range(limit - limit % 128, 0, -128) if all(w % t == 0 for w in widths)]


def _common_column_tile(widths, limit: int) -> int | None:
    """The largest multiple of 128 up to ``limit`` that divides every width, or None
    (q|k|v 768 + mlp 3072 per device, limit 256: 256)."""
    return next(iter(column_tiles(widths, limit)), None)


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


# ---------------------------------------------------------------------------
# Row-parallel matmul + reduce-scatter
# ---------------------------------------------------------------------------

#: Receive slots per direction for the travelling accumulators. Accumulators move
#: tile by tile, so a neighbour can run up to two hops ahead in one direction: it
#: writes a slot again only three hops later, after it has received this device's
#: other-direction tiles of the hop in between, which this device sends only after
#: it has read the slot. The CPU test's race detector catches two slots.
REDUCE_SCATTER_SLOTS = 3

#: Row chunks are DMA'd straight from the (batch, sequence, in_features) input when
#: their first row is a multiple of this (Mosaic only slices tiled dims in whole
#: tiles); otherwise the input is reshaped to (batch, ring_size, rows, ...) first.
_ROW_TILE = 8


def matmul_reduce_scatter_body(xs, weights, chunk_shards, neighbors, *, ring_size: int,
                               axis: str = "tp",
                               column_tile_limit: int = DEFAULT_COLUMN_TILE_LIMIT,
                               vmem_limit_bytes: int = VMEM_BUDGET_BYTES,
                               interpret=False, profile_ablation: str = ""):
    """Per-device row-parallel matmul over the (x_i, weight_i) pairs, summed, then
    reduce-scattered along the ring by sequence.

    The same bidirectional ring as ``collective_matmul``'s JAX ``rs`` op: the
    output's left half of the columns accumulates clockwise, the right half
    counter-clockwise. At each hop a device multiplies one row chunk of its
    inputs by its weights (the partial sum for the chunk's owner), adds the
    accumulator that arrived from upstream and sends the sum on; after
    ``ring_size`` hops each device holds the full sum for its own rows.

    - Tile by tile: each output column tile is multiplied, the upstream
      accumulator's same tile is added (it was sent about a hop earlier, so it
      has arrived) and the sum is sent on at once. The add runs in the matmul's
      epilogue, the transfers overlap the following tiles' matmuls.
    - Accumulators travel VMEM to VMEM; the last hop writes its sums straight to
      the output. No partial product, accumulator or concatenation goes to HBM.
    - The weights stream through VMEM one column tile at a time, one
      double-buffered stream over every (hop, direction, column tile); the input
      row chunks of the next hop load while the current hop computes.
    - Rounding as the JAX ring: each pair's product is rounded to the input
      dtype, the pairs are summed in the input dtype, then the accumulator.

    xs:           sequence of (batch, ring_size * rows, in_features_i): the full
                  sequence, this device's slice of the contracted features.
    weights:      sequence of (out_features, in_features_i), one per x; the same
                  out_features, even, its half divisible by one column tile (a
                  multiple of 128, at most ``column_tile_limit``).
    chunk_shards: int32 (2, ring_size) -- chunk_shards[0, hop] / [1, hop] is the
                  shard (row chunk) whose left / right half of the columns this
                  device computes at ``hop``. From :func:`reduce_scatter_tables`.
    neighbors:    int32 (2,) -- mesh index along ``axis`` of the clockwise
                  neighbour (receives the left halves) and the counter-clockwise one.
    profile_ablation: profiling only, wrong values -- ``"communication"`` skips
                  the transfers and the adds, ``"compute"`` skips the matmuls.

    Returns (batch, rows, out_features): this device's rows of the sum.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if profile_ablation not in ("", "communication", "compute"):
        raise ValueError(f"profile_ablation must be '', 'communication' or 'compute', "
                         f"got {profile_ablation!r}")
    num_pairs = len(xs)
    if num_pairs != len(weights) or num_pairs == 0:
        raise ValueError(f"need one weight per input, got {num_pairs} inputs and "
                         f"{len(weights)} weights")
    if ring_size < 2:
        raise ValueError(f"ring_size must be at least 2, got {ring_size}")
    batch, sequence, _ = xs[0].shape
    if sequence % ring_size:
        raise ValueError(f"sequence {sequence} does not split into ring_size={ring_size} chunks")
    rows = sequence // ring_size
    out_features = weights[0].shape[0]
    dtype = xs[0].dtype
    in_features = []
    for x, weight in zip(xs, weights):
        if x.shape[:2] != (batch, sequence) or x.dtype != dtype:
            raise ValueError(f"inputs differ in batch, sequence or dtype: {x.shape} {x.dtype}")
        if weight.shape != (out_features, x.shape[2]):
            raise ValueError(f"weight {weight.shape} does not match input {x.shape} "
                             f"and out_features={out_features}")
        in_features.append(x.shape[2])
    if out_features % 2:
        raise ValueError(f"out_features must be even (two ring halves), got {out_features}")
    half_columns = out_features // 2
    tile_width = _common_column_tile([half_columns], column_tile_limit)
    if tile_width is None:
        raise ValueError(f"no column tile (a multiple of 128) divides half the columns "
                         f"({half_columns})")
    tiles_per_direction = half_columns // tile_width
    tiles_per_hop = 2 * tiles_per_direction
    total_tiles = ring_size * tiles_per_hop
    slots = REDUCE_SCATTER_SLOTS
    aligned_rows = rows % _ROW_TILE == 0
    skip_communication = profile_ablation == "communication"
    skip_compute = profile_ablation == "compute"

    def kernel(chunk_shards_ref, neighbors_ref, *refs):
        x_refs = refs[:num_pairs]
        weight_refs = refs[num_pairs:2 * num_pairs]
        output_ref = refs[2 * num_pairs]
        scratch = refs[2 * num_pairs + 1:]
        x_chunks = scratch[:num_pairs]
        weight_tiles = scratch[num_pairs:2 * num_pairs]
        (partial_sums, received, chunk_semaphores, weight_semaphores, send_semaphores,
         receive_semaphores, output_semaphores) = scratch[2 * num_pairs:]

        # Both neighbours must be inside this kernel (their VMEM slots belong to
        # it) before anything is copied into them.
        barrier = pltpu.get_barrier_semaphore()
        for direction in range(2):
            pl.semaphore_signal(barrier, 1, device_id={axis: neighbors_ref[direction]})
        pl.semaphore_wait(barrier, 2)

        def columns_of(column_tile):
            return pl.ds(pl.multiple_of(column_tile * tile_width, tile_width), tile_width)

        def start_chunk_loads(hop, parity):
            copies = []
            for pair in range(num_pairs):
                for direction in range(2):
                    shard = chunk_shards_ref[direction, hop]
                    source = (x_refs[pair].at[:, pl.ds(pl.multiple_of(shard * rows, _ROW_TILE),
                                                       rows)]
                              if aligned_rows else x_refs[pair].at[:, shard])
                    copy = pltpu.make_async_copy(source, x_chunks[pair].at[direction, parity],
                                                 chunk_semaphores.at[pair, direction, parity])
                    copy.start()
                    copies.append(copy)
            return copies

        def start_weight_tile_load(tile, buffer_slot):
            # Tile u of a hop covers output columns [u * tile_width, (u + 1) * tile_width):
            # the left half's tiles first, then the right half's.
            column_tile = jnp.asarray(tile, jnp.int32) % tiles_per_hop
            for pair in range(num_pairs):
                pltpu.make_async_copy(weight_refs[pair].at[columns_of(column_tile), :],
                                      weight_tiles[pair].at[buffer_slot],
                                      weight_semaphores.at[pair, buffer_slot]).start()

        def wait_weight_tile_load(buffer_slot):
            for pair in range(num_pairs):
                pltpu.make_async_copy(weight_refs[pair].at[columns_of(0), :],
                                      weight_tiles[pair].at[buffer_slot],
                                      weight_semaphores.at[pair, buffer_slot]).wait()

        def tile_transfer(direction, column_tile, slot):
            """The descriptor of one accumulator tile into the neighbour's ``slot``
            (also used to wait for the same tile arriving here)."""
            columns = columns_of(column_tile)
            return pltpu.make_async_remote_copy(
                partial_sums.at[direction, :, :, columns],
                received.at[direction, slot, :, :, columns],
                send_semaphores.at[direction, column_tile],
                receive_semaphores.at[direction, slot, column_tile],
                device_id={axis: neighbors_ref[direction]})

        def output_write(direction, column_tile):
            columns = columns_of(column_tile)
            output_columns = pl.ds(
                pl.multiple_of(direction * half_columns + column_tile * tile_width, tile_width),
                tile_width)
            return pltpu.make_async_copy(partial_sums.at[direction, :, :, columns],
                                         output_ref.at[:, :, output_columns],
                                         output_semaphores.at[direction, column_tile])

        chunk_loads = start_chunk_loads(0, 0)
        start_weight_tile_load(0, 0)
        for hop in range(ring_size):
            parity = hop % 2
            for copy in chunk_loads:
                copy.wait()
            chunk_loads = start_chunk_loads(hop + 1, 1 - parity) if hop + 1 < ring_size else []

            def process_tile(tile_in_hop, carry, hop=hop, parity=parity):
                tile = hop * tiles_per_hop + tile_in_hop
                buffer_slot = tile % 2

                @pl.when(tile + 1 < total_tiles)
                def _():
                    start_weight_tile_load(tile + 1, 1 - buffer_slot)  # always one in flight

                wait_weight_tile_load(buffer_slot)
                direction = tile_in_hop // tiles_per_direction
                column_tile = tile_in_hop % tiles_per_direction
                columns = columns_of(column_tile)
                sums = []
                for batch_index in range(batch):
                    total = None
                    if not skip_compute:
                        for pair in range(num_pairs):
                            product = jax.lax.dot_general(
                                x_chunks[pair][direction, parity, batch_index],
                                weight_tiles[pair][buffer_slot],
                                (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32).astype(dtype)
                            total = product if total is None else (
                                total.astype(jnp.float32)
                                + product.astype(jnp.float32)).astype(dtype)
                    sums.append(total)
                communicate = not skip_communication
                if hop > 0 and communicate:
                    # This tile of the upstream accumulator (sent about a hop ago).
                    tile_transfer(direction, column_tile, hop % slots).wait_recv()
                    # The previous hop's send of this tile has left partial_sums.
                    tile_transfer(direction, column_tile, hop % slots).wait_send()
                for batch_index in range(batch):
                    total = sums[batch_index]
                    if hop > 0 and communicate:
                        upstream = received[direction, hop % slots, batch_index, :, columns]
                        total = upstream if total is None else (
                            total.astype(jnp.float32) + upstream.astype(jnp.float32)).astype(dtype)
                    if total is not None:
                        partial_sums[direction, batch_index, :, columns] = total
                if hop + 1 < ring_size:
                    if communicate:
                        tile_transfer(direction, column_tile, (hop + 1) % slots).start()
                else:  # the last hop holds this device's rows of the sum
                    output_write(direction, column_tile).start()
                return carry

            jax.lax.fori_loop(0, tiles_per_hop, process_tile, 0)

        def finish(tile_in_hop, carry):
            direction = tile_in_hop // tiles_per_direction
            column_tile = tile_in_hop % tiles_per_direction
            output_write(direction, column_tile).wait()
            return carry

        jax.lax.fori_loop(0, tiles_per_hop, finish, 0)

    any_memory = pl.BlockSpec(memory_space=pl.ANY)
    scratch_shapes = (
        [pltpu.VMEM((2, 2, batch, rows, k), dtype) for k in in_features]        # row chunks
        + [pltpu.VMEM((2, tile_width, k), dtype) for k in in_features]          # weight tiles
        + [pltpu.VMEM((2, batch, rows, half_columns), dtype),                   # partial sums
           pltpu.VMEM((2, slots, batch, rows, half_columns), dtype),            # received
           pltpu.SemaphoreType.DMA((num_pairs, 2, 2)),                  # row chunk loads
           pltpu.SemaphoreType.DMA((num_pairs, 2)),                     # weight tile loads
           pltpu.SemaphoreType.DMA((2, tiles_per_direction)),           # remote sends
           pltpu.SemaphoreType.DMA((2, slots, tiles_per_direction)),    # remote receives
           pltpu.SemaphoreType.DMA((2, tiles_per_direction))])          # output writes
    inputs = (xs if aligned_rows
              else [x.reshape(batch, ring_size, rows, x.shape[2]) for x in xs])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, rows, out_features), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[any_memory] * (2 * num_pairs),
            out_specs=any_memory,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=0,
                                             vmem_limit_bytes=vmem_limit_bytes),
        interpret=interpret,
    )(chunk_shards, neighbors, *inputs, *weights)


def reduce_scatter_unsupported_reason(x_shapes, weight_shapes, itemsize: int, ring_size: int,
                                      column_tile_limit: int = DEFAULT_COLUMN_TILE_LIMIT
                                      ) -> str | None:
    """Why :func:`matmul_reduce_scatter_body` cannot take these shapes, or None.
    Callers fall back to the JAX ring."""
    batch, sequence, _ = x_shapes[0]
    if sequence % ring_size:
        return f"sequence {sequence} does not split into {ring_size} chunks"
    rows = sequence // ring_size
    out_features = weight_shapes[0][0]
    if out_features % 2:
        return f"odd out_features ({out_features})"
    tile_width = _common_column_tile([out_features // 2], column_tile_limit)
    if tile_width is None:
        return f"no column tile (a multiple of 128) divides half the columns ({out_features // 2})"
    in_features = sum(shape[2] for shape in x_shapes)
    half_columns = out_features // 2
    needed = itemsize * (2 * 2 * batch * rows * in_features                  # row chunks
                         + 2 * tile_width * in_features                      # weight tiles
                         + (1 + REDUCE_SCATTER_SLOTS) * 2 * batch * rows * half_columns)
    needed += 4 * rows * tile_width                                          # one fp32 product
    if needed > VMEM_BUDGET_BYTES:
        return (f"needs {needed / 2**20:.0f} MiB of VMEM "
                f"(budget {VMEM_BUDGET_BYTES / 2**20:.0f})")
    return None


def reduce_scatter_tables(ring_position, shard_of_position, ring_size: int):
    """Runtime chunk shards and neighbours for ``ring_position`` (a traced int32).

    At hop h the device at position p computes the left half of the columns for
    position (p - 1 - h)'s rows (that accumulator travels clockwise) and the right
    half for position (p + 1 + h)'s; at the last hop both are its own position.
    Returns (chunk_shards int32 (2, ring_size), neighbors int32 (2,)).
    """
    import jax.numpy as jnp

    shard_of = jnp.asarray(shard_of_position, dtype=jnp.int32)
    hops = jnp.arange(ring_size, dtype=jnp.int32)
    chunk_shards = jnp.stack([shard_of[(ring_position - 1 - hops) % ring_size],
                              shard_of[(ring_position + 1 + hops) % ring_size]])
    neighbors = jnp.stack([(ring_position + 1) % ring_size,
                           (ring_position - 1) % ring_size]).astype(jnp.int32)
    return chunk_shards, neighbors
