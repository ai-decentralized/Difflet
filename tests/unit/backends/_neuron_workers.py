"""Worker bodies for ``_neuron_gloo.run_ranks`` (CPU gloo ranks; never the device).

Each worker is ``fn(rank, world_size, *args)`` and returns plain Python data so
the result pickles back to the parent. C4, C7 and C8 append their workers here.
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------
# C5: TP collectives and mesh
# --------------------------------------------------------------------------


def _ints(rank: int, shape, *, seed: int, dtype=torch.bfloat16, low: int = -8, high: int = 9):
    """Integer-valued tensor, distinct per rank: sums over 4 ranks are exact in bf16."""
    gen = torch.Generator().manual_seed(1000 * seed + rank)
    return torch.randint(low, high, tuple(shape), generator=gen).to(dtype)


def _recording_backend(graphs: list):
    aot_eager = torch._dynamo.lookup_backend("aot_eager")

    def backend(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    return backend


def _collective_count(gm) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and "_c10d_functional" in str(node.target)
        and "wait_tensor" not in str(node.target)
    )


def c5_collectives_worker(rank: int, world_size: int, mode: str) -> dict[str, bool]:
    """Check the neuron TP collective surface on a gloo world; returns {case: passed}.

    ``mode="compiled"`` runs every case through
    ``torch.compile(backend=aot_eager, fullgraph=True, dynamic=False)`` and also
    checks that each collective stayed in the graph.
    """
    import torch.distributed as dist

    from difflet import ops
    from difflet.backends.neuron.ops_impl import collectives as C
    from difflet.pipeline.parallel_mesh import MeshSpec

    checks: dict[str, bool] = {}
    world = world_size

    # Mesh guards with a live multi-rank process group.
    C.destroy_parallel_mesh()
    try:
        C.get_tp_size()
        checks["uninitialized mesh at world > 1 raises"] = False
    except RuntimeError:
        checks["uninitialized mesh at world > 1 raises"] = True
    try:
        C.init_parallel_mesh(MeshSpec(tp=world // 2))
        checks["spec smaller than the world is rejected"] = False
    except ValueError:
        checks["spec smaller than the world is rejected"] = not C.is_mesh_initialized()

    C.init_parallel_mesh(MeshSpec(tp=world))
    checks["ranks and sizes"] = (
        C.get_tp_size() == world
        and C.get_tp_rank() == rank
        and C.get_tensor_model_parallel_size() == world
        and C.get_tensor_model_parallel_rank() == rank
        and C.SPMDRank(world).get_rank() == rank
        and C.get_world_group().size() == world
        and C.get_cfg_group().size() == 1
        and C.get_cp_group().size() == 1
        and C.get_tp_group() is dist.group.WORLD
    )
    checks["difflet.ops dispatches here"] = (
        ops.reduce_tp is C.reduce_tp and ops.SPMDRank is C.SPMDRank
    )

    graphs: list = []

    def run(fn, *args, collectives: int | None = None):
        if mode == "eager":
            return fn(*args)
        torch._dynamo.reset()
        graphs.clear()
        out = torch.compile(fn, backend=_recording_backend(graphs), fullgraph=True, dynamic=False)(
            *args
        )
        if collectives is not None:
            found = sum(_collective_count(gm) for gm in graphs)
            checks[f"{fn.__name__}: {collectives} collective(s) in one graph"] = (
                len(graphs) == 1 and found == collectives
            )
        return out

    def every(seed, shape, **kw):
        return [_ints(r, shape, seed=seed, **kw) for r in range(world)]

    def same(out, ref):
        return out.dtype == ref.dtype and out.shape == ref.shape and torch.equal(out, ref)

    # All-reduce: bf16 activations and the Wan qk-norm fp32 [1, S, 1] sum of squares
    # (integer-valued below 2**16, so every summation order is exact).
    x = every(1, (4, 8, 12))
    mine = x[rank].clone()

    def all_reduce_bf16(t):
        return C.reduce_tp(t)

    def all_reduce_nxd_name(t):
        return C.reduce_from_tensor_model_parallel_region(t)

    checks["reduce_tp bf16"] = same(run(all_reduce_bf16, mine, collectives=1), sum(x))
    checks["reduce_tp leaves its input alone"] = torch.equal(mine, x[rank])
    checks["reduce_from_tensor_model_parallel_region"] = same(
        run(all_reduce_nxd_name, mine), sum(x)
    )
    sq = every(2, (1, 4680, 1), dtype=torch.float32, low=0, high=2**16)
    checks["reduce_tp fp32 [1, 4680, 1]"] = same(run(all_reduce_bf16, sq[rank]), sum(sq))

    # All-gather along leading, middle and last dims. The result is contiguous, as
    # torch.cat's is, so callers can .view() it.
    for dim in (0, 1, -1):

        def gather(t, dim=dim):
            return C.gather_tp_dim(t, dim=dim)

        gather.__name__ = f"gather_tp_dim_{dim}"
        out = run(gather, x[rank], collectives=1)
        checks[f"gather_tp_dim dim={dim}"] = (
            same(out, torch.cat(x, dim=dim)) and out.is_contiguous()
        )

    def gather_sp(t):
        return C.gather_from_sequence_parallel_region(t, dim=1)

    def gather_nxd_name(t):
        return C.gather_from_tensor_model_parallel_region_with_dim(t, 1)

    checks["gather_from_sequence_parallel_region"] = same(run(gather_sp, x[rank]), torch.cat(x, 1))
    checks["gather_from_tensor_model_parallel_region_with_dim"] = same(
        run(gather_nxd_name, x[rank]), torch.cat(x, 1)
    )

    # Reduce-scatter along leading, middle and last dims.
    for dim in (0, 1, -1):

        def reduce_scatter(t, dim=dim):
            return C.reduce_scatter_to_sequence_parallel_region(t, dim=dim)

        reduce_scatter.__name__ = f"reduce_scatter_{dim}"
        checks[f"reduce_scatter_to_sequence_parallel_region dim={dim}"] = same(
            run(reduce_scatter, x[rank], collectives=1), sum(x).chunk(world, dim=dim)[rank]
        )

    # Local scatters: this rank's chunk, rank taken from the mesh or SPMDRank.
    full = _ints(0, (4, 8, 12), seed=3)

    def scatter_tp(t):
        return C.scatter_tp_dim(t, dim=1)

    def scatter_sp(t):
        return C.scatter_to_sequence_parallel_region(t, dim=1)

    def scatter_nxd_default_dim(t):
        return C.scatter_to_tensor_model_parallel_region(t)

    rank_util = C.SPMDRank(world)  # built outside the graph, as Wan holds it on the model

    def scatter_spmd(t):
        return C.scatter_to_process_group_spmd(t, 1, rank_util.get_rank())

    checks["scatter_tp_dim"] = same(run(scatter_tp, full), full.chunk(world, 1)[rank])
    checks["scatter_to_sequence_parallel_region"] = same(
        run(scatter_sp, full), full.chunk(world, 1)[rank]
    )
    checks["scatter_to_tensor_model_parallel_region (dim -1)"] = same(
        run(scatter_nxd_default_dim, full), full.chunk(world, -1)[rank]
    )
    checks["scatter_to_process_group_spmd with SPMDRank"] = same(
        run(scatter_spmd, full), full.chunk(world, 1)[rank]
    )

    # Megatron-SP chain in one graph: scatter -> gather -> reduce-scatter -> gather.
    def sp_chain(t):
        shard = C.scatter_to_sequence_parallel_region(t, dim=1)
        whole = C.gather_from_sequence_parallel_region(shard, dim=1)
        part = C.reduce_scatter_to_sequence_parallel_region(whole, dim=1)
        return C.gather_tp_dim(part, dim=1)

    checks["sp chain round trip"] = same(run(sp_chain, full, collectives=3), full * world)

    # Errors are raised before any collective is issued, so no rank can hang.
    try:
        C.reduce_scatter_to_sequence_parallel_region(torch.zeros(2, 6), dim=1)
        checks["indivisible reduce-scatter raises"] = False
    except ValueError:
        checks["indivisible reduce-scatter raises"] = True
    try:
        C.gather_from_tensor_model_parallel_region_with_dim(x[rank], 0, dist.group.WORLD)
        checks["explicit multi-rank group raises"] = False
    except NotImplementedError:
        checks["explicit multi-rank group raises"] = True

    C.destroy_parallel_mesh()
    return checks
