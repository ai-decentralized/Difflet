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


# --------------------------------------------------------------------------
# C4: TP linear and embedding
# --------------------------------------------------------------------------


def c4_tp_mlp_worker(rank: int, world_size: int, compiled: bool) -> dict:
    """C4: neuron TP linear and embedding at TP=world_size against TP1 on gloo.

    ``compiled`` wraps every module in ``torch.compile(backend="aot_eager",
    fullgraph=True)``. Everything is compared inside the worker and only plain
    Python values come back (bools, floats, hex digests), so the result does not
    depend on how ``run_ranks`` transports tensors.
    """
    import hashlib

    import torch.distributed as dist
    import torch.nn.functional as F

    from difflet.backends.neuron.ops_impl import linear as L
    from difflet.backends.neuron.ops_impl.parallel_mesh import init_parallel_mesh
    from difflet.backends.tpu.core.weights import load_sharded_state_dict
    from difflet.pipeline.parallel_mesh import MeshSpec
    from tests.unit.backends._neuron_toy import ToyTPMLP, reference_mlp, toy_full_weights

    res: dict = {}
    # A multi-rank process without a mesh must refuse to size a layer, not guess tp=1.
    try:
        L.ColumnParallelLinear(8, 16)
    except RuntimeError:
        res["uninitialized_mesh_raises"] = True
    else:
        res["uninitialized_mesh_raises"] = False

    init_parallel_mesh(MeshSpec(tp=world_size))

    def load(module, full):
        load_sharded_state_dict(module, full, tp_size=world_size, tp_rank=rank)
        return module.eval()

    def run(module, *args):
        if compiled:
            module = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=False)
        with torch.no_grad():
            return module(*args)

    # Toy column -> GELU(tanh) -> row MLP on fp32 random data, against TP1.
    dim, hidden = 64, 256
    weights = toy_full_weights(dim, hidden, seed=0)
    x = torch.randn(2, 9, dim, generator=torch.Generator().manual_seed(1))
    mlp = load(ToyTPMLP(dim, hidden), weights)
    res["shapes_ok"] = (
        tuple(mlp.up.weight.shape) == (hidden // world_size, dim)
        and tuple(mlp.up.bias.shape) == (hidden // world_size,)
        and tuple(mlp.down.weight.shape) == (dim, hidden // world_size)
        and tuple(mlp.down.bias.shape) == (dim,)
    )
    got = run(mlp, x)
    res["mlp_max_abs_err"] = (got - reference_mlp(x, weights)).abs().max().item()
    res["mlp_digest"] = hashlib.sha256(got.contiguous().numpy().tobytes()).hexdigest()

    # Ternary {-1, 0, 1} data keeps every partial sum a small integer, so the
    # sharded result must equal TP1 bit for bit whatever the reduction order.
    gen = torch.Generator().manual_seed(2)

    def ternary(*shape):
        return torch.randint(-1, 2, shape, generator=gen).float()

    cw, cb, xc = ternary(128, 64), ternary(128), ternary(3, 5, 64)
    col = load(L.ColumnParallelLinear(64, 128, gather_output=True), {"weight": cw, "bias": cb})
    res["column_gather_exact"] = torch.equal(run(col, xc), F.linear(xc, cw, cb))

    rw, rb, xr = ternary(64, 128), ternary(64), ternary(3, 5, 128)
    full_row = F.linear(xr, rw) + rb
    row = load(L.RowParallelLinear(128, 64, input_is_parallel=False), {"weight": rw, "bias": rb})
    res["row_scatter_exact"] = torch.equal(run(row, xr), full_row)

    skip = load(
        L.RowParallelLinear(128, 64, input_is_parallel=False, skip_bias_add=True),
        {"weight": rw, "bias": rb},
    )
    out, bias = run(skip, xr)
    res["row_skip_bias_add_exact"] = torch.equal(out, F.linear(xr, rw)) and torch.equal(bias, rb)

    # reduce_output=False: this rank's partial plus the FULL bias (NxD semantics,
    # corrected by modeling_wan._sp_unbias). Summing the partials outside the layer
    # and removing the (tp - 1) extra biases must give the full output.
    partial_row = load(
        L.RowParallelLinear(128, 64, input_is_parallel=True, reduce_output=False),
        {"weight": rw, "bias": rb},
    )
    x_shard = xr.chunk(world_size, dim=-1)[rank]
    partial = run(partial_row, x_shard)
    total = partial.clone()
    dist.all_reduce(total)
    res["row_partial_plus_bias_exact"] = torch.equal(
        partial, F.linear(x_shard, rw.chunk(world_size, dim=1)[rank]) + rb
    ) and torch.equal(total - (world_size - 1) * rb, full_row)

    # Embeddings are a lookup plus zeros, or a gather, so exact for any values.
    ew = torch.randn(256, 32, generator=gen)
    ids = torch.randint(0, 256, (2, 7), generator=gen)
    vocab = load(L.ParallelEmbedding(256, 32), {"weight": ew})
    res["embedding_vocab_exact"] = torch.equal(run(vocab, ids), F.embedding(ids, ew))
    by_dim = load(L.ParallelEmbedding(256, 32, shard_across_embedding=True), {"weight": ew})
    res["embedding_dim_exact"] = torch.equal(run(by_dim, ids), F.embedding(ids, ew))
    return res


# ---------------------------------------------------------------- C7: checkpoint loader


def _proc_status_kib(field: str) -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def _reset_peak_rss() -> bool:
    """Reset VmHWM to the current RSS (Linux >= 4.0); False where /proc forbids it."""
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def c7_checkpoint_worker(rank, world_size, ckpt_dir, dim, hidden, dtype_name):
    """Load a toy TP MLP with tp size/rank from the mesh; check shards, reassembly, forward."""
    import torch

    from difflet.backends.neuron.core.checkpoint import (
        build_on_meta,
        load_sharded_checkpoint,
        shard_dim,
    )
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.backends.neuron.ops_impl.collectives import gather_tp_dim, get_tp_rank
    from difflet.pipeline.parallel_mesh import MeshSpec
    from tests.unit.backends._neuron_toy import ToyTPMLP, reference_mlp, toy_full_weights

    dtype = getattr(torch, dtype_name)
    pm.init_parallel_mesh(MeshSpec(tp=world_size))
    full = toy_full_weights(dim, hidden, seed=0)
    model = build_on_meta(lambda: ToyTPMLP(dim, hidden))
    report = load_sharded_checkpoint(model, ckpt_dir, device="cpu", dtype=dtype)
    state = model.state_dict()
    shards_exact = reassembled_exact = True
    for name, tensor in full.items():  # same order on every rank: collectives must match
        axis = shard_dim(model, name)
        whole = state[name] if axis is None else gather_tp_dim(state[name], dim=axis)
        expected = tensor if axis is None else tensor.chunk(world_size, dim=axis)[rank]
        shard_ok = state[name].dtype == dtype and torch.equal(state[name], expected.to(dtype))
        shards_exact = shards_exact and shard_ok
        reassembled_exact = reassembled_exact and torch.equal(whole, tensor.to(dtype))
    forward_err = None
    if dtype == torch.float32:
        x = torch.randn(2, 8, dim, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            ref = reference_mlp(x, full)
            err = (model(x) - ref).abs().max() / ref.abs().max().clamp_min(1.0)
        forward_err = err.item()
    return {
        "rank": rank,
        "tp_rank": get_tp_rank(),
        "missing": report["missing"],
        "unexpected": report["unexpected"],
        "shards_exact": bool(shards_exact),
        "reassembled_exact": bool(reassembled_exact),
        "forward_err": forward_err,
    }


def c7_peak_rss_worker(rank, world_size, ckpt_dir, n_layers, dim, hidden):
    """Peak host RSS of one rank's bf16 load, measured from a VmHWM reset just before it."""
    import gc

    import safetensors.torch  # noqa: F401  import cost stays outside the measured window
    import torch
    import torch.nn as nn

    from difflet.backends.neuron.core.checkpoint import build_on_meta, load_sharded_checkpoint
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.pipeline.parallel_mesh import MeshSpec
    from tests.unit.backends._neuron_toy import ToyTPMLP

    pm.init_parallel_mesh(MeshSpec(tp=world_size))
    model = build_on_meta(lambda: nn.ModuleList(ToyTPMLP(dim, hidden) for _ in range(n_layers)))
    gc.collect()
    if not _reset_peak_rss():
        return {"rank": rank, "peak_delta": None, "module_bytes": 0, "missing": []}
    before = _proc_status_kib("VmRSS")
    report = load_sharded_checkpoint(model, ckpt_dir, device="cpu", dtype=torch.bfloat16)
    peak = _proc_status_kib("VmHWM")
    module_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "rank": rank,
        "peak_delta": (peak - before) * 1024,
        "module_bytes": module_bytes,
        "missing": report["missing"],
    }


# ----------------------------------------------------------------------------- C8 workers


def c8_distributed_worker(rank: int, world_size: int) -> dict:
    """Rank-0 I/O primitives on gloo: status sync, phases, rank0_call, broadcast_tensor."""
    import math

    import torch

    from difflet.backends.neuron.core import distributed as nd

    device = torch.device("cpu")
    out: dict = {"world": nd.world_info(), "is_rank0": nd.is_rank0()}
    nd.sync_status(True, device=device, what="all healthy")

    try:
        with nd.collective_phase("rank-2 phase", device=device):
            if rank == 2:
                raise ValueError("boom on rank 2")
        out["phase"] = "completed"
    except nd.RankFailureError as exc:
        out["phase"] = f"RankFailureError: {exc}"
    except ValueError as exc:
        out["phase"] = f"ValueError: {exc}"

    out["rank0_call"] = nd.rank0_call(lambda a, *, b: a + b, 40, b=2, device=device, what="add")

    mismatches = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64):
        for shape in ((2, 3), (), (0, 4)):
            expected = (torch.arange(math.prod(shape)) + 1).reshape(shape).to(dtype)
            got = nd.broadcast_tensor(expected if rank == 0 else None, device=device)
            if got.dtype != dtype or tuple(got.shape) != shape or not torch.equal(got, expected):
                mismatches.append((str(dtype), shape, str(got.dtype), tuple(got.shape)))
    out["broadcast_mismatches"] = mismatches

    try:
        nd.broadcast_tensor(None, device=device)
        out["bad_root"] = "completed"
    except nd.RankFailureError:
        out["bad_root"] = "RankFailureError"
    except TypeError:
        out["bad_root"] = "TypeError"

    # The neuron rule (int64 crosses the process group as int32) applied on gloo: rank 0's
    # out-of-range payload must fail every rank, and in-range edge values still go through.
    narrows_int64 = nd._narrows_int64
    nd._narrows_int64 = lambda device: True
    try:
        try:
            too_big = torch.tensor([1, 2**31], dtype=torch.int64)
            nd.broadcast_tensor(too_big if rank == 0 else None, device=device)
            out["int64_out_of_range"] = "completed"
        except nd.RankFailureError as exc:
            out["int64_out_of_range"] = f"RankFailureError: {exc}"
        except ValueError as exc:
            out["int64_out_of_range"] = f"ValueError: {exc}"
        edges = torch.tensor([[-(2**31)], [2**31 - 1]], dtype=torch.int64)
        scalar = torch.tensor(-5, dtype=torch.int64)
        wire: list[tuple[str, tuple[int, ...]]] = []
        broadcast = nd.dist.broadcast

        def recording_broadcast(tensor, *args, **kwargs):
            wire.append((str(tensor.dtype), tuple(tensor.shape)))
            return broadcast(tensor, *args, **kwargs)

        nd.dist.broadcast = recording_broadcast
        try:
            edges = nd.broadcast_tensor(edges if rank == 0 else None, device=device)
            scalar = nd.broadcast_tensor(scalar if rank == 0 else None, device=device)
        finally:
            nd.dist.broadcast = broadcast
        # int32 and flat on the wire; int64 in the original shape on arrival
        out["int64_edges"] = (str(edges.dtype), edges.tolist())
        out["int64_scalar"] = (str(scalar.dtype), tuple(scalar.shape), scalar.item())
        out["int64_wire"] = wire  # [header, payload] per broadcast
    finally:
        nd._narrows_int64 = narrows_int64

    # A mismatched collective count above would hang here until run_ranks times out.
    nd.sync_status(True, device=device, what="still in lockstep")
    out["final"] = "ok"
    return out


def c8_lifecycle_worker(
    rank: int, world_size: int, model_dir: str, exec_mode: str, dtype_name: str
) -> dict:
    """Full ToyApplication lifecycle at TP=world on gloo; returns plain Python data."""
    import torch

    from difflet.backends.neuron.core.distributed import broadcast_tensor
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_toy import ToyApplication, toy_blocks_input

    dtype = getattr(torch, dtype_name)
    app = ToyApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=dtype,
        exec_mode=exec_mode,
        device="cpu",
    )
    app.load()
    x = toy_blocks_input(app.batch, app.seq, app.dim, dtype=dtype) if rank == 0 else None
    x = broadcast_tensor(x, device=app.device)
    with torch.no_grad():
        out = app(x)
    return {
        "rank": app.rank,
        "world_size": app.world_size,
        "is_loaded": app.is_loaded,
        "compiled_blocks": list(app.compiled_blocks),
        "warmup_shapes": app.warmup_shapes,
        "param_numel": sum(p.numel() for p in app.module.parameters()),
        "param_shapes": {name: tuple(p.shape) for name, p in app.module.named_parameters()},
        "out": out.float().tolist(),
    }


def c8_failure_worker(rank: int, world_size: int, model_dir: str, inject: str) -> dict:
    """Rank 0 fails before a collective; every rank must return instead of hanging."""
    import torch

    from difflet.backends.neuron.core.distributed import RankFailureError
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_toy import ToyApplication

    app = ToyApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        exec_mode="eager",
        device="cpu",
        inject_failure=inject,
    )
    try:
        app.load()
    except RankFailureError as exc:
        return {"outcome": "rank_failure", "message": str(exc), "is_loaded": app.is_loaded}
    except RuntimeError as exc:
        return {"outcome": "raised", "message": str(exc), "is_loaded": app.is_loaded}
    return {"outcome": "loaded", "message": "", "is_loaded": app.is_loaded}


def c8_forward_failure_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """FORWARD_FAILURE_RANK raises inside block 0 of load()'s warm-up forward.

    Nothing is caught: the failing rank's error leaves the worker, as it would leave a torchrun
    rank, while its peers stay blocked in block 0's MLP all-reduce until run_ranks kills them.
    """
    import torch

    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_toy import ToyApplication

    app = ToyApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        exec_mode="eager",
        device="cpu",
        inject_failure="forward",
    )
    app.load()
    return {"rank": rank, "is_loaded": app.is_loaded}
