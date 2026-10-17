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
    # An explicit multi-rank group (here the world, which is also the TP group) is gathered over.
    checks["explicit multi-rank group gathers"] = same(
        C.gather_from_tensor_model_parallel_region_with_dim(x[rank], 0, dist.group.WORLD),
        C.gather_tp_dim(x[rank], dim=0))

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


def c8_hbm_budget_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """ToyApplication at TP=world: the load's per-rank weight estimate (or the budget error every
    rank must raise together), plus the bytes the loaded module really holds."""
    import torch

    from difflet.backends.neuron.hbm import HbmBudgetError
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_toy import ToyApplication

    app = ToyApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        exec_mode="eager",
        device="cpu",
    )
    try:
        app.load(skip_warmup=True)
    except HbmBudgetError as exc:
        return {"outcome": "rejected", "message": str(exc), "weight_bytes": app.weight_bytes}
    held = sum(t.numel() * t.element_size()
               for t in (*app.module.parameters(), *app.module.buffers()))
    return {"outcome": "loaded", "weight_bytes": app.weight_bytes, "held": held}


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


# --------------------------------------------------------------------------
# C10: launch
# --------------------------------------------------------------------------


def c10_pipeline_worker(rank: int, world_size: int, exec_mode: str, work_dir: str) -> dict:
    """C10: one gloo rank through DiffletPipeline (cpu backend) into the toy lifecycle."""
    from tests.unit.backends._neuron_toy import run_toy_pipeline

    result = run_toy_pipeline(exec_mode=exec_mode, work_dir=work_dir, device="cpu")
    if (result["rank"], result["world_size"]) != (rank, world_size):
        raise AssertionError(f"worker {rank}/{world_size} got {result}")
    return result


# --------------------------------------------------------------------------
# W1: Wan DiT application and the composed neuron Wan application
# --------------------------------------------------------------------------


def _w1_transformer_app(model_dir: str, world_size: int, exec_mode: str):
    """The tiny Wan DiT (tests/unit/backends/_neuron_wan_toy.py) at TP=world, fp32, on gloo."""
    import os

    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import tiny_wan_config

    return NeuronWanTransformerApplication(
        model_path=os.path.join(model_dir, "transformer"),
        config=tiny_wan_config(tp_degree=world_size),
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        exec_mode=exec_mode,
        device="cpu",
    )


def w1_wan_lifecycle_worker(rank: int, world_size: int, model_dir: str, exec_mode: str) -> dict:
    """Load the tiny Wan DiT at TP=world and run one forward on the fixed inputs."""
    from tests.unit.backends._neuron_wan_toy import tiny_wan_inputs

    app = _w1_transformer_app(model_dir, world_size, exec_mode)
    app.load()
    out = app(*tiny_wan_inputs())
    return {
        "rank": app.rank,
        "world_size": app.world_size,
        "load_report": app.load_report,
        "warmup_shapes": app.warmup_shapes,
        "compiled_blocks": list(app.compiled_blocks),
        "unwarmed_shapes": list(app.unwarmed_shapes),
        "local_heads": app.module.blocks[0].attn1.local_heads,
        "param_numel": sum(p.numel() for p in app.module.parameters()),
        "out": out.float().tolist(),
    }


def w1_wan_truncated_worker(rank: int, world_size: int, model_dir: str, num_layers: int,
                            cp: int, exec_mode: str) -> dict:
    """The tiny Wan DiT cut to its first ``num_layers`` blocks at TP=world/cp x CP=cp, with the
    preflight's evidence hook: the sequence length the block stack hands ``norm_out``."""
    import hashlib
    import os

    from torch._dynamo.utils import counters

    from difflet.backends.neuron.wan.config import NeuronWanConfig
    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import TINY_SHAPE, TINY_TEXT_SEQ_LEN, tiny_wan_inputs

    transformer_dir = os.path.join(model_dir, "transformer")
    tp = world_size // cp
    config = NeuronWanConfig.from_pretrained(
        transformer_dir, num_layers=num_layers, tp_degree=tp, text_seq_len=TINY_TEXT_SEQ_LEN,
        context_parallel_enabled=cp > 1, cp_mode="ulysses" if cp > 1 else "gather_kv",
        torch_dtype=torch.float32, **TINY_SHAPE,
    )
    app = NeuronWanTransformerApplication(
        model_path=transformer_dir, config=config,
        parallel=DiffletParallelConfig(tp_degree=tp, cp_degree=cp,
                                       cp_mode="ulysses" if cp > 1 else "gather_kv"),
        dtype=torch.float32, exec_mode=exec_mode, device="cpu",
    )
    app.load()
    tokens: list[int] = []
    hook = app.module.norm_out.register_forward_pre_hook(
        lambda _mod, args: tokens.append(int(args[0].shape[1])))
    out = app(*tiny_wan_inputs())
    hook.remove()
    return {
        "num_blocks": len(app.module.blocks),
        "load_report": app.load_report,
        "tokens": tokens,
        "image_seq_len": config.image_seq_len,
        "graph_breaks": sum(counters["graph_break"].values()),
        "unwarmed_shapes": list(app.unwarmed_shapes),
        "out": out.float().tolist(),
        "sha256": hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest(),
    }


def w1_wan_one_graph_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """Compile mode at TP=world: how many graphs Dynamo traced for the two blocks.

    The blocks are compiled by a recording ``aot_eager`` (``_recording_backend``) so the traced
    graphs can be inspected; Dynamo's tracing and caching do not depend on the backend.
    """
    from torch._dynamo.utils import counters

    from tests.unit.backends._neuron_wan_toy import tiny_wan_inputs

    graphs: list = []
    app = _w1_transformer_app(model_dir, world_size, "compile")
    app.compile_backend = _recording_backend(graphs)
    app.load()
    after_warmup = counters["stats"]["unique_graphs"]
    latents, timestep, text = tiny_wan_inputs()
    app(latents, timestep, text)
    # The CFG pass: same shapes, other values (the unconditional embeddings).
    app(latents, timestep * 0.5, -text)
    return {
        "compiled_blocks": list(app.compiled_blocks),
        "unique_graphs_after_warmup": after_warmup,
        "unique_graphs_after_forwards": counters["stats"]["unique_graphs"],
        "graph_breaks": sum(counters["graph_break"].values()),
        "recorded_graphs": len(graphs),
        "collectives_per_graph": [_collective_count(gm) for gm in graphs],
        "unwarmed_shapes": list(app.unwarmed_shapes),
    }


def w1_fake_hidden(input_ids, text_dim: int):
    """The fake umT5's output for ``input_ids``: an arange ramp plus the token id."""
    batch, length = input_ids.shape
    ramp = torch.arange(batch * length * text_dim, dtype=torch.float32)
    return ramp.reshape(batch, length, text_dim) / 7 + input_ids[..., None].float()


class _W1FakeUMT5:
    """Stands in for umT5 on rank 0: ``w1_fake_hidden``, or a failure on demand."""

    def __init__(self, text_dim: int):
        self.text_dim = text_dim
        self.fail = False
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def __call__(self, input_ids, attention_mask):
        from types import SimpleNamespace

        self.calls.append((tuple(input_ids.shape), torch.get_num_threads()))
        if self.fail:
            raise ValueError("fake umT5 failure")
        return SimpleNamespace(last_hidden_state=w1_fake_hidden(input_ids, self.text_dim))


def w1_host_text_encoder_worker(rank: int, world_size: int) -> dict:
    """The rank-0 host text encoder on gloo: a broadcast result, a failing encode, a failing load.

    Every outcome comes back as a string, so a rank left waiting in a collective shows up as
    run_ranks' timeout rather than as a result.
    """
    from difflet.backends.neuron.core.distributed import RankFailureError
    from difflet.models.wan.neuron_application import _HostTextEncoder

    def attempt(fn):
        try:
            out = fn()
        except RankFailureError as exc:
            return {"outcome": f"RankFailureError: {exc}"}
        except Exception as exc:  # noqa: BLE001 - reported to the parent by type and message
            return {"outcome": f"{type(exc).__name__}: {exc}"}
        if out is None:
            return {"outcome": "ok"}
        return {"outcome": "ok", "dtype": str(out.dtype), "embeds": out.float().tolist()}

    text_dim, seq_len = 24, 8
    fake = _W1FakeUMT5(text_dim)
    encoder = _HostTextEncoder(
        "/nonexistent-w1-model", text_dim=text_dim, seq_len=seq_len, dtype=torch.bfloat16,
        device="cpu", threads=2,
    )
    if rank == 0:
        encoder.model = fake  # what load() leaves on rank 0; the real umT5 path is the e2e test's
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
    threads_before = torch.get_num_threads()
    result = {"ok": attempt(lambda: encoder(input_ids, attention_mask))}
    result["threads_restored"] = torch.get_num_threads() == threads_before
    fake.fail = True
    result["failed"] = attempt(lambda: encoder(input_ids, attention_mask))
    fake.fail = False
    result["recovered"] = attempt(lambda: encoder(input_ids, attention_mask))
    result["calls"] = fake.calls
    missing = _HostTextEncoder(
        "/nonexistent-w1-model", text_dim=text_dim, seq_len=seq_len, dtype=torch.bfloat16,
        device="cpu", threads=2,
    )
    result["load"] = attempt(missing.load)
    result["loaded_model"] = missing.model is not None
    return result


def w1_wan_pipeline_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """The composed TorchNeuronWanApplication: two steps at guidance 5.0 on the fixed inputs."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
    )
    app.load()
    steps: list[int] = []
    forward_seconds: list[float] = []
    encoder_inputs: list[tuple[int, ...]] = []
    negative_abs_sums: list[float] = []
    # Each stamp records how many forwards ran before it: 2 per denoise iteration at CFG.
    app.step_hook = lambda: steps.append(len(forward_seconds))
    app.transformer_adapter.forward_hook = forward_seconds.append
    if app.text_encoder.model is not None:  # rank 0 only
        app.text_encoder.model.register_forward_hook(
            lambda module, args, output: encoder_inputs.append(tuple(args[0].shape))
        )
    negative_prompt_embeds = app.pipeline._negative_prompt_embeds

    def recording_negative(prompt_embeds):
        out = negative_prompt_embeds(prompt_embeds)
        negative_abs_sums.append(float(out.abs().sum()))
        return out

    app.pipeline._negative_prompt_embeds = recording_negative
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(
        prompt_embeds=prompt_embeds,
        latents=latents,
        channels=TINY_LATENT_SHAPE[1],  # the orchestrator defaults to Wan's 16
        num_inference_steps=2,
        guidance_scale=5.0,
        output_type="latent",
    )
    final = out.latents
    return {
        "dtype": str(final.dtype),
        "latents": final.tolist(),
        "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest(),
        "steps": steps,
        "forward_seconds": forward_seconds,
        "encoder_inputs": encoder_inputs,
        "negative_abs_sums": negative_abs_sums,
        "phase_seconds": sorted(app.phase_seconds),
        "unwarmed_shapes": list(app.unwarmed_shapes),
    }


def w1_wan22_pipeline_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """The composed application with both Wan 2.2 experts: four steps at guidance 5.0 above the
    boundary and 3.0 below it, on the fixed inputs."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
        enable_transformer_2=True,
    )
    app.load()
    forward_seconds: list[float] = []
    app.transformer_adapter.forward_hook = forward_seconds.append  # times both experts
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(
        prompt_embeds=prompt_embeds,
        latents=latents,
        channels=TINY_LATENT_SHAPE[1],
        num_inference_steps=4,
        guidance_scale=5.0,
        guidance_scale_2=3.0,
        output_type="latent",
    )
    final = out.latents
    return {
        "latents": final.tolist(),
        "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest(),
        "expert_calls": app.expert_calls,
        "forwards_timed": len(forward_seconds),
        "phase_seconds": sorted(app.phase_seconds),
        "unwarmed_shapes": [list(shape) for shape in app.unwarmed_shapes],
    }


def w1_wan_device_text_encoder_worker(rank: int, world_size: int, model_dir: str,
                                      prompts: list) -> dict:
    """The composed application with umT5 on the (CPU) device, sharded at TP4: the embeddings
    the orchestrator gets for each prompt."""
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import TINY_SHAPE, TINY_TEXT_SEQ_LEN

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        text_encoder_device="neuron",
    )
    app.load()
    embeds = [app.pipeline.encode_prompt(prompt=prompt) for prompt in prompts]
    return {
        "embeds": [e.tolist() for e in embeds],
        "phase_seconds": sorted(app.phase_seconds),
        "unwarmed_shapes": [list(shape) for shape in app.unwarmed_shapes],
    }


def w1_wan_device_vae_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """The composed application with the device VAE: the two-step run decoded to frames."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
        enable_vae_decoder=True,
    )
    app.load()
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(
        prompt_embeds=prompt_embeds,
        latents=latents,
        channels=TINY_LATENT_SHAPE[1],
        num_inference_steps=2,
        guidance_scale=5.0,
        output_type="pt",
    )
    frames = out.frames
    return {
        "frames": frames.tolist(),
        "latents": out.latents.tolist(),
        "sha256": hashlib.sha256(frames.contiguous().numpy().tobytes()).hexdigest(),
        "phase_seconds": sorted(app.phase_seconds),
    }


def cfg_mesh_worker(rank: int, world_size: int) -> dict:
    """TP2 x CFG2 on a gloo world: each rank's group memberships and both collectives."""
    import torch.distributed as dist

    from difflet.backends.neuron.ops_impl import collectives as C
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.pipeline.parallel_mesh import MeshSpec

    C.init_parallel_mesh(MeshSpec(cfg=2, tp=2))
    tp_group, cfg_group = pm.get_tp_group(), C.get_cfg_group()
    x = torch.full((2, 3), float(rank + 1))
    reduced = C.reduce_from_tensor_model_parallel_region(x)  # sums over the TP group
    gathered = C.gather_from_tensor_model_parallel_region_with_dim(reduced, 0, process_group=cfg_group)
    branch = C.scatter_to_process_group_spmd(
        torch.arange(4.0).reshape(2, 2), 0, C.get_cfg_rank_spmd(C.SPMDRank(world_size).get_rank()),
        process_group=cfg_group)
    return {
        "tp_ranks": dist.get_process_group_ranks(tp_group),
        "cfg_ranks": dist.get_process_group_ranks(cfg_group),
        "tp_rank": pm.get_tp_rank(), "cfg_rank": pm.get_axis_rank("cfg"),
        "gathered_col0": gathered[:, 0].tolist(), "branch": branch.tolist(),
    }


def w1_wan_cfg_parallel_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """The composed application at TP2 x CFG2: two steps at guidance 5.0 on the fixed inputs."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size // 2, cfg_parallel_enabled=True),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
    )
    app.load()
    calls: list[float] = []
    app.transformer_adapter.forward_hook = calls.append
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(prompt_embeds=prompt_embeds, latents=latents, channels=TINY_LATENT_SHAPE[1],
              num_inference_steps=2, guidance_scale=5.0, output_type="latent")
    final = out.latents
    return {
        "latents": final.tolist(),
        "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest(),
        "dit_calls": len(calls),
        "unwarmed_shapes": [list(s) for s in app.unwarmed_shapes],
    }


def ulysses_worker(rank: int, world_size: int) -> dict:
    """TP2 x CP2: Ulysses attention on this rank's head block and sequence block, against plain
    attention over the full sequence; plus the two layout swaps as a round trip."""
    import torch.nn.functional as F

    from difflet.backends.neuron.ops_impl import attention as A
    from difflet.backends.neuron.ops_impl import collectives as C
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.pipeline.parallel_mesh import MeshSpec

    C.init_parallel_mesh(MeshSpec(cp=2, tp=2))
    tp, cp = pm.get_tp_rank(), pm.get_axis_rank("cp")
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, 8, 16, generator=gen) for _ in range(3))  # [B, H, S, d]
    heads, rows = slice(2 * tp, 2 * tp + 2), slice(4 * cp, 4 * cp + 4)
    local = [t[:, heads, rows].contiguous() for t in (q, k, v)]
    out = A.ulysses_attention(*local, scale=0.25)
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.25)[:, heads, rows]
    group = C.get_cp_group()
    back = C.heads_to_seq(C.seq_to_heads(local[0], group, 2), group, 2)
    return {"max_err": (out - ref).abs().max().item(), "shape": list(out.shape),
            "round_trip_exact": torch.equal(back, local[0])}


def w1_wan_initial_noise_worker(rank: int, world_size: int, model_dir: str) -> dict:
    """The composed application's initial latents when every rank's RNG differs: the global
    RNG, ``seed=``, and a per-rank generator. No DiT and no encoder are loaded, so the
    orchestrator returns the initial latents as its final ones."""
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import TINY_LATENT_SHAPE, TINY_SHAPE, TINY_TEXT_SEQ_LEN

    torch.manual_seed(100 + rank)
    app = TorchNeuronWanApplication(
        model_path=model_dir, parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager", device="cpu", enable_transformer=False, enable_text_encoder=False,
    )
    app.load()
    common = {"channels": TINY_LATENT_SHAPE[1], "output_type": "latent"}
    return {
        "global_rng": app(**common).latents.tolist(),
        "seed": app(seed=7, **common).latents.tolist(),
        "generator": app(generator=torch.Generator().manual_seed(7 + rank), **common)
        .latents.tolist(),
    }


def w1_wan_cp_worker(rank: int, world_size: int, model_dir: str, cp_mode: str) -> dict:
    """The composed application at TP2 x CP2: two steps at guidance 5.0 on the fixed inputs."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size // 2, cp_degree=2, cp_mode=cp_mode),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
    )
    app.load()
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(prompt_embeds=prompt_embeds, latents=latents, channels=TINY_LATENT_SHAPE[1],
              num_inference_steps=2, guidance_scale=5.0, output_type="latent")
    final = out.latents
    return {"latents": final.tolist(),
            "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest()}


def ring_worker(rank: int, world_size: int) -> dict:
    """TP2 x CP2: ring attention on this rank's head block and sequence block, against plain
    attention over the full sequence; plus one ring shift."""
    import torch.nn.functional as F

    from difflet.backends.neuron.ops_impl import attention as A
    from difflet.backends.neuron.ops_impl import collectives as C
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.pipeline.parallel_mesh import MeshSpec

    C.init_parallel_mesh(MeshSpec(cp=2, tp=2))
    tp, cp = pm.get_tp_rank(), pm.get_axis_rank("cp")
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 4, 8, 16, generator=gen) for _ in range(3))  # [B, H, S, d]
    heads, rows = slice(2 * tp, 2 * tp + 2), slice(4 * cp, 4 * cp + 4)
    local = [t[:, heads, rows].contiguous() for t in (q, k, v)]
    out = A.ring_attention(*local, scale=0.25)
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.25)[:, heads, rows]
    shifted = C.ring_shift(torch.full((2,), float(rank)), C.get_cp_group(), 2, cp)
    return {"max_err": (out - ref).abs().max().item(), "shape": list(out.shape),
            "received": shifted.tolist()}


def w1_wan_sp_worker(rank: int, world_size: int, model_dir: str, cfg: bool) -> dict:
    """The composed application with sequence parallelism: TP4 + SP, or TP2 x CFG2 + SP; two
    steps at guidance 5.0 on the fixed inputs."""
    import hashlib

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_wan_toy import (
        TINY_LATENT_SHAPE,
        TINY_SHAPE,
        TINY_TEXT_SEQ_LEN,
        tiny_wan_pipeline_inputs,
    )

    app = TorchNeuronWanApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=world_size // (2 if cfg else 1),
                                       cfg_parallel_enabled=cfg, sp_enabled=True),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode="eager",
        device="cpu",
        encoder_threads=2,
    )
    app.load()
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    out = app(prompt_embeds=prompt_embeds, latents=latents, channels=TINY_LATENT_SHAPE[1],
              num_inference_steps=2, guidance_scale=5.0, output_type="latent")
    final = out.latents
    return {"latents": final.tolist(),
            "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest()}


def dp_replica_claims_worker(rank: int, world_size: int, requests_dir: str, fail_index: int) -> dict:
    """A 2-rank DP replica draining a queue: rank 0 claims, both ranks run each request; a
    request at ``fail_index`` raises on every rank."""
    import argparse
    from pathlib import Path

    from difflet.cli.orchestrators.wan import _replica_request_scope, _replica_requests

    args = argparse.Namespace(requests_dir=Path(requests_dir), worker_index=0,
                              dp_schedule="round_robin")
    seen, error = [], None
    try:
        for req in _replica_requests(args, device=torch.device("cpu")):
            with _replica_request_scope(args, req):
                if req.index == fail_index:
                    raise RuntimeError("boom")
                seen.append((req.index, req.prompt))
    except RuntimeError as exc:
        error = str(exc)
    return {"seen": seen, "error": error}


# --------------------------------------------------------------------------
# Joint (text + image) context-parallel attention: joint_ring_attention and
# joint_ulysses_attention against one dense softmax over the full joint sequence
# --------------------------------------------------------------------------

JOINT_HEADS, JOINT_S_TXT, JOINT_S_IMG, JOINT_D = 4, 6, 16, 16


def _joint_reference(q, k, v, *, scale: float, valid: int | None = None):
    """fp64 softmax attention on ``[B, H, S, d]``; keys at index >= ``valid`` masked."""
    scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    if valid is not None:
        scores[..., valid:] = float("-inf")
    return torch.softmax(scores, dim=-1) @ v.double()


def joint_cp_worker(rank: int, world_size: int, tp: int, mode: str) -> dict:
    """TP ``tp`` x CP ``world_size // tp``: both joint CP attentions on this rank's head block
    (TP) and image sequence block (CP), text replicated, against a dense softmax over the full
    ``[text ‖ image]`` sequence; returns max errors and output shapes.

    ``mode="compiled"`` runs each op through ``torch.compile(backend="aot_eager",
    fullgraph=True, dynamic=False)``: the per-block compile the device path uses must trace
    the ring loop and the all-to-alls without a graph break.
    """
    from types import SimpleNamespace

    from difflet.backends.neuron.ops_impl import attention as A
    from difflet.backends.neuron.ops_impl import collectives as C
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm

    C.init_parallel_mesh(SimpleNamespace(tp_degree=tp, context_parallel_enabled=True))
    cp = pm.get_axis_size("cp")
    assert cp == world_size // tp
    tp_rank, cp_rank = pm.get_tp_rank(), pm.get_axis_rank("cp")
    scale = 0.25
    gen = torch.Generator().manual_seed(7)
    q_t, k_t, v_t = (torch.randn(1, JOINT_HEADS, JOINT_S_TXT, JOINT_D, generator=gen)
                     for _ in range(3))
    q_i, k_i, v_i = (torch.randn(1, JOINT_HEADS, JOINT_S_IMG, JOINT_D, generator=gen)
                     for _ in range(3))
    hl, rows = JOINT_HEADS // tp, JOINT_S_IMG // cp
    heads = slice(hl * tp_rank, hl * (tp_rank + 1))
    img_rows = slice(rows * cp_rank, rows * (cp_rank + 1))

    def txt(t):  # replicated over CP: this TP rank's heads, every text row
        return t[:, heads].contiguous()

    def img(t):  # sharded over CP: this TP rank's heads, this CP rank's image rows
        return t[:, heads, img_rows].contiguous()

    ring, ulysses = A.joint_ring_attention, A.joint_ulysses_attention
    if mode == "compiled":
        torch._dynamo.reset()
        ring = torch.compile(ring, backend="aot_eager", fullgraph=True, dynamic=False)
        ulysses = torch.compile(ulysses, backend="aot_eager", fullgraph=True, dynamic=False)

    # Ring, in Qwen's calling convention: joint query [text ‖ this rank's image rows].
    ref = _joint_reference(torch.cat([q_t, q_i], 2), torch.cat([k_t, k_i], 2),
                           torch.cat([v_t, v_i], 2), scale=scale)[:, heads]
    ref_txt, ref_img = ref[:, :, :JOINT_S_TXT], ref[:, :, JOINT_S_TXT:][:, :, img_rows]
    ring_out = ring(torch.cat([txt(q_t), img(q_i)], 2), img(k_i), img(v_i), txt(k_t), txt(v_t),
                    scale=scale, causal=False)

    # Ulysses: image and text streams apart, each returned in the sharding it came in.
    img_out, txt_out = ulysses(img(q_i), txt(q_t), img(k_i), img(v_i), txt(k_t), txt(v_t),
                               scale=scale, causal=False)

    # Ulysses with a key-padding count in the joint [image ‖ text] order: the last two text
    # keys are padding (HunyuanVideo's contract).
    valid = JOINT_S_IMG + JOINT_S_TXT - 2
    masked = _joint_reference(torch.cat([q_i, q_t], 2), torch.cat([k_i, k_t], 2),
                              torch.cat([v_i, v_t], 2), scale=scale, valid=valid)[:, heads]
    img_masked, txt_masked = ulysses(
        img(q_i), txt(q_t), img(k_i), img(v_i), txt(k_t), txt(v_t), scale=scale, causal=False,
        key_valid_len=torch.tensor([valid], dtype=torch.int32))

    # One head per rank cannot split over cp >= 2 ranks; raised before any collective.
    try:
        A.joint_ulysses_attention(*(t[:, :1] for t in (img(q_i), txt(q_t), img(k_i), img(v_i),
                                                        txt(k_t), txt(v_t))), scale=scale)
        head_error = ""
    except ValueError as exc:
        head_error = str(exc)

    def err(out, expected):
        return (out.double() - expected).abs().max().item()

    return {
        "cp": cp,
        "ring_shape": list(ring_out.shape),
        "ring_err": err(ring_out, torch.cat([ref_txt, ref_img], 2)),
        "ulysses_shapes": [list(img_out.shape), list(txt_out.shape)],
        "ulysses_img_err": err(img_out, ref_img),
        "ulysses_txt_err": err(txt_out, ref_txt),
        "masked_img_err": err(img_masked, masked[:, :, :JOINT_S_IMG][:, :, img_rows]),
        "masked_txt_err": err(txt_masked, masked[:, :, JOINT_S_IMG:]),
        "head_error": head_error,
    }
