r"""Manual check: neuron tensor-parallel linear and embedding on hardware (C4).

Run on a trn2 host with the TorchNeuron stack, once per execution mode:

    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=$PWD \
        torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_linear_c4.py \
        --exec-mode eager

Every case runs in bf16 on the device at TP=world. Compile mode is
``torch.compile(backend="neuron", fullgraph=True, dynamic=False)`` with
``fallback_execution`` off, so a graph that fails to lower or run raises instead of
silently running eagerly.

* dispatch: difflet.ops resolves the three layers to the neuron implementations.
* mlp: the toy column -> GELU(tanh) -> row MLP against the same MLP at TP1 (full
  weights, no collectives) on the same core, both measured against a CPU fp32
  reference on the same bf16 values. TP's mean error may be at most 3x TP1's plus
  1e-3; a wrong shard is off by O(0.1). The all-reduced output must be
  bit-identical on every rank.
* column_gather, row_scatter, row_partial: ternary {-1, 0, 1} weights and inputs
  keep every partial sum a small integer, so the result must equal CPU exactly.
* embedding_vocab, embedding_dim: lookups, exact.
* fallbacks: no op fell back to CPU.

Rank 0 prints one PASS/FAIL line per case; every rank prints [rank r] PASS|FAIL
and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from difflet import ops
from difflet.backends.neuron.ops_impl import linear as L
from difflet.backends.neuron.ops_impl.parallel_mesh import init_parallel_mesh
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.backends.registry import get_backend
from difflet.backends.tpu.core.weights import load_sharded_state_dict
from difflet.pipeline.parallel_config import DiffletParallelConfig
from tests.unit.backends._neuron_toy import ToyTPMLP, reference_mlp, toy_full_weights

DEVICE, DTYPE = "neuron", torch.bfloat16
DIM, HIDDEN, TOKENS = 512, 2048, 256
VOCAB, EMB_DIM = 1024, 256
# With fallback execution on (the neuron backend's default), a graph that fails
# to lower or run is executed op by op instead, and the check would still pass.
NEURON_COMPILE_OPTIONS = {"fallback_execution": False}


def build(module, full, *, mode, world, rank):
    """Load this rank's shards on the host, move to the device, compile if asked."""
    load_sharded_state_dict(module, full, tp_size=world, tp_rank=rank)
    module = module.eval().to(DEVICE)
    if mode == "compile":
        module = torch.compile(
            module,
            backend="neuron",
            fullgraph=True,
            dynamic=False,
            options=NEURON_COMPILE_OPTIONS,
        )
    return module


def exact(got: torch.Tensor, want: torch.Tensor) -> tuple[bool, str]:
    if got.dtype == want.dtype and torch.equal(got, want):
        return True, "exact"
    return False, f"max_abs_diff={(got.float() - want.float()).abs().max().item():.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 device check: TP linear and embedding")
    parser.add_argument("--exec-mode", choices=("eager", "compile"), default="eager")
    mode = parser.parse_args().exec_mode

    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    init_parallel_mesh(parallel.mesh_spec)
    rank = dist.get_rank() if dist.is_initialized() else 0
    kw = {"mode": mode, "world": world, "rank": rank}

    # Same seed on every rank: identical inputs and full weights everywhere.
    gen = torch.Generator().manual_seed(0)

    def ternary(*shape):
        return torch.randint(-1, 2, shape, generator=gen).to(DTYPE)

    weights = {k: v.to(DTYPE) for k, v in toy_full_weights(DIM, HIDDEN, seed=0).items()}
    x = torch.randn(1, TOKENS, DIM, generator=gen).to(DTYPE)
    ref32 = reference_mlp(x.float(), {k: v.float() for k, v in weights.items()})
    cw, cb, xc = ternary(128, 64), ternary(128), ternary(2, 16, 64)
    rw, rb, xr = ternary(64, 128), ternary(64), ternary(2, 16, 128)
    # Contiguous on the host: .to("neuron") keeps a chunk view's strides, and a
    # strided input makes aten::linear restride it through the CPU (reported as
    # aten::linear[prologue_cpu_roundtrip]), which would blame the layer for the
    # harness's input layout. The layers themselves are fed contiguous tensors.
    x_shard = xr.chunk(world, dim=-1)[rank].contiguous()
    rw_shard = rw.chunk(world, dim=1)[rank]
    ew = torch.randn(VOCAB, EMB_DIM, generator=gen).to(DTYPE)
    ids = torch.randint(0, VOCAB, (2, 16), generator=gen)

    with track_fallbacks() as fallbacks, torch.no_grad():
        mlp = build(ToyTPMLP(DIM, HIDDEN, dtype=DTYPE), weights, **kw)
        x_dev = x.to(DEVICE)
        tp_out = mlp(x_dev)
        tp1_out = reference_mlp(x_dev, {k: v.to(DEVICE) for k, v in weights.items()})
        flat = tp_out.reshape(-1).contiguous()
        gathered = torch.empty(world * flat.numel(), dtype=DTYPE, device=DEVICE)
        if world > 1:
            dist.all_gather_into_tensor(gathered, flat)
        else:
            gathered.copy_(flat)

        col = build(
            L.ColumnParallelLinear(64, 128, gather_output=True, dtype=DTYPE),
            {"weight": cw, "bias": cb},
            **kw,
        )
        col_out = col(xc.to(DEVICE))
        row = build(
            L.RowParallelLinear(128, 64, input_is_parallel=False, dtype=DTYPE),
            {"weight": rw, "bias": rb},
            **kw,
        )
        row_out = row(xr.to(DEVICE))
        part = build(
            L.RowParallelLinear(
                128, 64, input_is_parallel=True, reduce_output=False, dtype=DTYPE
            ),
            {"weight": rw, "bias": rb},
            **kw,
        )
        part_out = part(x_shard.to(DEVICE))
        vocab = build(L.ParallelEmbedding(VOCAB, EMB_DIM, dtype=DTYPE), {"weight": ew}, **kw)
        vocab_out = vocab(ids.to(DEVICE))
        by_dim = build(
            L.ParallelEmbedding(VOCAB, EMB_DIM, shard_across_embedding=True, dtype=DTYPE),
            {"weight": ew},
            **kw,
        )
        dim_out = by_dim(ids.to(DEVICE))

    tp_c, tp1_c = tp_out.float().cpu(), tp1_out.float().cpu()
    err_tp = (tp_c - ref32).abs().mean().item()
    err_tp1 = (tp1_c - ref32).abs().mean().item()
    max_diff = (tp_c - tp1_c).abs().max().item()
    per_rank = gathered.cpu().view(world, -1)
    identical = all(torch.equal(per_rank[0], per_rank[i]) for i in range(1, world))

    results = {
        "dispatch": (
            ops.ColumnParallelLinear is L.ColumnParallelLinear
            and ops.RowParallelLinear is L.RowParallelLinear
            and ops.ParallelEmbedding is L.ParallelEmbedding,
            "difflet.ops -> difflet.backends.neuron.ops_impl.linear",
        ),
        "mlp": (
            err_tp <= 3.0 * err_tp1 + 1e-3 and identical,
            f"mean_err tp{world}={err_tp:.2e} tp1={err_tp1:.2e} "
            f"max|tp{world}-tp1|={max_diff:.2e} ranks_identical={identical}",
        ),
        "column_gather": exact(
            col_out.float().cpu(), F.linear(xc.float(), cw.float(), cb.float())
        ),
        "row_scatter": exact(
            row_out.float().cpu(), F.linear(xr.float(), rw.float()) + rb.float()
        ),
        "row_partial": exact(
            part_out.float().cpu(), F.linear(x_shard.float(), rw_shard.float()) + rb.float()
        ),
        "embedding_vocab": exact(vocab_out.cpu(), F.embedding(ids, ew)),
        "embedding_dim": exact(dim_out.cpu(), F.embedding(ids, ew)),
        "fallbacks": (not fallbacks, str(sorted(set(fallbacks)))),
    }

    ok = all(passed for passed, _ in results.values())
    if rank == 0:
        for name, (passed, detail) in results.items():
            print(f"{'PASS' if passed else 'FAIL'} {name} [{mode}] {detail}", flush=True)
    failed = [name for name, (passed, _) in results.items() if not passed]
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL ' + ','.join(failed)}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
