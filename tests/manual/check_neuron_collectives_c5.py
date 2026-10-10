"""Manual check: neuron backend TP collectives on hardware, 4 ranks, eager and compiled.

Run on a trn2 host with the TorchNeuron stack while no other process holds the
NeuronCores (without --exec-mode both modes run):

    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=$PWD \
      torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_collectives_c5.py \
      --exec-mode eager

Through difflet.backends.neuron.ops_impl.collectives at Wan 2.2 A14B TP4 shapes
(4,680 video tokens, hidden 5,120, 512 text tokens), eager and under
torch.compile(backend="neuron", fullgraph=True, dynamic=False) with
fallback_execution off, so a graph that fails to lower or run raises instead of
silently running eagerly:

* bf16 all-reduce, all-gather (dims 0, 1, -1) and reduce-scatter (dims 0, 1, -1)
  are bit-exact against a CPU reference built from every rank's inputs. Inputs are
  small integers, so every sum is exact in bf16 whatever the reduction order.
* fp32 [1, 4680, 1] all-reduce (the Wan qk-norm sum of squares): exact for
  integer-valued inputs, within 1e-6 relative for real positive inputs.
* one function carrying a Wan block's 7 all-reduces (4 fp32 qk-norm, 3 bf16
  row-parallel outputs), and the Megatron-SP chain scatter, gather, reduce-scatter,
  gather.
* local scatters take this rank's chunk; no op falls back to CPU.

Rank 0 prints one PASS/FAIL line per case; every rank prints [rank r] PASS|FAIL;
the exit code is non-zero when any check fails on any rank.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("DIFFLET_BACKEND", "neuron")

import torch
import torch.distributed as dist

from difflet.backends.neuron.ops_impl import collectives as C

S, D, S_TEXT = 4680, 5120, 512
MODES = ("eager", "compile")
# With fallback execution on (the neuron backend's default), a graph that fails
# to lower or run is executed op by op instead, and the check would still pass.
NEURON_COMPILE_OPTIONS = {"fallback_execution": False}


def ints(seed, rank, shape, dtype=torch.bfloat16, low=-8, high=9):
    gen = torch.Generator().manual_seed(1000 * seed + rank)
    return torch.randint(low, high, shape, generator=gen).to(dtype)


def positive(seed, rank, shape):
    gen = torch.Generator().manual_seed(1000 * seed + rank)
    return torch.rand(shape, generator=gen, dtype=torch.float32) * 100.0 + 1.0


# One module-level function per graph, so each compiles once and is reused.
def all_reduce(t):
    return C.reduce_tp(t)


def gather_dim0(t):
    return C.gather_tp_dim(t, dim=0)


def gather_dim1(t):
    return C.gather_from_sequence_parallel_region(t, dim=1)


def gather_last(t):
    return C.gather_from_tensor_model_parallel_region_with_dim(t, -1)


def reduce_scatter_dim0(t):
    return C.reduce_scatter_to_sequence_parallel_region(t, dim=0)


def reduce_scatter_dim1(t):
    return C.reduce_scatter_to_sequence_parallel_region(t, dim=1)


def reduce_scatter_last(t):
    return C.reduce_scatter_to_sequence_parallel_region(t, dim=-1)


def scatter_dim1(t):
    return C.scatter_tp_dim(t, dim=1)


RANK_UTIL = C.SPMDRank(world_size=int(os.environ.get("WORLD_SIZE", "1")))


def scatter_spmd_last(t):
    return C.scatter_to_process_group_spmd(t, -1, RANK_UTIL.get_rank())


def wan_block_all_reduces(q1, k1, q2, k2, attn1_out, attn2_out, ffn_out):
    # Wan block order: self-attn q/k norm, self-attn out, cross-attn q/k norm,
    # cross-attn out, FFN out.
    return tuple(C.reduce_tp(t) for t in (q1, k1, attn1_out, q2, k2, attn2_out, ffn_out))


def sp_chain(t):
    shard = C.scatter_to_sequence_parallel_region(t, dim=1)
    whole = C.gather_from_sequence_parallel_region(shard, dim=1)
    part = C.reduce_scatter_to_sequence_parallel_region(whole, dim=1)
    return C.gather_tp_dim(part, dim=1)


def build_cases(rank, world):
    """(name, fn, this rank's inputs, expected outputs, rtol); rtol 0 means bit-exact."""

    def every(seed, shape, **kw):
        return [ints(seed, r, shape, **kw) for r in range(world)]

    cases = []
    x = every(1, (1, S, D))
    cases.append((f"all_reduce bf16 {[1, S, D]}", all_reduce, [x[rank]], [sum(x)], 0.0))
    sq = every(2, (1, S, 1), dtype=torch.float32, low=0, high=2**16)
    cases.append((f"all_reduce fp32 {[1, S, 1]} integer", all_reduce, [sq[rank]], [sum(sq)], 0.0))
    real = [positive(3, r, (1, S, 1)) for r in range(world)]
    real_sum = sum(t.double() for t in real).float()
    cases.append((f"all_reduce fp32 {[1, S, 1]} real", all_reduce, [real[rank]], [real_sum], 1e-6))

    g0, g1, g2 = every(4, (2, S // world, D // world)), every(5, (1, S // world, D)), every(
        6, (1, S, D // world)
    )
    cases.append(("all_gather dim 0", gather_dim0, [g0[rank]], [torch.cat(g0, 0)], 0.0))
    cases.append(("all_gather dim 1 (SP)", gather_dim1, [g1[rank]], [torch.cat(g1, 1)], 0.0))
    cases.append(("all_gather dim -1 (TP)", gather_last, [g2[rank]], [torch.cat(g2, -1)], 0.0))

    r0 = every(7, (2 * world, S // world, D // world))
    for name, fn, src, dim in (
        ("reduce_scatter dim 0", reduce_scatter_dim0, r0, 0),
        ("reduce_scatter dim 1 (SP)", reduce_scatter_dim1, x, 1),
        ("reduce_scatter dim -1", reduce_scatter_last, x, -1),
    ):
        cases.append((name, fn, [src[rank]], [sum(src).chunk(world, dim)[rank]], 0.0))

    full = ints(8, 0, (1, S, D))
    cases.append(("scatter_tp_dim dim 1", scatter_dim1, [full], [full.chunk(world, 1)[rank]], 0.0))
    cases.append(
        ("scatter_to_process_group_spmd dim -1", scatter_spmd_last, [full],
         [full.chunk(world, -1)[rank]], 0.0)
    )

    block = [
        every(9, (1, S, 1), dtype=torch.float32, low=0, high=2**16),
        every(10, (1, S, 1), dtype=torch.float32, low=0, high=2**16),
        every(11, (1, S, 1), dtype=torch.float32, low=0, high=2**16),
        every(12, (1, S_TEXT, 1), dtype=torch.float32, low=0, high=2**16),
        every(13, (1, S, D)),
        every(14, (1, S, D)),
        every(15, (1, S, D)),
    ]
    q1, k1, q2, k2, a1, a2, f = block
    order = (q1, k1, a1, q2, k2, a2, f)
    cases.append(
        ("wan block: 7 all-reduces in one graph", wan_block_all_reduces,
         [t[rank] for t in block], [sum(t) for t in order], 0.0)
    )
    cases.append(("megatron-SP chain", sp_chain, [full], [full * world], 0.0))
    return cases


def compare(out, expected, rtol):
    if out.shape != expected.shape or out.dtype != expected.dtype:
        want = f"{tuple(expected.shape)} {expected.dtype}"
        return False, f"got {tuple(out.shape)} {out.dtype}, want {want}"
    diff = (out.double() - expected.double()).abs()
    if rtol == 0.0:
        return torch.equal(out, expected), f"max_abs_err={diff.max().item():.1e}"
    rel = (diff / expected.double().abs().clamp_min(1e-30)).max().item()
    return rel <= rtol, f"max_rel_err={rel:.1e}"


def run_mode(mode, rank, world, device, backend, tracker, options=None):
    results = []
    cases = build_cases(rank, world)
    with tracker() as fallbacks:
        for name, fn, inputs, expected, rtol in cases:
            label = f"{name} ({mode})"
            try:
                call = fn
                if mode == "compile":
                    call = torch.compile(
                        fn, backend=backend, fullgraph=True, dynamic=False, options=options
                    )
                with torch.no_grad():
                    out = call(*[t.to(device) for t in inputs])
                outs = out if isinstance(out, tuple) else (out,)
                checks = [compare(o.cpu(), e, rtol) for o, e in zip(outs, expected, strict=True)]
                ok = all(c[0] for c in checks)
                detail = "; ".join(sorted({c[1] for c in checks}))
            except Exception as exc:  # report and keep going; the summary fails the run
                ok, detail = False, f"{type(exc).__name__}: {exc}"
                if rank == 0:
                    traceback.print_exc()
            results.append((label, ok, detail))
    results.append((f"no CPU fallbacks ({mode})", not fallbacks, f"fallbacks={list(fallbacks)}"))
    return results


def report(results, rank, device):
    local_failures = [name for name, ok, _ in results if not ok]
    if rank == 0:
        for name, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {detail}", flush=True)
    total = torch.tensor([float(len(local_failures))], device=device)
    dist.all_reduce(total)
    failed = int(total.item())
    print(f"[rank {rank}] {'PASS' if not local_failures else 'FAIL'}", flush=True)
    if rank == 0:
        verdict = "PASS" if failed == 0 else "FAIL"
        print(f"{verdict}  all ranks: {failed} failed check(s)", flush=True)
    return 0 if failed == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exec-mode", choices=MODES, default=None)
    args = parser.parse_args(argv)

    from difflet.backends.neuron.runtime import track_fallbacks
    from difflet.backends.registry import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    if world < 2 or not dist.is_initialized():
        print("FAIL  launch with torchrun --nproc-per-node 4", flush=True)
        return 2
    rank = dist.get_rank()
    C.init_parallel_mesh(parallel.mesh_spec)
    device = torch.device("neuron")
    results = []
    for mode in (args.exec_mode,) if args.exec_mode else MODES:
        results += run_mode(
            mode, rank, world, device, "neuron", track_fallbacks, NEURON_COMPILE_OPTIONS
        )
    code = report(results, rank, device)
    dist.destroy_process_group()
    return code


if __name__ == "__main__":
    sys.exit(main())
