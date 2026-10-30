"""Device probe: process subgroups on the neuron backend, eager and torch.compile.

    torchrun --standalone --nproc-per-node 4 tests/manual/probe_neuron_subgroups.py

Builds AoT's mesh layout for TP2 x CFG2 (rank = cfg * tp_size + tp): TP groups {0,1} {2,3},
CFG groups {0,2} {1,3}, each with ``dist.new_group`` (every rank creates every group, in the
same order). On each group it runs the functional collectives the TP and CFG paths use
(all_reduce, all_gather_single on dim 0) on a bf16 tensor whose values encode the rank, in eager
mode and inside ``torch.compile(backend="neuron", fullgraph=True)``, and checks the exact
expected result. Prints PASS/FAIL per check on rank 0 and exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys

# One NeuronCore per process, before torch is imported (torch autoloads torch_neuronx), as
# Difflet's neuron runtime does.
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", os.environ.get("LOCAL_RANK", "0"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as fc  # noqa: E402


def main() -> int:
    import torch_neuronx  # noqa: F401  registers the neuron device and its collectives

    dist.init_process_group("neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "run with --nproc-per-node 4"
    device = torch.device("neuron", int(os.environ.get("LOCAL_RANK", "0")))  # this rank's visible core (global index)

    tp_ranks = [[0, 1], [2, 3]]
    cfg_ranks = [[0, 2], [1, 3]]
    tp_groups = [dist.new_group(r) for r in tp_ranks]
    cfg_groups = [dist.new_group(r) for r in cfg_ranks]
    tp_group = tp_groups[rank // 2]
    cfg_group = cfg_groups[rank % 2]
    tp_members = tp_ranks[rank // 2]
    cfg_members = cfg_ranks[rank % 2]

    x = torch.full((4, 8), float(rank + 1), dtype=torch.bfloat16).to(device)

    def reduce_then_gather(t, reduce_group, gather_group):
        reduced = fc.wait_tensor(fc.all_reduce(t, "sum", reduce_group))
        return fc.wait_tensor(fc.all_gather_tensor(reduced, 0, gather_group))

    compiled = torch.compile(reduce_then_gather, backend="neuron", fullgraph=True, dynamic=False)

    results = []
    for mode, fn in (("eager", reduce_then_gather), ("compile", compiled)):
        for name, red, gat, red_members, gat_members in (
            ("tp reduce + cfg gather", tp_group, cfg_group, tp_members, cfg_members),
            ("cfg reduce + tp gather", cfg_group, tp_group, cfg_members, tp_members),
        ):
            with torch.no_grad():
                out = fn(x, red, gat).float().cpu()
            # all_reduce over red_members sums (r+1); the gather stacks the reduced value of each
            # member of the gather group, each of which reduced over ITS reduce group.
            expected_blocks = []
            for member in gat_members:
                if red is tp_group:
                    peers = tp_ranks[member // 2]
                else:
                    peers = cfg_ranks[member % 2]
                expected_blocks.append(torch.full((4, 8), float(sum(p + 1 for p in peers))))
            expected = torch.cat(expected_blocks, 0)
            ok = torch.equal(out, expected)
            results.append((mode, name, ok, out[:, 0].tolist()))

    all_ok = all(ok for _, _, ok, _ in results)
    for mode, name, ok, col in results:
        print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} {mode:7s} {name}: column0={col}", flush=True)
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
