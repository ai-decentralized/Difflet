#!/usr/bin/env python3
"""0b: the same collectives under torch_tpu, for a bit-for-bit diff.

Launched with ``torchrun --nproc_per_node=4``. Records, for every collective,
both the torch.distributed spelling and the torch.tpu.ops one where it exists,
so the migration can pick per collective rather than wholesale.
"""

import os
import traceback

import torch
import torch.distributed as dist

OUT = os.environ.get("TORCHTPU_SPIKE_OUT", "/mnt/models/tt-spikes/out")
SHAPE = (4, 8)

GROUP_RANKS = {
    "tp4": [[0, 1, 2, 3]],
    "tp2": [[0, 1], [2, 3]],
}


def payload(rank: int) -> torch.Tensor:
    base = torch.arange(SHAPE[0] * SHAPE[1], dtype=torch.float32).reshape(SHAPE)
    return base + rank * 1000.0


def attempt(res: dict, key: str, fn):
    """Record a result, or the exception — a missing API is data, not a crash."""
    try:
        res[key] = fn().cpu()
    except Exception as exc:  # noqa: BLE001 — the point is to learn what fails
        res[key] = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"  {key} -> {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Deliberately no set_device_index: the launcher hands each worker its own
    # chip, so this process sees one addressable device and any index but 0 is
    # out of bounds. An earlier version called it inside a try/except and the
    # failure went unnoticed.

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("tpu")
    res = {"rank": rank, "world": world, "local_rank": local_rank}
    print(f"[tpu] rank {rank}/{world} local={local_rank} device={device}", flush=True)

    # Every process must create every subgroup, in the same order.
    groups = {}
    for name, partition in GROUP_RANKS.items():
        made = [dist.new_group(ranks=r) for r in partition]
        groups[name] = next(
            g for g, r in zip(made, partition) if rank in r
        )

    has_tpu_ops = hasattr(torch, "tpu") and hasattr(torch.tpu, "ops")
    res["has_torch_tpu_ops"] = has_tpu_ops
    res["torch_tpu_ops_names"] = (
        sorted(n for n in dir(torch.tpu.ops) if not n.startswith("_"))
        if has_tpu_ops else []
    )

    for name, partition in GROUP_RANKS.items():
        g = groups[name]
        size = len(partition[0])

        def _all_gather():
            t = payload(rank).to(device)
            out = torch.empty((SHAPE[0] * size, SHAPE[1]), dtype=t.dtype, device=device)
            dist.all_gather_into_tensor(out, t, group=g)
            return out
        attempt(res, f"all_gather_dim0_{name}", _all_gather)

        def _all_gather_ops():
            t = payload(rank).to(device)
            return torch.tpu.ops.all_gather_dim(t, 0, group=g)
        if has_tpu_ops:
            attempt(res, f"all_gather_dim0_{name}__tpuops", _all_gather_ops)

        def _all_reduce():
            t = payload(rank).to(device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=g)
            return t
        attempt(res, f"all_reduce_sum_{name}", _all_reduce)

        def _reduce_scatter():
            t = payload(rank).to(device)
            out = torch.empty((SHAPE[0] // size, SHAPE[1]), dtype=t.dtype, device=device)
            dist.reduce_scatter_tensor(out, t, op=dist.ReduceOp.SUM, group=g)
            return out
        attempt(res, f"reduce_scatter_sum_dim0_{name}", _reduce_scatter)

        def _reduce_scatter_ops():
            t = payload(rank).to(device)
            return torch.tpu.ops.reduce_scatter_dim(t, 0, group=g)
        if has_tpu_ops:
            attempt(res, f"reduce_scatter_sum_dim0_{name}__tpuops", _reduce_scatter_ops)

    def _broadcast():
        t = payload(rank).to(device)
        dist.broadcast(t, src=0)
        return t
    attempt(res, "broadcast_src0", _broadcast)

    torch.accelerator.synchronize()
    torch.save(res, f"{OUT}_tpu_rank{rank}.pt")
    print(f"[tpu] rank {rank}/{world} wrote {OUT}_tpu_rank{rank}.pt", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
