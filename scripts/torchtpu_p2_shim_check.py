#!/usr/bin/env python3
"""Phase 2 check: the shim's torch_tpu side, against the torch_xla results.

Runs the same deterministic payload as the 0b spike, but through
``device_runtime`` rather than the runtime APIs directly, so what is verified
is the shim — the group-list -> ProcessGroup translation, the dim handling in
all_gather / reduce_scatter, and broadcast over a list.

Launch: torchtpu_launch.py 4 p2_shim.py
"""

import os

import torch

from difflet.backends.tpu.ops_impl import device_runtime as D

OUT = os.environ.get("TORCHTPU_SPIKE_OUT", "/mnt/models/tt-spikes/out")
SHAPE = (4, 8)
GROUPS = {"tp4": [[0, 1, 2, 3]], "tp2": [[0, 1], [2, 3]]}


def payload(rank: int) -> torch.Tensor:
    base = torch.arange(SHAPE[0] * SHAPE[1], dtype=torch.float32).reshape(SHAPE)
    return base + rank * 1000.0


def main():
    D.initialize_replica(0, 4)          # torch_tpu: init_process_group("tpu_dist")
    rank, world = D.rank(), D.world_size()
    dev = D.device()
    res = {"rank": rank, "world": world, "runtime": D.runtime_name(),
           "device_kind": D.device_kind()}
    print(f"[{D.runtime_name()}] rank {rank}/{world} on {dev} ({res['device_kind']})", flush=True)

    for name, groups in GROUPS.items():
        size = len(groups[0])
        res[f"all_gather_dim0_{name}"] = D.all_gather(
            payload(rank).to(dev), dim=0, groups=groups).cpu()
        res[f"all_reduce_sum_{name}"] = D.all_reduce_sum(
            payload(rank).to(dev), groups=groups).cpu()
        res[f"reduce_scatter_sum_dim0_{name}"] = D.reduce_scatter_sum(
            payload(rank).to(dev), dim=0, shard_count=size, groups=groups).cpu()

    t = payload(rank).to(dev)
    D.broadcast([t], src=0)
    res["broadcast_src0"] = t.cpu()

    D.sync()
    mem = D.memory_info()
    res["mem_has_peak"] = "peak_bytes_used" in mem
    torch.save(res, f"{OUT}_shim_rank{rank}.pt")
    print(f"[{D.runtime_name()}] rank {rank} wrote; mem_keys={sorted(mem)}", flush=True)


if __name__ == "__main__":
    main()
