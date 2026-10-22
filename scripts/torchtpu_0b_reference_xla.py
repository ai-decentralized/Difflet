#!/usr/bin/env python3
"""0b reference run: collective results on the shipped torch_xla path.

Writes one .pt per rank so the torch_tpu run can be diffed against it
bit-for-bit. Deliberately calls the same xm.* entry points, with the same
``groups=``/``pin_layout=False`` arguments, that
difflet/backends/tpu/ops_impl/collectives.py uses in production.
"""

import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

OUT = os.environ.get("TORCHTPU_SPIKE_OUT", "/mnt/models/tt-spikes/out")
SHAPE = (4, 8)

GROUPS = {
    "tp4": [[0, 1, 2, 3]],
    "tp2": [[0, 1], [2, 3]],
}


def payload(rank: int) -> torch.Tensor:
    """Deterministic, rank-distinct, exactly representable in fp32."""
    base = torch.arange(SHAPE[0] * SHAPE[1], dtype=torch.float32).reshape(SHAPE)
    return base + rank * 1000.0


def run(index):
    device = torch_xla.device()
    rank = int(xr.global_ordinal())
    world = int(xr.world_size())
    res = {"rank": rank, "world": world}

    for name, groups in GROUPS.items():
        size = len(groups[0])

        t = payload(rank).to(device)
        res[f"all_gather_dim0_{name}"] = xm.all_gather(
            t, dim=0, groups=groups, pin_layout=False
        ).cpu()
        xm.mark_step()

        t = payload(rank).to(device)
        res[f"all_reduce_sum_{name}"] = xm.all_reduce(
            xm.REDUCE_SUM, t, groups=groups, pin_layout=False
        ).cpu()
        xm.mark_step()

        t = payload(rank).to(device)
        res[f"reduce_scatter_sum_dim0_{name}"] = xm.reduce_scatter(
            xm.REDUCE_SUM, t, scale=1.0, scatter_dim=0, shard_count=size,
            groups=groups, pin_layout=False,
        ).cpu()
        xm.mark_step()

    # The shape every tpu_application uses to fan the encoded prompt out.
    t = payload(rank).to(device)
    xm.collective_broadcast([t], root_ordinal=0)
    xm.mark_step()
    res["broadcast_src0"] = t.cpu()

    xm.wait_device_ops()
    torch.save(res, f"{OUT}_xla_rank{rank}.pt")
    print(f"[xla] rank {rank}/{world} wrote {OUT}_xla_rank{rank}.pt", flush=True)


if __name__ == "__main__":
    torch_xla.launch(run, args=())
