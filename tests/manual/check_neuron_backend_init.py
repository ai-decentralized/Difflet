"""Manual check: neuron (TorchNeuron) backend runtime on hardware.

Run on a trn2 host with the TorchNeuron stack:

    NEURON_RT_NUM_CORES=4 torchrun --nproc-per-node 4 tests/manual/check_neuron_backend_init.py

Verifies that prepare_runtime binds each rank to its own NeuronCore and brings up
the ``neuron`` process group, even when torch_neuronx is imported first, and that
an all-reduce on the device is exact with no CPU fallback.
"""

from __future__ import annotations

import os
import sys

import torch
import torch_neuronx  # noqa: F401  imported before prepare_runtime on purpose

from difflet.backends.registry import get_backend
from difflet.backends.neuron.runtime import track_fallbacks
from difflet.pipeline.parallel_config import DiffletParallelConfig


def main() -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    backend = get_backend("neuron")
    backend.prepare_runtime(DiffletParallelConfig(tp_degree=world))

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    core = int(os.environ.get("NEURON_RT_VISIBLE_CORES", "-1"))
    ok = True
    with track_fallbacks() as fallbacks:
        cores = torch.tensor([core], dtype=torch.float32).to("neuron")
        gathered = torch.empty(world, dtype=torch.float32, device="neuron")
        if world > 1:
            dist.all_gather_into_tensor(gathered, cores)
        else:
            gathered.copy_(cores)
        total = torch.full((1024,), float(rank + 1)).to("neuron")
        if world > 1:
            dist.all_reduce(total)
    bound = [int(c) for c in gathered.cpu().tolist()]
    expected_sum = world * (world + 1) / 2
    if len(set(bound)) != world:
        ok = False
    if not torch.equal(total.cpu(), torch.full((1024,), expected_sum)):
        ok = False
    if fallbacks:
        ok = False
    if rank == 0:
        print(f"backend={backend.name} world={world} cores per rank={bound} "
              f"all_reduce exact={torch.equal(total.cpu(), torch.full((1024,), expected_sum))} "
              f"fallbacks={fallbacks}", flush=True)
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
