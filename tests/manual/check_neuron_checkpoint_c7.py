"""Manual check: lazy per-rank safetensors load onto the neuron device (C7).

Run on a trn2 host with the TorchNeuron stack and no other process on the cores:

    DIFFLET_BACKEND=neuron DIFFLET_DISABLE_PREWARM=1 NEURON_RT_NUM_CORES=4 PYTHONPATH=$PWD \
        torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_checkpoint_c7.py

Rank 0 writes a synthetic fp32 checkpoint (two files and an index). Every rank builds
a stack of toy column->row MLPs on meta and loads its own shards onto its NeuronCore
in bf16. Each check is aggregated over all ranks:

* placement: every parameter is bf16 on the neuron device, nothing left on meta,
  nothing missing or unexpected;
* shards: each rank's shard equals the matching slice of the checkpoint cast to bf16;
* reassembly: all-gathering the shards on the device rebuilds the full bf16 tensors;
* buffer: a computed fp32 buffer reaches the device unchanged and still fp32;
* forward: the loaded stack's TP forward is within 3x of a bf16 CPU forward's error
  against the fp32 CPU reference;
* fallbacks: zero CPU fallbacks during all device work.

The per-rank peak host RSS of the load is printed for information; the bound is
asserted on CPU by tests/unit/backends/test_neuron_checkpoint_c7.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "neuron")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

from difflet.backends.neuron.core.checkpoint import (  # noqa: E402
    build_on_meta,
    load_sharded_checkpoint,
    shard_dim,
)
from difflet.backends.neuron.ops_impl import parallel_mesh as pm  # noqa: E402
from difflet.backends.neuron.ops_impl.collectives import gather_tp_dim  # noqa: E402
from difflet.backends.neuron.runtime import track_fallbacks  # noqa: E402
from difflet.backends.registry import get_backend  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.pipeline.parallel_mesh import MeshSpec  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    ToyTPMLP,
    reference_mlp,
    toy_full_weights,
    write_toy_checkpoint,
)

MiB = 2**20
CASES = ("placement", "shards", "reassembly", "buffer", "forward", "fallbacks")


class ToyStack(nn.Module):
    def __init__(self, layers: int, dim: int, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList(ToyTPMLP(dim, hidden) for _ in range(layers))
        self.register_buffer("rope", torch.linspace(0.0, 1.0, 4096), persistent=False)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def reference_stack(x, per_layer):
    for weights in per_layer:
        x = x + reference_mlp(x, weights)
    return x


def proc_kib(field):
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def all_ranks_sum(values, world):
    """Sum a float vector over ranks with an eager neuron all-reduce (verified in C1)."""
    totals = torch.tensor(values, dtype=torch.float32).to("neuron")
    if world > 1:
        dist.all_reduce(totals)
    return totals.cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/tmp/difflet_check_c7")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=world))
    rank = dist.get_rank() if dist.is_initialized() else 0
    pm.init_parallel_mesh(MeshSpec(tp=world))

    per_layer = [toy_full_weights(args.dim, args.hidden, seed=i) for i in range(args.layers)]
    full = {f"layers.{i}.{k}": t for i, w in enumerate(per_layer) for k, t in w.items()}
    ckpt_dir = Path(args.work_dir)
    ok = dict.fromkeys(CASES, True)
    sharded = 0

    with track_fallbacks() as fallbacks:
        wrote = 1.0
        if rank == 0:
            try:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                write_toy_checkpoint(ckpt_dir, full, num_files=2)
            except Exception as exc:  # every rank stops at the sync below
                print(f"FAIL write checkpoint: {exc!r}", flush=True)
                wrote = 0.0
        if all_ranks_sum([wrote], world)[0] != world:  # also orders reads after the write
            print(f"[rank {rank}] FAIL", flush=True)
            return 1

        model = build_on_meta(lambda: ToyStack(args.layers, args.dim, args.hidden))
        rss_ok = reset_peak_rss()
        rss_before = proc_kib("VmRSS")
        report = load_sharded_checkpoint(model, ckpt_dir, device="neuron", dtype=torch.bfloat16)
        torch.neuron.synchronize()
        load_peak_mib = (proc_kib("VmHWM") - rss_before) / 1024 if rss_ok else -1.0
        model.eval().requires_grad_(False)

        placed = all(
            p.device.type == "neuron" and p.dtype == torch.bfloat16 for p in model.parameters()
        )
        no_meta = not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
        clean = report == {"missing": [], "unexpected": []}
        ok["placement"] = placed and no_meta and clean

        state = model.state_dict()
        for name, tensor in full.items():  # same order on every rank: gathers must match
            axis = shard_dim(model, name)
            expected = tensor if axis is None else tensor.chunk(world, dim=axis)[rank]
            ok["shards"] &= torch.equal(state[name].cpu(), expected.to(torch.bfloat16))
            if axis is not None:
                sharded += 1
                whole = gather_tp_dim(state[name], dim=axis).cpu()
                ok["reassembly"] &= torch.equal(whole, tensor.to(torch.bfloat16))

        ok["buffer"] = (
            model.rope.device.type == "neuron"
            and model.rope.dtype == torch.float32
            and torch.equal(model.rope.cpu(), torch.linspace(0.0, 1.0, 4096))
        )

        x = torch.randn(2, 64, args.dim, generator=torch.Generator().manual_seed(1))
        bf16_layers = [{k: v.to(torch.bfloat16) for k, v in w.items()} for w in per_layer]
        with torch.no_grad():
            ref = reference_stack(x, per_layer).double()
            cpu_bf16 = reference_stack(x.to(torch.bfloat16), bf16_layers).double()
            dev = model(x.to("neuron", torch.bfloat16)).cpu().double()
        dev_err = (dev - ref).abs().mean().item()
        cpu_err = (cpu_bf16 - ref).abs().mean().item()
        ok["forward"] = dev_err <= 3.0 * cpu_err + 1e-6
    ok["fallbacks"] = not fallbacks

    rss = [0.0] * world
    rss[rank] = load_peak_mib
    totals = all_ranks_sum([1.0 if ok[c] else 0.0 for c in CASES] + rss, world)
    passed = [int(round(v)) for v in totals[: len(CASES)]]
    if rank == 0:
        detail = {
            "placement": "params bf16 on neuron, nothing on meta, nothing missing",
            "shards": f"{len(full)} tensors equal their bf16 checkpoint slice",
            "reassembly": f"{sharded} sharded tensors all-gathered on device == full bf16",
            "buffer": "computed fp32 buffer on device, unchanged",
            "forward": f"rank 0 mean|err| device {dev_err:.2e} <= 3 x cpu bf16 {cpu_err:.2e}",
            "fallbacks": f"rank 0: {fallbacks}",
        }
        for case, count in zip(CASES, passed):
            verdict = "PASS" if count == world else "FAIL"
            print(f"{verdict} {case:10s} {detail[case]} ({count}/{world} ranks)", flush=True)
        ckpt_mib = sum(t.numel() * t.element_size() for t in full.values()) / MiB
        shard_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / MiB
        peaks = [round(v, 1) for v in totals[len(CASES) :]]
        print(
            f"INFO load peak host RSS per rank (MiB): {peaks}; checkpoint {ckpt_mib:.0f} MiB, "
            f"rank shard {shard_mib:.0f} MiB",
            flush=True,
        )
    print(f"[rank {rank}] {'PASS' if all(ok.values()) else 'FAIL'}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if all(count == world for count in passed) else 1


if __name__ == "__main__":
    sys.exit(main())
