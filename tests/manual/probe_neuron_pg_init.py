"""Probe: where the neuron backend's per-launch start-up goes (prepare_runtime took 10-18 s).

    torchrun --standalone --nproc-per-node 4 tests/manual/probe_neuron_pg_init.py

Every rank times, in order: import torch; import torch_neuronx; Difflet's compile-cache
configuration and core binding; dist.init_process_group("neuron"); the first device tensor (if
the runtime starts lazily, it starts here); two all-reduces and a barrier on the neuron group;
then a gloo subgroup over the same ranks (same store and rendezvous, no Neuron runtime) and one
all-reduce on it. One line per rank.
"""

from __future__ import annotations

import os
import time

T0 = time.perf_counter()


def _process_age() -> float:
    """Seconds since this process started (from /proc), to show interpreter start-up."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return float("nan")


AGE = _process_age()
times: dict[str, float] = {}
mark = time.perf_counter()


def lap(name: str) -> None:
    global mark
    now = time.perf_counter()
    times[name] = round(now - mark, 2)
    mark = now


import torch  # noqa: E402

lap("import torch")
import torch_neuronx  # noqa: E402,F401

lap("import torch_neuronx")
import torch.distributed as dist  # noqa: E402

from difflet.backends.neuron.runtime import bind_core, configure_compile_cache  # noqa: E402

configure_compile_cache()
bind_core()
lap("cache + bind_core")
dist.init_process_group("neuron")
lap("init_process_group(neuron)")
x = torch.ones(1024).to("neuron")
torch.neuron.synchronize()
lap("first device tensor")
dist.all_reduce(x)
torch.neuron.synchronize()
lap("all_reduce #1")
dist.all_reduce(x)
torch.neuron.synchronize()
lap("all_reduce #2")
dist.barrier()
lap("barrier")
gloo = dist.new_group(backend="gloo")
lap("new_group(gloo)")
y = torch.ones(1024)
dist.all_reduce(y, group=gloo)
lap("gloo all_reduce")

total = round(time.perf_counter() - T0, 2)
print(f"[probe] rank {dist.get_rank()}: process age at start {AGE:.2f} s; {times}; total {total} s; "
      f"sum check {x[0].item():.0f}/{y[0].item():.0f}", flush=True)
dist.destroy_process_group()
