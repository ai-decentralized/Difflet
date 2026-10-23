#!/usr/bin/env python3
"""Why is torch_tpu's host->device copy 20-27x slower than torch_xla's?

Each case moves 1 GiB, forced to arrive (a dependent read, back to the host),
best of 3. Process CPU time is recorded beside wall time: if the host is doing
the work (a relayout, a conversion) CPU ~ wall; if it is waiting on DMA, CPU << wall.
"""

import json
import os
import time

import torch
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
GiB = 2 ** 30


def timed(make_host, label, **to_kwargs):
    best = None
    for _ in range(3):
        host = make_host()
        w0, c0 = time.perf_counter(), time.process_time()
        t = host.to(dev, **to_kwargs)
        float(t.reshape(-1)[:1].float().sum().cpu())  # wait for the whole buffer
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        if best is None or wall < best[0]:
            best = (wall, cpu)
        del t, host
    wall, cpu = best
    gb = make_host().numel() * make_host().element_size() / GiB
    print("%-34s %6.2f GB/s   wall %6.3f s   cpu %6.3f s   cpu/wall %.2f"
          % (label, gb / wall, wall, cpu, cpu / wall), flush=True)
    return {"label": label, "GBps": round(gb / wall, 3), "cpu_over_wall": round(cpu / wall, 2)}


n32 = GiB // 4
print("TPU_PREMAPPED_BUFFER_SIZE=%s   threads=%d"
      % (os.environ.get("TPU_PREMAPPED_BUFFER_SIZE", "<unset>"), torch.get_num_threads()))
out = []
out.append(timed(lambda: torch.ones(n32), "A0 fp32 1-D pageable (baseline)"))
out.append(timed(lambda: torch.ones(n32).pin_memory(), "A1 fp32 1-D pinned"))
out.append(timed(lambda: torch.ones(n32), "E  fp32 1-D non_blocking", non_blocking=True))
out.append(timed(lambda: torch.ones(n32).reshape(64, 1024, 4096), "C  fp32 3-D (weight-like)"))
out.append(timed(lambda: torch.ones(n32 * 2, dtype=torch.bfloat16), "D  bf16 1-D same bytes"))
out.append(timed(lambda: torch.ones(n32 * 2, dtype=torch.bfloat16).reshape(4096, 131072),
                 "D2 bf16 2-D same bytes"))
print("H2DWHY " + json.dumps(out))
