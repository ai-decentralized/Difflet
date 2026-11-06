#!/usr/bin/env python3
"""Does a transfer take the fast path only when it fits in the premapped buffer?

Buffer fixed by the env var; transfer size varied across the boundary.
"""

import os
import time

import torch
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
MiB = 2 ** 20


def gbps(nbytes):
    best = None
    for _ in range(3):
        h = torch.ones(nbytes // 4).pin_memory()
        t0 = time.perf_counter()
        t = h.to(dev)
        float(t.reshape(-1)[:1].float().sum().cpu())
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        del t, h
    return nbytes / 2 ** 30 / best


buf = int(os.environ["TPU_PREMAPPED_BUFFER_SIZE"]) // MiB
for mib in (32, 64, 128, 192, 240, 256, 300, 512):
    side = "fits" if mib < buf else "exceeds"
    print("FIT buffer=%dMiB transfer=%4dMiB (%-7s) %6.2f GB/s" % (buf, mib, side, gbps(mib * MiB)),
          flush=True)
