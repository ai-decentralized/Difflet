#!/usr/bin/env python3
"""host->device bandwidth, same code both runtimes, on an empty chip.

Every copy is forced to materialise with a reduction read back to the host;
timing .to() plus a wait measures enqueue on a lazy runtime, not transfer.

The torch_tpu transformer load stage is 20-32 s against torch_xla's whole load
of 7.8 s, and a 1 GB copy measured 0.75-1.71 s on a chip that was already
nearly full. This measures it clean, single process, nothing else resident.
"""

import json
import time

import torch

from difflet.backends.tpu.ops_impl import device_runtime as D

SIZES_GB = (0.25, 1.0, 2.0)


def main():
    dev = D.device()
    out = {"runtime": D.runtime_name()}

    # warm the path once so the first-call cost is not in the numbers
    warm = torch.zeros(1024, 1024, dtype=torch.float32).to(dev)
    D.wait()
    del warm

    for gb in SIZES_GB:
        elems = int(gb * 2 ** 30 // 4)
        host = torch.empty(elems, dtype=torch.float32).uniform_()
        best = None
        for _ in range(3):
            mark = time.perf_counter()
            on_dev = host.to(dev)
            # Force the transfer to actually happen. torch_xla is lazy: .to()
            # records the copy and wait_device_ops() need not materialise it
            # when nothing consumes the tensor, so timing the pair measures the
            # enqueue rate. Pulling a reduction back to the host does not.
            checksum = float(on_dev.sum().cpu())
            D.wait()
            dt = time.perf_counter() - mark
            out.setdefault("checksum_%.2f" % gb, round(checksum, 1))
            best = dt if best is None else min(best, dt)
            del on_dev
        out["h2d_%.2fGB_s" % gb] = round(best, 3)
        out["h2d_%.2fGB_GBps" % gb] = round(gb / best, 2)
        del host

    print("H2D " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
