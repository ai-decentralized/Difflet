#!/usr/bin/env python3
"""Parent-side TPU launcher.

``torch.tpu.distributed.set_tpu_launch_env()`` has to run in the parent, before
any worker spawns — torchrun does not set TORCH_TPU_TOPOLOGY /
TORCH_TPU_SLICEBUILDER_ADDRESSES itself, and ``init_process_group("tpu_dist")``
refuses without them. exec'ing torchrun (rather than spawning it) hands the
whole environment over and drops whatever the parent had attached.

    python tt_launch.py <nproc> <script.py> [args...]
"""

import os
import sys

import torch
import torch_tpu  # noqa: F401 — registers the 'tpu' backend


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    nproc, script = int(sys.argv[1]), sys.argv[2]

    # Without this, process_group_tpu warns "Node-local store disabled" and every
    # rank publishes to the world store individually.
    os.environ.setdefault("TORCH_TPU_DEVICES_PER_HOST", str(nproc))
    torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=nproc)

    argv = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}", "--redirects=3", "--tee=3",
        script, *sys.argv[3:],
    ]
    print(f"exec: {' '.join(argv)}", flush=True)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
