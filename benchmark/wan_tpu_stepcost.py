"""Isolate Wan's DiT compute from the host round-trip the orchestrator needs.

The orchestrator holds latents on the host in fp32 (UniPC's order-2 corrector
collapses in bf16), so every step transfers and syncs. This measures the same
DiT calls two ways — synced per step, and pipelined with a single sync at the
end — to say how much of the per-step figure is the sync rather than compute.
Values are meaningless (the output is fed back as input); only shapes and
therefore the graph are real.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

SNAP = os.environ.get(
    "DIFFLET_WAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)
N = 10


def worker(rank, world, reply_q):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

    from difflet.models.wan.entry import create_wan_application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    app = create_wan_application(
        model_path=SNAP,
        parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": 480, "width": 832, "num_frames": 9},
        backend="tpu",
    )
    module = app.transformer._prepare_module().to(device)
    xm.mark_step()
    cfg = app.config

    latents = torch.zeros(
        1, cfg.in_channels, cfg.latent_frames, cfg.latent_height,
        cfg.latent_width, dtype=torch.bfloat16, device=device,
    )
    timestep = torch.full((1,), 500.0, dtype=torch.bfloat16, device=device)
    text = torch.zeros(1, cfg.text_seq_len, cfg.text_dim,
                       dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        # warm: pay the XLA compile once
        out = module(latents, timestep, text)
        xm.mark_step()
        xm.wait_device_ops()

        synced = []
        x = latents
        for _ in range(N):
            mark = time.monotonic()
            x = module(x, timestep, text)
            xm.mark_step()
            xm.wait_device_ops()
            synced.append(time.monotonic() - mark)

        # host round-trip, as the orchestrator actually does it
        roundtrip = []
        host = latents.cpu().float()
        for _ in range(N):
            mark = time.monotonic()
            y = module(host.to(torch.bfloat16).to(device), timestep, text)
            xm.mark_step()
            host = y.cpu().float()
            roundtrip.append(time.monotonic() - mark)

        # pipelined: cut the graph per step (same graph each time, so no
        # recompile) but do not wait — the analogue of the no-forced-sync
        # loop the Qwen-Image TPU benchmark reports. Without the per-step
        # mark_step, XLA instead fuses all N calls into one giant graph and
        # recompiles it, which measures nothing useful.
        x = latents
        for _ in range(2):  # warm both the graph and the queue
            x = module(x, timestep, text)
            xm.mark_step()
        xm.wait_device_ops()
        mark = time.monotonic()
        for _ in range(N):
            x = module(x, timestep, text)
            xm.mark_step()
        xm.wait_device_ops()
        pipelined = (time.monotonic() - mark) / N

    if rank == 0:
        reply_q.put({
            "synced_mean": sum(synced) / len(synced),
            "roundtrip_mean": sum(roundtrip) / len(roundtrip),
            "pipelined_mean": pipelined,
            "synced": synced, "roundtrip": roundtrip,
            "mem": {k: round(v / 2**30, 3)
                    for k, v in xm.get_memory_info(device).items()
                    if isinstance(v, (int, float))},
        })
    reply_q.put({"done": rank})


def main() -> None:
    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    seen = 0
    while seen < world:
        msg = reply_q.get()
        if "done" in msg:
            seen += 1
        else:
            print(json.dumps(msg, indent=2), flush=True)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
