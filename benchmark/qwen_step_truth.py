"""Is Qwen-Image's reported 0.276 s/step on v5e real device time?

SDPA alone at Qwen's per-chip attention shape (6 heads, 5120x5120) measures
6.64 ms, x60 layers = 0.399 s -- already more than the reported whole step. So
either the DiT is faster than its own attention (impossible) or the reported
figure is not device time.

Measures the same DiT three ways:
  enqueue   - mark_step per step, no wait: what a lazy loop's inter-step delta
              sees, which is the Python enqueue rate while the device lags
  throughput- N steps, one wait at the end, divided by N: real device time
  synced    - mark_step + wait every step: real, without cross-step overlap
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

SNAP = os.environ.get(
    "DIFFLET_QWEN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/"
    "75e0b4be04f60ec59a75f475837eced720f823b6",
)
N = 8


def worker(rank, world, reply_q):
    try:
        _worker(rank, world, reply_q)
    except Exception as exc:  # noqa: BLE001
        import traceback
        reply_q.put({"done": rank, "error": repr(exc),
                     "traceback": traceback.format_exc()})
        raise


def _worker(rank, world, reply_q):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

    from difflet.models.qwen_image.entry import create_qwen_image_application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    app = create_qwen_image_application(
        model_path=SNAP,
        parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": 1024, "width": 1024, "num_frames": None},
        backend="tpu",
        text_seq_len=1024,
    )
    module = app.transformer._prepare_module().to(device)
    xm.mark_step()
    xm.wait_device_ops()
    cfg = app.config

    latents = torch.zeros(1, cfg.image_seq_len, int(cfg.in_channels),
                          dtype=torch.bfloat16, device=device)
    timestep = torch.full((1,), 0.5, dtype=torch.bfloat16, device=device)
    text = torch.zeros(1, int(cfg.text_seq_len), int(cfg.joint_attention_dim),
                       dtype=torch.bfloat16, device=device)

    def call(x):
        return module(x, timestep, text, None, None)

    with torch.no_grad():
        x = latents
        for _ in range(2):                      # warm the graph
            x = call(x)
            xm.mark_step()
        xm.wait_device_ops()

        # enqueue rate: what an unsynced inter-step delta actually measures
        x = latents
        deltas = []
        for _ in range(N):
            mark = time.monotonic()
            x = call(x)
            xm.mark_step()
            deltas.append(time.monotonic() - mark)
        xm.wait_device_ops()
        enqueue = sum(deltas) / len(deltas)

        # throughput: N steps, one wait -- real device time per step
        x = latents
        mark = time.monotonic()
        for _ in range(N):
            x = call(x)
            xm.mark_step()
        xm.wait_device_ops()
        throughput = (time.monotonic() - mark) / N

        # synced: wait every step
        x = latents
        synced = []
        for _ in range(N):
            mark = time.monotonic()
            x = call(x)
            xm.mark_step()
            xm.wait_device_ops()
            synced.append(time.monotonic() - mark)

    if rank == 0:
        reply_q.put({
            "enqueue_s": enqueue,
            "throughput_s": throughput,
            "synced_s": sum(synced) / len(synced),
            "layers": int(cfg.num_layers),
            "image_seq_len": int(cfg.image_seq_len),
            "text_seq_len": int(cfg.text_seq_len),
            "mem_gb": xm.get_memory_info(device)["bytes_used"] / 2**30,
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
        if not any(p.is_alive() for p in procs) and reply_q.empty():
            print("workers exited early", flush=True)
            break
        try:
            msg = reply_q.get(timeout=180)
        except Exception:  # noqa: BLE001
            continue
        if "done" in msg:
            seen += 1
            if msg.get("traceback"):
                print(msg["traceback"], flush=True)
        else:
            layers = msg["layers"]
            print(f"Qwen-Image DiT, {layers} layers, seq {msg['image_seq_len']}"
                  f"+{msg['text_seq_len']}, tp=4, {msg['mem_gb']:.2f} GB/chip",
                  flush=True)
            for key, name in (("enqueue_s", "enqueue rate (no wait)"),
                              ("throughput_s", "throughput (one wait)"),
                              ("synced_s", "synced every step")):
                print(f"  {name:26s} {msg[key]*1000:8.1f} ms/step", flush=True)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
