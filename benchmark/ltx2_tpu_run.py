"""Run LTX-2 on 4 v5e chips through difflet's own TPU backend.

One process per chip, tp=4, eager. Drives ``TpuLTX2Application.load_eager`` and
the backend-neutral ``LTX2Orchestrator`` exactly as ``difflet serve`` does:
Gemma-3 on XLA ordinal 0 + broadcast, the sharded DiT on the chips, the video
VAE on rank 0's chip, audio VAE + vocoder on the host. Fork of
``benchmark/hunyuan_tpu_run.py``. Default shape is the Trainium MATRIX row
(480x704x49, 20 steps, guidance 1.0) so the row lands beside benchmark/trn2.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SNAP = os.environ.get(
    "DIFFLET_LTX2_SNAPSHOT",
    "/mnt/models/hf/hub/models--Lightricks--LTX-2/snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
)
PROMPT = "a cinematic shot of a red fox running through a snowy forest"


def _mem(xm, device):
    try:
        return {k: round(v / 2**30, 3) for k, v in xm.get_memory_info(device).items()
                if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def worker(rank, world, args, reply_q, decode_done):
    try:
        _worker(rank, world, args, reply_q, decode_done)
    except Exception as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"type": "error", "rank": rank, "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        decode_done.set()


def _worker(rank, world, args, reply_q, decode_done):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))
    # The primary replica (rank 0 here) is the one whose VAE is on the chip.
    os.environ["DIFFLET_REPLICA_RANK"] = str(rank)

    from difflet.models.ltx_2.entry import create_ltx_2_application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    log = lambda m: print(f"[rank{rank}] {m}", flush=True)  # noqa: E731

    started = time.monotonic()
    app = create_ltx_2_application(
        model_path=args["model_dir"], parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": args["height"], "width": args["width"], "num_frames": args["num_frames"]},
        backend="tpu", enable_host_pipeline=True,
        teacache_cadence=args.get("teacache_cadence"),
        teacache_online_delta_alpha=args.get("teacache_online_delta"),
    )
    log(f"config video_seq={app.config.video_seq_len} audio_seq={app.config.audio_seq_len}")
    mem_before = _mem(xm, device)
    app.load_eager()
    load_seconds = time.monotonic() - started
    mem_after_load = _mem(xm, device)
    log(f"loaded in {load_seconds:.1f}s mem={mem_after_load}")

    from benchmark.harness import RealLoopStepTimer

    sync_now = {"on": True}

    def _sync():
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_sync)
    inner_forward = app.forward_dit

    def timed_forward(bundle):
        out = inner_forward(bundle)
        timer.step()
        return out

    app.forward_dit = timed_forward
    orchestrator = app.pipeline

    total_iters = args["iters"] + args["natural_iters"]
    latents = None
    for iteration in range(total_iters):
        sync_now["on"] = iteration < args["iters"]
        timer.stamps.clear()
        wall = time.monotonic()
        out = orchestrator(
            prompt=PROMPT, negative_prompt=None, num_inference_steps=args["steps"],
            guidance_scale=args["guidance"], generator=torch.Generator().manual_seed(args["seed"]),
            output_type="latent",
        )
        total = time.monotonic() - wall
        steps = timer.deltas()
        latents = out.latents if hasattr(out, "latents") else out[0]
        basis = "synced" if sync_now["on"] else "natural"
        log(f"iter{iteration} [{basis}] e2e_latent={total:.2f}s steps={len(steps)} "
            f"mean_step={sum(steps) / max(1, len(steps)):.3f}s")
        controller = getattr(orchestrator, "_teacache_controller", None)
        if controller is not None:
            print(f"[teacache] stats: {controller.stats()}", flush=True)
        if rank == 0:
            reply_q.put({
                "type": "result", "iteration": iteration, "basis": basis,
                "wall_seconds": total, "load_seconds": load_seconds,
                "step_seconds": list(steps), "mem_before": mem_before,
                "mem_after_load": mem_after_load, "mem_after_run": _mem(xm, device),
                "latent_shape": list(latents.shape), "finite": bool(latents.isfinite().all()),
                "mean": float(latents.float().mean()), "std": float(latents.float().std()),
                "teacache": controller.stats() if controller is not None else None,
            })

    if rank != 0 and args["decode"]:
        decode_done.wait()
    if rank == 0 and args["decode"]:
        mark = time.monotonic()
        video, audio = orchestrator._decode_latents(latents, out.audio_latents if hasattr(out, "audio_latents") else None)
        seconds = time.monotonic() - mark
        reply_q.put({"type": "decoded", "seconds": seconds, "where": "device(video)+host(audio)",
                     "shape": list(video.shape), "finite": bool(video.isfinite().all())})
        import numpy as np

        frames = video[0].permute(1, 2, 3, 0).clamp(0, 1) if video.min() >= 0 else (video[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2
        np.save(args["out"] + ".npy", frames.float().numpy())
        try:
            import imageio

            imageio.mimsave(args["out"] + ".mp4", (frames.float().numpy() * 255).astype("uint8"), fps=24)
        except Exception as exc:  # noqa: BLE001
            print(f"[rank0] mp4 write skipped: {exc}", flush=True)
        decode_done.set()

    reply_q.put({"type": "done", "rank": rank})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=SNAP)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--natural-iters", type=int, default=2)
    parser.add_argument("--teacache-cadence", type=int, default=None, metavar="N")
    parser.add_argument("--teacache-online-delta", type=float, default=None, metavar="ALPHA")
    parser.add_argument("--no-decode", dest="decode", action="store_false")
    parser.add_argument("--out", default="/mnt/models/ltx2_tpu_out")
    parser.add_argument("--world", type=int, default=0)
    options = parser.parse_args()
    world = options.world or len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    args = {k: getattr(options, k) for k in ("model_dir", "height", "width", "num_frames", "steps", "guidance",
                                             "seed", "iters", "natural_iters", "decode", "teacache_cadence",
                                             "teacache_online_delta", "out")}
    print(f"world={world} args={args}", flush=True)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    decode_done = ctx.Event()
    procs = [ctx.Process(target=worker, args=(r, world, args, reply_q, decode_done), daemon=False) for r in range(world)]
    for p in procs:
        p.start()
    results, done, failed = [], 0, False
    while done < world:
        alive = any(p.is_alive() for p in procs)
        try:
            msg = reply_q.get(timeout=60 if alive else 5)
        except Exception:  # noqa: BLE001
            if not alive:
                break
            continue
        if msg["type"] == "done":
            done += 1
        elif msg["type"] == "error":
            failed = True
            print(f"ERROR rank{msg['rank']}: {msg['error']}\n{msg['traceback']}", file=sys.stderr, flush=True)
        else:
            results.append(msg)
            print(json.dumps(msg, indent=2), flush=True)
    for rank, p in enumerate(procs):
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
        if p.exitcode not in (0, None):
            failed = True
            print(f"rank{rank} exited with code {p.exitcode}", file=sys.stderr, flush=True)
    Path(options.out + ".json").write_text(json.dumps(results, indent=2))
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
