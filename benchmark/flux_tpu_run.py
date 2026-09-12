"""Run FLUX.1-dev on 4 v5e chips through difflet's own TPU backend.

One process per chip, tp=4, eager. Drives ``TpuFluxApplication.load_eager`` and
its device-resident denoise loop exactly as ``difflet serve`` does: T5-XXL on
XLA ordinal 0 + broadcast, CLIP-L on every rank, the sharded DiT on the chips,
the VAE on rank 0's chip. Fork of ``benchmark/ltx2_tpu_run.py``. Default shape
is the Trainium MATRIX row (1024x1024, 28 steps, guidance 3.5) so the row lands
beside benchmark/trn2.
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
    "DIFFLET_FLUX_SNAPSHOT",
    "/mnt/models/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
)
PROMPT = "a cinematic photo of a red fox in a snowy forest at golden hour, highly detailed"


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

    from difflet.models.flux.entry import create_flux_application
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    log = lambda m: print(f"[rank{rank}] {m}", flush=True)  # noqa: E731

    started = time.monotonic()
    app = create_flux_application(
        model_path=args["model_dir"], parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": args["height"], "width": args["width"], "num_frames": None},
        backend="tpu", enable_host_pipeline=True,
        teacache_cadence=args.get("teacache_cadence"),
        teacache_online_delta_alpha=args.get("teacache_online_delta"),
    )
    log(f"config image_seq={app.config.image_seq_len} text_seq={app.config.text_seq_len}")
    mem_before = _mem(xm, device)
    app.load_eager()
    load_seconds = time.monotonic() - started
    mem_after_load = _mem(xm, device)
    log(f"loaded in {load_seconds:.1f}s timings={app.last_timings} mem={mem_after_load}")

    from benchmark.harness import RealLoopStepTimer

    sync_now = {"on": True}

    def _sync():
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_sync)
    inner_forward = app.transformer.forward

    def timed_forward(*a, **k):
        out = inner_forward(*a, **k)
        timer.step()
        return out

    # The denoise loop looks up transformer.forward per call, so this wraps every step.
    app.transformer.forward = timed_forward

    total_iters = args["iters"] + args["natural_iters"]
    latents = None
    for iteration in range(total_iters):
        sync_now["on"] = iteration < args["iters"]
        timer.stamps.clear()
        wall = time.monotonic()
        out = app(
            prompt=PROMPT, num_inference_steps=args["steps"], guidance_scale=args["guidance"],
            generator=torch.Generator().manual_seed(args["seed"]), output_type="latent",
        )
        total = time.monotonic() - wall
        steps = timer.deltas()
        latents = out.latents
        basis = "synced" if sync_now["on"] else "natural"
        log(f"iter{iteration} [{basis}] e2e_latent={total:.2f}s steps={len(steps)} "
            f"mean_step={sum(steps) / max(1, len(steps)):.3f}s stage={app.last_timings}")
        if rank == 0:
            reply_q.put({
                "type": "result", "iteration": iteration, "basis": basis,
                "wall_seconds": total, "load_seconds": load_seconds,
                "stage_seconds": dict(app.last_timings),
                "step_seconds": list(steps), "mem_before": mem_before,
                "mem_after_load": mem_after_load, "mem_after_run": _mem(xm, device),
                "latent_shape": list(latents.shape), "finite": bool(latents.isfinite().all()),
                "mean": float(latents.float().mean()), "std": float(latents.float().std()),
                "teacache": app.teacache_last_stats,
            })

    if rank != 0 and args["decode"]:
        decode_done.wait()
    if rank == 0 and args["decode"]:
        torch.save({"latents": latents}, args["out"] + ".latents.pt")
        # Twice: the first decode carries the VAE graph's compile (117 s at
        # 1024x1024 measured), the second is the warm figure serving sees.
        seconds = []
        for _ in range(2):
            mark = time.monotonic()
            image = app.decode(latents, height=args["height"], width=args["width"])
            seconds.append(time.monotonic() - mark)
        image.save(args["out"] + ".png")
        reply_q.put({"type": "decoded", "seconds": seconds[-1], "first_decode_seconds": seconds[0],
                     "where": "device", "size": list(image.size), "mem_after_decode": _mem(xm, device)})
        decode_done.set()

    reply_q.put({"type": "done", "rank": rank})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=SNAP)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--natural-iters", type=int, default=2)
    parser.add_argument("--teacache-cadence", type=int, default=None, metavar="N")
    parser.add_argument("--teacache-online-delta", type=float, default=None, metavar="ALPHA")
    parser.add_argument("--no-decode", dest="decode", action="store_false")
    parser.add_argument("--out", default="/mnt/models/flux_tpu_out")
    parser.add_argument("--world", type=int, default=0)
    options = parser.parse_args()
    world = options.world or len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    args = {k: getattr(options, k) for k in ("model_dir", "height", "width", "steps", "guidance", "seed",
                                             "iters", "natural_iters", "decode", "teacache_cadence",
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
