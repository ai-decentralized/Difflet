"""Run HunyuanVideo on 4 v5e chips through difflet's own TPU backend.

One process per chip, tp=4, eager. Drives difflet's hardware-neutral
``HunyuanVideoOrchestrator`` through ``TpuHunyuanVideoApplication.load_eager``
— the same objects ``difflet serve`` uses — so the loop, the TeaCache
controller and the scheduler are the repo's, not a hand-rolled copy. Text
encoding (Llama-3 8B on XLA ordinal 0 + broadcast, CLIP-L everywhere) and the
VAE decode are on the host, as in serving. Fork of ``benchmark/wan_tpu_run.py``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Documented as `python benchmark/hunyuan_tpu_run.py`, which puts benchmark/
# at sys.path[0]; the spawned workers inherit that (see wan_tpu_run.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SNAP = os.environ.get(
    "DIFFLET_HUNYUAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--hunyuanvideo-community--HunyuanVideo/"
    "snapshots/e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
)
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
# The serving adapter's Llama contract (difflet/serving/models/hunyuan_video.py).
TEXT_SEQ_LEN = 256
LLAMA_CROP_START = 95
LLAMA_SEQ_LEN = TEXT_SEQ_LEN + LLAMA_CROP_START
LLAMA_CAPTURE_LAYER = 29
LLAMA_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the "
    "following aspects: 1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of "
    "the objects.3. Actions, events, behaviors temporal relationships, physical movement "
    "changes of the objects.4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


def _mem(xm, device):
    try:
        info = xm.get_memory_info(device)
        return {k: round(v / 2**30, 3) for k, v in info.items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def worker(rank, world, args, reply_q):
    try:
        _worker(rank, world, args, reply_q)
    except Exception as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"type": "error", "rank": rank, "error": repr(exc),
                     "traceback": traceback.format_exc()})
        raise


def _worker(rank, world, args, reply_q):
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    cores = os.cpu_count() or world
    torch.set_num_threads(max(1, cores // world))

    from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle
    from difflet.models.hunyuan_video.entry import create_hunyuan_video_application
    from difflet.models.hunyuan_video.tpu_application import TpuBroadcastLlamaEncoder
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    log = lambda m: print(f"[rank{rank}] {m}", flush=True)  # noqa: E731

    started = time.monotonic()
    app = create_hunyuan_video_application(
        model_path=args["model_dir"],
        parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": args["height"], "width": args["width"], "num_frames": args["num_frames"]},
        backend="tpu",
        text_seq_len=TEXT_SEQ_LEN,
        teacache_cadence=args.get("teacache_cadence"),
        teacache_online_delta_alpha=args.get("teacache_online_delta"),
    )
    log(f"config seq_len={app.config.image_seq_len} latent={app.config.latent_frames}x"
        f"{app.config.latent_height}x{app.config.latent_width}")
    mem_before = _mem(xm, device)
    app.load_eager()
    load_seconds = time.monotonic() - started
    mem_after_load = _mem(xm, device)
    log(f"DiT resident in {load_seconds:.1f}s mem={mem_after_load}")

    # --- host stages -------------------------------------------------------
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer

    mark = time.monotonic()
    llama = TpuBroadcastLlamaEncoder(
        args["model_dir"], seq_len=LLAMA_SEQ_LEN, capture_layer=LLAMA_CAPTURE_LAYER,
        dtype=torch.bfloat16,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(str(Path(args["model_dir"]) / "tokenizer"))
    clip_tokenizer = CLIPTokenizer.from_pretrained(str(Path(args["model_dir"]) / "tokenizer_2"))
    clip = CLIPTextModel.from_pretrained(
        str(Path(args["model_dir"]) / "text_encoder_2"), dtype=torch.float32
    ).eval()
    log(f"host encoders in {time.monotonic() - mark:.1f}s (ordinal {int(xr.global_ordinal())}; "
        f"llama {'loaded' if llama.is_encoder else 'skipped, not ordinal 0'})")

    # --- per-step timer around the on-device DiT ----------------------------
    from benchmark.harness import RealLoopStepTimer

    sync_now = {"on": True}

    def _sync():
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_sync)
    orchestrator = app.pipeline
    inner = orchestrator.transformer

    class _Timed:
        dtype = inner.dtype
        config = inner.config

        def __call__(self, bundle):
            out = inner(bundle)
            timer.step()
            return out

    orchestrator.transformer = _Timed()

    def encode(prompt: str):
        tokenized = llama_tokenizer(
            LLAMA_TEMPLATE.format(prompt), max_length=LLAMA_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt", return_attention_mask=True,
        )
        out = llama(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
        hidden = out.captured_tensors[0][:, LLAMA_CROP_START:]
        mask = tokenized.attention_mask[:, LLAMA_CROP_START:].to(torch.int64)
        clip_in = clip_tokenizer(prompt, padding="max_length", max_length=77, truncation=True,
                                 return_tensors="pt")
        with torch.no_grad():
            pooled = clip(input_ids=clip_in.input_ids, attention_mask=clip_in.attention_mask)
        pooled = pooled.pooler_output.to(torch.bfloat16).reshape(1, -1)
        return hidden, mask, pooled

    total_iters = args["iters"] + args["natural_iters"]
    latents = None
    for iteration in range(total_iters):
        sync_now["on"] = iteration < args["iters"]
        timer.stamps.clear()
        mark = time.monotonic()
        hidden, mask, pooled = encode(PROMPT)
        encode_seconds = time.monotonic() - mark
        generator = torch.Generator().manual_seed(args["seed"])
        latent_frames = (args["num_frames"] - 1) // 4 + 1
        noise = torch.randn(1, 16, latent_frames, args["height"] // 8, args["width"] // 8,
                            dtype=torch.bfloat16, generator=generator)
        bundle = HunyuanVideoDiTInputBundle(
            hidden_states=noise, timestep=torch.zeros(1, dtype=torch.bfloat16),
            encoder_hidden_states=hidden, encoder_attention_mask=mask,
            pooled_projections=pooled,
            guidance=torch.full([1], args["guidance"] * 1000.0, dtype=torch.bfloat16),
        )
        wall = time.monotonic()
        out = orchestrator(bundle=bundle, num_inference_steps=args["steps"],
                           output_type="latent", return_trajectory=False)
        total = time.monotonic() - wall
        steps = timer.deltas()
        latents = out.latents
        basis = "synced" if sync_now["on"] else "natural"
        log(f"iter{iteration} [{basis}] denoise={total:.2f}s encode={encode_seconds:.2f}s "
            f"steps={len(steps)} mean_step={sum(steps) / max(1, len(steps)):.3f}s "
            f"throughput_step={total / (len(steps) + 1):.3f}s")
        controller = orchestrator.teacache_controller
        if controller is not None:
            print(f"[teacache] stats: {controller.stats()}", flush=True)
        if rank == 0:
            reply_q.put({
                "type": "result", "iteration": iteration, "basis": basis,
                "throughput_step_seconds": total / (len(steps) + 1),
                "wall_seconds": total, "encode_seconds": encode_seconds,
                "load_seconds": load_seconds, "step_seconds": list(steps),
                "mem_before": mem_before, "mem_after_load": mem_after_load,
                "mem_after_run": _mem(xm, device),
                "latent_shape": list(latents.shape),
                "finite": bool(latents.isfinite().all()),
                "min": float(latents.min()), "max": float(latents.max()),
                "mean": float(latents.mean()), "std": float(latents.std()),
                "teacache": controller.stats() if controller is not None else None,
            })

    if rank == 0 and args["decode"]:
        from diffusers import AutoencoderKLHunyuanVideo

        mark = time.monotonic()
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            str(Path(args["model_dir"]) / "vae"), torch_dtype=torch.float32
        ).eval()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        scaling = float(getattr(vae.config, "scaling_factor", 1.0))
        with torch.no_grad():
            frames = vae.decode(latents.float() / scaling, return_dict=False)[0]
        reply_q.put({"type": "decoded", "seconds": time.monotonic() - mark, "where": "host",
                     "shape": list(frames.shape), "finite": bool(frames.isfinite().all())})
        torch.save(latents, args["out"] + ".latents.pt")
        video = (frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2
        import numpy as np

        np.save(args["out"] + ".npy", video.numpy())
        try:
            import imageio

            imageio.mimsave(args["out"] + ".mp4", (video.numpy() * 255).astype("uint8"), fps=24)
        except Exception as exc:  # noqa: BLE001
            print(f"[rank0] mp4 write skipped: {exc}", flush=True)

    reply_q.put({"type": "done", "rank": rank})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=SNAP)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--natural-iters", type=int, default=2,
                        help="extra iterations with no per-step device sync")
    parser.add_argument("--teacache-cadence", type=int, default=None, metavar="N")
    parser.add_argument("--teacache-online-delta", type=float, default=None, metavar="ALPHA")
    parser.add_argument("--no-decode", dest="decode", action="store_false")
    parser.add_argument("--out", default="/mnt/models/hunyuan_tpu_out")
    parser.add_argument("--world", type=int, default=0)
    options = parser.parse_args()
    world = options.world or len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    args = {
        "model_dir": options.model_dir, "height": options.height, "width": options.width,
        "num_frames": options.num_frames, "steps": options.steps, "guidance": options.guidance,
        "seed": options.seed, "iters": options.iters, "natural_iters": options.natural_iters,
        "decode": options.decode, "teacache_cadence": options.teacache_cadence,
        "teacache_online_delta": options.teacache_online_delta, "out": options.out,
    }
    print(f"world={world} args={args}", flush=True)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, args, reply_q), daemon=False)
             for r in range(world)]
    for p in procs:
        p.start()
    results, done, failed = [], 0, False
    while done < world:
        if not any(p.is_alive() for p in procs) and reply_q.empty():
            break
        try:
            msg = reply_q.get(timeout=60)
        except Exception:  # noqa: BLE001
            if not any(p.is_alive() for p in procs):
                break
            continue
        if msg["type"] == "done":
            done += 1
        elif msg["type"] == "error":
            failed = True
            print(f"ERROR rank{msg['rank']}: {msg['error']}\n{msg['traceback']}",
                  file=sys.stderr, flush=True)
        else:
            results.append(msg)
            print(json.dumps(msg, indent=2), flush=True)
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
    Path(options.out + ".json").write_text(json.dumps(results, indent=2))
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
