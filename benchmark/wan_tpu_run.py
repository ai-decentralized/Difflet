"""Run Wan 2.2 A14B on 4 v5e chips through difflet's own TPU backend.

One process per chip (SPMD and AOT export are mutually exclusive on
torch_xla 2.9), tp=4, single expert resident by default. Drives difflet's
hardware-neutral WanOrchestrator so the loop, boundary selection and
scheduler are the repo's, not a hand-rolled copy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

SNAP = os.environ.get(
    "DIFFLET_WAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)
PROMPT = "a cinematic shot of a red fox running through a snowy forest"


def _mem(xm, device):
    try:
        info = xm.get_memory_info(device)
        return {k: round(v / 2**30, 3) for k, v in info.items()
                if isinstance(v, (int, float))}
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

    # One thread per core *per process* oversubscribes the host world-fold;
    # a text encode measured 22 s in the worker against 0.43 s standalone.
    cores = os.cpu_count() or world
    torch.set_num_threads(max(1, cores // world))

    from difflet.models.wan.entry import create_wan_application
    from difflet.models.wan.pipeline import WanOrchestrator
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    log = lambda m: print(f"[rank{rank}] {m}", flush=True)  # noqa: E731

    parallel = DiffletParallelConfig(tp_degree=world)
    started = time.monotonic()
    app = create_wan_application(
        model_path=args["model_dir"],
        parallel=parallel,
        dtype=torch.bfloat16,
        shape={"height": args["height"], "width": args["width"],
               "num_frames": args["num_frames"]},
        backend="tpu",
        enable_transformer_2=args["both_experts"],
    )
    log(f"config seq_len={app.config.image_seq_len} "
        f"latent={app.config.latent_frames}x{app.config.latent_height}x"
        f"{app.config.latent_width} experts={len(app._experts())}")

    mem_before = _mem(xm, device)
    experts = []
    for name, expert in app._named_experts():
        mark = time.monotonic()
        module = expert._prepare_module().to(device)
        xm.mark_step()
        xm.wait_device_ops()
        experts.append((name, module))
        log(f"{name} resident in {time.monotonic() - mark:.1f}s "
            f"mem={_mem(xm, device)}")
    load_seconds = time.monotonic() - started
    mem_after_load = _mem(xm, device)

    # --- host stages -------------------------------------------------------
    from transformers import AutoTokenizer, UMT5EncoderModel

    mark = time.monotonic()
    # Only rank 0 holds the text encoder. Every replica needs the *same*
    # embeddings, so encoding four times is three times wasted -- and it is
    # what forces the dtype: four fp32 copies of umT5 is ~48 GB of host RAM
    # apiece and OOM-kills a 188 GB box (measured). With one copy, fp32 fits
    # comfortably, and fp32 is 4.5x faster than bf16 here because this VM is an
    # AMD EPYC without AVX512-BF16 so bf16 matmul is emulated.
    # The encoder must live on XLA *ordinal* 0, not on multiprocessing rank 0:
    # the two are a scrambled mapping (measured here: mp 0->2, 1->0, 2->3,
    # 3->1), so broadcasting with root_ordinal=0 while encoding on mp rank 0
    # ships whichever process happens to be ordinal 0 -- a zero placeholder --
    # to everyone. It does not fail; it just produces a different image.
    import torch_xla.runtime as xr

    is_encoder = int(xr.global_ordinal()) == 0
    text_encoder = tokenizer = None
    if is_encoder:
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(Path(args["model_dir"]) / "text_encoder"), dtype=torch.float32
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(args["model_dir"]) / "tokenizer")
        )
    log(f"umT5 on host in {time.monotonic() - mark:.1f}s "
        f"(ordinal {int(xr.global_ordinal())}; "
        f"{'loaded' if is_encoder else 'skipped, not ordinal 0'})")

    # --- device-moving adapters -------------------------------------------
    from benchmark.harness import RealLoopStepTimer

    # Same rule as every other device folder: a timestamp after each step of a
    # real generate, at a device-sync point, step 0 excluded. See
    # RealLoopStepTimer's docstring for why this is not left to each adapter.
    # Mutable so one process yields both bases: the comparable synced pass and
    # the natural pass a real serving loop runs.
    sync_now = {"on": True}

    def _sync():
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_sync)

    class _OnDevice:
        """Move a DiT call onto the chip and its result back.

        The scheduler runs on the host in fp32 (UniPC's order-2 corrector
        collapses in bf16), so each step round-trips regardless.
        """

        def __init__(self, module, config):
            self.module = module
            self.config = config
            self.dtype = torch.bfloat16

        def __call__(self, hidden_states, timestep, encoder_hidden_states):
            out = self.module(
                hidden_states.to(device),
                timestep.to(device),
                encoder_hidden_states.to(device),
            )
            xm.mark_step()
            out = out.cpu()
            timer.step()
            return out

    wrapped = {name: _OnDevice(module, app.config) for name, module in experts}

    orchestrator = WanOrchestrator(
        model_path=args["model_dir"],
        text_encoder=text_encoder,
        transformer=wrapped["transformer"],
        transformer_2=wrapped.get("transformer_2"),
        vae_decoder=None,
        dtype=torch.bfloat16,
        height=args["height"],
        width=args["width"],
        num_frames=args["num_frames"],
        tokenizer_path=str(Path(args["model_dir"]) / "tokenizer"),
    )
    orchestrator._tokenizer = tokenizer

    def encode(prompt: str) -> torch.Tensor:
        """Encode once on rank 0, then broadcast to every replica.

        Two things are folded in here. Rank 0 encodes at the prompt's *real*
        length rather than padding to max_length first -- difflet's shared
        ``encode_prompt`` pads because Trainium compiles a fixed 512-token text
        stage, but T5 attention is masked, so real tokens never attend to the
        padding and the padded positions are zeroed afterwards either way; 16
        tokens instead of 512 is the same tensor for 8x less host work.

        Then the result crosses to the other ranks over the interconnect
        instead of being recomputed three more times. The payload is
        1 x 512 x 4096 bf16 = 4 MB, which is nothing next to a second of host
        matmul.
        """
        seq_len, dim = int(args["text_seq_len"]), int(app.config.text_dim)
        if is_encoder:
            from difflet.models.wan.pipeline import _zero_padding_embeds

            tokenized = tokenizer(
                [prompt], padding="longest", truncation=True,
                max_length=seq_len, return_tensors="pt",
            )
            ids = tokenized["input_ids"].to(torch.int64)
            mask = tokenized["attention_mask"].to(torch.int32)
            with torch.no_grad():
                embeds = text_encoder(ids, mask).last_hidden_state
            embeds = _zero_padding_embeds(embeds, mask)
            pad = seq_len - embeds.shape[1]
            if pad > 0:
                embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad))
            embeds = embeds.to(torch.bfloat16)
        else:
            # Shape must match on every rank for the collective to be legal;
            # it is fixed by the compiled text length, not by the prompt.
            embeds = torch.zeros(1, seq_len, dim, dtype=torch.bfloat16)

        on_device = embeds.to(device)
        xm.collective_broadcast([on_device], root_ordinal=0)
        xm.mark_step()
        return on_device.cpu()
    log(f"boundary_ratio={orchestrator.boundary_ratio} "
        f"scheduler={type(orchestrator.scheduler).__name__}")

    total_iters = args["iters"] + args["natural_iters"]
    for iteration in range(total_iters):
        sync_now["on"] = iteration < args["iters"]
        timer.stamps.clear()
        generator = torch.Generator().manual_seed(args["seed"])
        mark = time.monotonic()
        prompt_embeds = encode(PROMPT)
        encode_seconds = time.monotonic() - mark
        wall = time.monotonic()
        out = orchestrator(
            prompt_embeds=prompt_embeds,
            num_inference_steps=args["steps"],
            guidance_scale=args["guidance"],
            generator=generator,
            output_type="latent",
        )
        total = time.monotonic() - wall
        steps = timer.deltas()
        latents = out.latents
        basis = "synced" if sync_now["on"] else "natural"
        log(f"iter{iteration} [{basis}] denoise={total:.2f}s "
            f"encode={encode_seconds:.2f}s steps={len(steps)} "
            f"mean_step={sum(steps) / max(1, len(steps)):.3f}s "
            f"throughput_step={total / (len(steps) + 1):.3f}s")

        if rank == 0:
            reply_q.put({
                "type": "result", "iteration": iteration, "basis": basis,
                "throughput_step_seconds": total / (len(steps) + 1),
                "wall_seconds": total, "encode_seconds": encode_seconds,
                "load_seconds": load_seconds,
                "step_seconds": list(steps),
                "mem_before": mem_before, "mem_after_load": mem_after_load,
                "mem_after_run": _mem(xm, device),
                "latent_shape": list(latents.shape),
                "finite": bool(latents.isfinite().all()),
                "min": float(latents.min()), "max": float(latents.max()),
                "mean": float(latents.mean()), "std": float(latents.std()),
            })

    # --- decode (rank 0 only, on host: the VAE is 3D-conv heavy and the
    # chips are the scarce resource, not the 112 host cores) ---------------
    if rank == 0 and args["decode"]:
        from diffusers import AutoencoderKLWan

        mark = time.monotonic()
        vae = AutoencoderKLWan.from_pretrained(
            str(Path(args["model_dir"]) / "vae"), torch_dtype=torch.float32
        ).eval()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        z = latents.float() * std + mean
        where = "device"
        try:
            # The DiT leaves ~8.7 GiB free on this chip, so unlike Qwen-Image
            # at 1024x1024 the VAE has room to visit the device.
            vae.to(device)
            with torch.no_grad():
                frames = vae.decode(z.to(device), return_dict=False)[0]
            xm.mark_step()
            frames = frames.cpu()
        except Exception as exc:  # noqa: BLE001
            log(f"device decode failed ({exc}); falling back to host")
            where = "host"
            vae.to("cpu")
            with torch.no_grad():
                frames = vae.decode(z, return_dict=False)[0]
        reply_q.put({"type": "decoded", "seconds": time.monotonic() - mark,
                     "where": where, "shape": list(frames.shape),
                     "finite": bool(frames.isfinite().all())})
        torch.save(latents, args["out"] + ".latents.pt")
        video = (frames[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2
        import numpy as np
        np.save(args["out"] + ".npy", video.numpy())
        try:
            import imageio
            imageio.mimsave(args["out"] + ".mp4",
                            (video.numpy() * 255).astype("uint8"), fps=8)
        except Exception as exc:  # noqa: BLE001
            print(f"[rank0] mp4 write skipped: {exc}", flush=True)

    reply_q.put({"type": "done", "rank": rank})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=SNAP)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--natural-iters", type=int, default=2,
                        help="extra iterations with no per-step device sync")
    parser.add_argument("--both-experts", action="store_true")
    parser.add_argument("--no-decode", dest="decode", action="store_false")
    parser.add_argument("--out", default="/mnt/models/wan_tpu_out")
    parser.add_argument("--world", type=int, default=0)
    options = parser.parse_args()

    world = options.world or len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    args = {
        "model_dir": options.model_dir, "height": options.height,
        "width": options.width, "num_frames": options.num_frames,
        "steps": options.steps, "guidance": options.guidance,
        "seed": options.seed, "iters": options.iters,
        "natural_iters": options.natural_iters,
        "both_experts": options.both_experts, "decode": options.decode,
        "text_seq_len": 512,
        "out": options.out,
    }
    print(f"world={world} args={args}", flush=True)

    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, args, reply_q), daemon=False)
             for r in range(world)]
    for p in procs:
        p.start()

    results = []
    done = 0
    failed = False
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
    raise SystemExit(main())
