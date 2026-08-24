"""Can Wan's denoise loop drop the per-step out.cpu()?

The DiT output goes to UniPCMultistepScheduler, which diffusers deliberately
keeps on the host (`self.sigmas = self.sigmas.to("cpu")  # to avoid too much
CPU/GPU communication`). Moving the loop onto the chip is therefore not just
deleting a transfer; the question is what XLA does with the per-step scalars.

Three variants, same real orchestrator:
  host     - as shipped: latents on host fp32, out.cpu() every step
  device   - latents on device, scheduler sigmas left on CPU
  device+  - latents on device, sigmas moved to the device too

If a variant recompiles per step it shows up immediately as a huge first-N
step time rather than a steady state.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

SNAP = os.environ.get(
    "DIFFLET_WAN_SNAPSHOT",
    "/mnt/models/hf/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
STEPS = 8


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

    from difflet.models.wan.entry import create_wan_application
    from difflet.models.wan.pipeline import WanOrchestrator
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    device = torch_xla.device()
    app = create_wan_application(
        model_path=SNAP, parallel=DiffletParallelConfig(tp_degree=world),
        dtype=torch.bfloat16,
        shape={"height": 480, "width": 832, "num_frames": 9},
        backend="tpu",
    )
    module = app.transformer._prepare_module().to(device)
    xm.mark_step()
    cfg = app.config

    # No text encoder: four replicas each holding an fp32 umT5 is ~48 GB of
    # host RAM apiece and OOM-kills the box, which has nothing to do with what
    # this probe measures. The loop's behaviour does not depend on the
    # embedding values, so feed it a correctly-shaped constant.
    embeds = torch.zeros(1, int(cfg.text_seq_len), int(cfg.text_dim),
                         dtype=torch.bfloat16)

    steps: list[float] = []

    class _Wrap:
        """Return the DiT output on the host or leave it on the chip."""

        def __init__(self, keep_on_device, wait=True):
            self.keep = keep_on_device
            self.wait = wait
            self.config = cfg
            self.dtype = torch.bfloat16

        def __call__(self, hidden_states, timestep, encoder_hidden_states):
            mark = time.monotonic()
            out = module(hidden_states.to(device), timestep.to(device),
                         encoder_hidden_states.to(device))
            xm.mark_step()
            if not self.keep:
                out = out.cpu()
            elif self.wait:
                xm.wait_device_ops()
            steps.append(time.monotonic() - mark)
            return out

    def build(keep_on_device, wait=True):
        o = WanOrchestrator(
            model_path=SNAP, text_encoder=None,
            transformer=_Wrap(keep_on_device, wait), transformer_2=None,
            vae_decoder=None, dtype=torch.bfloat16,
            height=480, width=832, num_frames=9,
            tokenizer_path=str(Path(SNAP) / "tokenizer"),
        )
        return o

    def run(label, keep_on_device, sigmas_to_device, wait=True):
        o = build(keep_on_device, wait)
        # Always seed on the host: XLA's RNG is a separate question from the
        # one under test, and this keeps the two variants bit-comparable.
        latents = o.prepare_latents(
            batch_size=1, dtype=torch.float32,
            generator=torch.Generator().manual_seed(42),
        )
        if keep_on_device:
            latents = latents.to(device)
            xm.mark_step()
        print(f"[rank{rank}] {label}: latents on {latents.device}", flush=True)
        if sigmas_to_device and o.scheduler is not None:
            o.scheduler.set_timesteps(STEPS, device=device)
            o.scheduler.sigmas = o.scheduler.sigmas.to(device)
        steps.clear()
        mark = time.monotonic()
        try:
            out = o(prompt_embeds=embeds, latents=latents,
                    num_inference_steps=STEPS, guidance_scale=1.0,
                    output_type="latent")
            lat0 = out.latents
            if lat0.device.type == "xla":
                xm.wait_device_ops()      # the loop's single sync
            total = time.monotonic() - mark
            lat = out.latents
            ok = bool(lat.isfinite().all())
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                reply_q.put({"label": label, "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
            return
        if rank == 0:
            reply_q.put({
                "label": label, "total": total, "finite": ok,
                "steps": [round(s, 3) for s in steps],
                "mean_tail": sum(steps[2:]) / max(1, len(steps) - 2),
            })

    run("host (as shipped)", False, False)
    run("device, sigmas on cpu", True, False)
    run("device, sigmas on device", True, True)
    run("device+sigmas, NO per-step wait", True, True, wait=False)
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
            msg = reply_q.get(timeout=300)
        except Exception:  # noqa: BLE001
            continue
        if "done" in msg:
            seen += 1
            if msg.get("traceback"):
                print(msg["traceback"], flush=True)
        elif "error" in msg:
            print(f"{msg['label']:26s} FAILED: {msg['error']}", flush=True)
        else:
            print(f"{msg['label']:26s} total {msg['total']:6.2f}s  "
                  f"tail-mean {msg['mean_tail']*1000:7.1f} ms  finite={msg['finite']}",
                  flush=True)
            print(f"{'':26s} steps: {msg['steps']}", flush=True)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
