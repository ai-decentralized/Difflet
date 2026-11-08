#!/usr/bin/env python3
"""Phase G's CPU fp32 reference for FLUX.1-dev: diffusers' ``FluxPipeline`` in fp32 on the host
CPU with the settings of the device runs it scores (``flux_d_tp4``: A1 bf16, B1 the shipped
MXFP8 default), from the device's initial noise, saved through ``save_run`` for ``score.py``.

Every setting is FLUX's research object's (``/home/ubuntu/fp8-bench/phase_g_research.json``,
``device_settings``); only the dtype and the device differ. The device draws its noise in its
bf16 application dtype on a CPU generator (diffusers' ``prepare_latents``), the fp32 draw rounded
to bf16, so the reference injects that draw upcast (exact), packed as the device packed it.
The final latents are kept packed (1, 4096, 64), the layout a device latents.pt would have (the
tp4 runs decode in process and kept none); the image is diffusers' PIL output, which the device
saved as is.

    taskset -c 0-47,96-143 env TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=<repo> \\
        python scripts/fp32_ref/flux.py --out-dir /path/to/ref --threads 96

(``TORCH_DEVICE_BACKEND_AUTOLOAD=0``: torch does not import torch_neuronx, so the run never
touches a Neuron device.)
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

from scripts.fp32_ref._common import (device_noise, require_fp32_math, save_run,
                                      set_threads, tensor_md5)

#: refs/main, the only FLUX snapshot on the host, which the device resolved with no --revision
#: (research: device_settings.revision; B1_gena.log's quantized dir names 3de623fc).
SNAPSHOT = Path("/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/"
                "snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
#: run_steps_2_5.sh's ``generate --steps 50 --seed 42`` (diffusers' own default is 28 steps).
STEPS = 50
SEED = 42
#: The device's FluxPipeline.__call__ arguments (research: device_settings).
CALL = {
    # run_steps_2_5.sh: --prompt 'a cat sitting on a bench' --height 1024 --width 1024.
    "prompt": "a cat sitting on a bench",
    "height": 1024,
    "width": 1024,
    # Embedded (distilled) guidance: _stage_transformer_neuron's
    # stage_loop.effective(req, args, "guidance_scale", 3.5), no --guidance-scale given.
    "guidance_scale": 3.5,
    # No true CFG: the orchestrator passes no negative prompt and leaves true_cfg_scale at 1.0.
    "negative_prompt": None,
    "true_cfg_scale": 1.0,
    # TorchNeuronFluxApplication.__call__'s setdefault from its text_seq_len 512.
    "max_sequence_length": 512,
}
#: The unpacked draw (1, 16, H/8, W/8) and the md5 of the device's packed bf16 draw at seed 42
#: (research: noise.evidence).
NOISE_SHAPE = (1, 16, 128, 128)
NOISE_MD5 = "be1ad0a63781cbc9b93ec9007d44b4ad"


def initial_latents() -> tuple[torch.Tensor, str]:
    """The device's initial noise as fp32, packed (1, 4096, 64) as ``prepare_latents`` packs it,
    and the md5 of its bf16 draw, refused unless it is the research's. ``prepare_latents``
    returns supplied latents unchanged, so the first step sees exactly this tensor."""
    _, channels, height, width = NOISE_SHAPE
    noise = device_noise(NOISE_SHAPE, SEED, bf16_draw=True)
    latents = FluxPipeline._pack_latents(noise, 1, channels, height, width)
    md5 = tensor_md5(latents.to(torch.bfloat16))
    if md5 != NOISE_MD5:
        raise RuntimeError(f"initial noise md5 {md5} is not the device's {NOISE_MD5}")
    return latents, md5


def require_fp32(components: dict) -> None:
    """Refuse a pipeline module holding any floating tensor other than fp32: a component left in
    bf16 (the hub's dtype for every FLUX checkpoint) would put its rounding into the reference."""
    for name, module in components.items():
        if not isinstance(module, torch.nn.Module):
            continue
        dtypes = {t.dtype for t in itertools.chain(module.parameters(), module.buffers())
                  if t.is_floating_point()}
        if dtypes - {torch.float32}:
            raise RuntimeError(f"{name} holds {sorted(map(str, dtypes))}, not only float32")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=96,
                   help="torch threads: the vCPUs of the NUMA node taskset pinned the run to")
    p.add_argument("--steps", type=int, default=STEPS,
                   help=f"denoise steps ({STEPS}, the device's; a smoke run uses 1)")
    args = p.parse_args(argv)
    require_fp32_math()
    set_threads(args.threads)
    torch.set_grad_enabled(False)
    latents, md5 = initial_latents()

    started = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(SNAPSHOT, torch_dtype=torch.float32)
    require_fp32(pipe.components)
    loaded = time.perf_counter()
    ends, final = [], []

    def on_step_end(_pipe, i, _t, callback_kwargs):
        """Each step's end; the last step's latents, which the pipeline then decodes."""
        ends.append(time.perf_counter())
        if i == args.steps - 1:
            final.append(callback_kwargs["latents"].detach().clone())
        return callback_kwargs

    out = pipe(**CALL, num_inference_steps=args.steps, latents=latents,
               callback_on_step_end=on_step_end, output_type="pil")
    done = time.perf_counter()
    save_run(args.out_dir, latents=final[0], output=out.images[0],
             settings={"snapshot": str(SNAPSHOT), **CALL, "steps": args.steps, "seed": SEED,
                       "noise": "bf16 CPU draw upcast, packed (1, 4096, 64)",
                       "cpu_capability": torch.backends.cpu.get_cpu_capability()},
             noise_md5=md5,
             timings={"load_s": loaded - started,
                      "to_first_step_end_s": ends[0] - loaded,
                      "step_s": [b - a for a, b in zip(ends, ends[1:])],
                      "after_last_step_s": done - ends[-1], "wall_s": done - started})
    print(f"[fp32-ref flux] wrote {args.out_dir} in {done - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
