#!/usr/bin/env python3
"""Phase G's CPU fp32 reference for Qwen-Image: diffusers' ``QwenImagePipeline`` in fp32 on the
host CPU with the settings of the device runs it scores (``qwen_bf16_tp4_ref2`` bf16,
``qwen_mx_hm0_tp4`` the shipped MXFP8 default), from the device's initial noise, saved through
``save_run`` for ``score.py``.

Every setting is Qwen-Image's research object's (``/home/ubuntu/fp8-bench/phase_g_research.json``,
``device_settings``); only the dtype and the device differ. The device draws its noise in bf16 on
rank 0's host from a CPU generator (difflet's ``prepare_latents``), the fp32 draw rounded to bf16,
so the reference injects that draw upcast (exact), checked against difflet's own. The prompt is
encoded in fp32 here (the device's 11 rows after the template's drop); the final latents are the
packed (1, 4096, 64) the device's latents.pt holds; the image is decoded in fp32 by difflet's
host decode, which feeds the VAE the vae stage's input and writes the PNG as the vae stage does
(round half up, where diffusers' postprocess rounds half to even).

    taskset -c 0-47,96-143 env TORCH_DEVICE_BACKEND_AUTOLOAD=0 DIFFLET_BACKEND=cpu \\
        PYTHONPATH=<repo> python scripts/fp32_ref/qwen_image.py --out-dir /path/to/ref

(``TORCH_DEVICE_BACKEND_AUTOLOAD=0``: torch does not import torch_neuronx, so the run never
touches a Neuron device.)
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
import time
from pathlib import Path

import torch
from diffusers import QwenImagePipeline
from PIL import Image

from difflet.cli.orchestrators.qwen_image import _decode_latents_host
from difflet.models.qwen_image.pipeline import QwenImageOrchestrator
from scripts.fp32_ref._common import (device_noise, require_fp32_math, save_run,
                                      set_threads, tensor_md5)

#: The local snapshot both device runs loaded with no --revision (research:
#: device_settings.revision; the gen logs' weight-store keys name 75e0b4be).
SNAPSHOT = Path("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/"
                "75e0b4be04f60ec59a75f475837eced720f823b6")
#: The device command's ``--prompt``, ``--steps 50`` and ``--seed 42`` (research:
#: device_settings.source).
PROMPT = "a cat sitting on a bench"
STEPS = 50
SEED = 42
#: The device's denoise, as QwenImagePipeline.__call__ arguments (research: device_settings).
CALL = {
    # The device command's --height 1024 --width 1024.
    "height": 1024,
    "width": 1024,
    # No true CFG, one DiT forward per step: no --negative-prompt or --true-cfg-scale, so
    # _true_cfg_kwargs returns {}. diffusers' own default true_cfg_scale is 4.0.
    "negative_prompt": None,
    "true_cfg_scale": 1.0,
    # The command's --guidance-scale 7.5 is dropped (_OnDeviceTransformer deletes it: the
    # checkpoint has guidance_embeds=false); diffusers would ignore it too, with a warning.
    "guidance_scale": None,
}
#: The prompt's rows after the template's 34 are dropped: 45 tokens keep 11 (research:
#: device_settings.other), which the device's DiT attends to unmasked.
TEXT_ROWS = 11
#: diffusers' and difflet's unpacked draw (1, 1, 16, H/8, W/8).
NOISE_SHAPE = (1, 1, 16, 128, 128)


def initial_latents() -> tuple[torch.Tensor, str]:
    """The device's initial noise as fp32, packed (1, 4096, 64), and the md5 of its bf16 draw:
    ``device_noise``'s bf16 law, refused unless torch.equal to difflet's own draw (the
    orchestrator's ``prepare_latents`` at the bf16 pipeline dtype from a CPU generator, which the
    device ran on rank 0's host). diffusers returns supplied latents unchanged."""
    _, _, channels, height, width = NOISE_SHAPE
    noise = device_noise(NOISE_SHAPE, SEED, bf16_draw=True)
    latents = QwenImagePipeline._pack_latents(noise, 1, channels, height, width)
    device = QwenImageOrchestrator(
        model_path=str(SNAPSHOT), transformer=None, dtype=torch.bfloat16,
        height=CALL["height"], width=CALL["width"],
    ).prepare_latents(batch_size=1, channels=channels, dtype=torch.bfloat16,
                      generator=torch.Generator().manual_seed(SEED))
    if not torch.equal(latents, device.float()):
        raise RuntimeError("the injected noise is not difflet's bf16 seed draw")
    return latents, tensor_md5(device)


def encode(pipe) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The prompt's fp32 embeddings and mask, refused unless they are the device's 11 unmasked
    rows: any other count would mean another prompt or template than the device encoded."""
    embeds, mask = pipe.encode_prompt(prompt=PROMPT, device="cpu")
    if embeds.shape[1] != TEXT_ROWS:
        raise RuntimeError(f"{embeds.shape[1]} prompt rows, not the device's {TEXT_ROWS}")
    if mask is not None:
        raise RuntimeError("the prompt came with a mask: the device's rows are all valid")
    return embeds, mask


def decode(packed: torch.Tensor) -> Image.Image:
    """The device's PNG of the packed latents, decoded in fp32: difflet's host decode
    (``_decode_latents_host``: diffusers' AutoencoderKLQwenImage loaded in fp32 on the vae stage's
    input, written through ``_save_image_pil``), which reads and writes files."""
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(packed, f"{tmp}/latents.pt")
        _decode_latents_host(f"{tmp}/latents.pt", str(SNAPSHOT), f"{tmp}/out.png",
                             height=CALL["height"], width=CALL["width"])
        with Image.open(f"{tmp}/out.png") as image:
            return image.copy()


def require_fp32(components: dict) -> None:
    """Refuse a pipeline module holding any floating tensor other than fp32: a component left in
    bf16 (the hub's dtype for the DiT and the text encoder) would put its rounding into the
    reference."""
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
                   help=f"denoise steps ({STEPS}, the device's; a smoke run uses 2: at 1 the "
                        "scheduler's shift_terminal stretch divides by zero)")
    args = p.parse_args(argv)
    require_fp32_math()
    set_threads(args.threads)
    torch.set_grad_enabled(False)
    latents, md5 = initial_latents()

    started = time.perf_counter()
    # The VAE loads in decode(), through the device's own host-decode path.
    pipe = QwenImagePipeline.from_pretrained(SNAPSHOT, vae=None, torch_dtype=torch.float32)
    require_fp32(pipe.components)
    loaded = time.perf_counter()
    embeds, mask = encode(pipe)
    encoded = time.perf_counter()
    ends = []

    def on_step_end(_pipe, _i, _t, callback_kwargs):
        ends.append(time.perf_counter())
        return callback_kwargs

    packed = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask, **CALL,
                  num_inference_steps=args.steps, latents=latents, output_type="latent",
                  callback_on_step_end=on_step_end).images
    if not torch.isfinite(packed).all():
        # One step's schedule is all NaN (shift_terminal); a NaN reference scores as garbage.
        raise RuntimeError(f"non-finite final latents after {args.steps} steps")
    del pipe
    image = decode(packed)
    done = time.perf_counter()
    save_run(args.out_dir, latents=packed, output=image,
             settings={"snapshot": str(SNAPSHOT), "prompt": PROMPT, **CALL, "steps": args.steps,
                       "seed": SEED, "text_rows": TEXT_ROWS,
                       "noise": "bf16 CPU draw upcast, packed (1, 4096, 64)",
                       "cpu_capability": torch.backends.cpu.get_cpu_capability()},
             noise_md5=md5,
             timings={"load_s": loaded - started, "encode_s": encoded - loaded,
                      "to_first_step_end_s": ends[0] - encoded,
                      "step_s": [b - a for a, b in zip(ends, ends[1:])],
                      "decode_s": done - ends[-1], "wall_s": done - started})
    print(f"[fp32-ref qwen_image] wrote {args.out_dir} in {done - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
