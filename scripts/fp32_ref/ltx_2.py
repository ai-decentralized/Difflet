#!/usr/bin/env python3
"""Phase G's CPU fp32 reference for LTX-2: diffusers' ``LTX2Pipeline`` in fp32 on the host CPU
with the settings and the initial noise of the device runs ``score.py`` scores (ltx2_tp4/
{bf16,mx}_gen50a, 2026-11-07), saved the way those runs saved theirs.

The device runs were ``difflet generate --model-id Lightricks/LTX-2 --backend neuron
--tp-degree 4 --height 512 --width 768 --num-frames 121 --steps 50 --seed 42 --prompt "a cat
walking through a garden" [--quant mxfp8]``; every constant below is copied from the LTX-2 object
of ``/home/ubuntu/fp8-bench/phase_g_research.json`` and names its device source. Where that run
differs from diffusers' defaults the reference passes the device's value: guidance 3.5 (diffusers
4.0) and 50 steps (40). The noise is the device's bf16 draw, injected through ``prepare_latents``
and ``prepare_audio_latents`` (``audio_latents=`` would be normalized by the audio VAE's stats).
The final latents are taken from the last step's callback, packed and normalized as the device's
``latents.pt`` holds them; diffusers' ``output_type="latent"`` returns them unpacked and
de-normalized.

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/ltx_2.py --out-dir <dir> --threads 96
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from diffusers import LTX2Pipeline
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub.constants import HF_HUB_CACHE

from difflet.models.ltx_2.pipeline import LTX2Orchestrator
from scripts.fp32_ref._common import (DEVICE_FPS, require_fp32_math, save_run, set_threads,
                                      tensor_md5)

#: The device command's --model-id, at the snapshot its runs loaded: refs/main of the local
#: snapshot (the CLI got no --revision; the logs' weight-store keys read ...__64f65d09).
MODEL_ID = "Lightricks/LTX-2"
REVISION = "64f65d09e0b76c96202b5bc09997673184cf9b5d"
SNAPSHOT = Path(HF_HUB_CACHE) / "models--Lightricks--LTX-2" / "snapshots" / REVISION
#: The device command: --height 512 --width 768 --num-frames 121 --steps 50 --seed 42 --prompt.
HEIGHT, WIDTH, NUM_FRAMES = 512, 768, 121
STEPS = 50
SEED = 42
PROMPT = "a cat walking through a garden"
#: The command gave no --negative-prompt: the device's text encoder encodes
#: ``[negative_prompt or "", prompt]`` (neuron_application.py ``_DeviceLtx2TextEncoder``), as
#: diffusers' ``encode_prompt`` does.
NEGATIVE_PROMPT = ""
#: The command gave no --guidance-scale: the driver's ``_DEFAULT_GUIDANCE`` (ltx_2.py), true CFG
#: with two DiT forwards per step; the audio guidance follows it (pipeline.py ``__call__``).
GUIDANCE_SCALE = 3.5
#: ``LTX2Orchestrator.__call__``'s defaults, which the device request leaves alone: no CFG
#: rescale, no STG, no modality guidance, no cross timestep.
GUIDANCE_RESCALE = 0.0
STG_SCALE = 0.0
MODALITY_SCALE = 1.0
USE_CROSS_TIMESTEP = False
#: The ranks' ``text_seq_len`` (ltx_2.py ``_TORCHRUN_APPLICATION_KWARGS``).
MAX_SEQUENCE_LENGTH = 1024
#: The application's ``frame_rate`` (neuron_application.py ``_KWARG_DEFAULTS``): it sets the
#: RoPE coords and the audio length; the mp4 is written at ``DEVICE_FPS["ltx_2"]``.
FRAME_RATE = 24.0
#: The unpacked noise both sides draw, in this order from one CPU generator: video [B, C, F, H, W]
#: = [1, 128, (121 - 1) // 8 + 1, 512 // 32, 768 // 32], then audio [B, C, L, M] = [1, 8,
#: round(121 / 24 * 16000 / 160 / 4), 64 // 4] (research: noise).
VIDEO_NOISE_SHAPE = (1, 128, 16, 16, 24)
AUDIO_NOISE_SHAPE = (1, 8, 126, 16)
#: ``tensor_md5`` of the packed bf16 device draw the research saved with the device runs' venv
#: and torch (/tmp/ltx2_phase_g_check/difflet_initial_seed42.pt): a torch whose generator drew
#: otherwise would pass the law check on both sides.
NOISE_MD5 = {"video": "d4d7266802e324f4553c3ea547e1ddfc",
             "audio": "acdb384e9f3763669471623ec320e9b2"}


def reference_noise(seed: int = SEED) -> dict[str, torch.Tensor]:
    """The noise the reference injects, packed as the pipeline packs it: diffusers' own fp32
    draws (video, then audio, from one CPU generator, as ``prepare_latents`` and
    ``prepare_audio_latents`` draw them) rounded to bf16 and back, the plan's law for LTX-2. Not
    ``device_noise``: it seeds a generator per call, and the audio is the generator's second
    draw."""
    generator = torch.Generator().manual_seed(seed)
    video = randn_tensor(VIDEO_NOISE_SHAPE, generator=generator, dtype=torch.float32)
    audio = randn_tensor(AUDIO_NOISE_SHAPE, generator=generator, dtype=torch.float32)
    return {"video": LTX2Pipeline._pack_latents(video).to(torch.bfloat16).float(),
            "audio": LTX2Pipeline._pack_audio_latents(audio).to(torch.bfloat16).float()}


def device_initial_latents(model_path=SNAPSHOT, seed: int = SEED) -> dict[str, torch.Tensor]:
    """The device's own initial noise, by its own code: ``LTX2Orchestrator``'s draws as
    ``TorchNeuronLtx2Application._initial_latents`` makes them on rank 0 (video then audio in the
    bf16 application dtype, packed; the request's CPU generator seeded by ltx_2.py's
    ``request_kwargs``). The orchestrator reads no weights, only the scheduler config."""
    orchestrator = LTX2Orchestrator(
        model_path=str(model_path), dtype=torch.bfloat16, height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES, text_seq_len=MAX_SEQUENCE_LENGTH, frame_rate=FRAME_RATE)
    generator = torch.Generator().manual_seed(seed)
    video = orchestrator.prepare_latents(batch_size=1, channels=128, dtype=torch.bfloat16,
                                         generator=generator)
    audio = orchestrator.prepare_audio_latents(batch_size=1, channels=8, dtype=torch.bfloat16,
                                               generator=generator)
    return {"video": video, "audio": audio}


def call_kwargs(steps: int) -> dict:
    """``LTX2Pipeline.__call__``'s arguments: the device run's settings, and frames out as
    (B, F, 3, H, W) in [0, 1] (``output_type="pt"``, what the device's decode hands
    ``_save_video``). The sigmas are left to diffusers: ``linspace(1, 1/N, N)`` shifted by mu at
    ``max_image_seq_len``, the device's ``LTX2Orchestrator._timesteps``. No generator: the noise
    is injected, and the VAE has no timestep conditioning, so nothing else draws."""
    return dict(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, height=HEIGHT, width=WIDTH,
                num_frames=NUM_FRAMES, frame_rate=FRAME_RATE, num_inference_steps=steps,
                guidance_scale=GUIDANCE_SCALE, audio_guidance_scale=GUIDANCE_SCALE,
                guidance_rescale=GUIDANCE_RESCALE, stg_scale=STG_SCALE,
                modality_scale=MODALITY_SCALE, use_cross_timestep=USE_CROSS_TIMESTEP,
                max_sequence_length=MAX_SEQUENCE_LENGTH, output_type="pt")


def assert_fp32(pipe) -> None:
    """Refuse a component holding a floating tensor in another dtype: the reference is fp32 end
    to end, text encoder, VAEs and vocoder included, or their rounding is scored as the device's
    error."""
    for name, component in pipe.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        dtypes = {t.dtype for t in (*component.parameters(), *component.buffers())
                  if t.is_floating_point()}
        if dtypes - {torch.float32}:
            raise RuntimeError(f"{name} holds {sorted(map(str, dtypes))}, not only float32")


def _inject(tensor: torch.Tensor):
    """A ``prepare_latents`` / ``prepare_audio_latents`` replacement returning ``tensor``."""
    return lambda *args, **kwargs: tensor


class LastStep:
    """The step callback: each step's wall time (the first includes the prompt encode) and the
    last step's video and audio latents, packed and normalized: the device's latents.pt."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.step_s: list[float] = []
        self.latents: dict[str, torch.Tensor] | None = None
        self.last = time.perf_counter()

    def __call__(self, pipe, step, timestep, callback_kwargs):
        now = time.perf_counter()
        self.step_s.append(now - self.last)
        self.last = now
        print(f"[fp32_ref ltx_2] step {step + 1}/{self.steps} {self.step_s[-1]:.1f} s", flush=True)
        if step == self.steps - 1:
            self.latents = {"video": callback_kwargs["latents"].detach().clone(),
                            "audio": callback_kwargs["audio_latents"].detach().clone()}
        return callback_kwargs


def device_clip(frames: torch.Tensor) -> np.ndarray:
    """The (F, H, W, 3) float32 frames the device's ``_save_video`` (ltx_2.py) hands
    ``export_to_video``: batch 0 of the (B, F, 3, H, W) [0, 1] decode, clamped."""
    return frames.detach().float().clamp(0.0, 1.0)[0].permute(0, 2, 3, 1).cpu().numpy()


def _steps(value: str) -> int:
    """At least 2: one LTX-2 step is a NaN schedule in diffusers (shift_terminal's stretch
    divides 0 by 0; the device's ``_timesteps`` refuses it too), refused before the load."""
    steps = int(value)
    if steps < 2:
        raise argparse.ArgumentTypeError(f"LTX-2 needs at least 2 steps, not {steps}")
    return steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=96)
    p.add_argument("--steps", type=_steps, default=STEPS)
    p.add_argument("--model-path", type=Path, default=SNAPSHOT)
    args = p.parse_args(argv)
    require_fp32_math()
    set_threads(args.threads)
    start = time.perf_counter()

    noise = reference_noise()
    device = device_initial_latents(args.model_path)
    for key, tensor in device.items():
        if not torch.equal(noise[key], tensor.float()):
            raise RuntimeError(f"the injected {key} noise is not the device's draw")
    md5 = {key: tensor_md5(tensor) for key, tensor in device.items()}
    if md5 != NOISE_MD5:
        raise RuntimeError(f"the device draw hashes to {md5}, not the research's {NOISE_MD5}")
    noise_md5 = ",".join(f"{key}={value}" for key, value in md5.items())

    load_start = time.perf_counter()
    pipe = LTX2Pipeline.from_pretrained(str(args.model_path), torch_dtype=torch.float32)
    assert_fp32(pipe)
    load_s = time.perf_counter() - load_start
    pipe.set_progress_bar_config(disable=True)
    pipe.prepare_latents = _inject(noise["video"])
    pipe.prepare_audio_latents = _inject(noise["audio"])
    # check_inputs admits only _callback_tensor_inputs, which leaves out the audio latents; it
    # reads the attribute, so an instance override admits them (the loop has them in locals()).
    pipe._callback_tensor_inputs = [*LTX2Pipeline._callback_tensor_inputs, "audio_latents"]
    recorder = LastStep(args.steps)
    kwargs = call_kwargs(args.steps)
    call_start = time.perf_counter()
    out = pipe(**kwargs, callback_on_step_end=recorder,
               callback_on_step_end_tensor_inputs=["latents", "audio_latents"])
    end = time.perf_counter()
    if recorder.latents is None:
        raise RuntimeError(f"the step callback never saw step {args.steps}")

    settings = {"model_id": MODEL_ID, "revision": REVISION, "model_path": str(args.model_path),
                "seed": SEED, **kwargs, "dtype": "float32", "device": "cpu",
                "noise": "difflet's bf16 draw (video then audio, CPU generator) as fp32",
                "latents": "packed and normalized, from the last step's callback",
                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "transformers": transformers.__version__}
    timings = {"load_s": load_s, "step_s": recorder.step_s,
               "decode_s": end - recorder.last, "call_s": end - call_start,
               "wall_s": end - start}
    save_run(args.out_dir, latents=recorder.latents, output=device_clip(out.frames),
             settings=settings, noise_md5=noise_md5, timings=timings, fps=DEVICE_FPS["ltx_2"])
    print(f"[fp32_ref ltx_2] wrote {args.out_dir} in {time.perf_counter() - start:.0f} s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
