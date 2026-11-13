"""What phase G's two Wan references (``wan21.py``, ``wan22.py``) share: diffusers' WanPipeline
in fp32 on the host CPU, run with the device runs' settings from the device's own initial noise
and saved as the device saved its run.

The device draws Wan's noise in fp32 from a CPU generator (rank 0's
``WanOrchestrator.prepare_latents``, broadcast in fp32), the same draw diffusers' fp32
``prepare_latents`` makes; the reference still injects it as ``latents=`` after asserting it
equals difflet's draw, so a change on either side fails here instead of being scored as device
error. The final latents are taken at the last step's callback: the tensor the pipeline then
decodes, in the scheduler's normalised space, which is what the device's ``latents.pt`` holds.
The frames are diffusers' own fp32 decode (``output_type="np"``: float [0, 1], the device's
clamp then (x + 1) / 2 in the same fp32 arithmetic).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler, WanPipeline

from difflet.models.wan.pipeline import WanOrchestrator
from difflet.pipeline.path_resolver import resolve_model_path
from scripts.fp32_ref import _common

EXPERTS = ("transformer", "transformer_2")


def reference_noise(settings: dict, snapshot) -> torch.Tensor:
    """The device's initial noise: the fp32 draw at ``noise_shape``, asserted bit-equal to
    difflet's own draw at the request's shape (``_initial_latents``' call, dtype included)."""
    noise = _common.device_noise(settings["noise_shape"], settings["seed"], bf16_draw=False)
    device = WanOrchestrator(model_path=str(snapshot)).prepare_latents(
        batch_size=1, channels=settings["noise_shape"][1], height=settings["height"],
        width=settings["width"], num_frames=settings["num_frames"], dtype=torch.float32,
        generator=torch.Generator().manual_seed(settings["seed"]))
    if not torch.equal(noise, device):
        raise AssertionError(f"the injected noise {tuple(noise.shape)} is not the device's draw "
                             f"{tuple(device.shape)} (seed {settings['seed']}, fp32)")
    return noise


def check_pipeline(pipe, settings: dict) -> None:
    """Refuse a pipeline that is not the device's model in fp32: a component holding another
    float dtype (the reference must be fp32 end to end, VAE and umT5 included), an expert set
    or ``boundary_ratio`` other than the device's, or a scheduler other than the snapshot's
    UniPC at the device's flow shift."""
    for name, component in pipe.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        dtypes = {t.dtype for t in (*component.parameters(), *component.buffers())
                  if t.is_floating_point()}
        if dtypes != {torch.float32}:
            raise ValueError(f"{name} holds {sorted(map(str, dtypes))}, not only torch.float32")
    if pipe.config.boundary_ratio != settings["boundary_ratio"]:
        raise ValueError(f"boundary_ratio {pipe.config.boundary_ratio}, the device ran "
                         f"{settings['boundary_ratio']}")
    if (pipe.transformer_2 is not None) != (settings["boundary_ratio"] is not None):
        raise ValueError("the low-noise expert must load exactly when the device switched experts")
    scheduler = pipe.scheduler
    if (not isinstance(scheduler, UniPCMultistepScheduler)
            or scheduler.config.flow_shift != settings["flow_shift"]):
        raise ValueError(f"scheduler {type(scheduler).__name__} flow_shift "
                         f"{getattr(scheduler.config, 'flow_shift', None)}, the device ran UniPC "
                         f"at {settings['flow_shift']}")


class _StepRecorder:
    """Step callback: wall seconds per step (the first carries the prompt encode) and the
    latents after the last step."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.step_s: list[float] = []
        self.final: torch.Tensor | None = None
        self.last = time.perf_counter()

    def __call__(self, pipe, step, timestep, callback_kwargs):
        now = time.perf_counter()
        self.step_s.append(now - self.last)
        self.last = now
        print(f"[fp32-ref] step {step + 1}/{self.steps} t={int(timestep)} "
              f"{self.step_s[-1]:.1f} s", flush=True)
        if step == self.steps - 1:
            self.final = callback_kwargs["latents"].detach().cpu().contiguous().clone()
        return callback_kwargs


def generate(pipe, settings: dict, steps: int, noise: torch.Tensor):
    """One device request through the fp32 pipeline: the decoded frames (F, H, W, 3) float
    [0, 1], the final latents, the per-step seconds, the decode seconds and the DiT forwards
    each expert ran (counted by hooks: the evidence for no CFG and for the expert switch)."""
    forwards: Counter = Counter()
    for name in EXPERTS:
        model = getattr(pipe, name, None)
        if model is not None:
            model.register_forward_pre_hook(
                lambda module, args, name=name: forwards.update([name]))
    recorder = _StepRecorder(steps)
    out = pipe(prompt=settings["prompt"], negative_prompt=settings["negative_prompt"],
               height=settings["height"], width=settings["width"],
               num_frames=settings["num_frames"], num_inference_steps=steps,
               guidance_scale=settings["guidance_scale"],
               guidance_scale_2=settings["guidance_scale_2"], latents=noise.clone(),
               max_sequence_length=settings["max_sequence_length"], output_type="np",
               callback_on_step_end=recorder)
    decode_s = time.perf_counter() - recorder.last
    return np.asarray(out.frames[0]), recorder.final, recorder.step_s, decode_s, dict(forwards)


def _check_outputs(frames: np.ndarray, latents: torch.Tensor, settings: dict) -> None:
    """The device layout: latents (1, 16, F_lat, H/8, W/8) fp32, ``num_frames`` frames at the
    device's height and width."""
    got = None if latents is None else (tuple(latents.shape), latents.dtype)
    if got != (tuple(settings["noise_shape"]), torch.float32):
        raise ValueError(f"final latents {got}, the device saved {settings['noise_shape']} fp32")
    expected = (settings["num_frames"], settings["height"], settings["width"], 3)
    if frames.shape != expected:
        raise ValueError(f"decoded frames {frames.shape}, the device wrote {expected}")


def main(key: str, settings: dict, dit_forwards: dict, argv=None) -> int:
    """Run ``key``'s reference: ``dit_forwards`` is the device's forwards per expert over the
    full ``settings["steps"]``, checked when the run takes them all (a smoke run takes one)."""
    p = argparse.ArgumentParser(
        description=f"CPU fp32 diffusers reference of {settings['model_id']} with the device "
                    f"runs' settings and noise (phase G).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=96,
                   help="torch intra-op threads: the vCPUs of the NUMA node taskset pins to")
    p.add_argument("--steps", type=int, default=settings["steps"],
                   help="denoising steps (the device ran %(default)s; a smoke run uses 1)")
    args = p.parse_args(argv)
    _common.require_fp32_math()
    _common.set_threads(args.threads)
    started = time.perf_counter()
    snapshot = resolve_model_path(settings["model_id"], revision=settings["revision"],
                                  local_files_only=True)
    noise = reference_noise(settings, snapshot)
    pipe = WanPipeline.from_pretrained(snapshot, torch_dtype=torch.float32)
    check_pipeline(pipe, settings)
    load_s = time.perf_counter() - started
    print(f"[fp32-ref] {key}: loaded {snapshot} in fp32 in {load_s:.0f} s, "
          f"{args.threads} threads", flush=True)
    frames, latents, step_s, decode_s, forwards = generate(pipe, settings, args.steps, noise)
    _check_outputs(frames, latents, settings)
    if sum(forwards.values()) != args.steps:
        raise RuntimeError(f"{forwards} DiT forwards over {args.steps} steps: the device ran one "
                           f"per step (guidance {settings['guidance_scale']}, no CFG)")
    if args.steps == settings["steps"] and forwards != dit_forwards:
        raise RuntimeError(f"DiT forwards per expert {forwards}, the device ran {dit_forwards}")
    _common.save_run(
        args.out_dir, latents=latents, output=frames,
        settings={**settings, "steps": args.steps, "snapshot": snapshot,
                  "dit_forwards": forwards},
        noise_md5=_common.tensor_md5(noise),
        timings={"load_s": load_s, "steps_s": step_s, "decode_s": decode_s,
                 "wall_s": time.perf_counter() - started},
        fps=_common.DEVICE_FPS[key])
    print(f"[fp32-ref] {key}: wrote {args.out_dir} in {time.perf_counter() - started:.0f} s",
          flush=True)
    return 0
