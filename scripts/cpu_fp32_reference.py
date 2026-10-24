#!/usr/bin/env python3
"""CPU fp32 reference generate for Qwen-Image or LTX-2, from a device run's noise.

Runs the diffusers pipeline in float32 on the host CPU, starting from the exact
latents a Difflet device run began its denoise loop with (the ``initial*.pt``
files written under ``DIFFLET_DUMP_LATENTS_DIR``). It writes ``final.pt``
(Qwen-Image) or ``final_video.pt`` (LTX-2) and the decoded ``out.png`` /
``out.mp4`` into ``--out-dir``, ready for ``scripts/quality_vs_reference.py``.

The supplied latents bypass the pipelines' ``prepare_latents`` (LTX-2's would
re-normalize and re-noise them), so both runs start from identical tensors.
To fit host RAM the prompt is encoded with only the text encoder loaded, which
is then freed before the transformer and decoders load.

LTX-2's step callback cannot see the audio latents, so only video latents are
recorded; audio is compared device-vs-device.

Usage:
    python scripts/cpu_fp32_reference.py --model qwen_image \
        --model-path <snapshot> --initial-latents-dir <device dump> --out-dir R
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

PROMPT = "a cinematic shot of a red fox running through a snowy forest"
CONFIGS = {
    "qwen_image": {"height": 1024, "width": 1024, "steps": 20},
    "ltx_2": {"height": 480, "width": 704, "num_frames": 49, "steps": 20, "frame_rate": 24.0},
}


def _fixed(tensor: torch.Tensor):
    """Replacement for a pipeline's prepare_*latents that returns ``tensor`` as is."""

    def prepare(*args, **kwargs):
        device = kwargs.get("device")
        return tensor.to(device=device, dtype=torch.float32)

    return prepare


class _StepRecorder:
    """Step callback: per-step wall time and the final latents."""

    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.step_times: list[float] = []
        self.final: torch.Tensor | None = None
        self._last = time.perf_counter()

    def __call__(self, pipe, step, timestep, callback_kwargs):
        now = time.perf_counter()
        self.step_times.append(now - self._last)
        self._last = now
        print(f"[cpu-ref] step {step + 1}/{self.num_steps} {self.step_times[-1]:.1f} s", flush=True)
        if step == self.num_steps - 1:
            self.final = callback_kwargs["latents"].detach().cpu().clone()
        return callback_kwargs


def run_qwen(model_path: str, initial_dir: Path, out_dir: Path, steps: int) -> dict:
    from diffusers import QwenImagePipeline

    cfg = CONFIGS["qwen_image"]
    encoder = QwenImagePipeline.from_pretrained(
        model_path, transformer=None, vae=None, torch_dtype=torch.float32
    )
    prompt_embeds, prompt_embeds_mask = encoder.encode_prompt(prompt=PROMPT, device="cpu")
    del encoder
    gc.collect()

    pipe = QwenImagePipeline.from_pretrained(
        model_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float32
    )
    pipe.prepare_latents = _fixed(torch.load(initial_dir / "initial.pt"))
    recorder = _StepRecorder(steps)
    image = pipe(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        true_cfg_scale=1.0,
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=steps,
        callback_on_step_end=recorder,
    ).images[0]
    torch.save(recorder.final, out_dir / "final.pt")
    image.save(out_dir / "out.png")
    return {"step_seconds": recorder.step_times}


def run_ltx2(model_path: str, initial_dir: Path, out_dir: Path, steps: int) -> dict:
    from diffusers import LTX2Pipeline
    from diffusers.utils import export_to_video

    cfg = CONFIGS["ltx_2"]
    encoder = LTX2Pipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        audio_vae=None,
        vocoder=None,
        connectors=None,
        torch_dtype=torch.float32,
    )
    prompt_embeds, prompt_attention_mask, _, _ = encoder.encode_prompt(
        prompt=PROMPT, do_classifier_free_guidance=False, device="cpu"
    )
    del encoder
    gc.collect()

    pipe = LTX2Pipeline.from_pretrained(
        model_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float32
    )
    pipe.prepare_latents = _fixed(torch.load(initial_dir / "initial_video.pt"))
    pipe.prepare_audio_latents = _fixed(torch.load(initial_dir / "initial_audio.pt"))
    recorder = _StepRecorder(steps)
    output = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["num_frames"],
        frame_rate=cfg["frame_rate"],
        num_inference_steps=steps,
        guidance_scale=1.0,
        callback_on_step_end=recorder,
    )
    torch.save(recorder.final, out_dir / "final_video.pt")
    export_to_video(output.frames[0], str(out_dir / "out.mp4"), fps=int(cfg["frame_rate"]))
    return {"step_seconds": recorder.step_times}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--initial-latents-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None, help="default: the measurement config")
    args = parser.parse_args()

    steps = args.steps or CONFIGS[args.model]["steps"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    start = time.perf_counter()
    run = run_qwen if args.model == "qwen_image" else run_ltx2
    stats = run(args.model_path, args.initial_latents_dir, args.out_dir, steps)
    stats.update(model=args.model, steps=steps, wall_seconds=time.perf_counter() - start)
    (args.out_dir / "reference.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
