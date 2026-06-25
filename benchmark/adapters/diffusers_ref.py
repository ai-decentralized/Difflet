"""Reference backend adapter: stock Hugging Face diffusers on CPU/CUDA.

Exists to prove the harness is genuinely backend-generic — the same metrics and
runner drive a non-Trainium backend with no changes. It runs the upstream
diffusers pipeline eagerly (no AOT compile, so ``compile_seconds`` is 0) and is
the natural apples-to-apples reference for the Trainium numbers.

Eager/heavyweight: intended for a small shape / few steps as a correctness+speed
reference, not for full-resolution video on CPU.
"""
from __future__ import annotations

import time

from benchmark.harness import BackendAdapter, OutputInfo, timed


class DiffusersRefAdapter(BackendAdapter):
    name = "diffusers"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def device_info(self) -> str:
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                return f"CUDA / {torch.cuda.get_device_name(0)}"
            import platform
            return f"CPU / {platform.processor() or platform.machine()}"
        except Exception:
            return self.device

    def toolchain(self) -> dict[str, str]:
        try:
            import importlib.metadata as m
            return {p: m.version(p) for p in ("torch", "diffusers", "transformers")
                    if _safe(m, p)}
        except Exception:
            return {}

    def prepare(self, spec) -> None:
        from diffusers import DiffusionPipeline
        DiffusionPipeline.download(spec.model_id)

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        return 0.0, {"eager": 0.0}  # eager backend: no AOT compile

    def run_generate(self, spec) -> dict:
        import torch
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            spec.model_id, torch_dtype=torch.bfloat16 if spec.dtype == "bf16" else torch.float32
        ).to(self.device)
        kwargs = dict(prompt=spec.prompt, num_inference_steps=spec.steps)
        if spec.guidance_scale is not None:
            kwargs["guidance_scale"] = spec.guidance_scale
        if spec.height:
            kwargs["height"] = spec.height
        if spec.width:
            kwargs["width"] = spec.width
        if spec.num_frames:
            kwargs["num_frames"] = spec.num_frames

        sync = torch.cuda.synchronize if self.device == "cuda" else None
        t0 = time.perf_counter()
        out = pipe(**kwargs)
        if sync:
            sync()
        wall = time.perf_counter() - t0
        return {"wall_seconds": wall,
                "output": OutputInfo(note="diffusers reference pipeline output").__dict__}


def _safe(m, p) -> bool:
    try:
        m.version(p)
        return True
    except Exception:
        return False
