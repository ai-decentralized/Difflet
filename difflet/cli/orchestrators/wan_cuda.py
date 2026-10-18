"""Wan 2.2 / 2.1 T2V on the CUDA backend — one process, one GPU (CUDA plan F1, D5).

The Trainium orchestrator (``wan.py``) runs a transformer stage and a VAE stage as Neuron
subprocesses. On CUDA at ``tp=1`` UMT5, both DiT experts and the VAE live in one process
(``difflet.models.wan.cuda_application.CudaWanApplication``), so ``difflet run`` is
``DiffletPipeline.from_pretrained(backend="cuda")`` followed by ``app.generate()`` — the
shared ``CudaOneProcessOrchestrator``. Selected by ``main._get_orchestrator`` for both Wan ids
when the backend is CUDA.

The application returns the ``(B, 3, F, H, W)`` video tensor in ``[-1, 1]``
(``OUTPUT_TYPE = "pt"``); it is written as an MP4 by the Trainium orchestrator's
``_save_video`` (16 fps), with the same ``.pt`` fallback when the export fails or the output
is not ``.mp4``. Default shape 480x832, 9 frames (``--height/--width/--num-frames``).

Release 1 scope (plan R3): tp=1, cp=1, dp=1, one request shape, no TeaCache, no TAEF1, no host
VAE, no ``torch.compile`` — refused here, before any download or load, with the follow-up that
owns each. ``--cfg-parallel`` (F2) is accepted: the conditional and unconditional branches run
on two GPUs, the shared orchestrator re-execing ``generate`` under ``torchrun``.
``--guidance-scale`` is the high-noise expert's CFG scale (default 4.0); the low-noise one
stays at the application's 3.0 (no CLI flag), and there is no ``--negative-prompt``
(diffusers' empty negative prompt).
"""

from __future__ import annotations

from pathlib import Path

from difflet.cli.orchestrators.qwen_image_cuda import CudaOneProcessOrchestrator
from difflet.cli.orchestrators.wan import _save_video

HF_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN21_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_STEPS = 40  # WanPipeline's Wan 2.2 default
DEFAULT_GUIDANCE_SCALE = 4.0  # Wan 2.2 high-noise stage
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_NUM_FRAMES = 9


class WanCudaOrchestrator(CudaOneProcessOrchestrator):
    HF_MODEL_ID = HF_MODEL_ID
    MODEL_IDS = (HF_MODEL_ID, WAN21_MODEL_ID)
    DEFAULT_STEPS = DEFAULT_STEPS
    ACCEPTS_GUIDANCE_SCALE = True
    DEFAULT_GUIDANCE_SCALE = DEFAULT_GUIDANCE_SCALE
    OUTPUT_TYPE = "pt"
    ACCEPTS_CFG_PARALLEL = True  # two-pass CFG: one branch per rank (F2)
    ACCEPTS_NEGATIVE_PROMPT = True  # a batch request's negative_prompt (F2 task 4)
    TAG = "[wan/cuda]"

    def refuse_model_flags(self) -> None:
        args = self.args
        if getattr(args, "shapes", None):
            self._refuse("--shapes (bucketed request shapes)", "F1b")
        for name in ("teacache_cadence", "teacache_speedup", "teacache_online_delta"):
            if getattr(args, name, None) is not None:
                self._refuse(f"--{name.replace('_', '-')} (TeaCache for Wan)", "F1b")
        # ``main._validate_teacache`` lets a lone calibration file through and nothing
        # forwards it for Wan, so it would be dropped silently: refuse it by name.
        if getattr(args, "teacache_calibration", None):
            self._refuse("--teacache-calibration (TeaCache for Wan)", "F1b")
        if getattr(args, "taef1", False) or getattr(args, "taef1_path", None):
            self._refuse("--taef1 / --taef1-path (tiny VAE decoder)", "F1b")
        if getattr(args, "host_vae", False):
            self._refuse("--host-vae (the VAE already runs in-process on the GPU)", "F1b")
        if getattr(args, "torch_compile", False):
            self._refuse("--torch-compile (compiled Wan DiT blocks)", "F1b")

    def _num_frames(self) -> int:
        return getattr(self.args, "num_frames", None) or DEFAULT_NUM_FRAMES

    def pipeline_kwargs(self, *, load: bool) -> dict:
        # The shared kwargs default to an image model's 1024x1024: Wan's shape is a video's.
        kwargs = super().pipeline_kwargs(load=load)
        kwargs["height"] = self.args.height or DEFAULT_HEIGHT
        kwargs["width"] = self.args.width or DEFAULT_WIDTH
        kwargs["num_frames"] = self._num_frames()
        return kwargs

    def generate_kwargs(self, request=None) -> dict:
        return {**super().generate_kwargs(request), "num_frames": self._num_frames()}

    def save_result(self, result, out: Path) -> str:
        """MP4 through the Trainium orchestrator's writer, else the tensor as ``.pt``."""
        _, _, frames, height, width = result.shape
        shape = f"({frames} frames, {height}x{width})"
        if out.suffix == ".mp4" and _save_video(result, str(out)):
            return f"{out} {shape}"
        pt = out.with_suffix(".pt")
        import torch

        torch.save(result, pt)
        return f"{pt} {shape}"


__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_STEPS",
    "HF_MODEL_ID",
    "WAN21_MODEL_ID",
    "WanCudaOrchestrator",
]
