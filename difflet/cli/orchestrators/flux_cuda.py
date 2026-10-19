"""Flux on the CUDA backend — one process, one GPU.

The Trainium orchestrator (``flux.py``) runs Neuron subprocess stages. On CUDA at ``tp=1``
CLIP, T5, the DiT and the VAE live in one process inside diffusers' ``FluxPipeline``, so
``difflet run`` is ``DiffletPipeline.from_pretrained(backend="cuda")`` followed by
``app.generate()`` — the shared ``CudaOneProcessOrchestrator``. Selected by
``main._get_orchestrator`` for FLUX.1-dev and FLUX.1-schnell when the backend is CUDA.

The CUDA backend runs tp=1, cp=1, dp=1, no CFG-parallel, one request shape, no
TeaCache, no TAEF1, no host VAE, no ``torch.compile`` — refused here, before any download or
load. ``--guidance-scale`` is FLUX.1-dev's distilled
guidance (one pass) and is forwarded, default 3.5.
"""

from __future__ import annotations

from difflet.cli.orchestrators.qwen_image_cuda import CudaOneProcessOrchestrator

HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
SCHNELL_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 28  # FluxPipeline's default
DEFAULT_GUIDANCE_SCALE = 3.5  # FluxPipeline's default


class FluxCudaOrchestrator(CudaOneProcessOrchestrator):
    HF_MODEL_ID = HF_MODEL_ID
    MODEL_IDS = (HF_MODEL_ID, SCHNELL_MODEL_ID)
    DEFAULT_STEPS = DEFAULT_STEPS
    ACCEPTS_GUIDANCE_SCALE = True
    DEFAULT_GUIDANCE_SCALE = DEFAULT_GUIDANCE_SCALE
    TAG = "[flux/cuda]"

    def refuse_model_flags(self) -> None:
        args = self.args
        if getattr(args, "shapes", None):
            self._refuse("--shapes (bucketed request shapes)")
        for name in ("teacache_cadence", "teacache_speedup", "teacache_online_delta"):
            if getattr(args, name, None) is not None:
                self._refuse(f"--{name.replace('_', '-')} (TeaCache for Flux)")
        # ``main._validate_teacache`` lets a lone calibration file through and nothing
        # forwards it for Flux, so it would be dropped silently: refuse it by name.
        if getattr(args, "teacache_calibration", None):
            self._refuse("--teacache-calibration (TeaCache for Flux)")
        if getattr(args, "taef1", False) or getattr(args, "taef1_path", None):
            self._refuse("--taef1 / --taef1-path (tiny VAE decoder)")
        if getattr(args, "host_vae", False):
            self._refuse("--host-vae (the VAE already runs in-process on the GPU)")

    def application_kwargs(self) -> dict:
        application_kwargs: dict = {}
        if getattr(self.args, "torch_compile", False):
            # Measured option: compiled DiT blocks.
            application_kwargs["torch_compile"] = True
        return application_kwargs


__all__ = ["DEFAULT_GUIDANCE_SCALE", "DEFAULT_STEPS", "HF_MODEL_ID", "FluxCudaOrchestrator"]
