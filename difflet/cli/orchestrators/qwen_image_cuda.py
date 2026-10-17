"""Qwen-Image on the CUDA backend — one process, one GPU (CUDA plan F3, D5).

The Trainium orchestrator (``qwen_image.py``) runs three Neuron subprocess stages. On CUDA at
``tp=1`` the text encoder, DiT and VAE all live in one process, so ``difflet run`` is
``DiffletPipeline.from_pretrained(backend="cuda")`` followed by ``app.generate()``. Selected by
``main._get_orchestrator`` when ``DIFFLET_BACKEND=cuda`` (or a CUDA device is auto-detected).

Release 1 scope: tp=1, cp=1, dp=1, no CFG-parallel, no TeaCache, one request shape. Anything
wider is refused here, before any weight is loaded, with the plan follow-up that owns it.

``CudaOneProcessOrchestrator`` is that ``download → compile → generate`` flow for any model
whose CUDA application exposes ``generate()``; ``CudaQwenImageOrchestrator`` and
``flux_cuda.FluxCudaOrchestrator`` (F5) set the model-specific parts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from difflet.cli.orchestrators.base import ModelOrchestrator

HF_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_STEPS = 50  # diffusers' QwenImagePipeline default; the Trainium CLI's 4 is a cost choice


def cuda_pipeline_kwargs(
    *,
    tp_degree: int | None,
    height: int | None,
    width: int | None,
    cache_dir: str | None,
    revision: str | None,
    force_compile: bool,
    load: bool,
) -> dict:
    """The ``DiffletPipeline.from_pretrained`` kwargs for Qwen-Image on CUDA.

    Shared with ``difflet serve`` (``difflet/serving/orchestrators/qwen_image_cuda.py``) so the
    resident server loads the model exactly as ``difflet run`` does.
    """
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    return dict(
        backend="cuda",
        parallel=DiffletParallelConfig(tp_degree=tp_degree or 1),
        dtype="bf16",
        height=height or 1024,
        width=width or 1024,
        compile_cache_dir=cache_dir,
        local_files_only=True,
        revision=revision,
        force_compile=bool(force_compile),
        load=load,
    )


class CudaOneProcessOrchestrator(ModelOrchestrator):
    """``difflet download/compile/generate/run`` for a one-process CUDA model.

    Subclasses set ``HF_MODEL_ID`` (the default repo), ``DEFAULT_STEPS``, ``TAG`` and
    ``ACCEPTS_GUIDANCE_SCALE`` (forward ``--guidance-scale``, defaulting to
    ``DEFAULT_GUIDANCE_SCALE``); ``MODEL_IDS`` lists the repos this orchestrator serves
    (the requested one is used when listed); ``refuse_model_flags`` and
    ``application_kwargs`` add the model-specific refusals and application options.
    """

    HF_MODEL_ID: str
    MODEL_IDS: tuple[str, ...] = ()
    DEFAULT_STEPS: int
    ACCEPTS_GUIDANCE_SCALE: bool = False
    DEFAULT_GUIDANCE_SCALE: float | None = None
    TAG: str

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._last_pipe = None

    # ------------------------------------------------------------- policy

    @property
    def model_id(self) -> str:
        requested = getattr(self.args, "model_id", None)
        return requested if requested in self.MODEL_IDS else self.HF_MODEL_ID

    def _refuse(self, flag: str, follow_up: str) -> None:
        raise SystemExit(
            f"{self.TAG} {flag} is not supported on the CUDA backend in Release 1 "
            f"(plan follow-up {follow_up})"
        )

    def refuse_unsupported(self) -> None:
        """Exit with the owning follow-up for every flag outside Release 1 scope."""
        args = self.args
        if (getattr(args, "tp_degree", None) or 1) > 1:
            self._refuse("--tp-degree > 1 (tensor parallel across PCIe)", "F6")
        if getattr(args, "sp_enabled", False):
            self._refuse("--sp (sequence parallel needs tp > 1)", "F6")
        if (getattr(args, "cp_degree", None) or 1) > 1:
            self._refuse("--cp-degree > 1 (context parallel, second card)", "F2")
        if getattr(args, "cfg_parallel", False):
            self._refuse("--cfg-parallel (second card)", "F2")
        self.refuse_model_flags()

    def refuse_model_flags(self) -> None:
        """Model-specific refusals; called after the shared ones."""

    def application_kwargs(self) -> dict:
        return {}

    def pipeline_kwargs(self, *, load: bool) -> dict:
        args = self.args
        kwargs = cuda_pipeline_kwargs(
            tp_degree=args.tp_degree,
            height=args.height,
            width=args.width,
            cache_dir=args.cache_dir,
            revision=getattr(args, "revision", None),
            force_compile=bool(getattr(args, "force", False)),
            load=load,
        )
        application_kwargs = self.application_kwargs()
        if application_kwargs:
            kwargs["application_kwargs"] = application_kwargs
        return kwargs

    def generate_kwargs(self) -> dict:
        kwargs = dict(
            num_inference_steps=self.args.steps or self.DEFAULT_STEPS,
            seed=self.args.seed,
            output_type="pil",
        )
        if self.ACCEPTS_GUIDANCE_SCALE:
            scale = getattr(self.args, "guidance_scale", None)
            kwargs["guidance_scale"] = self.DEFAULT_GUIDANCE_SCALE if scale is None else scale
        return kwargs

    # ------------------------------------------------------------ commands

    def download(self) -> None:
        # run() is download → compile → generate: refuse here too, before the snapshot pull.
        self.refuse_unsupported()
        from difflet.pipeline.path_resolver import resolve_model_path

        resolve_model_path(self.model_id, revision=self.args.revision, local_files_only=False)
        print(f"[difflet] weights ready for {self.model_id}")

    def compile(self) -> None:
        self.refuse_unsupported()
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        pipe = DiffletPipeline.from_pretrained(self.model_id, **self.pipeline_kwargs(load=False))
        self._last_pipe = pipe
        print(f"[difflet] CUDA artifact (eager manifest) at {pipe.compiled_path}")

    def generate(self) -> None:
        self.refuse_unsupported()
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        t0 = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(self.model_id, **self.pipeline_kwargs(load=True))
        self._last_pipe = pipe
        print(f"[difflet] loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        t1 = time.perf_counter()
        image = pipe.app.generate(self.args.prompt, **self.generate_kwargs())[0]
        out = Path(self.args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        print(
            f"[difflet] wrote {out} ({image.size[0]}x{image.size[1]}) "
            f"in {time.perf_counter() - t1:.1f} s"
        )


class CudaQwenImageOrchestrator(CudaOneProcessOrchestrator):
    HF_MODEL_ID = HF_MODEL_ID
    MODEL_IDS = (HF_MODEL_ID,)
    DEFAULT_STEPS = DEFAULT_STEPS
    # Qwen-Image's transformer has guidance_embeds=False: refused below, never forwarded.
    ACCEPTS_GUIDANCE_SCALE = False
    TAG = "[qwen_image/cuda]"

    def refuse_model_flags(self) -> None:
        args = self.args
        if getattr(args, "shapes", None):
            self._refuse("--shapes (bucketed request shapes)", "F3")
        # Probe-free fixed cadence (--teacache-cadence) runs on CUDA since F4; the calibrated
        # and online-delta modes need the probe path (F4b).
        for name in ("teacache_speedup", "teacache_online_delta"):
            if getattr(args, name, None) is not None:
                self._refuse(f"--{name.replace('_', '-')} (TeaCache probe path)", "F4b")
        if getattr(args, "guidance_scale", None) is not None:
            # The application refuses the value as a silent no-op, and the CLI must not hide
            # that by dropping the flag.
            raise SystemExit(
                f"{self.TAG} --guidance-scale has no effect on Qwen-Image (guidance_embeds=False; "
                "Release 1 generation is single-pass without classifier-free guidance)"
            )

    def application_kwargs(self) -> dict:
        application_kwargs: dict = {}
        if getattr(self.args, "torch_compile", False):
            # Measured option (CUDA plan D4 / F4 task 3): compiled DiT blocks, eager otherwise.
            application_kwargs["torch_compile"] = True
        if getattr(self.args, "teacache_cadence", None) is not None:
            application_kwargs["teacache_cadence"] = int(self.args.teacache_cadence)
        return application_kwargs


def refuse_unsupported(args: argparse.Namespace) -> None:
    """Qwen-Image's Release 1 refusals (kept as a module function for existing callers)."""
    CudaQwenImageOrchestrator(args).refuse_unsupported()


def pipeline_kwargs(args: argparse.Namespace, *, load: bool) -> dict:
    """Qwen-Image's ``DiffletPipeline.from_pretrained`` kwargs (module function kept)."""
    return CudaQwenImageOrchestrator(args).pipeline_kwargs(load=load)


__all__ = [
    "DEFAULT_STEPS",
    "HF_MODEL_ID",
    "CudaOneProcessOrchestrator",
    "CudaQwenImageOrchestrator",
    "cuda_pipeline_kwargs",
    "pipeline_kwargs",
    "refuse_unsupported",
]
