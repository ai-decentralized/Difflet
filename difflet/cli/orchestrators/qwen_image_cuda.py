"""Qwen-Image on the CUDA backend — one process, one GPU.

The Trainium orchestrator (``qwen_image.py``) runs three Neuron subprocess stages. On CUDA at
``tp=1`` the text encoder, DiT and VAE all live in one process, so ``difflet run`` is
``DiffletPipeline.from_pretrained(backend="cuda")`` followed by ``app.generate()``. Selected by
``main._get_orchestrator`` when ``DIFFLET_BACKEND=cuda`` (or a CUDA device is auto-detected).

Release 1 scope: tp=1, cp=1, dp=1, no CFG-parallel, no TeaCache, one request shape. Anything
wider is refused here, before any weight is loaded.

``CudaOneProcessOrchestrator`` is that ``download → compile → generate`` flow for any model
whose CUDA application exposes ``generate()``; ``CudaQwenImageOrchestrator``,
``flux_cuda.FluxCudaOrchestrator`` and ``wan_cuda.WanCudaOrchestrator`` set the
model-specific parts. When the parallel config needs more than one process (Wan's
``--cfg-parallel``), ``generate()`` re-execs itself under ``torchrun``
(``runner.run_torchrun``) and each rank runs ``generate`` in-process; only rank 0 writes.
"""

from __future__ import annotations

import argparse
import os
import sys
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
    cfg_parallel: bool = False,
    torch_compile: bool = False,
) -> dict:
    """The ``DiffletPipeline.from_pretrained`` kwargs for Qwen-Image on CUDA.

    Shared with ``difflet serve`` (``difflet/serving/orchestrators/qwen_image_cuda.py``) so the
    resident server loads the model exactly as ``difflet run`` does.
    """
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    kwargs = dict(
        backend="cuda",
        parallel=DiffletParallelConfig(
            tp_degree=tp_degree or 1, cfg_parallel_enabled=bool(cfg_parallel)
        ),
        dtype="bf16",
        height=height or 1024,
        width=width or 1024,
        compile_cache_dir=cache_dir,
        local_files_only=True,
        revision=revision,
        force_compile=bool(force_compile),
        load=load,
    )
    if torch_compile:
        kwargs["application_kwargs"] = {"torch_compile": True}
    return kwargs


class CudaOneProcessOrchestrator(ModelOrchestrator):
    """``difflet download/compile/generate/run`` for a one-process CUDA model.

    Subclasses set ``HF_MODEL_ID`` (the default repo), ``DEFAULT_STEPS``, ``TAG`` and
    ``ACCEPTS_GUIDANCE_SCALE`` (forward ``--guidance-scale``, defaulting to
    ``DEFAULT_GUIDANCE_SCALE``); ``MODEL_IDS`` lists the repos this orchestrator serves
    (the requested one is used when listed); ``refuse_model_flags`` and
    ``application_kwargs`` add the model-specific refusals and application options;
    ``OUTPUT_TYPE`` and ``save_result`` are what ``generate()`` asks for and how it writes it
    (a PIL image by default; a video model returns a tensor and writes an MP4).
    ``ACCEPTS_CFG_PARALLEL`` lets ``--cfg-parallel`` through (a two-pass CFG model; the
    distilled ones refuse it): its two branches run on two ranks under ``torchrun``.
    """

    HF_MODEL_ID: str
    MODEL_IDS: tuple[str, ...] = ()
    DEFAULT_STEPS: int
    ACCEPTS_GUIDANCE_SCALE: bool = False
    DEFAULT_GUIDANCE_SCALE: float | None = None
    OUTPUT_TYPE: str = "pil"
    ACCEPTS_CFG_PARALLEL: bool = False
    TAG: str

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._last_pipe = None

    # ------------------------------------------------------------- policy

    @property
    def model_id(self) -> str:
        requested = getattr(self.args, "model_id", None)
        return requested if requested in self.MODEL_IDS else self.HF_MODEL_ID

    @property
    def world_size(self) -> int:
        """Processes the parallel config needs: ``cfg × cp × tp`` (dp is the router's)."""
        args = self.args
        cfg = 2 if getattr(args, "cfg_parallel", False) else 1
        return (
            cfg * (getattr(args, "cp_degree", None) or 1) * (getattr(args, "tp_degree", None) or 1)
        )

    @property
    def is_rank_zero(self) -> bool:
        """True outside ``torchrun`` and on its rank 0: the one process that writes output."""
        return int(os.environ.get("RANK", "0")) == 0

    def _refuse(self, flag: str) -> None:
        raise SystemExit(f"{self.TAG} {flag} is not supported on the CUDA backend yet")

    def refuse_unsupported(self) -> None:
        """Exit for every flag the CUDA backend does not support."""
        args = self.args
        if (getattr(args, "tp_degree", None) or 1) > 1:
            self._refuse("--tp-degree > 1 (tensor parallel across PCIe)")
        if getattr(args, "sp_enabled", False):
            self._refuse("--sp (sequence parallel needs tp > 1)")
        if (getattr(args, "cp_degree", None) or 1) > 1:
            self._refuse("--cp-degree > 1 (context parallel across cards)")
        if getattr(args, "cfg_parallel", False) and not self.ACCEPTS_CFG_PARALLEL:
            # main._validate_cfg_parallel rejects it first for the distilled models; this
            # is the backstop for a direct orchestrator caller.
            raise SystemExit(
                f"{self.TAG} --cfg-parallel does not apply: {self.model_id} is a distilled "
                "model (one guidance pass, no conditional/unconditional pair to split)"
            )
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
            cfg_parallel=bool(getattr(args, "cfg_parallel", False)),
        )
        application_kwargs = self.application_kwargs()
        if application_kwargs:
            kwargs["application_kwargs"] = application_kwargs
        return kwargs

    def generate_kwargs(self) -> dict:
        kwargs = dict(
            num_inference_steps=self.args.steps or self.DEFAULT_STEPS,
            seed=self.args.seed,
            output_type=self.OUTPUT_TYPE,
        )
        if self.ACCEPTS_GUIDANCE_SCALE:
            scale = getattr(self.args, "guidance_scale", None)
            kwargs["guidance_scale"] = self.DEFAULT_GUIDANCE_SCALE if scale is None else scale
        return kwargs

    def save_result(self, result, out: Path) -> str:
        """Write ``generate()``'s result to ``out``; return what to print after "wrote"."""
        image = result[0]
        image.save(out)
        return f"{out} ({image.size[0]}x{image.size[1]})"

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

    def generate(self, argv: list[str] | None = None) -> None:
        """Generate in-process, or re-exec under ``torchrun`` when ``world_size > 1``.

        Outside ``torchrun`` (``WORLD_SIZE`` unset) a multi-process config launches
        ``world_size`` ranks of ``difflet generate`` with the user's own arguments after the
        command word (``argv``, default ``sys.argv``), so a ``run`` becomes ``generate`` in
        the ranks: download and compile already ran here. A non-zero rank exit exits this
        process with that code. Inside a rank every rank runs the full generate (SPMD); only
        rank 0 writes the output.
        """
        self.refuse_unsupported()
        if self.world_size > 1 and "WORLD_SIZE" not in os.environ:
            from difflet.cli import runner

            user_argv = sys.argv if argv is None else argv
            code = runner.run_torchrun(
                self.world_size, ["-m", "difflet.cli.main", "generate", *user_argv[2:]]
            )
            if code:
                raise SystemExit(code)
            return
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        t0 = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(self.model_id, **self.pipeline_kwargs(load=True))
        self._last_pipe = pipe
        print(f"[difflet] loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        t1 = time.perf_counter()
        result = pipe.app.generate(self.args.prompt, **self.generate_kwargs())
        if not self.is_rank_zero:
            return
        out = Path(self.args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = self.save_result(result, out)
        print(f"[difflet] wrote {written} in {time.perf_counter() - t1:.1f} s")


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
            self._refuse("--shapes (bucketed request shapes)")
        # Probe-free fixed cadence (--teacache-cadence) runs on CUDA now; the calibrated
        # and online-delta modes need the probe path.
        for name in ("teacache_speedup", "teacache_online_delta"):
            if getattr(args, name, None) is not None:
                self._refuse(f"--{name.replace('_', '-')} (TeaCache probe path)")
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
            # Measured option: compiled DiT blocks, eager otherwise.
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
