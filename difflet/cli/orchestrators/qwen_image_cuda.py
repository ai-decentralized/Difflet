"""Qwen-Image on the CUDA backend — one process, one GPU.

The Trainium orchestrator (``qwen_image.py``) runs three Neuron subprocess stages. On CUDA at
``tp=1`` the text encoder, DiT and VAE all live in one process, so ``difflet run`` is
``DiffletPipeline.from_pretrained(backend="cuda")`` followed by ``app.generate()``. Selected by
``main._get_orchestrator`` when ``DIFFLET_BACKEND=cuda`` (or a CUDA device is auto-detected).

Release 1 scope: tp=1, cp=1, dp=1, no CFG-parallel, no TeaCache, one request shape. Anything
wider is refused here, before any weight is loaded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from difflet.cli.orchestrators.base import ModelOrchestrator

HF_MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_STEPS = 50  # diffusers' QwenImagePipeline default; the Trainium CLI's 4 is a cost choice
_TAG = "[qwen_image/cuda]"


def _refuse(flag: str) -> None:
    raise SystemExit(
        f"{_TAG} {flag} is not supported on the CUDA backend yet"
    )


def refuse_unsupported(args: argparse.Namespace) -> None:
    """Exit with the owning follow-up for every flag outside Release 1 scope."""
    if getattr(args, "shapes", None):
        _refuse("--shapes (bucketed request shapes)")
    if (getattr(args, "tp_degree", None) or 1) > 1:
        _refuse("--tp-degree > 1 (tensor parallel across PCIe)")
    if getattr(args, "sp_enabled", False):
        _refuse("--sp (sequence parallel needs tp > 1)")
    if (getattr(args, "cp_degree", None) or 1) > 1:
        _refuse("--cp-degree > 1 (context parallel, second card)")
    if getattr(args, "cfg_parallel", False):
        _refuse("--cfg-parallel (second card)")
    for name in ("teacache_speedup", "teacache_cadence", "teacache_online_delta"):
        if getattr(args, name, None) is not None:
            _refuse(f"--{name.replace('_', '-')} (TeaCache port)")
    if getattr(args, "guidance_scale", None) is not None:
        # Qwen-Image's transformer has guidance_embeds=False: the application refuses the
        # value as a silent no-op, and the CLI must not hide that by dropping the flag.
        raise SystemExit(
            f"{_TAG} --guidance-scale has no effect on Qwen-Image (guidance_embeds=False; "
            "Release 1 generation is single-pass without classifier-free guidance)"
        )


def pipeline_kwargs(args: argparse.Namespace, *, load: bool) -> dict:
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    return dict(
        backend="cuda",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree or 1),
        dtype="bf16",
        height=args.height or 1024,
        width=args.width or 1024,
        compile_cache_dir=args.cache_dir,
        local_files_only=True,
        revision=getattr(args, "revision", None),
        force_compile=bool(getattr(args, "force", False)),
        load=load,
    )


class CudaQwenImageOrchestrator(ModelOrchestrator):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._last_pipe = None

    def download(self) -> None:
        # run() is download → compile → generate: refuse here too, before the 58 GB pull.
        refuse_unsupported(self.args)
        from difflet.pipeline.path_resolver import resolve_model_path

        resolve_model_path(HF_MODEL_ID, revision=self.args.revision, local_files_only=False)
        print(f"[difflet] weights ready for {HF_MODEL_ID}")

    def compile(self) -> None:
        refuse_unsupported(self.args)
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        pipe = DiffletPipeline.from_pretrained(HF_MODEL_ID, **pipeline_kwargs(self.args, load=False))
        self._last_pipe = pipe
        print(f"[difflet] CUDA artifact (eager manifest) at {pipe.compiled_path}")

    def generate(self) -> None:
        refuse_unsupported(self.args)
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        t0 = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(HF_MODEL_ID, **pipeline_kwargs(self.args, load=True))
        self._last_pipe = pipe
        print(f"[difflet] loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        t1 = time.perf_counter()
        image = pipe.app.generate(
            self.args.prompt,
            num_inference_steps=self.args.steps or DEFAULT_STEPS,
            seed=self.args.seed,
            output_type="pil",
        )[0]
        out = Path(self.args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        print(
            f"[difflet] wrote {out} ({image.size[0]}x{image.size[1]}) "
            f"in {time.perf_counter() - t1:.1f} s"
        )


__all__ = ["DEFAULT_STEPS", "HF_MODEL_ID", "CudaQwenImageOrchestrator", "pipeline_kwargs", "refuse_unsupported"]
