"""``difflet download/compile/generate/run`` for every model on the generic GPU flow.

On a GPU platform the whole model (encoders, DiT, decoder) lives in one process, so a run is
``DiffletPipeline.from_pretrained(backend=...)`` followed by ``app.generate()``. The model's
``ModelSpec`` supplies the defaults, the refusals and the output kind; nothing here is
per-model code. Selected by ``difflet.cli.main._get_orchestrator`` on a GPU platform.

When the parallel config needs more than one process (a CFG model's ``--cfg-parallel``),
``generate()`` re-execs itself under ``torchrun`` (``runner.run_torchrun``) and each rank runs
``generate`` in-process; only rank 0 writes. As a data-parallel worker (``--requests-dir``,
spawned by ``difflet/cli/dp/router.py``) ``generate()`` loads once and generates every request
it claims.

Importing this module must not import torch: ``difflet serve``'s parent process imports
``gpu_pipeline_kwargs`` from it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from difflet.cli.orchestrators.base import ModelOrchestrator
from difflet.backends.gpu.policy import (
    cfg_parallel_does_not_apply,
    is_set,
    unsupported_parallel,
)
from difflet.models.spec import ModelSpec


def gpu_pipeline_kwargs(
    spec: ModelSpec,
    *,
    tp_degree: int | None,
    height: int | None,
    width: int | None,
    cache_dir: str | None,
    revision: str | None,
    force_compile: bool,
    load: bool,
    num_frames: int | None = None,
    cfg_parallel: bool = False,
    torch_compile: bool = False,
) -> dict:
    """The ``DiffletPipeline.from_pretrained`` kwargs for ``spec`` on the GPU flow, shared by
    ``difflet run`` and ``difflet serve`` so the resident server loads the model exactly as a
    run does."""
    from difflet.backends.gpu.policy import gpu_platform
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    kwargs = dict(
        backend=gpu_platform().name,
        parallel=DiffletParallelConfig(
            tp_degree=tp_degree or 1, cfg_parallel_enabled=bool(cfg_parallel)
        ),
        dtype="bf16",
        height=height or spec.default_height,
        width=width or spec.default_width,
        compile_cache_dir=cache_dir,
        local_files_only=True,
        revision=revision,
        force_compile=bool(force_compile),
        load=load,
    )
    if spec.output == "video":
        kwargs["num_frames"] = num_frames or spec.default_num_frames
    if torch_compile:
        kwargs["application_kwargs"] = {"torch_compile": True}
    return kwargs


class GpuOrchestrator(ModelOrchestrator):
    """The one-process ``download → compile → generate`` flow for any model with a spec."""

    def __init__(self, args: argparse.Namespace, spec: ModelSpec) -> None:
        super().__init__(args)
        from difflet.backends.gpu.policy import gpu_platform

        self.spec = spec
        self.platform = gpu_platform()
        self.tag = f"[{spec.model_type}/{self.platform.name}]"
        self._last_pipe = None

    # ------------------------------------------------------------- policy

    @property
    def model_id(self) -> str:
        requested = getattr(self.args, "model_id", None)
        return requested if requested in self.spec.cli_model_ids else self.spec.cli_model_ids[0]

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
        raise SystemExit(
            f"{self.tag} {flag} is not supported on the {self.platform.display_name} backend yet"
        )

    def refuse_unsupported(self) -> None:
        """Exit for every option the platform or the model does not support, before anything
        is downloaded or loaded."""
        from difflet.platforms.base import PARALLEL_MODE_FLAGS

        args = self.args
        parallel = SimpleNamespace(
            tp_degree=getattr(args, "tp_degree", None),
            sp_enabled=getattr(args, "sp_enabled", False),
            cp_degree=getattr(args, "cp_degree", None),
            cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
        )
        for mode in unsupported_parallel(self.spec, parallel, self.platform):
            if mode == "cfg":
                # main._validate_cfg_parallel rejects it first for the distilled models; this
                # is the backstop for a direct orchestrator caller.
                raise SystemExit(f"{self.tag} {cfg_parallel_does_not_apply(self.spec)}")
            self._refuse(PARALLEL_MODE_FLAGS[mode])
        for attribute, label in self.spec.refused_cli_flags:
            if is_set(getattr(args, attribute, None)):
                self._refuse(label)
        if self.spec.guidance == "none" and getattr(args, "guidance_scale", None) is not None:
            # The application refuses the value as a silent no-op, and the CLI must not hide
            # that by dropping the flag.
            raise SystemExit(
                f"{self.tag} --guidance-scale has no effect on {self.spec.display_name} "
                f"({self.spec.no_guidance_reason})"
            )

    def application_kwargs(self) -> dict:
        kwargs: dict = {}
        for flag in self.spec.forwarded_cli_flags:
            value = getattr(self.args, flag, None)
            if is_set(value):
                kwargs[flag] = value
        return kwargs

    def _num_frames(self) -> int | None:
        if self.spec.output != "video":
            return None
        return getattr(self.args, "num_frames", None) or self.spec.default_num_frames

    def pipeline_kwargs(self, *, load: bool) -> dict:
        args = self.args
        kwargs = gpu_pipeline_kwargs(
            self.spec,
            tp_degree=args.tp_degree,
            height=args.height,
            width=args.width,
            num_frames=self._num_frames(),
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

    def generate_kwargs(self, request=None) -> dict:
        """``app.generate`` kwargs; a batch ``request`` (``RequestSpec``) overrides the flags.

        A request's steps/guidance win over ``--steps``/``--guidance-scale``, which win over
        the model defaults (``stage_loop.effective``); its seed replaces ``--seed``. A request
        field this model cannot honour raises, so that request is failed, not altered.
        """
        from difflet.cli.dp import stage_loop

        args = self.args
        spec = self.spec
        steps = stage_loop.effective(request, args, "steps", None) or spec.default_steps
        kwargs = dict(
            num_inference_steps=int(steps),
            seed=args.seed if request is None else request.seed,
            output_type="pt" if spec.output == "video" else "pil",
        )
        if spec.guidance != "none":
            kwargs["guidance_scale"] = stage_loop.effective(
                request, args, "guidance_scale", spec.default_guidance_scale
            )
        elif request is not None and request.guidance_scale is not None:
            raise ValueError(f"{self.tag} {self.model_id} takes no guidance_scale (request)")
        negative = None if request is None else request.negative_prompt
        if negative is not None:
            if not spec.accepts_negative_prompt:
                raise ValueError(f"{self.tag} {self.model_id} takes no negative_prompt (request)")
            kwargs["negative_prompt"] = negative
        if spec.output == "video":
            kwargs["num_frames"] = self._num_frames()
        return kwargs

    def save_result(self, result, out: Path) -> str:
        """Write ``generate()``'s result to ``out``; return what to print after "wrote"."""
        if self.spec.output != "video":
            image = result[0]
            image.save(out)
            return f"{out} ({image.size[0]}x{image.size[1]})"
        # A (B, 3, F, H, W) tensor in [-1, 1]: MP4 through the Trainium orchestrator's writer,
        # else the tensor as ``.pt``.
        from difflet.cli.orchestrators.wan import _save_video

        _, _, frames, height, width = result.shape
        shape = f"({frames} frames, {height}x{width})"
        if out.suffix == ".mp4" and _save_video(result, str(out)):
            return f"{out} {shape}"
        pt = out.with_suffix(".pt")
        import torch

        torch.save(result, pt)
        return f"{pt} {shape}"

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
        print(f"[difflet] GPU artifact (eager manifest) at {pipe.compiled_path}")

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
        from difflet.cli.dp import stage_loop

        if not stage_loop.batch_mode(self.args):
            self._generate_one(pipe, self.args.prompt, self.args.output, None)
            return
        from difflet.cli.dp import claims

        # One process: a failed request is marked and the loop continues. In a torchrun world
        # a failure on one rank would leave the ranks' collectives out of step (one in a CFG
        # gather, the other in the next broadcast: a hang), so the world fails fast and the
        # router marks the claimed-but-unfinished requests failed.
        multi_rank = int(os.environ.get("WORLD_SIZE", "1")) > 1
        for request in self._batch_requests():
            if self.is_rank_zero:
                # Marks .done on success, .failed (and continues) on an exception.
                with stage_loop.request_scope(self.args, request, final=True):
                    self._generate_one(pipe, request.prompt, request.output, request)
                if multi_rank and claims.is_failed(self.args.requests_dir, request.index):
                    print(
                        f"[dp-worker] request {request.index} failed; stopping the "
                        "torchrun world",
                        flush=True,
                    )
                    raise SystemExit(1)
                continue
            try:  # a follower rank: same work, no markers (rank 0 owns the queue)
                self._generate_one(pipe, request.prompt, request.output, request)
            except Exception as exc:
                print(f"[dp-worker] request {request.index} failed on this rank: {exc}", flush=True)
                raise

    def _generate_one(self, pipe, prompt: str, output: str, request) -> None:
        t1 = time.perf_counter()
        result = pipe.app.generate(prompt, **self.generate_kwargs(request))
        if not self.is_rank_zero:
            return
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = self.save_result(result, out)
        print(f"[difflet] wrote {written} in {time.perf_counter() - t1:.1f} s", flush=True)

    def _batch_requests(self):
        """This worker's requests, each claimed exactly once.

        One process: ``stage_loop.claim_requests``. Inside a ``torchrun`` world (a cfg
        replica) only rank 0 claims; it broadcasts each claimed ``RequestSpec`` (then
        ``None`` to stop) so every rank runs the same request in the same order (SPMD).
        """
        from difflet.cli.dp import claims, stage_loop

        if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            yield from stage_loop.claim_requests(self.args)
            return
        import torch.distributed

        args = self.args
        while True:
            box = [None]
            if self.is_rank_zero:
                box[0] = claims.claim_next(
                    args.requests_dir, int(args.worker_index), args.dp_schedule
                )
            torch.distributed.broadcast_object_list(box, src=0)
            if box[0] is None:
                return
            yield box[0]


__all__ = ["GpuOrchestrator", "gpu_pipeline_kwargs"]
