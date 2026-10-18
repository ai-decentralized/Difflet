from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import argparse

from difflet.cli import runner
from difflet.cli.orchestrators.base import ModelOrchestrator, require_request_shape_in_set
from difflet.cli.orchestrators.torchrun import (
    reject_unsupported_torchrun_options,
    replica_request_scope,
    replica_requests,
    root_compile_cache_at_cache_dir,
    torchrun_backend,
    torchrun_unsupported_options,
)

_HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
_MODEL_TYPE = "flux"
_CLI_NAME = "flux"
_TORCHRUN_RUNS = "tensor parallelism and data-parallel replicas"


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """What the torchrun (neuron) generate stage rejects, as the command line names it:
    every torchrun backend's (TeaCache), and what Flux's neuron path does not run yet. Torch-free."""
    found = torchrun_unsupported_options(args)
    if (getattr(args, "cp_degree", None) or 1) > 1:
        found.append("--cp-degree")
    if getattr(args, "cfg_parallel", False):
        found.append("--cfg-parallel")
    if getattr(args, "sp_enabled", False):
        found.append("--sp")
    if getattr(args, "taef1", False):
        found.append("--taef1")
    return found


class FluxOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision,
                           local_files_only=False,
                           allow_patterns=entry.download_patterns)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        if torchrun_backend() is not None:
            self._run_torchrun_stage("compile")
            return
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=True)
        DiffletPipeline.precompile(
            _HF_MODEL_ID,
            model_type=_MODEL_TYPE,
            parallel=self._parallel(),
            dtype=self._dtype(),
            height=self.args.height,
            width=self.args.width,
            shapes=self._compile_shapes(),
            compile_cache_dir=self.args.cache_dir,
            force_compile=self.args.force,
            revision=self.args.revision,
            **self._model_kwargs(),
        )

    def generate(self) -> None:
        if torchrun_backend() is not None:
            self._run_torchrun_stage("generate")
            return
        import torch

        from difflet.cli.dp import stage_loop

        # Fail fast if the request shape is outside the compiled bucket set.
        require_request_shape_in_set(self.args, default_shape=(1024, 1024), model_tag="flux")
        pipe = self._load_pipeline()
        args = self.args
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                output = pipe(
                    prompt=req.prompt,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 28)),
                    height=args.height or 1024,
                    width=args.width or 1024,
                    guidance_scale=float(
                        stage_loop.effective(req, args, "guidance_scale", 3.5)
                    ),
                    generator=torch.Generator().manual_seed(req.seed),
                )
                image = output.images[0]
                out = Path(req.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(out))
                print(f"[difflet] image saved to {out}")

    # ------------------------------------------------ torchrun (neuron) path

    def _run_torchrun_stage(self, mode: str) -> None:
        """One torchrun launch of the ``generate`` stage, tp ranks: ``compile`` loads and warms up
        every component (filling the compile cache); ``generate`` also runs the request(s), and
        rank 0 writes each image."""
        reject_unsupported_torchrun_options(_torchrun_unsupported_options(self.args),
                                            runs=_TORCHRUN_RUNS)
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="flux")
        runner.run_stage(_HF_MODEL_ID, "generate", num_cores=int(self.args.tp_degree or 4),
                         virtual_core_size=None, cli_args=self._stage_cli_args(mode))

    def _stage_cli_args(self, mode: str) -> list[str]:
        a = self.args
        parts = ["--model-id", _HF_MODEL_ID, "--tp-degree", str(a.tp_degree or 4),
                 "--height", str(a.height or 1024), "--width", str(a.width or 1024),
                 "--stage-mode", mode]
        for flag, name in (("--revision", "revision"), ("--prompt", "prompt"), ("--output", "output"),
                           ("--steps", "steps"), ("--guidance-scale", "guidance_scale"),
                           ("--seed", "seed"), ("--cache-dir", "cache_dir"), ("--work-dir", "work_dir"),
                           ("--requests-dir", "requests_dir"), ("--worker-index", "worker_index"),
                           ("--dp-schedule", "dp_schedule"), ("--exec-mode", "exec_mode")):
            value = getattr(a, name, None)
            if value is not None:
                parts += [flag, str(value)]
        if getattr(a, "keep_work_dir", False):
            parts.append("--keep-work-dir")
        return parts

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if stage != "generate":
            raise ValueError(f"Unknown Flux stage: {stage!r}")
        self._stage_generate_neuron(args)

    def _stage_generate_neuron(self, args: argparse.Namespace) -> None:
        """The generate stage on every rank: DiffletPipeline prepares the runtime and builds the
        registry's application (TorchNeuronFluxApplication); then each request (one, or a DP
        worker's queue: rank 0 claims for the replica) runs on every rank and rank 0 saves it."""
        import torch
        import torch.distributed as dist

        from difflet.cli.dp import stage_loop
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        import statistics
        import time

        h, w = int(args.height or 1024), int(args.width or 1024)
        app_kwargs = {"exec_mode": args.exec_mode} if getattr(args, "exec_mode", None) else None
        parallel = DiffletParallelConfig(tp_degree=int(args.tp_degree or 4))
        # Start-up breakdown: process start to here, prepare_runtime, the heavy imports
        # from_pretrained would do lazily, then from_pretrained itself (it prepares again: cheap).
        startup: dict[str, float] = {}
        mark = time.perf_counter()
        import torch_neuronx  # noqa: F401  (registers the device; does not start the runtime)

        startup["import torch_neuronx"] = round(time.perf_counter() - mark, 2)
        mark = time.perf_counter()
        from difflet.backends.registry import get_backend

        get_backend("neuron").prepare_runtime(parallel)  # binds the core; rendezvous of all ranks
        startup["prepare_runtime"] = round(time.perf_counter() - mark, 2)
        mark = time.perf_counter()
        import diffusers  # noqa: F401
        import transformers  # noqa: F401

        import difflet.models.flux.neuron_application  # noqa: F401
        startup["imports"] = round(time.perf_counter() - mark, 2)
        started = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(
            _HF_MODEL_ID, model_type=_MODEL_TYPE,
            parallel=parallel,
            dtype=torch.bfloat16, height=h, width=w, revision=getattr(args, "revision", None),
            compile_cache_dir=getattr(args, "cache_dir", None), local_files_only=True,
            application_kwargs=app_kwargs)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        app = pipe.app
        if rank == 0:
            loaded = time.perf_counter() - started
            components = getattr(app, "phase_seconds", {}) or {}
            startup["from_pretrained outside the components"] = round(loaded - sum(components.values()), 2)
            print(f"[flux] start-up {startup}", flush=True)
            print(f"[flux] load {loaded:.1f} s by component {components}", flush=True)
            for name in ("dit", "t5", "vae"):
                phases = getattr(getattr(app, name, None), "phase_seconds", None)
                if phases:
                    print(f"[flux]   {name} phases { {k: round(v, 2) for k, v in phases.items()} }",
                          flush=True)
        if args.stage_mode == "compile":
            if rank == 0:
                print("[flux] neuron: every component loaded and warmed up", flush=True)
            return
        for req in replica_requests(args, device=pipe.app.dit.device):
            with replica_request_scope(args, req, final=True):
                marks: list[float] = []

                def on_step_end(_pipe, _i, _t, callback_kwargs):
                    marks.append(time.perf_counter())
                    return callback_kwargs

                called = time.perf_counter()
                output = pipe(
                    prompt=req.prompt, height=h, width=w,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 28)),
                    guidance_scale=float(stage_loop.effective(req, args, "guidance_scale", 3.5)),
                    generator=torch.Generator().manual_seed(req.seed),
                    callback_on_step_end=on_step_end,
                )
                done = time.perf_counter()
                if rank == 0 and len(marks) > 1:
                    deltas = sorted(b - a for a, b in zip(marks, marks[1:]))
                    print(f"[flux] request: {done - called:.2f} s; to the first step's end "
                          f"{marks[0] - called:.2f} s (prompt encode + step 1); step median "
                          f"{statistics.median(deltas) * 1e3:.1f} ms (n={len(deltas)}, max "
                          f"{deltas[-1] * 1e3:.1f}); after the last step {done - marks[-1]:.2f} s "
                          f"(unpack + decode)", flush=True)
                if rank == 0:
                    out = Path(req.output)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    output.images[0].save(str(out))
                    print(f"[difflet] image saved to {out}", flush=True)

    def _load_pipeline(self):
        from difflet.pipeline.compile_cache import CacheSpec, cache_path, has_valid_manifest
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model

        try:
            model_path = resolve_model_path(_HF_MODEL_ID, revision=self.args.revision,
                                            local_files_only=True)
        except OSError:
            print(
                f"Error: model weights not found.\n"
                f"Run: difflet download --model-id {_HF_MODEL_ID}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        parallel = self._parallel()
        # Overlap the ~6.7s one-time NeuronCore bring-up with the host-side load.
        from difflet.cli.prewarm import prewarm_neuron_runtime
        prewarm_neuron_runtime(parallel.world_size)
        shape = entry.resolve_shape(height=self.args.height, width=self.args.width)
        spec = CacheSpec(
            model_id=_HF_MODEL_ID, model_path=model_path,
            model_name=entry.name, parallel=parallel,
            dtype=self._dtype(),
            height=shape.get("height"), width=shape.get("width"),
            num_frames=shape.get("num_frames"), revision=self.args.revision,
            application_kwargs=self._application_kwargs() or None,
            shapes=self._compile_shapes(),
        )
        compiled = cache_path(self.args.cache_dir, spec)
        if not has_valid_manifest(compiled, spec):
            print(
                f"Error: no compiled artifacts found for {_HF_MODEL_ID} at {compiled}.\n"
                f"Run: difflet compile --model-id {_HF_MODEL_ID} --tp-degree {parallel.tp_degree}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        return DiffletPipeline.from_pretrained(
            _HF_MODEL_ID,
            model_type=_MODEL_TYPE,
            parallel=parallel,
            dtype=self._dtype(),
            height=self.args.height,
            width=self.args.width,
            shapes=self._compile_shapes(),
            compile_cache_dir=self.args.cache_dir,
            revision=self.args.revision,
            skip_compile=True,
            **self._model_kwargs(),
        )

    def _compile_shapes(self):
        """Canonical (h, w) bucket set from --shapes, or None for single-shape."""
        from difflet.backends.trainium.core.bucketing import canonicalize_shapes
        from difflet.cli.orchestrators.base import parse_shapes_arg

        shapes = parse_shapes_arg(getattr(self.args, "shapes", None))
        return canonicalize_shapes(shapes) if shapes else None

    # ------------------------------------------------------------------ helpers

    def _parallel(self):
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        tp = self.args.tp_degree or entry.default_parallel.tp_degree
        cp = self.args.cp_degree or 1
        # CFG-parallel is not supported for Flux via the CLI (rejected in
        # main._validate_cfg_parallel); do not thread the flag here.
        return DiffletParallelConfig(
            tp_degree=tp,
            cp_degree=cp,
            cp_mode=getattr(self.args, "cp_mode", "gather_kv"),
            sp_enabled=getattr(self.args, "sp_enabled", False),
        )

    def _dtype(self):
        import torch
        return torch.bfloat16

    def _application_kwargs(self) -> dict[str, Any]:
        """Model-opt kwargs shared by compile and load (hashed into the cache key)."""
        app_kwargs: dict[str, Any] = {}
        if getattr(self.args, "teacache_cadence", None) is not None:
            app_kwargs["teacache_cadence"] = self.args.teacache_cadence
        if getattr(self.args, "teacache_online_delta", None) is not None:
            app_kwargs["teacache_online_delta_alpha"] = self.args.teacache_online_delta
        if getattr(self.args, "taef1", False):
            app_kwargs["taef1"] = True
            app_kwargs["taef1_path"] = self.args.taef1_path
        return app_kwargs

    def _model_kwargs(self) -> dict[str, Any]:
        """Kwargs passed to DiffletPipeline.from_pretrained / precompile.

        TeaCache and TAEF1 must reach BOTH the compile step (the VAE NEFF and
        the probe NEFF are built at compile time) and generate, so the cache
        spec and the loaded application agree.
        """
        kwargs: dict[str, Any] = {}
        if getattr(self.args, "teacache_speedup", None) is not None:
            kwargs["teacache_speedup"] = self.args.teacache_speedup
            kwargs["teacache_calibration_path"] = getattr(self.args, "teacache_calibration", None)
        app_kwargs = self._application_kwargs()
        if app_kwargs:
            kwargs["application_kwargs"] = app_kwargs
        return kwargs
