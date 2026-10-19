from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

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
_TORCHRUN_RUNS = "tensor, sequence and context parallelism, data-parallel replicas and TeaCache"


def _decode_in_process(tp_degree: int) -> bool:
    """Whether the transformer stage decodes too (no separate vae stage). At TP4 the 1024x1024
    fp32 decode fits on rank 0's core beside its DiT (~6 GB) and T5 (~1.2 GB) shards, and a second
    stage would only add a launch (~15-20 s per worker). At TP2 (~12 GB and ~2.4 GB) it does not:
    in process it crawled through OOM recovery (~10 s per decode against ~0.7 s), so the vae stage
    decodes after the transformer stage has exited, as for Wan."""
    return int(tp_degree) >= 4


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """What the torchrun (neuron) generate stage rejects, as the command line names it:
    every torchrun backend's (TeaCache), and what Flux's neuron path does not run yet. Torch-free."""
    # TeaCache runs on Flux's neuron path (probe on the DiT, controller in the pipeline): drop
    # it from the shared torchrun refusal.
    found = [f for f in torchrun_unsupported_options(args) if not f.startswith("--teacache")]
    if getattr(args, "cfg_parallel", False):
        found.append("--cfg-parallel")  # FLUX.1-dev is guidance-distilled: no two-pass CFG
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
        """The ``transformer`` stage (torchrun, tp ranks) encodes and denoises. Where the decode fits
        beside the shards (``_decode_in_process``: TP4) it also decodes, rank 0 saving each image;
        otherwise rank 0 writes each request's packed latents to the work directory and the ``vae``
        stage (one process, one core) decodes them after the transformer stage has exited, as for
        Wan. ``compile`` loads and warms up whatever the generate path will run."""
        reject_unsupported_torchrun_options(_torchrun_unsupported_options(self.args),
                                            runs=_TORCHRUN_RUNS)
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="flux")
        work_dir = Path(self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / "flux")
        work_dir.mkdir(parents=True, exist_ok=True)
        cli_args = self._stage_cli_args(mode, work_dir=str(work_dir))
        try:
            runner.run_stage(_HF_MODEL_ID, "transformer", num_cores=self._replica_cores(),
                             virtual_core_size=None, cli_args=cli_args)
            if not _decode_in_process(int(self.args.tp_degree or 4)):
                runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None,
                                 cli_args=cli_args)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not getattr(self.args, "keep_work_dir", False):
            shutil.rmtree(work_dir, ignore_errors=True)

    def _replica_cores(self) -> int:
        return int(self.args.tp_degree or 4) * int(getattr(self.args, "cp_degree", None) or 1)

    def _stage_cli_args(self, mode: str, *, work_dir: str) -> list[str]:
        a = self.args
        parts = ["--model-id", _HF_MODEL_ID, "--tp-degree", str(a.tp_degree or 4),
                 "--height", str(a.height or 1024), "--width", str(a.width or 1024),
                 "--stage-mode", mode, "--work-dir", work_dir]
        for flag, name in (("--revision", "revision"), ("--prompt", "prompt"), ("--output", "output"),
                           ("--steps", "steps"), ("--guidance-scale", "guidance_scale"),
                           ("--seed", "seed"), ("--cache-dir", "cache_dir"),
                           ("--requests-dir", "requests_dir"), ("--worker-index", "worker_index"),
                           ("--dp-schedule", "dp_schedule"), ("--exec-mode", "exec_mode")):
            value = getattr(a, name, None)
            if value is not None:
                parts += [flag, str(value)]
        if (getattr(a, "cp_degree", None) or 1) > 1:
            parts += ["--cp-degree", str(a.cp_degree), "--cp-mode", str(a.cp_mode or "gather_kv")]
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        for flag, name in (("--teacache-cadence", "teacache_cadence"),
                           ("--teacache-online-delta", "teacache_online_delta"),
                           ("--teacache-speedup", "teacache_speedup"),
                           ("--teacache-calibration", "teacache_calibration")):
            value = getattr(a, name, None)
            if value is not None:
                parts += [flag, str(value)]
        if getattr(a, "keep_work_dir", False):
            parts.append("--keep-work-dir")
        return parts

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if stage == "transformer":
            self._stage_transformer_neuron(args)
        elif stage == "vae":
            self._stage_vae_neuron(args)
        else:
            raise ValueError(f"Unknown Flux stage: {stage!r}")

    def _stage_transformer_neuron(self, args: argparse.Namespace) -> None:
        """The transformer stage on every rank: DiffletPipeline prepares the runtime and builds the
        registry's application (TorchNeuronFluxApplication) without its VAE; each request (one, or
        a DP worker's queue: rank 0 claims for the replica) is encoded and denoised on every rank,
        and rank 0 saves its packed latents for the vae stage."""
        import torch
        import torch.distributed as dist

        from difflet.cli.dp import stage_loop
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        import statistics
        import time

        h, w = int(args.height or 1024), int(args.width or 1024)
        in_process = _decode_in_process(int(args.tp_degree or 4))
        app_kwargs = {"decode_vae": in_process}
        for option, name in (("teacache_cadence", "teacache_cadence"),
                             ("teacache_online_delta_alpha", "teacache_online_delta"),
                             ("teacache_speedup", "teacache_speedup"),
                             ("teacache_calibration_path", "teacache_calibration")):
            value = getattr(args, name, None)
            if value is not None:
                app_kwargs[option] = value
        if getattr(args, "exec_mode", None):
            app_kwargs["exec_mode"] = args.exec_mode
        cp = int(getattr(args, "cp_degree", None) or 1)
        parallel = DiffletParallelConfig(
            tp_degree=int(args.tp_degree or 4), cp_degree=cp,
            cp_mode=(getattr(args, "cp_mode", None) or "gather_kv") if cp > 1 else "gather_kv",
            sp_enabled=bool(getattr(args, "sp_enabled", False)))
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
            with replica_request_scope(args, req, final=in_process):
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
                    output_type="pil" if in_process else "latent",
                )
                done = time.perf_counter()
                if rank == 0 and len(marks) > 1:
                    deltas = sorted(b - a for a, b in zip(marks, marks[1:]))
                    print(f"[flux] request: {done - called:.2f} s; to the first step's end "
                          f"{marks[0] - called:.2f} s (prompt encode + step 1); step median "
                          f"{statistics.median(deltas) * 1e3:.1f} ms (n={len(deltas)}, max "
                          f"{deltas[-1] * 1e3:.1f}); after the last step {done - marks[-1]:.2f} s",
                          flush=True)
                if rank == 0 and in_process:
                    out = Path(req.output)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    output.images[0].save(str(out))
                    print(f"[difflet] image saved to {out}", flush=True)
                elif rank == 0:
                    path = stage_loop.work_file(args, req, "latents.pt")
                    torch.save(output.images.cpu(), path)
                    print(f"[flux] latents saved to {path}", flush=True)

    def _stage_vae_neuron(self, args: argparse.Namespace) -> None:
        """The vae stage, one process on one core, after the transformer stage has exited: each
        claimed request's packed latents are unpacked and un-normalised as FluxPipeline does
        (``latents / scaling_factor + shift_factor``), decoded in fp32 (eager), post-processed and
        saved. ``compile`` mode loads with the warm-up decode, which compiles and caches the per-op
        kernels at the request shape; ``generate`` skips it (as for Wan's VAE stage)."""
        import time

        import torch
        from diffusers import FluxPipeline
        from diffusers.image_processor import VaeImageProcessor

        from difflet.backends.neuron.flux.vae import NeuronFluxVaeDecoderApplication
        from difflet.backends.registry import get_backend
        from difflet.cli.dp import stage_loop
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        h, w = int(args.height or 1024), int(args.width or 1024)
        parallel = DiffletParallelConfig(tp_degree=1)
        get_backend("neuron").prepare_runtime(parallel)
        root = resolve_model_path(_HF_MODEL_ID, revision=getattr(args, "revision", None),
                                  local_files_only=True)
        started = time.perf_counter()
        app = NeuronFluxVaeDecoderApplication(model_path=os.path.join(root, "vae"), height=h, width=w,
                                              parallel=parallel, device="neuron")
        app.load(skip_warmup=args.stage_mode != "compile")
        print(f"[flux] vae load {time.perf_counter() - started:.1f} s "
              f"{ {k: round(v, 2) for k, v in app.phase_seconds.items()} }", flush=True)
        if args.stage_mode == "compile":
            return
        cfg = app.vae_config
        scaling, shift = cfg.get("scaling_factor", 0.3611), cfg.get("shift_factor", 0.1159)
        processor = VaeImageProcessor(vae_scale_factor=app.scale_factor * 2)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                started = time.perf_counter()
                packed = torch.load(stage_loop.work_file(args, req, "latents.pt"), weights_only=True)
                latents = FluxPipeline._unpack_latents(packed, h, w, app.scale_factor)
                latents = latents / scaling + shift
                image = processor.postprocess(app.decode(latents).cpu(), output_type="pil")[0]
                out = Path(req.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(out))
                print(f"[flux] decode {time.perf_counter() - started:.2f} s; [difflet] image saved "
                      f"to {out}", flush=True)

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
