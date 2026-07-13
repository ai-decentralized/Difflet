from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from difflet.cli.orchestrators.base import ModelOrchestrator

_HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
_MODEL_TYPE = "flux"
_CLI_NAME = "flux"


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
            compile_cache_dir=self.args.cache_dir,
            force_compile=self.args.force,
            revision=self.args.revision,
        )

    def generate(self) -> None:
        import torch

        from difflet.cli.dp import stage_loop

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
            compile_cache_dir=self.args.cache_dir,
            revision=self.args.revision,
            skip_compile=True,
            **self._teacache_kwargs(),
        )

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

    def _teacache_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.args.teacache_speedup is not None:
            kwargs["teacache_speedup"] = self.args.teacache_speedup
            kwargs["teacache_calibration_path"] = self.args.teacache_calibration
        app_kwargs: dict[str, Any] = {}
        if self.args.teacache_cadence is not None:
            app_kwargs["teacache_cadence"] = self.args.teacache_cadence
        if self.args.teacache_online_delta is not None:
            app_kwargs["teacache_online_delta_alpha"] = self.args.teacache_online_delta
        if app_kwargs:
            kwargs["application_kwargs"] = app_kwargs
        return kwargs
