from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from difflet.cli.orchestrators.base import ModelOrchestrator

_HF_MODEL_ID = "Lightricks/LTX-2"
_MODEL_TYPE = "ltx_2"
_CLI_NAME = "ltx-2"


class LTX2Orchestrator(ModelOrchestrator):

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
            num_frames=self.args.num_frames,
            compile_cache_dir=self.args.cache_dir,
            force_compile=self.args.force,
            revision=self.args.revision,
        )

    def generate(self) -> None:
        import torch
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.compile_cache import CacheSpec, cache_path, has_valid_manifest
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
        shape = entry.resolve_shape(height=self.args.height, width=self.args.width,
                                    num_frames=self.args.num_frames)
        spec = CacheSpec(
            model_id=_HF_MODEL_ID, model_path=model_path,
            model_name=entry.name, parallel=parallel, dtype=self._dtype(),
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

        pipe = DiffletPipeline.from_pretrained(
            _HF_MODEL_ID,
            model_type=_MODEL_TYPE,
            parallel=parallel,
            dtype=self._dtype(),
            height=self.args.height,
            width=self.args.width,
            num_frames=self.args.num_frames,
            compile_cache_dir=self.args.cache_dir,
            revision=self.args.revision,
            skip_compile=True,
            # Load the host pipeline (text encoder + connectors + VAE/vocoder) so
            # prompt encoding and latent decode run on CPU around the Neuron DiT.
            # These are runtime-only and excluded from the compile cache key.
            application_kwargs={
                "enable_host_pipeline": True,
                "enable_decode_components": True,
            },
        )
        # Shape (height/width/num_frames) is baked into the compiled transformer and
        # the pipeline config; the pipeline derives latents/coords from it, so we do
        # NOT forward those kwargs here (pipeline.__call__ does not accept them).
        output = pipe(
            prompt=self.args.prompt,
            num_inference_steps=self.args.steps or 40,
            guidance_scale=self.args.guidance_scale or 3.5,
            generator=torch.Generator().manual_seed(self.args.seed),
            output_type="pt",
        )
        frames = output.frames if hasattr(output, "frames") else output[0]
        out = Path(self.args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frames.cpu(), out.with_suffix(".pt"))
        print(f"[difflet] video tensor saved to {out.with_suffix('.pt')}")

    def _parallel(self):
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        tp = self.args.tp_degree or entry.default_parallel.tp_degree
        return DiffletParallelConfig(tp_degree=tp, cp_degree=self.args.cp_degree or 1)

    def _dtype(self):
        import torch
        return torch.bfloat16
