from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from difflet.cli.orchestrators.base import ModelOrchestrator, adaptive_teacache_calibration

_HF_MODEL_ID = "Lightricks/LTX-2"
_MODEL_TYPE = "ltx_2"
_CLI_NAME = "ltx-2"
# LTX-2's native frame rate (difflet/models/ltx_2/application.py frame_rate default).
_FPS = 24


def _save_video(frames: "torch.Tensor", output_path: str) -> bool:
    """Write (B, F, C, H, W) frames in [0, 1] to MP4. Returns True on success.

    LTX-2's pipeline returns frames already denormalized to [0, 1] by diffusers'
    VideoProcessor.postprocess_video (frames axis before channels), unlike the
    Wan/Hunyuan (B, C, T, H, W) [-1, 1] layout — do not reuse their converter.
    """
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        return False
    clip = frames.detach().float().clamp(0.0, 1.0)[0].permute(0, 2, 3, 1)  # (F, H, W, C)
    # export_to_video multiplies ndarray frames by 255 itself; pass float [0, 1]
    # (uint8 input wraps to 256-v: color inversion).
    clip = clip.cpu().numpy().astype("float32")
    try:
        export_to_video(list(clip), output_path, fps=_FPS)
    except Exception as exc:
        print(f"[ltx_2] mp4 export failed ({exc}); saving as .pt instead", flush=True)
        return False
    print(f"[ltx_2] video saved to {output_path}", flush=True)
    return True


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

        from difflet.cli.dp import stage_loop

        pipe = self._load_pipeline()
        args = self.args
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                # Shape (height/width/num_frames) is baked into the compiled
                # transformer and the pipeline config; the pipeline derives
                # latents/coords from it, so we do NOT forward those kwargs here
                # (pipeline.__call__ does not accept them).
                output = pipe(
                    prompt=req.prompt,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 40)),
                    guidance_scale=float(
                        stage_loop.effective(req, args, "guidance_scale", 3.5)
                    ),
                    generator=torch.Generator().manual_seed(req.seed),
                    output_type="pt",
                )
                frames = output.frames if hasattr(output, "frames") else output[0]
                out = Path(req.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not (out.suffix == ".mp4" and _save_video(frames, str(out))):
                    torch.save(frames.cpu(), out.with_suffix(".pt"))
                    print(f"[difflet] video tensor saved to {out.with_suffix('.pt')}")

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

        # Load the host pipeline (text encoder + connectors + VAE/vocoder) so
        # prompt encoding and latent decode run on CPU around the Neuron DiT.
        # These are runtime-only and excluded from the compile cache key, as
        # are the probe-free TeaCache modes (host-side skip logic, no NEFF).
        application_kwargs: dict[str, Any] = {
            "enable_host_pipeline": True,
            "enable_decode_components": True,
        }
        if getattr(self.args, "teacache_cadence", None) is not None:
            application_kwargs["teacache_cadence"] = self.args.teacache_cadence
        if getattr(self.args, "teacache_online_delta", None) is not None:
            application_kwargs["teacache_online_delta_alpha"] = self.args.teacache_online_delta
        # Calibrated-adaptive TeaCache: LTX-2 computes its block-0 signal on the
        # host CPU transformer (no probe NEFF), so only the calibration path
        # reaches the application. It is runtime-only for the cache key; passing
        # teacache_speedup as well would flip the key to a probe-artifact identity
        # that does not exist for LTX-2 and miss the warm tp4 artifact.
        calibration = adaptive_teacache_calibration(
            self.args, model="ltx_2",
            shape_label=f"{shape['height']}x{shape['width']}x{shape['num_frames']}",
        )
        if calibration is not None:
            application_kwargs["teacache_calibration_path"] = calibration

        return DiffletPipeline.from_pretrained(
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
            application_kwargs=application_kwargs,
        )

    def _parallel(self):
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        tp = self.args.tp_degree or entry.default_parallel.tp_degree
        return DiffletParallelConfig(
            tp_degree=tp,
            cp_degree=self.args.cp_degree or 1,
            cfg_parallel_enabled=getattr(self.args, "cfg_parallel", False),
        )

    def _dtype(self):
        import torch
        return torch.bfloat16
