"""Wan 2.1 / 2.2 T2V orchestrator — 2-stage subprocess pipeline.

Stage 1 (transformer): text encoder + DiT backbone, tp×cp cores.
Stage 2 (vae):         VAE decoder, 1 core.
Inter-stage tensor: {work_dir}/latents.pt
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    bucketed_dir_token,
    cp_mode_token,
    parse_shapes_arg,
    require_request_shape_in_set,
)

# Model id comes from the CLI (--model-id); both Wan 2.2 A14B (MoE, dual
# transformer) and Wan 2.1 14B (single transformer) route here. The single-vs-
# dual difference is handled by the app/pipeline (transformer_2 + boundary_ratio
# are loaded only when present), so the orchestrator is version-agnostic.
_MODEL_TYPE = "wan"
_CLI_NAME = "wan"
_VIRTUAL_CORE_SIZE = None  # Wan does not require NEURON_RT_VIRTUAL_CORE_SIZE

# The historical model id keeps the bare "wan" compiled-dir prefix so existing
# compile caches stay valid (additive-only, same policy as
# DiffletParallelConfig.to_cache_dict).
_LEGACY_CACHE_PREFIX_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def _cache_prefix(model_id: str) -> str:
    """Per-model compiled-dir prefix: two Wan versions must never share
    artifacts (same architecture, different weights)."""
    if model_id == _LEGACY_CACHE_PREFIX_MODEL_ID:
        return "wan"
    return re.sub(r"[^a-z0-9]+", "_", model_id.split("/")[-1].lower()).strip("_")


def _save_video(tensor: "torch.Tensor", output_path: str) -> bool:
    """Write (1, C, T, H, W) tensor to MP4. Returns True on success."""
    try:
        import torch
        from diffusers.utils import export_to_video
    except ImportError:
        return False
    frames = tensor.detach().to(torch.float32).clamp(-1, 1)
    frames = ((frames + 1.0) / 2.0).clamp(0, 1)
    # export_to_video multiplies ndarray frames by 255 itself; pass float [0, 1]
    # (uint8 input wraps to 256-v: color inversion).
    frames = frames[0].permute(1, 2, 3, 0).cpu().numpy().astype("float32")
    try:
        export_to_video(list(frames), output_path, fps=16)
    except Exception as exc:
        print(f"[wan] mp4 export failed ({exc}); saving as .pt instead", flush=True)
        return False
    print(f"[wan] video saved to {output_path}", flush=True)
    return True


def _decode_latents_host(latents_path: str, model_id: str, output_path: str,
                         revision: str | None = None) -> None:
    """Decode DiT latents with the diffusers Wan VAE on host CPU.

    The compiled single-shot Neuron VAE exceeds the neuronx-cc instruction
    limit (NCC_EVRF007) beyond ~9 frames; diffusers decodes latent frames
    sequentially with a causal cache, so long clips work on host.
    """
    import torch
    from diffusers import AutoencoderKLWan

    from difflet.pipeline.path_resolver import resolve_model_path

    model_dir = resolve_model_path(model_id, revision=revision, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(
        str(Path(model_dir) / "vae"), torch_dtype=torch.float32
    ).eval()
    z = torch.load(latents_path, map_location="cpu").to(torch.float32)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    z = z / std + mean
    with torch.no_grad():
        frames = vae.decode(z, return_dict=False)[0]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4" and _save_video(frames.cpu(), str(out)):
        return
    torch.save(frames.cpu(), out.with_suffix(".pt"))
    print(f"[wan] video tensor saved to {out.with_suffix('.pt')}", flush=True)


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(480, 832, 9), model_tag="wan"
    )


class WanOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model
        entry = resolve_model(self.args.model_id, model_type=_MODEL_TYPE)
        resolve_model_path(self.args.model_id, local_files_only=False,
                           allow_patterns=entry.download_patterns)
        print(f"[difflet] weights ready for {self.args.model_id}")

    def compile(self) -> None:
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1) * (
            2 if getattr(self.args, "cfg_parallel", False) else 1
        )
        shared = self._shared_cli_args(stage_mode="compile")
        runner.run_stage(self.args.model_id, "transformer",
                         num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE,
                         cli_args=shared)
        if getattr(self.args, "host_vae", False):
            return
        runner.run_stage(self.args.model_id, "vae",
                         num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
                         cli_args=shared)

    def generate(self) -> None:
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(self.args.work_dir or
                        Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME)
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1) * (
            2 if getattr(self.args, "cfg_parallel", False) else 1
        )
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        try:
            runner.run_stage(self.args.model_id, "transformer",
                             num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE,
                             cli_args=shared)
            if getattr(self.args, "host_vae", False):
                _decode_latents_host(str(work_dir / "latents.pt"), self.args.model_id,
                                     self.args.output, revision=self.args.revision)
            else:
                runner.run_stage(self.args.model_id, "vae",
                                 num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
                                 cli_args=shared)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if stage == "transformer":
            self._stage_transformer(args)
        elif stage == "vae":
            self._stage_vae(args)
        else:
            raise ValueError(f"Unknown Wan stage: {stage!r}")

    # ---------------------------------------------------------------- stages

    def _stage_transformer(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.models.wan.application import NeuronWanApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id, local_files_only=True)
        compile_shapes = _require_request_shape_in_set(args)
        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
            sp_enabled=getattr(args, "sp_enabled", False),
        )
        compiled_dir = self._stage_compiled_dir("transformer", args)
        app = NeuronWanApplication(
            model_path=model_dir,
            parallel=parallel,
            dtype=torch.bfloat16,
            shape={
                "height": args.height or 480,
                "width": args.width or 832,
                "num_frames": args.num_frames or 9,
            },
            shapes=compile_shapes,
            text_seq_len=512,
            batch_size=1,
            enable_text_encoder=True,
            enable_transformer=True,
            enable_transformer_2=False,
            enable_vae_decoder=False,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop

        app.load(str(compiled_dir), start_rank_id=0,
                 local_ranks_size=parallel.world_size, skip_warmup=True)
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                out = app(
                    # Latent init is delegated to WanPipeline.prepare_latents, which
                    # draws unit-variance noise from this seeded generator.
                    generator=torch.Generator().manual_seed(req.seed),
                    prompt=req.prompt,
                    height=args.height or 480,
                    width=args.width or 832,
                    num_frames=args.num_frames or 9,
                    num_inference_steps=int(stage_loop.effective(req, args, "steps", 2)),
                    guidance_scale=float(
                        stage_loop.effective(req, args, "guidance_scale", 1.0)
                    ),
                    output_type="latent",
                )
                latents_out = out.latents if hasattr(out, "latents") else out[0]
                dest = stage_loop.work_file(args, req, "latents.pt")
                torch.save(latents_out.cpu(), dest)
                print(f"[wan] latents saved to {dest}")

    def _stage_vae(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.models.wan.application import NeuronWanApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id, local_files_only=True)
        compile_shapes = _require_request_shape_in_set(args)
        parallel = DiffletParallelConfig(tp_degree=1, cp_degree=1)
        compiled_dir = self._stage_compiled_dir("vae", args)
        app = NeuronWanApplication(
            model_path=model_dir,
            parallel=parallel,
            dtype=torch.bfloat16,
            shape={
                "height": args.height or 480,
                "width": args.width or 832,
                "num_frames": args.num_frames or 9,
            },
            shapes=compile_shapes,
            text_seq_len=512,
            batch_size=1,
            enable_text_encoder=False,
            enable_transformer=False,
            enable_transformer_2=False,
            enable_vae_decoder=True,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop

        app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(
                    stage_loop.work_file(args, req, "latents.pt")
                ).to(torch.bfloat16)
                out = app(
                    latents=latents,
                    height=args.height or 480,
                    width=args.width or 832,
                    num_frames=args.num_frames or 9,
                    num_inference_steps=1,
                    output_type="pt",
                )
                frames = out.frames if hasattr(out, "frames") else out[0]
                out_path = Path(req.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.suffix == ".mp4" and _save_video(frames.cpu(), str(out_path)):
                    pass
                else:
                    pt_path = out_path.with_suffix(".pt")
                    torch.save(frames.cpu(), pt_path)
                    print(f"[wan] video tensor saved to {pt_path}")

    # ------------------------------------------------------------ helpers

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        prefix = _cache_prefix(self.args.model_id)
        tp = args.tp_degree or 4
        cp = args.cp_degree or 1
        cfg = "cfg" if getattr(args, "cfg_parallel", False) else ""
        sp = "sp" if getattr(args, "sp_enabled", False) else ""
        cpm = cp_mode_token(args)
        h = args.height or 480
        w = args.width or 832
        f = args.num_frames or 9
        shapes = parse_shapes_arg(getattr(args, "shapes", None))
        if shapes is not None:
            from difflet.backends.trainium.core.bucketing import canonicalize_shapes

            canonical = canonicalize_shapes(shapes)
            h, w, f = canonical[0]
            token = f"_{bucketed_dir_token(canonical)}"
        else:
            token = ""
        if stage == "transformer":
            return base / f"{prefix}_transformer_tp{tp}cp{cp}{cpm}{cfg}{sp}_h{h}w{w}f{f}{token}"
        if stage == "vae":
            return base / f"{prefix}_vae_h{h}w{w}f{f}{token}"
        raise ValueError(f"unknown stage {stage!r}")

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id", self.args.model_id,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--cp-mode", str(getattr(a, "cp_mode", "gather_kv")),
            "--height", str(a.height or 480),
            "--width", str(a.width or 832),
            "--num-frames", str(a.num_frames or 9),
            "--steps", str(getattr(a, "steps", None) or 2),
            "--guidance-scale", str(getattr(a, "guidance_scale", None) or 1.0),
            "--seed", str(getattr(a, "seed", 42)),
            "--stage-mode", stage_mode,
        ]
        if getattr(a, "shapes", None):
            parts += ["--shapes", str(a.shapes)]
        if getattr(a, "cfg_parallel", False):
            parts.append("--cfg-parallel")
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        if getattr(a, "prompt", None):
            parts += ["--prompt", a.prompt]
        if getattr(a, "output", None):
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
        return parts
