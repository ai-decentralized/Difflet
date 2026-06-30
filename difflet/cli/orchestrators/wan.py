"""Wan 2.1 / 2.2 T2V orchestrator — 2-stage subprocess pipeline.

Stage 1 (transformer): text encoder + DiT backbone, tp×cp cores.
Stage 2 (vae):         VAE decoder, 1 core.
Inter-stage tensor: {work_dir}/latents.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import ModelOrchestrator

# Model id comes from the CLI (--model-id); both Wan 2.2 A14B (MoE, dual
# transformer) and Wan 2.1 14B (single transformer) route here. The single-vs-
# dual difference is handled by the app/pipeline (transformer_2 + boundary_ratio
# are loaded only when present), so the orchestrator is version-agnostic.
_MODEL_TYPE = "wan"
_CLI_NAME = "wan"
_VIRTUAL_CORE_SIZE = None  # Wan does not require NEURON_RT_VIRTUAL_CORE_SIZE


def _save_video(tensor: "torch.Tensor", output_path: str) -> bool:
    """Write (1, C, T, H, W) tensor to MP4. Returns True on success."""
    try:
        import torch
        from diffusers.utils import export_to_video
    except ImportError:
        return False
    frames = tensor.detach().to(torch.float32).clamp(-1, 1)
    frames = ((frames + 1.0) / 2.0).clamp(0, 1)
    frames = (frames[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype("uint8")
    try:
        export_to_video(list(frames), output_path, fps=16)
    except Exception as exc:
        print(f"[wan] mp4 export failed ({exc}); saving as .pt instead", flush=True)
        return False
    print(f"[wan] video saved to {output_path}", flush=True)
    return True


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
        runner.run_stage(self.args.model_id, "vae",
                         num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
                         cli_args=shared)

    def generate(self) -> None:
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

        from difflet.models.wan.application import NeuronWanApplication, _latent_num_frames
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id, local_files_only=True)
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

        latent_frames = _latent_num_frames(args.num_frames or 9)
        torch.manual_seed(args.seed)
        latents = torch.randn(
            1, 16, latent_frames,
            (args.height or 480) // 8, (args.width or 832) // 8,
            dtype=torch.bfloat16,
        ) * 0.1
        app.load(str(compiled_dir), start_rank_id=0,
                 local_ranks_size=parallel.world_size, skip_warmup=True)
        out = app(
            latents=latents,
            prompt=args.prompt,
            height=args.height or 480,
            width=args.width or 832,
            num_frames=args.num_frames or 9,
            num_inference_steps=args.steps or 2,
            guidance_scale=args.guidance_scale or 1.0,
            output_type="latent",
        )
        work_dir = Path(args.work_dir)
        latents_out = out.latents if hasattr(out, "latents") else out[0]
        torch.save(latents_out.cpu(), work_dir / "latents.pt")
        print(f"[wan] latents saved to {work_dir}/latents.pt")

    def _stage_vae(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.models.wan.application import NeuronWanApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id, local_files_only=True)
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

        latents = torch.load(Path(args.work_dir) / "latents.pt")
        app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
        out = app(
            latents=latents,
            height=args.height or 480,
            width=args.width or 832,
            num_frames=args.num_frames or 9,
            num_inference_steps=1,
            output_type="pt",
        )
        frames = out.frames if hasattr(out, "frames") else out[0]
        out_path = Path(args.output)
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
        tp = args.tp_degree or 4
        cp = args.cp_degree or 1
        cfg = "cfg" if getattr(args, "cfg_parallel", False) else ""
        sp = "sp" if getattr(args, "sp_enabled", False) else ""
        h = args.height or 480
        w = args.width or 832
        f = args.num_frames or 9
        if stage == "transformer":
            return base / f"wan_transformer_tp{tp}cp{cp}{cfg}{sp}_h{h}w{w}f{f}"
        if stage == "vae":
            return base / f"wan_vae_h{h}w{w}f{f}"
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
        return parts
