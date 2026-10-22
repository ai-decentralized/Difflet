"""HunyuanVideo 1.5 text-to-video orchestrator.

On the neuron (TorchNeuron) backend, three stages share a work directory:

- ``encode`` (host, one process): diffusers' own ``HunyuanVideo15Pipeline.encode_prompt`` (Qwen2.5-VL
  chat template, ``hidden_states[-3]``, crop; ByT5 for quoted text) for the prompt and the empty
  negative prompt, written to ``embeds.pt``;
- ``transformer`` (TP x CFG x CP cores; TP2 x CFG2 by default: the checkpoint's guider runs
  classifier-free guidance at 6.0 on every step, so each cfg group computes one branch): the
  denoising loop (50 steps by default), rank 0 writing ``latents.pt``;
- ``vae`` (one core): the tiled bf16 decode, diffusers' video post-processing, an MP4 at 24 fps.

The Trainium (AoT) path is not implemented: its application covers the DiT on cached inputs only.

Default HF model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
Registry defaults: height=480, width=848, num_frames=121
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import ModelOrchestrator
from difflet.cli.orchestrators.torchrun import (
    reject_unsupported_torchrun_options,
    root_compile_cache_at_cache_dir,
    torchrun_backend,
    torchrun_unsupported_options,
)

_HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
_MODEL_TYPE = "hunyuan_video_15"
_CLI_NAME = "hunyuan-video-1.5"
_RUNS = "tensor, context and CFG parallelism"
_FPS = 24
_DEFAULT_STEPS = 50


def _unsupported(args: argparse.Namespace) -> list[str]:
    found = list(torchrun_unsupported_options(args))  # TeaCache
    if getattr(args, "sp_enabled", False):
        found.append("--sp")
    if getattr(args, "requests_dir", None) or getattr(args, "requests", None) or (getattr(args, "dp", None) or 1) > 1:
        found.append("--requests/--dp (one video per launch)")
    return found


def _layout(args: argparse.Namespace) -> tuple[int, bool, int]:
    """(tp, cfg_parallel, cp): TP2 x CFG2 unless the user chose a layout."""
    tp, cfg = getattr(args, "tp_degree", None), bool(getattr(args, "cfg_parallel", False))
    cp = int(getattr(args, "cp_degree", None) or 1)
    if tp is None and not cfg and cp == 1:
        return 2, True, 1
    return int(tp or 1), cfg, cp


class HunyuanVideo15Orchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        self._run("compile")

    def generate(self) -> None:
        self._run("generate")

    # ------------------------------------------------------------ driver

    def _run(self, mode: str) -> None:
        if torchrun_backend() != "neuron":
            raise NotImplementedError(
                f"difflet {mode} --model-id {_HF_MODEL_ID} runs on the neuron (TorchNeuron) backend only; "
                "the Trainium (AoT) application covers the DiT on cached inputs, without a pipeline.")
        reject_unsupported_torchrun_options(_unsupported(self.args), runs=_RUNS)
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="hunyuan_video_15")
        # The key-bounded attention kernel at 121 frames (~51k joint tokens) takes ~10 min to trace
        # in every process unless torch-neuronx's NKI trace cache keeps it; its inputs are all call
        # arguments, so the stages opt in (a per-stack-version cache, see neuron.compile).
        os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1")
        tp, cfg, cp = _layout(self.args)
        work_dir = Path(getattr(self.args, "work_dir", None) or Path.home() / ".cache" / "difflet" / "work" / "hv15")
        work_dir.mkdir(parents=True, exist_ok=True)
        cli = self._stage_cli_args(mode, str(work_dir), tp, cfg, cp)
        try:
            if mode == "generate":
                runner.run_stage(_HF_MODEL_ID, "encode", num_cores=1, virtual_core_size=None, cli_args=cli)
            runner.run_stage(_HF_MODEL_ID, "transformer", num_cores=tp * (2 if cfg else 1) * cp,
                             virtual_core_size=None, cli_args=cli)
            runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None, cli_args=cli)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not getattr(self.args, "keep_work_dir", False):
            shutil.rmtree(work_dir, ignore_errors=True)

    def _stage_cli_args(self, mode: str, work_dir: str, tp: int, cfg: bool, cp: int) -> list[str]:
        a = self.args
        parts = ["--model-id", _HF_MODEL_ID, "--stage-mode", mode, "--work-dir", work_dir,
                 "--tp-degree", str(tp), "--cp-degree", str(cp),
                 "--height", str(a.height or 480), "--width", str(a.width or 848),
                 "--num-frames", str(getattr(a, "num_frames", None) or 121),
                 "--steps", str(getattr(a, "steps", None) or _DEFAULT_STEPS),
                 "--seed", str(42 if getattr(a, "seed", None) is None else a.seed)]
        if cfg:
            parts.append("--cfg-parallel")
        if cp > 1:
            parts += ["--cp-mode", "ulysses"]
        for flag, name in (("--prompt", "prompt"), ("--output", "output"), ("--guidance-scale", "guidance_scale"),
                           ("--cache-dir", "cache_dir"), ("--revision", "revision"), ("--exec-mode", "exec_mode")):
            value = getattr(a, name, None)
            if value is not None:
                parts += [flag, str(value)]
        return parts

    # ------------------------------------------------------------ stages

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if stage == "encode":
            self._stage_encode(args)
        elif stage == "transformer":
            self._stage_transformer(args)
        elif stage == "vae":
            self._stage_vae(args)
        else:
            raise ValueError(f"Unknown HunyuanVideo 1.5 stage: {stage!r}")

    def _root(self, args) -> str:
        from difflet.pipeline.path_resolver import resolve_model_path

        return resolve_model_path(_HF_MODEL_ID, revision=getattr(args, "revision", None), local_files_only=True)

    def _stage_encode(self, args: argparse.Namespace) -> None:
        """Host: diffusers' encode_prompt for the prompt and the (empty) negative prompt."""
        import torch
        from diffusers import HunyuanVideo15Pipeline

        started = time.perf_counter()
        pipe = HunyuanVideo15Pipeline.from_pretrained(self._root(args), transformer=None, torch_dtype=torch.bfloat16)
        loaded = time.perf_counter()
        with torch.no_grad():
            pos = pipe.encode_prompt(prompt=args.prompt, device="cpu", dtype=torch.bfloat16)
            neg = pipe.encode_prompt(prompt="", device="cpu", dtype=torch.bfloat16)
        names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
        embeds = dict(zip(names, pos)) | {f"negative_{n}": t for n, t in zip(names, neg)}
        torch.save(embeds, os.path.join(args.work_dir, "embeds.pt"))
        print(f"[hv15] encode: load {loaded - started:.1f} s, encode {time.perf_counter() - loaded:.1f} s "
              f"(prompt + negative)", flush=True)

    def _stage_transformer(self, args: argparse.Namespace) -> None:
        """TP x CFG x CP ranks: the denoising loop; rank 0 saves the latents."""
        import torch
        import torch.distributed as dist

        from difflet.backends.neuron.hunyuan_video15.pipeline import TorchNeuronHunyuanVideo15Pipeline
        from difflet.backends.registry import get_backend
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        cp = int(getattr(args, "cp_degree", None) or 1)
        parallel = DiffletParallelConfig(tp_degree=int(args.tp_degree or 1), cp_degree=cp,
                                         cfg_parallel_enabled=bool(getattr(args, "cfg_parallel", False)),
                                         cp_mode="ulysses" if cp > 1 else "gather_kv")
        get_backend("neuron").prepare_runtime(parallel)
        rank0 = not dist.is_initialized() or dist.get_rank() == 0
        started = time.perf_counter()
        pipe = TorchNeuronHunyuanVideo15Pipeline(
            model_path=self._root(args), parallel=parallel, height=int(args.height), width=int(args.width),
            num_frames=int(args.num_frames), exec_mode=getattr(args, "exec_mode", None),
            guidance_scale=getattr(args, "guidance_scale", None))
        pipe.load(skip_warmup=args.stage_mode != "compile")
        if rank0:
            print(f"[hv15] transformer load {time.perf_counter() - started:.1f} s "
                  f"{ {k: round(v, 1) for k, v in pipe.dit.phase_seconds.items()} }", flush=True)
        if args.stage_mode == "compile":
            return
        embeds = torch.load(os.path.join(args.work_dir, "embeds.pt"), weights_only=True)
        marks = [time.perf_counter()]
        latents = pipe(**embeds, num_inference_steps=int(args.steps),
                       generator=torch.Generator().manual_seed(int(args.seed)),
                       callback=lambda i, t, x: marks.append(time.perf_counter()))
        if rank0:
            steps = [b - a for a, b in zip(marks, marks[1:])]
            steps_sorted = sorted(steps)
            print(f"[hv15] denoise {marks[-1] - marks[0]:.1f} s, {len(steps)} steps, median "
                  f"{steps_sorted[len(steps) // 2] * 1e3:.0f} ms (first {steps[0] * 1e3:.0f} ms)", flush=True)
            torch.save(latents, os.path.join(args.work_dir, "latents.pt"))

    def _stage_vae(self, args: argparse.Namespace) -> None:
        """One core: tiled bf16 decode, diffusers' post-processing, MP4."""
        import inspect

        import diffusers
        import torch
        from diffusers.utils import export_to_video

        from difflet.backends.neuron.hunyuan_video15.vae import NeuronHunyuanVideo15VaeDecoderApplication
        from difflet.backends.registry import get_backend
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=1))
        started = time.perf_counter()
        app = NeuronHunyuanVideo15VaeDecoderApplication(
            model_path=os.path.join(self._root(args), "vae"), height=int(args.height), width=int(args.width),
            num_frames=int(args.num_frames))
        app.load(skip_warmup=args.stage_mode != "compile")
        print(f"[hv15] vae load {time.perf_counter() - started:.1f} s "
              f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} }", flush=True)
        if args.stage_mode == "compile":
            return
        latents = torch.load(os.path.join(args.work_dir, "latents.pt"), weights_only=True)
        started = time.perf_counter()
        video = app.decode(latents).float().cpu()
        decoded = time.perf_counter()
        processor_cls = inspect.getmodule(diffusers.HunyuanVideo15Pipeline).HunyuanVideo15ImageProcessor
        frames = processor_cls(vae_scale_factor=16).postprocess_video(video, output_type="np")[0]
        out = Path(args.output or "hunyuan_video_15.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(list(frames), str(out), fps=_FPS)
        print(f"[hv15] decode {decoded - started:.1f} s; [difflet] video saved to {out}", flush=True)
