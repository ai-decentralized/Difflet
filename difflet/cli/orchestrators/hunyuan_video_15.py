"""HunyuanVideo 1.5 text-to-video orchestrator.

On the neuron (TorchNeuron) backend, three stages share a work directory:

- ``encode`` (host, one process): diffusers' own ``HunyuanVideo15Pipeline.encode_prompt`` (Qwen2.5-VL
  chat template, ``hidden_states[-3]``, crop; ByT5 for quoted text) for the prompt and the empty
  negative prompt, written to ``embeds.pt``;
- ``transformer`` (TP x CFG x CP cores; TP2 x CFG2 by default: the checkpoint's guider runs
  classifier-free guidance at 6.0 on every step, so each cfg group computes one branch): the
  denoising loop (50 steps by default), rank 0 writing ``latents.pt``;
- ``vae`` (one core): the tiled bf16 decode, diffusers' video post-processing, an MP4 at 24 fps.

Batch requests and data parallelism (``--requests``, ``--dp``): the encode stage claims each of the
worker's requests (its prompt, negative prompt and seed, else the command line's) and writes that
request's embeddings; the transformer and VAE stages replay the claims, the VAE stage marking each
request done. A DP replica is ``tp x cp (x2 with CFG)`` cores: on 4 cores, ``--dp 2 --tp-degree 2``
runs two TP2 replicas with classifier-free guidance's two passes in sequence (the 8.3B DiT does not
fit one core beside 121 frames' activations, so DP4 is not an option).

On the Trainium (AoT) backend (``_run_aot``: any backend but neuron), the same layout runs three
stages: the same host ``encode``; ``aot_transformer``, one process whose NxD application drives
all ``tp x cp (x2 with CFG)`` cores (``aot15.build_aot_denoiser``: the compiled DiT, plus the
TeaCache probe compiled into the same artifact on the first adaptive run), writing each request's
latents; and ``aot_vae``, writing the MP4: by default on the host (diffusers' tiled VAE in fp32,
as Wan's Trainium path decodes; the compiled decoder does not build on neuronx-cc 2.26), or with
--device-vae the tiled bf16 decoder on one core (``vae15``), which `difflet compile` then
compiles. Each compiled artifact lives in a hashed dir under --cache-dir
(``_aot_stage_cache_inputs``). --cp-mode ring and a multi-shape --shapes are refused before any
stage launches (``_aot_reject_unsupported``).

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
_RUNS = "tensor, sequence, context and CFG parallelism, and TeaCache"
_FPS = 24
_DEFAULT_STEPS = 50
_NKI_TRACE_OPT_IN = "DIFFLET_NKI_TRACE_CACHE"


def _unsupported(args: argparse.Namespace) -> list[str]:
    return [f for f in torchrun_unsupported_options(args) if not f.startswith("--teacache")]  # all run


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
            return self._run_aot(mode)
        reject_unsupported_torchrun_options(_unsupported(self.args), runs=_RUNS)
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="hunyuan_video_15")
        if mode == "compile":  # once, as the Trainium path's pre-shards: the fp32 DiT stored in bf16
            from difflet.backends.neuron.core.weights_bf16 import ensure_bf16_copy
            from difflet.pipeline.path_resolver import resolve_model_path

            started = time.perf_counter()
            try:
                root = resolve_model_path(_HF_MODEL_ID, revision=getattr(self.args, "revision", None),
                                          local_files_only=True)
                copy = ensure_bf16_copy(os.path.join(root, "transformer"))
                print(f"[hv15] bf16 copy of transformer: {copy} ({time.perf_counter() - started:.1f} s)",
                      flush=True)
            except FileNotFoundError as exc:  # no local snapshot: the stages report it themselves
                print(f"[hv15] bf16 copy skipped: {exc}", flush=True)
        tp, cfg, cp = _layout(self.args)
        work_dir = Path(getattr(self.args, "work_dir", None) or Path.home() / ".cache" / "difflet" / "work" / "hv15")
        work_dir.mkdir(parents=True, exist_ok=True)
        cli = self._stage_cli_args(mode, str(work_dir), tp, cfg, cp)
        # The key-bounded attention kernel at 121 frames (~51k joint tokens) takes ~10 min to trace
        # in every process unless torch-neuronx's NKI trace cache keeps it; its inputs are all call
        # arguments, so the stages opt in (a per-stack-version cache, see neuron.compile). Set only
        # around the launches (the stages inherit the environment), then restored.
        previous = os.environ.get(_NKI_TRACE_OPT_IN)
        os.environ[_NKI_TRACE_OPT_IN] = previous if previous is not None else "1"
        try:
            if mode == "generate":
                runner.run_stage(_HF_MODEL_ID, "encode", num_cores=1, virtual_core_size=None, cli_args=cli)
            runner.run_stage(_HF_MODEL_ID, "transformer", num_cores=tp * (2 if cfg else 1) * cp,
                             virtual_core_size=None, cli_args=cli)
            runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None, cli_args=cli)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        finally:
            if previous is None:
                os.environ.pop(_NKI_TRACE_OPT_IN, None)
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
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        if cp > 1:
            parts += ["--cp-mode", str(getattr(a, "cp_mode", None) or "ulysses")]
        for flag, name in (("--prompt", "prompt"), ("--output", "output"), ("--guidance-scale", "guidance_scale"),
                           ("--teacache-cadence", "teacache_cadence"),
                           ("--teacache-online-delta", "teacache_online_delta"),
                           ("--teacache-speedup", "teacache_speedup"),
                           ("--teacache-calibration", "teacache_calibration"),
                           ("--cache-dir", "cache_dir"), ("--revision", "revision"), ("--exec-mode", "exec_mode"),
                           ("--requests-dir", "requests_dir"), ("--worker-index", "worker_index"),
                           ("--dp-schedule", "dp_schedule")):
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
        elif stage == "aot_transformer":
            self._stage_aot_transformer(args)
        elif stage == "aot_vae":
            self._stage_aot_vae(args)
        else:
            raise ValueError(f"Unknown HunyuanVideo 1.5 stage: {stage!r}")

    def _root(self, args) -> str:
        from difflet.pipeline.path_resolver import resolve_model_path

        return resolve_model_path(_HF_MODEL_ID, revision=getattr(args, "revision", None), local_files_only=True)

    def _stage_encode(self, args: argparse.Namespace) -> None:
        """Host: claims this worker's requests (one, outside batch mode) and runs diffusers'
        encode_prompt for each prompt and negative prompt (empty unless the request has one)."""
        import torch
        from diffusers import HunyuanVideo15Pipeline

        from difflet.cli.dp import stage_loop

        started = time.perf_counter()
        pipe = HunyuanVideo15Pipeline.from_pretrained(self._root(args), transformer=None, torch_dtype=torch.bfloat16)
        print(f"[hv15] encode: load {time.perf_counter() - started:.1f} s", flush=True)
        names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                started = time.perf_counter()
                with torch.no_grad():
                    pos = pipe.encode_prompt(prompt=req.prompt, device="cpu", dtype=torch.bfloat16)
                    neg = pipe.encode_prompt(prompt=req.negative_prompt or "", device="cpu", dtype=torch.bfloat16)
                embeds = dict(zip(names, pos)) | {f"negative_{n}": t for n, t in zip(names, neg)}
                torch.save(embeds, stage_loop.work_file(args, req, "embeds.pt"))
                print(f"[hv15] encode request {req.index}: {time.perf_counter() - started:.1f} s "
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
                                         sp_enabled=bool(getattr(args, "sp_enabled", False)),
                                         cp_mode=(getattr(args, "cp_mode", None) or "ulysses") if cp > 1 else "gather_kv")
        get_backend("neuron").prepare_runtime(parallel)
        rank0 = not dist.is_initialized() or dist.get_rank() == 0
        started = time.perf_counter()
        pipe = TorchNeuronHunyuanVideo15Pipeline(
            model_path=self._root(args), parallel=parallel, height=int(args.height), width=int(args.width),
            num_frames=int(args.num_frames), exec_mode=getattr(args, "exec_mode", None),
            guidance_scale=getattr(args, "guidance_scale", None),
            teacache_cadence=getattr(args, "teacache_cadence", None),
            teacache_online_delta_alpha=getattr(args, "teacache_online_delta", None),
            teacache_speedup=getattr(args, "teacache_speedup", None),
            teacache_calibration_path=getattr(args, "teacache_calibration", None))
        pipe.load(skip_warmup=args.stage_mode != "compile")
        if rank0:
            print(f"[hv15] transformer load {time.perf_counter() - started:.1f} s "
                  f"{ {k: round(v, 1) for k, v in pipe.dit.phase_seconds.items()} }", flush=True)
        if args.stage_mode == "compile":
            return
        from difflet.cli.dp import stage_loop
        from difflet.cli.orchestrators.torchrun import replica_request_scope

        default_scale = pipe.guidance_scale
        for req in stage_loop.claimed_requests(args):
            with replica_request_scope(args, req, final=False):
                scale = stage_loop.effective(req, args, "guidance_scale", default_scale)
                if (scale is None) != (default_scale is None) or (scale is not None and float(scale) <= 1.0):
                    raise ValueError(f"request {req.index}: guidance_scale={scale} would switch classifier-"
                                     "free guidance on or off, which the stage's layout fixes")
                pipe.guidance_scale = None if scale is None else float(scale)
                embeds = torch.load(stage_loop.work_file(args, req, "embeds.pt"), weights_only=True)
                marks = [time.perf_counter()]
                latents = pipe(**embeds, num_inference_steps=int(stage_loop.effective(req, args, "steps", 50)),
                               generator=torch.Generator().manual_seed(int(stage_loop.effective(req, args, "seed", 42))),
                               callback=lambda i, t, x: marks.append(time.perf_counter()))
                if rank0:
                    steps = [b - a for a, b in zip(marks, marks[1:])]
                    steps_sorted = sorted(steps)
                    print(f"[hv15] request {req.index}: denoise {marks[-1] - marks[0]:.1f} s, {len(steps)} steps, "
                          f"median {steps_sorted[len(steps) // 2] * 1e3:.0f} ms (first {steps[0] * 1e3:.0f} ms)",
                          flush=True)
                    torch.save(latents, stage_loop.work_file(args, req, "latents.pt"))

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
        from difflet.cli.dp import stage_loop

        processor_cls = inspect.getmodule(diffusers.HunyuanVideo15Pipeline).HunyuanVideo15ImageProcessor
        processor = processor_cls(vae_scale_factor=16)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"), weights_only=True)
                started = time.perf_counter()
                video = app.decode(latents).float().cpu()
                decoded = time.perf_counter()
                out = Path(req.output or args.output or "hunyuan_video_15.mp4")
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.suffix == ".pt":  # the decoded frames, free of video-encoding loss
                    torch.save(video, str(out))
                else:
                    frames = processor.postprocess_video(video, output_type="np")[0]
                    export_to_video(list(frames), str(out), fps=_FPS)
                print(f"[hv15] request {req.index}: decode {decoded - started:.1f} s; "
                      f"[difflet] video saved to {out}", flush=True)

    # ------------------------------------------------------------ Trainium (AoT) path

    def _run_aot(self, mode: str) -> None:
        """encode (host) -> aot_transformer (one NxD process on all the layout's cores) -> aot_vae
        (the host decode by default; the compiled decoder, compiled by `difflet compile`, only
        with --device-vae)."""
        _aot_reject_unsupported(self.args)
        tp, cfg, cp = _layout(self.args)
        device_vae = _aot_device_vae(self.args)
        work_dir = Path(getattr(self.args, "work_dir", None) or Path.home() / ".cache" / "difflet" / "work" / "hv15")
        work_dir.mkdir(parents=True, exist_ok=True)
        cli = self._stage_cli_args(mode, str(work_dir), tp, cfg, cp)
        try:
            if mode == "generate":
                runner.run_stage(_HF_MODEL_ID, "encode", num_cores=1, virtual_core_size=None, cli_args=cli)
            runner.run_stage(_HF_MODEL_ID, "aot_transformer", num_cores=tp * (2 if cfg else 1) * cp,
                             virtual_core_size=_AOT_VIRTUAL_CORE_SIZE, cli_args=cli)
            if device_vae:
                runner.run_stage(_HF_MODEL_ID, "aot_vae", num_cores=1, virtual_core_size=_AOT_VIRTUAL_CORE_SIZE,
                                 cli_args=cli + ["--device-vae"])
            elif mode == "generate":  # a host stage, launched as the encoder is
                runner.run_stage(_HF_MODEL_ID, "aot_vae", num_cores=1, virtual_core_size=None, cli_args=cli)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not getattr(self.args, "keep_work_dir", False):
            shutil.rmtree(work_dir, ignore_errors=True)

    def _aot_parallel(self, args: argparse.Namespace):
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        return DiffletParallelConfig(tp_degree=int(args.tp_degree or 1), cp_degree=int(args.cp_degree or 1),
                                     cfg_parallel_enabled=bool(getattr(args, "cfg_parallel", False)),
                                     sp_enabled=bool(getattr(args, "sp_enabled", False)),
                                     cp_mode=_aot_cp_mode(args))

    def _aot_stage_cache_inputs(self, args: argparse.Namespace, component: str) -> dict:
        """One AoT artifact's identity (hashed into its dir, recorded in its manifest). Not part of
        it: the TeaCache options (the probe is an additive component of the DiT artifact), the
        steps, seed and prompt."""
        from difflet.cli.orchestrators.base import stage_toolchain_versions

        shape = {"height": int(args.height or 480), "width": int(args.width or 848),
                 "num_frames": int(args.num_frames or 121)}
        if component == "transformer":
            # graph_version: bump backbone15._AOT_DIT_GRAPH_VERSION whenever the traced DiT
            # graph's inputs or structure change, so the change compiles into a fresh dir.
            from difflet.backends.trainium.hunyuan_video.backbone15 import _AOT_DIT_GRAPH_VERSION

            return {"component": "hunyuan_video_15_aot_dit", "graph_version": _AOT_DIT_GRAPH_VERSION,
                    "model": _HF_MODEL_ID,
                    "revision": getattr(args, "revision", None), "tp": int(args.tp_degree or 1),
                    "cp": int(args.cp_degree or 1), "cp_mode": _aot_cp_mode(args),
                    "sp": bool(getattr(args, "sp_enabled", False)), "cfg": bool(getattr(args, "cfg_parallel", False)),
                    "dtype": "bfloat16", **shape, "text_seq_len": 1000, "text_seq_len_2": 256,
                    "image_seq_len": 729, "toolchain": stage_toolchain_versions()}
        if component == "vae":
            return {"component": "hunyuan_video_15_aot_vae", "model": _HF_MODEL_ID,
                    "revision": getattr(args, "revision", None), "dtype": "bfloat16", "shape": shape,
                    "tile": {"sample_min_height": 256, "sample_min_width": 256, "overlap_factor": 0.25},
                    "toolchain": stage_toolchain_versions()}
        raise ValueError(f"unknown HunyuanVideo 1.5 AoT component {component!r}")

    def _aot_compiled_dir(self, args: argparse.Namespace, component: str) -> Path:
        from difflet.cli.orchestrators.base import hashed_stage_dir

        inputs = self._aot_stage_cache_inputs(args, component)
        base = Path(getattr(args, "cache_dir", None) or Path.home() / ".cache" / "difflet").expanduser()
        return hashed_stage_dir(base, str(inputs["component"]), inputs)

    def _aot_require_artifact(self, args: argparse.Namespace, component: str, compiled_dir: Path) -> None:
        from difflet.cli.orchestrators.base import has_valid_stage_manifest

        if not has_valid_stage_manifest(compiled_dir, self._aot_stage_cache_inputs(args, component)):
            raise SystemExit(f"[hv15] no valid compiled {component} artifact at {compiled_dir} (manifest missing "
                             "or configuration changed); run `difflet compile` with the same flags first.")

    def _stage_aot_transformer(self, args: argparse.Namespace) -> None:
        """One process, NxD driving every core of the layout: compile the DiT (with the probe for
        adaptive TeaCache), or load it once and denoise each claimed request into its latents."""
        import torch

        from difflet.cli.orchestrators.base import write_stage_manifest
        from difflet.models.hunyuan_video import aot15

        root, parallel = self._root(args), self._aot_parallel(args)
        compiled_dir = self._aot_compiled_dir(args, "transformer")
        shape = dict(height=int(args.height), width=int(args.width), num_frames=int(args.num_frames))
        adaptive = bool(getattr(args, "teacache_speedup", None) or getattr(args, "teacache_calibration", None))
        started = time.perf_counter()
        if args.stage_mode == "compile":
            app = aot15.build_aot_application(model_path=root, parallel=parallel, teacache_fused=adaptive, **shape)
            app.compile(str(compiled_dir))
            write_stage_manifest(compiled_dir, self._aot_stage_cache_inputs(args, "transformer"))
            print(f"[hv15] aot transformer compiled into {compiled_dir} ({time.perf_counter() - started:.1f} s)",
                  flush=True)
            return
        from difflet.cli.dp import stage_loop
        from difflet.models.hunyuan_video.pipeline15 import cfg_on, load_hv15_guidance

        self._aot_require_artifact(args, "transformer", compiled_dir)
        denoiser = aot15.build_aot_denoiser(
            model_path=root, compiled_model_path=str(compiled_dir), parallel=parallel, **shape,
            guidance_scale=getattr(args, "guidance_scale", None),
            teacache_cadence=getattr(args, "teacache_cadence", None),
            teacache_online_delta_alpha=getattr(args, "teacache_online_delta", None),
            teacache_speedup=getattr(args, "teacache_speedup", None),
            teacache_calibration_path=getattr(args, "teacache_calibration", None))
        app = denoiser.dit.app
        if adaptive and not app.has_compiled_artifacts(str(compiled_dir), select=["teacache_probe"]):
            # The probe is an additive component of the DiT artifact, built on first adaptive use.
            print(f"[hv15] compiling the TeaCache probe into {compiled_dir} (first adaptive run; "
                  "the DiT artifact is reused)", flush=True)
            app.compile(str(compiled_dir), select=["teacache_probe"])
        denoiser.load(skip_warmup=True)
        print(f"[hv15] aot transformer load {time.perf_counter() - started:.1f} s", flush=True)
        default_on = cfg_on(denoiser.guidance_scale)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                requested = stage_loop.effective(req, args, "guidance_scale", None)
                scale = load_hv15_guidance(root, requested)  # the guider's own scale, or None when off
                if cfg_on(scale) != default_on:
                    raise ValueError(f"request {req.index}: guidance_scale={requested} would switch classifier-"
                                     "free guidance on or off, which the stage's layout fixes")
                denoiser.guidance_scale = scale
                embeds = torch.load(stage_loop.work_file(args, req, "embeds.pt"), weights_only=True)
                marks = [time.perf_counter()]
                latents = denoiser(
                    **embeds, num_inference_steps=int(stage_loop.effective(req, args, "steps", 50)),
                    generator=torch.Generator().manual_seed(int(stage_loop.effective(req, args, "seed", 42))),
                    callback=lambda i, t, x: marks.append(time.perf_counter()))
                steps = sorted(b - a for a, b in zip(marks, marks[1:]))
                timing = (f", median {steps[len(steps) // 2] * 1e3:.0f} ms (first {(marks[1] - marks[0]) * 1e3:.0f} ms)"
                          if steps else "")
                print(f"[hv15] request {req.index}: denoise {marks[-1] - marks[0]:.1f} s, "
                      f"{len(marks) - 1} steps{timing}", flush=True)
                torch.save(latents, stage_loop.work_file(args, req, "latents.pt"))

    def _stage_aot_vae(self, args: argparse.Namespace) -> None:
        """The host decode (the default), or with --device-vae the compiled decoder on one core."""
        if getattr(args, "device_vae", False):
            self._stage_aot_device_vae(args)
        else:
            self._stage_aot_host_vae(args)

    def _stage_aot_host_vae(self, args: argparse.Namespace) -> None:
        """Host CPU: diffusers' AutoencoderKLHunyuanVideo15 in fp32, tiled, decoding
        ``latents / scaling_factor`` as diffusers' pipeline does; post-processing, MP4. No artifact
        (the compiled vae15 decoder does not build on neuronx-cc 2.26: NCC_EVRF007 and others)."""
        if args.stage_mode == "compile":
            return
        import torch
        from diffusers import AutoencoderKLHunyuanVideo15

        from difflet.cli.dp import stage_loop

        started = time.perf_counter()
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(os.path.join(self._root(args), "vae"),
                                                          torch_dtype=torch.float32).eval()
        vae.enable_tiling()
        print(f"[hv15] host vae load {time.perf_counter() - started:.1f} s", flush=True)
        scaling_factor = float(vae.config.scaling_factor)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"), weights_only=True)
                started = time.perf_counter()
                with torch.no_grad():
                    video = vae.decode(latents.to(torch.float32) / scaling_factor, return_dict=False)[0].float().cpu()
                out = Path(req.output or args.output or "hunyuan_video_15.mp4")
                _aot_write_output(video, out)
                print(f"[hv15] request {req.index}: host decode {time.perf_counter() - started:.1f} s; "
                      f"[difflet] video saved to {out}", flush=True)

    def _stage_aot_device_vae(self, args: argparse.Namespace) -> None:
        """One core: the tiled bf16 decoder (an artifact of its own), diffusers' post-processing, MP4."""
        import torch

        from difflet.backends.trainium.hunyuan_video.vae15 import NeuronHunyuanVideo15VAEDecoderApplication
        from difflet.cli.orchestrators.base import write_stage_manifest
        from difflet.models.hunyuan_video.application import create_hunyuan_video15_vae_decoder_config

        root = self._root(args)
        compiled_dir = self._aot_compiled_dir(args, "vae")
        config = create_hunyuan_video15_vae_decoder_config(
            model_path=root, world_size=1, tp_degree=1, dtype=torch.bfloat16, height=int(args.height),
            width=int(args.width), num_frames=int(args.num_frames))
        app = NeuronHunyuanVideo15VAEDecoderApplication(model_path=os.path.join(root, "vae"), config=config)
        started = time.perf_counter()
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            write_stage_manifest(compiled_dir, self._aot_stage_cache_inputs(args, "vae"))
            print(f"[hv15] aot vae compiled into {compiled_dir} ({time.perf_counter() - started:.1f} s)", flush=True)
            return
        from difflet.cli.dp import stage_loop

        self._aot_require_artifact(args, "vae", compiled_dir)
        app.load(str(compiled_dir), skip_warmup=True)
        scaling_factor = float(app.config.scaling_factor)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"), weights_only=True)
                started = time.perf_counter()
                # vae15 neither divides by the scaling factor nor returns a tensor by default.
                video = app.decode(latents.to(torch.bfloat16) / scaling_factor, return_dict=False)[0].float().cpu()
                out = Path(req.output or args.output or "hunyuan_video_15.mp4")
                _aot_write_output(video, out)
                print(f"[hv15] request {req.index}: decode {time.perf_counter() - started:.1f} s; "
                      f"[difflet] video saved to {out}", flush=True)


_AOT_VIRTUAL_CORE_SIZE = 2


def _aot_device_vae(args: argparse.Namespace) -> bool:
    """--device-vae opts into the compiled decoder; --host-vae (or neither) decodes on the host,
    as Wan's Trainium path does (``wan._decode_on_device``)."""
    return bool(getattr(args, "device_vae", False)) and not getattr(args, "host_vae", False)


def _aot_reject_unsupported(args: argparse.Namespace) -> None:
    """Before any stage launches: ring CP (the Trainium ring kernel has no text-mask path) and a
    --shapes naming more than one shape (the AoT application compiles one). Torch-free: the
    shapes are counted, not parsed."""
    if getattr(args, "cp_mode", None) == "ring":
        raise SystemExit("[hv15] --cp-mode ring is not supported on the Trainium (AoT) backend: its ring "
                         "attention kernel has no text-mask path. Use --cp-mode ulysses (or gather_kv).")
    shapes = getattr(args, "shapes", None)
    if shapes and len([token for token in str(shapes).split(",") if token.strip()]) > 1:
        raise SystemExit(f"[hv15] --shapes {shapes} names more than one shape; the Trainium (AoT) HunyuanVideo "
                         "1.5 application compiles one shape: pass --height/--width/--num-frames instead.")


def _aot_cp_mode(args: argparse.Namespace) -> str:
    return (getattr(args, "cp_mode", None) or "ulysses") if int(args.cp_degree or 1) > 1 else "gather_kv"


def _aot_write_output(video, out: Path) -> None:
    """``.pt``: the decoded frames as they are; otherwise diffusers' post-processing and an MP4
    (``_stage_vae``'s, repeated so the native stage stays as it is)."""
    import inspect

    import diffusers
    import torch
    from diffusers.utils import export_to_video

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".pt":
        torch.save(video, str(out))
        return
    processor = inspect.getmodule(diffusers.HunyuanVideo15Pipeline).HunyuanVideo15ImageProcessor(vae_scale_factor=16)
    frames = processor.postprocess_video(video, output_type="np")[0]
    export_to_video(list(frames), str(out), fps=_FPS)
