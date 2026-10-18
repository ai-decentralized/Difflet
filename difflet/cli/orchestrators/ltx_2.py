"""LTX-2 orchestrator.

On the Trainium AoT backend (the default) ``compile`` builds the transformer NEFF through
``DiffletPipeline.precompile`` and ``generate`` runs the whole pipeline in this process: the
Gemma-3 text path, the connectors and the decode (video VAE, audio VAE, vocoder) on the host CPU
around the DiT on the NeuronCores.

With DIFFLET_BACKEND naming a non-AoT torchrun backend (neuron), ``generate`` is a generate stage
of tp ranks (doubled by --cfg-parallel) that build the registry's application
(TorchNeuronLtx2Application) through DiffletPipeline (``_stage_generate_neuron``): rank 0 encodes
the prompt with the same diffusers text path on its host and broadcasts the two text streams, the
ranks denoise (the DiT TP-sharded on every rank's core; CFG as two forwards per step, or one
batch-2 call over two TP groups with --cfg-parallel) and rank 0 writes the packed video and audio
latents. The driver then decodes them on the host after the launch has exited
(``_decode_latents_host``: the Trainium path's decode, the same diffusers modules in bf16), so the
decoder never shares a core or a process with the DiT. ``compile`` is the stage's load + warm-up
(it fills the NEFF cache) and its stage manifest. --cache-dir becomes DIFFLET_COMPILE_CACHE unless
that is exported. DP replicas (--dp, --requests) are separate torchrun launches whose rank 0
claims the requests, as for Wan and Qwen-Image. The options the ranks do not run (TeaCache,
several --shapes, --cp-mode ring, --sp, --cfg-parallel without --guidance-scale > 1 or with
--cp-degree > 1) and a TP degree whose DiT shard cannot fit one core are refused here, before
torchrun starts. --cp-degree > 1 (gather_kv or ulysses) shards the video tokens over the CP group
(tp x cp ranks).
--device-text-encoder runs Gemma-3's language model on the device, TP-sharded beside the DiT, with
the connectors on rank 0's host; the driver keeps the host encoder when it cannot fit
(``_place_text_encoder``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from difflet import envs
from difflet.cli import runner
from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    adaptive_teacache_calibration,
    canonical_shapes_list,
    hashed_stage_dir,
    stage_toolchain_versions,
    write_stage_manifest,
)
from difflet.cli.orchestrators.torchrun import (
    fallback_tracker,
    reject_unsupported_torchrun_options,
    replica_request_scope,
    replica_requests,
    root_compile_cache_at_cache_dir,
    torchrun_backend,
    torchrun_unsupported_options,
)

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


#: The Trainium CLI's defaults for a request that names none.
_DEFAULT_STEPS = 40
_DEFAULT_GUIDANCE = 3.5
_DEFAULT_SHAPE = (512, 768, 121)
#: What the neuron LTX-2 ranks run, for the driver's rejection message.
_TORCHRUN_RUNS = "tensor and CFG parallelism at one static shape, and DP replicas"
#: TorchNeuronLtx2Application keyword arguments for the torchrun generate stage: the text path
#: on rank 0's host with 12 threads, no decode in the launch (the driver decodes after it) and
#: exec_mode=None (the application reads DIFFLET_EXEC_MODE, which --exec-mode exports).
_TORCHRUN_APPLICATION_KWARGS = {
    "text_seq_len": 1024,
    "batch_size": 1,
    "enable_text_encoder": True,
    "enable_transformer": True,
    "enable_decode_components": False,
    "exec_mode": None,
    "encoder_threads": 12,
}


def _guidance(args: argparse.Namespace) -> float:
    value = getattr(args, "guidance_scale", None)
    return float(_DEFAULT_GUIDANCE if value is None else value)


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options the torchrun (neuron) ranks do not run, as the command line names them:
    every torchrun backend's (TeaCache), several --shapes (one static shape per process), and
    the parallel modes LTX-2 has no implementation of (--cp-mode ring, --sp). Torch-free."""
    found = torchrun_unsupported_options(args)
    shapes = getattr(args, "shapes", None)
    if shapes and len([token for token in str(shapes).split(",") if token.strip()]) > 1:
        found.append("--shapes with more than one shape")
    if (getattr(args, "cp_degree", None) or 1) > 1 and getattr(args, "cp_mode", None) == "ring":
        found.append("--cp-mode ring")
    if getattr(args, "sp_enabled", False):
        found.append("--sp")
    return found


def _torchrun_conflicts(args: argparse.Namespace, *, generating: bool) -> list[str]:
    """--host-vae with --device-vae, and --cfg-parallel without CFG: the CFG groups split the two
    branches of a guidance_scale > 1 step (``compile`` only warms the batch-2 graph up).
    Torch-free."""
    conflicts = []
    if getattr(args, "host_vae", False) and getattr(args, "device_vae", False):
        conflicts.append("--host-vae and --device-vae are mutually exclusive")
    if getattr(args, "cfg_parallel", False) and (getattr(args, "cp_degree", None) or 1) > 1:
        conflicts.append("--cfg-parallel and --cp-degree > 1 are mutually exclusive")
    if generating and getattr(args, "cfg_parallel", False) and _guidance(args) <= 1.0:
        conflicts.append("--cfg-parallel splits the two branches of classifier-free guidance; "
                         "LTX-2 runs them only with --guidance-scale > 1")
    return conflicts


#: Where the neuron backend decodes when neither --host-vae nor --device-vae is given: on the
#: host in the driver, the Trainium decode's placement and dtype.
_DEVICE_VAE_DEFAULT = False


def _decodes_on_device(args: argparse.Namespace) -> bool:
    """Whether the neuron backend decodes the video in a vae stage on one core
    (``_stage_vae_neuron``) instead of on the host in the driver (``_decode_latents_host``).
    --host-vae wins, then --device-vae, then ``_DEVICE_VAE_DEFAULT``. Torch-free."""
    if getattr(args, "host_vae", False):
        return False
    if getattr(args, "device_vae", False):
        return True
    return _DEVICE_VAE_DEFAULT


def _decode_latents_host(latents_path: str, model_dir: str, output_path: str, *, height: int,
                         width: int, num_frames: int) -> None:
    """Decode the generate stage's packed latents on the host CPU, as the Trainium generate does
    in its own process: diffusers' video VAE, audio VAE and vocoder in bf16
    (``LTX2Orchestrator._decode_latents``, the same un-normalisation and post-processing), then
    the video frames to ``output_path`` (``_save_video``; the audio is decoded and not written,
    as on Trainium)."""
    import torch

    from difflet.models.ltx_2.neuron_application import _host_ltx2_pipeline
    from difflet.models.ltx_2.pipeline import LTX2Orchestrator

    pipe = _host_ltx2_pipeline(model_dir, dtype=torch.bfloat16, text=False, decode=True)
    orchestrator = LTX2Orchestrator(
        model_path=model_dir, vae=pipe.vae, audio_vae=pipe.audio_vae, vocoder=pipe.vocoder,
        video_processor=pipe.video_processor, host_pipeline=pipe, dtype=torch.bfloat16,
        height=height, width=width, num_frames=num_frames,
    )
    latents = torch.load(latents_path, map_location="cpu")
    with torch.no_grad():
        frames, _audio = orchestrator._decode_latents(latents["video"], latents["audio"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (out.suffix == ".mp4" and _save_video(frames, str(out))):
        torch.save(frames.cpu(), out.with_suffix(".pt"))
        print(f"[difflet] video tensor saved to {out.with_suffix('.pt')}", flush=True)


class LTX2Orchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision,
                           local_files_only=False,
                           allow_patterns=entry.download_patterns)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def _adaptive_teacache_kwargs(self, shape: dict) -> dict[str, Any]:
        """Application kwargs a calibrated-adaptive request adds: the calibration
        path. For an adaptive calibration the application builds the fused
        block-0 probe NEFF and DiffletPipeline keys it (calibrated_probe in
        _cache_application_kwargs), so compile and load must both pass it."""
        calibration = adaptive_teacache_calibration(
            self.args, model="ltx_2",
            shape_label=f"{shape['height']}x{shape['width']}x{shape['num_frames']}",
        )
        return {} if calibration is None else {"teacache_calibration_path": calibration}

    def _resolved_shape(self) -> dict:
        from difflet.registry import resolve_model
        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        return entry.resolve_shape(height=self.args.height, width=self.args.width,
                                   num_frames=self.args.num_frames)

    def run(self) -> None:
        self._reject_unsupported_torchrun_options()  # before the download, too
        super().run()

    def compile(self) -> None:
        if torchrun_backend() is not None:
            self._compile_torchrun()
            return
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=True)
        # A calibrated-adaptive request compiles the probe artifact (its own
        # cache identity); every other request keeps the plain tp4 key.
        adaptive = self._adaptive_teacache_kwargs(self._resolved_shape())
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
            **({"application_kwargs": adaptive} if adaptive else {}),
        )

    def generate(self) -> None:
        if torchrun_backend() is not None:
            self._generate_torchrun()
            return
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
        adaptive = self._adaptive_teacache_kwargs(shape)
        from difflet.pipeline.difflet_pipeline import _cache_application_kwargs
        spec = CacheSpec(
            model_id=_HF_MODEL_ID, model_path=model_path,
            model_name=entry.name, parallel=parallel, dtype=self._dtype(),
            height=shape.get("height"), width=shape.get("width"),
            num_frames=shape.get("num_frames"), revision=self.args.revision,
            # the same artifact identity from_pretrained resolves below, so a
            # missing probe artifact fails here with the compile hint
            application_kwargs=_cache_application_kwargs(adaptive, model_name=entry.name),
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
        # Calibrated-adaptive TeaCache: the calibration path reaches the
        # application, which builds the fused block-0 probe NEFF for an adaptive
        # calibration; DiffletPipeline keys that probe (calibrated_probe), and
        # compile() passes the same kwarg so the probe artifact exists.
        application_kwargs.update(adaptive)

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

    # ------------------------------------------------------- torchrun (neuron) backend

    def _reject_unsupported_torchrun_options(self, *, generating: bool = True) -> None:
        """A torchrun backend (neuron) asked for an option its ranks do not run, or a pair they
        cannot run together: print one message naming it and exit 2 before anything is
        launched (or downloaded, for ``run``). The ranks' own checks stay (defence in depth).
        No-op on the AoT backends."""
        reject_unsupported_torchrun_options(_torchrun_unsupported_options(self.args),
                                            runs=_TORCHRUN_RUNS)
        conflicts = _torchrun_conflicts(self.args, generating=generating)
        if conflicts and torchrun_backend() is not None:
            print(f"Error: {conflicts[0]}; nothing was launched.", file=sys.stderr, flush=True)
            raise SystemExit(2)

    def _replica_cores(self) -> int:
        """Cores of one generate launch: tp x cp, doubled by --cfg-parallel."""
        return ((self.args.tp_degree or 4) * (self.args.cp_degree or 1)
                * (2 if getattr(self.args, "cfg_parallel", False) else 1))

    def _model_dir(self) -> str:
        from difflet.pipeline.path_resolver import resolve_model_path

        return resolve_model_path(_HF_MODEL_ID, revision=self.args.revision,
                                  local_files_only=True)

    def _check_dit_fits(self) -> None:
        """Refuse a TP degree whose DiT shard clearly cannot fit one core
        (``difflet/backends/neuron/ltx_2/hbm.py``) before any rank starts: print one message and
        exit 2. Weights that are not local yet leave the decision to the application's own
        check. Torch-free."""
        from difflet.backends.neuron.hbm import GIB, suggested_tp
        from difflet.backends.neuron.ltx_2.hbm import dit_fits

        try:
            model_dir = self._model_dir()
        except (OSError, RuntimeError):
            return
        tp = self.args.tp_degree or 4
        fits, estimate, limit = dit_fits(model_dir, tp_degree=tp)
        if fits:
            return
        print(f"Error: the LTX-2 DiT needs at least {estimate / GIB:.1f} GiB of bf16 weights per "
              f"core at tp={tp}, over the {limit / GIB:.1f} GiB budget of one NeuronCore (every "
              f"DP replica is its own TP group); use --tp-degree "
              f"{suggested_tp(estimate, tp, limit)} or more; nothing was launched.",
              file=sys.stderr, flush=True)
        raise SystemExit(2)

    def _place_text_encoder(self) -> None:
        """With --device-text-encoder, keep the Gemma-3 encode on rank 0's host when its language
        model cannot fit on the device beside the DiT at this TP degree, and print one line
        saying so (as the Qwen-Image driver does). At tp4 the encoder adds about 5.5 GiB of bf16
        weights per core beside 8.8 GiB of DiT; a tp2 group (--cfg-parallel or DP replicas on
        four cores) doubles both, over a 24 GiB core (``difflet/backends/neuron/ltx_2/hbm.py``).
        Weights that are not local yet leave the flag alone (the application's own check still
        refuses). The setting lands on ``self.args``, so ``_shared_cli_args`` hands it to the
        stage. Torch-free."""
        args = self.args
        if not getattr(args, "device_text_encoder", False):
            return
        from difflet.backends.neuron.hbm import GIB
        from difflet.backends.neuron.ltx_2.hbm import device_text_encoder_fits

        try:
            model_dir = self._model_dir()
        except (OSError, RuntimeError):
            return
        tp = args.tp_degree or 4
        fits, estimate, limit = device_text_encoder_fits(model_dir, tp_degree=tp)
        if fits:
            return
        args.device_text_encoder = False
        print(f"[ltx_2] Gemma-3 on the device needs at least {estimate / GIB:.1f} GiB per core "
              f"beside the DiT at tp={tp}, over the {limit / GIB:.1f} GiB budget: encoding on "
              "rank 0's host instead (without --device-text-encoder)", flush=True)

    def _prepare_torchrun(self, *, generating: bool) -> None:
        self._reject_unsupported_torchrun_options(generating=generating)
        self._check_dit_fits()
        self._place_text_encoder()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="ltx_2")

    def _compile_torchrun(self) -> None:
        """``compile`` on a torchrun backend: the generate stage's load + warm-up (it fills the
        NEFF cache) and its stage manifest."""
        self._prepare_torchrun(generating=False)
        shared = self._shared_cli_args("compile")
        runner.run_stage(_HF_MODEL_ID, "generate", num_cores=self._replica_cores(),
                         virtual_core_size=None, cli_args=shared)
        if _decodes_on_device(self.args):
            runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None,
                             cli_args=shared)

    def _generate_torchrun(self) -> None:
        """``generate`` on a torchrun backend: the generate stage (prompt encode on rank 0's
        host, denoise, rank 0 writes latents.pt), then the host decode of every request it
        finished, after torchrun has exited."""
        self._prepare_torchrun(generating=True)
        work_dir = Path(getattr(self.args, "work_dir", None)
                        or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME)
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args("generate", work_dir=str(work_dir))
        try:
            runner.run_stage(_HF_MODEL_ID, "generate", num_cores=self._replica_cores(),
                             virtual_core_size=None, cli_args=shared)
            if _decodes_on_device(self.args):
                runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None,
                                 cli_args=shared)
            else:
                self._decode_on_host(work_dir)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not getattr(self.args, "keep_work_dir", False):
            shutil.rmtree(work_dir, ignore_errors=True)

    def _shape(self, args: argparse.Namespace | None = None) -> tuple[int, int, int]:
        a = args or self.args
        return (a.height or _DEFAULT_SHAPE[0], a.width or _DEFAULT_SHAPE[1],
                a.num_frames or _DEFAULT_SHAPE[2])

    def _decode_on_host(self, work_dir: Path) -> None:
        """The host decode of every request the generate launch finished: the one request of a
        plain launch, or, in a DP worker, each request this worker's replica claimed and did
        not fail, marked done after its video is written."""
        from difflet.cli.dp import stage_loop

        args = argparse.Namespace(**{**vars(self.args), "work_dir": str(work_dir)})
        height, width, num_frames = self._shape()
        model_dir = self._model_dir()
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                _decode_latents_host(str(stage_loop.work_file(args, req, "latents.pt")),
                                     model_dir, req.output, height=height, width=width,
                                     num_frames=num_frames)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        backend = torchrun_backend()
        if stage not in ("generate", "vae") or backend is None:
            raise ValueError(
                f"LTX-2 has stages 'generate' and 'vae' on a torchrun backend only (got "
                f"{stage!r}, backend {backend!r}); the Trainium path runs in one process"
            )
        if stage == "generate":
            self._stage_generate_neuron(args, backend)
        else:
            self._stage_vae_neuron(args)

    def _stage_vae_neuron(self, args: argparse.Namespace) -> None:
        """The neuron backend's vae stage (--device-vae): one rank, a launch of its own after the
        generate stage has exited, so the decoder never shares a core with the DiT. The video
        decodes on the device (``NeuronLtx2VaeDecoderApplication``, bf16, eager); the audio VAE
        and the vocoder run on the host, and the orchestrator's own ``_decode_latents`` does the
        unpacking, un-normalisation and post-processing, as in the host decode. ``compile`` mode
        loads with the warm-up decode (it fills the eager kernel cache) and writes the
        manifest; ``generate`` skips that warm-up."""
        import torch

        from difflet.backends.neuron.core import distributed
        from difflet.backends.neuron.ltx_2.vae import NeuronLtx2VaeDecoderApplication
        from difflet.cli.dp import stage_loop
        from difflet.models.ltx_2.neuron_application import (
            _DeviceLtx2VaeDecoder,
            _host_ltx2_audio_decoder,
        )
        from difflet.models.ltx_2.pipeline import LTX2Orchestrator
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        model_dir = self._model_dir()
        height, width, num_frames = self._shape(args)
        latent_shape = (1, 128, (num_frames - 1) // 8 + 1, height // 32, width // 32)
        app = NeuronLtx2VaeDecoderApplication(
            model_path=str(Path(model_dir) / "vae"), latent_shape=latent_shape,
            parallel=DiffletParallelConfig(tp_degree=1), dtype=torch.bfloat16, device="neuron",
        )
        decoder = _DeviceLtx2VaeDecoder(app, device="neuron")
        decoder.load(skip_warmup=args.stage_mode != "compile")
        if args.stage_mode == "compile":
            if distributed.is_rank0():
                vae_args = argparse.Namespace(**{**vars(args), "stage": "vae"})
                write_stage_manifest(self._stage_compiled_dir(vae_args),
                                     self._stage_cache_inputs(vae_args))
            return
        audio_vae, vocoder, video_processor = _host_ltx2_audio_decoder(
            model_dir, dtype=torch.bfloat16)
        orchestrator = LTX2Orchestrator(
            model_path=model_dir, vae=decoder, audio_vae=audio_vae, vocoder=vocoder,
            video_processor=video_processor, dtype=torch.bfloat16, height=height, width=width,
            num_frames=num_frames,
        )
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"),
                                     map_location="cpu")
                with torch.no_grad():
                    frames, _audio = orchestrator._decode_latents(latents["video"],
                                                                  latents["audio"])
                out = Path(req.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not (out.suffix == ".mp4" and _save_video(frames, str(out))):
                    torch.save(frames.cpu(), out.with_suffix(".pt"))
                    print(f"[difflet] video tensor saved to {out.with_suffix('.pt')}",
                          flush=True)

    def _stage_generate_neuron(self, args: argparse.Namespace, backend: str) -> None:
        """The generate stage on a non-AoT torchrun backend (neuron); runs on every rank.

        ``DiffletPipeline.from_pretrained`` prepares the runtime (NEFF cache, core binding, the
        process group) before it builds the registry's application (TorchNeuronLtx2Application),
        and ``load(skip_warmup=False)`` compiles or loads the blocks at the static shape, so the
        step loop compiles nothing. ``compile`` mode is that warm-up plus the stage manifest.
        Every rank runs the same generate and rank 0 alone writes files. The process group is
        destroyed after a successful stage only: a failing rank exits with its exception and
        torchrun stops its peers.
        """
        import time

        import torch
        import torch.distributed as dist

        from difflet.backends.neuron.compile import resolve_exec_mode
        from difflet.backends.neuron.core import distributed
        from difflet.cli.dp import stage_loop
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        height, width, num_frames = self._shape(args)
        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv") or "gather_kv",
            cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
            sp_enabled=getattr(args, "sp_enabled", False),
        )
        kwargs = dict(_TORCHRUN_APPLICATION_KWARGS)
        if getattr(args, "device_text_encoder", False):
            kwargs["text_encoder_device"] = "neuron"
        for app_key, arg_key in (("teacache_speedup", "teacache_speedup"),
                                 ("teacache_cadence", "teacache_cadence"),
                                 ("teacache_online_delta_alpha", "teacache_online_delta"),
                                 ("teacache_calibration_path", "teacache_calibration")):
            value = getattr(args, arg_key, None)
            if value is not None:
                kwargs[app_key] = value  # the application refuses it (phase 1)
        start = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(
            _HF_MODEL_ID, model_type=_MODEL_TYPE, revision=args.revision, local_files_only=True,
            parallel=parallel, dtype=torch.bfloat16, height=height, width=width,
            num_frames=num_frames, compile_cache_dir=args.cache_dir, backend=backend,
            application_kwargs=kwargs, load=False,
        )
        with fallback_tracker(pipe) as load_fallbacks:
            pipe.load(skip_warmup=False)
        load_s = time.perf_counter() - start

        generate_fallbacks: list = []
        if args.stage_mode == "compile":
            if distributed.is_rank0():
                write_stage_manifest(self._stage_compiled_dir(args),
                                     self._stage_cache_inputs(args))
        else:
            for req in replica_requests(args, device=getattr(pipe.app, "device", None)):
                with replica_request_scope(args, req):
                    with fallback_tracker(pipe) as fallbacks:
                        out = pipe(
                            prompt=req.prompt,
                            negative_prompt=getattr(req, "negative_prompt", None)
                            or getattr(args, "negative_prompt", None),
                            num_inference_steps=int(
                                stage_loop.effective(req, args, "steps", _DEFAULT_STEPS)),
                            guidance_scale=float(stage_loop.effective(
                                req, args, "guidance_scale", _DEFAULT_GUIDANCE)),
                            generator=torch.Generator().manual_seed(req.seed),
                            output_type="latent",
                        )
                    generate_fallbacks.extend(fallbacks)
                    if distributed.is_rank0():
                        dest = stage_loop.work_file(args, req, "latents.pt")
                        torch.save({"video": out.latents.cpu(), "audio": out.audio_latents.cpu()},
                                   dest)
                        print(f"[ltx_2] latents saved to {dest}", flush=True)

        app = pipe.app
        rank, world = distributed.world_info()
        print(
            f"[ltx_2] {backend} rank {rank}/{world}: "
            f"exec_mode={resolve_exec_mode(envs.DIFFLET_EXEC_MODE)} "
            f"compiled_blocks={len(getattr(app, 'compiled_blocks', []))} "
            f"unwarmed_shapes={list(getattr(app, 'unwarmed_shapes', []))} "
            f"fallbacks: load={list(load_fallbacks)} generate={generate_fallbacks} "
            f"load+warmup {load_s:.1f}s, stage {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        if dist.is_initialized():
            dist.destroy_process_group()

    def _stage_cache_inputs(self, args: argparse.Namespace) -> dict:
        from difflet.backends.neuron.compile import resolve_exec_mode

        height, width, num_frames = self._shape(args)
        if getattr(args, "stage", None) == "vae":
            return {
                "component": "ltx_2_vae",
                "model_id": _HF_MODEL_ID,
                "dtype": "bfloat16",
                "shapes": canonical_shapes_list(args, (height, width, num_frames)),
                "toolchain": stage_toolchain_versions(),
                "backend": torchrun_backend(),
                "exec_mode": "eager",
            }
        return {
            "component": "ltx_2_generate",
            "model_id": _HF_MODEL_ID,
            "tp": args.tp_degree or 4,
            "cp": args.cp_degree or 1,
            "cp_mode": str(getattr(args, "cp_mode", "gather_kv") or "gather_kv"),
            "cfg_parallel": bool(getattr(args, "cfg_parallel", False)),
            "dtype": "bfloat16",
            "text_seq_len": 1024,
            "shapes": canonical_shapes_list(args, (height, width, num_frames)),
            "toolchain": stage_toolchain_versions(),
            "backend": torchrun_backend(),
            "exec_mode": resolve_exec_mode(envs.DIFFLET_EXEC_MODE),
            # Additive: only a device-encoder stage carries the key.
            **({"text_encoder_device": "neuron"}
               if getattr(args, "device_text_encoder", False) else {}),
        }

    def _stage_compiled_dir(self, args: argparse.Namespace) -> Path:
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        inputs = self._stage_cache_inputs(args)
        return hashed_stage_dir(base, str(inputs["component"]), inputs)

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        height, width, num_frames = self._shape()
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--cp-mode", str(getattr(a, "cp_mode", "gather_kv") or "gather_kv"),
            "--height", str(height),
            "--width", str(width),
            "--num-frames", str(num_frames),
            "--steps", str(getattr(a, "steps", None) or _DEFAULT_STEPS),
            "--guidance-scale", str(_guidance(a)),
            "--seed", str(getattr(a, "seed", 42)),
            "--stage-mode", stage_mode,
        ]
        if getattr(a, "shapes", None):
            parts += ["--shapes", str(a.shapes)]
        if getattr(a, "cfg_parallel", False):
            parts.append("--cfg-parallel")
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        if getattr(a, "device_text_encoder", False):
            parts.append("--device-text-encoder")
        for flag, name in (("--teacache-cadence", "teacache_cadence"),
                           ("--teacache-online-delta", "teacache_online_delta"),
                           ("--teacache-speedup", "teacache_speedup"),
                           ("--teacache-calibration", "teacache_calibration"),
                           ("--prompt", "prompt"),
                           ("--negative-prompt", "negative_prompt"),
                           ("--output", "output"),
                           ("--cache-dir", "cache_dir"),
                           ("--revision", "revision")):
            value = getattr(a, name, None)
            if value is not None and value != "":
                parts += [flag, str(value)]
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
        return parts
