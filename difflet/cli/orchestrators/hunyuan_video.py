"""HunyuanVideo T2V orchestrator.

On the Trainium AoT backend (the default), a 3-stage subprocess pipeline:

Stage 1 (clip):     CLIP text encoder,   1 core, VIRTUAL_CORE_SIZE=2.
Stage 2 (llama):    Llama-3 encoder,     tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 3 (generate): DiT + VAE decoder,   tp×cp cores, VIRTUAL_CORE_SIZE=2.
Inter-stage tensors: {work_dir}/clip.pt, {work_dir}/llama.pt.

Stage logic migrated from examples/hunyuan_video_example.py.

With DIFFLET_BACKEND naming a non-AoT torchrun backend (neuron), ``generate`` is a generate stage
of tp x cp ranks that build the registry's application (TorchNeuronHunyuanVideoApplication)
through DiffletPipeline (``_stage_generate_neuron``): rank 0 encodes the prompt on its host with
diffusers' ``HunyuanVideoPipeline.encode_prompt`` (Llama-3 and CLIP-L, the tensors the clip and
llama stages produce) and broadcasts them, the ranks denoise (the DiT TP-sharded on every rank's
core, with SP, or CP in any mode) and rank 0 writes the latents. The decode follows LTX-2's
placement: on the host in the driver after the launch has exited (``_decode_latents_host``), or,
with --device-vae, in a vae stage of its own on one core (``_stage_vae_neuron``); both run
diffusers' AutoencoderKLHunyuanVideo in fp32, tiled in space and time
(NeuronHunyuanVideoVaeDecoderApplication), and diffusers' video post-processing. ``compile`` is
the generate stage's load + warm-up (it fills the NEFF cache) and its stage manifest, plus the vae
stage's warm-up decode with --device-vae. --cache-dir becomes DIFFLET_COMPILE_CACHE unless that is
exported. TeaCache runs in every mode. Refused here, before torchrun starts: several --shapes,
--cfg-parallel (the model is guidance-distilled), --host-vae with --device-vae, and a TP degree
whose DiT shard cannot fit one core (tp1: 23.9 GiB of bf16 DiT on a 24 GiB core). DP replicas
(--dp, --requests) are separate torchrun launches whose rank 0 claims the requests. Unset,
trainium and tpu keep the AoT path unchanged.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    canonical_shapes_list,
    has_valid_stage_manifest,
    hashed_stage_dir,
    require_request_shape_in_set,
    stage_toolchain_versions,
    write_stage_manifest,
)
from difflet.cli.orchestrators.torchrun import (
    decodes_on_device,
    refuse_dit_overflow,
    refuse_torchrun_conflicts,
    reject_unsupported_torchrun_options,
    replica_cores,
    root_compile_cache_at_cache_dir,
    run_generate_stage,
    run_vae_stage,
    torchrun_backend,
    torchrun_stage_keys,
    torchrun_unsupported_options,
    vae_placement_conflict,
)

_HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
_MODEL_TYPE = "hunyuan_video"
_CLI_NAME = "hunyuan-video"
_VIRTUAL_CORE_SIZE = 2

_LLAMA_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the "
    "following aspects: 1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
_LLAMA_CROP_START = 95
_LLAMA_CAPTURE = "layers.29"
_TEXT_SEQ_LEN = 256


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
        export_to_video(list(frames), output_path, fps=24)
    except Exception as exc:
        print(f"[hunyuan_video] mp4 export failed ({exc}); saving as .pt instead", flush=True)
        return False
    print(f"[hunyuan_video] video saved to {output_path}", flush=True)
    return True


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(320, 512, 61), model_tag="hunyuan_video"
    )


# ------------------------------------------------------------ torchrun (neuron) backend

#: The Trainium stage's defaults for a request that names none.
_DEFAULT_SHAPE = (320, 512, 61)
_DEFAULT_STEPS = 4
_DEFAULT_GUIDANCE = 6.0
_FPS = 24
#: What the neuron HunyuanVideo ranks run, for the driver's rejection message.
_TORCHRUN_RUNS = ("tensor, sequence and context parallelism (gather_kv, ulysses, ring) at one "
                  "static shape, DP replicas and TeaCache")
#: TorchNeuronHunyuanVideoApplication keyword arguments for the torchrun generate stage: the
#: Llama/CLIP encode on rank 0's host with 12 threads, no decode in the launch (the driver or the
#: vae stage decodes after it) and exec_mode=None (the application reads DIFFLET_EXEC_MODE,
#: which --exec-mode exports).
_TORCHRUN_APPLICATION_KWARGS = {
    "text_seq_len": _TEXT_SEQ_LEN,
    "batch_size": 1,
    "enable_vae_decoder": False,
    "exec_mode": None,
    "encoder_threads": 12,
    "text_encoder_device": "host",
}
#: Where the neuron backend decodes when neither --host-vae nor --device-vae is given: on the
#: host in the driver, LTX-2's default. The vae stage's launch and load have not been measured
#: against the host decode for HunyuanVideo yet.
_DEVICE_VAE_DEFAULT = False


def _shape(args: argparse.Namespace) -> tuple[int, int, int]:
    return (args.height or _DEFAULT_SHAPE[0], args.width or _DEFAULT_SHAPE[1],
            args.num_frames or _DEFAULT_SHAPE[2])


def _torchrun_application_kwargs(args: argparse.Namespace) -> dict:
    """The application kwargs, plus any TeaCache flag the user gave (the application builds the
    controller and, for adaptive TeaCache, the probe on the DiT's own modules)."""
    kwargs = dict(_TORCHRUN_APPLICATION_KWARGS)
    for app_key, arg_key in (("teacache_speedup", "teacache_speedup"),
                             ("teacache_cadence", "teacache_cadence"),
                             ("teacache_online_delta_alpha", "teacache_online_delta"),
                             ("teacache_calibration_path", "teacache_calibration")):
        value = getattr(args, arg_key, None)
        if value is not None:
            kwargs[app_key] = value
    return kwargs


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options the torchrun (neuron) ranks do not run, as the command line names them: every
    torchrun backend's but TeaCache, which runs on HunyuanVideo's neuron path (several --shapes:
    one static shape per process). Torch-free."""
    return [f for f in torchrun_unsupported_options(args) if not f.startswith("--teacache")]


def _torchrun_conflicts(args: argparse.Namespace) -> list[str]:
    """--cfg-parallel (there are no CFG branches to split) and --host-vae with --device-vae.
    Torch-free."""
    cfg = None
    if getattr(args, "cfg_parallel", False):
        cfg = ("--cfg-parallel does not apply: HunyuanVideo is guidance-distilled (one forward "
               "pass per step, the guidance scale baked into the timestep embedding)")
    return [c for c in (cfg, vae_placement_conflict(args)) if c is not None]


def _decodes_on_device(args: argparse.Namespace) -> bool:
    """Whether the neuron backend decodes in a vae stage on one core (``_stage_vae_neuron``)
    instead of on the host in the driver (``_decode_latents_host``). --host-vae wins, then
    --device-vae, then ``_DEVICE_VAE_DEFAULT``. Torch-free."""
    return decodes_on_device(args, _DEVICE_VAE_DEFAULT)


def _write_video(video, output: str) -> None:
    """The decoded clip ([1, 3, F, H, W] in [-1, 1]) to ``output``: diffusers' VideoProcessor
    post-processing, as HunyuanVideoPipeline does, then an MP4 when ``output`` ends in .mp4 and
    encoding works; else the decoded tensor as a .pt."""
    import torch

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        from diffusers.utils import export_to_video
        from diffusers.video_processor import VideoProcessor

        # Float frames in [0, 1]: export_to_video multiplies ndarray frames by 255 itself.
        frames = VideoProcessor(vae_scale_factor=8).postprocess_video(
            video.detach().float().cpu(), output_type="np")[0]
        try:
            export_to_video(list(frames), str(out), fps=_FPS)
            print(f"[hunyuan_video] video saved to {out}", flush=True)
            return
        except Exception as exc:
            print(f"[hunyuan_video] mp4 export failed ({exc}); saving as .pt instead", flush=True)
    pt_path = out.with_suffix(".pt")
    torch.save(video.cpu(), pt_path)
    print(f"[hunyuan_video] video tensor saved to {pt_path}", flush=True)


def _decode_latents_host(latents_path: str, model_dir: str, output_path: str, *, height: int,
                         width: int, num_frames: int) -> None:
    """Decode the generate stage's latents on the host CPU with the vae stage's decoder
    (NeuronHunyuanVideoVaeDecoderApplication: diffusers' AutoencoderKLHunyuanVideo in fp32,
    tiled in space and time), then write the frames to ``output_path`` (``_write_video``)."""
    import torch

    from difflet.backends.neuron.hunyuan_video.vae import NeuronHunyuanVideoVaeDecoderApplication

    vae = NeuronHunyuanVideoVaeDecoderApplication(
        model_path=str(Path(model_dir) / "vae"), height=height, width=width,
        num_frames=num_frames, device="cpu",
    )
    vae.load(skip_warmup=True)
    _write_video(vae.decode(torch.load(latents_path, map_location="cpu")), output_path)


class HunyuanVideoOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def run(self) -> None:
        self._reject_unsupported_torchrun_options()  # before the download, too
        super().run()

    def compile(self) -> None:
        if torchrun_backend() is not None:
            self._compile_torchrun()
            return
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
        runner.run_stage(_HF_MODEL_ID, "clip",
                         num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)
        runner.run_stage(_HF_MODEL_ID, "llama",
                         num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)
        runner.run_stage(_HF_MODEL_ID, "generate",
                         num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)

    def generate(self) -> None:
        if torchrun_backend() is not None:
            self._generate_torchrun()
            return
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(self.args.work_dir or
                        Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME)
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        try:
            runner.run_stage(_HF_MODEL_ID, "clip",
                             num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)
            runner.run_stage(_HF_MODEL_ID, "llama",
                             num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)
            runner.run_stage(_HF_MODEL_ID, "generate",
                             num_cores=full_cores, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared)
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------- torchrun (neuron) backend

    def _reject_unsupported_torchrun_options(self) -> None:
        """A torchrun backend (neuron) asked for an option its ranks do not run, or a pair they
        cannot run together: print one message naming it and exit 2 before anything is
        launched (or downloaded, for ``run``). The ranks' own checks stay (defence in depth).
        No-op on the AoT backends."""
        reject_unsupported_torchrun_options(_torchrun_unsupported_options(self.args),
                                            runs=_TORCHRUN_RUNS)
        refuse_torchrun_conflicts(_torchrun_conflicts(self.args))

    def _model_dir(self) -> str:
        from difflet.pipeline.path_resolver import resolve_model_path

        return resolve_model_path(_HF_MODEL_ID, revision=getattr(self.args, "revision", None),
                                  local_files_only=True)

    def _check_dit_fits(self) -> None:
        """Refuse a TP degree whose DiT shard clearly cannot fit one core
        (``difflet/backends/neuron/hunyuan_video/hbm.py``; tp1) before any rank starts: print
        one message and exit 2. Weights that are not local yet leave the decision to the
        application's own check. Torch-free."""
        from difflet.backends.neuron.hunyuan_video.hbm import dit_fits

        refuse_dit_overflow("HunyuanVideo", self._model_dir, dit_fits, self.args.tp_degree or 4)

    def _prepare_torchrun(self) -> None:
        self._reject_unsupported_torchrun_options()
        self._check_dit_fits()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None),
                                        tag="hunyuan_video")

    def _compile_torchrun(self) -> None:
        """``compile`` on a torchrun backend: the generate stage's load + warm-up (it fills the
        NEFF cache) and its stage manifest; with --device-vae, the vae stage's warm-up decode."""
        self._prepare_torchrun()
        shared = self._shared_cli_args(stage_mode="compile")
        runner.run_stage(_HF_MODEL_ID, "generate", num_cores=replica_cores(self.args),
                         virtual_core_size=None, cli_args=shared)
        if _decodes_on_device(self.args):
            runner.run_stage(_HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=None,
                             cli_args=shared)

    def _generate_torchrun(self) -> None:
        """``generate`` on a torchrun backend: the generate stage (prompt encode on rank 0's
        host, denoise, rank 0 writes latents.pt), then the decode of every request it finished,
        after torchrun has exited: in a one-core vae stage (--device-vae) or on the host."""
        self._prepare_torchrun()
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(getattr(self.args, "work_dir", None)
                        or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME)
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        try:
            runner.run_stage(_HF_MODEL_ID, "generate", num_cores=replica_cores(self.args),
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

    def _decode_on_host(self, work_dir: Path) -> None:
        """The host decode of every request the generate launch finished: the one request of a
        plain launch, or, in a DP worker, each request this worker's replica claimed and did
        not fail, marked done after its video is written."""
        from difflet.cli.dp import stage_loop

        args = argparse.Namespace(**{**vars(self.args), "work_dir": str(work_dir)})
        height, width, num_frames = _shape(args)
        model_dir = self._model_dir()
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                _decode_latents_host(str(stage_loop.work_file(args, req, "latents.pt")),
                                     model_dir, req.output, height=height, width=width,
                                     num_frames=num_frames)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        backend = torchrun_backend()
        if backend is not None:
            if stage == "generate":
                self._stage_generate_neuron(args, backend)
            elif stage == "vae":
                self._stage_vae_neuron(args)
            else:
                raise ValueError(
                    f"HunyuanVideo has stages 'generate' and 'vae' on a torchrun backend (got "
                    f"{stage!r}, backend {backend!r}); clip and llama are the Trainium path's"
                )
            return
        if stage == "clip":
            self._stage_clip(args)
        elif stage == "llama":
            self._stage_llama(args)
        elif stage == "generate":
            self._stage_generate(args)
        else:
            raise ValueError(f"Unknown HunyuanVideo stage: {stage!r}")

    # ---------------------------------------------------------------- stages

    def _stage_clip(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import CLIPTokenizer

        from difflet.backends.trainium.core.config import NeuronConfig
        from difflet.models.flux.clip.modeling_clip import (
            CLIPInferenceConfig,
            NeuronClipApplication,
        )
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.utils.diffusers_adapter import load_diffusers_config

        model_dir = resolve_model_path(_HF_MODEL_ID, local_files_only=True)
        clip_path = str(Path(model_dir) / "text_encoder_2")
        compiled_dir = self._stage_compiled_dir("clip", args)

        config = CLIPInferenceConfig(
            neuron_config=NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.bfloat16),
            load_config=load_diffusers_config(clip_path),
        )
        for key, val in {"output_attentions": False, "output_hidden_states": False,
                         "use_return_dict": True}.items():
            setattr(config, key, val)

        app = NeuronClipApplication(model_path=clip_path, config=config)
        if args.stage_mode == "compile":
            if not self._stage_is_compiled("clip", args, compiled_dir):
                app.compile(str(compiled_dir))
                self._finish_stage_compile("clip", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("clip", args, compiled_dir)
        app.load(str(compiled_dir))
        tok = CLIPTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                ids = tok(req.prompt, padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to(torch.int64)
                out = app(ids)
                pooled = out.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
                dest = stage_loop.work_file(args, req, "clip.pt")
                torch.save({"pooled_projections": pooled}, dest)
                print(f"[clip] pooled_projections {tuple(pooled.shape)} -> {dest}")

    def _stage_llama(self, args: argparse.Namespace) -> None:
        import torch
        from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        from transformers import AutoConfig, AutoTokenizer

        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(_HF_MODEL_ID, local_files_only=True)
        enc_path = str(Path(model_dir) / "text_encoder")
        seq = _TEXT_SEQ_LEN + _LLAMA_CROP_START
        compiled_dir = self._stage_compiled_dir("llama", args)
        tp_degree = args.tp_degree or 4

        hf_cfg = AutoConfig.from_pretrained(enc_path)
        if hf_cfg.pad_token_id is None:
            hf_cfg.pad_token_id = 0
        hf_cfg.tie_word_embeddings = True

        # This stage uses NxDI's stock NeuronConfig (not difflet's fork), whose
        # save_sharded_checkpoint default is False, so it always re-sharded the
        # Llama encoder on load (~28s CPU-bound cost per warm run). Stock
        # load_weights() has no missing-shard fallback (unlike difflet's own
        # application_base.py), so only request the presharded read path once
        # the shard files actually exist; always request the write path at
        # compile time.
        weights_dir = compiled_dir / "weights"
        shard_paths = [weights_dir / f"tp{r}_sharded_checkpoint.safetensors" for r in range(tp_degree)]
        save_sharded_checkpoint = (
            args.stage_mode == "compile" or all(p.exists() for p in shard_paths)
        )

        neuron_config = NeuronConfig(
            tp_degree=tp_degree, batch_size=1, seq_len=seq,
            torch_dtype=torch.bfloat16, on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=[_LLAMA_CAPTURE]),
            save_sharded_checkpoint=save_sharded_checkpoint,
        )
        config = NeuronLlamaForCausalLM.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
        )
        app = NeuronLlamaForCausalLM(enc_path, config)
        if args.stage_mode == "compile":
            if not self._stage_is_compiled("llama", args, compiled_dir):
                app.compile(str(compiled_dir))
                self._finish_stage_compile("llama", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("llama", args, compiled_dir)
        app.load(str(compiled_dir))
        tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                ti = tok(
                    _LLAMA_TEMPLATE.format(req.prompt), max_length=seq,
                    padding="max_length", truncation=True, return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids = ti.input_ids.to(torch.int32)
                attn = ti.attention_mask.to(torch.int32)
                out = app(
                    input_ids=input_ids,
                    attention_mask=attn,
                    position_ids=torch.arange(seq, dtype=torch.int32).unsqueeze(0),
                    sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                )
                hidden = out.captured_tensors[0][:, _LLAMA_CROP_START:].to(torch.bfloat16).cpu()
                mask = attn[:, _LLAMA_CROP_START:].to(torch.int64)
                dest = stage_loop.work_file(args, req, "llama.pt")
                torch.save({"encoder_hidden_states": hidden, "encoder_attention_mask": mask},
                           dest)
                print(f"[llama] encoder_hidden_states {tuple(hidden.shape)} -> {dest}")

    def _stage_generate(self, args: argparse.Namespace) -> None:
        import numpy as np
        import torch

        from difflet.models.hunyuan_video.application import (
            HunyuanVideoDiTInputBundle,
            NeuronHunyuanVideoApplication,
        )
        from difflet.models.hunyuan_video.pipeline import _retrieve_timesteps
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(_HF_MODEL_ID, local_files_only=True)
        compiled_dir = self._stage_compiled_dir("generate", args)
        work_dir = Path(args.work_dir)

        h, w, f = args.height or 320, args.width or 512, args.num_frames or 61
        latent_frames = (f - 1) // 4 + 1
        compile_shapes = _require_request_shape_in_set(args)

        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            sp_enabled=getattr(args, "sp_enabled", False),
        )
        # Adaptive TeaCache (--teacache-speedup + --teacache-calibration) is the
        # only mode that needs the probe NEFF (block-0 modulated input + the
        # on-device L2 delta). Plain and probe-free (cadence / online-delta)
        # runs opt out, so no probe is compiled or loaded for them.
        adaptive = getattr(args, "teacache_speedup", None) is not None
        app = NeuronHunyuanVideoApplication(
            model_path=model_dir, parallel=parallel, dtype=torch.bfloat16,
            shape={"height": h, "width": w, "num_frames": f},
            shapes=compile_shapes,
            text_seq_len=_TEXT_SEQ_LEN, enable_vae_decoder=True,
            # Probe-free TeaCache (fixed cadence / online-delta): host-side skip
            # logic only; not in _stage_cache_inputs, so the warm artifact hits.
            teacache_cadence=getattr(args, "teacache_cadence", None),
            teacache_online_delta_alpha=getattr(args, "teacache_online_delta", None),
            enable_teacache_probe=adaptive,
            teacache_speedup=getattr(args, "teacache_speedup", None),
            teacache_calibration_path=getattr(args, "teacache_calibration", None),
        )

        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            self._finish_stage_compile("generate", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("generate", args, compiled_dir)
        if adaptive and not app.has_compiled_artifacts(str(compiled_dir), select=["teacache_probe"]):
            # The probe is an additive component of the DiT stage artifact
            # (<compiled_dir>/teacache_probe/): its identity is the DiT's, the
            # DiT/VAE NEFFs are reused untouched, and it is built on first use.
            print(
                f"[hunyuan_video] compiling the TeaCache probe NEFF into {compiled_dir} "
                "(first adaptive run; the DiT artifact is reused)",
                flush=True,
            )
            app.compile(str(compiled_dir), select=["teacache_probe"])
        app.load(str(compiled_dir), skip_warmup=True)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                llama = torch.load(stage_loop.work_file(args, req, "llama.pt"))
                clip = torch.load(stage_loop.work_file(args, req, "clip.pt"))
                torch.manual_seed(req.seed)
                latents = torch.randn(
                    1, 16, latent_frames, h // 8, w // 8, dtype=torch.bfloat16
                )
                guidance = torch.full(
                    [1],
                    float(stage_loop.effective(req, args, "guidance_scale", 6.0)) * 1000.0,
                    dtype=torch.bfloat16,
                )
                steps = int(stage_loop.effective(req, args, "steps", 4))
                sigmas = np.linspace(1.0, 0.0, steps + 1)[:-1]
                timesteps, _ = _retrieve_timesteps(
                    app.pipeline.scheduler, steps, "cpu", sigmas=sigmas
                )
                bundle = HunyuanVideoDiTInputBundle(
                    hidden_states=latents,
                    timestep=timesteps[:1].clone(),
                    encoder_hidden_states=llama["encoder_hidden_states"],
                    encoder_attention_mask=llama["encoder_attention_mask"].to(torch.int64),
                    pooled_projections=clip["pooled_projections"],
                    guidance=guidance,
                )
                output = app(
                    bundle=bundle, timesteps=timesteps,
                    num_inference_steps=steps,
                    output_type="pt", return_trajectory=False,
                )
                frames = output.frames
                out_path = Path(req.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.suffix == ".mp4" and _save_video(frames.cpu(), str(out_path)):
                    pass
                else:
                    pt_path = out_path.with_suffix(".pt")
                    torch.save(frames.cpu(), pt_path)
                    print(f"[generate] video tensor saved to {pt_path}")

    def _stage_generate_neuron(self, args: argparse.Namespace, backend: str) -> None:
        """The generate stage on a non-AoT torchrun backend (neuron); runs on every rank.

        ``run_generate_stage`` builds the registry's application
        (TorchNeuronHunyuanVideoApplication) through ``DiffletPipeline``, loads it with the
        warm-up (``compile`` mode is that plus the stage manifest) and runs the replica's
        requests. HunyuanVideo's part: the Trainium stage's step and guidance defaults, the
        request's seed (the application draws the initial noise from it on rank 0), and the
        latents written by rank 0 to the request's ``latents.pt``.
        """
        import torch

        from difflet.cli.dp import stage_loop

        compile_shapes = _require_request_shape_in_set(args)
        height, width, num_frames = _shape(args)

        def request_kwargs(req) -> dict:
            return dict(
                prompt=req.prompt,
                generator=torch.Generator().manual_seed(req.seed),
                num_inference_steps=int(stage_loop.effective(req, args, "steps", _DEFAULT_STEPS)),
                guidance_scale=float(stage_loop.effective(
                    req, args, "guidance_scale", _DEFAULT_GUIDANCE)),
                output_type="latent",
            )

        def save(req, out) -> None:
            dest = stage_loop.work_file(args, req, "latents.pt")
            torch.save(out.latents.cpu(), dest)
            print(f"[hunyuan_video] latents saved to {dest}", flush=True)

        run_generate_stage(
            args, backend, tag="hunyuan_video", model_id=_HF_MODEL_ID,
            pipeline_kwargs=dict(model_type=_MODEL_TYPE, revision=getattr(args, "revision", None),
                                 height=height, width=width, num_frames=num_frames,
                                 shapes=compile_shapes),
            application_kwargs=_torchrun_application_kwargs(args),
            request_kwargs=request_kwargs,
            save=save,
            finish_compile=lambda: self._finish_stage_compile(
                "generate", args, self._stage_compiled_dir("generate", args)),
        )

    def _stage_vae_neuron(self, args: argparse.Namespace) -> None:
        """The neuron backend's vae stage (``run_vae_stage``; --device-vae): one rank, a launch
        of its own after the generate stage has exited, so the decoder never shares a core with
        the DiT. NeuronHunyuanVideoVaeDecoderApplication decodes each request's latents (fp32,
        eager, tiled in space and time), and the frames are written as the host decode writes
        them."""
        import torch

        from difflet.backends.neuron.hunyuan_video.vae import NeuronHunyuanVideoVaeDecoderApplication
        from difflet.cli.dp import stage_loop

        model_dir = self._model_dir()
        height, width, num_frames = _shape(args)

        def build(parallel):
            return NeuronHunyuanVideoVaeDecoderApplication(
                model_path=str(Path(model_dir) / "vae"), height=height, width=width,
                num_frames=num_frames, device="neuron",
            )

        def prepare_decode(decoder):
            def decode(req) -> None:
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"),
                                     map_location="cpu")
                _write_video(decoder.decode(latents), req.output)

            return decode

        run_vae_stage(
            args, build=build, prepare_decode=prepare_decode,
            finish_compile=lambda: self._finish_stage_compile(
                "vae", args, self._stage_compiled_dir("vae", args)),
        )

    # ------------------------------------------------------------ helpers

    def _stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        """Identity for one stage artifact (hashed into the dir name; recorded
        verbatim in the dir's manifest.json)."""
        if torchrun_backend() is not None:
            return self._torchrun_stage_cache_inputs(stage, args)
        if stage == "clip":
            return {
                "component": "hunyuan_video_clip",
                "model_id": _HF_MODEL_ID,
                "dtype": "bfloat16",
                "toolchain": stage_toolchain_versions(),
            }
        if stage == "llama":
            return {
                "component": "hunyuan_video_llama",
                "model_id": _HF_MODEL_ID,
                "tp": args.tp_degree or 4,
                "seq_len": _TEXT_SEQ_LEN + _LLAMA_CROP_START,
                "dtype": "bfloat16",
                "toolchain": stage_toolchain_versions(),
            }
        if stage == "generate":
            return {
                "component": "hunyuan_video_dit",
                "model_id": _HF_MODEL_ID,
                "tp": args.tp_degree or 4,
                "cp": args.cp_degree or 1,
                "cp_mode": str(getattr(args, "cp_mode", "gather_kv") or "gather_kv"),
                "sp": bool(getattr(args, "sp_enabled", False)),
                "dtype": "bfloat16",
                "text_seq_len": _TEXT_SEQ_LEN,
                "shapes": canonical_shapes_list(args, (320, 512, 61)),
                "toolchain": stage_toolchain_versions(),
            }
        raise ValueError(f"unknown stage {stage!r}")

    def _torchrun_stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        """The neuron stages' identities: the generate stage under a component of its own (it
        holds the text encode too, and never matches an AoT DiT artifact) with the backend and
        the exec mode (``torchrun_stage_keys``), and the vae stage, eager whatever --exec-mode
        says. There is no clip or llama stage."""
        shapes = canonical_shapes_list(args, _DEFAULT_SHAPE)
        if stage == "generate":
            return torchrun_stage_keys({
                "component": "hunyuan_video_generate",
                "model_id": _HF_MODEL_ID,
                "tp": args.tp_degree or 4,
                "cp": args.cp_degree or 1,
                "cp_mode": str(getattr(args, "cp_mode", "gather_kv") or "gather_kv"),
                "sp": bool(getattr(args, "sp_enabled", False)),
                "dtype": "bfloat16",
                "text_seq_len": _TEXT_SEQ_LEN,
                "shapes": shapes,
                "toolchain": stage_toolchain_versions(),
            })
        if stage == "vae":
            return {
                "component": "hunyuan_video_vae",
                "model_id": _HF_MODEL_ID,
                "dtype": "float32",
                "shapes": shapes,
                "toolchain": stage_toolchain_versions(),
                "backend": torchrun_backend(),
                "exec_mode": "eager",
            }
        raise ValueError(f"unknown stage {stage!r} on a torchrun backend")

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        inputs = self._stage_cache_inputs(stage, args)
        return hashed_stage_dir(base, str(inputs["component"]), inputs)

    def _finish_stage_compile(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        write_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args))

    def _require_stage_artifact(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        if not has_valid_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args)):
            raise SystemExit(
                f"[hunyuan_video] no valid compiled artifact for stage {stage!r} at "
                f"{compiled_dir} (manifest missing or configuration changed); run "
                "`difflet compile` with the same flags first."
            )

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--cp-mode", str(getattr(a, "cp_mode", "gather_kv")),
            "--height", str(a.height or 320),
            "--width", str(a.width or 512),
            "--num-frames", str(a.num_frames or 61),
            "--steps", str(getattr(a, "steps", None) or 4),
            "--guidance-scale", str(getattr(a, "guidance_scale", None) or 6.0),
            "--seed", str(getattr(a, "seed", 42)),
            "--stage-mode", stage_mode,
        ]
        if getattr(a, "shapes", None):
            parts += ["--shapes", str(a.shapes)]
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        if getattr(a, "teacache_cadence", None) is not None:
            parts += ["--teacache-cadence", str(a.teacache_cadence)]
        if getattr(a, "teacache_online_delta", None) is not None:
            parts += ["--teacache-online-delta", str(a.teacache_online_delta)]
        if getattr(a, "teacache_speedup", None) is not None:
            parts += ["--teacache-speedup", str(a.teacache_speedup)]
        if getattr(a, "teacache_calibration", None):
            parts += ["--teacache-calibration", str(a.teacache_calibration)]
        if getattr(a, "prompt", None):
            parts += ["--prompt", a.prompt]
        if getattr(a, "output", None):
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        # Only when given (--exec-mode exists on the torchrun backends only), so an AoT command
        # line stays byte-identical.
        if getattr(a, "exec_mode", None):
            parts += ["--exec-mode", a.exec_mode]
        if getattr(a, "revision", None):
            parts += ["--revision", a.revision]
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
        return parts
