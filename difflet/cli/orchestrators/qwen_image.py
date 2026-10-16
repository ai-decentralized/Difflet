"""Qwen-Image T2I orchestrator — 3-stage subprocess pipeline.

Stage 1 (text):     Qwen2.5-VL text encoder,  tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 2 (generate): DiT backbone,              tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 3 (vae):      VAE decoder,               1 core,      VIRTUAL_CORE_SIZE=2.
Inter-stage tensors: {work_dir}/text.pt, {work_dir}/latents.pt.

Stage logic migrated from examples/qwen_image_example.py.

With DIFFLET_BACKEND naming a non-AoT torchrun backend (neuron), the plan is one launch and a
host decode: the generate stage runs as tp*cp torchrun ranks (default tp 4, every core of a
trn2.3xlarge) that build the registry's application (TorchNeuronQwenImageApplication) through
DiffletPipeline (``_stage_generate_neuron``). The text stage is folded into it: rank 0 encodes
the prompt with Qwen2.5-VL on its host and broadcasts the embeddings, the ranks denoise, and rank
0 writes latents.pt in the packed format the vae stage reads. The driver then decodes it on the
host (diffusers ``AutoencoderKLQwenImage``, fp32: ``_decode_latents_host``), so there is no text
or vae launch and --host-vae changes nothing. ``compile`` is the generate stage's load + warm-up
(it fills the NEFF cache) and its stage manifest; the stage cache key names the backend and exec
mode; --cache-dir becomes DIFFLET_COMPILE_CACHE unless that is exported; the options the ranks
reject (--sp, --cfg-parallel, TeaCache, several --shapes, --requests/--dp) are refused by the
driver before torchrun starts. Unset, trainium and tpu keep their three stages unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from difflet import envs
from difflet.common.orchestrators import qwen_image as qwen_common
from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    canonical_shapes_list,
    has_valid_stage_manifest,
    hashed_stage_dir,
    parse_shapes_arg,
    require_request_shape_in_set,
    stage_toolchain_versions,
    write_stage_manifest,
)
from difflet.cli.orchestrators.torchrun import (
    fallback_tracker,
    reject_unsupported_torchrun_options,
    root_compile_cache_at_cache_dir,
    sp_and_batch_options,
    torchrun_backend,
    torchrun_unsupported_options,
)
from difflet.cli import runner

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_CLI_NAME = qwen_common.CLI_NAME
_VIRTUAL_CORE_SIZE = qwen_common.VIRTUAL_CORE_SIZE
_ENC_SEQ = qwen_common.ENC_SEQ
_TEXT_SEQ_LEN = qwen_common.TEXT_SEQ_LEN

_QWEN_TEMPLATE = qwen_common.QWEN_TEMPLATE
_QWEN_DROP_IDX = qwen_common.QWEN_DROP_IDX


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(1024, 1024), model_tag="qwen_image"
    )


# TorchNeuronQwenImageApplication keyword arguments for the torchrun generate stage: the
# Trainium generate stage's set, Qwen2.5-VL on rank 0's host (in its default bf16, the Trainium
# text stage's dtype) with 12 threads, no device VAE (the driver decodes on the host) and
# exec_mode=None (the application reads DIFFLET_EXEC_MODE, which --exec-mode exports).
_TORCHRUN_APPLICATION_KWARGS = {
    "text_seq_len": _TEXT_SEQ_LEN,
    "batch_size": 1,
    "enable_text_encoder": True,
    "enable_transformer": True,
    "enable_vae_decoder": False,
    "exec_mode": None,
    "encoder_threads": 12,
}
#: What the neuron Qwen-Image ranks support, for the driver's rejection message.
_TORCHRUN_RUNS = "tensor and context parallelism at one static shape"


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options the torchrun (neuron) generate stage rejects in phase 1, as the command line
    names them: every torchrun backend's (TeaCache), --sp and request batches (the Qwen ranks
    run no sequence parallelism and no DP replica loop, unlike Wan's), plus --cfg-parallel
    (Qwen-Image is guidance-distilled: one forward per step) and a --shapes with more than one
    shape (the application holds one static shape per process). Torch-free: the shapes are
    counted, not parsed, so a malformed --shapes fails where it always did."""
    found = torchrun_unsupported_options(args) + sp_and_batch_options(args)
    if getattr(args, "cfg_parallel", False):
        found.append("--cfg-parallel")
    shapes = getattr(args, "shapes", None)
    if shapes and len([token for token in str(shapes).split(",") if token.strip()]) > 1:
        found.append("--shapes with more than one shape")
    return found


def _torchrun_application_kwargs(args: argparse.Namespace) -> dict:
    """The application kwargs, plus any TeaCache flag the user gave: the neuron application
    rejects TeaCache (phase 1) instead of the stage dropping the flag silently."""
    kwargs = dict(_TORCHRUN_APPLICATION_KWARGS)
    for app_key, arg_key in (
        ("teacache_speedup", "teacache_speedup"),
        ("teacache_cadence", "teacache_cadence"),
        ("teacache_online_delta_alpha", "teacache_online_delta"),
        ("teacache_calibration_path", "teacache_calibration"),
    ):
        value = getattr(args, arg_key, None)
        if value is not None:
            kwargs[app_key] = value
    return kwargs


def _unpack_latents(packed, height: int, width: int):
    """Packed DiT latents ``[B, (H/16)*(W/16), 4*C]`` -> the VAE's one-frame latent video
    ``[B, C, 1, H/8, W/8]`` (diffusers ``QwenImagePipeline._unpack_latents``: the VAE's 8x
    scale and the DiT's 2x2 patch). The grid follows the REQUEST shape, which need not be
    square."""
    b, seq, packed_channels = packed.shape
    hh = height // 16
    ww = width // 16
    if hh * ww != seq:
        raise ValueError(
            f"packed latent length {seq} does not match request grid {hh}x{ww}"
        )
    channels = packed_channels // 4
    return (
        packed.view(b, hh, ww, channels, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, channels, hh * 2, ww * 2)
        .unsqueeze(2)
    )


def _save_image_torchvision(image, path: Path) -> None:
    from torchvision.utils import save_image

    save_image(image, str(path))


def _save_image_pil(image, path: Path) -> None:
    """``torchvision.utils.save_image`` of one ``[3, H, W]`` image in [0, 1], pixel for pixel
    (``x * 255 + 0.5`` truncated to uint8), through PIL, which diffusers already requires: the
    neuron venv has no torchvision."""
    import torch
    from PIL import Image

    pixels = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)
    Image.fromarray(pixels.numpy()).save(str(path))


def _write_image(img, output: str, save=_save_image_torchvision) -> None:
    """Decoded images ``[B, 3, H, W]`` in [-1, 1]: the first written by ``save`` as an image
    file at ``output``, or, when that fails, the tensor next to it with a .pt suffix."""
    import torch

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save((img[0] * 0.5 + 0.5).clamp(0, 1), out_path)
        print(f"[vae] image saved to {out_path}")
    except Exception as exc:
        torch.save(img, out_path.with_suffix(".pt"))
        print(
            f"[vae] tensor saved to {out_path.with_suffix('.pt')} "
            f"(png skipped: {exc})"
        )


def _decode_latents_host(
    latents_path: str, model_dir: str, output_path: str, *, height: int, width: int
) -> None:
    """Decode the generate stage's packed latents with diffusers' ``AutoencoderKLQwenImage`` on
    the host CPU in fp32: the neuron backend's VAE path. The driver runs it after the torchrun
    generate stage has exited, so the decoder never shares a core (or a process) with the DiT.

    The decode of the Trainium vae stage and of diffusers' ``QwenImagePipeline``: unpack to a
    one-frame latent video, un-normalise (``z * latents_std + latents_mean``), decode, keep
    frame 0. fp32 rather than the Trainium stage's bf16: the host has the memory for it. The
    image goes through PIL (``_save_image_pil``: the pixels torchvision would write).
    """
    import torch
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        str(Path(model_dir) / "vae"), torch_dtype=torch.float32
    ).eval()
    packed = torch.load(latents_path, map_location="cpu").to(torch.float32)
    z = _unpack_latents(packed, height, width)
    config = vae.config
    mean = torch.tensor(config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        frames = vae.decode(z * std + mean, return_dict=False)[0]
    _write_image(frames[:, :, 0].float().cpu(), output_path, save=_save_image_pil)


class QwenImageOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path

        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def _reject_unsupported_torchrun_options(self) -> None:
        """A torchrun backend (neuron) asked for an option its ranks reject in phase 1: print
        one message naming it and exit 2 before anything is launched (or downloaded, for
        ``run``). The ranks' own checks stay (defence in depth)."""
        reject_unsupported_torchrun_options(
            _torchrun_unsupported_options(self.args), runs=_TORCHRUN_RUNS
        )

    def run(self) -> None:
        self._reject_unsupported_torchrun_options()  # before the download, too
        super().run()

    def compile(self) -> None:
        self._reject_unsupported_torchrun_options()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="qwen_image")
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
        if torchrun_backend() is not None:
            # One launch: the generate stage's load + warm-up (the text encoder is part of it)
            # fills the NEFF cache and writes the manifest; the host VAE compiles nothing.
            runner.run_stage(
                _HF_MODEL_ID,
                "generate",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            return
        runner.run_stage(
            _HF_MODEL_ID,
            "text",
            num_cores=full_cores,
            virtual_core_size=_VIRTUAL_CORE_SIZE,
            cli_args=shared,
        )
        runner.run_stage(
            _HF_MODEL_ID,
            "generate",
            num_cores=full_cores,
            virtual_core_size=_VIRTUAL_CORE_SIZE,
            cli_args=shared,
        )
        runner.run_stage(
            _HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE, cli_args=shared
        )

    def generate(self) -> None:
        self._reject_unsupported_torchrun_options()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="qwen_image")
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(
            self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        if torchrun_backend() is not None:
            self._generate_torchrun(work_dir, full_cores, shared)
            return
        try:
            runner.run_stage(
                _HF_MODEL_ID,
                "text",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            runner.run_stage(
                _HF_MODEL_ID,
                "generate",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            runner.run_stage(
                _HF_MODEL_ID,
                "vae",
                num_cores=1,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _generate_torchrun(self, work_dir: Path, full_cores: int, shared: list[str]) -> None:
        """``generate`` on a torchrun backend (neuron): the generate stage as ``full_cores``
        ranks (prompt encode on rank 0's host, denoise, rank 0 writes latents.pt), then the host
        decode here in the driver, after torchrun has exited (so after every rank has)."""
        from difflet.pipeline.path_resolver import resolve_model_path

        try:
            runner.run_stage(
                _HF_MODEL_ID,
                "generate",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            _decode_latents_host(
                str(work_dir / "latents.pt"),
                self._model_dir(self.args, resolve_model_path),
                self.args.output,
                height=self.args.height or 1024,
                width=self.args.width or 1024,
            )
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if stage == "text":
            self._stage_text(args)
        elif stage == "generate":
            self._stage_generate(args)
        elif stage == "vae":
            self._stage_vae(args)
        else:
            raise ValueError(f"Unknown QwenImage stage: {stage!r}")

    # ---------------------------------------------------------------- stages

    @staticmethod
    def _require_aot_stage(stage: str) -> None:
        """The text and vae stages exist on the AoT backends only: on a torchrun backend the
        generate stage encodes the prompt and the driver decodes on the host."""
        backend = torchrun_backend()
        if backend is not None:
            raise RuntimeError(
                f"[qwen_image] the {backend} backend has no {stage!r} stage: the generate stage "
                "encodes the prompt (rank 0's host) and the driver decodes on the host"
            )

    def _stage_text(self, args: argparse.Namespace) -> None:
        self._require_aot_stage("text")
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
            NeuronQwen2VLTextForCausalLM,
        )
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        enc_path = str(Path(model_dir) / "text_encoder")
        compiled_dir = self._stage_compiled_dir("text", args)

        tp_degree = args.tp_degree or 4
        text_cfg = AutoConfig.from_pretrained(enc_path).text_config
        if getattr(text_cfg, "pad_token_id", None) is None:
            text_cfg.pad_token_id = 0

        # This stage uses NxDI's stock NeuronConfig (not difflet's fork), whose
        # save_sharded_checkpoint default is False (see the identical fix in
        # hunyuan_video.py's _stage_llama). Stock load_weights() has no
        # missing-shard fallback, so only request the presharded read path
        # once the shard files actually exist; always request the write path
        # at compile time.
        weights_dir = compiled_dir / "weights"
        shard_paths = [
            weights_dir / f"tp{r}_sharded_checkpoint.safetensors" for r in range(tp_degree)
        ]
        save_sharded_checkpoint = args.stage_mode == "compile" or all(
            p.exists() for p in shard_paths
        )

        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            batch_size=1,
            seq_len=_ENC_SEQ,
            torch_dtype=torch.bfloat16,
            on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=["norm"]),
            save_sharded_checkpoint=save_sharded_checkpoint,
        )
        config = NeuronQwen2VLTextForCausalLM.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(hf_config=text_cfg)
        )
        app = NeuronQwen2VLTextForCausalLM(enc_path, config)
        if args.stage_mode == "compile":
            if not self._stage_is_compiled("text", args, compiled_dir):
                app.compile(str(compiled_dir))
                self._finish_stage_compile("text", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("text", args, compiled_dir)
        app.load(str(compiled_dir))
        tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                ti = tok(
                    _QWEN_TEMPLATE.format(req.prompt),
                    max_length=_ENC_SEQ,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids = ti.input_ids.to(torch.int32)
                attn = ti.attention_mask.to(torch.int32)
                out = app(
                    input_ids=input_ids,
                    attention_mask=attn,
                    position_ids=torch.arange(_ENC_SEQ, dtype=torch.int32).unsqueeze(0),
                    sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                )
                hs = out.captured_tensors[0].float()
                valid = int(attn.sum())
                dev = hs[:, _QWEN_DROP_IDX:valid]
                seq = dev.shape[1]
                ehs = torch.zeros(1, _TEXT_SEQ_LEN, dev.shape[-1], dtype=torch.bfloat16)
                ehs[:, :seq] = dev.to(torch.bfloat16)
                mask = torch.zeros(1, _TEXT_SEQ_LEN, dtype=torch.bool)
                mask[:, :seq] = True
                dest = stage_loop.work_file(args, req, "text.pt")
                torch.save({"encoder_hidden_states": ehs, "encoder_hidden_states_mask": mask}, dest)
                print(f"[text] encoder_hidden_states {tuple(ehs.shape)} -> {dest}")

    def _stage_generate(self, args: argparse.Namespace) -> None:
        backend = torchrun_backend()
        if backend is not None:
            self._stage_generate_neuron(args, backend)
            return

        import numpy as np
        import torch
        from difflet.models.qwen_image.application import NeuronQwenImageApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        compiled_dir = self._stage_compiled_dir("generate", args)

        h, w = args.height or 1024, args.width or 1024

        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            sp_enabled=bool(getattr(args, "sp_enabled", None)),
        )
        compile_shapes = _require_request_shape_in_set(args)
        app = NeuronQwenImageApplication(
            model_path=model_dir,
            parallel=parallel,
            dtype=torch.bfloat16,
            shape={"height": h, "width": w, "num_frames": None},
            shapes=compile_shapes,
            text_seq_len=_TEXT_SEQ_LEN,
            enable_transformer=True,
            teacache_fused=getattr(args, "teacache_speedup", None) is not None,
            teacache_speedup=getattr(args, "teacache_speedup", None),
            teacache_calibration_path=getattr(args, "teacache_calibration", None),
            teacache_cadence=getattr(args, "teacache_cadence", None),
            teacache_online_delta_alpha=getattr(args, "teacache_online_delta", None),
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            self._finish_stage_compile("generate", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("generate", args, compiled_dir)
        adaptive = getattr(args, "teacache_speedup", None) is not None
        if adaptive and not app.has_compiled_artifacts(str(compiled_dir), select=["teacache_probe"]):
            # Same additive-probe rule as hunyuan_video: the fused probe is a
            # component of the DiT stage artifact (identity unchanged; the DiT
            # NEFF is reused). `difflet compile --teacache-speedup` builds it;
            # a generate against an artifact compiled without it builds it here.
            print(
                f"[qwen_image] compiling the TeaCache probe NEFF into {compiled_dir} "
                "(first adaptive run; the DiT artifact is reused)",
                flush=True,
            )
            app.compile(str(compiled_dir), select=["teacache_probe"])
        app.load(str(compiled_dir), skip_warmup=True)
        sched = app.pipeline.scheduler
        sc = sched.config
        image_seq_len = (h // 16) * (w // 16)
        slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
        mu = image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                text = torch.load(stage_loop.work_file(args, req, "text.pt"))
                num_steps = int(stage_loop.effective(req, args, "steps", 4))
                sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
                sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
                guidance = torch.full(
                    [1],
                    float(stage_loop.effective(req, args, "guidance_scale", 4.0)),
                    dtype=torch.bfloat16,
                )
                torch.manual_seed(req.seed)
                out = app.pipeline(
                    encoder_hidden_states=text["encoder_hidden_states"],
                    encoder_hidden_states_mask=text["encoder_hidden_states_mask"],
                    guidance=guidance,
                    timesteps=sched.timesteps,
                    num_inference_steps=num_steps,
                    output_type="latent",
                )
                packed = out.latents.cpu()
                dest = stage_loop.work_file(args, req, "latents.pt")
                torch.save(packed, dest)
                print(f"[generate] packed latents {tuple(packed.shape)} -> {dest}")

    def _stage_generate_neuron(self, args: argparse.Namespace, backend: str) -> None:
        """The generate stage on a non-AoT torchrun backend (neuron); runs on every rank.

        ``DiffletPipeline.from_pretrained`` prepares the runtime (NEFF cache, core binding, the
        process group) before it builds the registry's application
        (TorchNeuronQwenImageApplication), and ``load(skip_warmup=False)`` compiles or loads the
        DiT blocks at the static shape and reads Qwen2.5-VL on rank 0, so the step loop compiles
        nothing. Nothing is built ahead of time: ``compile`` mode is that load plus the stage
        manifest; ``generate`` requires no artifact. Per request, the application encodes the
        prompt on rank 0's host (the Trainium text stage's template, drop and zero-pad to
        TEXT_SEQ_LEN; the DiT attends the padded text unmasked, as on Trainium), draws the noise
        from the seeded generator on rank 0 and broadcasts it, and sets the shifted FlowMatch
        schedule as ``_stage_generate`` does. Every rank runs the same generate and rank 0 alone
        writes files: ``work_file`` and the compiled dir are the same path on every rank. The
        process group is destroyed after a successful stage only: a failing rank exits with its
        exception and torchrun stops its peers, which may be blocked in a collective.
        """
        import time

        import torch
        import torch.distributed as dist

        from difflet.backends.neuron.compile import resolve_exec_mode
        from difflet.backends.neuron.core import distributed
        from difflet.cli.dp import stage_loop
        from difflet.pipeline.difflet_pipeline import DiffletPipeline
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        if stage_loop.batch_mode(args):
            raise NotImplementedError(
                f"[qwen_image] the {backend} backend runs one request per launch (phase 1): "
                "every rank would claim a different request; --requests and data parallelism "
                "are not supported"
            )
        compile_shapes = _require_request_shape_in_set(args)
        h, w = args.height or 1024, args.width or 1024
        # The whole parallel request: the backend and the application reject what they do not
        # run (cfg, sp) rather than the stage dropping it.
        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
            sp_enabled=bool(getattr(args, "sp_enabled", None)),
        )
        # --model-path (a local snapshot) binds the pipeline to it; the compiled path is the
        # stage dir, which the neuron application never reads.
        bound = {}
        if getattr(args, "model_path", None):
            bound = {
                "model_path_override": self._model_dir(args, None),
                "resolved_source_id": args.revision or "local",
                "compiled_path_override": str(self._stage_compiled_dir("generate", args)),
            }
        start = time.perf_counter()
        pipe = DiffletPipeline.from_pretrained(
            _HF_MODEL_ID,
            revision=args.revision,
            local_files_only=True,
            parallel=parallel,
            dtype=torch.bfloat16,
            height=h,
            width=w,
            shapes=compile_shapes,
            compile_cache_dir=args.cache_dir,
            backend=backend,
            application_kwargs=_torchrun_application_kwargs(args),
            load=False,
            **bound,
        )
        with fallback_tracker(pipe) as load_fallbacks:
            pipe.load(skip_warmup=False)
        load_s = time.perf_counter() - start

        generate_fallbacks: list = []
        if args.stage_mode == "compile":
            if distributed.is_rank0():
                self._finish_stage_compile(
                    "generate", args, self._stage_compiled_dir("generate", args)
                )
        else:
            for req in stage_loop.claim_requests(args):
                with stage_loop.request_scope(args, req, final=False):
                    with fallback_tracker(pipe) as fallbacks:
                        out = pipe(
                            # The noise the Trainium stage draws after torch.manual_seed(seed).
                            generator=torch.Generator().manual_seed(req.seed),
                            prompt=req.prompt,
                            num_inference_steps=int(stage_loop.effective(req, args, "steps", 4)),
                            # A bf16 [1] tensor of it, as the Trainium stage passes (the DiT
                            # drops it: guidance_embeds is off).
                            guidance_scale=float(
                                stage_loop.effective(req, args, "guidance_scale", 4.0)
                            ),
                            output_type="latent",
                        )
                    generate_fallbacks.extend(fallbacks)
                    if distributed.is_rank0():
                        packed = out.latents.cpu()
                        dest = stage_loop.work_file(args, req, "latents.pt")
                        torch.save(packed, dest)
                        print(f"[generate] packed latents {tuple(packed.shape)} -> {dest}",
                              flush=True)

        app = pipe.app
        rank, world = distributed.world_info()
        print(
            f"[qwen_image] {backend} rank {rank}/{world}: "
            f"exec_mode={resolve_exec_mode(envs.DIFFLET_EXEC_MODE)} "
            f"compiled_blocks={len(getattr(app, 'compiled_blocks', []))} "
            f"unwarmed_shapes={list(getattr(app, 'unwarmed_shapes', []))} "
            f"fallbacks: load={list(load_fallbacks)} generate={generate_fallbacks} "
            f"load+warmup {load_s:.1f}s, stage {time.perf_counter() - start:.1f}s",
            flush=True,
        )
        if dist.is_initialized():
            dist.destroy_process_group()

    def _stage_vae(self, args: argparse.Namespace) -> None:
        self._require_aot_stage("vae")
        import torch
        from difflet.backends.trainium.core.config import NeuronConfig
        from difflet.backends.trainium.wan.vae import (
            NeuronWanVAEDecoderApplication,
            WanVAEDecoderInferenceConfig,
        )
        from difflet.utils.diffusers_adapter import load_diffusers_config
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        vae_path = str(Path(model_dir) / "vae")
        compiled_dir = Path(
            getattr(args, "compiled_dir", None) or self._stage_compiled_dir("vae", args)
        )
        h, w = args.height or 1024, args.width or 1024
        vae_tp_degree = getattr(args, "vae_tp_degree", None) or 1

        # Image bucket set -> single-frame video shapes for the reused Wan VAE
        # decoder config ((h, w) entries become (h, w, 1)).
        compile_shapes = _require_request_shape_in_set(args)
        extra = {}
        if compile_shapes:
            extra["compile_shapes"] = tuple(
                (shape[0], shape[1], 1) for shape in compile_shapes
            )
        config = WanVAEDecoderInferenceConfig(
            neuron_config=NeuronConfig(
                tp_degree=vae_tp_degree,
                world_size=vae_tp_degree,
                torch_dtype=torch.bfloat16,
            ),
            load_config=load_diffusers_config(vae_path),
            height=h,
            width=w,
            num_frames=1,
            **extra,
        )
        app = NeuronWanVAEDecoderApplication(model_path=vae_path, config=config)
        if args.stage_mode == "compile":
            if not self._stage_is_compiled("vae", args, compiled_dir):
                app.compile(str(compiled_dir))
                self._finish_stage_compile("vae", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("vae", args, compiled_dir)
        app.load(str(compiled_dir))
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                packed = torch.load(stage_loop.work_file(args, req, "latents.pt")).float()
                z = _unpack_latents(packed, args.height or 1024, args.width or 1024)
                mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
                std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
                z = (
                    (z * std + mean).to(torch.bfloat16)
                    if len(config.latents_mean)
                    else z.to(torch.bfloat16)
                )
                img = app(z)
                img = (img[0] if isinstance(img, (tuple, list)) else img).float().cpu()
                _write_image(img[:, :, 0], req.output)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _model_dir(args: argparse.Namespace, resolver) -> str:
        model_path = getattr(args, "model_path", None)
        if model_path:
            return str(Path(model_path).expanduser().resolve())
        return resolver(
            _HF_MODEL_ID,
            revision=args.revision,
            local_files_only=True,
        )

    def _stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        inputs = self._backend_neutral_stage_cache_inputs(stage, args)
        backend = torchrun_backend()
        if backend is None:
            return inputs
        from difflet.backends.neuron.compile import resolve_exec_mode

        # Additive, torchrun backends only: their stage dirs never match a Trainium/TPU
        # artifact, and eager vs compile differ (an omitted --exec-mode keys as the default).
        return {
            **inputs,
            "backend": backend,
            "exec_mode": resolve_exec_mode(envs.DIFFLET_EXEC_MODE),
        }

    def _backend_neutral_stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        if stage == "text":
            return {
                "component": "qwen_image_enc",
                "model_id": _HF_MODEL_ID,
                "tp": args.tp_degree or 4,
                "cp": args.cp_degree or 1,
                "seq_len": _ENC_SEQ,
                "dtype": "bfloat16",
                "toolchain": stage_toolchain_versions(),
            }
        if stage == "generate":
            return {
                "component": "qwen_image_dit",
                "model_id": _HF_MODEL_ID,
                "tp": args.tp_degree or 4,
                "cp": args.cp_degree or 1,
                "cp_mode": str(getattr(args, "cp_mode", "gather_kv") or "gather_kv"),
                # sp changes the DiT graph (the modeling_qwen fork); without
                # this field tp4sp and tp4 hash to the same dir and silently
                # reuse each other's artifact (the 9a00746 naming bug, now on
                # the hashed system). Same encoding wan/hunyuan use.
                "sp": bool(getattr(args, "sp_enabled", False)),
                "dtype": "bfloat16",
                "text_seq_len": _TEXT_SEQ_LEN,
                "shapes": canonical_shapes_list(args, (1024, 1024)),
                "toolchain": stage_toolchain_versions(),
            }
        if stage == "vae":
            return {
                "component": "qwen_image_vae",
                "model_id": _HF_MODEL_ID,
                "vae_tp": getattr(args, "vae_tp_degree", None) or 1,
                "dtype": "bfloat16",
                "shapes": canonical_shapes_list(args, (1024, 1024)),
                "toolchain": stage_toolchain_versions(),
            }
        raise ValueError(f"unknown Qwen stage {stage!r}")

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        compiled_dir = getattr(args, "compiled_dir", None)
        if compiled_dir:
            return Path(compiled_dir)
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        inputs = self._stage_cache_inputs(stage, args)
        return hashed_stage_dir(base, str(inputs["component"]), inputs)

    def _finish_stage_compile(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        write_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args))

    def _require_stage_artifact(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        if getattr(args, "compiled_dir", None):
            return  # explicit dir override bypasses manifest gating
        if not has_valid_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args)):
            raise SystemExit(
                f"[qwen_image] no valid compiled artifact for stage {stage!r} at "
                f"{compiled_dir} (manifest missing or configuration changed); run "
                "`difflet compile` with the same flags first."
            )

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id",
            _HF_MODEL_ID,
            "--tp-degree",
            str(a.tp_degree or 4),
            "--cp-degree",
            str(a.cp_degree or 1),
            "--cp-mode",
            str(getattr(a, "cp_mode", "gather_kv")),
            "--height",
            str(a.height or 1024),
            "--width",
            str(a.width or 1024),
            "--steps",
            str(getattr(a, "steps", None) or 4),
            "--guidance-scale",
            str(getattr(a, "guidance_scale", None) or 4.0),
            "--seed",
            str(getattr(a, "seed", 42)),
            "--stage-mode",
            stage_mode,
        ]
        if getattr(a, "shapes", None):
            parts += ["--shapes", str(a.shapes)]
        if getattr(a, "prompt", None):
            parts += ["--prompt", a.prompt]
        if getattr(a, "output", None):
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if getattr(a, "revision", None):
            parts += ["--revision", a.revision]
        if getattr(a, "sp_enabled", None):
            parts += ["--sp"]
        if getattr(a, "teacache_speedup", None) is not None:
            parts += ["--teacache-speedup", str(a.teacache_speedup)]
        if getattr(a, "teacache_calibration", None):
            parts += ["--teacache-calibration", a.teacache_calibration]
        if getattr(a, "teacache_cadence", None) is not None:
            parts += ["--teacache-cadence", str(a.teacache_cadence)]
        if getattr(a, "teacache_online_delta", None) is not None:
            parts += ["--teacache-online-delta", str(a.teacache_online_delta)]
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += [
                "--requests-dir",
                str(a.requests_dir),
                "--worker-index",
                str(a.worker_index),
                "--dp-schedule",
                str(getattr(a, "dp_schedule", "round_robin")),
            ]
        return parts
