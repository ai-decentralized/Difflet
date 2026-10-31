"""Qwen-Image T2I orchestrator — 3-stage subprocess pipeline.

Stage 1 (text):     Qwen2.5-VL text encoder,  tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 2 (generate): DiT backbone,              tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 3 (vae):      VAE decoder,               1 core,      VIRTUAL_CORE_SIZE=2.
Inter-stage tensors: {work_dir}/text.pt, {work_dir}/latents.pt.

Stage logic migrated from examples/qwen_image_example.py.

With DIFFLET_BACKEND naming a non-AoT torchrun backend (neuron), the plan is the generate launch
and then a decode: the generate stage runs as tp*cp torchrun ranks (default tp 4, every core of
a trn2.3xlarge) that build the registry's application (TorchNeuronQwenImageApplication) through
DiffletPipeline (``_stage_generate_neuron``). The text stage is folded into it: rank 0 encodes
the prompt with Qwen2.5-VL on its host and broadcasts the embeddings (or, with
--device-text-encoder, every rank encodes it with the language model TP-sharded on the device
beside the DiT; the driver keeps the host encoder when that cannot fit: ``_place_text_encoder``),
the ranks denoise, and rank 0 writes latents.pt in the packed format the vae stage reads. The decode then runs after that
launch has exited, in one of two places (``_decodes_on_device``):

* by default (or with --device-vae), in a vae stage of its own on one NeuronCore
  (``_stage_vae_neuron``): the Wan decoder (Qwen-Image's VAE is Wan 2.1's), eager, in bf16 by
  default (``_DEFAULT_VAE_DTYPE``; norms, attention scores and the output head stay fp32; 56.7
  dB from the fp32 decode on the device, phase 3 of the 2026-10-26 native-vs-AoT plan), or in
  fp32 with --vae-dtype fp32, which adds ``vae_dtype`` to the vae stage key; --vae-exec-mode
  compile (phase 7's opt-in, which adds ``vae_exec_mode``) compiles the decoder region by
  region instead of running it eager;
* with --host-vae, on the host: the driver decodes with diffusers ``AutoencoderKLQwenImage``
  in fp32 (``_decode_latents_host``), about 23.6 s per 1024x1024 image on the trn2.3xlarge host.

The decoder runs in a launch of its own rather than beside the DiT on rank 0's core, as Wan
does, because of the HBM arithmetic: at tp4 a core already holds about 10.1 GiB of DiT shard
(measured, docs/verification/qwen-image-neuron-evidence.md) plus its activations and NEFFs,
and the fp32 decoder adds 0.5 GB of weights and several GiB of full-resolution activations
(a 96-channel 1024x1024 fp32 feature map alone is 0.4 GiB, and the up blocks keep a few of them
and their convolution caches live). That leaves too little margin at tp4, and at tp2 (CP2,
CFG2 or dp2tp2, 19.75 GiB of DiT per core) it cannot fit at all. After the generate launch has
exited, the vae launch has a whole core to itself whatever the topology was. Both decodes share
the unpack and the un-normalisation (``_vae_latents``), so they see the same fp32 latent.

``compile`` is the generate stage's load + warm-up (it fills the NEFF cache) and its stage
manifest, plus the vae stage's warm-up decode with --device-vae; the stage cache key names the
backend and exec mode; --cache-dir becomes DIFFLET_COMPILE_CACHE unless that is exported; the
options the ranks reject (TeaCache, several --shapes), the pairs they cannot run together
(--sp or --cfg-parallel with --cp-degree > 1, --cfg-parallel without --true-cfg-scale > 1) and a
TP degree whose DiT shard cannot fit one core (``_check_dit_fits``) are refused by the driver
before torchrun starts. The ranks run TP, SP (--sp), CP and, with opt-in true CFG
(--true-cfg-scale > 1, --negative-prompt: diffusers' two forwards per step), CFG parallelism
(--cfg-parallel: the two branches on two TP groups, twice the cores). DP replicas (--dp, --requests) are separate torchrun launches, as Wan's: each DP
worker runs this driver, its generate launch drains the request queue (rank 0 claims:
``replica_requests``), and its decode, host or vae stage, writes and marks done each request
the replica finished. Unset, trainium and tpu keep their three stages unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from difflet.common.orchestrators import qwen_image as qwen_common
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
    cfg_cp_conflict,
    component_dtype,
    decodes_on_device,
    device_encoder_overflow,
    refuse,
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
    vae_dtype_conflict,
    vae_exec_mode_conflict,
    vae_exec_mode_key,
    vae_placement_conflict,
)
from difflet.cli import runner

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_CLI_NAME = qwen_common.CLI_NAME
_VIRTUAL_CORE_SIZE = qwen_common.VIRTUAL_CORE_SIZE
_ENC_SEQ = qwen_common.ENC_SEQ
_TEXT_SEQ_LEN = qwen_common.TEXT_SEQ_LEN
#: The neuron vae stage's decoder precision when --vae-dtype is not given (a --vae-dtype value).
_DEFAULT_VAE_DTYPE = "bf16"

_QWEN_TEMPLATE = qwen_common.QWEN_TEMPLATE
_QWEN_DROP_IDX = qwen_common.QWEN_DROP_IDX


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(1024, 1024), model_tag="qwen_image"
    )


# TorchNeuronQwenImageApplication keyword arguments for the torchrun generate stage: the
# Trainium generate stage's set, Qwen2.5-VL on rank 0's host (in its default bf16, the Trainium
# text stage's dtype) with 12 threads, no VAE in the generate launch (the decode runs after it,
# on the host or in the vae stage: _decodes_on_device) and exec_mode=None (the application
# reads DIFFLET_EXEC_MODE, which --exec-mode exports).
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
_TORCHRUN_RUNS = (
    "tensor, sequence, context and CFG parallelism at one static shape, and DP replicas, "
    "no TeaCache"
)


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options the torchrun (neuron) generate stage rejects in phase 1, as the command line
    names them: every torchrun backend's (TeaCache), plus a --shapes with more than one shape
    (the application holds one static shape per process). Request batches and DP replicas run,
    as Wan's: each replica's rank 0 claims the requests (``replica_requests``); --cfg-parallel
    runs with true CFG (``_torchrun_conflicts``). Torch-free: the shapes are counted, not
    parsed, so a malformed --shapes fails where it always did."""
    return torchrun_unsupported_options(args)


def _true_cfg_scale(args: argparse.Namespace) -> float | None:
    """--true-cfg-scale when it turns true CFG on (above 1), else None."""
    scale = getattr(args, "true_cfg_scale", None)
    return float(scale) if scale is not None and float(scale) > 1.0 else None


def _torchrun_conflicts(args: argparse.Namespace, *, generating: bool) -> list[str]:
    """Option pairs the torchrun (neuron) path cannot honour together, one message each:
    --host-vae with --device-vae; --sp or --cfg-parallel with --cp-degree > 1 (SP and CP both
    shard the image sequence, CFG and CP both take the data-parallel lanes;
    ``NeuronQwenImageConfig`` refuses either pair in every rank, as Wan's config does); and,
    for a ``generating`` command, --cfg-parallel without --true-cfg-scale > 1 (the CFG groups
    split true CFG's two branches; ``compile`` only warms the batch-2 graph up). Torch-free."""
    sp_cp = None
    if getattr(args, "sp_enabled", False) and (getattr(args, "cp_degree", None) or 1) > 1:
        sp_cp = ("--sp and --cp-degree > 1 are mutually exclusive for Qwen-Image (both shard "
                 "the image sequence)")
    cfg_without_true_cfg = None
    if getattr(args, "cfg_parallel", False) and generating and _true_cfg_scale(args) is None:
        cfg_without_true_cfg = ("--cfg-parallel splits the two branches of true classifier-free "
                                "guidance; Qwen-Image runs them only with --true-cfg-scale > 1")
    found = (vae_placement_conflict(args), vae_dtype_conflict(args), vae_exec_mode_conflict(args),
             sp_cp, cfg_cp_conflict(args), cfg_without_true_cfg)
    return [conflict for conflict in found if conflict is not None]


def _true_cfg_misuse(args: argparse.Namespace) -> str | None:
    """--true-cfg-scale or --negative-prompt off the neuron backend, else None. Torch-free:
    the Trainium and TPU stages run one forward per step and would ignore them."""
    flags = [flag for flag, name in (("--true-cfg-scale", "true_cfg_scale"),
                                     ("--negative-prompt", "negative_prompt"))
             if getattr(args, name, None) is not None]
    if not flags or torchrun_backend() is not None:
        return None
    return (f"{' and '.join(flags)} run on the neuron backend only (pass --backend neuron): "
            "the trainium and tpu Qwen-Image paths run one forward per step")


def _true_cfg_kwargs(args: argparse.Namespace, req) -> dict:
    """The application's true-CFG arguments for one request: none by default (one forward per
    step, the call unchanged), or ``true_cfg_scale`` and the request's negative prompt (its
    manifest line's, else --negative-prompt, else diffusers' empty negative prompt ``" "``)
    when --true-cfg-scale > 1."""
    from difflet.cli.dp import stage_loop

    scale = _true_cfg_scale(args)
    if scale is None:
        return {}
    negative = stage_loop.effective(req, args, "negative_prompt", None)
    return {"true_cfg_scale": scale, "negative_prompt": " " if negative is None else negative}


def _torchrun_application_kwargs(args: argparse.Namespace) -> dict:
    """The application kwargs, plus any TeaCache flag the user gave: the neuron application
    rejects TeaCache (phase 1) instead of the stage dropping the flag silently.
    --device-text-encoder puts Qwen2.5-VL's language model on the device (the driver has already
    dropped the flag when it cannot fit: ``QwenImageOrchestrator._place_text_encoder``)."""
    kwargs = dict(_TORCHRUN_APPLICATION_KWARGS)
    if getattr(args, "device_text_encoder", False):
        kwargs["text_encoder_device"] = "neuron"
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


#: Where the neuron backend decodes when neither --host-vae nor --device-vae is given: the vae
#: stage on one core. Measured on a trn2.3xlarge (TP4, 1024x1024, 50 steps, warm, alternating
#: runs): the vae stage takes 19.2 s against the host decode's 23.6 s, the process wall 79.4 s
#: against 83.2 s, the image within one level of the host's (fp32 cosine 1 - 3e-13).
#: --host-vae is the opt-out.
_DEVICE_VAE_DEFAULT = True


def _decodes_on_device(args: argparse.Namespace) -> bool:
    """Whether the neuron backend decodes in a vae stage on one core (``_stage_vae_neuron``)
    instead of on the host in the driver (``_decode_latents_host``). --host-vae wins, then
    --device-vae, then ``_DEVICE_VAE_DEFAULT``. Torch-free."""
    return decodes_on_device(args, _DEVICE_VAE_DEFAULT)


def _vae_latents(packed, height: int, width: int, latents_mean, latents_std):
    """The VAE's fp32 input for the generate stage's packed latents: unpacked to a one-frame
    latent video (``_unpack_latents``) and un-normalised, ``z * latents_std + latents_mean`` (the
    Trainium vae stage's, diffusers' ``latents / (1 / std) + mean`` up to rounding, and the
    orchestrator's ``_decode_latents``). Both neuron decodes, host and device, start from this
    tensor, so they decode bit-identical inputs. Without a mean/std the latent is unchanged."""
    import torch

    z = _unpack_latents(packed.to(torch.float32), height, width)
    if latents_mean and latents_std:
        mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        z = z * std + mean
    return z


def _decode_latents_host(
    latents_path: str, model_dir: str, output_path: str, *, height: int, width: int
) -> None:
    """Decode the generate stage's packed latents with diffusers' ``AutoencoderKLQwenImage`` on
    the host CPU in fp32: the neuron backend's default VAE path. The driver runs it after the
    torchrun generate stage has exited, so the decoder never shares a core (or a process) with
    the DiT.

    The decode of the Trainium vae stage and of diffusers' ``QwenImagePipeline``: unpack to a
    one-frame latent video, un-normalise (``_vae_latents``), decode, keep frame 0. fp32 rather
    than the Trainium stage's bf16: the host has the memory for it. The image goes through PIL
    (``_save_image_pil``: the pixels torchvision would write).
    """
    import torch
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        str(Path(model_dir) / "vae"), torch_dtype=torch.float32
    ).eval()
    packed = torch.load(latents_path, map_location="cpu")
    z = _vae_latents(packed, height, width, vae.config.latents_mean, vae.config.latents_std)
    with torch.no_grad():
        frames = vae.decode(z, return_dict=False)[0]
    _write_image(frames[:, :, 0].float().cpu(), output_path, save=_save_image_pil)


class QwenImageOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path

        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def _reject_unsupported_torchrun_options(self, *, generating: bool = True) -> None:
        """A torchrun backend (neuron) asked for an option its ranks reject in phase 1, or a
        pair they cannot run together: print one message naming it and exit 2 before anything
        is launched (or downloaded, for ``run``). The ranks' own checks stay (defence in
        depth). Off the neuron backend, the true-CFG flags are refused the same way."""
        reject_unsupported_torchrun_options(
            _torchrun_unsupported_options(self.args), runs=_TORCHRUN_RUNS
        )
        refuse_torchrun_conflicts(_torchrun_conflicts(self.args, generating=generating))
        refuse(_true_cfg_misuse(self.args))

    def _replica_cores(self) -> int:
        """Cores of one generate launch: tp x cp, doubled by --cfg-parallel (each CFG branch
        on its own TP group)."""
        return replica_cores(self.args)

    def _check_dit_fits(self) -> None:
        """On a torchrun backend (neuron), refuse a TP degree whose DiT shard clearly cannot fit
        one core (``difflet/backends/neuron/qwen_image/hbm.py``) before any rank starts: print
        one message and exit 2.

        The application base refuses it too, but only once torchrun has started every rank and
        each has built the model on meta; a DP router would start every replica for it. The
        per-rank share depends on the TP degree alone, so a DP replica (``--dp 4 --tp-degree
        1``: 38 GiB per core) is caught the same way as a plain launch, while dp2tp2 (about
        19 GiB per core) passes. Weights that are not local yet (``compile`` before a download)
        leave the decision to the application's check. Torch-free, like every driver check.
        """
        if torchrun_backend() is None:
            return
        from difflet.backends.neuron.qwen_image.hbm import dit_fits
        from difflet.pipeline.path_resolver import resolve_model_path

        refuse_dit_overflow("Qwen-Image", lambda: self._model_dir(self.args, resolve_model_path),
                            dit_fits, self.args.tp_degree or 4)

    def _place_text_encoder(self) -> None:
        """On a torchrun backend (neuron) with --device-text-encoder, keep Qwen2.5-VL on rank
        0's host when its language model cannot fit on the device beside the DiT at this TP
        degree, and print one line saying so (Wan's ``_place_text_encoder``, in reverse: Qwen
        defaults to the host and the device is the opt-in).

        At tp4 the encoder adds about 3.3 GiB of bf16 weights per core beside 9.5 GiB of DiT;
        a narrower TP group (--cp-degree 2 with --tp-degree 2, DP replicas of tp2, a plain tp2)
        doubles both, about 6.6 + 19.0 GiB, over a 24 GiB core. The estimate
        (``difflet/backends/neuron/qwen_image/hbm.py``) depends on the stage's tp and the local
        checkpoint's sizes. Weights that are not local yet leave the flag alone, and the
        application's own check still refuses. The setting lands on ``self.args``, so
        ``_shared_cli_args`` hands it to the stages and a ``run`` that dispatches DP workers
        after ``compile`` forwards it to them (``worker_cli_args``). Torch-free."""
        args = self.args
        if torchrun_backend() is None or not getattr(args, "device_text_encoder", False):
            return
        from difflet.backends.neuron.qwen_image.hbm import device_text_encoder_fits
        from difflet.pipeline.path_resolver import resolve_model_path

        overflow = device_encoder_overflow(lambda: self._model_dir(args, resolve_model_path),
                                           device_text_encoder_fits, args.tp_degree or 4)
        if overflow is None:
            return
        args.device_text_encoder = False
        print(f"[qwen_image] Qwen2.5-VL on the device {overflow}: encoding on rank 0's host "
              "instead (without --device-text-encoder)", flush=True)

    def run(self) -> None:
        self._reject_unsupported_torchrun_options()  # before the download, too
        super().run()

    def compile(self) -> None:
        self._reject_unsupported_torchrun_options(generating=False)
        self._check_dit_fits()
        self._place_text_encoder()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="qwen_image")
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = self._replica_cores()
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
        if torchrun_backend() is not None:
            # The generate stage's load + warm-up (the text encoder is part of it) fills the
            # NEFF cache and writes the manifest; the host VAE compiles nothing, the device
            # VAE's warm-up decode fills the eager kernel cache at the request shape.
            runner.run_stage(
                _HF_MODEL_ID,
                "generate",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            if _decodes_on_device(self.args):
                runner.run_stage(
                    _HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
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
        self._check_dit_fits()
        self._place_text_encoder()
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="qwen_image")
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(
            self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = self._replica_cores()
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
        ranks (prompt encode on rank 0's host, denoise, rank 0 writes latents.pt), then the
        decode after torchrun has exited (so after every rank has): the vae stage on one core
        with --device-vae, else the host decode here in the driver."""
        from difflet.pipeline.path_resolver import resolve_model_path

        try:
            runner.run_stage(
                _HF_MODEL_ID,
                "generate",
                num_cores=full_cores,
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )
            if _decodes_on_device(self.args):
                runner.run_stage(
                    _HF_MODEL_ID, "vae", num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
                    cli_args=shared,
                )
            else:
                self._decode_on_host(work_dir, self._model_dir(self.args, resolve_model_path))
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _decode_on_host(self, work_dir: Path, model_dir: str) -> None:
        """The host decode of every request the generate launch finished: the one request of a
        plain launch (``work_dir/latents.pt`` to --output), or, in a DP worker, each request
        this worker's replica claimed and did not fail, from its own latents file to its own
        output, marked done after its image is written (the vae stage's ``final=True``). A
        request whose decode fails is marked failed and the next one decodes, as the AoT vae
        stage's ``request_scope`` does."""
        from difflet.cli.dp import stage_loop

        args = argparse.Namespace(**{**vars(self.args), "work_dir": str(work_dir)})
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                _decode_latents_host(
                    str(stage_loop.work_file(args, req, "latents.pt")),
                    model_dir,
                    req.output,
                    height=self.args.height or 1024,
                    width=self.args.width or 1024,
                )

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
        """The text stage exists on the AoT backends only: on a torchrun backend the generate
        stage encodes the prompt (the vae stage is ``_stage_vae_neuron`` there)."""
        backend = torchrun_backend()
        if backend is not None:
            raise RuntimeError(
                f"[qwen_image] the {backend} backend has no {stage!r} stage: the generate stage "
                "encodes the prompt (rank 0's host)"
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
            tp_reduce_dtype=getattr(args, "tp_reduce_dtype", None),
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

        ``run_generate_stage`` builds the registry's application (TorchNeuronQwenImageApplication)
        through ``DiffletPipeline``, loads the DiT blocks with the warm-up and Qwen2.5-VL on rank
        0 (``compile`` mode is that load plus the stage manifest) and runs the replica's
        requests. Qwen-Image's part: --model-path (a local snapshot) binds the pipeline to it;
        per request, the application encodes the prompt on rank 0's host (the Trainium text
        stage's template, drop and zero-pad to TEXT_SEQ_LEN; the DiT masks the padded rows, as
        diffusers does, where Trainium attends them), draws the noise from the seeded generator
        on rank 0 (the noise the Trainium stage draws after ``torch.manual_seed(seed)``), sets
        the shifted FlowMatch schedule as ``_stage_generate`` does, and runs true CFG only with
        --true-cfg-scale > 1 (``_true_cfg_kwargs``); the guidance scale is passed as the
        Trainium stage passes it (the DiT drops it: guidance_embeds is off); rank 0 writes the
        packed latents the vae stage reads.
        """
        import torch

        from difflet.cli.dp import stage_loop

        compile_shapes = _require_request_shape_in_set(args)
        h, w = args.height or 1024, args.width or 1024
        # --model-path (a local snapshot) binds the pipeline to it; the compiled path is the
        # stage dir, which the neuron application never reads.
        bound = {}
        if getattr(args, "model_path", None):
            bound = {
                "model_path_override": self._model_dir(args, None),
                "resolved_source_id": args.revision or "local",
                "compiled_path_override": str(self._stage_compiled_dir("generate", args)),
            }

        def request_kwargs(req) -> dict:
            return dict(
                **_true_cfg_kwargs(args, req),
                generator=torch.Generator().manual_seed(req.seed),
                prompt=req.prompt,
                num_inference_steps=int(stage_loop.effective(req, args, "steps", 4)),
                guidance_scale=float(stage_loop.effective(req, args, "guidance_scale", 4.0)),
                output_type="latent",
            )

        def save(req, out) -> None:
            packed = out.latents.cpu()
            dest = stage_loop.work_file(args, req, "latents.pt")
            torch.save(packed, dest)
            print(f"[generate] packed latents {tuple(packed.shape)} -> {dest}", flush=True)

        run_generate_stage(
            args, backend, tag="qwen_image", model_id=_HF_MODEL_ID,
            pipeline_kwargs=dict(revision=args.revision, height=h, width=w, shapes=compile_shapes,
                                 **bound),
            application_kwargs=_torchrun_application_kwargs(args),
            request_kwargs=request_kwargs,
            save=save,
            finish_compile=lambda: self._finish_stage_compile(
                "generate", args, self._stage_compiled_dir("generate", args)),
        )

    def _stage_vae(self, args: argparse.Namespace) -> None:
        if torchrun_backend() is not None:
            self._stage_vae_neuron(args)
            return
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

    def _stage_vae_neuron(self, args: argparse.Namespace) -> None:
        """The neuron backend's vae stage (``run_vae_stage``; --device-vae, the default): one
        rank, a launch of its own after the generate stage has exited, so the decoder has a
        whole core (see the module docstring). Wan's vae stage with Qwen-Image's one-frame
        latent: the Wan decoder (``NeuronWanVaeDecoderApplication``, which Qwen-Image's VAE maps
        onto key for key) in fp32, eager, through the composed application's adapter
        (``_DeviceQwenVaeDecoder``, which drops the frame axis). The packed latents are unpacked
        and un-normalised by ``_vae_latents``, as the host decode does, and the image is written
        as the host decode writes it (``_write_image`` through PIL)."""
        import json

        import torch

        from difflet.backends.neuron.core import distributed
        from difflet.backends.neuron.wan import vae as vae_mod
        from difflet.cli.dp import stage_loop
        from difflet.models.qwen_image import neuron_application as na
        from difflet.pipeline.path_resolver import resolve_model_path

        vae_dir = Path(self._model_dir(args, resolve_model_path)) / "vae"
        with open(vae_dir / "config.json", encoding="utf-8") as f:
            z_dim = int(json.load(f)["z_dim"])
        h, w = args.height or 1024, args.width or 1024

        dtype = getattr(torch, component_dtype(getattr(args, "vae_dtype", None), _DEFAULT_VAE_DTYPE))
        exec_mode = getattr(args, "vae_exec_mode", None) or "eager"

        def build(parallel):
            return vae_mod.NeuronWanVaeDecoderApplication(
                model_path=str(vae_dir),
                latent_shape=(1, z_dim, 1, 2 * (h // 16), 2 * (w // 16)),
                parallel=parallel,
                dtype=dtype,
                exec_mode=exec_mode,
                device="neuron",
            )

        def prepare_decode(app):
            decoder = na._DeviceQwenVaeDecoder(app, device="neuron")
            config = app.vae_config

            def decode(req) -> None:
                packed = torch.load(stage_loop.work_file(args, req, "latents.pt"),
                                    map_location="cpu")
                z = _vae_latents(packed, h, w, config.latents_mean, config.latents_std)
                with torch.no_grad():
                    images = decoder(z)
                if distributed.is_rank0():
                    _write_image(images.float().cpu(), req.output, save=_save_image_pil)

            return decode

        run_vae_stage(
            args, build=build, prepare_decode=prepare_decode,
            finish_compile=lambda: self._finish_stage_compile(
                "vae", args, self._stage_compiled_dir("vae", args)),
        )

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
        inputs = torchrun_stage_keys(self._backend_neutral_stage_cache_inputs(stage, args))
        # Trainium only: its DiT masks the prompt padding by the valid-row count, where an older
        # DiT's graph attends it (the neuron NEFF cache keys the graph itself).
        if stage == "generate" and torchrun_backend() is None:
            inputs["text_mask"] = "valid-len"
        # Additive, neuron only: a decoder in the non-default precision (--vae-dtype fp32) names
        # its dtype, so the default (bf16) vae stage, given --vae-dtype bf16 or not, keeps one key.
        vae_dtype = getattr(args, "vae_dtype", None)
        if (stage == "vae" and torchrun_backend() is not None
                and vae_dtype not in (None, _DEFAULT_VAE_DTYPE)):
            inputs["vae_dtype"] = component_dtype(vae_dtype)
        # Likewise the decoder compiled by region (--vae-exec-mode compile, phase 7).
        if stage == "vae" and torchrun_backend() is not None:
            inputs.update(vae_exec_mode_key(args))
        return inputs

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
            import torch

            from difflet.common.tp_reduce import parse_tp_reduce_dtype, tp_reduce_cache_fields

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
                # Additive: only a CFG-parallel artifact names it (its warm-up is the batch-2
                # pair), so every existing artifact keeps its identity.
                **({"cfg_parallel": True} if getattr(args, "cfg_parallel", False) else {}),
                "dtype": "bfloat16",
                "text_seq_len": _TEXT_SEQ_LEN,
                "shapes": canonical_shapes_list(args, (1024, 1024)),
                "toolchain": stage_toolchain_versions(),
                # Additive: pre-bf16 DiT artifacts reduced in fp32 and keep their key.
                **tp_reduce_cache_fields(
                    parse_tp_reduce_dtype(getattr(args, "tp_reduce_dtype", None)),
                    legacy=torch.float32,
                ),
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
        if getattr(a, "cfg_parallel", False):
            parts += ["--cfg-parallel"]
        if getattr(a, "true_cfg_scale", None) is not None:
            parts += ["--true-cfg-scale", str(a.true_cfg_scale)]
        if getattr(a, "negative_prompt", None) is not None:
            parts += ["--negative-prompt", a.negative_prompt]
        if getattr(a, "device_text_encoder", False):
            parts += ["--device-text-encoder"]
        if getattr(a, "vae_dtype", None):  # only when given: a default line stays byte-identical
            parts += ["--vae-dtype", a.vae_dtype]
        if getattr(a, "vae_exec_mode", None):
            parts += ["--vae-exec-mode", a.vae_exec_mode]
        if getattr(a, "tp_reduce_dtype", None):
            parts += ["--tp-reduce-dtype", a.tp_reduce_dtype]
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
