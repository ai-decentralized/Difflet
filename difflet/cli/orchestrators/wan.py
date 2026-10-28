"""Wan 2.1 / 2.2 T2V orchestrator — 2-stage subprocess pipeline.

Stage 1 (transformer): text encoder + DiT backbone, tp×cp cores.
Stage 2 (vae):         VAE decoder, 1 core.
Inter-stage tensor: {work_dir}/latents.pt

With DIFFLET_BACKEND naming a non-AoT torchrun backend (neuron), the transformer stage runs as
torchrun ranks that build the registry's application through DiffletPipeline
(_stage_transformer_neuron). Those ranks run TP, SP, CFG and CP at one static shape, and the
DiT sits on every rank's core. umT5 also runs on the device (bf16, TP-sharded) unless
--host-text-encoder is given. When the TP group is too narrow for umT5 to fit beside the DiT
(_place_text_encoder), the driver sets that flag itself and says so. The VAE decodes in a vae
stage of its own on one core (_stage_vae_neuron, eager; bf16 up to 9 frames, fp32 beyond),
or on the host with --host-vae. --text-encoder-dtype and --vae-dtype override umT5's and the
device decoder's precision; a stage key gains a field only for a precision other than its
default (_component_dtype_keys), so a default command line keeps its keys.
DP replicas (--dp) are separate torchrun launches; each one's rank 0 claims the requests
(_replica_requests). The stage cache key names the backend and exec mode, and --cache-dir
becomes DIFFLET_COMPILE_CACHE (the NEFF/HLO cache root) unless that is exported. Unset,
trainium and tpu keep their paths unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import (
    ModelOrchestrator,
    adaptive_teacache_calibration,
    canonical_shapes_list,
    has_valid_stage_manifest,
    hashed_stage_dir,
    require_request_shape_in_set,
    stage_toolchain_versions,
    write_stage_manifest,
)
from difflet.cli.orchestrators.torchrun import (
    component_dtype,
    device_encoder_overflow,
    refuse,
    refuse_torchrun_conflicts,
    reject_unsupported_torchrun_options,
    replica_cores,
    root_compile_cache_at_cache_dir,
    run_generate_stage,
    run_vae_stage,
    torchrun_backend as _torchrun_backend,
    torchrun_stage_keys,
    torchrun_unsupported_options,
    vae_dtype_conflict,
)

# Model id comes from the CLI (--model-id); both Wan 2.2 A14B (MoE, dual
# transformer) and Wan 2.1 14B (single transformer) route here. The single-vs-
# dual difference is handled by the app/pipeline (transformer_2 + boundary_ratio
# are loaded only when present), so the orchestrator is version-agnostic.
_MODEL_TYPE = "wan"
_CLI_NAME = "wan"
_VIRTUAL_CORE_SIZE = None  # Wan does not require NEURON_RT_VIRTUAL_CORE_SIZE


def _transformer_virtual_core_size(args) -> int | None:
    """Ring CP is the one Wan path that needs NEURON_RT_VIRTUAL_CORE_SIZE=2.

    The nkilib ring kernel allocates per-core shared_hbm send/recv buffers and
    a core_barrier that only exist in its LNC2 SPMD-grid variant, selected in
    ops_impl.attention.ring_attention off this env var; without it neuronx-cc
    fails with NCC_ILLC059 ("Could not find MemoryLocation ... send_k_buf on
    core 1"). Every other Wan stage/mode stays at None so its existing compile
    caches remain valid (ring's transformer cache is a separate key anyway).
    """
    if getattr(args, "cp_mode", None) == "ring" and (args.cp_degree or 1) > 1:
        return 2
    return _VIRTUAL_CORE_SIZE

# The historical model id keeps the bare "wan" compiled-dir prefix so existing
# compile caches stay valid (additive-only, same policy as
# DiffletParallelConfig.to_cache_dict).
_LEGACY_CACHE_PREFIX_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def _teacache_probe_required(args: argparse.Namespace) -> bool:
    """Whether this request's transformer artifact must carry the TeaCache probe
    NEFF: a calibrated-adaptive request (--teacache-speedup) whose calibration is
    adaptive -- the same predicate the application uses to build the probe
    (difflet.pipeline.teacache.requires_teacache_probe)."""
    if getattr(args, "teacache_speedup", None) is None:
        return False
    path = getattr(args, "teacache_calibration", None)
    if not path:
        return False
    from difflet.pipeline.teacache import requires_teacache_probe

    return requires_teacache_probe({"teacache_calibration_path": path})


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


def _write_frames(frames, output: str) -> None:
    """Decoded frames to ``output``: MP4 when it ends in .mp4 and encoding works, else a .pt."""
    import torch

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".mp4" and _save_video(frames.cpu(), str(out_path)):
        return
    pt_path = out_path.with_suffix(".pt")
    torch.save(frames.cpu(), pt_path)
    print(f"[wan] video tensor saved to {pt_path}")


def _torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options the torchrun (neuron) ranks reject in phase 1, as the command line names
    them: every torchrun backend's (TeaCache), plus a --shapes with more than one shape. The
    application holds one static shape per process (``_resolve_shape``), so a bucketed set
    used to fail only after torchrun had started every rank. Torch-free: the shapes are
    counted, not parsed, so a malformed --shapes fails where it always did."""
    return torchrun_unsupported_options(args)


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(480, 832, 9), model_tag="wan"
    )


# TorchNeuronWanApplication keyword arguments for the torchrun transformer stage: the Trainium
# stage's set, exec_mode=None (the application reads DIFFLET_EXEC_MODE, which --exec-mode
# exports), umT5 on the device (bf16, the application's default, TP-sharded; --host-text-encoder
# keeps it on rank 0's host, and the driver sets that flag itself when the TP group is too narrow
# for the encoder to fit beside the DiT: WanOrchestrator._place_text_encoder) and, for the host
# encoder, 12 threads
# (D70: its fp32 output bits depend on the count).
_TORCHRUN_APPLICATION_KWARGS = {
    "text_seq_len": 512,
    "batch_size": 1,
    "enable_text_encoder": True,
    "enable_transformer": True,
    "enable_transformer_2": False,
    "enable_vae_decoder": False,
    "exec_mode": None,
    "encoder_threads": 12,
    "text_encoder_device": "neuron",
}


def _dual_expert_misuse(args: argparse.Namespace, torchrun_backend: str | None) -> str | None:
    """Why --dual-expert / --guidance-scale-2 cannot run as given, else None. Torch-free.

    The trainium and tpu Wan paths hold the high-noise expert alone (expert residency) and run
    it for every step; only the neuron application loads ``transformer_2``. A low-noise
    guidance scale without the low-noise expert would be ignored, so it is refused too.
    """
    dual = getattr(args, "dual_expert", False)
    if getattr(args, "guidance_scale_2", None) is not None and not dual:
        return "--guidance-scale-2 sets the low-noise expert's guidance; it needs --dual-expert"
    if dual and torchrun_backend is None:
        return ("--dual-expert runs on the neuron backend only (pass --backend neuron): the "
                "trainium and tpu Wan paths hold the high-noise expert alone and run it for "
                "every step")
    return None


def _torchrun_application_kwargs(args: argparse.Namespace) -> dict:
    """The application kwargs, plus any TeaCache flag the user gave (the application builds the
    controller and, for a calibrated signal, one probe per expert). --dual-expert
    loads Wan 2.2 A14B's low-noise expert (``transformer_2``). --text-encoder-dtype, only when
    given, sets umT5's precision (``text_encoder_dtype``; the application's default is bf16)."""
    kwargs = dict(_TORCHRUN_APPLICATION_KWARGS)
    if getattr(args, "dual_expert", False):
        kwargs["enable_transformer_2"] = True
    if getattr(args, "host_text_encoder", False):
        kwargs["text_encoder_device"] = "host"
    if getattr(args, "text_encoder_dtype", None):
        kwargs["text_encoder_dtype"] = component_dtype(args.text_encoder_dtype)
    for app_key, arg_key in (
        ("teacache_cadence", "teacache_cadence"),
        ("teacache_online_delta_alpha", "teacache_online_delta"),
        ("teacache_calibration_path", "teacache_calibration"),
    ):
        value = getattr(args, arg_key, None)
        if value is not None:
            kwargs[app_key] = value
    return kwargs


#: ``difflet.backends.neuron.wan.vae.BF16_MAX_FRAMES``, copied so the driver stays torch-free
#: (a unit test pins the two equal): the device decoder runs in bf16 up to this many frames and
#: in fp32 beyond, as the Trainium path decodes.
_VAE_BF16_MAX_FRAMES = 9


def _default_vae_dtype_flag(args: argparse.Namespace) -> str:
    """``default_vae_dtype`` of the request's clip as a --vae-dtype value. Torch-free."""
    return "bf16" if int(args.num_frames or 9) <= _VAE_BF16_MAX_FRAMES else "fp32"


def _vae_dtype_name(args: argparse.Namespace) -> str:
    """The neuron vae stage's decoder dtype as torch names it: --vae-dtype when given, else
    ``default_vae_dtype`` of the clip. Torch-free."""
    return component_dtype(getattr(args, "vae_dtype", None) or _default_vae_dtype_flag(args))


def _component_dtype_keys(stage: str, args: argparse.Namespace) -> dict:
    """The neuron stage key fields of --text-encoder-dtype and --vae-dtype:
    ``text_encoder_dtype`` on the transformer stage (umT5 runs there) and ``vae_dtype`` on the
    vae stage, each only when the flag asks for a precision other than the default (umT5: bf16;
    the decoder: ``default_vae_dtype`` of the clip), so a default stage, given the flag or not,
    keeps the key, and the manifests, it had. Nothing on the AoT backends, whose paths these
    flags do not change. Torch-free."""
    if _torchrun_backend() is None:
        return {}
    if stage == "transformer":
        flag, default = "text_encoder_dtype", "bf16"
    elif stage == "vae":
        flag, default = "vae_dtype", _default_vae_dtype_flag(args)
    else:
        return {}
    value = getattr(args, flag, None)
    if value is None or value == default:
        return {}
    return {flag: component_dtype(value)}


class WanOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model
        entry = resolve_model(self.args.model_id, model_type=_MODEL_TYPE)
        resolve_model_path(self.args.model_id, revision=self.args.revision,
                           local_files_only=False,
                           allow_patterns=entry.download_patterns)
        print(f"[difflet] weights ready for {self.args.model_id}")

    def _reject_unsupported_torchrun_options(self) -> None:
        """A torchrun backend (neuron) asked for an option its ranks reject in phase 1: print
        one message naming it and exit 2 before anything is launched (or downloaded, for
        ``run``). The ranks' own checks stay (defence in depth)."""
        reject_unsupported_torchrun_options(
            # TeaCache runs on Wan's neuron path (the stage forwards its flags): not refused.
            [f for f in _torchrun_unsupported_options(self.args) if not f.startswith("--teacache")],
            runs="tensor, sequence, CFG and context parallelism at one static shape, DP "
            "replicas and TeaCache",
        )
        # --vae-dtype sets the device decoder: refused with --host-vae (the AoT backends keep
        # their own paths for these flags).
        refuse_torchrun_conflicts([c for c in (vae_dtype_conflict(self.args),) if c])

    def _reject_dual_expert_misuse(self) -> None:
        """--dual-expert off the neuron backend, or --guidance-scale-2 without it: print one
        message and exit 2 before anything is launched (or downloaded, for ``run``)."""
        refuse(_dual_expert_misuse(self.args, _torchrun_backend()))

    def _place_text_encoder(self) -> None:
        """On a torchrun backend (neuron), move umT5 to rank 0's host when it cannot fit on the
        device beside the DiT at this TP degree, and print one line saying so.

        The CLI default is the device encoder in bf16 (``_TORCHRUN_APPLICATION_KWARGS``, the
        application's ``text_encoder_dtype``). At tp2 (``--cfg-parallel`` or ``--cp-degree>1``
        on four cores, ``--mode latency``'s CFG, DP replicas of tp2) that is about 5.3 GiB of
        encoder beside 13.3 GiB of DiT, which fits a core; tp1, or tp2 with both of Wan 2.2's
        experts resident, does not; with ``--text-encoder-dtype fp32`` the encoder doubles and
        tp2 does not fit either. Before this check the application refused it only after
        torchrun had started every rank. The estimate
        (``difflet/backends/neuron/wan/hbm.py``) depends on the tp the transformer stage runs,
        the local checkpoint's sizes and umT5's precision. Weights that are not local yet
        (``compile`` before any download) leave the flag alone, and the application's own check
        still refuses. The
        setting lands on ``self.args``, so ``_shared_cli_args`` hands it to the stages, and a
        ``run`` that dispatches DP workers after ``compile`` forwards it to them
        (``worker_cli_args``). Torch-free, like every driver check.
        """
        args = self.args
        if _torchrun_backend() is None or getattr(args, "host_text_encoder", False):
            return
        from difflet.backends.neuron.wan.hbm import device_text_encoder_fits
        from difflet.pipeline.path_resolver import resolve_model_path

        experts = (("transformer", "transformer_2") if getattr(args, "dual_expert", False)
                   else ("transformer",))
        encoder_bytes = 4 if getattr(args, "text_encoder_dtype", None) == "fp32" else 2
        overflow = device_encoder_overflow(
            lambda: resolve_model_path(args.model_id, revision=args.revision,
                                       local_files_only=True),
            lambda model_dir, tp_degree: device_text_encoder_fits(
                model_dir, tp_degree=tp_degree, experts=experts,
                encoder_dtype_bytes=encoder_bytes),
            args.tp_degree or 4,
        )
        if overflow is None:
            return
        args.host_text_encoder = True
        print(f"[wan] umT5 on the device {overflow}: encoding on rank 0's host instead "
              "(--host-text-encoder)", flush=True)

    def run(self) -> None:
        self._reject_unsupported_torchrun_options()  # before the download, too
        self._reject_dual_expert_misuse()
        super().run()

    def _root_compile_cache_at_cache_dir(self) -> None:
        """--cache-dir roots a torchrun backend's NEFF/HLO compile cache
        (``root_compile_cache_at_cache_dir``)."""
        root_compile_cache_at_cache_dir(getattr(self.args, "cache_dir", None), tag="wan")

    def compile(self) -> None:
        self._reject_unsupported_torchrun_options()
        self._reject_dual_expert_misuse()
        self._place_text_encoder()
        self._root_compile_cache_at_cache_dir()
        if _torchrun_backend() == "neuron":
            self._make_bf16_dit_copies()
        full_cores = replica_cores(self.args)
        shared = self._shared_cli_args(stage_mode="compile")
        runner.run_stage(self.args.model_id, "transformer",
                         num_cores=full_cores,
                         virtual_core_size=_transformer_virtual_core_size(self.args),
                         cli_args=shared)
        if getattr(self.args, "host_vae", False):
            return
        runner.run_stage(self.args.model_id, "vae",
                         num_cores=1, virtual_core_size=_VIRTUAL_CORE_SIZE,
                         cli_args=shared)

    def _make_bf16_dit_copies(self) -> None:
        """Once, at compile time (as the Trainium path's pre-shards): bf16 copies of the fp32
        experts the stages will load, so a cold start reads half the bytes."""
        import os
        import time

        from difflet.backends.neuron.core.weights_bf16 import ensure_bf16_copy
        from difflet.pipeline.path_resolver import resolve_model_path

        try:
            root = resolve_model_path(self.args.model_id, revision=getattr(self.args, "revision", None),
                                      local_files_only=True)
        except Exception as exc:  # no local snapshot: the stages report it themselves
            print(f"[wan] bf16 copies skipped: snapshot not resolvable ({exc})", flush=True)
            return
        experts = ["transformer"] + (["transformer_2"] if getattr(self.args, "dual_expert", False) else [])
        for name in experts:
            started = time.perf_counter()
            copy = ensure_bf16_copy(os.path.join(root, name))
            print(f"[wan] bf16 copy of {name}: {copy or 'not needed (no fp32 tensors)'} "
                  f"({time.perf_counter() - started:.1f} s)", flush=True)

    def generate(self) -> None:
        self._reject_unsupported_torchrun_options()
        self._reject_dual_expert_misuse()
        self._place_text_encoder()
        self._root_compile_cache_at_cache_dir()
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(self.args.work_dir or
                        Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME)
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = replica_cores(self.args)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        try:
            runner.run_stage(self.args.model_id, "transformer",
                             num_cores=full_cores,
                             virtual_core_size=_transformer_virtual_core_size(self.args),
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
        backend = _torchrun_backend()
        if backend is not None:
            self._stage_transformer_neuron(args, backend)
            return

        import torch

        from difflet.models.wan.application import NeuronWanApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id,
                                       revision=self.args.revision,
                                       local_files_only=True)
        compile_shapes = _require_request_shape_in_set(args)
        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            cfg_parallel_enabled=getattr(args, "cfg_parallel", False),
            sp_enabled=getattr(args, "sp_enabled", False),
        )
        compiled_dir = self._stage_compiled_dir("transformer", args)
        h, w, f = args.height or 480, args.width or 832, args.num_frames or 9
        # Calibrated-adaptive TeaCache (--teacache-speedup + --teacache-calibration):
        # the application builds the fused block-0 probe NEFF for an adaptive
        # calibration (application.py, requires_teacache_probe), so the probe is
        # part of the transformer artifact identity (_teacache_probe_required in
        # _stage_cache_inputs). Validated here the way the probe pipelines do.
        teacache_calibration = adaptive_teacache_calibration(
            args, model="wan", shape_label=f"{h}x{w}x{f}"
        )
        app = NeuronWanApplication(
            model_path=model_dir,
            parallel=parallel,
            dtype=torch.bfloat16,
            shape={"height": h, "width": w, "num_frames": f},
            shapes=compile_shapes,
            text_seq_len=512,
            batch_size=1,
            enable_text_encoder=True,
            enable_transformer=True,
            enable_transformer_2=False,
            enable_vae_decoder=False,
            # Probe-free TeaCache (fixed cadence / online-delta): host-side skip
            # logic only; not in _stage_cache_inputs, so the warm artifact hits.
            teacache_cadence=getattr(args, "teacache_cadence", None),
            teacache_online_delta_alpha=getattr(args, "teacache_online_delta", None),
            teacache_calibration_path=teacache_calibration,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            self._finish_stage_compile("transformer", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("transformer", args, compiled_dir)
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

    def _stage_transformer_neuron(self, args: argparse.Namespace, backend: str) -> None:
        """The transformer stage on a non-AoT torchrun backend (neuron); runs on every rank.

        ``run_generate_stage`` builds the registry's application (TorchNeuronWanApplication)
        through ``DiffletPipeline``, loads it with the warm-up (``compile`` mode is that plus the
        stage manifest) and runs the replica's requests. Wan's part: the request's shape on the
        call (the orchestrator draws the noise for it), two steps and guidance 1.0 unless the
        request or the command line say otherwise, the low-noise expert's guidance only when
        --guidance-scale-2 is given (the orchestrator otherwise uses --guidance-scale
        throughout), and the latents written by rank 0 to the request's ``latents.pt``.
        """
        import torch

        from difflet.cli.dp import stage_loop

        compile_shapes = _require_request_shape_in_set(args)
        h, w, f = args.height or 480, args.width or 832, args.num_frames or 9
        low_noise = ({"guidance_scale_2": float(args.guidance_scale_2)}
                     if getattr(args, "guidance_scale_2", None) is not None else {})

        def request_kwargs(req) -> dict:
            return dict(
                generator=torch.Generator().manual_seed(req.seed),
                prompt=req.prompt,
                height=h,
                width=w,
                num_frames=f,
                num_inference_steps=int(stage_loop.effective(req, args, "steps", 2)),
                guidance_scale=float(stage_loop.effective(req, args, "guidance_scale", 1.0)),
                output_type="latent",
                **low_noise,
            )

        def save(req, out) -> None:
            latents_out = out.latents if hasattr(out, "latents") else out[0]
            dest = stage_loop.work_file(args, req, "latents.pt")
            torch.save(latents_out.cpu(), dest)
            print(f"[wan] latents saved to {dest}", flush=True)

        run_generate_stage(
            args, backend, tag="wan", model_id=self.args.model_id,
            pipeline_kwargs=dict(revision=self.args.revision, height=h, width=w, num_frames=f,
                                 shapes=compile_shapes),
            application_kwargs=_torchrun_application_kwargs(args),
            request_kwargs=request_kwargs,
            save=save,
            finish_compile=lambda: self._finish_stage_compile(
                "transformer", args, self._stage_compiled_dir("transformer", args)),
            timing=True,
        )

    def _stage_vae(self, args: argparse.Namespace) -> None:
        if _torchrun_backend() is not None:
            self._stage_vae_neuron(args)
            return
        import torch

        from difflet.models.wan.application import NeuronWanApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id,
                                       revision=self.args.revision,
                                       local_files_only=True)
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
            self._finish_stage_compile("vae", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("vae", args, compiled_dir)
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
                _write_frames(frames, req.output)

    def _stage_vae_neuron(self, args: argparse.Namespace) -> None:
        """The neuron backend's vae stage (``run_vae_stage``), structured as Trainium's: a launch
        of its own after the transformer stage has exited, so the decoder's activations (about
        12 GiB per core at 480x832 in fp32) never share a core with the DiT and the text
        encoder. One rank decodes eager (``NeuronWanVaeDecoderApplication``) at the request's
        latent video shape, in --vae-dtype when given, else ``default_vae_dtype`` of the clip
        (bf16 up to 9 frames, fp32 beyond; the frames come back fp32 either way). The latents are un-normalised in fp32 as the host decode and
        the orchestrator do, and rank 0 writes the frames."""
        import torch

        from difflet.backends.neuron.core import distributed
        from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication
        from difflet.cli.dp import stage_loop
        from difflet.models.wan.neuron_application import _DeviceVaeDecoder
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = resolve_model_path(self.args.model_id, revision=self.args.revision,
                                       local_files_only=True)
        vae_dir = Path(model_dir) / "vae"
        with open(vae_dir / "config.json", encoding="utf-8") as f:
            z_dim = int(json.load(f)["z_dim"])
        h, w, frames_count = args.height or 480, args.width or 832, args.num_frames or 9
        dtype = getattr(torch, _vae_dtype_name(args))

        def build(parallel):
            return NeuronWanVaeDecoderApplication(
                model_path=str(vae_dir),
                latent_shape=(1, z_dim, (frames_count - 1) // 4 + 1, h // 8, w // 8),
                parallel=parallel,
                dtype=dtype,
                device="neuron",
            )

        def prepare_decode(app):
            decoder = _DeviceVaeDecoder(app, device="neuron")
            config = app.vae_config

            def decode(req) -> None:
                z = torch.load(stage_loop.work_file(args, req, "latents.pt"),
                               map_location="cpu").to(torch.float32)
                if config.latents_mean and config.latents_std:
                    z = (z * torch.tensor(config.latents_std).view(1, z_dim, 1, 1, 1)
                         + torch.tensor(config.latents_mean).view(1, z_dim, 1, 1, 1))
                frames = decoder(z)
                if distributed.is_rank0():
                    _write_frames(frames, req.output)

            return decode

        run_vae_stage(
            args, build=build, prepare_decode=prepare_decode,
            finish_compile=lambda: self._finish_stage_compile(
                "vae", args, self._stage_compiled_dir("vae", args)),
        )

    # ------------------------------------------------------------ helpers

    def _stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        return {**torchrun_stage_keys(self._backend_neutral_stage_cache_inputs(stage, args)),
                **_component_dtype_keys(stage, args)}

    def _backend_neutral_stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        prefix = _cache_prefix(self.args.model_id)
        if stage == "transformer":
            # Additive: the probe key appears only for a calibrated-adaptive
            # request, so every existing artifact keeps its identity.
            probe = {"teacache_probe": True} if _teacache_probe_required(args) else {}
            return {
                **probe,
                "component": f"{prefix}_transformer",
                "model_id": self.args.model_id,
                "tp": args.tp_degree or 4,
                "cp": args.cp_degree or 1,
                "cp_mode": str(getattr(args, "cp_mode", "gather_kv") or "gather_kv"),
                "cfg_parallel": bool(getattr(args, "cfg_parallel", False)),
                "sp": bool(getattr(args, "sp_enabled", False)),
                "dtype": "bfloat16",
                "text_seq_len": 512,
                "shapes": canonical_shapes_list(args, (480, 832, 9)),
                "toolchain": stage_toolchain_versions(),
            }
        if stage == "vae":
            return {
                "component": f"{prefix}_vae",
                "model_id": self.args.model_id,
                "dtype": "bfloat16",
                "shapes": canonical_shapes_list(args, (480, 832, 9)),
                "toolchain": stage_toolchain_versions(),
            }
        raise ValueError(f"unknown stage {stage!r}")

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        inputs = self._stage_cache_inputs(stage, args)
        return hashed_stage_dir(base, str(inputs["component"]), inputs)

    def _finish_stage_compile(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        write_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args))

    def _require_stage_artifact(self, stage: str, args: argparse.Namespace, compiled_dir: Path) -> None:
        if not has_valid_stage_manifest(compiled_dir, self._stage_cache_inputs(stage, args)):
            raise SystemExit(
                f"[wan] no valid compiled artifact for stage {stage!r} at {compiled_dir} "
                "(manifest missing or configuration changed); run `difflet compile` "
                "with the same flags first."
            )

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
        if getattr(a, "dual_expert", False):
            parts.append("--dual-expert")
        if getattr(a, "host_text_encoder", False):
            parts.append("--host-text-encoder")
        # Only when given, so a default command line stays byte-identical.
        if getattr(a, "text_encoder_dtype", None):
            parts += ["--text-encoder-dtype", a.text_encoder_dtype]
        if getattr(a, "vae_dtype", None):
            parts += ["--vae-dtype", a.vae_dtype]
        if getattr(a, "guidance_scale_2", None) is not None:
            parts += ["--guidance-scale-2", str(a.guidance_scale_2)]
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
        # Only when pinned: a command line without --revision stays byte-identical.
        if getattr(a, "revision", None):
            parts += ["--revision", a.revision]
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
        return parts
