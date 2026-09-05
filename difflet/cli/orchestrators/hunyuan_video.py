"""HunyuanVideo T2V orchestrator — 3-stage subprocess pipeline.

Stage 1 (clip):     CLIP text encoder,   1 core, VIRTUAL_CORE_SIZE=2.
Stage 2 (llama):    Llama-3 encoder,     tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 3 (generate): DiT + VAE decoder,   tp×cp cores, VIRTUAL_CORE_SIZE=2.
Inter-stage tensors: {work_dir}/clip.pt, {work_dir}/llama.pt.

Stage logic migrated from examples/hunyuan_video_example.py.
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


class HunyuanVideoOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
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

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
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

    # ------------------------------------------------------------ helpers

    def _stage_cache_inputs(self, stage: str, args: argparse.Namespace) -> dict:
        """Identity for one stage artifact (hashed into the dir name; recorded
        verbatim in the dir's manifest.json)."""
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
        if work_dir:
            parts += ["--work-dir", work_dir]
        if getattr(a, "requests_dir", None):
            parts += ["--requests-dir", str(a.requests_dir),
                      "--worker-index", str(a.worker_index),
                      "--dp-schedule", str(getattr(a, "dp_schedule", "round_robin"))]
        return parts
