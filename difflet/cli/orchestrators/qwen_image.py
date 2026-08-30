"""Qwen-Image T2I orchestrator — 3-stage subprocess pipeline.

Stage 1 (text):     Qwen2.5-VL text encoder,  tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 2 (generate): DiT backbone,              tp×cp cores, VIRTUAL_CORE_SIZE=2.
Stage 3 (vae):      VAE decoder,               1 core,      VIRTUAL_CORE_SIZE=2.
Inter-stage tensors: {work_dir}/text.pt, {work_dir}/latents.pt.

Stage logic migrated from examples/qwen_image_example.py.
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
    parse_shapes_arg,
    require_request_shape_in_set,
    stage_toolchain_versions,
    write_stage_manifest,
)
from difflet.cli import runner

_HF_MODEL_ID = "Qwen/Qwen-Image"
_MODEL_TYPE = "qwen_image"
_CLI_NAME = qwen_common.CLI_NAME
_VIRTUAL_CORE_SIZE = qwen_common.VIRTUAL_CORE_SIZE
_ENC_SEQ = qwen_common.ENC_SEQ
_TEXT_SEQ_LEN = qwen_common.TEXT_SEQ_LEN

_QWEN_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_QWEN_DROP_IDX = 34


def _require_request_shape_in_set(args: argparse.Namespace):
    return require_request_shape_in_set(
        args, default_shape=(1024, 1024), model_tag="qwen_image"
    )


class QwenImageOrchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path

        resolve_model_path(_HF_MODEL_ID, revision=self.args.revision, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
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
        _require_request_shape_in_set(self.args)  # fail fast before any stage runs
        work_dir = Path(
            self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        full_cores = (self.args.tp_degree or 4) * (self.args.cp_degree or 1)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
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

    def _stage_text(self, args: argparse.Namespace) -> None:
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

    def _stage_vae(self, args: argparse.Namespace) -> None:
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
            app.compile(str(compiled_dir))
            self._finish_stage_compile("vae", args, compiled_dir)
            return

        from difflet.cli.dp import stage_loop

        self._require_stage_artifact("vae", args, compiled_dir)
        app.load(str(compiled_dir))
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                packed = torch.load(stage_loop.work_file(args, req, "latents.pt")).float()
                b, seq, _ = packed.shape
                # Packed grid follows the REQUEST shape (not necessarily square).
                hh = (args.height or 1024) // 16
                ww = (args.width or 1024) // 16
                if hh * ww != seq:
                    raise ValueError(
                        f"packed latent length {seq} does not match request grid {hh}x{ww}"
                    )
                z = (
                    packed.view(b, hh, ww, 16, 2, 2)
                    .permute(0, 3, 1, 4, 2, 5)
                    .reshape(b, 16, hh * 2, ww * 2)
                )
                z = z.unsqueeze(2)
                mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
                std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
                z = (
                    (z * std + mean).to(torch.bfloat16)
                    if len(config.latents_mean)
                    else z.to(torch.bfloat16)
                )
                img = app(z)
                img = (img[0] if isinstance(img, (tuple, list)) else img).float().cpu()
                img = img[:, :, 0]
                out_path = Path(req.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    from torchvision.utils import save_image

                    save_image((img[0] * 0.5 + 0.5).clamp(0, 1), str(out_path))
                    print(f"[vae] image saved to {out_path}")
                except Exception as exc:
                    torch.save(img, out_path.with_suffix(".pt"))
                    print(
                        f"[vae] tensor saved to {out_path.with_suffix('.pt')} "
                        f"(png skipped: {exc})"
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
