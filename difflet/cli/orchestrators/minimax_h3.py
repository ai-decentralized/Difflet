"""MiniMax-H3 T2VA orchestrator — four sequential Trainium stages.

Stage 1 (text):      Qwen3-VL layer-50 conditioner, TP4.
Stage 2 (generate):  33B H3 Omni Transformer,          TP4.
Stage 3 (video_vae): H3 visual VAE decoder,             1 core initially.
Stage 4 (audio_vae): H3 audio VAE decoder and AV mux,   1 core initially.

Only light tokenization, scheduling, tensor layout, normalization and media
post-processing remain on the host.  Staging is required by HBM capacity, not
because the heavy components are intended to execute on the host.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from difflet.cli import runner
from difflet.cli.orchestrators.base import ModelOrchestrator
from difflet.common.orchestrators import minimax_h3 as h3_common

_HF_MODEL_ID = h3_common.HF_MODEL_ID
_MODEL_TYPE = h3_common.MODEL_TYPE
_CLI_NAME = h3_common.CLI_NAME
_VIRTUAL_CORE_SIZE = h3_common.VIRTUAL_CORE_SIZE
_STAGES = ("text", "generate", "video_vae", "audio_vae")


class MiniMaxH3Orchestrator(ModelOrchestrator):
    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model

        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        resolve_model_path(
            _HF_MODEL_ID,
            revision=self.args.revision,
            local_files_only=False,
            allow_patterns=entry.download_patterns,
        )
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        self._validate_topology()
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
        for stage in _STAGES:
            runner.run_stage(
                _HF_MODEL_ID,
                stage,
                num_cores=self._stage_cores(stage),
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )

    def generate(self) -> None:
        self._validate_topology()
        work_dir = Path(
            self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        try:
            for stage in _STAGES:
                runner.run_stage(
                    _HF_MODEL_ID,
                    stage,
                    num_cores=self._stage_cores(stage),
                    virtual_core_size=_VIRTUAL_CORE_SIZE,
                    cli_args=shared,
                )
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        handlers = {
            "text": self._stage_text,
            "generate": self._stage_generate,
            "video_vae": self._stage_video_vae,
            "audio_vae": self._stage_audio_vae,
        }
        try:
            handler = handlers[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown MiniMax-H3 stage: {stage!r}") from exc
        handler(args)

    def _stage_text(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLTextForCausalLM,
        )
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        encoder_path = str(Path(model_dir) / "text_encoder")
        compiled_dir = self._stage_compiled_dir("text", args)
        tp_degree = args.tp_degree or 4

        hf_config = AutoConfig.from_pretrained(encoder_path)
        text_config = hf_config.text_config
        if getattr(text_config, "pad_token_id", None) is None:
            text_config.pad_token_id = 0
        if int(getattr(text_config, "num_hidden_layers", 0)) <= h3_common.TEXT_ENCODER_LAYER:
            raise ValueError(
                "MiniMax-H3 requires Qwen3-VL hidden_states[50], but the text encoder "
                f"has only {getattr(text_config, 'num_hidden_layers', None)} layers."
            )

        weights_dir = compiled_dir / "weights"
        shard_paths = [
            weights_dir / f"tp{rank}_sharded_checkpoint.safetensors" for rank in range(tp_degree)
        ]
        save_sharded_checkpoint = args.stage_mode == "compile" or all(
            path.exists() for path in shard_paths
        )
        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            batch_size=1,
            seq_len=h3_common.TEXT_SEQ_LEN,
            torch_dtype=torch.bfloat16,
            on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=["layers.49"]),
            save_sharded_checkpoint=save_sharded_checkpoint,
        )
        config = NeuronQwen3VLTextForCausalLM.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(hf_config=text_config),
        )
        app = NeuronQwen3VLTextForCausalLM(encoder_path, config)
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop

        app.load(str(compiled_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                tokenized = tokenizer(
                    req.prompt,
                    add_special_tokens=False,
                    max_length=h3_common.TEXT_SEQ_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids = tokenized.input_ids.to(torch.int32)
                attention_mask = tokenized.attention_mask.to(torch.int32)
                token_count = int(attention_mask.sum().item())
                if token_count < 1:
                    raise ValueError("MiniMax-H3 prompt must produce at least one token.")
                output = app(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=torch.arange(h3_common.TEXT_SEQ_LEN, dtype=torch.int32).unsqueeze(
                        0
                    ),
                    sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                )
                hidden_states = output.captured_tensors[0].to(torch.bfloat16)
                # Padding rows are never part of the official packed document.
                # Preserve a fixed inter-stage tensor and carry the live length
                # explicitly so the DiT stage can build its masked tail layout.
                hidden_states[:, token_count:] = 0
                destination = stage_loop.work_file(args, req, "text.pt")
                torch.save(
                    {
                        "encoder_hidden_states": hidden_states,
                        "attention_mask": attention_mask.to(torch.bool),
                        "num_text_tokens": token_count,
                        "text_encoder_layer": h3_common.TEXT_ENCODER_LAYER,
                    },
                    destination,
                )
                print(
                    f"[text] Qwen3-VL hidden_states[50] {tuple(hidden_states.shape)} "
                    f"({token_count} live tokens) -> {destination}"
                )

    def _stage_generate(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.backends.trainium.minimax_h3.transformer import (
            NeuronMiniMaxH3TransformerApplication,
        )
        from difflet.models.minimax_h3.application import (
            create_minimax_h3_transformer_config,
        )
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        transformer_path = str(Path(model_dir) / "transformer")
        compiled_dir = self._stage_compiled_dir("generate", args)
        config = create_minimax_h3_transformer_config(
            model_path=model_dir,
            tp_degree=args.tp_degree or 4,
            dtype=torch.bfloat16,
            height=args.height or h3_common.DEFAULT_HEIGHT,
            width=args.width or h3_common.DEFAULT_WIDTH,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
        )
        app = NeuronMiniMaxH3TransformerApplication(
            model_path=transformer_path,
            config=config,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.pipeline import (
            denoise_minimax_h3_t2va,
            load_minimax_h3_schedulers,
        )

        app.load(str(compiled_dir), skip_warmup=True)
        height = args.height or h3_common.DEFAULT_HEIGHT
        width = args.width or h3_common.DEFAULT_WIDTH
        num_frames = args.num_frames or h3_common.DEFAULT_NUM_FRAMES
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                text = torch.load(stage_loop.work_file(args, req, "text.pt"))
                video_scheduler, audio_scheduler = load_minimax_h3_schedulers(model_dir)
                output = denoise_minimax_h3_t2va(
                    app,
                    encoder_hidden_states=text["encoder_hidden_states"],
                    encoder_attention_mask=text["attention_mask"],
                    num_text_tokens=int(text["num_text_tokens"]),
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=int(
                        stage_loop.effective(req, args, "steps", h3_common.DEFAULT_STEPS)
                    ),
                    seed=req.seed,
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    model_dtype=torch.bfloat16,
                )
                destination = stage_loop.work_file(args, req, "latents.pt")
                torch.save(
                    {
                        "video_latents": output.video_latents.cpu(),
                        "audio_latents": output.audio_latents.cpu(),
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                    },
                    destination,
                )
                print(
                    f"[generate] video latents {tuple(output.video_latents.shape)}, "
                    f"audio latents {tuple(output.audio_latents.shape)} -> {destination}"
                )

    def _stage_video_vae(self, args: argparse.Namespace) -> None:
        self._pending_stage("video_vae", "H3 visual VAE Neuron decoder")

    def _stage_audio_vae(self, args: argparse.Namespace) -> None:
        self._pending_stage("audio_vae", "H3 audio VAE Neuron decoder")

    @staticmethod
    def _pending_stage(stage: str, component: str) -> None:
        raise NotImplementedError(
            f"MiniMax-H3 stage {stage!r} is reserved for the {component}; "
            "no host fallback is used."
        )

    def _validate_topology(self) -> None:
        if (self.args.tp_degree or 4) != 4:
            raise NotImplementedError("MiniMax-H3's first qualified graph is fixed at TP4.")
        if (self.args.cp_degree or 1) != 1:
            raise NotImplementedError("MiniMax-H3 CP is deferred until full-attention parity.")
        if getattr(self.args, "cfg_parallel", False):
            raise NotImplementedError("MiniMax-H3 is guidance-distilled and has no CFG branch.")
        if getattr(self.args, "sp_enabled", False):
            raise NotImplementedError(
                "MiniMax-H3 SP is deferred until the dense TP4 graph is qualified."
            )

    def _stage_cores(self, stage: str) -> int:
        return (self.args.tp_degree or 4) if stage in {"text", "generate"} else 1

    @staticmethod
    def _model_dir(args: argparse.Namespace, resolver) -> str:
        model_path = getattr(args, "model_path", None)
        if model_path:
            return str(Path(model_path).expanduser().resolve())
        return resolver(_HF_MODEL_ID, revision=args.revision, local_files_only=True)

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        compiled_dir = getattr(args, "compiled_dir", None)
        if compiled_dir:
            return Path(compiled_dir)
        return h3_common.stage_compiled_dir_from_values(
            stage,
            cache_dir=args.cache_dir,
            tp_degree=args.tp_degree or 4,
            height=args.height or h3_common.DEFAULT_HEIGHT,
            width=args.width or h3_common.DEFAULT_WIDTH,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
        )

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        args = self.args
        parts = [
            "--model-id",
            _HF_MODEL_ID,
            "--tp-degree",
            str(args.tp_degree or 4),
            "--cp-degree",
            str(args.cp_degree or 1),
            "--height",
            str(args.height or h3_common.DEFAULT_HEIGHT),
            "--width",
            str(args.width or h3_common.DEFAULT_WIDTH),
            "--num-frames",
            str(args.num_frames or h3_common.DEFAULT_NUM_FRAMES),
            "--steps",
            str(getattr(args, "steps", None) or h3_common.DEFAULT_STEPS),
            "--seed",
            str(getattr(args, "seed", 42)),
            "--stage-mode",
            stage_mode,
        ]
        for flag, value in (
            ("--prompt", getattr(args, "prompt", None)),
            ("--output", getattr(args, "output", None)),
            ("--cache-dir", getattr(args, "cache_dir", None)),
            ("--revision", getattr(args, "revision", None)),
            ("--work-dir", work_dir),
        ):
            if value:
                parts.extend([flag, str(value)])
        if getattr(args, "requests_dir", None):
            parts.extend(
                [
                    "--requests-dir",
                    str(args.requests_dir),
                    "--worker-index",
                    str(args.worker_index),
                    "--dp-schedule",
                    str(getattr(args, "dp_schedule", "round_robin")),
                ]
            )
        return parts
