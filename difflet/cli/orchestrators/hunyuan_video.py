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
from difflet.cli.orchestrators.base import ModelOrchestrator

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
            return

        app.load(str(compiled_dir))
        tok = CLIPTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer_2"))
        ids = tok(args.prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(torch.int64)
        out = app(ids)
        pooled = out.pooler_output.to(torch.bfloat16).cpu().reshape(1, -1)
        work_dir = Path(args.work_dir)
        torch.save({"pooled_projections": pooled}, work_dir / "clip.pt")
        print(f"[clip] pooled_projections {tuple(pooled.shape)} -> {work_dir}/clip.pt")

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

        hf_cfg = AutoConfig.from_pretrained(enc_path)
        if hf_cfg.pad_token_id is None:
            hf_cfg.pad_token_id = 0
        hf_cfg.tie_word_embeddings = True

        neuron_config = NeuronConfig(
            tp_degree=args.tp_degree or 4, batch_size=1, seq_len=seq,
            torch_dtype=torch.bfloat16, on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=[_LLAMA_CAPTURE]),
        )
        config = NeuronLlamaForCausalLM.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
        )
        app = NeuronLlamaForCausalLM(enc_path, config)
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        app.load(str(compiled_dir))
        tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        ti = tok(
            _LLAMA_TEMPLATE.format(args.prompt), max_length=seq,
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
        work_dir = Path(args.work_dir)
        torch.save({"encoder_hidden_states": hidden, "encoder_attention_mask": mask},
                   work_dir / "llama.pt")
        print(f"[llama] encoder_hidden_states {tuple(hidden.shape)} -> {work_dir}/llama.pt")

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
        torch.manual_seed(args.seed)
        latents = torch.randn(1, 16, latent_frames, h // 8, w // 8, dtype=torch.bfloat16)
        guidance = torch.full([1], (args.guidance_scale or 6.0) * 1000.0, dtype=torch.bfloat16)

        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
            sp_enabled=getattr(args, "sp_enabled", False),
        )
        app = NeuronHunyuanVideoApplication(
            model_path=model_dir, parallel=parallel, dtype=torch.bfloat16,
            shape={"height": h, "width": w, "num_frames": f},
            text_seq_len=_TEXT_SEQ_LEN, enable_vae_decoder=True,
        )
        app.teacache_probe = None

        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        llama = torch.load(work_dir / "llama.pt")
        clip = torch.load(work_dir / "clip.pt")

        app.load(str(compiled_dir), skip_warmup=True)
        sigmas = np.linspace(1.0, 0.0, (args.steps or 4) + 1)[:-1]
        timesteps, _ = _retrieve_timesteps(
            app.pipeline.scheduler, args.steps or 4, "cpu", sigmas=sigmas
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
            num_inference_steps=args.steps or 4,
            output_type="pt", return_trajectory=False,
        )
        frames = output.frames
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".mp4" and _save_video(frames.cpu(), str(out_path)):
            pass
        else:
            pt_path = out_path.with_suffix(".pt")
            torch.save(frames.cpu(), pt_path)
            print(f"[generate] video tensor saved to {pt_path}")

    # ------------------------------------------------------------ helpers

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        base = Path(args.cache_dir or Path.home() / ".cache" / "difflet").expanduser()
        tp = args.tp_degree or 4
        cp = args.cp_degree or 1
        sp = "sp" if getattr(args, "sp_enabled", False) else ""
        h, w, f = args.height or 320, args.width or 512, args.num_frames or 61
        if stage == "clip":
            return base / "hunyuan_video_clip"
        if stage == "llama":
            return base / f"hunyuan_video_llama_seq{_TEXT_SEQ_LEN + _LLAMA_CROP_START}"
        if stage == "generate":
            return base / f"hunyuan_video_dit_tp{tp}cp{cp}{sp}_h{h}w{w}f{f}"
        raise ValueError(f"unknown stage {stage!r}")

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
        if getattr(a, "sp_enabled", False):
            parts.append("--sp")
        if getattr(a, "prompt", None):
            parts += ["--prompt", a.prompt]
        if getattr(a, "output", None):
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        return parts
