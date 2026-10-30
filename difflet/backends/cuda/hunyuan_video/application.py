"""HunyuanVideo GPU pipeline: Difflet DiT inside the pinned Diffusers driver."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from difflet.backends.gpu.application import GpuApplication
from difflet.backends.gpu.policy import validate_gpu_steps
from difflet.models.hunyuan_video.dit_config import HunyuanVideoDiTConfig
from difflet.models.hunyuan_video.prompt import LLAMA_CROP_START, LLAMA_TEMPLATE, TEXT_SEQ_LEN
from difflet.models.hunyuan_video.spec import SPEC


class HunyuanVideoDiTAdapter(torch.nn.Module):
    """The Diffusers interface: upcast received time values without undoing driver rounding."""

    def __init__(self, component):
        super().__init__()
        self.component = component
        self.config = component.dit_config

    @property
    def dtype(self):
        return self.component.dtype

    @property
    def device(self):
        return self.component.device

    @contextmanager
    def cache_context(self, name):
        yield

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        if attention_kwargs:
            raise NotImplementedError("HunyuanVideo CUDA takes no attention_kwargs")
        if guidance is None:
            raise ValueError("HunyuanVideo requires its embedded guidance tensor")
        result = self.component(
            hidden_states.to(self.device, self.dtype),
            timestep.to(self.device, torch.float32),
            encoder_hidden_states.to(self.device, self.dtype),
            encoder_attention_mask.to(self.device, torch.bool),
            pooled_projections.to(self.device, self.dtype),
            guidance.to(self.device, self.dtype),
            return_dict=False,
        )[0]
        return Transformer2DModelOutput(sample=result) if return_dict else (result,)


class CudaHunyuanVideoApplication(GpuApplication):
    spec = SPEC

    def _configure(self, kwargs):
        self.text_seq_len = int(kwargs.get("text_seq_len", TEXT_SEQ_LEN))
        self.load_text_encoders = bool(kwargs.get("load_text_encoders", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.local_files_only = bool(kwargs.get("local_files_only", True))
        self.pipe = None
        self.adapter = None

    def _build_transformers(self):
        from difflet.backends.cuda.hunyuan_video.transformer import (
            CudaHunyuanVideoTransformerApplication,
        )

        self.config = HunyuanVideoDiTConfig.from_pretrained(
            self.transformer_path,
            **self.shape,
            text_seq_len=self.text_seq_len,
            torch_dtype=self.dtype,
        )
        self.transformer = CudaHunyuanVideoTransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
        )
        self.adapter = HunyuanVideoDiTAdapter(self.transformer)

    def load_companions(self):
        from diffusers import HunyuanVideoPipeline

        skipped = {"transformer": None}
        if not self.load_text_encoders:
            skipped.update(text_encoder=None, text_encoder_2=None)
        if not self.load_vae:
            skipped["vae"] = None
        base = HunyuanVideoPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            **skipped,
        )
        for name in ("text_encoder", "text_encoder_2", "vae"):
            module = getattr(base, name, None)
            if module is not None:
                module.to(self.device).requires_grad_(False).eval()
        self.pipe = HunyuanVideoPipeline(**{**base.components, "transformer": self.adapter})

    def dit_input_contract(self):
        cfg = self._require_transformer("input contract").dit_config
        return {
            "hidden_states": {
                "shape": (
                    1,
                    cfg.in_channels,
                    cfg.latent_frames,
                    cfg.latent_height,
                    cfg.latent_width,
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (1,), "dtype": torch.float32},
            "encoder_hidden_states": {
                "shape": (1, cfg.text_seq_len, cfg.text_embed_dim),
                "dtype": self.dtype,
            },
            "encoder_attention_mask": {"shape": (1, cfg.text_seq_len), "dtype": torch.bool},
            "pooled_projections": {"shape": (1, cfg.pooled_projection_dim), "dtype": self.dtype},
            "guidance": {"shape": (1,), "dtype": self.dtype},
        }

    def generate(
        self,
        prompt,
        *,
        num_inference_steps=SPEC.default_steps,
        guidance_scale=SPEC.default_guidance_scale,
        seed=None,
        latents=None,
        num_frames=None,
        height=None,
        width=None,
        output_type="pt",
        negative_prompt=None,
        true_cfg_scale=None,
        guidance_scale_2=None,
    ):
        if self.pipe is None:
            raise RuntimeError("application is not loaded; call load() first")
        if not isinstance(prompt, str):
            if not isinstance(prompt, (list, tuple)) or len(prompt) != 1:
                raise NotImplementedError("HunyuanVideo generates one prompt at a time")
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HunyuanVideo prompt must be a nonempty string")
        if any(v is not None for v in (negative_prompt, true_cfg_scale, guidance_scale_2)):
            raise NotImplementedError("HunyuanVideo CUDA supports distilled guidance only")
        for key, value in (("height", height), ("width", width), ("num_frames", num_frames)):
            if value is not None and value != self.shape[key]:
                raise ValueError(f"{key} differs from the loaded shape {self.shape[key]}")
        num_inference_steps = validate_gpu_steps(num_inference_steps)
        if not math.isfinite(float(guidance_scale)) or not 0 <= float(guidance_scale) <= 20:
            raise ValueError("guidance_scale must be finite and between 0 and 20")
        if output_type not in {"pt", "latent"}:
            raise ValueError("HunyuanVideo output_type must be pt or latent")
        if not self.load_text_encoders or getattr(self.pipe, "tokenizer", None) is None:
            raise RuntimeError("prompt generation requires loaded text encoders and tokenizer")
        if output_type == "pt" and not self.load_vae:
            raise RuntimeError("video generation requires a loaded VAE")
        encoded = self.pipe.tokenizer(
            LLAMA_TEMPLATE.format(prompt), padding=False, truncation=False, return_tensors="pt"
        )
        if len(encoded.input_ids[0]) - LLAMA_CROP_START > self.text_seq_len:
            raise ValueError("HunyuanVideo prompt exceeds Llama's 256 retained tokens")
        generator = None
        if latents is not None:
            expected = (
                1,
                self.config.in_channels,
                self.config.latent_frames,
                self.config.latent_height,
                self.config.latent_width,
            )
            if tuple(latents.shape) != expected or latents.dtype != torch.float32:
                raise ValueError(f"latents must have shape {expected} and dtype float32")
        elif seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            true_cfg_scale=1.0,
            generator=generator,
            latents=latents,
            output_type=output_type,
            max_sequence_length=self.text_seq_len,
        ).frames
        if output_type == "latent":
            return out
        return out.permute(0, 2, 1, 3, 4).to(device="cpu", dtype=torch.float32) * 2 - 1
