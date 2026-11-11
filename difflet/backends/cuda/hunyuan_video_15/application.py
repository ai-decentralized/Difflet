"""Standard 1.5 conditioning, CFG and scheduler semantics from the Diffusers driver."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from difflet.backends.cuda.hunyuan_video_15.conditioning import configure_conditioning
from difflet.backends.cuda.hunyuan_video_15.dit_config import HunyuanVideo15DiTConfig
from difflet.backends.gpu.application import GpuApplication
from difflet.models.hunyuan_video_15.spec import SPEC


class HunyuanVideo15DiTAdapter(torch.nn.Module):
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
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
        attention_kwargs=None,
        return_dict=True,
    ):
        # Every input is cast to the transformer's dtype, as Diffusers' HunyuanVideo 1.5 driver
        # does. Unlike the original HunyuanVideo adapter, the timestep is not kept in FP32.
        values = (
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image_embeds,
        )
        inputs = tuple(
            t.to(device=self.device, dtype=torch.bool if i in (3, 5) else self.dtype)
            for i, t in enumerate(values)
        )
        result = self.component(*inputs, return_dict=False)[0]
        return Transformer2DModelOutput(sample=result) if return_dict else (result,)


class CudaHunyuanVideo15Application(GpuApplication):
    spec = SPEC

    def _configure(self, kwargs):
        self.local_files_only = bool(kwargs.get("local_files_only", True))
        self.pipe = None
        self.adapter = None
        self._guider_config = None

    def _build_transformers(self):
        from difflet.backends.cuda.hunyuan_video_15.transformer import (
            CudaHunyuanVideo15TransformerApplication,
        )

        self.config = HunyuanVideo15DiTConfig.from_pretrained(
            self.transformer_path, **self.shape, torch_dtype=self.dtype
        )
        self.transformer = CudaHunyuanVideo15TransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
        )
        self.adapter = HunyuanVideo15DiTAdapter(self.transformer)

    def load_companions(self):
        from diffusers import HunyuanVideo15Pipeline

        base = HunyuanVideo15Pipeline.from_pretrained(
            self.model_path,
            transformer=None,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        )
        # Diffusers' native overlapping spatial tiles (256 px, 25% overlap) keep the decode
        # within one GPU at 121 frames.
        base.vae.enable_tiling()
        for name in ("text_encoder", "text_encoder_2", "vae"):
            getattr(base, name).to(self.device).requires_grad_(False).eval()
        self.pipe = HunyuanVideo15Pipeline(**{**base.components, "transformer": self.adapter})
        configure_conditioning(self.pipe)
        self._guider_config = dict(base.guider.config)

    def dit_input_contract(self):
        c = self._require_transformer("input contract").dit_config
        shapes = (
            ("hidden_states", (1, c.in_channels, c.latent_frames, c.latent_height, c.latent_width)),
            ("timestep", (1,)),
            ("encoder_hidden_states", (1, c.text_seq_len, c.text_embed_dim)),
            ("encoder_attention_mask", (1, c.text_seq_len)),
            ("encoder_hidden_states_2", (1, c.text_2_seq_len, c.text_embed_2_dim)),
            ("encoder_attention_mask_2", (1, c.text_2_seq_len)),
            ("image_embeds", (1, c.image_tokens, c.image_embed_dim)),
        )
        return {
            name: {"shape": shape, "dtype": torch.bool if "mask" in name else self.dtype}
            for name, shape in shapes
        }

    def generate(
        self,
        prompt,
        *,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=6.0,
        seed=None,
        latents=None,
        output_type="pt",
        height=None,
        width=None,
        num_frames=None,
    ):
        from diffusers import ClassifierFreeGuidance

        if self.pipe is None:
            raise RuntimeError("application is not loaded; call load() first")
        if not isinstance(prompt, str):
            raise ValueError("HunyuanVideo 1.5 generates one prompt per request")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        for key, value in (("height", height), ("width", width), ("num_frames", num_frames)):
            if value is not None and value != self.shape[key]:
                raise ValueError(f"{key} differs from the loaded shape {self.shape[key]}")
        if output_type not in ("pt", "latent"):
            raise ValueError("HunyuanVideo 1.5 output_type must be pt or latent")
        generator = None
        if latents is not None:
            c = self.config
            expected = (1, c.out_channels, c.latent_frames, c.latent_height, c.latent_width)
            if tuple(latents.shape) != expected:
                raise ValueError(f"latents must have shape {expected}")
        elif seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.pipe.guider = ClassifierFreeGuidance.from_config(
            {**self._guider_config, "guidance_scale": float(guidance_scale)}
        )
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            num_inference_steps=num_inference_steps,
            generator=generator,
            latents=latents,
            output_type=output_type,
        ).frames
        if output_type == "latent":
            return output
        return output.permute(0, 2, 1, 3, 4).to(device="cpu", dtype=torch.float32) * 2 - 1
