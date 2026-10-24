"""Qwen-Image on the GPU flow: the DiT through ``difflet.ops``, the Qwen2.5-VL text encoder and
the ``AutoencoderKLQwenImage`` VAE on the same device, denoising and decoding through the
backend-neutral ``QwenImageOrchestrator``.

``__call__`` accepts a ``QwenImageDiTInputBundle`` or the five DiT tensors as keywords (parity
scripts), and ``prompt=`` for the whole text-to-image path.
"""

from __future__ import annotations

from typing import Any

import torch

from difflet.backends.gpu.application import GpuApplication
from difflet.models.qwen_image.contract import QwenImageDiTInputBundle
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
from difflet.models.qwen_image.spec import SPEC


def dynamic_shift_mu(scheduler_config, image_seq_len: int) -> float:
    """``mu`` for a ``use_dynamic_shifting`` FlowMatch scheduler — the formula
    difflet/cli/orchestrators/qwen_image.py and diffusers' ``calculate_shift`` share."""
    sc = scheduler_config
    slope = (sc.max_shift - sc.base_shift) / (sc.max_image_seq_len - sc.base_image_seq_len)
    return float(image_seq_len * slope + (sc.base_shift - slope * sc.base_image_seq_len))


def qwen_image_timesteps(scheduler, *, num_inference_steps: int, image_seq_len: int):
    """Timesteps the way the Trainium CLI builds them: linear sigmas plus the dynamic
    shift ``mu`` (the model's scheduler config has ``use_dynamic_shifting=True``, and
    ``set_timesteps`` refuses to run without ``mu``)."""
    import numpy as np

    steps = max(int(num_inference_steps), 1)
    sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
    mu = dynamic_shift_mu(scheduler.config, image_seq_len)
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
    return scheduler.timesteps


class CudaQwenImageApplication(GpuApplication):
    spec = SPEC
    input_bundle_type = QwenImageDiTInputBundle

    def _configure(self, kwargs: dict[str, Any]) -> None:
        self.text_seq_len = int(kwargs.get("text_seq_len", 1024))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.load_text_encoder = bool(kwargs.get("load_text_encoder", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        # TeaCache: the probe-free fixed cadence runs on the host-side controller the Qwen
        # orchestrator already has; the calibrated and online-delta modes are refused (spec).
        cadence = kwargs.get("teacache_cadence")
        self.teacache_cadence = int(cadence) if cadence is not None else None
        self.text_pipeline = None
        self.vae = None
        self.pipeline = None

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.qwen_image.transformer import (
            CudaQwenImageTransformerApplication,
        )

        parallel = self.parallel
        self.config = QwenImageDiTConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
            batch_size=self.batch_size,
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
            context_parallel_enabled=int(getattr(parallel, "cp_degree", 1)) > 1,
            cp_mode=getattr(parallel, "cp_mode", "gather_kv"),
        )
        self.transformer = CudaQwenImageTransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
        )

    # ------------------------------------------------ encoder / VAE / loop

    def load_companions(self) -> None:
        """Text encoder, VAE and the denoise/decode orchestrator, all on the device."""
        from difflet.models.qwen_image.pipeline import QwenImageOrchestrator
        from difflet.models.qwen_image.prompt import load_text_pipeline

        if self.load_text_encoder and self.text_pipeline is None:
            self.text_pipeline = load_text_pipeline(
                model_path=self.model_path, dtype=self.dtype, device=self.device
            )
        if self.load_vae and self.vae is None:
            from diffusers import AutoencoderKLQwenImage

            self.vae = AutoencoderKLQwenImage.from_pretrained(
                self.model_path, subfolder="vae", torch_dtype=self.dtype, local_files_only=True
            ).to(self.device)
            self.vae.eval()
        scheduler = self.text_pipeline.scheduler if self.text_pipeline is not None else None
        self.pipeline = QwenImageOrchestrator(
            model_path=self.model_path,
            transformer=self,
            vae=self.vae,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
            scheduler=scheduler,
            teacache_cadence=self.teacache_cadence,
        )

    def encode_prompt(self, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        from difflet.models.qwen_image.prompt import encode_prompt_fixed_length

        if self.text_pipeline is None:
            raise RuntimeError("text encoder is not loaded; call load() first")
        return encode_prompt_fixed_length(
            self.text_pipeline, prompt, device=self.device, max_sequence_length=self.text_seq_len
        )

    # ------------------------------------------------------------- DiT API

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").dit_config
        batch = int(cfg.batch_size)
        return {
            "hidden_states": {
                "shape": (batch, int(cfg.image_seq_len), int(cfg.in_channels)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch, int(cfg.text_seq_len), int(cfg.joint_attention_dim)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states_mask": {
                "shape": (batch, int(cfg.text_seq_len)),
                "dtype": torch.bool,
            },
            "guidance": {"shape": (batch,), "dtype": self.dtype},
        }

    # ---------------------------------------------------------- generation

    def generate(
        self,
        prompt: str | list[str],
        *,
        num_inference_steps: int = SPEC.default_steps,
        seed: int | None = None,
        guidance_scale: float | None = None,
        output_type: str = "pil",
        latents: torch.Tensor | None = None,
    ):
        """Text → image on the device: encode, denoise through the Difflet DiT, decode.

        ``latents`` is packed ``(B, S, 64)`` initial noise used verbatim (the end-to-end
        oracle feeds the same tensor to diffusers); otherwise ``seed`` draws it.

        One pass per step without classifier-free guidance, the path the Trainium CLI runs.
        The transformer has ``guidance_embeds=False``, so the modeling code drops the guidance
        tensor; a ``guidance_scale`` would be a silent no-op and is refused.
        """
        if guidance_scale is not None and not bool(getattr(self.config, "guidance_embeds", False)):
            raise ValueError(f"guidance_scale has no effect: {SPEC.no_guidance_reason}")
        if self.pipeline is None:
            raise RuntimeError("application is not loaded; call load() first")
        encoder_hidden_states, encoder_hidden_states_mask = self.encode_prompt(prompt)
        batch = int(encoder_hidden_states.shape[0])
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = self.pipeline.prepare_latents(
            batch_size=batch,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            latents=latents,
        )
        timesteps = None
        if self.pipeline.scheduler is not None:
            timesteps = qwen_image_timesteps(
                self.pipeline.scheduler,
                num_inference_steps=int(num_inference_steps),
                image_seq_len=int(latents.shape[1]),
            ).to(self.device)
        # The Trainium CLI fills the ignored guidance tensor with this value; mirrored so the
        # DiT inputs are identical, not because it changes the image.
        guidance_scale = (
            SPEC.serving_noop_guidance_scale if guidance_scale is None else float(guidance_scale)
        )
        out = self.pipeline(
            latents=latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            batch_size=batch,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            output_type=(
                "pt" if output_type in {"pt", "pil", "np"} and self.vae is not None else "latent"
            ),
            generator=generator,
        )
        if output_type == "latent" or self.vae is None:
            return out
        images = out.images
        if images.ndim == 5:  # (B, C, 1, H, W) from the 3D VAE
            images = images[:, :, 0]
        if output_type == "pt":
            return out
        from diffusers.image_processor import VaeImageProcessor

        processor = VaeImageProcessor(vae_scale_factor=self.pipeline.vae_scale_factor * 2)
        return processor.postprocess(images.float(), output_type=output_type)


__all__ = ["CudaQwenImageApplication", "dynamic_shift_mu", "qwen_image_timesteps"]
