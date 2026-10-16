"""Qwen-Image on the CUDA backend: DiT through ``difflet.ops``, encoder and VAE on the
same device, one process (CUDA plan Phase 4, decisions D5 and D8).

Mirrors ``TpuQwenImageApplication``'s outward contract — the shape ``DiffletPipeline``
calls reflectively — and adds the two things TPU kept host-side: the Qwen2.5-VL text
encoder and the ``AutoencoderKLQwenImage`` VAE, both diffusers/transformers modules
placed on the GPU at ``load()``. Denoising and decoding go through the existing
``QwenImageOrchestrator``; only the DiT is Difflet code, which is the part under test.

``__call__`` accepts, like the TPU application, a ``QwenImageDiTInputBundle`` or the
five DiT tensors as keywords (parity scripts), and additionally ``prompt=`` for the
whole text-to-image path.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.models.qwen_image.contract import QwenImageDiTInputBundle, normalize_dtype
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig


class CudaQwenImageApplication(torch.nn.Module):
    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from difflet.backends.cuda.qwen_image.transformer import (
            CudaQwenImageTransformerApplication,
            transformer_dir,
        )

        self.model_path = model_path
        self.parallel = parallel
        self.dtype = normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or 1024),
            "width": int(shape.get("width") or 1024),
            "num_frames": None,
        }
        self.kwargs = kwargs
        self.device = torch.device(kwargs.get("device", "cuda"))
        self.text_seq_len = int(kwargs.get("text_seq_len", 1024))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.load_text_encoder = bool(kwargs.get("load_text_encoder", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.transformer_path = transformer_dir(model_path)
        self.transformer = None
        self.text_pipeline = None
        self.vae = None
        self.pipeline = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return

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
            model_path=self.transformer_path, config=self.config, device=self.device
        )

    # --------------------------------------------------- DiffletPipeline API

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"Qwen-Image {action} requires transformer/config.json under {self.model_path!r}"
            )
        return self.transformer

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        self._require_transformer("compile").compile(compiled_model_path, debug=debug)

    def load(
        self,
        compiled_model_path: str,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup: bool = False,
    ) -> None:
        self._require_transformer("load").load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )
        self._load_companions()

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        if self.transformer is None:
            return False
        return self.transformer.has_compiled_artifacts(compiled_model_path)

    # ------------------------------------------------ encoder / VAE / loop

    def _load_companions(self) -> None:
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

    def forward_dit(self, bundle: QwenImageDiTInputBundle):
        transformer = self._require_transformer("forward")
        inputs = tuple(
            t.to(self.device) if isinstance(t, torch.Tensor) else t
            for t in bundle.as_model_inputs()
        )
        return transformer(*inputs)

    # ---------------------------------------------------------- generation

    def generate(
        self,
        prompt: str | list[str],
        *,
        num_inference_steps: int = 50,
        seed: int | None = None,
        guidance_scale: float | None = None,
        output_type: str = "pil",
    ):
        """Text → image on the device: encode, denoise through the Difflet DiT, decode."""
        if self.pipeline is None:
            raise RuntimeError("application is not loaded; call load() first")
        encoder_hidden_states, encoder_hidden_states_mask = self.encode_prompt(prompt)
        batch = int(encoder_hidden_states.shape[0])
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = self.pipeline.prepare_latents(
            batch_size=batch, dtype=self.dtype, device=self.device, generator=generator
        )
        out = self.pipeline(
            latents=latents,
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

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], QwenImageDiTInputBundle):
            return self.forward_dit(args[0])
        if "prompt" in kwargs and not args:
            return self.generate(**kwargs)
        if kwargs and not args:
            return self.forward_dit(QwenImageDiTInputBundle(**kwargs))
        return self._require_transformer("forward")(*args, **kwargs)


__all__ = ["CudaQwenImageApplication"]
