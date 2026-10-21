"""Flux on the GPU flow: Difflet's DiT inside diffusers' ``FluxPipeline``, one process.

The DiT is ``CudaFluxTransformerApplication`` (the neutral ``NeuronFluxTransformer2DModel`` on
``difflet.ops``), wrapped in ``FluxDiTAdapter`` so that diffusers' own ``FluxPipeline`` drives
CLIP, T5, the scheduler, the DiT and the VAE on one device. Only the DiT is Difflet code.

``__call__`` accepts a ``FluxDiTInputBundle`` or the six DiT tensors as keywords (parity
scripts), and ``prompt=`` for the whole text-to-image path.
"""

from __future__ import annotations

from typing import Any

import torch

from difflet.backends.gpu.application import GpuApplication
from difflet.models.flux.contract import FluxDiTInputBundle
from difflet.models.flux.dit_config import FluxDiTConfig
from difflet.models.flux.spec import MAX_SEQUENCE_LENGTH, SPEC

DEFAULT_STEPS = SPEC.default_steps
DEFAULT_GUIDANCE_SCALE = SPEC.default_guidance_scale


class CudaFluxApplication(GpuApplication):
    spec = SPEC
    input_bundle_type = FluxDiTInputBundle

    def _configure(self, kwargs: dict[str, Any]) -> None:
        self.load_text_encoders = bool(kwargs.get("load_text_encoders", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.local_files_only = bool(kwargs.get("local_files_only", True))
        self.adapter = None
        self.pipe = None

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.flux.transformer import (
            CudaFluxTransformerApplication,
            FluxDiTAdapter,
        )

        self.config = FluxDiTConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            tp_degree=int(self.parallel.tp_degree),
            torch_dtype=self.dtype,
        )
        self.transformer = CudaFluxTransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
            fuse_proj_out=bool(self.kwargs.get("fuse_proj_out", False)),
        )
        self.adapter = FluxDiTAdapter(self.transformer)

    # ------------------------------------------------ encoders / VAE / loop

    def load_companions(self) -> None:
        """CLIP, T5, tokenizers, scheduler and VAE from the snapshot, on the device, around
        the adapter.

        ``from_pretrained`` checks a passed component against ``model_index.json``'s class
        (``FluxTransformer2DModel``, a diffusers ``ModelMixin``) and refuses the adapter, so
        the transformer is skipped there (``transformer=None``) and the pipeline is rebuilt
        with the adapter through the constructor, which takes any module. The components are
        moved one by one: ``pipe.to(device)`` would also move the adapter's DiT.
        """
        from diffusers import FluxPipeline

        skipped: dict[str, None] = {"transformer": None}
        if not self.load_text_encoders:
            skipped.update(text_encoder=None, text_encoder_2=None)
        if not self.load_vae:
            skipped["vae"] = None
        base = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            **skipped,
        )
        for name in ("text_encoder", "text_encoder_2", "vae"):
            module = getattr(base, name, None)
            if module is not None:
                module.to(self.device)
        self.pipe = FluxPipeline(**{**base.components, "transformer": self.adapter})

    # ------------------------------------------------------------- DiT API

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        cfg = self._require_transformer("input contract").dit_config
        batch = 1
        image_seq_len = int(cfg.image_seq_len)
        return {
            "hidden_states": {
                "shape": (batch, image_seq_len, int(cfg.in_channels)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states": {
                "shape": (batch, MAX_SEQUENCE_LENGTH, int(cfg.joint_attention_dim)),
                "dtype": self.dtype,
            },
            "pooled_projections": {
                "shape": (batch, int(cfg.pooled_projection_dim)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch,), "dtype": self.dtype},
            # schnell (guidance_embeds=False) takes an empty guidance tensor.
            "guidance": {"shape": (batch,) if cfg.guidance_embeds else (0,), "dtype": self.dtype},
            "image_rotary_emb": {
                "shape": (MAX_SEQUENCE_LENGTH + image_seq_len, int(cfg.attention_head_dim), 2),
                "dtype": self.dtype,
            },
        }

    # ---------------------------------------------------------- generation

    def generate(
        self,
        prompt: str | list[str],
        *,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: str = "pil",
    ):
        """Text → image on the device through ``FluxPipeline`` with the Difflet DiT.

        ``latents`` is packed ``(B, S, 64)`` initial noise used verbatim (the end-to-end
        oracle feeds the same tensor to stock diffusers); otherwise ``seed`` draws it.
        ``guidance_scale`` is FLUX.1-dev's distilled guidance (one pass, no CFG); for
        schnell (``guidance_embeds=False``) FluxPipeline does not pass it to the DiT.

        Returns the images for ``"pil"``/``"np"``, and the ``FluxPipelineOutput`` for
        ``"pt"``/``"latent"``.
        """
        if self.pipe is None:
            raise RuntimeError("application is not loaded; call load() first")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            height=int(height or self.shape["height"]),
            width=int(width or self.shape["width"]),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            latents=latents,
            output_type=output_type,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        if output_type in {"pt", "latent"}:
            return out
        return out.images


__all__ = ["CudaFluxApplication", "DEFAULT_GUIDANCE_SCALE", "DEFAULT_STEPS"]
