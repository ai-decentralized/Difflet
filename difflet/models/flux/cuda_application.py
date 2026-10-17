"""Flux on the CUDA backend: Difflet's DiT inside diffusers' ``FluxPipeline``, one process
(CUDA plan F5, decisions D5 and D8).

Mirrors ``CudaQwenImageApplication``'s outward contract — the shape ``DiffletPipeline``
calls reflectively. The DiT is ``CudaFluxTransformerApplication`` (the neutral
``NeuronFluxTransformer2DModel`` on ``difflet.ops``), wrapped in ``FluxDiTAdapter`` so that
diffusers' own ``FluxPipeline`` drives CLIP, T5, the scheduler, the DiT and the VAE on one
device. Only the DiT is Difflet code, which is the part under test.

Release 1 scope (plan R3): tp=1, one request shape, no TeaCache, no TAEF1, no SP, no
``torch.compile`` — each is refused by name with follow-up F5b, before any weight loads.

``__call__`` accepts a ``FluxDiTInputBundle`` or the six DiT tensors as keywords (parity
scripts), and ``prompt=`` for the whole text-to-image path.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.models.flux.contract import FluxDiTInputBundle, normalize_dtype
from difflet.models.flux.dit_config import FluxDiTConfig

DEFAULT_STEPS = 28  # FluxPipeline's default for FLUX.1-dev
DEFAULT_GUIDANCE_SCALE = 3.5  # FluxPipeline's default (distilled guidance, single pass)
MAX_SEQUENCE_LENGTH = 512  # T5 prompt length; also the DiT warm-up's text length

# Application kwargs that request a feature outside Release 1 (plan R3 → F5b). A key is
# refused only when it carries a request (not None / False): the CLI and DiffletPipeline
# pass several of them with their "off" value.
_F5B_KWARGS = ("taef1", "taef1_path", "shapes", "sp_enabled", "torch_compile")


def _refuse_f5b_kwargs(kwargs: dict[str, Any]) -> None:
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        if key in _F5B_KWARGS or key.startswith("teacache_"):
            raise NotImplementedError(
                f"{key} for Flux on the CUDA backend is follow-up F5b of the CUDA backend "
                "plan; Release 1 runs one shape at tp=1 without TeaCache, TAEF1, SP or "
                "torch.compile"
            )


class CudaFluxApplication(torch.nn.Module):
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
        _refuse_f5b_kwargs(kwargs)
        from difflet.backends.cuda.flux.transformer import (
            CudaFluxTransformerApplication,
            FluxDiTAdapter,
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
        self.load_text_encoders = bool(kwargs.get("load_text_encoders", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.local_files_only = bool(kwargs.get("local_files_only", True))
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.adapter = None
        self.pipe = None

        if not os.path.exists(os.path.join(self.transformer_path, "config.json")):
            return

        self.config = FluxDiTConfig.from_pretrained(
            self.transformer_path,
            height=self.shape["height"],
            width=self.shape["width"],
            tp_degree=int(parallel.tp_degree),
            torch_dtype=self.dtype,
        )
        self.transformer = CudaFluxTransformerApplication(
            model_path=self.transformer_path, config=self.config, device=self.device
        )
        self.adapter = FluxDiTAdapter(self.transformer)

    # --------------------------------------------------- DiffletPipeline API

    def _require_transformer(self, action: str):
        if self.transformer is None:
            raise NotImplementedError(
                f"Flux {action} requires transformer/config.json under {self.model_path!r}"
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

    # ------------------------------------------------ encoders / VAE / loop

    def _load_companions(self) -> None:
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

    def forward_dit(self, bundle: FluxDiTInputBundle):
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
        ``"pt"``/``"latent"`` (as ``CudaQwenImageApplication.generate`` does).
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

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], FluxDiTInputBundle):
            return self.forward_dit(args[0])
        if "prompt" in kwargs and not args:
            return self.generate(**kwargs)
        if kwargs and not args:
            return self.forward_dit(FluxDiTInputBundle(**kwargs))
        return self._require_transformer("forward")(*args, **kwargs)


__all__ = ["CudaFluxApplication", "DEFAULT_GUIDANCE_SCALE", "DEFAULT_STEPS"]
