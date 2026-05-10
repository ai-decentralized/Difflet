"""Wan Trainium application."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from nova.backends.trainium.core.config import NeuronConfig
from nova.utils.diffusers_adapter import load_diffusers_config


def create_wan_backbone_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int = 1,
    subfolder: str = "transformer",
):
    from nova.backends.trainium.wan.backbone import WanBackboneInferenceConfig

    transformer_path = os.path.join(model_path, subfolder)
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        # W3 starts with compile validation. Real weight sharding/loading lands
        # after checkpoint conversion is verified.
        skip_sharding=True,
    )
    return WanBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
    )


def create_wan_text_encoder_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    text_seq_len: int = 512,
    batch_size: int = 1,
):
    from nova.backends.trainium.wan.text_encoder import WanTextEncoderInferenceConfig

    text_encoder_path = os.path.join(model_path, "text_encoder")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return WanTextEncoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(text_encoder_path),
        text_seq_len=text_seq_len,
    )


def create_wan_vae_decoder_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int = 1,
):
    from nova.backends.trainium.wan.vae import WanVAEDecoderInferenceConfig

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return WanVAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Wan dtype: {dtype!r}")


def _latent_num_frames(num_frames: int) -> int:
    return (int(num_frames) - 1) // 4 + 1


class NeuronWanApplication(nn.Module):
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
        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = shape
        self.kwargs = kwargs
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer_2_path = os.path.join(model_path, "transformer_2")
        self.text_encoder_path = os.path.join(model_path, "text_encoder")
        self.vae_decoder_path = os.path.join(model_path, "vae")
        self.transformer = None
        self.transformer_2 = None
        self.text_encoder = None
        self.vae_decoder = None
        enable_text_encoder = bool(kwargs.get("enable_text_encoder", True))
        enable_transformer = bool(kwargs.get("enable_transformer", True))
        enable_transformer_2 = bool(kwargs.get("enable_transformer_2", True))
        enable_vae_decoder = bool(kwargs.get("enable_vae_decoder", True))

        text_seq_len = int(kwargs.get("text_seq_len", 512))
        batch_size = int(kwargs.get("batch_size", 1))
        height = int(shape.get("height") or 480)
        width = int(shape.get("width") or 832)
        num_frames = int(shape.get("num_frames") or 9)
        latent_num_frames = _latent_num_frames(num_frames)

        if enable_transformer and os.path.exists(os.path.join(self.transformer_path, "config.json")):
            from nova.backends.trainium.wan.backbone import NeuronWanBackboneApplication

            config = create_wan_backbone_config(
                model_path=model_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=height,
                width=width,
                num_frames=latent_num_frames,
                batch_size=batch_size,
            )
            self.transformer = NeuronWanBackboneApplication(
                model_path=self.transformer_path,
                config=config,
            )

        if enable_transformer_2 and os.path.exists(os.path.join(self.transformer_2_path, "config.json")):
            from nova.backends.trainium.wan.backbone import NeuronWanBackboneApplication

            config = create_wan_backbone_config(
                model_path=model_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=height,
                width=width,
                num_frames=latent_num_frames,
                batch_size=batch_size,
                subfolder="transformer_2",
            )
            self.transformer_2 = NeuronWanBackboneApplication(
                model_path=self.transformer_2_path,
                config=config,
            )

        if enable_text_encoder and os.path.exists(os.path.join(self.text_encoder_path, "config.json")):
            from nova.backends.trainium.wan.text_encoder import (
                NeuronWanTextEncoderApplication,
            )

            te_config = create_wan_text_encoder_config(
                model_path=model_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                text_seq_len=text_seq_len,
                batch_size=batch_size,
            )
            self.text_encoder = NeuronWanTextEncoderApplication(
                model_path=self.text_encoder_path,
                config=te_config,
            )

        if enable_vae_decoder and os.path.exists(os.path.join(self.vae_decoder_path, "config.json")):
            from nova.backends.trainium.wan.vae import NeuronWanVAEDecoderApplication

            vae_config = create_wan_vae_decoder_config(
                model_path=model_path,
                world_size=1,
                tp_degree=1,
                dtype=self.dtype,
                height=height,
                width=width,
                num_frames=num_frames,
                batch_size=batch_size,
            )
            self.vae_decoder = NeuronWanVAEDecoderApplication(
                model_path=self.vae_decoder_path,
                config=vae_config,
            )

        from nova.models.wan.pipeline import WanOrchestrator

        self.pipeline = WanOrchestrator(
            model_path=model_path,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            transformer_2=self.transformer_2,
            vae_decoder=self.vae_decoder,
            dtype=self.dtype,
            height=height,
            width=width,
            num_frames=num_frames,
            max_text_length=text_seq_len,
        )

    def _components(self):
        """Yield ``(name, component)`` for every active sub-app, in compile order.

        UMT5 first so its communicator establishes before the transformer
        (Flux M1.4 lesson, cclogs/05 §9.1). VAE compiles last as an
        independent one-core component.
        """
        if self.text_encoder is not None:
            yield "text_encoder", self.text_encoder
        if self.transformer is not None:
            yield "transformer", self.transformer
        if self.transformer_2 is not None:
            yield "transformer_2", self.transformer_2
        if self.vae_decoder is not None:
            yield "vae_decoder", self.vae_decoder

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        components = list(self._components())
        if not components:
            raise NotImplementedError(
                "Wan compile requires diffusers transformer/ or text_encoder/ "
                "config.json. The current app is in W2 skeleton mode for this "
                "model path."
            )
        for name, component in components:
            component.compile(os.path.join(compiled_model_path, name), debug=debug)

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        components = list(self._components())
        if not components:
            return True
        for name, _component in components:
            component_path = os.path.join(compiled_model_path, name)
            if not os.path.exists(os.path.join(component_path, "model.pt")):
                return False
            if not os.path.exists(os.path.join(component_path, "neuron_config.json")):
                return False
        return True

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
    ) -> None:
        components = list(self._components())
        if not components:
            raise NotImplementedError(
                "Wan load requires compiled artifacts. The current app is in "
                "W2 skeleton mode for this model path."
            )
        for name, component in components:
            component_start_rank_id, component_local_ranks_size = self._component_load_range(
                component,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
            )
            component.load(
                os.path.join(compiled_model_path, name),
                start_rank_id=component_start_rank_id,
                local_ranks_size=component_local_ranks_size,
                skip_warmup=skip_warmup,
            )

    @staticmethod
    def _component_load_range(
        component,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        """Clamp single-core components when the parent Wan app uses TP>1."""
        config = getattr(component, "config", None)
        neuron_config = getattr(config, "neuron_config", None)
        world_size = getattr(neuron_config, "world_size", None)
        if world_size == 1:
            return 0 if start_rank_id is not None else None, 1
        return start_rank_id, local_ranks_size

    def __call__(self, *args: Any, **kwargs: Any):
        if self.transformer is not None and len(args) >= 3:
            return self.transformer(*args, **kwargs)
        if self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        batch = int(kwargs.get("batch_size", 1))
        channels = int(kwargs.get("channels", 16))
        frames = int(kwargs.get("num_latent_frames", 1))
        height = int(kwargs.get("latent_height", 1))
        width = int(kwargs.get("latent_width", 1))
        dtype = self.dtype if isinstance(self.dtype, torch.dtype) else torch.bfloat16
        return torch.zeros((batch, channels, frames, height, width), dtype=dtype)
