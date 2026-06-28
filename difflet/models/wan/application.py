"""Wan Trainium application."""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.backends.trainium.core.config import NeuronConfig
from difflet.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)
from difflet.utils.diffusers_adapter import load_diffusers_config


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
    context_parallel_enabled: bool = False,
    cp_mode: str = "gather_kv",
):
    from difflet.backends.trainium.wan.backbone import WanBackboneInferenceConfig

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
        context_parallel_enabled=context_parallel_enabled,
        cp_mode=cp_mode,
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
    from difflet.backends.trainium.wan.text_encoder import WanTextEncoderInferenceConfig

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
    from difflet.backends.trainium.wan.vae import WanVAEDecoderInferenceConfig

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
        # VAE is a single-core stage; pin LNC=1 so the NEFF matches the 1-core
        # runtime. Without this it inherits the trn2 platform default (LNC=2) and
        # nrt_load fails: "compiled with --lnc=2" vs runtime NEURON_LOGICAL_NC_CONFIG=1.
        logical_nc_config=1,
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


class NeuronWanApplication(MultiComponentApplication):
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
            from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication

            config = create_wan_backbone_config(
                model_path=model_path,
                world_size=parallel.world_size,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=height,
                width=width,
                num_frames=latent_num_frames,
                batch_size=batch_size,
                context_parallel_enabled=parallel.cp_degree > 1,
                cp_mode=parallel.cp_mode,
            )
            self.transformer = NeuronWanBackboneApplication(
                model_path=self.transformer_path,
                config=config,
            )

        if enable_transformer_2 and os.path.exists(os.path.join(self.transformer_2_path, "config.json")):
            from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication

            config = create_wan_backbone_config(
                model_path=model_path,
                world_size=parallel.world_size,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=height,
                width=width,
                num_frames=latent_num_frames,
                batch_size=batch_size,
                subfolder="transformer_2",
                context_parallel_enabled=parallel.cp_degree > 1,
                cp_mode=parallel.cp_mode,
            )
            self.transformer_2 = NeuronWanBackboneApplication(
                model_path=self.transformer_2_path,
                config=config,
            )

        if enable_text_encoder and os.path.exists(os.path.join(self.text_encoder_path, "config.json")):
            from difflet.backends.trainium.wan.text_encoder import (
                NeuronWanTextEncoderApplication,
            )

            te_config = create_wan_text_encoder_config(
                model_path=model_path,
                world_size=parallel.world_size,
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
            from difflet.backends.trainium.wan.vae import NeuronWanVAEDecoderApplication

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

        from difflet.models.wan.pipeline import WanOrchestrator

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
            teacache_calibration_path=self.kwargs.get("teacache_calibration_path"),
        )

    def components(self) -> list[ComponentSpec]:
        """Yield ``(name, component)`` for every active sub-app, in compile order.

        UMT5 first so its communicator establishes before the transformer
        (Flux M1.4 lesson, cclogs/05 §9.1). VAE compiles last as an
        independent one-core component.
        """
        components: list[ComponentSpec] = []
        if self.text_encoder is not None:
            components.append(ComponentSpec("text_encoder", self.text_encoder))
        if self.transformer is not None:
            components.append(ComponentSpec("transformer", self.transformer))
        if self.transformer_2 is not None:
            components.append(ComponentSpec("transformer_2", self.transformer_2))
        if self.vae_decoder is not None:
            components.append(ComponentSpec("vae_decoder", self.vae_decoder))
        return components

    def no_components_message(self, action: str) -> str:
        if action == "compile":
            return (
                "Wan compile requires diffusers transformer/ or text_encoder/ "
                "config.json. The current app is in W2 skeleton mode for this "
                "model path."
            )
        if action == "load":
            return (
                "Wan load requires compiled artifacts. The current app is in "
                "W2 skeleton mode for this model path."
            )
        return super().no_components_message(action)

    @staticmethod
    def _component_load_rank_range(
        component,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        """Clamp each component to its own world_size.

        Text encoder uses tp-only (world_size=tp), transformer uses tp*cp.
        When the app-level local_ranks_size=tp*cp, text encoder must still
        load on only tp ranks or weight initialization crashes.
        """
        config = getattr(component, "config", None)
        neuron_config = getattr(config, "neuron_config", None)
        world_size = getattr(neuron_config, "world_size", None)
        if world_size == 1:
            return 0 if start_rank_id is not None else None, 1
        if world_size is not None and local_ranks_size is not None and world_size < local_ranks_size:
            return start_rank_id, world_size
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
