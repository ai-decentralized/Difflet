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
    cfg_parallel_enabled: bool = False,
    sp_enabled: bool = False,
    compile_shapes=None,
):
    from difflet.backends.trainium.wan.backbone import WanBackboneInferenceConfig

    transformer_path = os.path.join(model_path, subfolder)
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    extra = {}
    if compile_shapes:
        extra["compile_shapes"] = compile_shapes
    return WanBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        context_parallel_enabled=context_parallel_enabled,
        cp_mode=cp_mode,
        cfg_parallel_enabled=cfg_parallel_enabled,
        sp_enabled=sp_enabled,
        **extra,
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
    compile_shapes=None,
):
    from difflet.backends.trainium.wan.vae import WanVAEDecoderInferenceConfig

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        # VAE is a single-core stage; pin LNC=1 so the NEFF matches the 1-core
        # runtime. Without this it inherits the trn2 platform default (LNC=2) and
        # nrt_load fails: "compiled with --lnc=2" vs runtime NEURON_LOGICAL_NC_CONFIG=1.
        logical_nc_config=1,
    )
    extra = {}
    if compile_shapes:
        extra["compile_shapes"] = compile_shapes
    return WanVAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
        **extra,
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
        # CFG parallel stacks [uncond, cond] into batch=2 before scattering one
        # branch to each data-parallel rank, so the transformer compiles at
        # batch=2 while the rest of the components stay at the base batch.
        cfg_parallel_enabled = bool(getattr(parallel, "cfg_parallel_enabled", False))
        backbone_batch_size = 2 if cfg_parallel_enabled else batch_size
        height = int(shape.get("height") or 480)
        width = int(shape.get("width") or 832)
        num_frames = int(shape.get("num_frames") or 9)
        # Optional bucket shape set (PIXEL h/w/frames): one artifact with K
        # backbone/VAE NEFFs sharing one weight copy. Largest shape becomes the
        # single-shape defaults; frames are converted to latent counts for the
        # backbone configs (their num_frames semantics) and stay pixel for VAE.
        self.compile_shapes = None
        backbone_compile_shapes = None
        raw_shapes = kwargs.get("shapes")
        if raw_shapes:
            from difflet.backends.trainium.core.bucketing import canonicalize_shapes

            self.compile_shapes = canonicalize_shapes(raw_shapes)
            height, width, num_frames = (
                self.compile_shapes[0][0],
                self.compile_shapes[0][1],
                int(self.compile_shapes[0][2]),
            )
            self.shape = {"height": height, "width": width, "num_frames": num_frames}
            backbone_compile_shapes = tuple(
                (h, w, _latent_num_frames(f)) for h, w, f in self.compile_shapes
            )
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
                batch_size=backbone_batch_size,
                context_parallel_enabled=parallel.cp_degree > 1,
                cp_mode=parallel.cp_mode,
                cfg_parallel_enabled=cfg_parallel_enabled,
                sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
                compile_shapes=backbone_compile_shapes,
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
                batch_size=backbone_batch_size,
                subfolder="transformer_2",
                context_parallel_enabled=parallel.cp_degree > 1,
                cp_mode=parallel.cp_mode,
                cfg_parallel_enabled=cfg_parallel_enabled,
                sp_enabled=bool(getattr(parallel, "sp_enabled", False)),
                compile_shapes=backbone_compile_shapes,
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
                compile_shapes=self.compile_shapes,
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
            # Probe-free modes: runtime-only, never part of the artifact identity.
            teacache_cadence=self.kwargs.get("teacache_cadence"),
            teacache_online_delta_alpha=self.kwargs.get("teacache_online_delta_alpha"),
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

    # No _component_load_rank_range override: every co-resident Wan component
    # (text encoder, transformer, transformer_2) declares world_size =
    # parallel.world_size since cc9316c, and the VAE (world_size=1) runs in its
    # own stage subprocess. The former override's "clamp a smaller world to a
    # sub-range" branch described a text-encoder wiring that never shipped and
    # was dead code from the day it landed; the base class now rejects mixed
    # worlds outright (world_check.py).

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
