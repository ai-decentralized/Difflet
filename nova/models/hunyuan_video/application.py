"""HunyuanVideo Trainium application skeleton.

M3 starts with registration, shape/cache plumbing, and explicit scope guards.
The transformer/text/VAE components land after the dual-stream attention
reference path is covered by CPU tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from nova.backends.trainium.core.config import NeuronConfig
from nova.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)
from nova.utils.diffusers_adapter import load_diffusers_config


@dataclass(frozen=True)
class HunyuanVideoDiTInputBundle:
    """Host-side contract for one HunyuanVideo DiT call.

    M3 v0 keeps Llama3, CLIP, scheduler setup, and VAE decode outside the
    Trainium graph. The Trainium boundary is exactly this tuple.
    """

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    pooled_projections: torch.Tensor
    guidance: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            self.pooled_projections,
            self.guidance,
        )


@dataclass(frozen=True)
class HunyuanVideo15DiTInputBundle:
    """Host-side contract for one HunyuanVideo 1.5 DiT call.

    HunyuanVideo 1.5 uses Qwen2.5-VL embeddings, ByT5 glyph embeddings, and
    image-semantic embeddings in addition to the latent tensor.
    """

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    timestep_r: torch.Tensor
    encoder_hidden_states_2: torch.Tensor
    encoder_attention_mask_2: torch.Tensor
    image_embeds: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            self.timestep_r,
            self.encoder_hidden_states_2,
            self.encoder_attention_mask_2,
            self.image_embeds,
        )


def validate_hunyuan_video_dit_inputs(
    bundle: HunyuanVideoDiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M3 v0 embedding/latent contract before Trainium dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 256))
    latent_shape = (
        batch_size,
        int(config.in_channels),
        int(config.latent_frames),
        int(config.latent_height),
        int(config.latent_width),
    )
    expected = {
        "hidden_states": (latent_shape, dtype),
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.text_embed_dim)),
            dtype,
        ),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.int64),
        "pooled_projections": ((batch_size, int(config.pooled_projection_dim)), dtype),
        "guidance": ((batch_size,), dtype),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "pooled_projections": bundle.pooled_projections,
        "guidance": bundle.guidance,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"HunyuanVideo DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"HunyuanVideo DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def validate_hunyuan_video15_dit_inputs(
    bundle: HunyuanVideo15DiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the HunyuanVideo 1.5 embedding/latent contract before dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 1000))
    text_seq_len_2 = int(getattr(config, "text_seq_len_2", 256))
    image_seq_len = int(getattr(config, "image_seq_len", 729))
    latent_shape = (
        batch_size,
        int(config.in_channels),
        int(config.latent_frames),
        int(config.latent_height),
        int(config.latent_width),
    )
    expected = {
        "hidden_states": (latent_shape, dtype),
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.text_embed_dim)),
            dtype,
        ),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.int64),
        "timestep_r": ((batch_size,), dtype),
        "encoder_hidden_states_2": (
            (batch_size, text_seq_len_2, int(config.text_embed_2_dim)),
            dtype,
        ),
        "encoder_attention_mask_2": ((batch_size, text_seq_len_2), torch.int64),
        "image_embeds": ((batch_size, image_seq_len, int(config.image_embed_dim)), dtype),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "timestep_r": bundle.timestep_r,
        "encoder_hidden_states_2": bundle.encoder_hidden_states_2,
        "encoder_attention_mask_2": bundle.encoder_attention_mask_2,
        "image_embeds": bundle.image_embeds,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"HunyuanVideo 1.5 DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"HunyuanVideo 1.5 DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def create_hunyuan_video_backbone_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = 256,
    batch_size: int = 1,
):
    from nova.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
    )

    transformer_path = os.path.join(model_path, "transformer")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return HunyuanVideoBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
    )


def create_hunyuan_video15_backbone_config(
    *,
    transformer_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = 1000,
    text_seq_len_2: int = 256,
    image_seq_len: int = 729,
    batch_size: int = 1,
):
    from nova.backends.trainium.hunyuan_video.backbone15 import (
        HunyuanVideo15BackboneInferenceConfig,
    )

    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return HunyuanVideo15BackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        text_seq_len_2=text_seq_len_2,
        image_seq_len=image_seq_len,
    )


def create_hunyuan_video_vae_decoder_config(
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
    from nova.backends.trainium.hunyuan_video.vae import HunyuanVideoVAEDecoderInferenceConfig

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return HunyuanVideoVAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
    )


def create_hunyuan_video15_vae_decoder_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    batch_size: int = 1,
    tile_sample_min_height: int = 256,
    tile_sample_min_width: int = 256,
    tile_overlap_factor: float = 0.25,
):
    from nova.backends.trainium.hunyuan_video.vae15 import (
        HunyuanVideo15VAEDecoderInferenceConfig,
    )

    vae_path = os.path.join(model_path, "vae")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return HunyuanVideo15VAEDecoderInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(vae_path),
        height=height,
        width=width,
        num_frames=num_frames,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_overlap_factor=tile_overlap_factor,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported HunyuanVideo dtype: {dtype!r}")


class NeuronHunyuanVideoApplication(MultiComponentApplication):
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
        self.model_version = str(kwargs.get("model_version", "1.0"))
        default_shape = (
            {"height": 480, "width": 848, "num_frames": 121}
            if self.model_version == "1.5"
            else {"height": 320, "width": 512, "num_frames": 61}
        )
        self.shape = {
            "height": int(shape.get("height") or default_shape["height"]),
            "width": int(shape.get("width") or default_shape["width"]),
            "num_frames": int(shape.get("num_frames") or default_shape["num_frames"]),
        }
        self.kwargs = kwargs
        transformer_subfolder = str(kwargs.get("transformer_subfolder", "transformer"))
        self.transformer_path = os.path.join(model_path, transformer_subfolder)
        self.vae_decoder_path = os.path.join(model_path, "vae")
        self.transformer = None
        self.vae_decoder = None
        self.pipeline = None
        self.text_seq_len = int(kwargs.get("text_seq_len", 256))
        self.text_seq_len_2 = int(kwargs.get("text_seq_len_2", 256))
        self.image_seq_len = int(kwargs.get("image_seq_len", 729))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.transformer_runtime = str(
            kwargs.get(
                "transformer_runtime",
                os.environ.get("NOVA_HUNYUAN15_TRANSFORMER_RUNTIME", "monolithic"),
            )
        )

        enable_transformer = bool(kwargs.get("enable_transformer", True))
        enable_vae_decoder = bool(kwargs.get("enable_vae_decoder", False))
        transformer_config_path = os.path.join(self.transformer_path, "config.json")
        self.transformer_config = None
        if os.path.exists(transformer_config_path):
            self.transformer_config = load_diffusers_config(self.transformer_path)
        if self.model_version == "1.5" and enable_transformer and os.path.exists(transformer_config_path):
            from nova.backends.trainium.hunyuan_video.backbone15 import (
                NeuronHunyuanVideo15BackboneApplication,
            )

            config = create_hunyuan_video15_backbone_config(
                transformer_path=self.transformer_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=int(kwargs.get("text_seq_len", 1000)),
                text_seq_len_2=self.text_seq_len_2,
                image_seq_len=self.image_seq_len,
                batch_size=self.batch_size,
            )
            if self.transformer_runtime == "segmented":
                from nova.backends.trainium.hunyuan_video.segmented15 import (
                    DEFAULT_ATTENTION_COMPILER_ARGS,
                    DEFAULT_BLOCK_COMPILER_ARGS,
                    HunyuanVideo15SegmentedTransformerApplication,
                )

                self.transformer = HunyuanVideo15SegmentedTransformerApplication(
                    model_path=self.transformer_path,
                    config=config,
                    query_tile_size=int(kwargs.get("segmented_query_tile_size", 2051)),
                    key_tile_size=int(kwargs.get("segmented_key_tile_size", 2051)),
                    block_load_mode=str(kwargs.get("segmented_block_load_mode", "all")),
                    block_compiler_args=str(
                        kwargs.get("segmented_block_compiler_args", DEFAULT_BLOCK_COMPILER_ARGS)
                    ),
                    attention_compiler_args=str(
                        kwargs.get(
                            "segmented_attention_compiler_args",
                            DEFAULT_ATTENTION_COMPILER_ARGS,
                        )
                    ),
                )
            elif self.transformer_runtime == "monolithic":
                self.transformer = NeuronHunyuanVideo15BackboneApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
            else:
                raise ValueError(
                    "HunyuanVideo 1.5 transformer_runtime must be 'monolithic' or "
                    f"'segmented', got {self.transformer_runtime!r}."
                )
        elif enable_transformer and os.path.exists(transformer_config_path):
            from nova.backends.trainium.hunyuan_video.backbone import (
                NeuronHunyuanVideoBackboneApplication,
            )

            config = create_hunyuan_video_backbone_config(
                model_path=model_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=self.text_seq_len,
                batch_size=self.batch_size,
            )
            self.transformer = NeuronHunyuanVideoBackboneApplication(
                model_path=self.transformer_path,
                config=config,
            )

        vae_config_path = os.path.join(self.vae_decoder_path, "config.json")
        if enable_vae_decoder and os.path.exists(vae_config_path):
            if self.model_version == "1.5":
                from nova.backends.trainium.hunyuan_video.vae15 import (
                    NeuronHunyuanVideo15VAEDecoderApplication,
                )

                vae_config = create_hunyuan_video15_vae_decoder_config(
                    model_path=model_path,
                    world_size=1,
                    tp_degree=1,
                    dtype=self.dtype,
                    height=self.shape["height"],
                    width=self.shape["width"],
                    num_frames=self.shape["num_frames"],
                    batch_size=self.batch_size,
                    tile_sample_min_height=int(kwargs.get("vae_tile_sample_min_height", 256)),
                    tile_sample_min_width=int(kwargs.get("vae_tile_sample_min_width", 256)),
                    tile_overlap_factor=float(kwargs.get("vae_tile_overlap_factor", 0.25)),
                )
                self.vae_decoder = NeuronHunyuanVideo15VAEDecoderApplication(
                    model_path=self.vae_decoder_path,
                    config=vae_config,
                )
            else:
                from nova.backends.trainium.hunyuan_video.vae import (
                    NeuronHunyuanVideoVAEDecoderApplication,
                )

                vae_config = create_hunyuan_video_vae_decoder_config(
                    model_path=model_path,
                    world_size=parallel.tp_degree,
                    tp_degree=1,
                    dtype=self.dtype,
                    height=self.shape["height"],
                    width=self.shape["width"],
                    num_frames=self.shape["num_frames"],
                    batch_size=self.batch_size,
                )
                self.vae_decoder = NeuronHunyuanVideoVAEDecoderApplication(
                    model_path=self.vae_decoder_path,
                    config=vae_config,
                )
        from nova.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

        self.pipeline = HunyuanVideoOrchestrator(
            model_path=model_path,
            transformer=self if self.transformer is not None else None,
            vae=self.vae_decoder,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
        )

    def components(self) -> list[ComponentSpec]:
        components: list[ComponentSpec] = []
        if self.transformer is not None:
            if hasattr(self.transformer, "component_specs"):
                components.extend(self.transformer.component_specs(prefix="transformer"))
            else:
                components.append(ComponentSpec("transformer", self.transformer))
        if self.vae_decoder is not None:
            if hasattr(self.vae_decoder, "component_specs"):
                components.extend(self.vae_decoder.component_specs(prefix="vae_decoder"))
            else:
                components.append(ComponentSpec("vae_decoder", self.vae_decoder))
        return components

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
        select=None,
    ) -> None:
        if (
            self.model_version == "1.5"
            and self.transformer is not None
            and getattr(self.transformer, "block_load_mode", None) == "process"
        ):
            self.transformer.set_compiled_model_path(str(compiled_model_path))
            if self.vae_decoder is not None:
                self.vae_decoder.load(
                    os.path.join(str(compiled_model_path), "vae_decoder"),
                    start_rank_id=0 if start_rank_id is not None else None,
                    local_ranks_size=1,
                    skip_warmup=skip_warmup,
                )
            return
        return super().load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
            select=select,
        )

    def no_components_message(self, action: str) -> str:
        if self.model_version == "1.5":
            return (
                "HunyuanVideo 1.5 compile/load requires a transformer variant "
                "config under the selected transformer_subfolder, for example "
                "'transformer' for community Diffusers repos or 'transformer/480p_t2v' "
                "for the Tencent original layout."
            )
        if action == "compile":
            return (
                "HunyuanVideo compile requires transformer/config.json. "
                "The current app has no active compile component."
            )
        if action == "load":
            return "HunyuanVideo load requires compiled component artifacts"
        return super().no_components_message(action)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo DiT contract requires an active transformer.")
        config = self.transformer.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        if self.model_version == "1.5":
            text_seq_len = int(getattr(config, "text_seq_len", 1000))
            text_seq_len_2 = int(getattr(config, "text_seq_len_2", 256))
            image_seq_len = int(getattr(config, "image_seq_len", 729))
            return {
                "hidden_states": {
                    "shape": (
                        batch_size,
                        int(config.in_channels),
                        int(config.latent_frames),
                        int(config.latent_height),
                        int(config.latent_width),
                    ),
                    "dtype": self.dtype,
                },
                "timestep": {"shape": (batch_size,), "dtype": self.dtype},
                "encoder_hidden_states": {
                    "shape": (batch_size, text_seq_len, int(config.text_embed_dim)),
                    "dtype": self.dtype,
                },
                "encoder_attention_mask": {
                    "shape": (batch_size, text_seq_len),
                    "dtype": torch.int64,
                },
                "timestep_r": {"shape": (batch_size,), "dtype": self.dtype},
                "encoder_hidden_states_2": {
                    "shape": (batch_size, text_seq_len_2, int(config.text_embed_2_dim)),
                    "dtype": self.dtype,
                },
                "encoder_attention_mask_2": {
                    "shape": (batch_size, text_seq_len_2),
                    "dtype": torch.int64,
                },
                "image_embeds": {
                    "shape": (batch_size, image_seq_len, int(config.image_embed_dim)),
                    "dtype": self.dtype,
                },
            }
        text_seq_len = int(getattr(config, "text_seq_len", 256))
        return {
            "hidden_states": {
                "shape": (
                    batch_size,
                    int(config.in_channels),
                    int(config.latent_frames),
                    int(config.latent_height),
                    int(config.latent_width),
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch_size,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch_size, text_seq_len, int(config.text_embed_dim)),
                "dtype": self.dtype,
            },
            "encoder_attention_mask": {
                "shape": (batch_size, text_seq_len),
                "dtype": torch.int64,
            },
            "pooled_projections": {
                "shape": (batch_size, int(config.pooled_projection_dim)),
                "dtype": self.dtype,
            },
            "guidance": {"shape": (batch_size,), "dtype": self.dtype},
        }

    def forward_dit(self, bundle: HunyuanVideoDiTInputBundle):
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo forward_dit requires an active transformer.")
        if isinstance(bundle, HunyuanVideo15DiTInputBundle):
            validate_hunyuan_video15_dit_inputs(
                bundle,
                config=self.transformer.config,
                dtype=self.dtype,
            )
            return self.transformer(*bundle.as_model_inputs())
        validate_hunyuan_video_dit_inputs(
            bundle,
            config=self.transformer.config,
            dtype=self.dtype,
        )
        return self.transformer(*bundle.as_model_inputs())

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], (HunyuanVideoDiTInputBundle, HunyuanVideo15DiTInputBundle)):
            return self.forward_dit(args[0])
        if self.model_version == "1.5":
            direct_keys_15 = {
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
                "encoder_attention_mask",
                "timestep_r",
                "encoder_hidden_states_2",
                "encoder_attention_mask_2",
                "image_embeds",
            }
            if not args and direct_keys_15.issubset(kwargs):
                bundle = HunyuanVideo15DiTInputBundle(**kwargs)
                return self.forward_dit(bundle)
        direct_keys = {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "pooled_projections",
            "guidance",
        }
        if not args and direct_keys.issubset(kwargs):
            bundle = HunyuanVideoDiTInputBundle(**kwargs)
            return self.forward_dit(bundle)
        if self.transformer is not None and args:
            return self.transformer(*args, **kwargs)
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        del args, kwargs
        raise NotImplementedError("HunyuanVideo end-to-end inference is not implemented yet")
