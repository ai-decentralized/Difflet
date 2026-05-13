"""Trainium application wrapper for the HunyuanVideo 1.5 DiT backbone."""

from __future__ import annotations

import math
import os
from typing import List

import torch
import torch.nn as nn

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper


class _HunyuanVideo15NoMaskAttnProcessor:
    """Experimental HV1.5 attention processor for all-valid token compiles."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, *, query_chunk_size: int | None = None) -> None:
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if self.query_chunk_size is None or query.shape[1] <= self.query_chunk_size:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            chunks = []
            for query_chunk in torch.split(query, self.query_chunk_size, dim=1):
                chunks.append(
                    dispatch_attention_fn(
                        query_chunk,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=self._attention_backend,
                        parallel_config=self._parallel_config,
                    )
                )
            hidden_states = torch.cat(chunks, dim=1)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class _HunyuanVideo15NkiNoMaskAttnProcessor(_HunyuanVideo15NoMaskAttnProcessor):
    """Experimental no-mask processor using Nova's NKI attention kernel."""

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        from nova.ops import attention

        batch_size, query_len, heads, head_dim = query.shape
        key_len = key.shape[1]
        value_len = value.shape[1]
        query = query.permute(0, 2, 1, 3).reshape(batch_size * heads, query_len, head_dim)
        key = key.permute(0, 2, 1, 3).reshape(batch_size * heads, key_len, head_dim)
        value = value.permute(0, 2, 1, 3).reshape(batch_size * heads, value_len, head_dim)
        hidden_states = attention(
            query,
            key,
            value,
            scale=1 / math.sqrt(head_dim),
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        return hidden_states.reshape(batch_size, heads, query_len, head_dim).permute(0, 2, 1, 3)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        hidden_states = self._attention(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class _HunyuanVideo15ZeroAttnProcessor:
    """Experimental processor that removes block attention for capacity probes."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attn, attention_mask, image_rotary_emb
        context = None if encoder_hidden_states is None else torch.zeros_like(encoder_hidden_states)
        return torch.zeros_like(hidden_states), context


class _ZeroLikeModule(nn.Module):
    """Experimental replacement for FFN capacity probes."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class HunyuanVideo15BackboneInferenceConfig(InferenceConfig):
    """Inference config for the HunyuanVideo 1.5 transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = 32
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 1000
        if not hasattr(self, "text_seq_len_2"):
            self.text_seq_len_2 = 256
        if not hasattr(self, "image_seq_len"):
            self.image_seq_len = 729
        if not hasattr(self, "spatial_compression_ratio"):
            self.spatial_compression_ratio = 16
        if not hasattr(self, "temporal_compression_ratio"):
            self.temporal_compression_ratio = 4

    def get_required_attributes(self) -> List[str]:
        return [
            "in_channels",
            "out_channels",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "num_refiner_layers",
            "mlp_ratio",
            "patch_size",
            "patch_size_t",
            "qk_norm",
            "text_embed_dim",
            "text_embed_2_dim",
            "image_embed_dim",
            "rope_theta",
            "rope_axes_dim",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.spatial_compression_ratio)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.spatial_compression_ratio)

    @property
    def latent_frames(self) -> int:
        ratio = int(self.temporal_compression_ratio)
        return (int(self.num_frames) - 1) // ratio + 1

    def validate_config(self):
        super().validate_config()
        if isinstance(self.rope_axes_dim, list):
            self.rope_axes_dim = tuple(self.rope_axes_dim)
        if self.qk_norm != "rms_norm":
            raise NotImplementedError(
                "HunyuanVideo 1.5 currently supports only qk_norm='rms_norm'."
            )
        spatial_ratio = int(self.spatial_compression_ratio)
        if self.height % spatial_ratio != 0 or self.width % spatial_ratio != 0:
            raise ValueError(
                "HunyuanVideo 1.5 compile height/width must be divisible by "
                f"{spatial_ratio}."
            )
        if self.latent_height % self.patch_size != 0 or self.latent_width % self.patch_size != 0:
            raise ValueError("HunyuanVideo 1.5 latent height/width must be divisible by patch_size.")
        if self.latent_frames % self.patch_size_t != 0:
            raise ValueError("HunyuanVideo 1.5 latent frame count must be divisible by patch_size_t.")


class ModelWrapperHunyuanVideo15Backbone(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo 1.5 DiT compile inputs."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag: str = "",
        compiler_args: str | None = None,
        priority_model_idx: int | None = None,
        model_init_kwargs=None,
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            model_init_kwargs or {},
        )
        self.bucket_config = None

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 1000))
        text_seq_len_2 = int(getattr(self.config, "text_seq_len_2", 256))
        image_seq_len = int(getattr(self.config, "image_seq_len", 729))

        return [
            (
                torch.randn(
                    [
                        batch_size,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, self.config.text_embed_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.int64),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len_2, self.config.text_embed_2_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len_2], dtype=torch.int64),
                torch.zeros([batch_size, image_seq_len, self.config.image_embed_dim], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = _HunyuanVideo15TraceModule(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep_r,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep_r,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image_embeds,
        )


class NeuronHunyuanVideo15BackboneApplication(NeuronApplicationBase):
    """Compile/load wrapper for diffusers ``HunyuanVideo15Transformer3DModel``."""

    _model_cls = object

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="HunyuanVideo15Transformer3DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideo15BackboneInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideo15Backbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        override = os.environ.get("NOVA_HUNYUAN15_COMPILER_ARGS")
        if override:
            os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
            return override
        compiler_args = (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        return {
            key if key.startswith("transformer.") else f"transformer.{key}": value
            for key, value in state_dict.items()
        }

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


def _model_kwargs_from_config(config: InferenceConfig) -> dict[str, object]:
    names = (
        "in_channels",
        "out_channels",
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "num_refiner_layers",
        "mlp_ratio",
        "patch_size",
        "patch_size_t",
        "qk_norm",
        "text_embed_dim",
        "text_embed_2_dim",
        "image_embed_dim",
        "rope_theta",
        "rope_axes_dim",
        "target_size",
        "task_type",
        "use_meanflow",
    )
    kwargs = {name: getattr(config, name) for name in names if hasattr(config, name)}
    if isinstance(kwargs.get("rope_axes_dim"), list):
        kwargs["rope_axes_dim"] = tuple(kwargs["rope_axes_dim"])
    return kwargs


class _HunyuanVideo15TraceModule(nn.Module):
    """Fixed-tuple wrapper around diffusers HunyuanVideo15Transformer3DModel."""

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        self.use_meanflow = bool(getattr(config, "use_meanflow", False))
        self.transformer = HunyuanVideo15Transformer3DModel(**_model_kwargs_from_config(config))
        if os.environ.get("NOVA_HUNYUAN15_NKI_ATTENTION") == "1":
            processor = _HunyuanVideo15NkiNoMaskAttnProcessor()
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        elif os.environ.get("NOVA_HUNYUAN15_NO_ATTENTION_MASK") == "1":
            raw_chunk_size = os.environ.get("NOVA_HUNYUAN15_ATTENTION_QUERY_CHUNK_SIZE")
            chunk_size = int(raw_chunk_size) if raw_chunk_size else None
            processor = _HunyuanVideo15NoMaskAttnProcessor(query_chunk_size=chunk_size)
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        if os.environ.get("NOVA_HUNYUAN15_ZERO_BLOCK_ATTN") == "1":
            processor = _HunyuanVideo15ZeroAttnProcessor()
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        if os.environ.get("NOVA_HUNYUAN15_ZERO_BLOCK_FF") == "1":
            for block in self.transformer.transformer_blocks:
                block.ff = _ZeroLikeModule()
                block.ff_context = _ZeroLikeModule()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep_r=timestep_r if self.use_meanflow else None,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            image_embeds=image_embeds,
            return_dict=False,
        )[0]
