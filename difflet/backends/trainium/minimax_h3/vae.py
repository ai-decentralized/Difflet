"""Trainium wrappers for the two MiniMax-H3 decoder-only VAE stages."""

from __future__ import annotations

import math
import os
from typing import List

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.common.orchestrators import minimax_h3 as h3_common
from difflet.models.minimax_h3.modeling_vae import (
    MiniMaxH3AudioBigVGANDecoder,
    MiniMaxH3VideoViTDecoder3d,
    apply_h3_video_rotary,
)
from difflet.ops import attention


class MiniMaxH3VideoVAEDecoderInferenceConfig(InferenceConfig):
    """Fixed official 256px tile and seven-latent-frame visual decoder graph."""

    def add_derived_config(self):
        super().add_derived_config()
        self.spatial_compression_ratio = math.prod(self.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(self.temporal_downsample_factors)
        self.tokens_chunk_size = math.ceil(
            int(self.clip_length) / int(self.temporal_compression_ratio)
        )
        self.token_overlap = (-int(self.token_drop)) % int(self.tokens_chunk_size)
        self.tile_sample_height = int(getattr(self, "tile_sample_height", 256))
        self.tile_sample_width = int(getattr(self, "tile_sample_width", 256))
        self.tile_latent_height = self.tile_sample_height // self.spatial_compression_ratio
        self.tile_latent_width = self.tile_sample_width // self.spatial_compression_ratio
        self.tile_latent_frames = self.tokens_chunk_size + self.token_overlap

    def get_required_attributes(self) -> List[str]:
        return [
            "out_channels",
            "latent_channels",
            "spatial_downsample_factors",
            "temporal_downsample_factors",
            "decoder_num_layers",
            "decoder_num_attention_heads",
            "decoder_attention_head_dim",
            "decoder_num_register_tokens",
            "decoder_ffn_mult",
            "decoder_rope_theta",
            "decoder_rope_dim_ratio",
            "decoder_norm_eps",
            "clip_length",
            "token_drop",
            "latents_mean",
            "latents_std",
            "height",
            "width",
            "num_frames",
        ]

    def validate_config(self):
        super().validate_config()
        if int(self.neuron_config.tp_degree) != 1:
            raise NotImplementedError("MiniMax-H3 visual VAE's first tile graph is TP1")
        if self.spatial_compression_ratio != 16 or self.temporal_compression_ratio != 4:
            raise NotImplementedError("MiniMax-H3 visual VAE expects f4/d16 compression")
        if self.tile_sample_height != 256 or self.tile_sample_width != 256:
            raise NotImplementedError("MiniMax-H3 visual VAE parity requires official 256px tiles")
        if self.tile_latent_frames != 7:
            raise NotImplementedError("MiniMax-H3 visual VAE expects seven latent frames per clip")
        if len(self.latents_mean) != int(self.latent_channels) or len(self.latents_std) != int(
            self.latent_channels
        ):
            raise ValueError("MiniMax-H3 visual VAE normalization must match latent_channels")


class MiniMaxH3AudioVAEDecoderInferenceConfig(InferenceConfig):
    """Fixed stereo-as-batch audio decoder graph."""

    def add_derived_config(self):
        super().add_derived_config()
        self.audio_latent_frames = int(round(int(self.num_frames) / 24 * 40))
        self.audio_chunk_core_frames = h3_common.AUDIO_VAE_CHUNK_CORE_FRAMES
        self.audio_chunk_halo = h3_common.AUDIO_VAE_CHUNK_HALO_FRAMES
        self.audio_chunk_latent_frames = h3_common.AUDIO_VAE_CHUNK_LATENT_FRAMES
        self.audio_num_chunks = math.ceil(self.audio_latent_frames / self.audio_chunk_core_frames)

    def get_required_attributes(self) -> List[str]:
        return [
            "latent_dim",
            "latent_channels",
            "decoder_dim",
            "decoder_rates",
            "decoder_kernel_sizes",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
            "sampling_rate",
            "latents_mean",
            "latents_std",
            "num_frames",
        ]

    def validate_config(self):
        super().validate_config()
        if int(self.neuron_config.tp_degree) != 1:
            raise NotImplementedError("MiniMax-H3 audio VAE's first graph is TP1")
        if math.prod(self.decoder_rates) != 800:
            raise NotImplementedError("MiniMax-H3 audio VAE expects an 800-sample decoder hop")
        if self.audio_chunk_latent_frames > self.audio_latent_frames:
            raise NotImplementedError(
                "MiniMax-H3 audio VAE chunk graph cannot exceed the requested latent length"
            )
        if len(self.latents_mean) != int(self.latent_channels) or len(self.latents_std) != int(
            self.latent_channels
        ):
            raise ValueError("MiniMax-H3 audio VAE normalization must match latent_channels")


class _MiniMaxH3VideoVAETrainiumAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query.float()).to(torch.float16)
        key = attn.norm_k(key.float()).to(torch.float16)
        value = value.to(torch.float16)
        query = apply_h3_video_rotary(query, rotary_emb)
        key = apply_h3_video_rotary(key, rotary_emb)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        batch_size, num_heads, query_length, head_dim = q.shape
        kwargs = {}
        if attention_mask is not None:
            valid_count = attention_mask.sum(dim=-1, keepdim=True).to(torch.int32)
            bound_max = (
                valid_count[:, None, None, :]
                .expand(batch_size, num_heads, query_length, 1)
                .reshape(batch_size * num_heads, query_length, 1)
                .contiguous()
            )
            kwargs = {"bound_min": torch.zeros_like(bound_max), "bound_max": bound_max}
        output = attention(
            q.reshape(batch_size * num_heads, query_length, head_dim),
            k.reshape(batch_size * num_heads, query_length, head_dim),
            v.reshape(batch_size * num_heads, query_length, head_dim),
            scale=1.0 / math.sqrt(head_dim),
            causal=False,
            # attention_cte's [B*H, S, D] layout is selected by the tp_q/tp_k
            # flags even for this TP1 graph.
            tp_q=True,
            tp_k=True,
            tp_out=False,
            **kwargs,
        )
        output = output.reshape(batch_size, num_heads, query_length, head_dim)
        output = output.transpose(1, 2).flatten(2, 3).to(hidden_states.dtype)
        return attn.to_out[0](output.contiguous())


class MiniMaxH3VideoVAEDecoderModel(nn.Module):
    def __init__(self, config: MiniMaxH3VideoVAEDecoderInferenceConfig) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv3d(
            int(config.latent_channels),
            int(config.latent_channels),
            kernel_size=1,
        )
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=int(config.latent_channels),
            out_channels=int(config.out_channels),
            patch_size=int(config.spatial_compression_ratio),
            patch_size_t=int(config.temporal_compression_ratio),
            num_layers=int(config.decoder_num_layers),
            num_attention_heads=int(config.decoder_num_attention_heads),
            attention_head_dim=int(config.decoder_attention_head_dim),
            num_register_tokens=int(config.decoder_num_register_tokens),
            ffn_mult=int(config.decoder_ffn_mult),
            rope_theta=float(config.decoder_rope_theta),
            rope_dim_ratio=float(config.decoder_rope_dim_ratio),
            norm_eps=float(config.decoder_norm_eps),
            sequence_alignment=128,
        )
        for block in self.decoder.transformer_blocks:
            block.attn.set_processor(_MiniMaxH3VideoVAETrainiumAttnProcessor())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latents))


class MiniMaxH3AudioVAEDecoderModel(nn.Module):
    def __init__(self, config: MiniMaxH3AudioVAEDecoderInferenceConfig) -> None:
        super().__init__()
        self.dec_in_proj = nn.Conv1d(int(config.latent_channels), int(config.latent_dim), 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=int(config.latent_dim),
            upsample_initial_channel=int(config.decoder_dim),
            upsample_rates=tuple(int(value) for value in config.decoder_rates),
            upsample_kernel_sizes=tuple(int(value) for value in config.decoder_kernel_sizes),
            resblock_kernel_sizes=tuple(int(value) for value in config.resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(
                tuple(int(value) for value in dilation)
                for dilation in config.resblock_dilation_sizes
            ),
            use_weight_norm=False,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(latents))


class _MiniMaxH3VAEModelWrapper(ModelWrapper):
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

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, latents):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(latents)


class ModelWrapperMiniMaxH3VideoVAE(_MiniMaxH3VAEModelWrapper):
    def input_generator(self) -> list[tuple[torch.Tensor]]:
        return [
            (
                torch.randn(
                    1,
                    int(self.config.latent_channels),
                    int(self.config.tile_latent_frames),
                    int(self.config.tile_latent_height),
                    int(self.config.tile_latent_width),
                    dtype=torch.float32,
                ),
            )
        ]


class ModelWrapperMiniMaxH3AudioVAE(_MiniMaxH3VAEModelWrapper):
    def input_generator(self) -> list[tuple[torch.Tensor]]:
        return [
            (
                torch.randn(
                    2,
                    int(self.config.latent_channels),
                    int(self.config.audio_chunk_latent_frames),
                    dtype=torch.float32,
                ),
            )
        ]


class _MiniMaxH3VAEApplication(NeuronApplicationBase):
    wrapper_cls = _MiniMaxH3VAEModelWrapper

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.wrapper_cls(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = torch.float32

    def get_model_wrapper_cls(self):
        return self.wrapper_cls

    def forward(self, latents):
        return self.models[0](latents)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


class NeuronMiniMaxH3VideoVAEDecoderApplication(_MiniMaxH3VAEApplication):
    _model_cls = MiniMaxH3VideoVAEDecoderModel
    wrapper_cls = ModelWrapperMiniMaxH3VideoVAE

    @classmethod
    def get_config_cls(cls):
        return MiniMaxH3VideoVAEDecoderInferenceConfig

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        return (
            "--model-type=transformer -O1 "
            "--auto-cast=matmult --auto-cast-type=fp16 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("post_quant_conv.", "decoder."))
        }


class NeuronMiniMaxH3AudioVAEDecoderApplication(_MiniMaxH3VAEApplication):
    _model_cls = MiniMaxH3AudioVAEDecoderModel
    wrapper_cls = ModelWrapperMiniMaxH3AudioVAE

    @classmethod
    def get_config_cls(cls):
        return MiniMaxH3AudioVAEDecoderInferenceConfig

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        return (
            "--model-type=unet-inference -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        converted = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("dec_in_proj.", "decoder."))
        }
        # Legacy torch weight_norm is a Python pre-hook and aborts PJRT tracing.
        # Materialize its mathematically identical convolution weight once on
        # the host, then load ordinary Conv1d/ConvTranspose1d modules.
        for key in list(converted):
            if not key.endswith(".weight_v"):
                continue
            prefix = key[: -len(".weight_v")]
            value = converted.pop(key)
            scale = converted.pop(f"{prefix}.weight_g")
            converted[f"{prefix}.weight"] = torch._weight_norm(value, scale, 0)
        return converted
