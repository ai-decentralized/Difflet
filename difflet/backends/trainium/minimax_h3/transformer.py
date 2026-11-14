"""Trainium TP4 wrapper for the MiniMax-H3 Omni Transformer."""

from __future__ import annotations

import os
from typing import List

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.minimax_h3.contracts import (
    PATCH_SIZE,
    VAE_SPATIAL_COMPRESSION,
    build_padded_t2va_layout,
    video_latent_num_frames,
)
# The TP rewrite is backend-neutral and shared with the neuron backend; re-exported here under
# the names this module has always had.
from difflet.models.minimax_h3.modeling_minimax_h3_tp import (  # noqa: F401
    MiniMaxH3TPAttnProcessor,
    _column_parallel_like,
    _H3TensorParallelSwiGLU,
    _MiniMaxH3TrainiumAttnProcessor,
    _MiniMaxH3TransformerTraceModule,
    _prefix_bounds,
    _replace_attention_for_tp,
    _replace_block_for_tp,
    _replace_h3_linears_for_tp,
    _row_parallel_like,
    _safe_tp_size,
    ff_prefixes,
    split_fused_swiglu_keys,
)


class MiniMaxH3TransformerInferenceConfig(InferenceConfig):
    """Fixed-shape H3 T2VA Transformer compile contract."""

    def add_derived_config(self):
        super().add_derived_config()
        self.patch_size = tuple(getattr(self, "patch_size", PATCH_SIZE))
        self.text_seq_len = int(getattr(self, "text_seq_len", 1024))
        self.vae_spatial_compression = int(
            getattr(self, "vae_spatial_compression", VAE_SPATIAL_COMPRESSION)
        )
        # When enabled, the 13B adaln_proj branch is dropped from the device
        # graph; the host feeds each step's precomputed modulation table as an
        # extra graph input instead (schedule-dependent data, not weights).
        self.precomputed_adaln = bool(getattr(self, "precomputed_adaln", False))

    def get_required_attributes(self) -> List[str]:
        return [
            "num_attention_heads",
            "attention_head_dim",
            "hidden_size",
            "num_layers",
            "num_refiner_layers",
            "ffn_dim",
            "in_channels",
            "audio_in_channels",
            "patch_size",
            "text_dim",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
            "rope_freq_dim",
            "rope_theta",
            "norm_eps",
            "qk_norm_eps",
            "final_norm_eps",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.vae_spatial_compression)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.vae_spatial_compression)

    @property
    def num_video_latent_frames(self) -> int:
        return video_latent_num_frames(int(self.num_frames))

    @property
    def video_seq_len(self) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        return (
            self.num_video_latent_frames
            * (self.latent_height // int(patch_h))
            * (self.latent_width // int(patch_w))
            // int(patch_t)
        )

    @property
    def video_patch_dim(self) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        return int(self.in_channels) * int(patch_t) * int(patch_h) * int(patch_w)

    @property
    def audio_seq_len(self) -> int:
        # 40 latent rows/s/channel, stereo, at the model's fixed 24 fps.
        return int(round(int(self.num_frames) / 24 * 40)) * 2

    @property
    def packed_seq_len(self) -> int:
        unaligned = int(self.text_seq_len) + self.audio_seq_len + self.video_seq_len
        return (unaligned + 127) // 128 * 128

    def validate_config(self):
        super().validate_config()
        if self.patch_size != PATCH_SIZE:
            raise NotImplementedError(
                f"MiniMax-H3 currently supports patch_size={PATCH_SIZE}, got {self.patch_size}."
            )
        if int(self.height) % 32 or int(self.width) % 32:
            raise ValueError("MiniMax-H3 height and width must be divisible by 32.")
        # Also validates the 17*n+5 temporal contract.
        video_latent_num_frames(int(self.num_frames))
        tp_degree = int(self.neuron_config.tp_degree)
        if tp_degree != 4:
            raise NotImplementedError(f"MiniMax-H3's first graph is TP4, got TP{tp_degree}.")
        if int(self.num_attention_heads) % tp_degree:
            raise ValueError(
                f"MiniMax-H3 heads ({self.num_attention_heads}) must divide TP{tp_degree}."
            )
        rotary_dim = 2 * 3 * int(self.rope_freq_dim)
        if rotary_dim > int(self.attention_head_dim):
            raise ValueError(
                f"MiniMax-H3 rotary dim ({rotary_dim}) exceeds attention head dim "
                f"({self.attention_head_dim})."
            )


class ModelWrapperMiniMaxH3Transformer(ModelWrapper):
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
        config = self.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        dtype = config.neuron_config.torch_dtype
        live_text_tokens = min(64, int(config.text_seq_len))
        layout = build_padded_t2va_layout(
            num_text_tokens=live_text_tokens,
            max_text_tokens=int(config.text_seq_len),
            height=int(config.height),
            width=int(config.width),
            num_frames=int(config.num_frames),
            sequence_alignment=128,
        )
        timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
        timestep_indices[layout.audio_indices] = 0
        encoder_attention_mask = torch.zeros(batch_size, int(config.text_seq_len), dtype=torch.bool)
        encoder_attention_mask[:, :live_text_tokens] = True
        modulation_inputs: tuple[torch.Tensor, ...] = ()
        if bool(config.precomputed_adaln):
            modulation_inputs = (
                torch.randn(
                    int(config.num_layers),
                    6,
                    2 * 3,  # two timestep slots x three modalities
                    int(config.hidden_size),
                    dtype=dtype,
                ),
            )
        return [
            (
                torch.randn(
                    batch_size,
                    config.video_seq_len,
                    config.video_patch_dim,
                    dtype=dtype,
                ),
                torch.randn(
                    batch_size,
                    config.audio_seq_len,
                    int(config.audio_in_channels),
                    dtype=dtype,
                ),
                torch.randn(
                    batch_size,
                    int(config.text_seq_len),
                    int(config.text_dim),
                    dtype=dtype,
                ),
                torch.tensor([0.3, 0.7], dtype=torch.float32),
                timestep_indices,
                layout.token_tags.to(torch.int64),
                layout.position_ids.to(torch.float32),
                layout.video_indices.to(torch.int64),
                layout.audio_indices.to(torch.int64),
                layout.text_indices.to(torch.int64),
                encoder_attention_mask,
                layout.attention_mask.unsqueeze(0),
                *modulation_inputs,
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.restore_checkpoint_fp32_modules()
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *model_inputs):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(*model_inputs)


class NeuronMiniMaxH3TransformerApplication(NeuronApplicationBase):
    _model_cls = _MiniMaxH3TransformerTraceModule

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="MiniMaxH3Transformer3DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return MiniMaxH3TransformerInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperMiniMaxH3Transformer

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict,
        config: InferenceConfig,
    ) -> dict:
        converted = {
            key if key.startswith("transformer.") else f"transformer.{key}": value
            for key, value in state_dict.items()
        }
        if bool(getattr(config, "precomputed_adaln", False)):
            # The host feeds the precomputed modulation table per step; the 13B
            # adaln_proj branch never becomes device weights.
            converted = {
                key: value for key, value in converted.items() if ".adaln_proj." not in key
            }
        split_fused_swiglu_keys(
            converted,
            ff_prefixes(int(config.num_refiner_layers), int(config.num_layers)),
        )
        return converted

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
