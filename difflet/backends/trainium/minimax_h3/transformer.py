"""Trainium TP4 wrapper for the MiniMax-H3 Omni Transformer."""

from __future__ import annotations

import math
import os
from typing import List

import torch
from torch import nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.minimax_h3.contracts import (
    PATCH_SIZE,
    VAE_SPATIAL_COMPRESSION,
    build_padded_t2va_layout,
    video_latent_num_frames,
)
from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    attention,
    get_tensor_model_parallel_size,
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


class _H3TensorParallelSwiGLU(nn.Module):
    """Shard both SwiGLU branches without mixing their checkpoint row layout."""

    def __init__(self, hidden_size: int, ffn_dim: int) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size, ffn_dim, bias=False, gather_output=False)
        self.gate_proj = ColumnParallelLinear(hidden_size, ffn_dim, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(ffn_dim, hidden_size, bias=False, input_is_parallel=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.up_proj(hidden_states) * torch.nn.functional.silu(self.gate_proj(hidden_states))
        )


def _column_parallel_like(linear: nn.Linear, *, gather_output: bool) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=gather_output,
    )


def _row_parallel_like(linear: nn.Linear) -> RowParallelLinear:
    return RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=True,
    )


def _safe_tp_size() -> int:
    try:
        return int(get_tensor_model_parallel_size())
    except AssertionError:
        return 1


def _replace_attention_for_tp(attn, tp_degree: int) -> None:
    if int(attn.heads) % tp_degree:
        raise ValueError(f"MiniMax-H3 attention heads {attn.heads} must divide TP{tp_degree}.")
    attn.heads = int(attn.heads) // tp_degree
    attn.to_q = _column_parallel_like(attn.to_q, gather_output=False)
    attn.to_k = _column_parallel_like(attn.to_k, gather_output=False)
    attn.to_v = _column_parallel_like(attn.to_v, gather_output=False)
    attn.to_out[0] = _row_parallel_like(attn.to_out[0])
    attn.processor = _MiniMaxH3TrainiumAttnProcessor()


def _replace_block_for_tp(block, tp_degree: int, hidden_size: int, ffn_dim: int) -> None:
    _replace_attention_for_tp(block.attn, tp_degree)
    block.ff = _H3TensorParallelSwiGLU(hidden_size, ffn_dim)


def _replace_h3_linears_for_tp(transformer: nn.Module) -> None:
    tp_degree = _safe_tp_size()
    if tp_degree <= 1:
        return

    hidden_size = int(transformer.config.hidden_size)
    ffn_dim = int(transformer.config.ffn_dim)
    transformer.context_embedder = _column_parallel_like(
        transformer.context_embedder, gather_output=True
    )
    transformer.norm_out.linear = _column_parallel_like(
        transformer.norm_out.linear, gather_output=True
    )
    for block in transformer.token_refiner.refiner_blocks:
        _replace_block_for_tp(block, tp_degree, hidden_size, ffn_dim)
    for block in transformer.transformer_blocks:
        _replace_block_for_tp(block, tp_degree, hidden_size, ffn_dim)
        # The 13B AdaLN branch is sharded in HBM; its tiny per-step output is
        # gathered because the residual stream remains replicated after TP.
        block.adaln_proj.linear = _column_parallel_like(block.adaln_proj.linear, gather_output=True)


def _prefix_bounds(
    attention_mask: torch.Tensor,
    *,
    num_heads: int,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact prefix-validity mask to attention_cte per-query bounds."""

    valid = attention_mask if attention_mask.dtype == torch.bool else attention_mask != 0
    count = valid.sum(dim=-1, keepdim=True).to(torch.int32)
    batch_size = int(valid.shape[0])
    bound_max = (
        count[:, None, None, :]
        .expand(batch_size, num_heads, query_length, 1)
        .reshape(batch_size * num_heads, query_length, 1)
        .contiguous()
    )
    return torch.zeros_like(bound_max), bound_max


class _MiniMaxH3TrainiumAttnProcessor:
    """Full H3 self-attention through attention_cte, including tail bounds."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from difflet.models.minimax_h3.modeling_minimax_h3 import _apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        batch_size, num_heads, query_length, head_dim = q.shape
        key_length = int(k.shape[2])
        kwargs = {}
        if attention_mask is not None:
            bound_min, bound_max = _prefix_bounds(
                attention_mask,
                num_heads=num_heads,
                query_length=query_length,
            )
            kwargs.update(bound_min=bound_min, bound_max=bound_max)
        output = attention(
            q.reshape(batch_size * num_heads, query_length, head_dim),
            k.reshape(batch_size * num_heads, key_length, head_dim),
            v.reshape(batch_size * num_heads, key_length, head_dim),
            scale=1.0 / math.sqrt(head_dim),
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
            **kwargs,
        )
        output = output.reshape(batch_size, num_heads, query_length, head_dim)
        output = output.transpose(1, 2).flatten(2, 3).to(query.dtype)
        output = attn.to_out[0](output.contiguous())
        return attn.to_out[1](output)


class _MiniMaxH3TransformerTraceModule(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerInferenceConfig) -> None:
        super().__init__()
        from difflet.models.minimax_h3.modeling_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )

        self.config = config
        self.transformer = MiniMaxH3Transformer3DModel(
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            num_refiner_layers=int(config.num_refiner_layers),
            ffn_dim=int(config.ffn_dim),
            in_channels=int(config.in_channels),
            audio_in_channels=int(config.audio_in_channels),
            patch_size=tuple(config.patch_size),
            text_dim=int(config.text_dim),
            freq_dim=int(config.freq_dim),
            time_embed_hidden_dim=int(config.time_embed_hidden_dim),
            time_embed_dim=int(config.time_embed_dim),
            rope_freq_dim=int(config.rope_freq_dim),
            rope_theta=float(config.rope_theta),
            norm_eps=float(config.norm_eps),
            qk_norm_eps=float(config.qk_norm_eps),
            final_norm_eps=float(config.final_norm_eps),
        )
        _replace_h3_linears_for_tp(self.transformer)

    def restore_checkpoint_fp32_modules(self) -> None:
        transformer = self.transformer
        transformer.proj_in.float()
        transformer.audio_proj_in.float()
        transformer.time_embedder.float()
        transformer.proj_out.float()
        transformer.audio_proj_out.float()

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        encoder_attention_mask,
        attention_mask,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            return_dict=False,
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
        prefixes = [
            *[
                f"transformer.token_refiner.refiner_blocks.{index}.ff"
                for index in range(int(config.num_refiner_layers))
            ],
            *[
                f"transformer.transformer_blocks.{index}.ff"
                for index in range(int(config.num_layers))
            ],
        ]
        for prefix in prefixes:
            fused_key = f"{prefix}.net.0.proj.weight"
            if fused_key not in converted:
                continue
            up_weight, gate_weight = converted.pop(fused_key).chunk(2, dim=0)
            converted[f"{prefix}.up_proj.weight"] = up_weight.contiguous()
            converted[f"{prefix}.gate_proj.weight"] = gate_weight.contiguous()
            converted[f"{prefix}.down_proj.weight"] = converted.pop(f"{prefix}.net.2.weight")
        return converted

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
