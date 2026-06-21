"""Segmented Trainium runtime pieces for HunyuanVideo 1.5 transformer blocks."""

from __future__ import annotations

import math
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
from difflet.backends.trainium.core.multi_component_application import ComponentSpec


DEFAULT_BLOCK_COMPILER_ARGS = (
    "--model-type=transformer -O1 --auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)
DEFAULT_ATTENTION_COMPILER_ARGS = (
    "--model-type=generic -O1 --auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


@dataclass(frozen=True)
class HunyuanVideo15SegmentedMetrics:
    """Last forward counters for the host-stitched segmented path."""

    stream_tile_calls: int
    stream_forward_elapsed_s: float
    block_count: int = 0
    blocks_elapsed_s: float = 0.0


def _load_block_state_dict_from_dir(
    transformer_dir: Path,
    block_index: int,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    prefix = f"transformer_blocks.{block_index}."
    block_state_dict = {}

    if index_path.exists():
        from safetensors import safe_open

        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard_name in weight_map.items():
            if key.startswith(prefix):
                shard_to_keys.setdefault(str(shard_name), []).append(key)
        for shard_name, keys in shard_to_keys.items():
            with safe_open(transformer_dir / shard_name, framework="pt", device="cpu") as shard:
                for key in keys:
                    value = shard.get_tensor(key)
                    tensor = (
                        value.to(dtype=dtype)
                        if dtype is not None and torch.is_floating_point(value)
                        else value
                    )
                    block_state_dict[f"block.{key[len(prefix):]}"] = tensor
        if block_state_dict:
            return block_state_dict

    state_dict = load_state_dict(str(transformer_dir))
    for key, value in state_dict.items():
        if key.startswith(prefix):
            tensor = value.to(dtype=dtype) if dtype is not None and torch.is_floating_point(value) else value
            block_state_dict[f"block.{key[len(prefix):]}"] = tensor
    if not block_state_dict:
        raise KeyError(f"no state_dict keys found for {prefix!r} under {transformer_dir}")
    return block_state_dict


def _full_attention_valid_mask(context_attention_mask: torch.Tensor, latent_seq_len: int) -> torch.Tensor:
    latent_valid = torch.ones(
        [context_attention_mask.shape[0], latent_seq_len],
        dtype=torch.bool,
        device=context_attention_mask.device,
    )
    return torch.cat([latent_valid, context_attention_mask.to(dtype=torch.bool)], dim=1)


def merge_manual_stats_tiles(
    tile_numerators: list[torch.Tensor],
    tile_max_scores: list[torch.Tensor],
    tile_denoms: list[torch.Tensor],
) -> torch.Tensor:
    if not tile_numerators:
        raise ValueError("at least one attention tile is required")
    stacked_max = torch.stack(tile_max_scores, dim=0)
    global_max = stacked_max.max(dim=0).values
    numerator_accum = torch.zeros_like(tile_numerators[0])
    denom_accum = torch.zeros_like(tile_denoms[0])
    finite_global = torch.isfinite(global_max)
    for numerator, max_score, denom in zip(tile_numerators, tile_max_scores, tile_denoms):
        scale = torch.where(
            finite_global,
            torch.exp(max_score - global_max),
            torch.zeros_like(max_score),
        )
        numerator_accum = numerator_accum + numerator * scale.unsqueeze(-1)
        denom_accum = denom_accum + denom * scale
    return torch.where(
        denom_accum.unsqueeze(-1) > 0,
        numerator_accum / denom_accum.unsqueeze(-1),
        torch.zeros_like(numerator_accum),
    )


def run_streaming_manual_stats_attention(
    app: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_tile_size: int,
    key_tile_size: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, HunyuanVideo15SegmentedMetrics]:
    """Run the compiled masked attention tile over full BSHD Q/K/V tensors."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be BSHD tensors")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(f"incompatible query/key shapes: {tuple(query.shape)} vs {tuple(key.shape)}")
    if query.shape[1] % query_tile_size != 0:
        raise ValueError("query sequence length must be divisible by query_tile_size")
    if key.shape[1] % key_tile_size != 0:
        raise ValueError("key sequence length must be divisible by key_tile_size")
    if tuple(valid_mask.shape) != (query.shape[0], key.shape[1]):
        raise ValueError(
            f"valid_mask must have shape {(query.shape[0], key.shape[1])}, "
            f"got {tuple(valid_mask.shape)}"
        )

    import time

    valid_mask_bool = valid_mask.to(dtype=torch.bool)
    valid_mask = valid_mask.to(dtype=torch.int64)
    forward_elapsed = 0.0
    output_chunks = []
    stream_query_tiles = query.shape[1] // query_tile_size
    stream_key_tiles = key.shape[1] // key_tile_size
    for query_index in range(stream_query_tiles):
        query_start = query_index * query_tile_size
        query_end = query_start + query_tile_size
        query_chunk = query[:, query_start:query_end].contiguous()
        tile_numerators = []
        tile_max_scores = []
        tile_denoms = []
        for key_index in range(stream_key_tiles):
            key_start = key_index * key_tile_size
            key_end = key_start + key_tile_size
            t0 = time.perf_counter()
            with torch.no_grad():
                numerator, max_score, denom = app(
                    query_chunk,
                    key[:, key_start:key_end].contiguous(),
                    value[:, key_start:key_end].contiguous(),
                    valid_mask[:, query_start:query_end].contiguous(),
                    valid_mask[:, key_start:key_end].contiguous(),
                )
            forward_elapsed += time.perf_counter() - t0
            numerator = numerator.detach().cpu().float()
            max_score = max_score.detach().cpu().float()
            denom = denom.detach().cpu().float()
            if not bool(valid_mask_bool[:, key_start:key_end].any()):
                numerator.zero_()
                max_score.fill_(-float("inf"))
                denom.zero_()
            tile_numerators.append(numerator)
            tile_max_scores.append(max_score)
            tile_denoms.append(denom)

        merged = merge_manual_stats_tiles(tile_numerators, tile_max_scores, tile_denoms)
        merged = torch.where(
            valid_mask_bool[:, query_start:query_end].unsqueeze(-1).unsqueeze(-1).cpu(),
            merged,
            torch.zeros_like(merged),
        )
        output_chunks.append(merged)

    return (
        torch.cat(output_chunks, dim=1),
        HunyuanVideo15SegmentedMetrics(
            stream_tile_calls=stream_query_tiles * stream_key_tiles,
            stream_forward_elapsed_s=forward_elapsed,
        ),
    )


class _Hunyuan15BlockPreQkvModule(nn.Module):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15TransformerBlock,
        )

        self.block = HunyuanVideo15TransformerBlock(
            config.heads,
            config.head_dim,
            mlp_ratio=config.mlp_ratio,
            qk_norm=config.qk_norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.models.embeddings import apply_rotary_emb

        norm_hidden_states, *_ = self.block.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, *_ = self.block.norm1_context(encoder_hidden_states, emb=temb)
        attn = self.block.attn

        query = attn.to_q(norm_hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(norm_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(norm_hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = apply_rotary_emb(query, (freqs_cos, freqs_sin), sequence_dim=1)
        key = apply_rotary_emb(key, (freqs_cos, freqs_sin), sequence_dim=1)

        encoder_query = attn.add_q_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_key = attn.add_k_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_value = attn.add_v_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        return (
            torch.cat([query, encoder_query], dim=1),
            torch.cat([key, encoder_key], dim=1),
            torch.cat([value, encoder_value], dim=1),
        )


class _Hunyuan15BlockPostModule(nn.Module):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15TransformerBlock,
        )

        self.block = HunyuanVideo15TransformerBlock(
            config.heads,
            config.head_dim,
            mlp_ratio=config.mlp_ratio,
            qk_norm=config.qk_norm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_seq_len = hidden_states.shape[1]
        _norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.norm1(
            hidden_states,
            emb=temb,
        )
        (
            _norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.block.norm1_context(encoder_hidden_states, emb=temb)

        attention_states = attention_states.flatten(2, 3).to(hidden_states.dtype)
        attn_output = attention_states[:, :latent_seq_len]
        context_attn_output = attention_states[:, latent_seq_len:]
        attn_output = self.block.attn.to_out[0](attn_output)
        attn_output = self.block.attn.to_out[1](attn_output)
        context_attn_output = self.block.attn.to_add_out(context_attn_output)

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.block.norm2(hidden_states)
        norm_encoder_hidden_states = self.block.norm2_context(encoder_hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        ff_output = self.block.ff(norm_hidden_states)
        context_ff_output = self.block.ff_context(norm_encoder_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        return hidden_states, encoder_hidden_states


class _AttentionTileModule(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * (1 / math.sqrt(query.shape[-1]))
        key_valid = key_valid.to(dtype=torch.bool).view(key.shape[0], 1, 1, key.shape[2])
        scores = torch.where(key_valid, scores, torch.full_like(scores, -1.0e9))
        max_score = scores.max(dim=-1).values
        weights = torch.exp(scores - max_score.unsqueeze(-1))
        denom = weights.sum(dim=-1)
        numerator = torch.matmul(weights.to(value.dtype), value)
        query_valid = query_valid.to(dtype=torch.bool).view(query.shape[0], 1, query.shape[2])
        numerator = torch.where(query_valid.unsqueeze(-1), numerator, torch.zeros_like(numerator))
        max_score = torch.where(query_valid, max_score, torch.full_like(max_score, -float("inf")))
        denom = torch.where(query_valid, denom, torch.zeros_like(denom))
        return numerator.permute(0, 2, 1, 3), max_score.permute(0, 2, 1), denom.permute(0, 2, 1)


class HunyuanVideo15BlockSegmentConfig(InferenceConfig):
    def get_required_attributes(self):
        return [
            "part",
            "latent_seq_len",
            "context_seq_len",
            "total_seq_len",
            "inner_dim",
            "heads",
            "head_dim",
            "mlp_ratio",
            "qk_norm",
            "source_model_dir",
            "block_index",
        ]


class HunyuanVideo15AttentionTileConfig(InferenceConfig):
    def get_required_attributes(self):
        return ["query_len", "key_len", "heads", "head_dim"]


class HunyuanVideo15BlockSegmentWrapper(ModelWrapper):
    def input_generator(self):
        dtype = self.config.neuron_config.torch_dtype
        hidden_shape = [1, self.config.latent_seq_len, self.config.inner_dim]
        context_shape = [1, self.config.context_seq_len, self.config.inner_dim]
        temb_shape = [1, self.config.inner_dim]
        if self.config.part == "pre-qkv":
            return [
                (
                    torch.randn(hidden_shape, dtype=dtype),
                    torch.randn(context_shape, dtype=dtype),
                    torch.randn(temb_shape, dtype=dtype),
                    torch.randn([self.config.latent_seq_len, self.config.head_dim], dtype=dtype),
                    torch.randn([self.config.latent_seq_len, self.config.head_dim], dtype=dtype),
                )
            ]
        return [
            (
                torch.randn(hidden_shape, dtype=dtype),
                torch.randn(context_shape, dtype=dtype),
                torch.randn(temb_shape, dtype=dtype),
                torch.randn(
                    [1, self.config.total_seq_len, self.config.heads, self.config.head_dim],
                    dtype=dtype,
                ),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            module = (
                _Hunyuan15BlockPreQkvModule(self.config)
                if self.config.part == "pre-qkv"
                else _Hunyuan15BlockPostModule(self.config)
            )
            module = module.to(dtype=self.config.neuron_config.torch_dtype)
            module.eval()
            return module

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *model_inputs):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(*model_inputs)


class HunyuanVideo15AttentionTileWrapper(ModelWrapper):
    def input_generator(self):
        dtype = self.config.neuron_config.torch_dtype
        shape_q = [1, self.config.query_len, self.config.heads, self.config.head_dim]
        shape_kv = [1, self.config.key_len, self.config.heads, self.config.head_dim]
        return [
            (
                torch.randn(shape_q, dtype=dtype),
                torch.randn(shape_kv, dtype=dtype),
                torch.randn(shape_kv, dtype=dtype),
                torch.ones([1, self.config.query_len], dtype=torch.int64),
                torch.ones([1, self.config.key_len], dtype=torch.int64),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = _AttentionTileModule()
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *model_inputs):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(*model_inputs)


class HunyuanVideo15BlockSegmentApplication(NeuronApplicationBase):
    _model_cls = object

    def __init__(self, *args, compiler_args: str = DEFAULT_BLOCK_COMPILER_ARGS, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_compiled_model_path: str | None = None
        self._loaded_start_rank_id: int | None = None
        self._loaded_local_ranks_size: int | None = None
        self._loaded_block_index = int(self.config.block_index)
        self.model = HunyuanVideo15BlockSegmentWrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=f"HunyuanVideo15Block_{self.config.block_index}_{self.config.part}",
            compiler_args=compiler_args,
            priority_model_idx=0,
        )
        self.models.append(self.model)

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideo15BlockSegmentConfig

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        del model_name_or_path
        return _load_block_state_dict_from_dir(
            Path(config.source_model_dir),
            int(config.block_index),
            dtype=config.neuron_config.torch_dtype,
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        self._loaded_compiled_model_path = str(compiled_model_path)
        self._loaded_start_rank_id = start_rank_id
        self._loaded_local_ranks_size = local_ranks_size
        self._loaded_block_index = int(self.config.block_index)
        return super().load(
            compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

    def reload_block_weights(self, block_index: int) -> None:
        block_index = int(block_index)
        if block_index == self._loaded_block_index:
            return
        if self.traced_model is None or self._loaded_compiled_model_path is None:
            raise RuntimeError("reload_block_weights called before load")
        self.config.block_index = block_index
        for model in self.models:
            model.config.block_index = block_index
        self._builder = None
        self.load_weights(
            self._loaded_compiled_model_path,
            start_rank_id=self._loaded_start_rank_id,
            local_ranks_size=self._loaded_local_ranks_size,
        )
        self._loaded_block_index = block_index


class HunyuanVideo15AttentionTileApplication(NeuronApplicationBase):
    _model_cls = object

    def __init__(self, *args, compiler_args: str = DEFAULT_ATTENTION_COMPILER_ARGS, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = HunyuanVideo15AttentionTileWrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="HunyuanVideo15MaskedAttentionTile",
            compiler_args=compiler_args,
            priority_model_idx=0,
        )
        self.models.append(self.model)

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideo15AttentionTileConfig

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        del model_name_or_path, config
        return {}

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)


class HunyuanVideo15SegmentedTransformerApplication(nn.Module):
    """Host-stitched HunyuanVideo 1.5 transformer runtime using segmented blocks."""

    def __init__(
        self,
        *,
        model_path: str,
        config: InferenceConfig,
        query_tile_size: int,
        key_tile_size: int,
        block_compiler_args: str = DEFAULT_BLOCK_COMPILER_ARGS,
        attention_compiler_args: str = DEFAULT_ATTENTION_COMPILER_ARGS,
        block_load_mode: str = "all",
    ) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.config = config
        self.dtype = config.neuron_config.torch_dtype
        self.query_tile_size = int(query_tile_size)
        self.key_tile_size = int(key_tile_size)
        self.block_load_mode = str(block_load_mode)
        self.block_compiler_args = block_compiler_args
        self.attention_compiler_args = attention_compiler_args
        self._compiled_model_path: str | None = None
        self._cpu_transformer: nn.Module | None = None
        self.last_metrics: HunyuanVideo15SegmentedMetrics | None = None
        if self.block_load_mode not in {"all", "streaming", "process"}:
            raise ValueError(
                "HunyuanVideo 1.5 segmented block_load_mode must be 'all', "
                f"'streaming', or 'process', got {self.block_load_mode!r}."
            )

        self.meta = self._shape_meta(config)
        if self.meta["total_seq_len"] % self.query_tile_size != 0:
            raise ValueError("HunyuanVideo 1.5 segmented total_seq_len must divide query_tile_size")
        if self.meta["total_seq_len"] % self.key_tile_size != 0:
            raise ValueError("HunyuanVideo 1.5 segmented total_seq_len must divide key_tile_size")

        self.pre_blocks = nn.ModuleList()
        self.post_blocks = nn.ModuleList()
        block_count = 1 if self.block_load_mode in {"streaming", "process"} else int(config.num_layers)
        for block_index in range(block_count):
            self.pre_blocks.append(
                HunyuanVideo15BlockSegmentApplication(
                    model_path=self.model_path,
                    config=self._block_config(config, block_index, "pre-qkv"),
                    compiler_args=block_compiler_args,
                )
            )
            self.post_blocks.append(
                HunyuanVideo15BlockSegmentApplication(
                    model_path=self.model_path,
                    config=self._block_config(config, block_index, "post"),
                    compiler_args=block_compiler_args,
                )
            )
        self.attention = HunyuanVideo15AttentionTileApplication(
            model_path=self.model_path,
            config=self._attention_config(config),
            compiler_args=attention_compiler_args,
        )

    @staticmethod
    def _shape_meta(config: InferenceConfig) -> dict[str, int]:
        latent_seq_len = (
            int(config.latent_frames)
            // int(config.patch_size_t)
            * int(config.latent_height)
            // int(config.patch_size)
            * int(config.latent_width)
            // int(config.patch_size)
        )
        context_seq_len = int(config.text_seq_len) + int(config.text_seq_len_2) + int(config.image_seq_len)
        inner_dim = int(config.num_attention_heads) * int(config.attention_head_dim)
        return {
            "latent_seq_len": latent_seq_len,
            "context_seq_len": context_seq_len,
            "total_seq_len": latent_seq_len + context_seq_len,
            "inner_dim": inner_dim,
        }

    def _block_config(
        self,
        base: InferenceConfig,
        block_index: int,
        part: str,
    ) -> HunyuanVideo15BlockSegmentConfig:
        return HunyuanVideo15BlockSegmentConfig(
            neuron_config=NeuronConfig(
                batch_size=1,
                tp_degree=base.neuron_config.tp_degree,
                world_size=base.neuron_config.world_size,
                torch_dtype=base.neuron_config.torch_dtype,
                skip_sharding=True,
            ),
            part=part,
            latent_seq_len=self.meta["latent_seq_len"],
            context_seq_len=self.meta["context_seq_len"],
            total_seq_len=self.meta["total_seq_len"],
            inner_dim=self.meta["inner_dim"],
            heads=int(base.num_attention_heads),
            head_dim=int(base.attention_head_dim),
            mlp_ratio=float(base.mlp_ratio),
            qk_norm=str(base.qk_norm),
            source_model_dir=self.model_path,
            block_index=block_index,
        )

    def _attention_config(self, base: InferenceConfig) -> HunyuanVideo15AttentionTileConfig:
        return HunyuanVideo15AttentionTileConfig(
            neuron_config=NeuronConfig(
                batch_size=1,
                tp_degree=base.neuron_config.tp_degree,
                world_size=base.neuron_config.world_size,
                torch_dtype=base.neuron_config.torch_dtype,
                skip_sharding=True,
            ),
            query_len=self.query_tile_size,
            key_len=self.key_tile_size,
            heads=int(base.num_attention_heads),
            head_dim=int(base.attention_head_dim),
        )

    def component_specs(self, prefix: str = "transformer") -> list[ComponentSpec]:
        specs = [
            ComponentSpec(f"{prefix}_attention_tile", self.attention),
        ]
        if self.block_load_mode in {"streaming", "process"}:
            specs.append(ComponentSpec(f"{prefix}_block_pre_qkv", self.pre_blocks[0]))
            specs.append(ComponentSpec(f"{prefix}_block_post", self.post_blocks[0]))
            return specs
        for index, (pre, post) in enumerate(zip(self.pre_blocks, self.post_blocks)):
            specs.append(
                ComponentSpec(
                    f"{prefix}_block_{index:02d}_pre_qkv",
                    pre,
                    artifact_name=f"{prefix}_block_pre_qkv",
                )
            )
            specs.append(
                ComponentSpec(
                    f"{prefix}_block_{index:02d}_post",
                    post,
                    artifact_name=f"{prefix}_block_post",
                )
            )
        return specs

    def set_compiled_model_path(self, compiled_model_path: str) -> None:
        self._compiled_model_path = str(compiled_model_path)

    def _load_cpu_transformer(self) -> nn.Module:
        if self._cpu_transformer is None:
            from diffusers.models.transformers.transformer_hunyuan_video15 import (
                HunyuanVideo15Transformer3DModel,
            )

            self._cpu_transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            ).eval()
        return self._cpu_transformer

    def _prepare_frontend(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        model = self._load_cpu_transformer()
        timestep_r = timestep_r if bool(getattr(self.config, "use_meanflow", False)) else None
        batch_size = hidden_states.shape[0]
        image_rotary_emb = model.rope(hidden_states)
        temb = model.time_embed(timestep, timestep_r=timestep_r)
        hidden_states = model.x_embedder(hidden_states)

        encoder_hidden_states = model.context_embedder(
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
        )
        encoder_hidden_states = encoder_hidden_states + model.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )

        encoder_hidden_states_2 = model.context_embedder_2(encoder_hidden_states_2)
        encoder_hidden_states_2 = encoder_hidden_states_2 + model.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )

        encoder_hidden_states_3 = model.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3 = encoder_hidden_states_3 + model.cond_type_embed(
            2 * torch.ones_like(encoder_hidden_states_3[:, :, 0], dtype=torch.long)
        )

        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []
        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],
                        text_2[text_mask_2],
                        text[text_mask],
                        image[~image_mask],
                        torch.zeros_like(text_2[~text_mask_2]),
                        torch.zeros_like(text[~text_mask]),
                    ],
                    dim=0,
                )
            )
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        return (
            hidden_states.to(dtype=self.dtype).contiguous(),
            torch.stack(new_encoder_hidden_states).to(dtype=self.dtype).contiguous(),
            temb.to(dtype=self.dtype).contiguous(),
            torch.stack(new_encoder_attention_mask).to(dtype=torch.int64).contiguous(),
            tuple(t.to(dtype=self.dtype).contiguous() for t in image_rotary_emb),
        )

    def _final_projection(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        model = self._load_cpu_transformer()
        p_t = int(model.config.patch_size_t)
        p_h = int(model.config.patch_size)
        p_w = int(model.config.patch_size)
        hidden_states = model.norm_out(hidden_states.to(dtype=temb.dtype), temb)
        hidden_states = model.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            int(self.config.latent_frames) // p_t,
            int(self.config.latent_height) // p_h,
            int(self.config.latent_width) // p_w,
            -1,
            p_t,
            p_h,
            p_w,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _save_process_tensors(self, path: Path, tensors: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, path)

    def _load_process_tensors(self, path: Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)

    def _process_runtime_config(self, compiled_model_path: str) -> dict[str, Any]:
        return {
            "transformer_dir": self.model_path,
            "compiled_path": str(compiled_model_path),
            "dtype": "bf16" if self.dtype is torch.bfloat16 else "fp32",
            "tp_degree": int(self.config.neuron_config.tp_degree),
            "query_tile_size": self.query_tile_size,
            "key_tile_size": self.key_tile_size,
            "block_compiler_args": self.block_compiler_args,
            "attention_compiler_args": self.attention_compiler_args,
            "heads": int(self.config.num_attention_heads),
            "head_dim": int(self.config.attention_head_dim),
            "mlp_ratio": float(self.config.mlp_ratio),
            "qk_norm": str(self.config.qk_norm),
            **self.meta,
        }

    def _run_process_block(
        self,
        *,
        runtime_config: Path,
        input_tensors: Path,
        output_tensors: Path,
        block_index: int,
    ) -> None:
        script_path = (
            Path(__file__).resolve().parents[4] / "scripts" / "hunyuan15_segmented_process_stream.py"
        )
        if not script_path.exists():
            raise FileNotFoundError(
                "HunyuanVideo 1.5 process streaming requires "
                f"{script_path}; use block_load_mode='streaming' in packaged installs."
            )
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[4])
        env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env.setdefault("DIFFLET_BACKEND", "trainium")
        env["LOCAL_WORLD_SIZE"] = str(int(self.config.neuron_config.tp_degree))
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--worker",
                "--runtime-config",
                str(runtime_config),
                "--input-tensors",
                str(input_tensors),
                "--output-tensors",
                str(output_tensors),
                "--block-index",
                str(block_index),
                "--model-dir",
                self.model_path,
                "--bundle",
                str(input_tensors),
            ],
            env=env,
            check=True,
        )

    def _forward_process_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self._compiled_model_path is None:
            raise RuntimeError(
                "HunyuanVideo 1.5 process streaming requires load() before forward so "
                "the compiled artifact path is known."
            )

        work_dir = Path(
            os.environ.get(
                "DIFFLET_HUNYUAN15_PROCESS_WORK_DIR",
                f"/tmp/difflet_hunyuan15_process_runtime_{os.getpid()}",
            )
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        runtime_config = work_dir / "runtime_config.json"
        runtime_config.write_text(
            json.dumps(
                self._process_runtime_config(self._compiled_model_path),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        freqs_cos, freqs_sin = image_rotary_emb
        valid_mask = _full_attention_valid_mask(attention_mask, self.meta["latent_seq_len"])
        static = {
            "temb": temb.detach().cpu(),
            "freqs_cos": freqs_cos.detach().cpu(),
            "freqs_sin": freqs_sin.detach().cpu(),
            "valid_mask": valid_mask.detach().cpu(),
        }
        block_metrics = []
        blocks_start = time.perf_counter()
        block_count = int(self.config.num_layers)
        for block_index in range(block_count):
            input_path = work_dir / f"block_{block_index:02d}_input.pt"
            output_path = work_dir / f"block_{block_index:02d}_output.pt"
            self._save_process_tensors(
                input_path,
                {
                    "hidden_states": hidden_states.detach().cpu(),
                    "encoder_hidden_states": encoder_hidden_states.detach().cpu(),
                    **static,
                },
            )
            t0 = time.perf_counter()
            self._run_process_block(
                runtime_config=runtime_config,
                input_tensors=input_path,
                output_tensors=output_path,
                block_index=block_index,
            )
            result = self._load_process_tensors(output_path)
            hidden_states = result["hidden_states"]
            encoder_hidden_states = result["encoder_hidden_states"]
            metrics = dict(result["metrics"])
            metrics["block_wall_elapsed_s"] = time.perf_counter() - t0
            block_metrics.append(metrics)
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.last_metrics = HunyuanVideo15SegmentedMetrics(
            stream_tile_calls=int(sum(m["stream_tile_calls"] for m in block_metrics)),
            stream_forward_elapsed_s=float(
                sum(m["stream_forward_elapsed_s"] for m in block_metrics)
            ),
            block_count=block_count,
            blocks_elapsed_s=time.perf_counter() - blocks_start,
        )
        return self._final_projection(hidden_states.detach().cpu(), temb.detach().cpu())

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
        with torch.no_grad():
            hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb = (
                self._prepare_frontend(
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_r,
                    encoder_hidden_states_2,
                    encoder_attention_mask_2,
                    image_embeds,
                )
            )
            freqs_cos, freqs_sin = image_rotary_emb
            total_tile_calls = 0
            total_attention_elapsed = 0.0
            if self.block_load_mode == "process":
                return self._forward_process_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                )
            valid_mask = _full_attention_valid_mask(attention_mask, self.meta["latent_seq_len"])
            if self.block_load_mode == "streaming":
                block_iter = (
                    (block_index, self.pre_blocks[0], self.post_blocks[0])
                    for block_index in range(int(self.config.num_layers))
                )
            else:
                block_iter = (
                    (block_index, pre, post)
                    for block_index, (pre, post) in enumerate(zip(self.pre_blocks, self.post_blocks))
                )
            for block_index, pre, post in block_iter:
                if self.block_load_mode == "streaming":
                    pre.reload_block_weights(block_index)
                    post.reload_block_weights(block_index)
                query, key, value = pre(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    freqs_cos,
                    freqs_sin,
                )
                attention_states, metrics = run_streaming_manual_stats_attention(
                    self.attention,
                    query.detach().cpu(),
                    key.detach().cpu(),
                    value.detach().cpu(),
                    query_tile_size=self.query_tile_size,
                    key_tile_size=self.key_tile_size,
                    valid_mask=valid_mask,
                )
                total_tile_calls += metrics.stream_tile_calls
                total_attention_elapsed += metrics.stream_forward_elapsed_s
                hidden_states, encoder_hidden_states = post(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_states.to(dtype=self.dtype),
                )
            self.last_metrics = HunyuanVideo15SegmentedMetrics(
                stream_tile_calls=total_tile_calls,
                stream_forward_elapsed_s=total_attention_elapsed,
                block_count=int(self.config.num_layers),
            )
            return self._final_projection(hidden_states.detach().cpu(), temb.detach().cpu())
