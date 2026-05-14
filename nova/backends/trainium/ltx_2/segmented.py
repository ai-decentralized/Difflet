"""Segmented Trainium runtime pieces for LTX-2 transformer blocks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig, NeuronConfig
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from nova.backends.trainium.core.modules.checkpoint import load_state_dict
from nova.backends.trainium.core.multi_component_application import ComponentSpec
from nova.backends.trainium.ltx_2.transformer import (
    LTX2TransformerInferenceConfig,
    _make_audio_coords,
    _make_video_coords,
    _nova_apply_split_rotary_emb,
)
from nova.models.ltx_2.application import LTX_2_DEFAULT_TEXT_SEQ_LEN


DEFAULT_BLOCK_COMPILER_ARGS = (
    "--model-type=transformer -O1 "
    "--tensorizer-options='--enable-ccop-compute-overlap' "
    "--auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)


def _patch_ltx2_rope() -> Any:
    import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

    ltx2_transformer.apply_split_rotary_emb = _nova_apply_split_rotary_emb
    return ltx2_transformer


def _load_block_state_dict_from_dir(
    transformer_dir: Path,
    block_index: int,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    prefix = f"transformer_blocks.{block_index}."
    block_state_dict: dict[str, torch.Tensor] = {}

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


def _block_inner_dim(config: InferenceConfig) -> int:
    return int(config.num_attention_heads) * int(config.attention_head_dim)


def _block_audio_inner_dim(config: InferenceConfig) -> int:
    return int(config.audio_num_attention_heads) * int(config.audio_attention_head_dim)


def _rotary_width(head_dim: int) -> int:
    return int(head_dim) // 2


def _copy_config_loader(config: InferenceConfig):
    values = {
        key: value
        for key, value in config.__dict__.items()
        if key
        not in {
            "neuron_config",
            "fused_spec_config",
            "metadata",
        }
    }

    def _load(target: InferenceConfig) -> None:
        for key, value in values.items():
            setattr(target, key, value)

    return _load


def _make_ltx2_block(config: InferenceConfig) -> nn.Module:
    ltx2_transformer = _patch_ltx2_rope()
    return ltx2_transformer.LTX2VideoTransformerBlock(
        dim=_block_inner_dim(config),
        num_attention_heads=int(config.num_attention_heads),
        attention_head_dim=int(config.attention_head_dim),
        cross_attention_dim=int(config.cross_attention_dim),
        audio_dim=_block_audio_inner_dim(config),
        audio_num_attention_heads=int(config.audio_num_attention_heads),
        audio_attention_head_dim=int(config.audio_attention_head_dim),
        audio_cross_attention_dim=int(config.audio_cross_attention_dim),
        video_gated_attn=bool(getattr(config, "gated_attn", False)),
        video_cross_attn_adaln=bool(getattr(config, "cross_attn_mod", False)),
        audio_gated_attn=bool(getattr(config, "audio_gated_attn", False)),
        audio_cross_attn_adaln=bool(getattr(config, "audio_cross_attn_mod", False)),
        qk_norm=str(getattr(config, "qk_norm", "rms_norm_across_heads")),
        activation_fn=str(getattr(config, "activation_fn", "gelu-approximate")),
        attention_bias=bool(getattr(config, "attention_bias", True)),
        attention_out_bias=bool(getattr(config, "attention_out_bias", True)),
        eps=float(getattr(config, "norm_eps", 1e-6)),
        elementwise_affine=bool(getattr(config, "norm_elementwise_affine", False)),
        rope_type=str(getattr(config, "rope_type", "interleaved")),
        perturbed_attn=bool(getattr(config, "perturbed_attn", False)),
    )


class _LTX2BlockModule(nn.Module):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.block = _make_ltx2_block(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        temb_ca_scale_shift: torch.Tensor,
        temb_ca_audio_scale_shift: torch.Tensor,
        temb_ca_gate: torch.Tensor,
        temb_ca_audio_gate: torch.Tensor,
        video_rotary_cos: torch.Tensor,
        video_rotary_sin: torch.Tensor,
        audio_rotary_cos: torch.Tensor,
        audio_rotary_sin: torch.Tensor,
        ca_video_rotary_cos: torch.Tensor,
        ca_video_rotary_sin: torch.Tensor,
        ca_audio_rotary_cos: torch.Tensor,
        ca_audio_rotary_sin: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            temb=temb,
            temb_audio=temb_audio,
            temb_ca_scale_shift=temb_ca_scale_shift,
            temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
            temb_ca_gate=temb_ca_gate,
            temb_ca_audio_gate=temb_ca_audio_gate,
            temb_prompt=None,
            temb_prompt_audio=None,
            video_rotary_emb=(video_rotary_cos, video_rotary_sin),
            audio_rotary_emb=(audio_rotary_cos, audio_rotary_sin),
            ca_video_rotary_emb=(ca_video_rotary_cos, ca_video_rotary_sin),
            ca_audio_rotary_emb=(ca_audio_rotary_cos, ca_audio_rotary_sin),
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            self_attention_mask=None,
            audio_self_attention_mask=None,
            a2v_cross_attention_mask=None,
            v2a_cross_attention_mask=None,
            use_a2v_cross_attention=True,
            use_v2a_cross_attention=True,
            perturbation_mask=None,
            all_perturbed=False,
        )


class LTX2BlockSegmentConfig(LTX2TransformerInferenceConfig):
    def get_required_attributes(self) -> list[str]:
        return [
            *super().get_required_attributes(),
            "source_model_dir",
            "block_index",
        ]


class LTX2BlockSegmentWrapper(ModelWrapper):
    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
        audio_text_seq_len = int(getattr(self.config, "audio_text_seq_len", text_seq_len))
        inner_dim = _block_inner_dim(self.config)
        audio_inner_dim = _block_audio_inner_dim(self.config)
        video_head_dim = int(self.config.attention_head_dim)
        audio_head_dim = int(self.config.audio_attention_head_dim)
        cross_head_dim = int(self.config.audio_cross_attention_dim) // int(
            self.config.audio_num_attention_heads
        )
        return [
            (
                torch.randn([batch_size, self.config.video_seq_len, inner_dim], dtype=dtype),
                torch.randn([batch_size, self.config.audio_seq_len, audio_inner_dim], dtype=dtype),
                torch.randn([batch_size, text_seq_len, inner_dim], dtype=dtype),
                torch.randn([batch_size, audio_text_seq_len, audio_inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, 6 * inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, 6 * audio_inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, 4 * inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, 4 * audio_inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, inner_dim], dtype=dtype),
                torch.randn([batch_size, 1, audio_inner_dim], dtype=dtype),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.num_attention_heads),
                        self.config.video_seq_len,
                        _rotary_width(video_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.num_attention_heads),
                        self.config.video_seq_len,
                        _rotary_width(video_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.audio_num_attention_heads),
                        self.config.audio_seq_len,
                        _rotary_width(audio_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.audio_num_attention_heads),
                        self.config.audio_seq_len,
                        _rotary_width(audio_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.num_attention_heads),
                        self.config.video_seq_len,
                        _rotary_width(cross_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.num_attention_heads),
                        self.config.video_seq_len,
                        _rotary_width(cross_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.audio_num_attention_heads),
                        self.config.audio_seq_len,
                        _rotary_width(cross_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.randn(
                    [
                        batch_size,
                        int(self.config.audio_num_attention_heads),
                        self.config.audio_seq_len,
                        _rotary_width(cross_head_dim),
                    ],
                    dtype=dtype,
                ),
                torch.zeros([batch_size, 1, text_seq_len], dtype=dtype),
                torch.zeros([batch_size, 1, audio_text_seq_len], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            module = _LTX2BlockModule(self.config)
            module = module.to(dtype=self.config.neuron_config.torch_dtype)
            module.eval()
            return module

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *model_inputs):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(*model_inputs)


class LTX2BlockSegmentApplication(NeuronApplicationBase):
    _model_cls = object

    def __init__(self, *args, compiler_args: str = DEFAULT_BLOCK_COMPILER_ARGS, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_compiled_model_path: str | None = None
        self._loaded_start_rank_id: int | None = None
        self._loaded_local_ranks_size: int | None = None
        self._loaded_block_index = int(self.config.block_index)
        self.model = LTX2BlockSegmentWrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=f"LTX2Block_{self.config.block_index}",
            compiler_args=compiler_args,
            priority_model_idx=0,
        )
        self.models.append(self.model)

    @classmethod
    def get_config_cls(cls):
        return LTX2BlockSegmentConfig

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


class LTX2SegmentedTransformerApplication(nn.Module):
    """Host-stitched LTX-2 transformer runtime using one compiled block artifact."""

    supports_ltx_2_extra_kwargs = False

    def __init__(
        self,
        *,
        model_path: str,
        config: LTX2TransformerInferenceConfig,
        block_compiler_args: str = DEFAULT_BLOCK_COMPILER_ARGS,
        block_load_mode: str = "streaming",
    ) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.config = config
        self.dtype = config.neuron_config.torch_dtype
        self.block_compiler_args = block_compiler_args
        self.block_load_mode = str(block_load_mode)
        self._compiled_model_path: str | None = None
        self._cpu_transformer: nn.Module | None = None
        if self.block_load_mode not in {"streaming", "process"}:
            raise ValueError(
                "LTX-2 segmented block_load_mode must be 'streaming' or 'process', "
                f"got {self.block_load_mode!r}."
            )
        self.block = LTX2BlockSegmentApplication(
            model_path=self.model_path,
            config=self._block_config(config, block_index=0),
            compiler_args=block_compiler_args,
        )

    def _block_config(
        self,
        base: LTX2TransformerInferenceConfig,
        *,
        block_index: int,
    ) -> LTX2BlockSegmentConfig:
        return LTX2BlockSegmentConfig(
            neuron_config=NeuronConfig(
                batch_size=int(getattr(base.neuron_config, "batch_size", 1)),
                tp_degree=base.neuron_config.tp_degree,
                world_size=base.neuron_config.world_size,
                torch_dtype=base.neuron_config.torch_dtype,
                skip_sharding=True,
            ),
            load_config=_copy_config_loader(base),
            source_model_dir=self.model_path,
            block_index=block_index,
        )

    def component_specs(self, prefix: str = "transformer") -> list[ComponentSpec]:
        return [ComponentSpec(f"{prefix}_block", self.block)]

    def set_compiled_model_path(self, compiled_model_path: str) -> None:
        self._compiled_model_path = str(compiled_model_path)

    def _load_cpu_transformer(self) -> nn.Module:
        if self._cpu_transformer is None:
            _patch_ltx2_rope()
            from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

            self._cpu_transformer = LTX2VideoTransformer3DModel.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )
            self._cpu_transformer = self._cpu_transformer.to(dtype=self.dtype).eval()
        return self._cpu_transformer

    def _prepare_frontend(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_encoder_attention_mask: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        model = self._load_cpu_transformer()
        batch_size = hidden_states.shape[0]
        audio_timestep = timestep
        audio_sigma = sigma

        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        audio_encoder_attention_mask = (
            1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
        ) * -10000.0
        audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        video_rotary_emb = model.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = model.audio_rope(audio_coords, device=audio_hidden_states.device)
        video_cross_attn_rotary_emb = model.cross_attn_rope(
            video_coords[:, 0:1, :],
            device=hidden_states.device,
        )
        audio_cross_attn_rotary_emb = model.cross_attn_audio_rope(
            audio_coords[:, 0:1, :],
            device=audio_hidden_states.device,
        )

        hidden_states = model.proj_in(hidden_states)
        audio_hidden_states = model.audio_proj_in(audio_hidden_states)

        timestep_cross_attn_gate_scale_factor = (
            model.config.cross_attn_timestep_scale_multiplier
            / model.config.timestep_scale_multiplier
        )
        temb, embedded_timestep = model.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = model.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(
            batch_size,
            -1,
            audio_embedded_timestep.size(-1),
        )

        video_ca_timestep = timestep.flatten()
        video_cross_attn_scale_shift, _ = model.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = model.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size,
            -1,
            video_cross_attn_scale_shift.shape[-1],
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
            batch_size,
            -1,
            video_cross_attn_a2v_gate.shape[-1],
        )

        audio_ca_timestep = audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = model.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = model.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size,
            -1,
            audio_cross_attn_scale_shift.shape[-1],
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
            batch_size,
            -1,
            audio_cross_attn_v2a_gate.shape[-1],
        )

        if model.config.use_prompt_embeddings:
            encoder_hidden_states = model.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size,
                -1,
                hidden_states.size(-1),
            )
            audio_encoder_hidden_states = model.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size,
                -1,
                audio_hidden_states.size(-1),
            )

        return (
            hidden_states.to(dtype=self.dtype).contiguous(),
            audio_hidden_states.to(dtype=self.dtype).contiguous(),
            encoder_hidden_states.to(dtype=self.dtype).contiguous(),
            audio_encoder_hidden_states.to(dtype=self.dtype).contiguous(),
            temb.to(dtype=self.dtype).contiguous(),
            temb_audio.to(dtype=self.dtype).contiguous(),
            video_cross_attn_scale_shift.to(dtype=self.dtype).contiguous(),
            audio_cross_attn_scale_shift.to(dtype=self.dtype).contiguous(),
            video_cross_attn_a2v_gate.to(dtype=self.dtype).contiguous(),
            audio_cross_attn_v2a_gate.to(dtype=self.dtype).contiguous(),
            *(t.to(dtype=self.dtype).contiguous() for t in video_rotary_emb),
            *(t.to(dtype=self.dtype).contiguous() for t in audio_rotary_emb),
            *(t.to(dtype=self.dtype).contiguous() for t in video_cross_attn_rotary_emb),
            *(t.to(dtype=self.dtype).contiguous() for t in audio_cross_attn_rotary_emb),
            encoder_attention_mask.to(dtype=self.dtype).contiguous(),
            audio_encoder_attention_mask.to(dtype=self.dtype).contiguous(),
            embedded_timestep.to(dtype=self.dtype).contiguous(),
            audio_embedded_timestep.to(dtype=self.dtype).contiguous(),
        )

    def _final_projection(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        audio_embedded_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._load_cpu_transformer()
        scale_shift_values = model.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = model.norm_out(hidden_states.to(dtype=embedded_timestep.dtype))
        hidden_states = hidden_states * (1 + scale) + shift
        output = model.proj_out(hidden_states)

        audio_scale_shift_values = (
            model.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
        audio_hidden_states = model.audio_norm_out(
            audio_hidden_states.to(dtype=audio_embedded_timestep.dtype)
        )
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = model.audio_proj_out(audio_hidden_states)
        return output, audio_output

    def _process_runtime_config(self, compiled_model_path: str) -> dict[str, Any]:
        return {
            "model_dir": str(Path(self.model_path).parent),
            "compiled_model_path": str(compiled_model_path),
            "dtype": "bf16" if self.dtype is torch.bfloat16 else "fp32",
            "tp_degree": int(self.config.neuron_config.tp_degree),
            "height": int(self.config.height),
            "width": int(self.config.width),
            "num_frames": int(self.config.num_frames),
            "audio_num_frames": int(self.config.audio_num_frames),
            "text_seq_len": int(getattr(self.config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN)),
            "audio_text_seq_len": int(
                getattr(
                    self.config,
                    "audio_text_seq_len",
                    getattr(self.config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN),
                )
            ),
            "frame_rate": float(getattr(self.config, "frame_rate", 24.0)),
        }

    @staticmethod
    def _save_process_tensors(path: Path, tensors: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, path)

    @staticmethod
    def _load_process_tensors(path: Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)

    def _run_process_block(
        self,
        *,
        runtime_config: Path,
        input_tensors: Path,
        output_tensors: Path,
        block_index: int,
    ) -> None:
        script_path = (
            Path(__file__).resolve().parents[4] / "scripts" / "ltx_2_segmented_process_block.py"
        )
        if not script_path.exists():
            raise FileNotFoundError(
                "LTX-2 segmented process mode requires "
                f"{script_path}; use block_load_mode='streaming' in packaged installs."
            )
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[4])
        env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env.setdefault("NOVA_BACKEND", "trainium")
        env["LOCAL_WORLD_SIZE"] = str(int(self.config.neuron_config.tp_degree))
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--runtime-config",
                str(runtime_config),
                "--input-tensors",
                str(input_tensors),
                "--output-tensors",
                str(output_tensors),
                "--block-index",
                str(block_index),
            ],
            env=env,
            check=True,
        )

    def _forward_process_block(
        self,
        *,
        runtime_config: Path,
        work_dir: Path,
        block_index: int,
        block_inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = work_dir / f"ltx2_block_{block_index:02d}_input.pt"
        output_path = work_dir / f"ltx2_block_{block_index:02d}_output.pt"
        self._save_process_tensors(
            input_path,
            {"inputs": [tensor.detach().cpu() for tensor in block_inputs]},
        )
        self._run_process_block(
            runtime_config=runtime_config,
            input_tensors=input_path,
            output_tensors=output_path,
            block_index=block_index,
        )
        result = self._load_process_tensors(output_path)
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result["hidden_states"], result["audio_hidden_states"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_encoder_attention_mask: torch.Tensor,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise NotImplementedError(f"LTX-2 segmented Trainium blocks do not support {names}.")
        batch_size = hidden_states.shape[0]
        if video_coords is None:
            video_coords = _make_video_coords(
                batch_size=batch_size,
                num_frames=int(self.config.latent_num_frames),
                height=int(self.config.latent_height),
                width=int(self.config.latent_width),
                patch_size=int(self.config.patch_size),
                patch_size_t=int(self.config.patch_size_t),
                scale_factors=tuple(self.config.vae_scale_factors),
                causal_offset=int(getattr(self.config, "causal_offset", 1)),
                fps=float(getattr(self.config, "frame_rate", 24.0)),
            )
        if audio_coords is None:
            audio_coords = _make_audio_coords(
                batch_size=batch_size,
                audio_num_frames=int(self.config.audio_num_frames),
                patch_size_t=int(self.config.audio_patch_size_t),
                scale_factor=int(self.config.audio_scale_factor),
                causal_offset=int(getattr(self.config, "causal_offset", 1)),
                sampling_rate=int(self.config.audio_sampling_rate),
                hop_length=int(self.config.audio_hop_length),
            )

        with torch.no_grad():
            (
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                temb_ca_scale_shift,
                temb_ca_audio_scale_shift,
                temb_ca_gate,
                temb_ca_audio_gate,
                video_rotary_cos,
                video_rotary_sin,
                audio_rotary_cos,
                audio_rotary_sin,
                ca_video_rotary_cos,
                ca_video_rotary_sin,
                ca_audio_rotary_cos,
                ca_audio_rotary_sin,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                embedded_timestep,
                audio_embedded_timestep,
            ) = self._prepare_frontend(
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                timestep,
                sigma,
                encoder_attention_mask,
                audio_encoder_attention_mask,
                video_coords,
                audio_coords,
            )

            process_runtime_config = None
            process_work_dir = None
            if self.block_load_mode == "process":
                if self._compiled_model_path is None:
                    raise RuntimeError(
                        "LTX-2 segmented process mode requires load() before forward so "
                        "the compiled artifact path is known."
                    )
                process_work_dir = Path(
                    os.environ.get(
                        "NOVA_LTX2_PROCESS_WORK_DIR",
                        f"/tmp/nova_ltx2_process_runtime_{os.getpid()}",
                    )
                )
                process_work_dir.mkdir(parents=True, exist_ok=True)
                process_runtime_config = process_work_dir / "runtime_config.json"
                process_runtime_config.write_text(
                    json.dumps(
                        self._process_runtime_config(self._compiled_model_path),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            for block_index in range(int(self.config.num_layers)):
                block_inputs = (
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    temb_ca_scale_shift,
                    temb_ca_audio_scale_shift,
                    temb_ca_gate,
                    temb_ca_audio_gate,
                    video_rotary_cos,
                    video_rotary_sin,
                    audio_rotary_cos,
                    audio_rotary_sin,
                    ca_video_rotary_cos,
                    ca_video_rotary_sin,
                    ca_audio_rotary_cos,
                    ca_audio_rotary_sin,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                )
                if self.block_load_mode == "process":
                    assert process_runtime_config is not None and process_work_dir is not None
                    hidden_states, audio_hidden_states = self._forward_process_block(
                        runtime_config=process_runtime_config,
                        work_dir=process_work_dir,
                        block_index=block_index,
                        block_inputs=block_inputs,
                    )
                else:
                    self.block.reload_block_weights(block_index)
                    hidden_states, audio_hidden_states = self.block(*block_inputs)
                hidden_states = hidden_states.detach().cpu().to(dtype=self.dtype)
                audio_hidden_states = audio_hidden_states.detach().cpu().to(dtype=self.dtype)

            return self._final_projection(
                hidden_states,
                audio_hidden_states,
                embedded_timestep.detach().cpu(),
                audio_embedded_timestep.detach().cpu(),
            )
