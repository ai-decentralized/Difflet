"""Qwen-Image TeaCache fused-A probe (cclog 81).

Same recipe as the HunyuanVideo fused probe (cclog 80): prev_mod is a
persistent on-device ``nn.Parameter`` updated in place via
``input_output_aliases``; forward returns ``(delta, mod_input)`` and only the
4-byte delta reaches host. The block-0 modulated input comes from
``_QwenImageTransformerTraceModule.teacache_mod_input`` (CPU-parity verified
bit-exact vs the diffusers forward).

The only per-architecture difference from HV is the block-0 hook (Qwen:
img_in -> time_text_embed -> block0 img_mod/img_norm1/_modulate). The alias
machinery (nn.Parameter, distinct extra output, custom ModelInstance.get())
is identical.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from neuronx_distributed.trace.model_builder import BaseModelInstance
from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import ModelWrapper
from nova.backends.trainium.qwen_image.transformer import (
    QwenImageTransformerInferenceConfig,
    _QwenImageTransformerTraceModule,
)


class QwenImageTeacacheProbeFusedModel(nn.Module):
    """prev_mod nn.Parameter; forward returns (delta, mod_input), mod_input
    aliased back to prev_mod."""

    def __init__(self, config, *, seq_len: int, inner_dim: int, batch_size: int = 1) -> None:
        super().__init__()
        self.trace_module = _QwenImageTransformerTraceModule(config)
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del encoder_hidden_states_mask
        mod_input = self.trace_module.teacache_mod_input(
            hidden_states, timestep, encoder_hidden_states, guidance
        )
        # cclog 83: relative-L1 signal (the TeaCache-paper / vLLM-Omni metric),
        # not absolute L2. mean|mod-prev| / mean|prev| is dimensionless O(0.05-
        # 0.3), which divides out the flow-matching latent-magnitude trend that
        # made the absolute-L2 signal a weak predictor on Qwen (cclog 82).
        # Reduce in float32; mod_input (out[1]) stays bf16 to keep the alias into
        # the bf16 prev_mod Parameter.
        m = mod_input.float()
        p = self.prev_mod.float()
        rel_l1 = (m - p).abs().mean() / (p.abs().mean() + 1e-8)
        return rel_l1, mod_input


class _QwenFusedProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # out 0 = delta (kept); out 1 = mod_input aliased to prev_mod (in-place).
        return self.module, {self.module.prev_mod: 1}


def _seq_len(config) -> int:
    return int(config.packed_height) * int(config.packed_width)


class ModelWrapperQwenImageTeacacheProbeFused(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, model_init_kwargs=None):
        super().__init__(config, model_cls, tag, compiler_args,
                         priority_model_idx, model_init_kwargs or {})
        self.bucket_config = None

    def input_generator(self):
        bs = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        tsl = int(getattr(self.config, "text_seq_len", 1024))
        seq = _seq_len(self.config)
        return [
            (
                torch.randn([bs, seq, self.config.in_channels], dtype=dtype),
                torch.ones([bs], dtype=dtype),
                torch.randn([bs, tsl, self.config.joint_attention_dim], dtype=dtype),
                torch.ones([bs, tsl], dtype=torch.int64),
                torch.ones([bs], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = self.config.neuron_config.torch_dtype
        seq = _seq_len(config)
        inner = int(config.num_attention_heads) * int(config.attention_head_dim)
        bs = int(getattr(config.neuron_config, "batch_size", 1))

        def _create():
            m = QwenImageTeacacheProbeFusedModel(
                config, seq_len=seq, inner_dim=inner, batch_size=bs
            )
            return m.to(dtype=dtype).eval()

        return _QwenFusedProbeModelInstance(module_builder=_create)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, guidance):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        return self._forward(
            hidden_states, timestep, encoder_hidden_states,
            encoder_hidden_states_mask, guidance,
        )


class NeuronQwenImageTeacacheProbeFusedApplication(NeuronApplicationBase):
    """Qwen-Image fused-A probe app."""

    _model_cls = QwenImageTeacacheProbeFusedModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperQwenImageTeacacheProbeFused
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return QwenImageTransformerInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        # the fused model nests the diffusers model under
        # trace_module.transformer.* ; prev_mod stays the init Parameter.
        del config
        return {f"trace_module.transformer.{k}": v for k, v in state_dict.items()}

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(self, hidden_states, timestep, encoder_hidden_states,
                       encoder_hidden_states_mask, guidance):
        out = self.models[0](
            hidden_states, timestep, encoder_hidden_states,
            encoder_hidden_states_mask, guidance,
        )
        return out[0] if isinstance(out, (tuple, list)) else out
