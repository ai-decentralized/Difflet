"""LTX-2 video TeaCache prefix, shared by single and segmented DiT runtimes."""

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2VideoTransformerBlock,
)

from difflet.backends.trainium.core.teacache_probe import (
    FusedProbeApplication,
    FusedProbeWrapper,
    RelativeL1Probe,
)
from difflet.backends.trainium.ltx_2.transformer import LTX2TransformerInferenceConfig


class LTX2TeacacheProbeFusedModel(RelativeL1Probe):
    def __init__(self, config):
        inner = int(config.num_attention_heads) * int(config.attention_head_dim)
        super().__init__(
            batch_size=config.neuron_config.batch_size,
            seq_len=config.video_seq_len,
            inner_dim=inner,
        )
        self.transformer = nn.Module()
        self.transformer.proj_in = nn.Linear(config.in_channels, inner)
        num_mod_params = 9 if getattr(config, "cross_attn_mod", False) else 6
        self.transformer.time_embed = LTX2AdaLayerNormSingle(inner, num_mod_params=num_mod_params)
        block = nn.Module()
        block.norm1 = RMSNorm(
            inner,
            eps=float(getattr(config, "norm_eps", 1e-6)),
            elementwise_affine=bool(getattr(config, "norm_elementwise_affine", False)),
        )
        block.scale_shift_table = nn.Parameter(torch.zeros(num_mod_params, inner))
        self.transformer.transformer_blocks = nn.ModuleList([block])

    def teacache_mod_input(self, hidden_states, timestep):
        model = self.transformer
        batch_size = hidden_states.shape[0]
        h = model.proj_in(hidden_states)
        temb, _ = model.time_embed(timestep.flatten(), batch_size=batch_size, hidden_dtype=h.dtype)
        temb = temb.view(batch_size, -1, temb.size(-1))
        block = model.transformer_blocks[0]
        shift, scale = LTX2VideoTransformerBlock.get_mod_params(
            block.scale_shift_table, temb, batch_size
        )[:2]
        return block.norm1(h) * (1 + scale) + shift

    def forward(self, hidden_states, timestep):
        return self.delta_and_state(self.teacache_mod_input(hidden_states, timestep))


class ModelWrapperLTX2TeacacheProbeFused(FusedProbeWrapper):
    def input_generator(self):
        bs = self.config.neuron_config.batch_size
        dtype = self.config.neuron_config.torch_dtype
        return [
            (
                torch.randn(bs, self.config.video_seq_len, self.config.in_channels, dtype=dtype),
                torch.ones(bs, dtype=dtype),
            )
        ]


class NeuronLTX2TeacacheProbeFusedApplication(FusedProbeApplication):
    _model_cls = LTX2TeacacheProbeFusedModel
    wrapper_cls = ModelWrapperLTX2TeacacheProbeFused
    weight_prefixes = (
        "proj_in.",
        "time_embed.",
        "transformer_blocks.0.norm1.",
        "transformer_blocks.0.scale_shift_table",
    )

    @classmethod
    def get_config_cls(cls):
        return LTX2TransformerInferenceConfig

    @classmethod
    def convert_hf_to_neuron_state_dict(cls, state_dict, config):
        return {
            f"transformer.{k}": v
            for k, v in state_dict.items()
            if k.startswith(cls.weight_prefixes)
        }
