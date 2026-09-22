"""Wan block-0 TeaCache probe with persistent device state and scalar host output."""

import torch
import torch.nn as nn
from diffusers.models.normalization import FP32LayerNorm

from difflet.backends.trainium.core.bucketing import (
    ShapeBucketedInputGenerator,
    resolve_compile_shapes,
)
from difflet.backends.trainium.core.teacache_probe import (
    FusedProbeApplication,
    FusedProbeWrapper,
    RelativeL1Probe,
)
from difflet.backends.trainium.wan.backbone import WanBackboneInferenceConfig
from difflet.models.wan.modeling_wan import WanTimeTextEmbedding


def _seq_len(config, shape):
    height, width, frames = shape
    pt, ph, pw = config.patch_size
    return (int(frames) // pt) * (height // 8 // ph) * (width // 8 // pw)


class WanTeacacheProbeFusedModel(RelativeL1Probe):
    def __init__(self, config):
        inner = config.inner_dim
        super().__init__(
            batch_size=config.neuron_config.batch_size,
            seq_len=max(_seq_len(config, s) for s in resolve_compile_shapes(config)),
            inner_dim=inner,
        )
        # Canonical checkpoint names, without constructing attention or FFNs.
        self.patch_embedding = nn.Conv3d(
            config.in_channels, inner, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.condition_embedder = WanTimeTextEmbedding(
            inner, config.freq_dim, 6 * inner, config.text_dim
        )
        del self.condition_embedder.text_embedder
        block = nn.Module()
        block.norm1 = FP32LayerNorm(inner, getattr(config, "eps", 1e-6), elementwise_affine=False)
        block.scale_shift_table = nn.Parameter(torch.zeros(1, 6, inner))
        self.blocks = nn.ModuleList([block])

    def teacache_mod_input(self, hidden_states, timestep):
        h = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        embedder = self.condition_embedder
        t = embedder.timesteps_proj(timestep).to(h.dtype)
        temb = embedder.time_embedder(t).type_as(h)
        modulation = embedder.time_proj(embedder.act_fn(temb)).unflatten(1, (6, -1))
        block = self.blocks[0]
        shift, scale = (block.scale_shift_table + modulation.float()).chunk(6, dim=1)[:2]
        return (block.norm1(h.float()) * (1 + scale) + shift).type_as(h)

    def forward(self, hidden_states, timestep):
        return self.delta_and_state(self.teacache_mod_input(hidden_states, timestep))


class ModelWrapperWanTeacacheProbeFused(ShapeBucketedInputGenerator, FusedProbeWrapper):
    def example_inputs_for_shape(self, shape):
        height, width, frames = shape
        bs = self.config.neuron_config.batch_size
        dtype = self.config.neuron_config.torch_dtype
        return (
            torch.randn(
                bs, self.config.in_channels, int(frames), height // 8, width // 8, dtype=dtype
            ),
            torch.ones(bs, dtype=dtype),
        )


class NeuronWanTeacacheProbeFusedApplication(FusedProbeApplication):
    _model_cls = WanTeacacheProbeFusedModel
    wrapper_cls = ModelWrapperWanTeacacheProbeFused
    weight_prefixes = (
        "patch_embedding.",
        "condition_embedder.time_embedder.",
        "condition_embedder.time_proj.",
        "blocks.0.norm1.",
        "blocks.0.scale_shift_table",
    )

    @classmethod
    def get_config_cls(cls):
        return WanBackboneInferenceConfig

    @classmethod
    def convert_hf_to_neuron_state_dict(cls, state_dict, config):
        return {k: v for k, v in state_dict.items() if k.startswith(cls.weight_prefixes)}
