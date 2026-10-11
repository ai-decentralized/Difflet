# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
"""Neuron-only Wan decoder operators and persistent temporal state.

Operator choices follow aws-neuron/vllm-omni-neuron cb7cf2b33968f271174803cc39eefbac909cfdec,
vllm_omni_neuron/diffusion/models/wan2_2/autoencoder_kl_wan.py.
The model, checkpoint names and causal-cache transitions remain owned by Difflet.
NxD input/output aliases own device state; no history tensors cross the host API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.models.wan.vae.modeling_vae import (
    WanAttentionBlock,
    WanCausalConv3d,
    WanRMSNorm,
    WanUpsample,
    WanVAEDecoderModel,
    wan_vae_decode_cache,
)


class NeuronWanRMSNorm(WanRMSNorm):
    def forward(self, x):
        dim = 1 if self.channel_first else -1
        rms = x.float().square().mean(dim, keepdim=True).clamp(min=1e-8).rsqrt()
        return (x * rms).to(self.gamma.dtype) * self.gamma + self.bias


class NeuronWanUpsample(WanUpsample):
    def forward(self, x):
        return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


class NeuronWanAttentionBlock(WanAttentionBlock):
    def forward(self, x):
        from difflet.backends.trainium.ops_impl.attention import attention
        import nki.language as nl

        identity = x
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        q, k, v = self.to_qkv(self.norm(x)).flatten(2).transpose(1, 2).chunk(3, dim=-1)
        x = attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            scale=channels**-0.5, tp_q=True, tp_k=True, tp_out=False,
            softmax_dtype=nl.float32, mm_out_dtype=nl.float32,
        )
        x = x.transpose(1, 2).reshape(batch * time, channels, height, width)
        x = self.proj(x).reshape(batch, time, channels, height, width)
        return x.permute(0, 2, 1, 3, 4) + identity


class NeuronWanCausalConv3d(WanCausalConv3d):
    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and padding[4] > 0:
            x = torch.cat((cache_x, x), dim=2)
            padding[4] -= cache_x.shape[2]
        if (
            self.logical_nc_config == 2
            and self.kernel_size == (3, 3, 3)
            and self.stride == self.dilation == (1, 1, 1)
            and self.groups == 1 and self.bias is not None
            and min(self.in_channels, self.out_channels) >= 96
            and padding == [1, 1, 1, 1, 0, 0]
            and x.shape[2] - 2 >= 2
        ):
            from nkilib.experimental.conv.conv3d import conv3d

            return conv3d[2](
                x, self.weight.permute(2, 3, 4, 1, 0).contiguous(), self.bias,
                padding=(0, 0, 1, 1, 1, 1), lnc_shard=True,
            )
        return F.conv3d(F.pad(x, padding), self.weight, self.bias,
                        self.stride, 0, self.dilation, self.groups)


def _optimize_decoder(module, logical_nc_config):
    # Change only this newly constructed Neuron model. Preserve parameter names
    # and the upstream block/control flow instead of maintaining another decoder.
    replacements = {
        WanRMSNorm: NeuronWanRMSNorm,
        WanUpsample: NeuronWanUpsample,
        WanAttentionBlock: NeuronWanAttentionBlock,
        WanCausalConv3d: NeuronWanCausalConv3d,
    }
    for child in module.modules():
        replacement = replacements.get(type(child))
        if replacement is not None:
            child.__class__ = replacement
            child.logical_nc_config = logical_nc_config


class NeuronWanVAEDecoderStepModel(WanVAEDecoderModel):
    """One tile/frame, sharing aliased cache storage across all phase graphs.

    All graphs share maximum tile state shapes. Two banks alternate reads and
    writes so updating history cannot overwrite a sliced temporal/spatial view
    still consumed by the decoder. Edge graphs retain their true dimensions for
    attention. The first phase replaces the complete bank used by the next step.
"""

    def __init__(self, config, step, tile_shape, max_tile_shape, batch_size=1,
                 logical_nc_config=1):
        super().__init__(config)
        self.step = min(step, 2)
        self.write_bank = step % 2
        self.read_bank = 1 - self.write_bank
        shape = (batch_size, config.z_dim, 1, *tile_shape)
        layout = wan_vae_decode_cache(config, shape, self.step, torch.float32)
        maximum = wan_vae_decode_cache(
            config, (batch_size, config.z_dim, 1, *max_tile_shape), 2, torch.float32,
        )
        self.cache_layout = tuple(t.shape if isinstance(t, torch.Tensor) else t for t in layout)
        self.cache_indices = tuple(i for i, t in enumerate(maximum) if isinstance(t, torch.Tensor))
        self.decode_cache = nn.ParameterList([
            nn.Parameter(torch.zeros(maximum[i].shape), requires_grad=False)
            for _ in range(2) for i in self.cache_indices
        ])
        _optimize_decoder(self, logical_nc_config)

    def forward(self, z, phase_marker):
        cache = list(self.cache_layout)
        count = len(self.cache_indices)
        read_states = self.decode_cache[self.read_bank * count:(self.read_bank + 1) * count]
        for index, state in zip(self.cache_indices, read_states):
            layout = self.cache_layout[index]
            if isinstance(layout, torch.Size):
                # Preserve the original first/second/steady history shapes.
                cache[index] = state[:, :, state.shape[2] - layout[2]:, :layout[3], :layout[4]]
        chunk = self.decoder(self.post_quant_conv(z), feat_cache=cache,
                             feat_idx=[0], first_chunk=self.step == 0)
        updated = list(self.decode_cache)
        for position, index in enumerate(self.cache_indices):
            destination = self.write_bank * count + position
            state = self.decode_cache[destination]
            value = cache[index]
            if isinstance(value, torch.Tensor):
                value = F.pad(value, (0, state.shape[4] - value.shape[4],
                                     0, state.shape[3] - value.shape[3],
                                     state.shape[2] - value.shape[2], 0))
            else:
                value = torch.zeros_like(state)
            # Keep aliased inputs visible in the first graph as well. The marker
            # is always true at runtime; its length routes the phase/bank pair.
            updated[destination] = torch.where(phase_marker[0], value, state)
        return chunk, *updated
