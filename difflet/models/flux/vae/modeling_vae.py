# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# This implementation is derived from the Diffusers and VAE library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/models/diffusers/flux/vae/modeling_vae.py
# Fork date: 2026-05-08
# Modifications:
#   2026-07-30 _model_cls:
#     * support TAEF1 via DecoderTiny; DecoderTiny has no GroupNorm so skip
#       the PatchedGroupNorm monkey-patch; its constructor signature differs
#       from the standard Decoder — use get_decoder_config() to dispatch.
# <<< NxDI fork banner <<<
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F, init

from diffusers.models.autoencoders.vae import DecoderTiny


# Replace torch.nn.GroupNorm with PatchedGroupNorm when running the decoder in bfloat16.
class PatchedGroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": torch.float32}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        out_dtype = input.dtype
        return F.group_norm(input.to(torch.float32), self.num_groups, self.weight, self.bias, self.eps).to(out_dtype)

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )


def get_decoder_config(model_cls, load_config_dict: dict, height: int, width: int,
                       transformer_in_channels: int | None = None) -> dict:
    """Build the decoder constructor kwargs dict for standard Decoder or DecoderTiny.

    The standard diffusers ``Decoder`` expects::

        in_channels, out_channels, up_block_types, block_out_channels,
        layers_per_block, norm_num_groups, act_fn, mid_block_add_attention

    ``DecoderTiny`` (TAEF1) expects::

        in_channels, out_channels, num_blocks, block_out_channels,
        upsampling_scaling_factor, act_fn, upsample_fn
    """
    if model_cls is DecoderTiny:
        latent_channels = load_config_dict.get("latent_channels", 16)
        return {
            "in_channels": latent_channels,
            "out_channels": load_config_dict.get("out_channels", 3),
            "num_blocks": load_config_dict.get("num_decoder_blocks", [3, 3, 3, 1]),
            "block_out_channels": load_config_dict.get("decoder_block_out_channels", [64, 64, 64, 64]),
            "upsampling_scaling_factor": load_config_dict.get("upsampling_scaling_factor", 2),
            "act_fn": load_config_dict.get("act_fn", "relu"),
            "upsample_fn": load_config_dict.get("upsample_fn", "nearest"),
        }

    # Standard Decoder
    return {
        "in_channels": load_config_dict.get("latent_channels", 16),
        "out_channels": load_config_dict.get("out_channels", 3),
        "up_block_types": load_config_dict.get("up_block_types", []),
        "block_out_channels": load_config_dict.get("block_out_channels", []),
        "layers_per_block": load_config_dict.get("layers_per_block", 1),
        "norm_num_groups": load_config_dict.get("norm_num_groups", 32),
        "act_fn": load_config_dict.get("act_fn", "silu"),
        "mid_block_add_attention": load_config_dict.get("mid_block_add_attention", True),
    }


def get_vae_scale_factor(model_cls, decoder_config: dict) -> int:
    """Return the VAE's total downsampling factor (always 8 for FLUX-compatible VAEs)."""
    if model_cls is DecoderTiny:
        # DecoderTiny has len(num_blocks) stages; each stage except the last
        # doubles resolution. 4 stages → 3 upsamples → 8x.
        num_stages = len(decoder_config.get("num_blocks", [3, 3, 3, 1]))
        return 2 ** (num_stages - 1)
    return 2 ** (len(decoder_config.get("block_out_channels", [])) - 1)


# The Trainium (AoT) application, inference config and model wrapper live in
# difflet.backends.trainium.flux.vae; re-exported lazily so existing imports keep working while
# importing the model does not load the AoT backend.
_AOT_EXPORTS = ('VAEDecoderInferenceConfig', 'ModelWrapperVAEDecoder', 'NeuronVAEDecoderApplication')


def __getattr__(name):
    if name in _AOT_EXPORTS:
        from difflet.backends.trainium.flux import vae

        return getattr(vae, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
