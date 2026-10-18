"""Neuron application for Flux's VAE decoder.

Builds diffusers' 2D ``Decoder`` from the VAE config the way the Trainium path does
(``get_decoder_config`` in ``difflet/models/flux/vae/modeling_vae.py``), plus ``post_quant_conv``
when the config uses one (FLUX.1-dev's does not). The checkpoint is diffusers' ``AutoencoderKL``
layout key for key; its encoder half is skipped.

fp32 and eager, on one rank: it runs once per request, has no tensor-parallel layers (the whole
~0.2 GB decoder per rank that runs it), and its 1024x1024 activations fit one core. ``decode``
takes latents already unscaled (``latents / scaling_factor + shift_factor``), as diffusers'
``AutoencoderKL.decode`` does.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class _FluxVaeDecoder(nn.Module):
    def __init__(self, decoder_kwargs: dict, latent_channels: int, post_quant_conv: bool) -> None:
        super().__init__()
        from diffusers.models.autoencoders.vae import Decoder

        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if post_quant_conv else None
        self.decoder = Decoder(**decoder_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)


class NeuronFluxVaeDecoderApplication(TorchNeuronApplicationBase):
    """Flux's VAE decoder at one static image shape, fp32, eager.

    ``model_path`` is the VAE's directory (``<model>/vae``).
    """

    block_attrs = ()
    #: The checkpoint's encoder half, unused by decoding.
    skipped_checkpoint_prefixes = ("encoder.", "quant_conv.")

    def __init__(self, *, model_path, height: int = 1024, width: int = 1024, batch_size: int = 1,
                 parallel=None, dtype=torch.float32, exec_mode: str | None = "eager",
                 device="neuron", **kwargs) -> None:
        if exec_mode not in (None, "eager"):
            raise ValueError(f"the Flux VAE decoder runs eager only, got exec_mode={exec_mode!r}")
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype, exec_mode="eager",
                         device=device, **kwargs)
        from diffusers.models.autoencoders.vae import Decoder

        from difflet.models.flux.vae.modeling_vae import get_decoder_config, get_vae_scale_factor

        with open(os.path.join(str(self.model_path), "config.json"), encoding="utf-8") as f:
            self.vae_config = json.load(f)
        self.decoder_kwargs = get_decoder_config(Decoder, self.vae_config, int(height), int(width))
        self.scale_factor = get_vae_scale_factor(Decoder, self.decoder_kwargs)
        self.latent_channels = int(self.vae_config.get("latent_channels", 16))
        self.latent_shape = (int(batch_size), self.latent_channels,
                             int(height) // self.scale_factor, int(width) // self.scale_factor)

    def build_module(self) -> torch.nn.Module:
        return _FluxVaeDecoder(self.decoder_kwargs, self.latent_channels,
                               bool(self.vae_config.get("use_post_quant_conv", True)))

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' AutoencoderKL layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(self.latent_shape, dtype=self.dtype),)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Images in [-1, 1], ``[batch, 3, height, width]``, from unscaled latents."""
        return self(latents.to(self.device, self.dtype))


__all__ = ["NeuronFluxVaeDecoderApplication"]
