"""Neuron application for Wan's VAE decoder.

Wraps the backend-neutral ``WanVAEDecoderModel`` (``difflet/models/wan/vae/modeling_vae.py``),
which mirrors diffusers' decoder key for key (``post_quant_conv.*``, ``decoder.*``) and decodes
the latent frame by frame with diffusers' causal convolution cache. The checkpoint's encoder
tensors match no parameter and are skipped.

Precision and placement: fp32, as upstream decodes (Wan's model cards load the VAE in fp32 under
a bf16 pipeline) and as the host decode does; the decoder writes pixels directly and runs once
per request. It has no tensor-parallel layers, so every rank that runs it holds the whole 0.5 GB
decoder; the CLI runs it on one rank, in a launch after the transformer stage (its fp32
activations are about 12 GiB per core at 480x832), as the Trainium path runs its VAE stage.

Eager only: the decode loop threads a mutable per-convolution frame cache through the blocks,
which compiled regions would have to guard on every frame. ``exec_mode`` is accepted for the
common constructor and must be ``"eager"``.
"""

from __future__ import annotations

import json
import os

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class NeuronWanVaeDecoderApplication(TorchNeuronApplicationBase):
    """The whole Wan VAE decoder on each rank, at one static latent shape, in fp32, eager.

    ``model_path`` is the VAE's directory (``<model>/vae``); ``latent_shape`` is the
    ``(batch, z_dim, latent_frames, latent_height, latent_width)`` the warm-up decodes.
    """

    block_attrs = ()
    #: The checkpoint's encoder half, unused by decoding.
    skipped_checkpoint_prefixes = ("encoder.", "quant_conv.")

    def __init__(
        self,
        *,
        model_path,
        latent_shape: tuple[int, ...],
        parallel=None,
        dtype=torch.float32,
        exec_mode: str | None = "eager",
        device="neuron",
        **kwargs,
    ) -> None:
        if exec_mode not in (None, "eager"):
            raise ValueError(f"the Wan VAE decoder runs eager only, got exec_mode={exec_mode!r}")
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode="eager",
            device=device,
            **kwargs,
        )
        self.latent_shape = tuple(int(n) for n in latent_shape)
        self.vae_config = self._read_config()
        if len(self.latent_shape) != 5 or self.latent_shape[1] != int(self.vae_config.z_dim):
            raise ValueError(
                f"latent_shape {self.latent_shape} must be (batch, z_dim={self.vae_config.z_dim}, "
                "frames, height, width)"
            )

    def _read_config(self):
        from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

        path = os.path.join(str(self.model_path), "config.json")
        with open(path, encoding="utf-8") as f:
            return WanVAEDecoderConfig.from_diffusers_dict(json.load(f))

    def build_module(self) -> torch.nn.Module:
        from difflet.models.wan.vae.modeling_vae import WanVAEDecoderModel

        return WanVAEDecoderModel(self.vae_config)

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' decoder layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(self.latent_shape, dtype=self.dtype),)


__all__ = ["NeuronWanVaeDecoderApplication"]
