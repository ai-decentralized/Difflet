"""Neuron application for LTX-2's video VAE decoder.

Wraps diffusers' own ``LTX2VideoDecoder3d`` (built from the checkpoint's ``vae/config.json`` the
way ``AutoencoderKLLTX2Video`` builds it, so its parameter names are the checkpoint's
``decoder.*`` keys); the encoder half and the ``latents_mean``/``latents_std`` buffers are
skipped (the caller un-normalises on the host, as ``LTX2Orchestrator._decode_latents`` does).
The whole latent clip is decoded in one call, without diffusers' tiling (the checkpoint's VAE
has ``timestep_conditioning: false`` and no noise injection, so the decode is a pure function of
the latents).

Precision and placement: bf16 by default, the dtype the Trainium path decodes in (its host
``LTX2Pipeline`` is loaded in the pipeline dtype); fp32 is accepted. It has no tensor-parallel
layers, so the rank that runs it holds the whole decoder; the CLI runs it on one rank, in a
launch after the generate stage, as the Wan and Qwen-Image vae stages run.

Eager only: the decoder is one stack of 3D convolutions run once per request, with no repeated
block a per-block compile would share. ``exec_mode`` is accepted for the common constructor and
must be ``"eager"``.
"""

from __future__ import annotations

import json
import os

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class _Ltx2VideoDecoder(torch.nn.Module):
    """``decoder`` is diffusers' ``LTX2VideoDecoder3d``; ``forward(z)`` is its call without
    timestep conditioning, causal as configured (``decoder_causal``)."""

    def __init__(self, vae_config: dict) -> None:
        super().__init__()
        from diffusers import AutoencoderKLLTX2Video

        # Built through the VAE so the decoder gets exactly the VAE's arguments; the encoder
        # (on meta under build_on_meta) is dropped with the VAE object.
        self.decoder = AutoencoderKLLTX2Video.from_config(vae_config).decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, None, causal=None)


class NeuronLtx2VaeDecoderApplication(TorchNeuronApplicationBase):
    """The whole LTX-2 video decoder on one rank, at one static latent shape, eager.

    ``model_path`` is the VAE's directory (``<model>/vae``); ``latent_shape`` is the
    ``(batch, 128, latent_frames, latent_height, latent_width)`` the warm-up decodes.
    """

    block_attrs = ()
    #: The checkpoint's encoder half and the normalisation buffers (applied on the host).
    skipped_checkpoint_prefixes = ("encoder.", "latents_mean", "latents_std")

    def __init__(self, *, model_path, latent_shape: tuple[int, ...], parallel=None,
                 dtype=torch.bfloat16, exec_mode: str | None = "eager", device="neuron",
                 **kwargs) -> None:
        if exec_mode not in (None, "eager"):
            raise ValueError(
                f"the LTX-2 VAE decoder runs eager only, got exec_mode={exec_mode!r}")
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode="eager", device=device, **kwargs)
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the LTX-2 VAE decoder runs in bf16 or fp32, got {self.dtype}")
        with open(os.path.join(str(self.model_path), "config.json"), encoding="utf-8") as f:
            self.vae_config = json.load(f)
        self.latent_shape = tuple(int(n) for n in latent_shape)
        channels = int(self.vae_config.get("latent_channels", 128))
        if len(self.latent_shape) != 5 or self.latent_shape[1] != channels:
            raise ValueError(
                f"latent_shape {self.latent_shape} must be (batch, latent_channels={channels}, "
                "frames, height, width)"
            )
        if self.vae_config.get("timestep_conditioning") or any(
                self.vae_config.get("decoder_inject_noise") or ()):
            raise NotImplementedError(
                "the LTX-2 device decoder runs without timestep conditioning or noise injection "
                "(the LTX-2 checkpoint has neither)"
            )

    def build_module(self) -> torch.nn.Module:
        return _Ltx2VideoDecoder(self.vae_config)

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(self.latent_shape, dtype=self.dtype),)


__all__ = ["NeuronLtx2VaeDecoderApplication"]
