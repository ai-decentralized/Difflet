"""Neuron application for Wan's VAE decoder.

Wraps the backend-neutral ``WanVAEDecoderModel`` (``difflet/models/wan/vae/modeling_vae.py``),
which mirrors diffusers' decoder key for key (``post_quant_conv.*``, ``decoder.*``) and decodes
the latent frame by frame with diffusers' causal convolution cache. The checkpoint's encoder
tensors match no parameter and are skipped.

Precision (``default_vae_dtype``) follows the Trainium path, clip length by clip length: bf16 up to
``BF16_MAX_FRAMES`` frames, where it decodes on the device in its pipeline dtype (bf16), and fp32
beyond, where its compiled decoder exceeds the compiler's instruction limit and it decodes with
diffusers' VAE on the host in fp32 (as Wan's model cards load it). This application runs eagerly,
so long clips also decode on the device, at the same fp32. An explicit ``dtype`` (fp32 or bf16;
the composed application's ``vae_dtype``, the CLI's ``--vae-dtype``) overrides that choice. In
bf16 the numerically sensitive steps stay fp32 (``difflet/models/wan/vae/modeling_vae.py``): the
RMS norms normalise in fp32, as HunyuanVideo's bf16 decoder needed for its GroupNorm statistics
(F2); the written-out attention's scores and softmax are fp32; and the output head
(``fp32_output_head=True``: ``conv_out`` and the clamp) runs in fp32, so the frames come back fp32
instead of rounded to bf16. It has no tensor-parallel layers, so every rank that runs it holds the
whole 0.5 GB decoder (0.25 GB in bf16); the CLI runs it on one rank, in a launch after the
transformer stage (its fp32 activations are about 12 GiB per core at 480x832), as the Trainium
path runs its VAE stage.

Eager only: the decode loop threads a mutable per-convolution frame cache through the blocks,
which compiled regions would have to guard on every frame. ``exec_mode`` is accepted for the
common constructor and must be ``"eager"``.
"""

from __future__ import annotations

import json
import os

import torch

#: The longest clip the Trainium path decodes on the device (in bf16); longer clips decode on its
#: host fallback in fp32.
BF16_MAX_FRAMES = 9


def default_vae_dtype(num_frames: int) -> torch.dtype:
    """The VAE's precision for a clip of ``num_frames`` frames, matching the Trainium path."""
    return torch.bfloat16 if int(num_frames) <= BF16_MAX_FRAMES else torch.float32

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class NeuronWanVaeDecoderApplication(TorchNeuronApplicationBase):
    """The whole Wan VAE decoder on each rank, at one static latent shape, eager, in fp32 or
    bf16 (the callers pick ``default_vae_dtype`` unless told otherwise).

    ``model_path`` is the VAE's directory (``<model>/vae``); ``latent_shape`` is the
    ``(batch, z_dim, latent_frames, latent_height, latent_width)`` the warm-up decodes. Its
    input is ``dtype``; its output is fp32 in either dtype (the fp32 output head).
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
        if self.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"the Wan VAE decoder runs in fp32 or bf16, got {self.dtype}")
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

        # The fp32 output head is a no-op in fp32 (the fp32 decode is unchanged).
        return WanVAEDecoderModel(self.vae_config, fp32_output_head=True)

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' decoder layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(self.latent_shape, dtype=self.dtype),)


__all__ = ["NeuronWanVaeDecoderApplication"]
