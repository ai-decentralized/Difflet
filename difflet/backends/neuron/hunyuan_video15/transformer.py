"""Neuron application for HunyuanVideo 1.5's DiT: diffusers' ``HunyuanVideo15Transformer3DModel``.

The module is diffusers' own (no Difflet port exists; the Trainium path wraps the same class), built
on ``meta`` and loaded from diffusers' checkpoint key for key, fp32 -> the application dtype; in
compile mode its 54 identical ``transformer_blocks`` are compiled in place into one graph.

diffusers' forward runs as is: its text-token reorder (valid tokens of the image, ByT5 and
Qwen2.5-VL streams first, padding zeroed after) is data-dependent but fixed-length (729 + 256 + 1000
= 1,985 tokens), and it runs in the un-compiled top-level forward, so the compiled blocks see one
shape. (The Trainium path forks it because it compiles the whole model as one graph.)

Parallelism: TP1 (the 8.3B DiT is 15.5 GiB in bf16, one core); context parallelism comes next.
Numerics depend on two backend workarounds installed by ``prepare_runtime``: the bool bitwise guard
(the token refiner's mask) and fp32 LayerNorm for bf16 inputs.
"""

from __future__ import annotations

import json
import os

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase

#: Spatial and temporal compression of HunyuanVideo 1.5's VAE (latent = (F-1)/4+1, H/16, W/16).
VAE_SPATIAL, VAE_TEMPORAL = 16, 4


class NeuronHunyuanVideo15TransformerApplication(TorchNeuronApplicationBase):
    """HunyuanVideo 1.5's DiT at one static shape (TP1)."""

    block_attrs = ("transformer_blocks",)

    def __init__(self, *, model_path, height: int = 480, width: int = 848, num_frames: int = 121,
                 text_seq_len: int = 1000, text_2_seq_len: int = 256, image_tokens: int = 729,
                 batch_size: int = 1, parallel=None, dtype=torch.bfloat16,
                 exec_mode: str | None = None, device="neuron", **kwargs) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        if int(self.parallel.tp_degree) != 1 or int(getattr(self.parallel, "cp_degree", 1) or 1) != 1:
            raise NotImplementedError("HunyuanVideo 1.5 on the neuron backend runs at TP1 (context "
                                      "parallelism is not implemented yet)")
        with open(os.path.join(str(self.model_path), "config.json"), encoding="utf-8") as f:
            self.model_config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        if int(num_frames) % VAE_TEMPORAL != 1:
            raise ValueError(f"num_frames={num_frames} must be 4k+1")
        for name, value in (("height", height), ("width", width)):
            if int(value) % VAE_SPATIAL:
                raise ValueError(f"{name}={value} must be a multiple of {VAE_SPATIAL}")
        self.latent_shape = (int(batch_size), int(self.model_config["in_channels"]),
                             (int(num_frames) - 1) // VAE_TEMPORAL + 1,
                             int(height) // VAE_SPATIAL, int(width) // VAE_SPATIAL)
        self.text_seq_len, self.text_2_seq_len = int(text_seq_len), int(text_2_seq_len)
        self.image_tokens, self.batch_size = int(image_tokens), int(batch_size)

    def build_module(self) -> torch.nn.Module:
        return _tensor_signature_model()(**self.model_config)

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' layout, key for key

    def _text_to_video_defaults(self, encoder_hidden_states_2, encoder_attention_mask_2, image_embeds):
        """What the pipeline passes for text-to-video without quoted text: an empty glyph stream and
        zero image embeddings."""
        b, dt = self.batch_size, self.dtype
        if encoder_hidden_states_2 is None:
            encoder_hidden_states_2 = torch.zeros(b, self.text_2_seq_len,
                                                  int(self.model_config["text_embed_2_dim"]), dtype=dt)
            encoder_attention_mask_2 = torch.zeros(b, self.text_2_seq_len, dtype=torch.int64)
        if image_embeds is None:
            image_embeds = torch.zeros(b, self.image_tokens, int(self.model_config["image_embed_dim"]), dtype=dt)
        return encoder_hidden_states_2, encoder_attention_mask_2, image_embeds

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        b, dt = self.batch_size, self.dtype
        mask = torch.zeros(b, self.text_seq_len, dtype=torch.int64)
        mask[:, :64] = 1
        e2, m2, img = self._text_to_video_defaults(None, None, None)
        return self._positional(torch.zeros(self.latent_shape, dtype=dt), torch.ones(b, dtype=dt),
                                torch.zeros(b, self.text_seq_len, int(self.model_config["text_embed_dim"]), dtype=dt),
                                mask, e2, m2, img)

    def _positional(self, *tensors) -> tuple[torch.Tensor, ...]:
        """The module's seven tensors on this rank's device: floats in the application dtype, the
        masks (integer) as they are."""
        dev, dt = self.device, self.dtype
        return tuple(t.to(dev, dt) if t.is_floating_point() else t.to(dev) for t in tensors)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                timestep_r=None, encoder_hidden_states_2=None, encoder_attention_mask_2=None,
                image_embeds=None, attention_kwargs=None, return_dict=False):
        """The pipeline's call; returns ``(noise_pred,)`` on the device."""
        if timestep_r is not None:
            raise NotImplementedError("timestep_r (meanflow) is not supported")
        e2, m2, img = self._text_to_video_defaults(encoder_hidden_states_2, encoder_attention_mask_2,
                                                   image_embeds)
        return (super().forward(*self._positional(hidden_states, timestep, encoder_hidden_states,
                                                  encoder_attention_mask, e2, m2, img)),)


def _tensor_signature_model():
    """diffusers' model with a tensor-only forward (the application base warms up and compiles
    with tensor inputs): ``(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
    encoder_hidden_states_2, encoder_attention_mask_2, image_embeds) -> noise_pred``. A subclass,
    so parameter names and ``transformer_blocks`` are diffusers' own."""
    from diffusers import HunyuanVideo15Transformer3DModel

    class HunyuanVideo15TensorSignature(HunyuanVideo15Transformer3DModel):
        def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                    encoder_hidden_states_2, encoder_attention_mask_2, image_embeds):
            return super().forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                   encoder_hidden_states_2=encoder_hidden_states_2,
                                   encoder_attention_mask_2=encoder_attention_mask_2,
                                   image_embeds=image_embeds, return_dict=False)[0]

    return HunyuanVideo15TensorSignature


__all__ = ["NeuronHunyuanVideo15TransformerApplication", "VAE_SPATIAL", "VAE_TEMPORAL"]
