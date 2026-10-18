"""Backend-neutral host-side contract for one LTX-2 transformer call.

Split out of ``application.py`` for the reason Qwen-Image's ``contract.py`` was: that module
imports NxD at module level, so anything importing it (the orchestrator in ``pipeline.py``)
dragged the Trainium toolchain in, and the neuron backend has none. These pieces are pure: the
default shape constants, a tensor dataclass and its shape/dtype check. ``application.py``
re-exports them, so existing importers are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

LTX_2_DEFAULT_TEXT_SEQ_LEN = 1024
LTX_2_DEFAULT_HEIGHT = 512
LTX_2_DEFAULT_WIDTH = 768
LTX_2_DEFAULT_NUM_FRAMES = 121


@dataclass(frozen=True)
class LTX2DiTInputBundle:
    """Host-side contract for one LTX-2 audiovisual transformer call."""

    hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    audio_encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    encoder_attention_mask: torch.Tensor
    audio_encoder_attention_mask: torch.Tensor
    video_coords: torch.Tensor
    audio_coords: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.audio_hidden_states,
            self.encoder_hidden_states,
            self.audio_encoder_hidden_states,
            self.timestep,
            self.sigma,
            self.encoder_attention_mask,
            self.audio_encoder_attention_mask,
            self.video_coords,
            self.audio_coords,
        )


def validate_ltx_2_dit_inputs(
    bundle: LTX2DiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M4c dual-stream latent/text embedding contract."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
    audio_text_seq_len = int(getattr(config, "audio_text_seq_len", text_seq_len))
    video_text_dim = int(getattr(config, "video_text_dim"))
    audio_text_dim = int(getattr(config, "audio_text_dim"))
    expected = {
        "hidden_states": ((batch_size, int(config.video_seq_len), int(config.in_channels)), dtype),
        "audio_hidden_states": (
            (batch_size, int(config.audio_seq_len), int(config.audio_in_channels)),
            dtype,
        ),
        "encoder_hidden_states": ((batch_size, text_seq_len, video_text_dim), dtype),
        "audio_encoder_hidden_states": (
            (batch_size, audio_text_seq_len, audio_text_dim),
            dtype,
        ),
        "timestep": ((batch_size,), dtype),
        "sigma": ((batch_size,), dtype),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.bool),
        "audio_encoder_attention_mask": ((batch_size, audio_text_seq_len), torch.bool),
        "video_coords": ((batch_size, 3, int(config.video_seq_len), 2), torch.float32),
        "audio_coords": ((batch_size, 1, int(config.audio_seq_len), 2), torch.float32),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "audio_hidden_states": bundle.audio_hidden_states,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "audio_encoder_hidden_states": bundle.audio_encoder_hidden_states,
        "timestep": bundle.timestep,
        "sigma": bundle.sigma,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "audio_encoder_attention_mask": bundle.audio_encoder_attention_mask,
        "video_coords": bundle.video_coords,
        "audio_coords": bundle.audio_coords,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"LTX-2 DiT input {name!r} has shape {tuple(tensor.shape)}, expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"LTX-2 DiT input {name!r} has dtype {tensor.dtype}, expected {tensor_dtype}."
            )


__all__ = [
    "LTX2DiTInputBundle",
    "LTX_2_DEFAULT_HEIGHT",
    "LTX_2_DEFAULT_NUM_FRAMES",
    "LTX_2_DEFAULT_TEXT_SEQ_LEN",
    "LTX_2_DEFAULT_WIDTH",
    "validate_ltx_2_dit_inputs",
]
