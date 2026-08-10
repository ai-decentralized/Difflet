"""MiniMax-H3 Trainium application boundary.

The production CLI is deliberately staged because the BF16 Qwen3-VL encoder,
33B Omni Transformer, visual VAE, and audio VAE cannot all remain resident on a
four-core instance.  This application exposes the static model contract while
the individual Neuron components are integrated behind those four stages.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from difflet.backends.trainium.core.multi_component_application import MultiComponentApplication
from difflet.common.orchestrators import minimax_h3 as h3_common
from difflet.models.minimax_h3.contracts import build_padded_t2va_layout


def create_minimax_h3_transformer_config(
    *,
    model_path: str,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = h3_common.TEXT_SEQ_LEN,
):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.minimax_h3.transformer import (
        MiniMaxH3TransformerInferenceConfig,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_path = os.path.join(model_path, "transformer")
    return MiniMaxH3TransformerInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=tp_degree,
            world_size=tp_degree,
            torch_dtype=dtype,
        ),
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"MiniMax-H3 currently supports BF16 only, got {dtype!r}")


class NeuronMiniMaxH3Application(MultiComponentApplication):
    """Static H3 T2VA contract; stages own component lifecycle independently."""

    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or h3_common.DEFAULT_HEIGHT),
            "width": int(shape.get("width") or h3_common.DEFAULT_WIDTH),
            "num_frames": int(shape.get("num_frames") or h3_common.DEFAULT_NUM_FRAMES),
        }
        self.text_seq_len = int(kwargs.get("text_seq_len", h3_common.TEXT_SEQ_LEN))
        self.layout = build_padded_t2va_layout(
            num_text_tokens=self.text_seq_len,
            max_text_tokens=self.text_seq_len,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            sequence_alignment=128,
        )

    def components(self):
        # H3 is not a resident MultiComponentApplication: each large component
        # is compiled/loaded in its own subprocess stage to stay within HBM.
        return []

    def no_components_message(self, action: str) -> str:
        return (
            f"MiniMax-H3 {action} uses the four-stage CLI; invoke `difflet {action} "
            "--model-id MiniMaxAI/MiniMax-H3` instead of a resident application."
        )

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        layout = self.layout
        video_rows = int(layout.video_indices.numel())
        audio_rows = int(layout.audio_indices.numel())
        return {
            "hidden_states": {"shape": (1, video_rows, 96), "dtype": self.dtype},
            "audio_hidden_states": {"shape": (1, audio_rows, 32), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (1, self.text_seq_len, 5120),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (2,), "dtype": torch.float32},
            "timestep_indices": {
                "shape": (layout.sequence_length,),
                "dtype": torch.int64,
            },
            "token_tags": {"shape": (layout.sequence_length,), "dtype": torch.int64},
            "position_ids": {
                "shape": (layout.sequence_length, 3),
                "dtype": torch.float64,
            },
        }
