"""Neuron application for MiniMax-H3's conditioner: Qwen3-VL's language model to hidden_states[50].

Wraps the backend-neutral ``Qwen3VLTextEncoderModel``
(``difflet/models/minimax_h3/modeling_qwen3vl_text_encoder.py``): the embedding and the first 50
decoder layers, TP-sharded over the DiT's TP group (about 12.6 GB per rank in bf16 at tp4). One
static ``(1, 1024)`` token shape (the Trainium text stage's bucket); in compile mode the 50
identical layers compile into one shared graph.

Checkpoint: transformers' Qwen3-VL layout, ``model.language_model.*``; the vision tower,
``lm_head``, the final norm and layers 50-63 are skipped on purpose.

``encode(prompt)`` returns the Trainium text stage's record (``text.pt``): the hidden states with
the padding rows zeroed, the bool mask, the live token count and the layer, identical on every
rank (every rank runs the encode, the TP collectives need them all).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.common.orchestrators import minimax_h3 as h3_common

_PREFIX = "model.language_model."


class NeuronMiniMaxH3TextEncoderApplication(TorchNeuronApplicationBase):
    """``model_path`` is ``<model>/text_encoder``; the tokenizer is read from
    ``<model>/tokenizer``."""

    block_attrs = ("layers",)

    def __init__(self, *, model_path, seq_len: int = h3_common.TEXT_SEQ_LEN,
                 layer: int = h3_common.TEXT_ENCODER_LAYER, parallel=None,
                 dtype=torch.bfloat16, exec_mode: str | None = None, device="neuron",
                 **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        from difflet.models.minimax_h3.modeling_qwen3vl_text_encoder import (
            Qwen3VLTextEncoderConfig,
        )

        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the Qwen3-VL text encoder runs in bf16 or fp32, got {self.dtype}")
        self.seq_len = int(seq_len)
        self.layer = int(layer)
        self.encoder_config = Qwen3VLTextEncoderConfig.from_pretrained(
            self.model_path, num_layers=self.layer)
        self.encoder_config.validate(int(self.parallel.tp_degree))
        total = int(self.encoder_config.num_hidden_layers)
        self.skipped_checkpoint_prefixes = (
            "model.visual.", "lm_head.", f"{_PREFIX}norm.",
            *(f"{_PREFIX}layers.{i}." for i in range(self.layer, total)),
        )
        self.tokenizer = None

    def build_module(self) -> torch.nn.Module:
        from difflet.models.minimax_h3.modeling_qwen3vl_text_encoder import (
            Qwen3VLTextEncoderModel,
        )

        return Qwen3VLTextEncoderModel(self.encoder_config, max_positions=self.seq_len)

    def checkpoint_key(self, name: str) -> str:
        return _PREFIX + name

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.seq_len, dtype=torch.int64),)

    def load(self, *args: Any, **kwargs: Any) -> None:
        super().load(*args, **kwargs)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(Path(str(self.model_path)).parent / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, prompt: str) -> dict[str, Any]:
        """The Trainium text stage's tokenization (raw prompt, no special tokens, right-padded to
        ``seq_len``) and record."""
        tokens = self.tokenizer(prompt, add_special_tokens=False, max_length=self.seq_len,
                                padding="max_length", truncation=True, return_tensors="pt",
                                return_attention_mask=True)
        mask = tokens.attention_mask
        count = int(mask.sum())
        if count < 1:
            raise ValueError("MiniMax-H3 prompt must produce at least one token")
        hidden = self(tokens.input_ids.to(torch.int64).to(self.device))
        hidden = hidden.cpu().to(torch.bfloat16)
        hidden[:, count:] = 0  # padding rows are never part of the packed document
        return {"encoder_hidden_states": hidden, "attention_mask": mask.to(torch.bool),
                "num_text_tokens": count, "text_encoder_layer": self.layer}

    def release(self) -> None:
        """Drop the weights from the device (the generate stage frees this core's share before
        the video decoder loads)."""
        self.module = None
        self.is_loaded = False
        import gc

        gc.collect()
        if self.device.type == "neuron":
            torch.neuron.empty_cache()


__all__ = ["NeuronMiniMaxH3TextEncoderApplication"]
