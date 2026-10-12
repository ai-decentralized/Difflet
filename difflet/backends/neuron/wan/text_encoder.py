"""Neuron application for Wan's umT5 text encoder.

Wraps the backend-neutral ``WanUmT5EncoderModel`` (``difflet/models/wan/umt5/modeling_umt5.py``),
the module the Trainium path compiles (``difflet/backends/trainium/wan/text_encoder.py``), with
the same tensor-parallel layout at the pipeline's TP degree: token embedding and relative
position biases split across the embedding dimension, attention projections column-parallel
with gathered outputs, the gated FFN column- then row-parallel.

It runs in fp32 (about a quarter of the 22.7 GB fp32 encoder per rank), as the host path does:
umT5 is fragile in bf16 (its FFN output projection ``wo`` produces large activations, which is
why transformers keeps ``wo`` in fp32 for bf16 loads), its bf16 error against fp32 ranges over
several x with CPU kernel choice alone, and it runs twice per request, so fp32 costs memory and
nothing measurable in time. The embeddings are cast to the DiT's dtype afterwards.

Like the AoT path it runs one static shape, ``(batch, text_seq_len)`` token ids and mask: a
prompt is padded to ``text_seq_len`` and encoded whole, the mask keeping padding out of the real
tokens, so no prompt length compiles a new graph. The checkpoint layout is Hugging Face's UMT5
key for key (``convert_text_encoder_state_dict`` is the identity); the tied ``encoder.
embed_tokens`` is the ``shared`` module, loaded once as ``shared.weight``.
"""

from __future__ import annotations

import json
import os

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class NeuronWanTextEncoderApplication(TorchNeuronApplicationBase):
    """umT5 sharded across the TP group, at one static ``(batch_size, text_seq_len)`` shape.

    ``model_path`` is the encoder's directory (``<model>/text_encoder``). Every rank runs every
    encode (the TP collectives need them all) and every rank gets the full embeddings.
    """

    block_attrs = ("encoder.block",)

    def __init__(
        self,
        *,
        model_path,
        text_seq_len: int = 512,
        batch_size: int = 1,
        parallel=None,
        dtype=torch.float32,
        exec_mode: str | None = None,
        device="neuron",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            exec_mode=exec_mode,
            device=device,
            **kwargs,
        )
        self.text_seq_len = int(text_seq_len)
        self.batch_size = int(batch_size)
        self.encoder_config = self._read_config()
        heads, tp = int(self.encoder_config.num_heads), int(self.parallel.tp_degree)
        if heads % tp:
            raise ValueError(f"umT5 has {heads} heads, not divisible by tp_degree={tp}")

    def _read_config(self):
        from difflet.models.wan.umt5.modeling_umt5 import WanUmT5Config

        path = os.path.join(str(self.model_path), "config.json")
        with open(path, encoding="utf-8") as f:
            return WanUmT5Config.from_diffusers_dict(json.load(f))

    def build_module(self) -> torch.nn.Module:
        from difflet.models.wan.umt5.modeling_umt5 import WanUmT5EncoderModel

        return WanUmT5EncoderModel(self.encoder_config)

    def checkpoint_key(self, name: str) -> str:
        return name  # Hugging Face UMT5 layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(input_ids, attention_mask)`` at the static shape: what every encode sends."""
        shape = (self.batch_size, self.text_seq_len)
        return (torch.zeros(shape, dtype=torch.int64), torch.ones(shape, dtype=torch.int32))


__all__ = ["NeuronWanTextEncoderApplication"]
