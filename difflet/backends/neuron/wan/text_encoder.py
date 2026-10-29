"""Neuron application for Wan's umT5 text encoder.

Wraps the backend-neutral ``WanUmT5EncoderModel`` (``difflet/models/wan/umt5/modeling_umt5.py``),
the module the Trainium path compiles (``difflet/backends/trainium/wan/text_encoder.py``), with
the same tensor-parallel layout at the pipeline's TP degree: token embedding and relative
position biases split across the embedding dimension, attention projections column-parallel
with gathered outputs, the gated FFN column- then row-parallel.

Its precision is the caller's; the application default is bf16 throughout, the precision the
Trainium path runs it at and diffusers' Wan pipeline loads it in (transformers'
``_keep_in_fp32_modules`` keeps the FFN output projection ``wo`` in fp32 for fp16 loads only, so a
bf16 load casts ``wo`` to bf16 too, as the AoT path's bf16 umT5 does). fp32 remains an option (the
composed application's ``text_encoder_dtype``, the CLI's ``--text-encoder-dtype fp32``): umT5's
bf16 error against fp32 varies with kernel choice (its ``wo`` produces large activations), so the
parity gate's floor spans several bf16 implementations. The weight store keys and writes entries
in the load dtype (``core/weight_store.py``): the fp32 encoder gains nothing from an fp32 entry and
gets none, the bf16 one gets an 11.36 GB entry (2.84 GB per rank at tp4) instead of re-reading the
22.7 GB checkpoint on every cold start. The embeddings are cast to the DiT's dtype afterwards.

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
from difflet.backends.neuron.wan.compile_options import apply_ccop_overlap


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
        # Before anything compiles: Wan's ccop overlap flag, held while this application lives
        # (wan/compile_options.py).
        apply_ccop_overlap(self.device, self)
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
