"""Neuron application for Flux's T5-XXL text encoder.

Wraps the backend-neutral ``NeuronT5EncoderModel`` (``difflet/models/flux/t5/modeling_t5.py``)
with the tensor-parallel layout the Trainium path compiles: the token embedding split across the
embedding dimension, attention projections column-parallel, the gated FFN column- then
row-parallel. The checkpoint is Hugging Face's T5 layout key for key (the Trainium conversion is
the identity); the encoder's ``embed_tokens`` is the ``shared`` module, loaded once as
``shared.weight``.

One static shape, ``(batch, text_seq_len)`` token ids (FLUX.1-dev: 512 tokens): a prompt is
padded to ``text_seq_len`` and encoded whole, **without an attention mask**, as diffusers'
FluxPipeline does (the model was trained attending to the padding; the model class asserts the
mask is ``None``), so no prompt length compiles a new graph. The
dtype is the caller's choice (fp32 by default, as for umT5, until a parity check shows bf16 is
safe for this encoder).
"""

from __future__ import annotations

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class NeuronFluxT5Application(TorchNeuronApplicationBase):
    """T5 encoder sharded across the TP group, at one static ``(batch_size, text_seq_len)`` shape.

    ``model_path`` is the encoder's directory (``<model>/text_encoder_2``). Every rank runs every
    encode (the TP collectives need them all) and every rank gets the full embeddings.
    """

    block_attrs = ("encoder.block",)

    def __init__(self, *, model_path, text_seq_len: int = 512, batch_size: int = 1, parallel=None,
                 dtype=torch.float32, exec_mode: str | None = None, device="neuron", **kwargs) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        from transformers import T5Config

        self.text_seq_len = int(text_seq_len)
        self.batch_size = int(batch_size)
        self.encoder_config = T5Config.from_pretrained(str(self.model_path))
        heads, tp = int(self.encoder_config.num_heads), int(self.parallel.tp_degree)
        if heads % tp:
            raise ValueError(f"T5 has {heads} heads, not divisible by tp_degree={tp}")

    def build_module(self) -> torch.nn.Module:
        from difflet.models.flux.t5.modeling_t5 import NeuronT5EncoderModel

        return NeuronT5EncoderModel(self.encoder_config)

    def checkpoint_key(self, name: str) -> str:
        return name  # Hugging Face T5 layout, key for key

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(input_ids,)`` at the static shape: what every encode sends (no mask, see above)."""
        return (torch.zeros(self.batch_size, self.text_seq_len, dtype=torch.int64),)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``[batch, text_seq_len, d_model]`` embeddings for padded token ids, on this rank's device."""
        out = self(input_ids.to(self.device))
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0] if isinstance(out, (tuple, list)) else out


__all__ = ["NeuronFluxT5Application"]
