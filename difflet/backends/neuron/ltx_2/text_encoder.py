"""Neuron application for LTX-2's Gemma-3 12B text encoder (its language model).

Wraps the backend-neutral ``Gemma3TextEncoder`` (``difflet/models/ltx_2/
modeling_gemma3_text_encoder.py``) on the base application's lifecycle: built on meta after the
mesh, this rank's TP shards loaded straight from transformers' checkpoint in
``<model>/text_encoder`` (``language_model.model.*``; the vision tower and the multimodal
projector are skipped), the HBM check, the 48 identical decoder layers compiled in place into one
graph, and a warm-up at the static ``(1, seq_len)`` token shape. It shares each rank's core with
the DiT shard: at tp4 about 5.5 GiB of bf16 encoder beside 9.3 GiB of DiT.

``forward(input_ids)`` returns the 49 hidden states stacked, ``[1, seq_len, 3840, 49]``, on every
rank (the row-parallel outputs are all-reduced), for right-padded tokens; see the modeling
module for the padding semantics.
"""

from __future__ import annotations

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase


class NeuronLtx2TextEncoderApplication(TorchNeuronApplicationBase):
    """Gemma-3's language model across the TP group, at ``seq_len`` tokens, in ``dtype``."""

    block_attrs = ("layers",)

    def __init__(self, *, model_path, seq_len: int = 1024, parallel=None, dtype=torch.bfloat16,
                 exec_mode: str | None = None, device="neuron", num_layers: int | None = None,
                 **kwargs) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        from difflet.models.ltx_2.modeling_gemma3_text_encoder import (
            LANGUAGE_MODEL_PREFIX,
            SKIPPED_PREFIXES,
            Gemma3TextEncoderConfig,
        )

        self.config = Gemma3TextEncoderConfig.from_pretrained(self.model_path)
        self.seq_len = int(seq_len)
        self.num_layers = None if num_layers is None else int(num_layers)
        self._prefix = LANGUAGE_MODEL_PREFIX
        skipped = list(SKIPPED_PREFIXES)
        total = int(self.config.num_hidden_layers)
        if self.num_layers is not None and self.num_layers < total:
            skipped += [f"{LANGUAGE_MODEL_PREFIX}layers.{i}." for i in range(self.num_layers, total)]
        self.skipped_checkpoint_prefixes = tuple(skipped)
        if self.device.type == "neuron" and self.dtype is not torch.bfloat16:
            raise ValueError(
                f"the Gemma-3 text encoder runs in torch.bfloat16 on the neuron device, got "
                f"{self.dtype} (decision D42, as for the DiT)"
            )

    def build_module(self) -> torch.nn.Module:
        # Imported here: the modeling binds difflet.ops to the selected backend at import.
        from difflet.models.ltx_2.modeling_gemma3_text_encoder import Gemma3TextEncoder

        return Gemma3TextEncoder(self.config, max_positions=self.seq_len,
                                 num_layers=self.num_layers)

    def checkpoint_key(self, name: str) -> str:
        return self._prefix + name

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.seq_len, dtype=torch.int64),)


__all__ = ["NeuronLtx2TextEncoderApplication"]
