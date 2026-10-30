"""Neuron application for Qwen-Image's prompt encoder: Qwen2.5-VL's language model on the device.

Wraps the backend-neutral ``QwenTextEncoderModel`` (``difflet/models/qwen_image/
modeling_qwen_text_encoder.py``), the text-only decoder stack of Qwen2.5-VL with its large linears
TP-sharded over the pipeline's TP group, the way ``NeuronWanTextEncoderApplication`` puts umT5 on
the device for Wan. Opt-in (``text_encoder_device="neuron"``, CLI ``--device-text-encoder``): the
host encode on rank 0 stays the default until the device stage has measured both.

Precision: the application's dtype, bf16 by default, the dtype diffusers runs the encoder in and
the host path's default; fp32 is accepted (CPU parity tests, and on the device for a precision
A/B). The embeddings are cast to bf16 afterwards by the caller (``qwen_prompt_embeds``).

Shape: one static ``(1, enc_seq_len)`` of token ids (``ENC_SEQ`` = 256, the Trainium text stage's
budget): every prompt is right-padded to it and encoded whole, so no prompt length compiles a new
graph. No attention mask is passed: attention is causal, so the valid rows never see the padding
after them (see the modeling's docstring). In compile mode the 28 identical decoder layers compile
into one shared graph (``block_attrs = ("layers",)``); the embedding and the final norm stay eager.

HBM: about 7.07 B parameters (13.2 GiB in bf16), so about 3.3 GiB per rank at tp4 beside the
DiT's 10.1 GiB, and 6.6 GiB at tp2 beside 19.75 GiB, which is over a 24 GiB core: the composed
application refuses it before anything loads and the CLI driver falls back to the host encoder
(``difflet/backends/neuron/qwen_image/hbm.py``).

Checkpoint: the transformers Qwen2.5-VL layout. The language model is ``model.embed_tokens``,
``model.layers.*`` and ``model.norm`` (``model.language_model.*`` in the layout newer
transformers versions save; the prefix is read from the checkpoint's own weight map); the vision
tower (``visual.*``, or ``model.visual.*``) and ``lm_head`` are skipped on purpose.
"""

from __future__ import annotations

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.common.orchestrators.qwen_image import ENC_SEQ

#: Checkpoint prefixes of the language model, oldest layout first.
_LM_PREFIXES = ("model.language_model.", "model.")


class NeuronQwenTextEncoderApplication(TorchNeuronApplicationBase):
    """Qwen2.5-VL's language model sharded across the TP group at one static ``(1,
    enc_seq_len)`` shape; ``model_path`` is ``<model>/text_encoder``. Every rank runs every
    encode (the TP collectives need them all) and every rank gets the full last hidden state."""

    block_attrs = ("layers",)
    skipped_checkpoint_prefixes = ("visual.", "model.visual.", "lm_head.")

    def __init__(
        self,
        *,
        model_path,
        enc_seq_len: int = ENC_SEQ,
        parallel=None,
        dtype=torch.bfloat16,
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
        from difflet.models.qwen_image.modeling_qwen_text_encoder import QwenTextEncoderConfig

        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"the Qwen2.5-VL text encoder runs in bfloat16 or float32, got {self.dtype}"
            )
        self.enc_seq_len = int(enc_seq_len)
        self.encoder_config = QwenTextEncoderConfig.from_pretrained(self.model_path)
        self.encoder_config.validate(int(self.parallel.tp_degree))
        self._prefix: str | None = None

    def build_module(self) -> torch.nn.Module:
        from difflet.models.qwen_image.modeling_qwen_text_encoder import QwenTextEncoderModel

        return QwenTextEncoderModel(self.encoder_config, max_positions=self.enc_seq_len)

    def checkpoint_key(self, name: str) -> str:
        if self._prefix is None:
            from difflet.backends.neuron.core.checkpoint import build_weight_map

            keys = build_weight_map(self.model_path)
            self._prefix = next(
                (p for p in _LM_PREFIXES if f"{p}embed_tokens.weight" in keys), "model."
            )
        return self._prefix + name

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        """``(input_ids,)`` at the static shape: what every encode sends."""
        return (torch.zeros(1, self.enc_seq_len, dtype=torch.int64),)


__all__ = ["NeuronQwenTextEncoderApplication"]
