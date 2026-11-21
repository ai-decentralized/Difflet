"""The part of the Qwen3-VL text encoder MiniMax-H3 reads: the vision tower and the first 50
language layers, without the final norm.

H3 conditions on the language model's ``hidden_states[50]`` (before the final norm). A
text-only prompt runs ``encoder.language_model``; a first-frame prompt also presents the
keyframe as image tokens, through the vision tower and its DeepStack features.
"""

from __future__ import annotations

import copy

import torch

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends._torch_megatron.weights import materialize_meta_

_SELECTED_LAYER = 50


def load_text_encoder(model_dir, *, dtype: torch.dtype, device) -> torch.nn.Module:
    """A ``Qwen3VLModel`` whose ``last_hidden_state`` is the original model's
    ``hidden_states[50]``; only its own tensors are read from the checkpoint."""
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = copy.deepcopy(Qwen3VLConfig.from_pretrained(model_dir, local_files_only=True))
    config.text_config.num_hidden_layers = _SELECTED_LAYER
    config.text_config.use_cache = False
    # Keep constructor-computed rotary frequencies real and FP32: building buffers on meta,
    # or casting the whole module later, would change them.
    with init_empty_weights(include_buffers=False):
        encoder = Qwen3VLModel(config)
        encoder.language_model.norm = torch.nn.Identity()
    # Allocate on the device, then stream one checkpoint tensor at a time.
    materialize_meta_(encoder, device=device, dtype=dtype)
    load_checkpoint_into(
        encoder,
        model_dir,
        tp_size=1,
        tp_rank=0,
        dtype=dtype,
        strict=True,
        rename=lambda name: "model." + name,
    )
    # Device-only movement keeps the FP32 nonpersistent rotary buffers.
    return encoder.to(device=device).eval().requires_grad_(False)


__all__ = ["load_text_encoder"]
