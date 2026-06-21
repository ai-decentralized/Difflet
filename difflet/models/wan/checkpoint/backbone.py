"""HF diffusers WanTransformer3DModel → Difflet WanTransformer3DModel state dict.

The Difflet modeling layer mostly mirrors the upstream attribute names so most
keys carry over unchanged. The only structural divergence is the FFN
sub-module: diffusers wraps the up- and down-projections in a
``ModuleList`` with non-trivial indices, while Difflet uses explicit
``net_in`` / ``net_out`` attributes.

Diffusers FFN layout (`activation_fn="gelu-approximate"`):
    blocks.{i}.ffn.net.0.proj.weight      # GELU's projection (= up-proj)
    blocks.{i}.ffn.net.0.proj.bias
    blocks.{i}.ffn.net.2.weight           # down-proj Linear
    blocks.{i}.ffn.net.2.bias

Difflet FFN layout (``WanFeedForward``):
    blocks.{i}.ffn.net_in.weight
    blocks.{i}.ffn.net_in.bias
    blocks.{i}.ffn.net_out.weight
    blocks.{i}.ffn.net_out.bias

Everything else (attention q/k/v/out, norm_q/norm_k, scale_shift_table,
patch_embedding, condition_embedder, norm_out, proj_out, root
scale_shift_table) carries over verbatim.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import torch

# Substring regex rewrites applied in order. Each (pattern, replacement) pair
# rewrites every match in a key.
BACKBONE_KEY_RENAMES: list[tuple[str, str]] = [
    (r"\.ffn\.net\.0\.proj\.", ".ffn.net_in."),
    (r"\.ffn\.net\.2\.", ".ffn.net_out."),
]


def convert_backbone_state_dict(
    state_dict: Dict[str, Any],
    config: Any | None = None,
) -> Dict[str, Any]:
    """Rename keys in a diffusers WanTransformer3DModel state dict to match
    Difflet's ``WanTransformer3DModel`` attribute layout.

    Args:
        state_dict: HF-style state dict (key, tensor pairs). Loaded from
            safetensors / pytorch_model.bin via
            ``difflet.backends.trainium.core.modules.checkpoint.load_state_dict``.
        config: optional ``WanTransformerConfig`` or ``WanBackboneInferenceConfig``
            instance. Currently unused; reserved for future structural
            transforms (e.g. weight splitting).

    Returns:
        Renamed state dict. Tensor objects are passed through unchanged
        (no clone). Caller may apply ``.clone().detach().contiguous()`` later
        before saving.
    """
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in BACKBONE_KEY_RENAMES:
            new_key = re.sub(pattern, replacement, new_key)
        out[new_key] = value

    if config is not None and getattr(config, "context_parallel_enabled", False):
        world_size = config.neuron_config.world_size
        out["global_rank.rank"] = torch.arange(0, world_size, dtype=torch.int32)

    return out


__all__ = ["BACKBONE_KEY_RENAMES", "convert_backbone_state_dict"]
