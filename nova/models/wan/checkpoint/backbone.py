"""HF diffusers WanTransformer3DModel → Nova WanTransformer3DModel state dict.

The Nova modeling layer mostly mirrors the upstream attribute names so most
keys carry over unchanged. The only structural divergence is the FFN
sub-module: diffusers wraps the up- and down-projections in a
``ModuleList`` with non-trivial indices, while Nova uses explicit
``net_in`` / ``net_out`` attributes.

Diffusers FFN layout (`activation_fn="gelu-approximate"`):
    blocks.{i}.ffn.net.0.proj.weight      # GELU's projection (= up-proj)
    blocks.{i}.ffn.net.0.proj.bias
    blocks.{i}.ffn.net.2.weight           # down-proj Linear
    blocks.{i}.ffn.net.2.bias

Nova FFN layout (``WanFeedForward``):
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
    Nova's ``WanTransformer3DModel`` attribute layout.

    Args:
        state_dict: HF-style state dict (key, tensor pairs). Loaded from
            safetensors / pytorch_model.bin via
            ``nova.core.modules.checkpoint.load_state_dict``.
        config: optional ``WanTransformerConfig`` or ``WanBackboneInferenceConfig``
            instance. Currently unused; reserved for future structural
            transforms (e.g. weight splitting).

    Returns:
        Renamed state dict. Tensor objects are passed through unchanged
        (no clone). Caller may apply ``.clone().detach().contiguous()`` later
        before saving.
    """
    del config  # unused for now; kept for forward-compat
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in BACKBONE_KEY_RENAMES:
            new_key = re.sub(pattern, replacement, new_key)
        out[new_key] = value
    return out


__all__ = ["BACKBONE_KEY_RENAMES", "convert_backbone_state_dict"]
