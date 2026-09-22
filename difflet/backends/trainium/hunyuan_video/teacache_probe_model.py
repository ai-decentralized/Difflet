"""HunyuanVideo probes with only the block-0 modulation prefix weights.

Canonical parameter names are preserved, but unused attention, FFN, token
refiner and later-block parameters are never constructed or loaded. Both
probe variants retain the backbone's signal computation and L2 reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.normalization import AdaLayerNormZero

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoPatchEmbed,
    HunyuanVideoTransformer3DModel,
)

#: Tensors the fused probe adds on top of the backbone. Aliased in place by
#: its ModelInstance, so NEFF state (zero-initialised at load), not weights.
PROBE_STATE_TENSORS: frozenset[str] = frozenset({"prev_mod"})


class HunyuanVideoTeacachePrefix(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.inner_dim
        self.x_embedder = HunyuanVideoPatchEmbed(
            (config.patch_size_t, config.patch_size, config.patch_size),
            config.in_channels,
            inner_dim,
        )
        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim,
            config.pooled_projection_dim,
            config.guidance_embeds,
            getattr(config, "image_condition_type", None),
        )
        block = nn.Module()
        block.norm1 = AdaLayerNormZero(inner_dim, norm_type="layer_norm")
        self.transformer_blocks = nn.ModuleList([block])

    # Reuse the reference computation without constructing the full backbone.
    teacache_mod_input = HunyuanVideoTransformer3DModel.teacache_mod_input


class HunyuanVideoTeacacheProbeFusedModel(HunyuanVideoTeacachePrefix):
    """fused-A probe (cclog 80): prev_mod lives ON the model as an nn.Parameter
    (NOT an input, NOT a buffer — the alias machinery in hlo_conversion.py only
    scans named_parameters()). forward returns ``(delta, mod_input)`` where
    ``mod_input`` (output index 1) is aliased back to ``prev_mod`` in-place; only
    the scalar ``delta`` (output 0) is returned to host.

    This removes the ~25.8 ms/step mod_input output-marshaling that the v1 probe
    paid (cclog 79 Test 1): the 63 MB mod_input never becomes a host-visible
    output — it is written in place to prev_mod's HBM via input_output_aliases.

    prev_mod must be sized (batch, seq_len, inner_dim). Recipe verified by the
    counter NEFF in scripts/probe_persistent_buffer_test.py.
    """

    def __init__(self, config, *, seq_len: int, inner_dim: int, batch_size: int = 1) -> None:
        super().__init__(config)
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def forward(  # type: ignore[override] — the probe traces its own signature
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.teacache_mod_input(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
        diff = (mod_input - self.prev_mod).reshape(-1)
        delta = torch.linalg.vector_norm(diff)
        # output 0 = delta (4 B → host); output 1 = mod_input (aliased to
        # prev_mod, stripped from host-visible outputs).
        return delta, mod_input


class HunyuanVideoTeacacheProbeModel(HunyuanVideoTeacachePrefix):
    """Compile target for the (v1, non-fused) TeaCache probe NEFF.

    The forward returns ``(delta_scalar, mod_input_tensor)`` where ``delta_scalar``
    is ``||mod_input − prev_mod_input||_2`` on device. Callers in calibration
    mode pass a zero ``prev_mod_input`` and ignore ``delta_scalar``. Callers
    in T1 production mode pass the previous step's ``mod_input`` (a device
    tensor handle retained across denoise steps) and use the scalar to drive
    the host-side skip decision. Adds no tensors of its own.
    """

    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(  # type: ignore[override] — the probe traces its own signature
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        prev_mod_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.teacache_mod_input(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
        diff = (mod_input - prev_mod_input).reshape(-1)
        delta = torch.linalg.vector_norm(diff)
        return delta, mod_input
