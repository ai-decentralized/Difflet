"""HunyuanVideo TeaCache probe model.

A minimal nn.Module whose forward executes ONLY the block-0 modulated-input
path of the full HunyuanVideo transformer, plus a device-side L2-norm diff
against a caller-supplied ``prev_mod_input``. Compiles to a small NEFF that
exposes the modulated-input scalar/tensor without forcing a full 40-block
DiT forward.

The CPU reference is ``HunyuanVideoTransformer3DModel.teacache_mod_input``
in ``difflet/models/hunyuan_video/modeling_hunyuan_video.py``.

Designed per ``cclogs/m9-teacache/72-teacache-probe-neff-design.md``
(decoupled architecture: the production DiT NEFF stays untouched; this probe
NEFF is additive). The wrapper holds a full ``HunyuanVideoTransformer3DModel``
for weight-shape compatibility with the production HF state dict; XLA dead-code
elimination prunes the unused layers when the NEFF is compiled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
)


class HunyuanVideoTeacacheProbeFusedModel(nn.Module):
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
        super().__init__()
        self.model = HunyuanVideoTransformer3DModel(config)
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.model.teacache_mod_input(
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


class HunyuanVideoTeacacheProbeModel(nn.Module):
    """Compile target for the TeaCache probe NEFF.

    The forward returns ``(delta_scalar, mod_input_tensor)`` where ``delta_scalar``
    is ``||mod_input − prev_mod_input||_2`` on device. Callers in calibration
    mode pass a zero ``prev_mod_input`` and ignore ``delta_scalar``. Callers
    in T1 production mode pass the previous step's ``mod_input`` (a device
    tensor handle retained across denoise steps) and use the scalar to drive
    the host-side skip decision.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.model = HunyuanVideoTransformer3DModel(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        prev_mod_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.model.teacache_mod_input(
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
