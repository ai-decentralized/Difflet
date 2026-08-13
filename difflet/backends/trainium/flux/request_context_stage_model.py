"""Host-runnable request-context staging primitive for FLUX H1d."""

from __future__ import annotations

import torch
import torch.nn as nn


class FluxRequestContextStageModel(nn.Module):
    """Turn immutable request tensors into independent graph outputs.

    The graph intentionally has no parameters or aliases.  Its outputs are
    opaque device tensors whose lifetime is owned by the request in the host
    runtime, unlike anchors/latents whose contents are mutated through aliases.
    """

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        rotary_embedding: torch.Tensor,
        slot_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Explicit clones prevent graph-output/input identity from being
        # represented as an input passthrough by the tracing/runtime boundary.
        token_zero = slot_token.float().sum() * 0.0
        return tuple(
            value.clone() + token_zero.to(dtype=value.dtype)
            for value in (
                encoder_hidden_states,
                pooled_projections,
                guidance,
                rotary_embedding,
            )
        )


__all__ = ["FluxRequestContextStageModel"]
