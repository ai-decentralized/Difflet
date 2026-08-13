"""Host-runnable math for the experimental FLUX K=2 resident cache state."""

from __future__ import annotations

import torch
import torch.nn as nn


RESET_ACTION = 0
ANCHOR_ACTION = 1
PREDICT_ACTION = 2


class FluxResidentCacheStateModel(nn.Module):
    """Two post-gather anchors stored as aliased runtime parameters."""

    def __init__(self, *, seq_len: int, channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        shape = (1, int(seq_len), int(channels))
        self.anchor0 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        self.anchor1 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)

    def forward(
        self,
        candidate: torch.Tensor,
        coefficients: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_value = action.reshape(-1)[0]
        is_reset = action_value == RESET_ACTION
        is_anchor = action_value == ANCHOR_ACTION

        # Match the current host predictor: BF16 anchors are promoted to FP32,
        # combined in FP32, then rounded once to the original dtype.
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        zeros = torch.zeros_like(self.anchor0)

        selected = torch.where(is_reset, zeros, torch.where(is_anchor, candidate, predicted))
        new_anchor0 = torch.where(
            is_reset,
            zeros,
            torch.where(is_anchor, self.anchor1, self.anchor0),
        )
        new_anchor1 = torch.where(
            is_reset,
            zeros,
            torch.where(is_anchor, candidate, self.anchor1),
        )
        checksum = selected.float().square().mean()
        return selected, checksum, new_anchor0, new_anchor1


__all__ = [
    "ANCHOR_ACTION",
    "PREDICT_ACTION",
    "RESET_ACTION",
    "FluxResidentCacheStateModel",
]
