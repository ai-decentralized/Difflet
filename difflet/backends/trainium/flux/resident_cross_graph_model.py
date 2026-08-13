"""Host-runnable models for the FLUX H1b cross-graph resident-state spike.

Every entry point declares the same two aliased parameters.  The compiled
runtime can therefore bind the update, predict, consume, and reset NEFFs to
one pair of device allocations while giving each entry point a different
host-visible input signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidentAnchors(nn.Module):
    def __init__(self, *, seq_len: int, channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        shape = (1, int(seq_len), int(channels))
        self.anchor0 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        self.anchor1 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)

    def _outputs(
        self,
        selected: torch.Tensor,
        new_anchor0: torch.Tensor,
        new_anchor1: torch.Tensor,
        checksum: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if checksum is None:
            checksum = selected.float().square().mean()
        return selected, checksum, new_anchor0, new_anchor1


class ResidentAnchorUpdateModel(_ResidentAnchors):
    """Shift the K=2 ring and install a newly computed anchor."""

    def forward(
        self, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Both aliased parameters must participate in lowering even though an
        # update overwrites anchor1.  Otherwise the legacy XLA alias map refers
        # to a parameter absent from the lowering context.
        selected = candidate
        new_anchor0 = self.anchor1 + self.anchor0 * 0.0
        new_anchor1 = candidate + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(selected, new_anchor0, new_anchor1)


class ResidentPredictModel(_ResidentAnchors):
    """Read shared anchors using a scalar-only host signature."""

    def forward(
        self, coefficients: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        return self._outputs(predicted, self.anchor0, self.anchor1)


class ResidentConsumeModel(_ResidentAnchors):
    """Stand in for scheduler/latent update while preserving ranked device I/O."""

    def forward(
        self, predicted: torch.Tensor, step_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The scalar is deliberately used so it remains in the compiled
        # signature.  A value of one makes this consumer an exact passthrough.
        selected = (predicted.float() * step_scale.float().reshape(-1)[0]).to(
            dtype=predicted.dtype
        )
        return self._outputs(selected, self.anchor0, self.anchor1)


class ResidentResetModel(_ResidentAnchors):
    """Clear request-local state using only a scalar request token."""

    def forward(
        self, request_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Express reset through the old allocations so both state parameters
        # are visible to the alias-aware lowering pass.
        zeros = self.anchor0 * 0.0 + self.anchor1 * 0.0
        new_anchor0 = self.anchor0 * 0.0
        new_anchor1 = self.anchor1 * 0.0
        # Keep request_token in the graph without changing the reset result.
        checksum = request_token.float().sum() * 0.0
        return self._outputs(zeros, new_anchor0, new_anchor1, checksum)


__all__ = [
    "ResidentAnchorUpdateModel",
    "ResidentConsumeModel",
    "ResidentPredictModel",
    "ResidentResetModel",
]
