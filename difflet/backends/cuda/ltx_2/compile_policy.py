"""Preserve native learned RMSNorm rounding inside compiled LTX-2 blocks."""

from __future__ import annotations

from types import MethodType
from typing import NamedTuple

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch import nn
from torch.nn import functional as F


@torch.library.custom_op("difflet_ltx2::native_rms_norm", mutates_args=())
def _native_rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float | None
) -> torch.Tensor:
    # Inductor's decomposition can cross BF16 rounding boundaries after its
    # FP32 reduction. Keep the native operation, rather than changing precision
    # or making the rest of the repeated block eager.
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, eps).contiguous()


@_native_rms_norm.register_fake
def _native_rms_norm_fake(hidden_states, weight, eps):
    return torch.empty_like(hidden_states, memory_format=torch.contiguous_format)


def _native_forward(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    return _native_rms_norm(hidden_states, self.weight, self.eps)


class _DiffusersNormParameters(NamedTuple):
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    eps: float


def _stock_diffusers_norm(hidden_states, weight, bias, eps):
    # Call the stock implementation itself, including its FP32 intermediates
    # and dtype/affine behavior. F.rms_norm has different rounding semantics.
    parameters = _DiffusersNormParameters(weight, bias, eps)
    return DiffusersRMSNorm.forward(parameters, hidden_states).contiguous()


@torch.library.custom_op("difflet_ltx2::diffusers_rms_norm", mutates_args=())
def _diffusers_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return _stock_diffusers_norm(hidden_states, weight, bias, eps)


@_diffusers_rms_norm.register_fake
def _diffusers_rms_norm_fake(hidden_states, weight, bias, eps):
    # Running stock arithmetic on fake tensors also preserves dtype promotion
    # when affine weights or bias have a different dtype from the input.
    return _stock_diffusers_norm(hidden_states, weight, bias, eps)


def _diffusers_forward(self: DiffusersRMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    return _diffusers_rms_norm(hidden_states, self.weight, self.bias, self.eps)


def preserve_native_rms_norm(module: nn.Module) -> int:
    """Keep stock one-dimensional RMSNorm arithmetic opaque inside compiled blocks.

    Preserve both learned torch norms and Diffusers' manual normalization.
    Custom classes and multi-dimensional normalized shapes retain their existing
    implementation. Installed once, at load.
    """
    count = 0
    for norm in module.modules():
        if type(norm) is nn.RMSNorm and len(norm.normalized_shape) == 1:
            forward = _native_forward
        elif type(norm) is DiffusersRMSNorm and len(norm.dim) == 1:
            forward = _diffusers_forward
        else:
            continue
        norm.forward = MethodType(forward, norm)
        count += 1
    return count
