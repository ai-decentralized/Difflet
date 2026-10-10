"""Shared toy modules for the neuron backend tests.

Used by the CPU unit tests, the gloo workers in ``_neuron_workers.py`` and the
device checks in ``tests/manual``. Later tasks append their toys here (C7
checkpoint writer, C9 repeated blocks, C8 application, C10 launch). Not collected
by pytest.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear, RowParallelLinear


class ToyTPMLP(nn.Module):
    """A DiT feed-forward in miniature: column-parallel up, tanh-GELU, row-parallel down.

    ``up`` keeps its output sharded and ``down`` consumes the shard directly, so the
    only collective is the row-parallel all-reduce, as in ``WanFeedForward``.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.up = ColumnParallelLinear(
            dim, hidden, bias=True, gather_output=False, dtype=dtype, device=device
        )
        self.down = RowParallelLinear(
            hidden, dim, bias=True, input_is_parallel=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="tanh"))


def toy_full_weights(
    dim: int, hidden: int, *, seed: int = 0, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Full (unsharded) ToyTPMLP weights, scaled so activations stay O(1)."""
    gen = torch.Generator().manual_seed(seed)
    weights = {
        "up.weight": torch.randn(hidden, dim, generator=gen) / dim**0.5,
        "up.bias": torch.randn(hidden, generator=gen) * 0.1,
        "down.weight": torch.randn(dim, hidden, generator=gen) / hidden**0.5,
        "down.bias": torch.randn(dim, generator=gen) * 0.1,
    }
    return {name: tensor.to(dtype) for name, tensor in weights.items()}


def reference_mlp(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """ToyTPMLP at TP1 from full weights, with no collectives.

    The op sequence mirrors the layers (bias folded into the up projection, added
    after the down projection), so ToyTPMLP at tp=1 matches it bit for bit.
    """
    h = F.gelu(F.linear(x, weights["up.weight"], weights["up.bias"]), approximate="tanh")
    return F.linear(h, weights["down.weight"]) + weights["down.bias"]
