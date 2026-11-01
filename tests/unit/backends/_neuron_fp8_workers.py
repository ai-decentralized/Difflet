"""Gloo workers and toys for the neuron FP8 linear tests (not collected by pytest).

``ToyFP8MLP`` is ``ToyTPMLP`` with both linears quantized: a column-parallel fp8 up projection
that keeps its output sharded and a row-parallel fp8 down projection whose all-reduce is the
only collective. ``fp8_toy_full_weights`` quantizes the toy's full fp32 weights with the
``difflet.quant`` law, so every rank slices the same fp8 tensors the checkpoint writer would
produce, and ``fp8_toy_reference`` runs the same layer math unsharded on one process.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.fp8_linear import (
    FP8ColumnParallelLinear,
    FP8RowParallelLinear,
)
from difflet.quant import fp8 as law
from tests.unit.backends._neuron_toy import toy_full_weights


class ToyFP8MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, *, compute: str = "dequant", dtype=None,
                 device=None):
        super().__init__()
        self.up = FP8ColumnParallelLinear(dim, hidden, bias=True, gather_output=False,
                                          dtype=dtype, device=device, compute=compute)
        self.down = FP8RowParallelLinear(hidden, dim, bias=True, input_is_parallel=True,
                                         dtype=dtype, device=device, compute=compute)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


def fp8_toy_full_weights(dim: int, hidden: int, *, seed: int = 0, input_amax: float = 4.0
                         ) -> dict[str, torch.Tensor]:
    """The toy's full weights in the quantized checkpoint's form."""
    full = toy_full_weights(dim, hidden, seed=seed)
    out = {}
    for name in ("up", "down"):
        q, s = law.quantize_weight(full[f"{name}.weight"])
        out[f"{name}.weight"] = q
        out[f"{name}.weight_scale"] = s
        out[f"{name}.input_scale"] = law.activation_scale(input_amax, margin=1.25)
        out[f"{name}.bias"] = full[f"{name}.bias"]
    return out


def fp8_toy_reference(weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """The dequant-mode forward, unsharded: activations in ``x``'s dtype, dequantized weights;
    the column bias fused into its matmul, the row bias added after (where the reduce is)."""
    def dequant(name, h):
        return (weights[f"{name}.weight"].float() * weights[f"{name}.weight_scale"]).to(h.dtype)

    up = F.linear(x, dequant("up", x), weights["up.bias"].to(x.dtype))
    h = F.gelu(up, approximate="tanh")
    return F.linear(h, dequant("down", h)) + weights["down.bias"].to(h.dtype)


def fp8_tp_mlp_worker(rank: int, world_size: int, dim: int, hidden: int, x_list: list,
                      mode: str) -> list:
    """Build the fp8 toy at tp=world, load its slices of the full fp8 weights, run ``x``
    eagerly or compiled (aot_eager, fullgraph) and return the output as a list."""
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.backends.tpu.core.weights import load_sharded_state_dict
    from difflet.pipeline.parallel_mesh import MeshSpec

    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=world_size))
    model = ToyFP8MLP(dim, hidden, dtype=torch.float32)
    load_sharded_state_dict(model, fp8_toy_full_weights(dim, hidden), tp_size=world_size,
                            tp_rank=rank)
    x = torch.tensor(x_list, dtype=torch.float32)
    if mode == "compiled":
        model = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)
    with torch.no_grad():
        out = model(x)
    return out.tolist()
