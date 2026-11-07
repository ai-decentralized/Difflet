"""Gloo workers and toys for the neuron MXFP8 linear tests (not collected by pytest).

``ToyFP8MLP`` is ``ToyTPMLP`` with both linears quantized: a column-parallel mxfp8 up projection
that keeps its output sharded and a row-parallel mxfp8 down projection whose all-reduce is the
only collective. ``fp8_toy_full_weights`` quantizes the toy's full fp32 weights with the
``difflet.quant`` block law (the row-parallel weight per rank K range, as the checkpoint writer
does), so every rank slices the same fp8 tensors the writer would produce, and
``fp8_toy_reference`` runs the same layer math unsharded on one process.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.fp8_linear import (
    FP8ColumnParallelLinear,
    FP8RowParallelLinear,
)
from difflet.quant import mx
from tests.unit.backends._neuron_toy import toy_full_weights

#: The toy's shape: K of both linears a multiple of 128 per rank at tp4 (the MX layout).
DIM, HIDDEN = 128, 512


class ToyFP8MLP(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = HIDDEN, *, compute: str = "dequant",
                 dtype=None, device=None):
        super().__init__()
        self.up = FP8ColumnParallelLinear(dim, hidden, bias=True, gather_output=False,
                                          dtype=dtype, device=device, compute=compute)
        self.down = FP8RowParallelLinear(hidden, dim, bias=True, input_is_parallel=True,
                                         dtype=dtype, device=device, compute=compute)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


def fp8_toy_full_weights(dim: int = DIM, hidden: int = HIDDEN, *, tp: int = 1, seed: int = 0
                         ) -> dict[str, torch.Tensor]:
    """The toy's full weights in the quantized checkpoint's form for a ``tp``-way launch (the
    row-parallel ``down`` quantized per rank K range)."""
    full = toy_full_weights(dim, hidden, seed=seed)
    out = {}
    for name, k_shards in (("up", 1), ("down", tp)):
        q, s = mx.quantize_mx_sharded(full[f"{name}.weight"], k_shards)
        out[f"{name}.weight"] = q
        out[f"{name}.weight_scale"] = s
        out[f"{name}.bias"] = full[f"{name}.bias"]
    return out


def fp8_toy_reference(weights: dict[str, torch.Tensor], x: torch.Tensor, *, tp: int = 1
                      ) -> torch.Tensor:
    """The dequant-mode forward, unsharded: activations in ``x``'s dtype, dequantized weights;
    the column bias fused into its matmul, the row bias added after (where the reduce is)."""
    def dequant(name, h, k_shards):
        return mx.dequantize_mx_sharded(weights[f"{name}.weight"], weights[f"{name}.weight_scale"],
                                        k_shards).to(h.dtype)

    up = F.linear(x, dequant("up", x, 1), weights["up.bias"].to(x.dtype))
    h = F.gelu(up, approximate="tanh")
    return F.linear(h, dequant("down", h, tp)) + weights["down.bias"].to(h.dtype)


def fp8_tp_mlp_worker(rank: int, world_size: int, dim: int, hidden: int, x_list: list,
                      mode: str) -> list:
    """Build the mxfp8 toy at tp=world, load its slices of the full quantized weights, prepare
    (dequant mode: the bf16 copy), run ``x`` eagerly or compiled (aot_eager, fullgraph) and
    return the output as a list."""
    from difflet.backends.neuron.ops_impl import parallel_mesh as pm
    from difflet.backends.neuron.ops_impl.fp8_linear import prepare_fp8_kernels_
    from difflet.backends.tpu.core.weights import load_sharded_state_dict
    from difflet.pipeline.parallel_mesh import MeshSpec

    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=world_size))
    model = ToyFP8MLP(dim, hidden, dtype=torch.float32)
    load_sharded_state_dict(model, fp8_toy_full_weights(dim, hidden, tp=world_size),
                            tp_size=world_size, tp_rank=rank)
    prepare_fp8_kernels_(model)
    x = torch.tensor(x_list, dtype=torch.float32)
    if mode == "compiled":
        model = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)
    with torch.no_grad():
        out = model(x)
    return out.tolist()
