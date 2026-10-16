"""Tensor-parallel linear/embedding layers for the CUDA backend.

The bodies live in ``difflet.backends._torch_megatron.linear`` (plan D10) and are
built here against the CUDA collectives — identity at world size 1, NCCL in
follow-up F2 — so the same sharding code serves TPU and CUDA.
"""

from difflet.backends._torch_megatron.linear import build_parallel_layers
from difflet.backends.cuda.ops_impl import collectives as _collectives

_layers = build_parallel_layers(_collectives)
ColumnParallelLinear = _layers.ColumnParallelLinear
RowParallelLinear = _layers.RowParallelLinear
ParallelEmbedding = _layers.ParallelEmbedding

__all__ = ["ColumnParallelLinear", "ParallelEmbedding", "RowParallelLinear"]
