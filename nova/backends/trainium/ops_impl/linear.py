"""Trainium parallel linear passthroughs."""

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
