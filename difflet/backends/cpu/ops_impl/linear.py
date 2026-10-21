"""Torch-native linear and embedding layers for CPU numerical checks."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=True, gather_output=True, **kwargs):
        del gather_output, kwargs
        super().__init__(input_size, output_size, bias=bias)


class RowParallelLinear(nn.Linear):
    """``nn.Linear`` with the NxD ``skip_bias_add`` contract (tp=1: nothing to reduce).

    ``skip_bias_add=True`` returns ``(x @ W^T, bias)`` with the bias left for the
    caller to add (``None`` when the layer has none); ``reduce_output`` is accepted
    and has no effect at tp=1; ``tensor_parallel_group`` is ``None`` (no group).
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        reduce_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        del input_is_parallel, reduce_output, kwargs
        super().__init__(input_size, output_size, bias=bias)
        self.skip_bias_add = skip_bias_add
        self.tensor_parallel_group = None

    def forward(self, x):
        if self.skip_bias_add:
            return F.linear(x, self.weight), self.bias
        return super().forward(x)


class ParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        shard_across_embedding=False,
        pad=False,
        **kwargs,
    ):
        del shard_across_embedding, pad, kwargs
        super().__init__(num_embeddings, embedding_dim)


__all__ = ["ColumnParallelLinear", "ParallelEmbedding", "RowParallelLinear"]
