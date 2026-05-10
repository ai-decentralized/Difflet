"""Torch-native linear and embedding layers for CPU numerical checks."""

from __future__ import annotations

import torch.nn as nn


class ColumnParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=True, gather_output=True, **kwargs):
        del gather_output, kwargs
        super().__init__(input_size, output_size, bias=bias)


class RowParallelLinear(nn.Linear):
    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, **kwargs):
        del input_is_parallel, kwargs
        super().__init__(input_size, output_size, bias=bias)


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
