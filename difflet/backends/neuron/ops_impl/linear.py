"""Megatron tensor-parallel linear and embedding layers for the neuron backend.

Ported from ``difflet/backends/tpu/ops_impl/linear.py`` by copy, not by import:
the TPU classes size their shards from the TPU mesh at construction, and
importing them would also register the ``difflet_tpu::*`` custom ops in every
neuron process. The loader-side helpers in ``difflet/backends/tpu/core/weights.py``
depend only on ``_difflet_shard`` and are reused by import.

Sharding follows Megatron:

* ``ColumnParallelLinear`` owns ``[out/tp, in]`` plus its bias slice and needs
  no communication; ``gather_output=True`` all-gathers the last dim.
* ``RowParallelLinear`` owns ``[out, in/tp]`` and a replicated bias. The partial
  sums are all-reduced and the bias is added once, after the reduce.
* ``ParallelEmbedding`` shards the vocab (masked lookup + all-reduce) or, with
  ``shard_across_embedding=True``, the feature dim (lookup + all-gather).

Unlike the TPU layer, ``RowParallelLinear`` honours NxD's ``reduce_output`` and
``skip_bias_add`` instead of dropping them in ``**kwargs``. ``reduce_output=False``
returns this rank's partial plus the FULL bias, as NxD does, which is what
``modeling_wan._sp_unbias`` corrects; ``skip_bias_add=True`` returns
``(output_without_bias, bias)``. The other NxD keywords (``reduce_dtype``,
``pad``, ...) are accepted and ignored, except ``sequence_parallel_enabled=True``:
NxD would scatter and gather the sequence inside the layer, so ignoring it would
be silently wrong.

Collectives come from ``neuron.ops_impl.collectives`` (functional collectives on
the TP process group, the identity at tp == 1), so one module runs eagerly and
under ``torch.compile(backend="neuron", fullgraph=True)``.

Each layer declares ``_difflet_shard = {parameter: split dim or None}`` on the
module, not on the Parameter, because ``accelerate.init_empty_weights``
re-creates parameters. Construction needs only ``get_tp_size()``, and the rank is
resolved in ``forward``, so layers build on the meta device before any process
group exists.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.collectives import (
    gather_tp_dim,
    get_tp_rank,
    get_tp_size,
    reduce_tp,
    scatter_tp_dim,
)


def _split_evenly(total: int, what: str) -> int:
    tp = get_tp_size()
    if total % tp != 0:
        raise ValueError(f"cannot shard {what} of size {total} across tp={tp}")
    return total // tp


def _reject_sequence_parallel(layer: str, kwargs: dict) -> None:
    if kwargs.get("sequence_parallel_enabled"):
        raise NotImplementedError(
            f"{layer}(sequence_parallel_enabled=True) is not supported on the neuron "
            "backend; difflet models apply sequence parallelism outside the layer"
        )


def _init_linear_(weight: torch.Tensor, bias: torch.Tensor | None, fan_in: int) -> None:
    """Initialise like ``nn.Linear``; the bias bound uses the UNSHARDED fan-in.

    As on TPU: never leave parameters as ``torch.empty``, so a checkpoint that
    misses one yields wrong-but-finite numbers instead of uninitialised memory.
    A no-op on the meta device.
    """
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)


class ColumnParallelLinear(nn.Module):
    """``y = x A^T + b`` with ``A`` split along its output dim."""

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        if kwargs.get("skip_bias_add"):
            raise NotImplementedError(
                "ColumnParallelLinear(skip_bias_add=True) is not supported on the neuron "
                "backend; no difflet model uses it"
            )
        _reject_sequence_parallel("ColumnParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.output_size_per_partition = _split_evenly(output_size, "output_size")

        factory = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, **factory))
        else:
            self.register_parameter("bias", None)
        # The weight rows and the bias both belong to this rank's output slice.
        self._difflet_shard = {"weight": 0, "bias": 0}
        _init_linear_(self.weight, self.bias, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The bias is per-partition, so it folds straight into the local matmul.
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = gather_tp_dim(out, dim=-1)
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.input_size}, out={self.output_size}, "
            f"out_per_partition={self.output_size_per_partition}, "
            f"gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    """``y = x A^T + b`` with ``A`` split along its input dim."""

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        dtype=None,
        device=None,
        reduce_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        super().__init__()
        _reject_sequence_parallel("RowParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.input_size_per_partition = _split_evenly(input_size, "input_size")

        factory = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, **factory))
        else:
            self.register_parameter("bias", None)
        # The bias is replicated (None), not sharded: it is added once, after the
        # all-reduce; sharding it would drop (tp - 1)/tp of it.
        self._difflet_shard = {"weight": 1, "bias": None}
        _init_linear_(self.weight, self.bias, input_size)

    def forward(self, x: torch.Tensor):
        if not self.input_is_parallel:
            x = scatter_tp_dim(x, dim=-1)
        # Partial sum over this rank's input slice; the all-reduce completes it.
        out = F.linear(x, self.weight)
        if self.reduce_output:
            out = reduce_tp(out)
        if self.skip_bias_add:
            return out, self.bias
        if self.bias is not None:
            # After the reduce: adding before it would sum the bias tp times. With
            # reduce_output=False the caller owns the reduce and the (tp - 1)
            # extra biases it then carries (modeling_wan._sp_unbias).
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.input_size}, out={self.output_size}, "
            f"in_per_partition={self.input_size_per_partition}, "
            f"input_is_parallel={self.input_is_parallel}, "
            f"reduce_output={self.reduce_output}, skip_bias_add={self.skip_bias_add}"
        )


class ParallelEmbedding(nn.Module):
    """Embedding sharded across the vocab (default) or the embedding dim."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        shard_across_embedding=False,
        pad=False,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        del pad
        _reject_sequence_parallel("ParallelEmbedding", kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.shard_across_embedding = shard_across_embedding
        factory = {"dtype": dtype, "device": device}

        if shard_across_embedding:
            self.embedding_dim_per_partition = _split_evenly(embedding_dim, "embedding_dim")
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, self.embedding_dim_per_partition, **factory)
            )
            self._difflet_shard = {"weight": 1}
        else:
            self.num_embeddings_per_partition = _split_evenly(num_embeddings, "num_embeddings")
            # vocab_start depends on the rank, so it is resolved in forward():
            # construction must not need a live process group.
            self.weight = nn.Parameter(
                torch.empty(self.num_embeddings_per_partition, embedding_dim, **factory)
            )
            self._difflet_shard = {"weight": 0}
        nn.init.normal_(self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.shard_across_embedding:
            # Each rank holds a slice of the feature dim; gather to full width.
            return gather_tp_dim(F.embedding(input_ids, self.weight), dim=-1)

        if get_tp_size() == 1:
            return F.embedding(input_ids, self.weight)

        # Vocab-parallel: look up only the ids this rank owns, zero the rest, then
        # all-reduce so every rank ends up with the full result.
        vocab_start = get_tp_rank() * self.num_embeddings_per_partition
        vocab_end = vocab_start + self.num_embeddings_per_partition
        mask = (input_ids < vocab_start) | (input_ids >= vocab_end)
        local_ids = (input_ids - vocab_start).masked_fill(mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return reduce_tp(out)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"shard_across_embedding={self.shard_across_embedding}"
        )


__all__ = ["ColumnParallelLinear", "ParallelEmbedding", "RowParallelLinear"]
