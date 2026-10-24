"""Megatron-style tensor-parallel linear/embedding layers, shared by torch backends.

Sharding follows Megatron:

* ``ColumnParallelLinear`` splits the OUTPUT dim. Each rank owns
  ``[out/tp, in]`` and computes an independent slice of the output; the
  optional ``gather_output`` all-gathers the slices back on the last dim.
* ``RowParallelLinear`` splits the INPUT dim. Each rank owns
  ``[out, in/tp]`` and produces a *partial* sum, finished with an all-reduce.
  **Bias is added after the all-reduce**, never before — folding it in first
  would add it ``tp`` times.
* ``ParallelEmbedding`` shards the vocab by default (masked lookup +
  all-reduce), or the embedding dim when ``shard_across_embedding=True``.

The layers are pure torch; the only backend-specific thing they touch is the
collectives module they call. ``build_parallel_layers(collectives)`` returns
fresh classes closed over that module, so the TPU backend (XLA custom-op
collectives) and the GPU backends (identity at world size 1, NCCL later) share
one implementation without sharing state. The bodies are the TPU backend's,
unchanged.

**Weight loading:** parameters are allocated already-sharded, so a checkpoint
loader must split host weights before ``load_state_dict``. Each layer declares
``_difflet_shard`` — ``{parameter name: split dim or None}`` — and the rank
comes from ``get_tp_rank()`` at load time. Do not assume the Trainium
remapping transfers: NxD splits some fused projections differently.

The declaration lives on the *module*, deliberately not on the parameter.
Sharding is a property of the layer's design, and module-level metadata
survives parameter re-creation — which happens in `accelerate`'s
`init_empty_weights`, in `to_empty()`, and in wrapper layers. (Attaching it to
the Parameter broke `init_empty_weights` outright: it re-creates parameters
and forwards their attributes as constructor kwargs.)
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["build_parallel_layers"]


def _init_linear_(weight: torch.Tensor, bias: torch.Tensor | None, fan_in: int) -> None:
    """Initialize like ``nn.Linear`` does, using the UNSHARDED fan-in.

    Never leave parameters as ``torch.empty``: a checkpoint that misses a
    parameter then yields NaN weights that propagate silently instead of
    something obviously wrong. (Observed for real — uninitialized memory made
    a test pass or fail depending on what ran before it.)

    fan_in is the full input size, not this rank's slice, so that sharding
    does not change the initialization distribution.
    """
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)


def build_parallel_layers(collectives) -> SimpleNamespace:
    """Build ``ColumnParallelLinear``, ``RowParallelLinear`` and ``ParallelEmbedding``
    against ``collectives``.

    ``collectives`` must provide ``gather_tp_dim(t, *, dim)``, ``scatter_tp_dim(t, *, dim)``,
    ``reduce_tp(t)``, ``get_tp_size() -> int``, ``get_tp_rank() -> int`` and
    ``get_tp_groups() -> list[list[int]]``. Each call returns new classes, so two
    backends in one process never share them.
    """

    gather_tp_dim = collectives.gather_tp_dim
    scatter_tp_dim = collectives.scatter_tp_dim
    reduce_tp = collectives.reduce_tp
    get_tp_size = collectives.get_tp_size
    get_tp_rank = collectives.get_tp_rank
    get_tp_groups = collectives.get_tp_groups

    def _split_evenly(total: int, what: str) -> int:
        tp = get_tp_size()
        if total % tp != 0:
            raise ValueError(f"cannot shard {what} of size {total} across tp={tp}")
        return total // tp

    class ColumnParallelLinear(nn.Module):
        """``y = xA^T + b`` with ``A`` split along its output dim."""

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
            del kwargs
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
            # Both the weight rows and the bias belong to this rank's output slice.
            self._difflet_shard = {"weight": 0, "bias": 0}
            _init_linear_(self.weight, self.bias, input_size)

        def forward(self, x):
            # Bias is per-partition here (each rank owns a distinct output slice),
            # so unlike RowParallelLinear it can be folded straight in.
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
        """``y = xA^T + b`` with ``A`` split along its input dim.

        ``reduce_output`` and ``skip_bias_add`` follow the NxD layer the modeling code
        was written against (defaults are NxD's, so callers that pass neither are
        unchanged):

        * ``reduce_output=False`` returns the un-reduced partial; the caller reduces
          (Flux's single block merges two row-parallel reduces into one). As in NxD,
          the full bias is then added to the partial — the modeling's ``_sp_unbias``
          removes the ``tp-1`` extra copies after its own reduce.
        * ``skip_bias_add=True`` returns ``(out, bias)`` with the bias NOT added
          (``bias`` is ``None`` when the layer has none), for the caller to add once.

        ``tensor_parallel_group`` is the TP replica group the caller reduces over
        (``reduce_from_tensor_model_parallel_region(..., process_group=)``); ``None``
        at tp=1, where there is nothing to reduce.
        """

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
            del kwargs
            self.input_size = input_size
            self.output_size = output_size
            self.input_is_parallel = input_is_parallel
            self.reduce_output = reduce_output
            self.skip_bias_add = skip_bias_add
            self.input_size_per_partition = _split_evenly(input_size, "input_size")
            self.tensor_parallel_group = None if get_tp_size() == 1 else get_tp_groups()

            factory = {"dtype": dtype, "device": device}
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition, **factory)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(output_size, **factory))
            else:
                self.register_parameter("bias", None)
            # bias is replicated (None), NOT sharded: it is added once, after the
            # all-reduce — sharding it would drop 3/4 of it at tp=4.
            self._difflet_shard = {"weight": 1, "bias": None}
            _init_linear_(self.weight, self.bias, input_size)

        def forward(self, x):
            if not self.input_is_parallel:
                x = scatter_tp_dim(x, dim=-1)
            # Partial sum over this rank's input slice; the all-reduce completes it.
            out = F.linear(x, self.weight)
            if self.reduce_output:
                out = reduce_tp(out)
            if self.skip_bias_add:
                return out, self.bias
            if self.bias is not None:
                # After the reduce — adding before would sum the bias tp times. With
                # reduce_output=False this is NxD's partial + bias (see the docstring).
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
            del pad, kwargs
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
                # vocab_start is rank-dependent, so it is resolved in forward()
                # rather than here — construction must not need a live runtime.
                self.weight = nn.Parameter(
                    torch.empty(self.num_embeddings_per_partition, embedding_dim, **factory)
                )
                self._difflet_shard = {"weight": 0}
            nn.init.normal_(self.weight)

        def forward(self, input_ids):
            if self.shard_across_embedding:
                # Each rank holds a slice of the feature dim; gather to full width.
                return gather_tp_dim(F.embedding(input_ids, self.weight), dim=-1)

            if get_tp_size() == 1:
                return F.embedding(input_ids, self.weight)

            # Vocab-parallel: look up only ids owned by this rank, zero the rest,
            # then all-reduce so every rank ends up with the full result.
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

    return SimpleNamespace(
        ColumnParallelLinear=ColumnParallelLinear,
        RowParallelLinear=RowParallelLinear,
        ParallelEmbedding=ParallelEmbedding,
    )
