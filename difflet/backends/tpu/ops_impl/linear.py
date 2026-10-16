"""Megatron-style tensor-parallel linear/embedding layers for TPU.

Unlike the CPU backend — whose `ColumnParallelLinear` is a plain `nn.Linear`
because it only ever runs at tp=1 — these shard for real. The plan originally
deferred sharding to Phase 5 and started TPU at tp=1, but v5e has 16 GB HBM
per chip and no candidate model's DiT fits at tp=1 in bf16, so TP moved up.

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

Every collective goes through ``ops_impl.collectives``, so it inherits the
opaque-custom-op wrapper that keeps ``torch.export`` working (Phase 0
finding).

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

Phase 2b of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# The layer bodies live in difflet.backends._torch_megatron.linear (CUDA plan D10)
# and are built here against the TPU collectives, so the XLA custom-op wrappers
# that keep torch.export working are the ones every collective goes through.
from difflet.backends._torch_megatron.linear import build_parallel_layers
from difflet.backends.tpu.ops_impl import collectives as _collectives

_layers = build_parallel_layers(_collectives)
ColumnParallelLinear = _layers.ColumnParallelLinear
RowParallelLinear = _layers.RowParallelLinear
ParallelEmbedding = _layers.ParallelEmbedding

__all__ = ["ColumnParallelLinear", "ParallelEmbedding", "RowParallelLinear"]
