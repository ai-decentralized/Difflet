"""Split full host checkpoints into this rank's tensor-parallel shards.

This is the TPU analogue of NxD's ``convert_hf_to_neuron_state_dict``. The
plan warned the Trainium version would not transfer, and it does not: NxD
splits some fused projections in its own layout (see the Flux
``proj_out_attn``/``proj_out_mlp`` split), whereas the TPU layers here declare
their own split axis via a module-level ``_difflet_shard`` mapping.

Driving the split off that declaration — rather than a per-model table of
weight names — means a layer and its loader cannot disagree: if
``ColumnParallelLinear`` ever changed which dim it shards, the loader follows
automatically.

Phase 3 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# Lifted into difflet.backends._torch_megatron (CUDA plan D10); re-exported here so
# the TPU backend and its tests keep their import paths.
from difflet.backends._torch_megatron.weights import (  # noqa: F401
    shard_dim,
    narrow_to_rank,
    shard_state_dict,
    load_sharded_state_dict,
    materialize_meta_,
)

__all__ = [
    "load_sharded_state_dict",
    "materialize_meta_",
    "narrow_to_rank",
    "shard_dim",
    "shard_state_dict",
]
