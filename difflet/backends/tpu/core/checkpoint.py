"""Load a safetensors checkpoint straight into this rank's shards.

``weights.shard_state_dict`` splits an already-materialized state dict. That
is fine for a toy module and impossible for a real one: Qwen-Image's
transformer is 38 GiB, and four ranks each materializing the full tensor set
would need ~152 GiB of host RAM before a single shard is taken.

So this reads *lazily*. ``safetensors``' slice API fetches only the requested
sub-range off disk, so each rank pays roughly ``38 GiB / tp`` instead of the
whole file. The split axis still comes from the layer's own ``_difflet_shard``
declaration on the owning module, exactly as in ``weights.py`` — one source of
truth for how a parameter is partitioned.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# Lifted into difflet.backends._torch_megatron (CUDA plan D10); re-exported here so
# the TPU backend and its tests keep their import paths.
from difflet.backends._torch_megatron.checkpoint import (  # noqa: F401
    INDEX_SUFFIX,
    build_weight_map,
    load_checkpoint_into,
)

__all__ = ["build_weight_map", "load_checkpoint_into"]
