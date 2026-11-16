"""Decode a VAE's spatial tiles across the ranks of a CPU process group.

For a decoder that splits its tiled decode into ``tile_latents(z)``, ``decode_tile(tile)`` and
``blend_tiles(z, tiles)`` (the Wan VAE decoder), rank r decodes tiles r, r + N, ... on its own
core, the decoded tiles go to group rank 0 through the host, and rank 0 blends them on its device
with the same ``blend_tiles`` the single-core tiled decode uses, so the frames are bit-identical
to it. No device collectives: the ranks share nothing but the host exchange.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def decode_tiles_across_ranks(model, z: torch.Tensor, group) -> torch.Tensor | None:
    """``model.tiled_decode(z)`` with its tiles decoded across ``group``; the frames on group
    rank 0, ``None`` on the others."""
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    latents = model.tile_latents(z)
    mine = {k: model.decode_tile(t).cpu() for k, t in enumerate(latents) if k % world == rank}
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(mine, gathered, dst=dist.get_global_rank(group, 0), group=group)
    if rank != 0:
        return None
    tiles = {k: t for part in gathered for k, t in part.items()}
    return model.blend_tiles(z, [tiles[k].to(z.device) for k in range(len(latents))])
