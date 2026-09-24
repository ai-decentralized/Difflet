"""Chunked Wan VAE decode, so the decoder fits a compiled graph.

``WanVAEDecoderModel.forward`` loops over latent frames and tracing flattens that
loop, so the compiled graph grows with the frame count. At 480x832 neuronx-cc
rejects it past 3 latent frames (NCC_EBVF030, limit 5,000,000 instructions;
measured 6,675,705 at 4 latent frames and 39,093,968 at the 21 that 81 output
frames need). The transformer is unaffected -- its attention is tiled -- so the
VAE alone is what pins Wan to 9 frames on device.

The loop body already runs one latent frame at a time, carrying a causal
``feat_cache`` between iterations, exactly as upstream diffusers does
(``AutoencoderKLWan._decode``). Decoding in fixed-size chunks and passing that
cache between them is therefore an equivalence, not an approximation:
``scripts/wan_vae_chunked_parity.py`` checks it is bit-identical.

Two graphs suffice. The cache has three regimes, not two -- frame 0 leaves the
upsample3d slot holding the string sentinel ``"Rep"``, frame 1 reads that
sentinel and takes a different ``time_conv`` call than later frames, and from
frame 2 on every slot is a fixed-shape tensor. A chunk of at least 3 latent
frames swallows all three inside the first chunk, so only two graph shapes are
ever compiled: the first chunk, and every later chunk.
"""

from __future__ import annotations

import torch
from torch import nn

from difflet.models.wan.vae.modeling_vae import WanVAEDecoderModel

# Latent frames per compiled chunk. Three is the smallest that contains the
# first-chunk cache regimes, and is also what already compiles at 480x832.
DEFAULT_CHUNK_LATENT_FRAMES = 3


class WanVAEDecoderChunk(nn.Module):
    """Decode one chunk of latent frames, threading the causal cache explicitly.

    ``first`` selects the regime: the first chunk starts from an empty cache and
    returns a full set of tensors; every later chunk takes those tensors and
    returns their successors with identical shapes, so on device the cache is
    graph state rather than a host round trip.
    """

    def __init__(self, decoder: WanVAEDecoderModel, *, first: bool) -> None:
        super().__init__()
        self.decoder = decoder
        self.first = bool(first)

    def forward(self, z_chunk: torch.Tensor, *cache: torch.Tensor):
        feat_cache: list = self.decoder._clear_cache() if self.first else list(cache)
        zq = self.decoder.post_quant_conv(z_chunk)
        chunks = []
        for i in range(int(zq.shape[2])):
            conv_idx = [0]
            chunks.append(
                self.decoder.decoder(
                    zq[:, :, i : i + 1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                    first_chunk=self.first and i == 0,
                )
            )
        return torch.cat(chunks, dim=2), feat_cache


def decode_chunked(
    model: WanVAEDecoderModel,
    z: torch.Tensor,
    *,
    chunk: int = DEFAULT_CHUNK_LATENT_FRAMES,
    clamp: bool = True,
) -> torch.Tensor:
    """Reference driver: decode ``z`` in chunks of ``chunk`` latent frames.

    Equivalent to ``model(z)``. This is the shape the device path takes, with the
    cache owned by the caller instead of by one traced loop.
    """
    total = int(z.shape[2])
    if chunk < DEFAULT_CHUNK_LATENT_FRAMES and total > chunk:
        raise ValueError(
            f"chunk must be at least {DEFAULT_CHUNK_LATENT_FRAMES} latent frames so the "
            f"first chunk contains the first-frame cache regimes; got {chunk}."
        )

    first = WanVAEDecoderChunk(model, first=True)
    later = WanVAEDecoderChunk(model, first=False)

    out, cache = first(z[:, :, :chunk, :, :])
    pieces = [out]
    for start in range(chunk, total, chunk):
        out, cache = later(z[:, :, start : start + chunk, :, :], *cache)
        pieces.append(out)

    result = torch.cat(pieces, dim=2)
    return torch.clamp(result, min=-1.0, max=1.0) if clamp else result


def cache_spec(model: WanVAEDecoderModel, z_chunk: torch.Tensor) -> list[tuple]:
    """Shapes and dtypes a later-chunk graph must declare for its cache inputs.

    Slots the decoder never writes are reported as ``None`` rather than raised
    on: ``_clear_cache`` sizes the list by counting every ``WanCausalConv3d``,
    but the last one (``conv_out``) is called outside the cache-writing path, so
    its slot stays empty however many frames are decoded. Only the written slots
    have to cross between chunks.
    """
    _, cache = WanVAEDecoderChunk(model, first=True)(z_chunk)
    spec: list[tuple] = []
    for i, slot in enumerate(cache):
        if isinstance(slot, torch.Tensor):
            spec.append((i, tuple(slot.shape), slot.dtype))
        elif slot is None:
            spec.append((i, None, None))
        else:
            # A string sentinel here means the chunk ended before the cache
            # settled, which would make a later-chunk graph's inputs ill-defined.
            raise RuntimeError(
                f"cache slot {i} is {slot!r} after the first chunk; the chunk is too "
                f"short to settle the cache (need >= {DEFAULT_CHUNK_LATENT_FRAMES} latent frames)."
            )
    return spec


__all__ = ["WanVAEDecoderChunk", "decode_chunked", "cache_spec", "DEFAULT_CHUNK_LATENT_FRAMES"]
