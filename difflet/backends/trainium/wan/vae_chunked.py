"""Stateful chunked Wan VAE decode: the causal cache lives in HBM, not on the host.

Why this exists: ``WanVAEDecoderModel.forward`` loops over latent frames and
tracing flattens that loop, so one graph for the 21 latent frames behind 81
output frames needs 39,093,968 instructions against neuronx-cc's 5,000,000
ceiling (NCC_EBVF030; 6,675,705 already fails at 4 latent frames). Decoding in
chunks keeps each graph small, and ``difflet/models/wan/vae/chunked.py`` shows
the split is bit-identical to the single-shot decode, because the loop body was
already per-frame.

Why the cache has to be device state rather than a returned tensor: at 480x832
the 32 written cache slots hold 1,889 MB in bf16, the largest four being
``(1, 96, 2, 480, 832)`` at 153 MB each near the end of the decoder. Returning
them to the host and passing them back for each of the 7 chunks would move about
13 GB per decode, which would cost more than the host decode this replaces. So
each slot is an aliased ``nn.Parameter``, updated in place on device exactly as
the TeaCache probe does with ``prev_mod``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.models.wan.vae.chunked import DEFAULT_CHUNK_LATENT_FRAMES
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderModel


class StatefulChunkedWanVAEDecoder(nn.Module):
    """Decode one chunk, reading and rewriting the causal cache held on device.

    ``first`` picks the regime. The first chunk starts from an empty cache and
    takes the ``first_chunk`` path, where one slot briefly carries the string
    sentinel ``"Rep"``; every later chunk reads settled tensors. Keeping them as
    two modules means neither graph contains a branch on cache state.
    """

    def __init__(
        self,
        decoder: WanVAEDecoderModel,
        cache_shapes: list[tuple | None],
        *,
        first: bool,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.first = bool(first)
        # One Parameter per written slot, in slot order. Slots the decoder never
        # writes (conv_out's, whose call skips the cache path) get no Parameter:
        # they stay None inside the loop and never cross a chunk boundary.
        self.slot_index: list[int] = [i for i, s in enumerate(cache_shapes) if s is not None]
        self._cache_shapes = [cache_shapes[i] for i in self.slot_index]
        self._dtype = dtype
        # Only the later-chunk graph holds Parameters. The first chunk starts
        # from an empty cache by definition, so its Parameters would be read by
        # nothing -- and a Parameter that never enters the computation is absent
        # from the lowering context, which fails the trace with "parameter not
        # found in lowering context". It returns its cache as plain outputs
        # instead, which is all the caller needs to seed the later graph.
        self.cache = (
            nn.ParameterList()
            if self.first
            else nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)
                    for shape in self._cache_shapes
                ]
            )
        )
        self._num_slots = len(cache_shapes)

    def _load_cache(self) -> list:
        """Rebuild the decoder's cache list from the device-resident slots."""
        feat_cache: list = [None] * self._num_slots
        for param, slot in zip(self.cache, self.slot_index):
            feat_cache[slot] = param
        return feat_cache

    def forward(self, z_chunk: torch.Tensor):
        feat_cache = self.decoder._clear_cache() if self.first else self._load_cache()
        zq = self.decoder.post_quant_conv(z_chunk)
        pieces = []
        for i in range(int(zq.shape[2])):
            conv_idx = [0]
            pieces.append(
                self.decoder.decoder(
                    zq[:, :, i : i + 1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                    first_chunk=self.first and i == 0,
                )
            )
        pixels = torch.cat(pieces, dim=2)

        # Outputs: the chunk's pixels, then each written slot in slot order, so
        # the aliases below map Parameter -> output index deterministically.
        updated = []
        for position, slot in enumerate(self.slot_index):
            value = feat_cache[slot]
            if not isinstance(value, torch.Tensor):
                # The first chunk can leave the sentinel in the upsample3d slot.
                # Zeros are what the next chunk's zero-pad path would build, and
                # keep every output the declared shape. Built from the recorded
                # shape rather than from a Parameter, so nothing is read that the
                # graph did not otherwise compute.
                value = torch.zeros(
                    *self._cache_shapes[position], dtype=self._dtype, device=pixels.device
                )
            updated.append(value)
        return (pixels, *updated)


class _ChunkedVAEInstance(BaseModelInstance):
    """Alias each cache Parameter onto its output, so updates stay in HBM."""

    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        module = self.module
        # Output 0 is the pixels; the slots follow in order.
        aliases = {param: 1 + i for i, param in enumerate(module.cache)}
        return module, aliases


def cache_shapes_for(
    decoder: WanVAEDecoderModel,
    latent_height: int,
    latent_width: int,
    *,
    z_dim: int,
    chunk: int = DEFAULT_CHUNK_LATENT_FRAMES,
    batch_size: int = 1,
) -> list[tuple | None]:
    """Slot shapes a later-chunk graph declares, measured by running one chunk.

    Shapes are a property of the decoder and the spatial size, not of how many
    frames have been decoded, so one first-chunk run on CPU settles them.
    """
    from difflet.models.wan.vae.chunked import WanVAEDecoderChunk

    probe = torch.randn(batch_size, z_dim, chunk, latent_height, latent_width)
    with torch.no_grad():
        _, cache = WanVAEDecoderChunk(decoder, first=True)(probe)
    shapes: list[tuple | None] = []
    for i, slot in enumerate(cache):
        if isinstance(slot, torch.Tensor):
            shapes.append(tuple(slot.shape))
        elif slot is None:
            shapes.append(None)
        else:
            raise RuntimeError(
                f"cache slot {i} is {slot!r} after a {chunk}-frame chunk; a later-chunk "
                "graph cannot declare it."
            )
    return shapes


__all__ = [
    "StatefulChunkedWanVAEDecoder",
    "_ChunkedVAEInstance",
    "cache_shapes_for",
]
