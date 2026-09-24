"""Chunked Wan VAE decode must equal the single-shot decode exactly.

The chunked path exists so the decoder fits a compiled graph: tracing flattens
the per-latent-frame loop, so one graph for 21 latent frames needs 39,093,968
instructions against neuronx-cc's 5,000,000 ceiling (NCC_EBVF030). Chunking is
only worth doing if it changes nothing numerically, so that is what these pin.
"""

from __future__ import annotations

import pytest
import torch

from difflet.models.wan.vae.chunked import (
    DEFAULT_CHUNK_LATENT_FRAMES,
    cache_spec,
    decode_chunked,
)
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


@pytest.fixture(scope="module")
def decoder() -> WanVAEDecoderModel:
    """A small decoder with the real module structure, including a temporal upsample.

    The temporal upsample matters: it is the path that parks the ``"Rep"``
    sentinel in its cache slot on the first frame, which is what makes the first
    chunk a different graph from every later one.
    """
    torch.manual_seed(0)
    config = WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True],
        dropout=0.0,
        out_channels=3,
    )
    return WanVAEDecoderModel(config).eval()


def _latents(decoder: WanVAEDecoderModel, frames: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, decoder.config.z_dim, frames, 16, 16)


@pytest.mark.parametrize("frames,chunk", [(3, 3), (6, 3), (9, 3), (21, 3), (6, 6), (21, 6)])
def test_chunked_decode_is_bit_identical(decoder, frames, chunk):
    z = _latents(decoder, frames)
    with torch.no_grad():
        reference = decoder(z)
        chunked = decode_chunked(decoder, z, chunk=chunk)
    assert chunked.shape == reference.shape
    assert torch.equal(chunked, reference), (
        f"chunked decode diverged at frames={frames} chunk={chunk}: "
        f"max |diff| = {(chunked - reference).abs().max().item():.3e}"
    )


def test_frame_count_not_a_multiple_of_the_chunk(decoder):
    """A trailing short chunk must still decode exactly."""
    z = _latents(decoder, 8)  # 2 full chunks of 3, then a chunk of 2
    with torch.no_grad():
        reference = decoder(z)
        chunked = decode_chunked(decoder, z, chunk=3)
    assert torch.equal(chunked, reference)


def test_cache_slots_are_fixed_shape_tensors_after_the_first_chunk(decoder):
    """Every slot a later-chunk graph declares must be a tensor of known shape."""
    spec = cache_spec(decoder, _latents(decoder, DEFAULT_CHUNK_LATENT_FRAMES))
    assert spec, "the decoder reported no cache slots"
    assert all(isinstance(shape, tuple) and len(shape) == 5 for _, shape, _ in spec)
    # Each slot keeps CACHE_T temporal entries, which is what makes the shape
    # independent of how many frames have been decoded so far.
    assert {shape[2] for _, shape, _ in spec} == {2}


def test_cache_shapes_do_not_drift_between_chunks(decoder):
    """A later chunk must return the same slot shapes it was given.

    Otherwise the compiled graph's outputs could not be fed back as its inputs.
    """
    from difflet.models.wan.vae.chunked import WanVAEDecoderChunk

    z = _latents(decoder, 12)
    first = WanVAEDecoderChunk(decoder, first=True)
    later = WanVAEDecoderChunk(decoder, first=False)
    with torch.no_grad():
        _, cache = first(z[:, :, :3, :, :])
        before = [tuple(c.shape) for c in cache]
        _, cache = later(z[:, :, 3:6, :, :], *cache)
        after = [tuple(c.shape) for c in cache]
        _, cache = later(z[:, :, 6:9, :, :], *cache)
        after2 = [tuple(c.shape) for c in cache]
    assert before == after == after2


def test_chunk_shorter_than_the_first_regime_is_rejected(decoder):
    """A 1- or 2-frame chunk cannot settle the cache, so it must not be allowed."""
    z = _latents(decoder, 9)
    with pytest.raises(ValueError, match="at least"):
        decode_chunked(decoder, z, chunk=1)
