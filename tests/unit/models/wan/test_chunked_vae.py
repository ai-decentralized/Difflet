"""Numerical and trace-boundary regression tests for streaming Wan VAE."""

from dataclasses import asdict

import pytest
import torch

from difflet.models.wan.vae.chunked import WanVAEChunkModel, cache_layouts, decode_chunks
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = WanVAEDecoderConfig(base_dim=8, z_dim=4, num_res_blocks=1)
    full = WanVAEDecoderModel(config).eval()
    chunked = WanVAEChunkModel(config).eval()
    chunked.load_state_dict(full.state_dict(), strict=True)
    yield config, full, chunked
    torch.set_num_threads(previous)


@pytest.mark.parametrize("latent_frames", [1, 2, 3, 9, 21])
def test_chunked_matches_full_decode_and_resets_between_requests(models, latent_frames):
    _, full, chunked = models
    z = torch.randn(1, 4, latent_frames, 2, 3)
    with torch.no_grad():
        expected = full(z)
        actual = decode_chunks(chunked, z)
        # A second request must start from an empty cache, even after 81 frames.
        again = decode_chunks(chunked, z)
    assert actual.shape == (1, 3, 1 + 4 * (latent_frames - 1), 16, 24)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(again, actual, rtol=0, atol=0)


def test_chunked_matches_diffusers_reference(models):
    from diffusers import AutoencoderKLWan

    config, full, chunked = models
    reference = AutoencoderKLWan(**asdict(config)).eval()
    reference.load_state_dict(full.state_dict(), strict=False)
    z = torch.randn(1, 4, 4, 2, 3)
    with torch.no_grad():
        expected = reference.decode(z, return_dict=False)[0].clamp(-1, 1)
        actual = decode_chunks(chunked, z)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_three_traces_preserve_temporal_state_for_longer_requests(models):
    _, full, chunked = models
    z = torch.randn(1, 4, 5, 2, 2)
    graphs = {}
    state = ()
    with torch.no_grad():
        for index in range(3):
            state = state or tuple(z.new_zeros((1,)) for _ in chunked.state_slots)
            inputs = (z[:, :, index:index + 1].contiguous(), *state)
            graphs[index] = torch.jit.trace(chunked, inputs, check_trace=False)
            state = chunked(*inputs)[1:]
        def routed(frame, *cache):
            phase = 0 if not cache else (1 if cache[0].shape[2] == 1 else 2)
            cache = cache or tuple(frame.new_zeros((1,)) for _ in chunked.state_slots)
            return graphs[phase](frame, *cache)
        actual = decode_chunks(routed, z)
        expected = full(z)
    assert len(graphs) == 3
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_meta_layout_matches_real_cache_shapes(models):
    config, _, chunked = models
    z = torch.randn(1, 4, 1, 2, 3)
    layouts = cache_layouts(config, tuple(z.shape))
    with torch.no_grad():
        first = chunked(z)[1:]
        second = chunked(z, *first)[1:]
        third = chunked(z, *second)[1:]
    for layout, state in zip(layouts, (first, second)):
        assert [layout[i] if isinstance(layout[i], tuple) else (1,) for i in chunked.state_slots] == [tuple(t.shape) for t in state]
    assert [t.shape for t in second] == [t.shape for t in third]


def test_empty_video_rejected(models):
    with pytest.raises(ValueError, match="nonempty"):
        decode_chunks(models[2], torch.empty(1, 4, 0, 2, 2))


@pytest.mark.parametrize("latent_frames", [1, 2, 3, 21])
def test_split_chunks_match_full_decoder(models, latent_frames):
    from difflet.models.wan.vae.chunked import WanVAESplitChunkModel, decode_split_chunks

    config, full, _ = models
    split = WanVAESplitChunkModel(config).eval()
    split.load_state_dict(full.state_dict(), strict=True)
    z = torch.randn(1, config.z_dim, latent_frames, 2, 3)
    with torch.no_grad():
        expected = full(z)
        actual = decode_split_chunks(split, z)
        again = decode_split_chunks(split, z)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(again, actual, rtol=0, atol=0)


def test_split_trace_signatures_cover_long_video(models):
    from difflet.models.wan.vae.chunked import (
        WanVAESplitChunkModel, decode_split_chunks, split_input_shapes,
    )

    config, full, _ = models
    split = WanVAESplitChunkModel(config).eval()
    split.load_state_dict(full.state_dict(), strict=True)
    signatures = split_input_shapes(config, (1, config.z_dim, 1, 2, 2))
    # Never carry the other partition's histories through a graph: those
    # otherwise allocate persistent HBM I/O buffers for every bucket.
    assert split.state_count < len(split.state_slots)
    assert all(len(s) == split.state_count + 1 for s in signatures)
    graphs = {}
    with torch.no_grad():
        for signature in signatures:
            inputs = tuple(torch.randn(shape) for shape in signature)
            graphs[signature] = torch.jit.trace(split, inputs, check_trace=False)

        def routed(x, *state):
            state = state or tuple(x.new_zeros((1,)) for _ in range(split.state_count))
            signature = tuple(tuple(t.shape) for t in (x, *state))
            return graphs[signature](x, *state)

        z = torch.randn(1, config.z_dim, 21, 2, 2)
        actual = decode_split_chunks(routed, z)
        expected = full(z)
    assert len(graphs) == 6
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
