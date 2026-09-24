"""The chunked Wan VAE application must build and declare chunk-sized inputs.

These pin two mistakes that only surface at compile time, after minutes of setup:

* ``ChunkedVAEWrapper`` must inherit ``ShapeBucketedInputGenerator`` before
  ``ModelWrapper``. Without it, ``ModelWrapper.input_generator`` builds LLM-shaped
  inputs and calls ``prepare_sampling_params``, which is None for a VAE and dies
  with "'NoneType' object is not callable" -- which is exactly how the first
  81-frame compile failed.
* The graph's temporal extent must be the chunk, not the request's frame count.
  Declaring 21 latent frames would rebuild the graph the chunking exists to avoid.
"""

from __future__ import annotations

import pytest
import torch

from difflet.backends.trainium.core.bucketing import ShapeBucketedInputGenerator
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.backends.trainium.wan.vae_chunked_app import (
    ChunkedVAEWrapper,
    NeuronWanVAEDecoderChunkedApplication,
)
from difflet.models.wan.application import (
    MAX_SINGLE_SHOT_LATENT_FRAMES,
    _vae_chunk_for,
    create_wan_vae_decoder_config,
)

CHUNK = 3
HEIGHT, WIDTH, FRAMES = 480, 832, 81


def _config(model_dir, frames=FRAMES):
    return create_wan_vae_decoder_config(
        model_path=str(model_dir),
        world_size=1,
        tp_degree=1,
        dtype=torch.bfloat16,
        height=HEIGHT,
        width=WIDTH,
        num_frames=frames,
        batch_size=1,
        compile_shapes=None,
    )


def test_wrapper_resolves_input_generator_to_the_shape_bucketed_one():
    """MRO order decides which input_generator runs; the VAE needs the bucketed one."""
    mro = ChunkedVAEWrapper.__mro__
    assert mro.index(ShapeBucketedInputGenerator) < mro.index(ModelWrapper)


@pytest.mark.parametrize(
    "frames,latent,expect_chunked",
    [(9, 3, False), (17, 5, True), (25, 7, True), (81, 21, True)],
)
def test_chunking_kicks_in_above_the_single_shot_ceiling(frames, latent, expect_chunked):
    assert latent == (frames - 1) // 4 + 1
    needs_chunking = latent > MAX_SINGLE_SHOT_LATENT_FRAMES
    assert needs_chunking is expect_chunked
    if needs_chunking:
        first, later = _vae_chunk_for(latent)
        assert first == MAX_SINGLE_SHOT_LATENT_FRAMES
        assert (latent - first) % later == 0


def test_the_default_frame_count_decomposes():
    """81 frames is 21 latent frames, which must split as first + n x later."""
    first, later = _vae_chunk_for(21)
    assert (21 - first) % later == 0
    assert first + later * ((21 - first) // later) == 21


def test_latent_count_that_does_not_decompose_is_rejected():
    """4 latent frames leaves 1 after the first chunk, which 2 does not divide."""
    with pytest.raises(ValueError, match="multiple"):
        _vae_chunk_for(4)


@pytest.mark.skipif(
    not list(__import__("glob").glob(
        "/home/ubuntu/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*/vae/config.json"
    )),
    reason="Wan VAE checkpoint not downloaded",
)
class TestAgainstTheRealCheckpoint:
    @pytest.fixture(scope="class")
    def model_dir(self):
        import glob
        from pathlib import Path

        cfg = glob.glob(
            "/home/ubuntu/.cache/huggingface/hub/"
            "models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*/vae/config.json"
        )[0]
        return Path(cfg).parent.parent

    def test_application_builds_two_chunk_graphs(self, model_dir):
        app = NeuronWanVAEDecoderChunkedApplication(
            model_path=str(model_dir / "vae"), config=_config(model_dir), chunk=CHUNK
        )
        assert [m.tag for m in app.models] == [
            "WanVAEDecoderFirstChunk",
            "WanVAEDecoderLaterChunk",
        ]
        assert app.chunk == CHUNK

    def test_declared_inputs_span_one_chunk_not_the_whole_clip(self, model_dir):
        app = NeuronWanVAEDecoderChunkedApplication(
            model_path=str(model_dir / "vae"), config=_config(model_dir), chunk=CHUNK
        )
        generated = app.models[0].input_generator()
        tensors = generated[0] if isinstance(generated[0], (list, tuple)) else generated
        latent = tensors[0]
        # (batch, z_dim, chunk latent frames, latent H, latent W)
        assert latent.shape[2] == CHUNK, (
            f"graph declares {latent.shape[2]} latent frames; chunking exists so it "
            f"declares {CHUNK}"
        )
        assert latent.shape[3] == HEIGHT // 8
        assert latent.shape[4] == WIDTH // 8

    def test_chunk_below_the_cache_settling_minimum_is_rejected(self, model_dir):
        with pytest.raises(ValueError, match="at least"):
            NeuronWanVAEDecoderChunkedApplication(
                model_path=str(model_dir / "vae"), config=_config(model_dir), chunk=2
            )
