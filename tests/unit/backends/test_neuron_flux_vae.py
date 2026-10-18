"""Flux's VAE decoder on the neuron backend (CPU, eager) against diffusers' AutoencoderKL.decode."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.flux.vae import NeuronFluxVaeDecoderApplication  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from tests.unit.backends._neuron_flux_toy import VAE_HW, vae_latents, vae_reference, write_tiny_vae  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


def test_decode_matches_diffusers(tmp_path):
    path = write_tiny_vae(tmp_path)
    app = NeuronFluxVaeDecoderApplication(model_path=path, height=VAE_HW, width=VAE_HW, device="cpu")
    app.load()
    assert app.load_report["missing"] == []
    assert app.latent_shape == tuple(vae_latents().shape)
    torch.testing.assert_close(app.decode(vae_latents()), vae_reference(path), atol=1e-5, rtol=1e-5)


def test_compiled_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="eager only"):
        NeuronFluxVaeDecoderApplication(model_path=write_tiny_vae(tmp_path), exec_mode="compile",
                                        device="cpu")
