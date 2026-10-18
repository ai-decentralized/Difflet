"""The composed Flux application (CPU, eager, TP1) against diffusers' FluxPipeline on the same tiny
components: same prompt embeddings, same seed; final latents and the decoded image."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from difflet.models.flux.neuron_application import TorchNeuronFluxApplication  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_flux_toy import (  # noqa: E402
    PIPE_HW,
    diffusers_pipeline_reference,
    pipeline_kwargs,
    write_tiny_pipeline,
)


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return write_tiny_pipeline(tmp_path_factory.mktemp("fluxpipe"))


@pytest.fixture
def app(root):
    a = TorchNeuronFluxApplication(model_path=root, parallel=DiffletParallelConfig(tp_degree=1),
                                   dtype=torch.float32, shape={"height": PIPE_HW, "width": PIPE_HW},
                                   device="cpu", exec_mode="eager", text_seq_len=8)
    a.load()
    return a


@pytest.mark.parametrize("output_type", ["latent", "pt"])
def test_matches_diffusers_flux_pipeline(app, root, output_type):
    ours = app(**pipeline_kwargs(output_type)).images
    ref = diffusers_pipeline_reference(root, output_type)
    torch.testing.assert_close(torch.as_tensor(ours), torch.as_tensor(ref), atol=1e-4, rtol=1e-4)


def test_unknown_options_are_rejected(root):
    with pytest.raises(TypeError, match="unexpected keyword arguments: bogus"):
        TorchNeuronFluxApplication(model_path=root, parallel=DiffletParallelConfig(), shape={"height": 16, "width": 16},
                                   bogus=1)


def test_without_the_vae_the_application_returns_diffusers_latents(root):
    a = TorchNeuronFluxApplication(model_path=root, parallel=DiffletParallelConfig(tp_degree=1),
                                   dtype=torch.float32, shape={"height": PIPE_HW, "width": PIPE_HW},
                                   device="cpu", exec_mode="eager", text_seq_len=8, decode_vae=False)
    a.load()
    assert a.vae is None
    ours = a(**pipeline_kwargs("pt")).images  # forced to latents without a VAE
    torch.testing.assert_close(ours, diffusers_pipeline_reference(root, "latent"), atol=1e-4, rtol=1e-4)
