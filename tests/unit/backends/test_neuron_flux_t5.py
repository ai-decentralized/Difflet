"""Flux's T5 encoder on the neuron backend (CPU, eager) against transformers' T5EncoderModel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from difflet.backends.neuron.flux.text_encoder import NeuronFluxT5Application  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from tests.unit.backends._neuron_flux_toy import T5_SEQ, t5_inputs, t5_reference, write_tiny_t5  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny_t5(tmp_path_factory):
    return write_tiny_t5(tmp_path_factory.mktemp("t5"))


def test_tp1_matches_transformers(tiny_t5):
    app = NeuronFluxT5Application(model_path=tiny_t5, text_seq_len=T5_SEQ, dtype=torch.float32,
                                  exec_mode="eager", device="cpu")
    app.load()
    assert app.load_report == {"missing": [], "unexpected": []}
    torch.testing.assert_close(app.encode(*t5_inputs()), t5_reference(tiny_t5), atol=1e-5, rtol=1e-5)


def test_tp4_on_four_ranks_matches_transformers(tiny_t5):
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import flux_t5_worker

    reference = t5_reference(tiny_t5)
    for r in run_ranks(flux_t5_worker, tiny_t5, world_size=4):
        assert r["report"] == {"missing": [], "unexpected": []}
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-5, rtol=1e-5)
