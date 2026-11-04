"""The registry suite builds Trainium pipelines/applications (DiffletPipeline,
Neuron*Application); pin the backend so a torch_xla venv (auto-detects ``tpu``)
does not divert them into the TPU gates. TPU-specific tests set ``tpu`` themselves."""

import pytest


@pytest.fixture(autouse=True)
def _registry_tests_assume_trainium(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
