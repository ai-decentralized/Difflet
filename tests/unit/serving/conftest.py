"""The serving suite describes Trainium profiles (FLUX, LTX-2, Neuron
placements). ``resolve_serving_model`` gates on the current backend
(d384b5c) and a venv with torch_xla auto-detects ``tpu``, which would reject
those profiles before the behaviour under test. Pin the backend here; the
TPU-specific tests set ``DIFFLET_BACKEND=tpu`` or patch ``_backend_is_tpu``
themselves, after this fixture."""

import pytest


@pytest.fixture(autouse=True)
def _serving_tests_assume_trainium(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
