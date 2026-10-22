"""Pin the backend for the serving suite.

``resolve_serving_model`` picks the GPU serving metadata on a GPU platform, so on a host where
``DIFFLET_BACKEND=cuda`` is set or auto-detected the tests that pin no backend would describe
the GPU adapters instead of the Trainium ones. They run on the CPU reference backend, as the CPU
gate does; tests that want CUDA set it with ``monkeypatch.setenv`` after this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _serving_tests_pin_the_cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
