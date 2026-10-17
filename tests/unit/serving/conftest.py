"""Pin the backend for the serving suite.

``resolve_serving_model`` selects a CUDA serving overlay by backend name (plan F3b), so the
serving tests that pin no backend would resolve Qwen-Image to the CUDA metadata — and every
other model to a "not on CUDA yet" error — on a host where ``DIFFLET_BACKEND=cuda`` is set or
auto-detected (97 failures, review finding). They describe the Trainium adapters, so they run
on the CPU reference backend as the CPU gate does. Tests that want CUDA set it themselves via
``monkeypatch.setenv`` after this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _serving_tests_pin_the_cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
