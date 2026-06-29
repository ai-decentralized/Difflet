"""Unit-test isolation for the hardware-dispatched op backend.

Several forward-pass test modules set ``DIFFLET_BACKEND=cpu`` at *import time* so
that ``difflet.ops`` binds to the torch-native CPU backend (the only one that runs
without Trainium hardware). That mutation of ``os.environ`` happens during
collection and leaks into the shared pytest process, which then breaks
backend-sensitive tests (e.g. the registry/pipeline suites) that resolve the
backend lazily at run time and expect the real ambient default.

This conftest is imported before any test module's module-level code runs, so it
captures the true ambient ``DIFFLET_BACKEND`` and restores it at the *start* of
every test. Forward-pass tests that genuinely need the CPU backend at run time
re-assert it through their own (inner, function-scoped) autouse fixtures, which
run after this package-scoped one — so they are unaffected.
"""

import os

import pytest

_TRUE_DIFFLET_BACKEND = os.environ.get("DIFFLET_BACKEND")


@pytest.fixture(autouse=True)
def _restore_difflet_backend_default():
    if _TRUE_DIFFLET_BACKEND is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = _TRUE_DIFFLET_BACKEND
    yield
