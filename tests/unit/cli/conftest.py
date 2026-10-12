"""CLI test isolation: the orchestrators export Neuron runtime variables into the process
environment by design (LTX-2's compile / load set NEURON_RT_VIRTUAL_CORE_SIZE=2, FLUX's backbone
does the same), so every test here restores ``os.environ`` afterwards; otherwise one
orchestrator test leaks into the runner tests that assert absence.

Only the variables that changed are restored, and Neuron prewarm is off: the prewarm thread
some stage code starts would touch the device from a unit test, and clearing the whole
environment while it runs crashes the interpreter (getenv racing unsetenv)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_environ(monkeypatch):
    monkeypatch.setenv("DIFFLET_DISABLE_PREWARM", "1")
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        for name in [k for k in os.environ if k not in snapshot]:
            del os.environ[name]
        for name, value in snapshot.items():
            if os.environ.get(name) != value:
                os.environ[name] = value
