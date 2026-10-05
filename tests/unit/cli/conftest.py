"""CLI test isolation: the orchestrators export Neuron runtime variables into the process
environment by design (LTX-2's compile / load set NEURON_RT_VIRTUAL_CORE_SIZE=2, FLUX's backbone
does the same), so every test here runs against a snapshot of ``os.environ`` that is restored
afterwards; otherwise one orchestrator test leaks into the runner tests that assert absence."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_environ():
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
