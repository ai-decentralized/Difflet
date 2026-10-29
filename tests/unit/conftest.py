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


@pytest.fixture(autouse=True)
def _no_hardware_probe(monkeypatch):
    """Keep unit tests off the host's real Neuron topology.

    ``difflet.planner.hardware`` shells out to ``neuron-ls`` so the CLI can
    enforce a core budget it actually knows. Left live, that would make unit
    tests pass or fail depending on the box they run on -- the same core-budget
    assertion would hold on a laptop and trip on a trn2. Stub the probe so every
    unit test sees the undetected/fallback path, and let the planner suite
    exercise real parses by injecting ``neuron-ls`` payloads directly.
    """

    from difflet.planner import hardware

    hardware._probe_neuron_ls_cached.cache_clear()
    monkeypatch.setattr(hardware, "_run_neuron_ls", lambda: [])
    yield
    hardware._probe_neuron_ls_cached.cache_clear()


@pytest.fixture(autouse=True)
def _neuron_cc_flags_restored(monkeypatch):
    """A Wan application built for the neuron device appends the ccop overlap flag to this
    process's NEURON_CC_FLAGS while it lives (``difflet/backends/neuron/wan/compile_options.py``);
    restore the variable after every test, so no later test's device compile gets the flag, and
    drop the test's holds first, so an application of this test collected during a later one
    leaves that test's NEURON_CC_FLAGS alone."""
    current = os.environ.get("NEURON_CC_FLAGS")
    monkeypatch.setenv("NEURON_CC_FLAGS", current or "")
    if current is None:
        monkeypatch.delenv("NEURON_CC_FLAGS")
    yield
    from difflet.backends.neuron.wan import compile_options

    compile_options._forget_holders()


@pytest.fixture(autouse=True)
def _neuron_weight_store_off(monkeypatch):
    """The neuron backend's weight store is on by default and lives under the compile cache
    (``~/.cache/difflet``); unit tests must not write there. Tests of the store point
    DIFFLET_NEURON_WEIGHT_STORE at a temp dir themselves (gloo workers inherit it)."""
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", "0")
