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
from pathlib import Path

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
def _neuron_weight_store_off(monkeypatch):
    """The neuron backend's weight store is on by default and lives under the compile cache
    (``~/.cache/difflet``); unit tests must not write there. Tests of the store point
    DIFFLET_NEURON_WEIGHT_STORE at a temp dir themselves (gloo workers inherit it)."""
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", "0")


def _hub_cache() -> Path:
    """The Hugging Face hub cache, read when called: a test that points
    ``huggingface_hub.constants.HF_HUB_CACHE`` at a fake hub moves the guards below with it."""
    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover
        return Path("~/.cache/huggingface/hub").expanduser().resolve()
    return Path(constants.HF_HUB_CACHE).expanduser().resolve()


@pytest.fixture(autouse=True)
def _no_bf16_copy_of_a_real_snapshot(monkeypatch):
    """``ensure_bf16_copy`` (the drivers' compile step for fp32 DiTs: Wan, HunyuanVideo-1.5)
    is refused for anything under the Hugging Face hub cache: a driver test that reaches it
    with the real weights local wrote a 27 GB copy per call under tmp_path and filled the host
    (2026-11-07). Toys under tmp_path still copy; a test that wants the real thing is not a
    unit test."""
    from difflet.backends.neuron.core import weights_bf16

    hub = _hub_cache()
    real = weights_bf16.ensure_bf16_copy

    def guarded(component_dir, cache_root=None):
        if Path(component_dir).resolve().is_relative_to(hub):
            raise AssertionError(
                f"a unit test asked for a bf16 copy of a real snapshot: {component_dir}")
        return real(component_dir, cache_root)

    monkeypatch.setattr(weights_bf16, "ensure_bf16_copy", guarded)


@pytest.fixture(autouse=True)
def _no_quantized_dir_of_a_real_snapshot(monkeypatch):
    """``write_quantized_dir`` (every driver's compile with --quant) is refused for a source
    under the Hugging Face hub cache, as ``ensure_bf16_copy`` is: HunyuanVideo, FLUX, LTX-2
    and Qwen-Image are quantized straight from the hub snapshot (about 19 GB for
    HunyuanVideo), so a driver test whose resolver patch missed would write one per call. The
    hub is read per call, so a test can point it at a fake one. A test module that imports the
    writer by name keeps the unguarded function for its own toys; the drivers reach it only
    through ``ensure_quantized_dir``, which looks up the module global."""
    from difflet.quant import checkpoint

    real = checkpoint.write_quantized_dir

    def guarded(model_dir, out_dir, spec, **kwargs):
        if Path(model_dir).resolve().is_relative_to(_hub_cache()):
            raise AssertionError(
                f"a unit test asked for a quantized directory of a real snapshot: {model_dir}")
        return real(model_dir, out_dir, spec, **kwargs)

    monkeypatch.setattr(checkpoint, "write_quantized_dir", guarded)
