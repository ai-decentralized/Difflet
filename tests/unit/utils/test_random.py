"""Unit tests for difflet.utils.random.set_random_seed.

The module seeds ``neuronx_distributed``'s model-parallel RNG only when that
package is importable and model parallelism is initialised. Tests drive that
through the ``_nxd()`` seam, so they run CPU-only with no Neuron SDK installed.
"""

import random as _stdlib_random
from types import SimpleNamespace

import numpy as np
import torch

import difflet.utils.random as random_mod


def _fake_nxd(initialized: bool, calls: list) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_layers=SimpleNamespace(
            parallel_state=SimpleNamespace(model_parallel_is_initialized=lambda: initialized),
            random=SimpleNamespace(model_parallel_xla_manual_seed=lambda seed: calls.append(seed)),
        )
    )


def test_set_random_seed_is_reproducible(monkeypatch):
    monkeypatch.setattr(random_mod, "_nxd", lambda: None)

    random_mod.set_random_seed(0)
    a_py = _stdlib_random.random()
    a_np = np.random.rand()
    a_torch = torch.rand(3)

    random_mod.set_random_seed(0)
    assert _stdlib_random.random() == a_py
    assert np.random.rand() == a_np
    assert torch.equal(torch.rand(3), a_torch)


def test_set_random_seed_without_neuron_sdk_does_not_touch_nxd(monkeypatch):
    # No neuronx_distributed importable: seeding must still work and import nothing.
    monkeypatch.setattr(random_mod, "_nxd", lambda: None)
    random_mod.set_random_seed(5)


def test_set_random_seed_invokes_xla_manual_seed_when_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(random_mod, "_nxd", lambda: _fake_nxd(True, calls))

    random_mod.set_random_seed(123)
    assert calls == [123]


def test_set_random_seed_skips_xla_when_not_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(random_mod, "_nxd", lambda: _fake_nxd(False, calls))

    random_mod.set_random_seed(7)
    assert calls == []


def test_nxd_seam_returns_none_when_package_absent(monkeypatch):
    monkeypatch.setattr(random_mod.importlib.util, "find_spec", lambda name: None)
    assert random_mod._nxd() is None
