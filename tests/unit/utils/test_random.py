"""Unit tests for difflet.utils.random.set_random_seed.

The module calls into ``neuronx_distributed`` parallel-state helpers; we
monkeypatch those so the test runs CPU-only with no Neuron hardware.
"""

import random as _stdlib_random

import numpy as np
import torch

import difflet.utils.random as random_mod


def test_set_random_seed_is_reproducible(monkeypatch):
    # Pretend model parallel is not initialized so the nxd branch is skipped.
    monkeypatch.setattr(
        random_mod.nxd.parallel_layers.parallel_state,
        "model_parallel_is_initialized",
        lambda: False,
    )

    random_mod.set_random_seed(0)
    a_py = _stdlib_random.random()
    a_np = np.random.rand()
    a_torch = torch.rand(3)

    random_mod.set_random_seed(0)
    assert _stdlib_random.random() == a_py
    assert np.random.rand() == a_np
    assert torch.equal(torch.rand(3), a_torch)


def test_set_random_seed_invokes_xla_manual_seed_when_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(
        random_mod.nxd.parallel_layers.parallel_state,
        "model_parallel_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        random_mod.nxd.parallel_layers.random,
        "model_parallel_xla_manual_seed",
        lambda seed: calls.append(seed),
    )

    random_mod.set_random_seed(123)
    assert calls == [123]


def test_set_random_seed_skips_xla_when_not_initialized(monkeypatch):
    calls = []
    monkeypatch.setattr(
        random_mod.nxd.parallel_layers.parallel_state,
        "model_parallel_is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        random_mod.nxd.parallel_layers.random,
        "model_parallel_xla_manual_seed",
        lambda seed: calls.append(seed),
    )

    random_mod.set_random_seed(7)
    assert calls == []
