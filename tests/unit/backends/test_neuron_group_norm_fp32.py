"""The fp32 GroupNorm workaround: bf16 inputs on a neuron device are normalized in fp32 and cast
back; anything else runs the original. The device is simulated (the predicate is patched)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from difflet.backends.neuron.core import workarounds


@pytest.fixture
def installed(monkeypatch):
    original = F.group_norm
    workarounds.install_group_norm_fp32()
    yield original
    workarounds.uninstall_neuron_workarounds()
    assert F.group_norm is original


def _inputs(dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 64, 32, 32, generator=g) * 3 + 1
    w, b = torch.randn(64, generator=g), torch.randn(64, generator=g)
    return x.to(dtype), w.to(dtype), b.to(dtype)


def test_bf16_on_the_device_normalizes_in_fp32(installed, monkeypatch):
    monkeypatch.setattr(workarounds, "_on_neuron", lambda t: True)
    x, w, b = _inputs(torch.bfloat16)
    out = F.group_norm(x, 32, w, b, 1e-6)
    assert out.dtype == torch.bfloat16
    expected = installed(x.float(), 32, w.float(), b.float(), 1e-6).to(torch.bfloat16)
    assert torch.equal(out, expected)
    norm = torch.nn.GroupNorm(32, 64, eps=1e-6).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(w), norm.bias.copy_(b)
        assert torch.equal(norm(x), expected)  # nn.GroupNorm goes through the wrapper


def test_other_dtypes_and_devices_run_unchanged(installed, monkeypatch):
    x, w, b = _inputs(torch.float32)
    monkeypatch.setattr(workarounds, "_on_neuron", lambda t: True)
    assert torch.equal(F.group_norm(x, 32, w, b, 1e-6), installed(x, 32, w, b, 1e-6))  # fp32 on device
    monkeypatch.setattr(workarounds, "_on_neuron", lambda t: False)
    xb, wb, bb = _inputs(torch.bfloat16)
    assert torch.equal(F.group_norm(xb, 32, wb, bb, 1e-6), installed(xb, 32, wb, bb, 1e-6))  # bf16 off device


def test_install_is_idempotent(installed):
    wrapped = F.group_norm
    workarounds.install_group_norm_fp32()
    assert F.group_norm is wrapped
