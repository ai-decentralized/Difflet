"""The guard for torch-neuronx's boolean bitwise ops on non-contiguous operands (CPU, with the
device check faked): bool operands on a neuron device reach the kernel contiguous; anything else
reaches it unchanged; installing twice wraps once."""

from __future__ import annotations

import pytest
import torch

from difflet.backends.neuron.core import workarounds as W


@pytest.fixture
def spy(monkeypatch):
    seen = []
    real_and = torch.Tensor.__and__

    def spy_and(self, other):
        seen.append((self.is_contiguous(), other.is_contiguous()))
        return real_and(self, other)

    W.uninstall_bool_bitwise_guard()
    monkeypatch.setattr(torch.Tensor, "__and__", spy_and)
    monkeypatch.setattr(W, "_on_neuron", lambda t: isinstance(t, torch.Tensor))  # CPU stands in
    W.install_bool_bitwise_guard()
    yield seen
    W.uninstall_bool_bitwise_guard()


def _mask(dtype):
    m = torch.zeros(1, 256, dtype=dtype)
    m[:, :50] = 1
    return m.view(1, 1, 1, 256).repeat(1, 1, 256, 1)


def test_bool_strided_operand_reaches_the_kernel_contiguous(spy):
    m = _mask(torch.bool)
    out = m & m.transpose(2, 3)
    assert spy == [(True, True)]
    assert int(out.sum()) == 2500


def test_other_dtypes_are_untouched(spy):
    m = _mask(torch.int64)
    m & m.transpose(2, 3)
    assert spy == [(True, False)]


def test_install_is_idempotent(spy):
    W.install_bool_bitwise_guard()
    m = _mask(torch.bool)
    m & m.transpose(2, 3)
    assert len(spy) == 1


def test_function_form_is_guarded(spy, monkeypatch):
    m = _mask(torch.bool)
    torch.testing.assert_close(torch.bitwise_and(m, m.transpose(2, 3)), m & m.transpose(2, 3).contiguous())


def test_layer_norm_runs_in_fp32_for_bf16_neuron_inputs(monkeypatch):
    import torch.nn.functional as F

    W.uninstall_neuron_workarounds()
    seen = []
    real = F.layer_norm

    def spy(x, shape, weight=None, bias=None, eps=1e-5):
        seen.append(x.dtype)
        return real(x, shape, weight, bias, eps)

    monkeypatch.setattr(F, "layer_norm", spy)
    monkeypatch.setattr(W, "_on_neuron", lambda t: isinstance(t, torch.Tensor))  # CPU stands in
    W.install_layer_norm_fp32()
    try:
        ln = torch.nn.LayerNorm(16).to(torch.bfloat16)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        out = ln(x)
        assert seen == [torch.float32] and out.dtype == torch.bfloat16
        F.layer_norm(torch.randn(4, 16), (16,))  # fp32 input: unchanged path
        assert seen[-1] == torch.float32 and len(seen) == 2
    finally:
        W.uninstall_neuron_workarounds()
