"""The fp32 reduction switch: bf16 sums run in fp32 and come back in bf16 when it is set; otherwise
the reduction runs on the input as is."""

from __future__ import annotations

import torch

from difflet.backends.neuron.ops_impl import collectives as c


def test_bf16_sums_run_in_fp32_when_set(monkeypatch):
    seen = []

    def reduce(t):
        seen.append(t.dtype)
        return t * 2

    x = torch.randn(8, dtype=torch.bfloat16)
    monkeypatch.setenv(c.FP32_REDUCE_ENV, "1")
    out = c._summed(x, reduce)
    assert seen == [torch.float32] and out.dtype == torch.bfloat16
    assert torch.equal(out, (x.float() * 2).to(torch.bfloat16))


def test_off_by_default_and_fp32_untouched(monkeypatch):
    seen = []

    def reduce(t):
        seen.append(t.dtype)
        return t

    monkeypatch.delenv(c.FP32_REDUCE_ENV, raising=False)
    c._summed(torch.randn(4, dtype=torch.bfloat16), reduce)
    monkeypatch.setenv(c.FP32_REDUCE_ENV, "1")
    c._summed(torch.randn(4, dtype=torch.float32), reduce)
    assert seen == [torch.bfloat16, torch.float32]
