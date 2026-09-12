"""Exercise Trainium's SDPA routing on real CPU tensors without Neuron hardware."""

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from difflet.ops.attention_config import attention_implementation


@pytest.fixture
def op(monkeypatch):
    kernel = ModuleType("nkilib.core.attention.attention_cte")
    kernel.attention_cte = Mock(side_effect=AssertionError("unexpected CTE call"))
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    path = Path(__file__).parents[3] / "difflet/backends/trainium/ops_impl/attention.py"
    spec = importlib.util.spec_from_file_location("_attention_impl_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", list(itertools.product([False, True], repeat=3)))
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive", "bounds"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.37])
def test_sdpa_matches_dense_reference(op, layout, mask_kind, causal, scale):
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(2, 5, 4, generator=generator)
    k, v = [torch.randn(2, 7, 4, generator=generator) for _ in range(2)]
    mask = None
    kwargs = {}
    if mask_kind != "none":
        mask = torch.arange(7).view(1, 1, -1) < torch.tensor([3, 6]).view(2, 1, 1)
        if mask_kind == "bounds":
            kwargs = dict(bound_min=torch.zeros(2, 5, 1, dtype=torch.int32),
                          bound_max=torch.tensor([3, 6], dtype=torch.int32)
                          .view(2, 1, 1).expand(2, 5, 1))
        else:
            if mask_kind == "additive":
                mask = torch.where(mask, 0.0, -3.0)
            kwargs["attention_mask"] = mask
    scores = q @ k.transpose(-1, -2) * (1.0 if scale is None else scale)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    if causal:
        scores = scores.masked_fill(~torch.ones(5, 7, dtype=torch.bool).tril(), -torch.inf)
    expected = scores.softmax(-1) @ v
    tp_q, tp_k, tp_out = layout
    with attention_implementation("sdpa"):
        actual = op.attention(
            q if tp_q else q.transpose(-1, -2),
            k if tp_k else k.transpose(-1, -2), v,
            scale=scale, causal=causal, tp_q=tp_q, tp_k=tp_k, tp_out=tp_out, **kwargs,
        )
    if tp_out:
        actual = actual.transpose(-1, -2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_megakernel_calls_cte(op):
    op.attention_cte = Mock(return_value="cte output")
    with attention_implementation("megakernel"):
        assert op.attention(None, None, None) == "cte output"
    op.attention_cte.assert_called_once()


def test_traced_sdpa_keeps_bounds_dynamic(op):
    class Attention(torch.nn.Module):
        def forward(self, q, k, v, lower, upper):
            return op.attention(q, k, v, scale=0.5, tp_q=True, tp_k=True,
                                bound_min=lower, bound_max=upper)

    q, k, v = [torch.randn(2, 5, 4) for _ in range(3)]
    lower = torch.zeros(2, 5, 1, dtype=torch.int32)
    upper = torch.full_like(lower, 5)
    with attention_implementation("sdpa"):
        traced = torch.jit.trace(Attention(), (q, k, v, lower, upper))
    assert "aten::scaled_dot_product_attention" in str(traced.graph)
    upper = torch.full_like(upper, 2)
    actual = traced(q, k, v, lower, upper)
    expected = (q @ k[:, :2].transpose(-1, -2) * 0.5).softmax(-1) @ v[:, :2]
    torch.testing.assert_close(actual, expected)


def test_ulysses_dense_step_honors_sdpa(op):
    q, k, v = [torch.randn(1, 2, 5, 4) for _ in range(3)]
    with attention_implementation("sdpa"):
        actual = op._dense_attention(q, k, v, scale=0.5, causal=False)
    torch.testing.assert_close(actual, (q @ k.transpose(-1, -2) * 0.5).softmax(-1) @ v)


def test_sdpa_rejects_ring_and_kernel_only_options(op):
    q = torch.randn(2, 5, 4)
    with attention_implementation("sdpa"):
        for call in [lambda: op.ring_attention(None, None, None, scale=1),
                     lambda: op.joint_ring_attention(None, None, None, None, None, scale=1)]:
            with pytest.raises(NotImplementedError, match="ring"):
                call()
        with pytest.raises(NotImplementedError, match="cache_softmax"):
            op.attention(q, q, q, tp_q=True, tp_k=True, cache_softmax=True)


def test_sdpa_rejects_ambiguous_bounds(op):
    q = torch.randn(2, 5, 4)
    bounds = torch.zeros(2, 5, 1, dtype=torch.int32)
    with attention_implementation("sdpa"):
        with pytest.raises(ValueError, match="both"):
            op.attention(q, q, q, bound_min=bounds)
        with pytest.raises(ValueError, match="mutually exclusive"):
            op.attention(q, q, q, bound_min=bounds, bound_max=bounds,
                         attention_mask=torch.ones(5, 5, dtype=torch.bool))
