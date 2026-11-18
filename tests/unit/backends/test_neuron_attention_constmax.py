"""Experiment E1's opt-in const-max route in the neuron attention (DIFFLET_EXP_ATTN_CONSTMAX), on
CPU tensors with the kernels faked; no Neuron hardware needed."""

import math

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.ops_impl import attention as neuron_attention  # noqa: E402

BH, SQ, SK, D = 4, 24, 40, 128
SIGMA = 1.5


@pytest.fixture
def faked(monkeypatch):
    """Tensors read as on neuron, the env on, and both kernels recording their calls."""
    calls = {"constmax": [], "cte": []}

    def const_max(q, k, v, scale, softmax_max):
        calls["constmax"].append(dict(shape=tuple(q.shape), scale=scale, softmax_max=softmax_max))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    def cte(q, k, v, *, is_causal, scale, training):
        calls["cte"].append(tuple(q.shape))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    monkeypatch.setattr(neuron_attention, "_on_neuron", lambda t: True)
    monkeypatch.setattr(neuron_attention, "_flash_kernel", lambda: cte)
    monkeypatch.setattr(neuron_attention, "_CONST_MAX_OP", const_max)
    monkeypatch.setattr(neuron_attention, "_CONSTMAX", True)
    return calls


def _qkv(d=D, dtype=torch.bfloat16, s_k=SK):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(BH, SQ, d, generator=g).to(dtype), torch.randn(BH, s_k, d, generator=g).to(dtype),
            torch.randn(BH, s_k, d, generator=g).to(dtype))


def _sigma():
    return torch.full((128, 1), SIGMA)


def test_routes_unbounded_bf16_d128_calls_with_a_sigma(faked):
    q, k, v = _qkv()
    with torch.no_grad():
        out = neuron_attention.attention(q, k, v, scale=D ** -0.5, tp_q=True, tp_k=True,
                                         constmax_sigma=_sigma())
    assert faked["cte"] == [] and len(faked["constmax"]) == 1
    call = faked["constmax"][0]
    assert call["shape"] == (BH, SQ, D) and call["scale"] == D ** -0.5
    expected = 0.88 * SIGMA * math.sqrt(2 * math.log(SK))  # the docstring's rule at this Sk
    assert call["softmax_max"].shape == (128, 1) and call["softmax_max"].dtype == torch.float32
    torch.testing.assert_close(call["softmax_max"], torch.full((128, 1), expected))
    assert out.shape == (BH, SQ, D)


def test_4d_inputs_fold_heads_into_the_batch(faked):
    q, k, v = (t.reshape(2, BH // 2, *t.shape[1:]) for t in _qkv())
    with torch.no_grad():
        out = neuron_attention.attention(q, k, v, scale=0.1, tp_q=True, tp_k=True, constmax_sigma=_sigma())
    assert faked["constmax"][0]["shape"] == (BH, SQ, D) and out.shape == (2, BH // 2, SQ, D)


@pytest.mark.parametrize("case", ["no_sigma", "causal", "mask", "bounds", "fp16", "fp32", "d64", "env_off",
                                  "autograd"])
def test_every_other_call_stays_off_the_route(faked, monkeypatch, case):
    q, k, v = _qkv(s_k=SQ)
    kw = dict(scale=0.1, tp_q=True, tp_k=True, constmax_sigma=_sigma())
    if case == "no_sigma":
        kw.pop("constmax_sigma")
    elif case == "causal":
        kw["causal"] = True
    elif case == "mask":
        kw["attention_mask"] = torch.ones(BH, SQ, SQ, dtype=torch.bool)
    elif case == "bounds":
        hi = torch.full((BH, SQ, 1), SQ, dtype=torch.int32)
        kw.update(bound_min=torch.zeros_like(hi), bound_max=hi)
        monkeypatch.setattr(neuron_attention, "_FLASH_BOUNDED_OP", None)  # the masked SDPA fallback
    elif case == "fp16":
        q, k, v = _qkv(dtype=torch.float16, s_k=SQ)
    elif case == "fp32":
        q, k, v = _qkv(dtype=torch.float32, s_k=SQ)
    elif case == "d64":
        q, k, v = _qkv(d=64, s_k=SQ)
    elif case == "env_off":
        monkeypatch.setattr(neuron_attention, "_CONSTMAX", False)
    elif case == "autograd":
        q.requires_grad_(True)
    out = neuron_attention.attention(q, k, v, **kw)
    assert faked["constmax"] == [] and out.shape == q.shape
    # the shipped path: cte for the eligible calls, SDPA for the rest
    assert len(faked["cte"]) == (1 if case in ("no_sigma", "causal", "fp16", "d64", "env_off") else 0)


def test_env_unset_loads_no_op_and_keeps_the_module_constant():
    import os
    import subprocess
    import sys

    code = ("import difflet.backends.neuron.ops_impl.attention as a; "
            "print(a._CONSTMAX, a._CONST_MAX_OP)")
    env = {k: v for k, v in os.environ.items() if k != "DIFFLET_EXP_ATTN_CONSTMAX"}
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "None"]
