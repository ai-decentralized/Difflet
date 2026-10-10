"""Unit tests for the neuron backend's attention contract, run on CPU tensors; no Neuron hardware needed."""

import math

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.cpu.ops_impl import attention as cpu_attention  # noqa: E402
from difflet.backends.neuron.ops_impl import attention as neuron_attention  # noqa: E402

BH, SQ, SK, D = 6, 33, 47, 16


def _reference(q, k, v, *, scale=None, causal=False, mask=None, tp_q=False, tp_k=False, tp_out=False):
    """Independent fp64 implementation of the attention_cte calling contract."""
    q = (q if tp_q else q.transpose(-1, -2)).double()
    k = (k if tp_k else k.transpose(-1, -2)).double()
    scores = q @ k.transpose(-1, -2) * (1.0 if scale is None else scale)
    if causal:
        tril = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        scores = scores.masked_fill(~tril, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.double()
    out = torch.softmax(scores, dim=-1) @ v.double()
    return out.transpose(-1, -2) if tp_out else out


def _inputs(tp_q=True, tp_k=True, s_k=SK):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(BH, SQ, D, generator=g)
    k = torch.randn(BH, s_k, D, generator=g)
    v = torch.randn(BH, s_k, D, generator=g)
    return (q if tp_q else q.transpose(-1, -2).contiguous(),
            k if tp_k else k.transpose(-1, -2).contiguous(), v)


def _close(out, ref):
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp_q", [True, False])
@pytest.mark.parametrize("tp_k", [True, False])
@pytest.mark.parametrize("tp_out", [True, False])
def test_layout_flags(tp_q, tp_k, tp_out):
    q, k, v = _inputs(tp_q, tp_k)
    flags = dict(tp_q=tp_q, tp_k=tp_k, tp_out=tp_out)
    _close(neuron_attention.attention(q, k, v, scale=0.3, **flags), _reference(q, k, v, scale=0.3, **flags))


@pytest.mark.parametrize("scale", [None, 1.0 / math.sqrt(D), 0.05])
def test_scale_none_means_one(scale):
    q, k, v = _inputs()
    out = neuron_attention.attention(q, k, v, scale=scale, tp_q=True, tp_k=True)
    _close(out, _reference(q, k, v, scale=scale, tp_q=True, tp_k=True))


def test_matches_cpu_backend_in_the_standard_layout():
    q, k, v = _inputs()
    flags = dict(scale=1.0 / math.sqrt(D), tp_q=True, tp_k=True, tp_out=False)
    torch.testing.assert_close(
        neuron_attention.attention(q, k, v, **flags), cpu_attention.attention(q, k, v, **flags),
        rtol=1e-5, atol=1e-5,
    )


def test_causal():
    q, k, v = _inputs(s_k=SQ)
    out = neuron_attention.attention(q, k, v, scale=0.2, causal=True, tp_q=True, tp_k=True)
    _close(out, _reference(q, k, v, scale=0.2, causal=True, tp_q=True, tp_k=True))


def test_bool_and_float_masks():
    q, k, v = _inputs()
    keep = torch.rand(BH, SQ, SK, generator=torch.Generator().manual_seed(1)) > 0.3
    keep[..., 0] = True  # every row keeps at least one key
    bias = torch.randn(BH, SQ, SK)
    for mask in (keep, bias):
        out = neuron_attention.attention(q, k, v, scale=0.2, attention_mask=mask, tp_q=True, tp_k=True)
        _close(out, _reference(q, k, v, scale=0.2, mask=mask, tp_q=True, tp_k=True))


def test_causal_with_mask():
    q, k, v = _inputs(s_k=SQ)
    keep = torch.ones(BH, SQ, SQ, dtype=torch.bool)
    keep[..., 1] = False
    keep[..., 0] = True
    out = neuron_attention.attention(q, k, v, scale=0.2, causal=True, attention_mask=keep, tp_q=True, tp_k=True)
    _close(out, _reference(q, k, v, scale=0.2, causal=True, mask=keep, tp_q=True, tp_k=True))


def test_bounds():
    q, k, v = _inputs()
    bound_min = torch.zeros(BH, SQ, 1, dtype=torch.int32)
    bound_max = torch.randint(1, SK + 1, (BH, SQ, 1), dtype=torch.int32)
    out = neuron_attention.attention(
        q, k, v, scale=0.2, bound_min=bound_min, bound_max=bound_max, tp_q=True, tp_k=True
    )
    keep = torch.arange(SK).view(1, 1, -1) < bound_max
    _close(out, _reference(q, k, v, scale=0.2, mask=keep, tp_q=True, tp_k=True))


def test_bounds_contract_errors():
    q, k, v = _inputs()
    bound = torch.zeros(BH, SQ, 1, dtype=torch.int32)
    with pytest.raises(ValueError, match="both be provided"):
        neuron_attention.attention(q, k, v, bound_min=bound, tp_q=True, tp_k=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        neuron_attention.attention(
            q, k, v, bound_min=bound, bound_max=bound + 1,
            attention_mask=torch.ones(BH, SQ, SK, dtype=torch.bool), tp_q=True, tp_k=True,
        )


def test_cross_attention_uses_the_transposed_layout():
    q, k, v = _inputs(tp_q=False, tp_k=False)
    _close(neuron_attention.cross_attention(q, k, v, scale=0.2), _reference(q, k, v, scale=0.2))


def test_unknown_kwargs_raise():
    q, k, v = _inputs()
    with pytest.raises(NotImplementedError, match="attention_cte-only kwargs: sliding_window"):
        neuron_attention.attention(q, k, v, tp_q=True, tp_k=True, sliding_window=4)


@pytest.mark.parametrize(
    "name", ["ring_attention", "joint_ring_attention", "ulysses_attention", "joint_ulysses_attention"]
)
def test_context_parallel_entry_points_raise(name):
    with pytest.raises(NotImplementedError, match="context parallelism"):
        getattr(neuron_attention, name)(None, None, None, scale=1.0)


# ---- routing between the NKI flash kernel and SDPA ----------------------------------


@pytest.fixture
def fake_kernel(monkeypatch):
    """Pretend tensors are on neuron and record kernel calls; the fake computes exact attention."""
    calls = []

    def kernel(q, k, v, *, is_causal, scale, training):
        calls.append(dict(shape=tuple(q.shape), is_causal=is_causal, scale=scale, training=training))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    monkeypatch.setattr(neuron_attention, "_on_neuron", lambda t: True)
    monkeypatch.setattr(neuron_attention, "_flash_kernel", lambda: kernel)
    return calls


def _bf16(*tensors):
    return [t.to(torch.bfloat16) for t in tensors]


def _close_bf16(out, ref):
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)


def test_unmasked_attention_uses_the_kernel_at_non_512_lengths(fake_kernel):
    q, k, v = _bf16(*_inputs())  # SQ=33, SK=47: neither a multiple of 512
    with torch.no_grad():
        out = neuron_attention.attention(q, k, v, scale=0.2, tp_q=True, tp_k=True)
    assert fake_kernel == [dict(shape=(1, BH, SQ, D), is_causal=False, scale=0.2, training=False)]
    _close_bf16(out, _reference(q, k, v, scale=0.2, tp_q=True, tp_k=True))


def test_kernel_keeps_scale_none_as_one_and_4d_inputs(fake_kernel):
    q, k, v = (t.reshape(2, BH // 2, *t.shape[1:]) for t in _bf16(*_inputs()))
    with torch.no_grad():
        out = neuron_attention.attention(q, k, v, causal=False, tp_q=True, tp_k=True)
    assert fake_kernel[0]["shape"] == (2, BH // 2, SQ, D) and fake_kernel[0]["scale"] == 1.0
    _close_bf16(out, _reference(q, k, v, scale=1.0, tp_q=True, tp_k=True))


def test_kernel_handles_the_transposed_layout(fake_kernel):
    q, k, v = _bf16(*_inputs(tp_q=False, tp_k=False))
    with torch.no_grad():
        out = neuron_attention.cross_attention(q, k, v, scale=0.2)
    assert len(fake_kernel) == 1
    _close_bf16(out, _reference(q, k, v, scale=0.2))


@pytest.mark.parametrize("case", ["fp32", "mask", "head_dim", "batch_heads", "kv_shape", "autograd"])
def test_falls_back_to_sdpa_outside_the_kernel_limits(fake_kernel, case):
    q, k, v = _bf16(*_inputs())
    kwargs = {}
    if case == "fp32":
        q, k, v = _inputs()
    elif case == "mask":
        kwargs["attention_mask"] = torch.ones(BH, SQ, SK, dtype=torch.bool)
    elif case == "head_dim":
        q, k, v = _bf16(*(torch.randn(2, 8, 160) for _ in range(3)))
    elif case == "batch_heads":
        q, k, v = _bf16(*(torch.randn(513, 8, 16) for _ in range(3)))
    elif case == "kv_shape":
        v = torch.randn(BH, SK, D + 8).to(torch.bfloat16)
    elif case == "autograd":
        q.requires_grad_(True)
    out = neuron_attention.attention(q, k, v, scale=0.2, tp_q=True, tp_k=True, **kwargs)
    assert fake_kernel == []
    if case == "fp32":
        _close(out, _reference(q, k, v, scale=0.2, tp_q=True, tp_k=True))
    elif case in ("mask", "autograd"):
        _close_bf16(out.detach(), _reference(q.detach(), k, v, scale=0.2, tp_q=True, tp_k=True))
