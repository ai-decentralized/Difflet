"""The CPU reference of MXFP8 attention: exact attention with MX off, bounds and the 512-key
padding leave no trace in P's MX blocks, and the perms decide the block membership. The perms are
``swizzle_perm`` until the kernel exports its own (ruling BR1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from difflet.quant.mx import swizzle_perm
from difflet.quant.mx_attention import mx_attention_reference

BH, S, D = 2, 1024, 128
SCALE = D ** -0.5
MX_ON = dict(qk_mx=True, pv_mx=True, qk_perm=swizzle_perm, pv_perm=swizzle_perm)
MX_OFF = dict(qk_mx=False, pv_mx=False)


def _qkv(dtype=torch.float32, s_k=S):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(BH, n, D, generator=g) for n in (S, s_k, s_k))
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _bounds(lo, hi, s_q=S):
    return (torch.full((BH, s_q, 1), lo, dtype=torch.int32),
            torch.full((BH, s_q, 1), hi, dtype=torch.int32))


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("section_len", [None, 8192, 512])
@pytest.mark.parametrize("s_k", [S, 1300])
def test_mx_off_equals_sdpa_in_fp32(s_k, section_len, bounded):
    """512-key sections put real boundaries (and, bounded, whole masked sections) in play."""
    q, k, v = _qkv(s_k=s_k)
    kw, mask = dict(scale=SCALE), None
    if bounded:
        g = torch.Generator().manual_seed(1)
        lo = torch.randint(0, s_k - 1, (BH, S, 1), generator=g, dtype=torch.int32)
        hi = lo + 1 + (torch.rand(lo.shape, generator=g) * (s_k - lo)).int()
        j = torch.arange(s_k)
        kw, mask = dict(scale=1.0, bound_min=lo, bound_max=hi), (j >= lo) & (j < hi)
    q_in = q * SCALE if bounded else q  # the bounded path takes scale 1.0
    out = mx_attention_reference(q_in, k, v, section_len=section_len, **kw, **MX_OFF)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALE)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_k_smoothing_alone_changes_nothing():
    q, k, v = _qkv()
    k = k + 3.0 * torch.randn(1, 1, D, generator=torch.Generator().manual_seed(2))
    base = mx_attention_reference(q, k, v, scale=SCALE, **MX_OFF)
    smoothed = mx_attention_reference(q, k - k.mean(dim=1, keepdim=True), v, scale=SCALE, **MX_OFF)
    torch.testing.assert_close(smoothed, base, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("section_len", [None, 512])
def test_perturbing_out_of_bounds_keys_changes_nothing(section_len):
    """Per row group: its out-of-range keys scaled up 100x. With 512-key sections, rows bounded to
    [600, 1024) have a wholly masked first section."""
    q, k, v = _qkv(torch.bfloat16)
    groups = [(0, 300), (300, 1000), (600, 1024), (37, 811)]
    rows = S // len(groups)
    lo, hi = _bounds(0, 0)
    for i, (a, b) in enumerate(groups):
        lo[:, i * rows:(i + 1) * rows], hi[:, i * rows:(i + 1) * rows] = a, b
    kw = dict(scale=1.0, bound_min=lo, bound_max=hi, section_len=section_len, **MX_ON)
    base = mx_attention_reference(q * SCALE, k, v, **kw)
    j = torch.arange(S)
    for i, (a, b) in enumerate(groups):
        k2 = k.clone()
        out_of_range = (j < a) | (j >= b)
        k2[:, out_of_range] = (k2[:, out_of_range].float() * 100).to(k.dtype)
        out = mx_attention_reference(q * SCALE, k2, v, **kw)
        sl = slice(i * rows, (i + 1) * rows)
        assert torch.equal(out[:, sl], base[:, sl]), (a, b)


def test_mx_block_straddling_the_bound_equals_the_truncated_sequence():
    hi = 300
    blocks = swizzle_perm(512).view(-1, 32)
    assert ((blocks < hi).any(1) & (blocks >= hi).any(1)).any(), "the bound must split a block"
    q, k, v = _qkv()
    k[:, hi:] *= 100  # unmasked, they would own the row max and P's block scales
    # V is quantized once for every query whatever its bounds, so its out-of-range keys share
    # block scales with in-range ones; zero here, they match the truncated call's zero padding.
    v[:, hi:] = 0
    lo_t, hi_t = _bounds(0, hi)
    q = q * SCALE  # the bounded path takes scale 1.0
    bounded = mx_attention_reference(q, k, v, scale=1.0, bound_min=lo_t, bound_max=hi_t, **MX_ON)
    truncated = mx_attention_reference(q, k[:, :hi], v[:, :hi], scale=1.0, **MX_ON)
    torch.testing.assert_close(bounded, truncated, atol=1e-6, rtol=1e-6)


def test_unbounded_key_length_off_512_ignores_the_padding():
    """Keys 700..1023 are padding: zero keys score 0, not -inf, so they must be kept out of P (its
    last tile's tail zeroed) whatever they hold; V's padding is zero."""
    n = 700
    q, k, v = _qkv()
    q = q * SCALE  # the bounded path takes scale 1.0
    unbounded = mx_attention_reference(q, k[:, :n], v[:, :n], scale=1.0, **MX_ON)
    lo, hi = _bounds(0, n)
    v_pad = torch.cat([v[:, :n], torch.zeros_like(v[:, n:])], dim=1)
    for k_pad in (torch.zeros_like(k[:, n:]), k[:, n:] * 100):
        out = mx_attention_reference(q, torch.cat([k[:, :n], k_pad], dim=1), v_pad, scale=1.0,
                                     bound_min=lo, bound_max=hi, **MX_ON)
        torch.testing.assert_close(out, unbounded, atol=1e-6, rtol=1e-6)


def test_a_non_default_perm_changes_the_scales():
    q, k, v = _qkv(torch.bfloat16)
    base = mx_attention_reference(q, k, v, scale=SCALE, **MX_ON)
    for which in ("qk_perm", "pv_perm"):
        out = mx_attention_reference(q, k, v, scale=SCALE, **{**MX_ON, which: torch.arange})
        assert not torch.equal(out, base), which


@pytest.mark.parametrize("qk_mx,pv_mx", [(False, True), (True, False), (True, True)])
def test_mx_output_stays_close_to_fp32(qk_mx, pv_mx):
    q, k, v = _qkv(torch.bfloat16)
    out = mx_attention_reference(q, k, v, scale=SCALE, qk_mx=qk_mx, pv_mx=pv_mx,
                                 qk_perm=swizzle_perm, pv_perm=swizzle_perm)
    fp32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=SCALE)
    assert out.dtype == torch.bfloat16 and out.shape == (BH, S, D)
    assert F.cosine_similarity(out.float().flatten(), fp32.flatten(), dim=0) > 0.99


def test_pv_mx_divides_by_the_unquantized_row_sum():
    """With v = 1 (exact in MX) the output is sum(P_mx) / sum(exp): not 1, since exponential's
    fused fp32 accumulator never sees P's bf16 or MX rounding."""
    q, k, _ = _qkv()
    out = mx_attention_reference(q, k, torch.ones(BH, S, D), scale=SCALE, qk_mx=False,
                                 pv_mx=True, pv_perm=swizzle_perm)
    assert (out - 1).abs().max() > 1e-3


def test_p_is_rounded_to_bf16_before_its_mx_quantization():
    """Key 0 scores 0 (E = 1), every other key s with E = exp(s) just below 0.59375, which bf16
    rounds onto the e4m3 midpoint 152/256 (blocks without key 0: 1.1875 x 2^7 after the 2^-8
    scale; key 0's: 76/128), so ties-to-even takes it up to 0.625; quantized straight from fp32
    it goes down to 0.5625. v = 1: the output is sum(P) / sum(E), P = 1 for key 0."""
    s = torch.log(torch.tensor(0.5925))
    e = torch.exp(s)
    assert e.bfloat16().item() == 0.59375 and e.item() != 0.59375
    q = torch.zeros(1, 4, D)
    q[..., 0] = 1
    k = torch.zeros(1, 512, D)
    k[0, 1:, 0] = s
    out = mx_attention_reference(q, k, torch.ones(1, 512, D), scale=1.0, qk_mx=False, pv_mx=True,
                                 pv_perm=swizzle_perm)
    expected = (1 + 511 * 0.625) / (1 + 511 * e)
    torch.testing.assert_close(out, torch.full_like(out, expected.item()), atol=1e-6, rtol=1e-6)


def test_the_accumulator_goes_through_q_dtype_between_sections():
    """Two 512-key sections, every row's max in the first (key 0), so the correction is 1 and P
    is the same with one section: the difference is the accumulator's bf16 round trip between
    sections (attention_cte's ``o``), which moves a good share of the bf16 outputs; summation
    order alone moves almost none."""
    g = torch.Generator().manual_seed(3)
    q = torch.zeros(BH, S, D)
    q[..., 0] = 0.5 + torch.rand(BH, S, generator=g)
    k = torch.randn(BH, S, D, generator=g)
    k[..., 0] = -2 + 3.9 * torch.rand(BH, S, generator=g)
    k[:, 0, 0] = 2
    v = torch.randn(BH, S, D, generator=g)
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    sectioned = mx_attention_reference(q, k, v, scale=1.0, section_len=512, **MX_OFF)
    whole = mx_attention_reference(q, k, v, scale=1.0, section_len=S, **MX_OFF)
    assert (sectioned != whole).float().mean() > 0.05


@pytest.mark.parametrize("s_k,default_len", [(10752, 8192), (10240, 10240)])
def test_the_default_sections_are_the_kernels(s_k, default_len):
    """attention_cte walks 8192-key sections only above 10240 keys."""
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, 128, D, generator=g).bfloat16()
    k, v = (torch.randn(1, s_k, D, generator=g).bfloat16() for _ in range(2))
    kw = dict(scale=SCALE, **MX_ON)
    default = mx_attention_reference(q, k, v, **kw)
    other_len = s_k if default_len == 8192 else 8192
    assert torch.equal(default, mx_attention_reference(q, k, v, section_len=default_len, **kw))
    assert not torch.equal(default, mx_attention_reference(q, k, v, section_len=other_len, **kw))


def test_the_bounded_path_takes_scale_one_and_both_bounds():
    q, k, v = _qkv()
    lo, hi = _bounds(0, S)
    with pytest.raises(ValueError, match="scale"):
        mx_attention_reference(q, k, v, scale=SCALE, bound_min=lo, bound_max=hi, **MX_OFF)
    for one in (dict(bound_min=lo), dict(bound_max=hi)):
        with pytest.raises(ValueError, match="bound_min and bound_max"):
            mx_attention_reference(q, k, v, scale=1.0, **one, **MX_OFF)
