"""CPU tests of Difflet's attention_cte fork (``ops_impl/mx_attention_kernel.py``): its argument
checks run in Python at trace time, before any NKI instruction, so no device is needed."""

from types import SimpleNamespace

import pytest

pytest.importorskip("nki")

from difflet.backends.neuron.ops_impl import mx_attention_kernel as fork  # noqa: E402


def _t(*shape):
    return SimpleNamespace(shape=shape)


Q = K = V = _t(2, 256, 128)  # the tp_q / tp_k layout: [batch, seqlen, d]

#: Every attention_cte feature Difflet never calls, with a value that turns it on.
REFUSED = {
    "causal_mask": True,
    "k_prior": _t(2, 128, 128),
    "v_prior": _t(2, 128, 128),
    "prior_used_len": _t(1),
    "sink": _t(2, 1),
    "sliding_window": 64,
    "cache_softmax": True,
    "cp_offset": _t(1, 1),
    "global_cp_deg": 2,
    "cp_strided_q_slicing": True,
    "cp_striped_input": True,
    "position_bias": _t(2, 256, 256),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_attention_cte_refuses_what_difflet_never_uses(name):
    with pytest.raises(AssertionError, match=f"refuses {name}"):
        fork._attention_cte(q=Q, k=K, v=V, **{"causal_mask": False, name: REFUSED[name]})


@pytest.mark.parametrize("q, tp_q", [(_t(2, 256, 64), True), (_t(2, 64, 256), False)])
def test_attention_cte_refuses_head_dim_other_than_128(q, tp_q):
    with pytest.raises(AssertionError, match="head dim 128 only, got 64"):
        fork._attention_cte(q=q, k=K, v=V, causal_mask=False, tp_q=tp_q)


@pytest.mark.parametrize("q, tp_q, sliding_window", [(Q, True, None), (_t(2, 128, 256), False, 0)])
def test_difflet_subset_passes(q, tp_q, sliding_window):
    """What ops_impl/attention.py passes on its unbounded and bounded paths (attention_cte's
    defaults but causal_mask=False; bounds are not refused), in either q layout."""
    fork._check_difflet_subset(q=q, tp_q=tp_q, causal_mask=False, k_prior=None, v_prior=None,
                               prior_used_len=None, sink=None, sliding_window=sliding_window,
                               cache_softmax=False, cp_offset=None, global_cp_deg=None,
                               cp_strided_q_slicing=False, cp_striped_input=False,
                               position_bias=None)


@pytest.mark.parametrize("pv_mx", [False, True])
def test_qk_mx_is_refused_until_mm1_lands(pv_mx):
    with pytest.raises(AssertionError, match="qk_mx: MM1 on nc_matmul_mx"):
        fork.mx_attention_fwd.func(Q, K, V, scale=1.0, pv_mx=pv_mx, qk_mx=True)


def test_entry_point_keeps_attention_cte_compile_options():
    """mx_attention_fwd gets the address-rotation / non-SSA options attention_cte gets, so both
    flags off compile to the same kernel."""
    for option in ("address_rotation", "non_ssa_legalization"):
        assert getattr(fork.mx_attention_fwd, option) == getattr(fork.attention_cte, option)


def test_the_reference_defaults_to_the_kernels_block_membership():
    """Rulings BR1 / BR16: the perms the kernel exports (contiguous blocks of 32, locked on the
    device on MM2's own loaders by scripts/mx_attention/membership.py) are the CPU reference's
    defaults."""
    import torch

    from difflet.quant.mx import swizzle_perm
    from difflet.quant.mx_attention import mx_attention_reference

    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 1024, 128, generator=g).bfloat16() for _ in range(3))
    mx = dict(scale=128 ** -0.5, qk_mx=True, pv_mx=True)
    default = mx_attention_reference(q, k, v, **mx)
    assert torch.equal(default, mx_attention_reference(q, k, v, **mx, qk_perm=fork.qk_perm,
                                                       pv_perm=fork.pv_perm))
    assert not torch.equal(default, mx_attention_reference(q, k, v, **mx, qk_perm=swizzle_perm,
                                                           pv_perm=swizzle_perm))


# --- the kernel itself on the NKI simulator (CPU), small shapes ---------------------------------

def _simulate(kernel, q, k, v, bounds=(), **kwargs):
    import ml_dtypes
    import numpy as np
    import torch
    from nki._simulator import simulate_kernel

    def bf16(t):
        return t.float().numpy().astype(ml_dtypes.bfloat16)

    if bounds:
        kwargs.update(bound_min=bounds[0].numpy(), bound_max=bounds[1].numpy())
    out = simulate_kernel(kernel, (bf16(q), bf16(k), bf16(v)), kwargs)
    return torch.from_numpy(np.asarray(out).astype(np.float32))


def _inputs(bh, s_q, s_k, valid_k=None):
    import torch

    g = torch.Generator().manual_seed(0)
    q = torch.randn(bh, s_q, 128, generator=g).bfloat16()
    k, v = (torch.randn(bh, s_k, 128, generator=g).bfloat16() for _ in range(2))
    if valid_k is None:
        return q, k, v, ()
    q = (q.float() * 128 ** -0.5).bfloat16()  # the bounded path takes scale 1.0
    hi = torch.full((bh, s_q, 1), valid_k, dtype=torch.int32)
    return q, k, v, (torch.zeros_like(hi), hi)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


#: (bh, s_q, s_k, valid_k): a partial query group and a partial key tile whose length is not a
#: multiple of 4; bounded with bound_max % 32 != 0; two flash sections (10,752 keys).
SIM_SHAPES = {"partial": (1, 602, 602, None), "bounded": (2, 640, 640, 600),
              "sections": (1, 128, 10752, None)}


@pytest.mark.parametrize("case", sorted(SIM_SHAPES))
def test_pv_mx_simulated_matches_the_mirror(case):
    """MM2 on nc_matmul_mx (Step 3): the simulated kernel equals the Task 1 mirror (P and V
    MX-quantized per 512-key tile by ``pv_perm``, the last tile zero-padded) to summation order,
    and is not the bf16 kernel."""
    from difflet.quant.mx_attention import mx_attention_reference

    bh, s_q, s_k, valid = SIM_SHAPES[case]
    q, k, v, bounds = _inputs(bh, s_q, s_k, valid)
    scale = 1.0 if bounds else 128 ** -0.5
    out = _simulate(fork.mx_attention_fwd, q, k, v, bounds, scale=scale, pv_mx=True, qk_mx=False)
    kw = dict(bound_min=bounds[0], bound_max=bounds[1]) if bounds else {}
    mirror = mx_attention_reference(q, k, v, scale=scale, qk_mx=False, pv_mx=True,
                                    pv_perm=fork.pv_perm, **kw).float()
    bf16 = mx_attention_reference(q, k, v, scale=scale, qk_mx=False, pv_mx=False, **kw).float()
    assert _rel(out, mirror) < 2e-3, _rel(out, mirror)
    assert _rel(out, bf16) > 4 * _rel(out, mirror)


@pytest.mark.parametrize("case", ["partial", "bounded"])
def test_both_flags_off_simulates_bit_identical_to_attention_cte(case):
    """The pv_mx edits leave the flags-off path alone: bit for bit nkilib's attention_cte."""
    import torch
    from nkilib.core.attention.attention_cte import attention_cte

    bh, s_q, s_k, valid = SIM_SHAPES[case]
    q, k, v, bounds = _inputs(bh, s_q, s_k, valid)
    scale = 1.0 if bounds else 128 ** -0.5
    out = _simulate(fork.mx_attention_fwd, q, k, v, bounds, scale=scale, pv_mx=False, qk_mx=False)
    cte = _simulate(attention_cte, q, k, v, bounds, scale=scale, causal_mask=False, tp_q=True,
                    tp_k=True)
    assert torch.equal(out, cte)
