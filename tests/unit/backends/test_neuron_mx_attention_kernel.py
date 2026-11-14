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


@pytest.mark.parametrize("flags", [{"pv_mx": True, "qk_mx": False},
                                   {"pv_mx": False, "qk_mx": True}])
def test_mx_flags_are_refused_until_the_mx_matmuls_land(flags):
    with pytest.raises(AssertionError, match="nc_matmul_mx"):
        fork.mx_attention_fwd.func(Q, K, V, scale=1.0, **flags)


def test_entry_point_keeps_attention_cte_compile_options():
    """mx_attention_fwd gets the address-rotation / non-SSA options attention_cte gets, so both
    flags off compile to the same kernel."""
    for option in ("address_rotation", "non_ssa_legalization"):
        assert getattr(fork.mx_attention_fwd, option) == getattr(fork.attention_cte, option)

