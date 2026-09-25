import pytest
import torch
import torch.nn.functional as F

B, T, H, K, V = 2, 16, 3, 8, 8
SCALE = K**-0.5
TOL = dict(atol=1e-4, rtol=1e-4)


def _inputs(t=T):
    q = torch.randn(B, t, H, K)
    k = torch.randn(B, t, H, K)
    v = torch.randn(B, t, H, V)
    g = F.logsigmoid(torch.randn(B, t, H, K))  # log-space gate, <= 0
    return q, k, v, g


def _reference_causal(q, k, v, g, scale, initial_state=None):
    """Independent recurrent reference, fp64, written from the definition.

    Kept alongside the FLA parity test below so causal correctness is still
    covered when `fla` is not installed, and so a disagreement between two
    independently written references is visible rather than assumed away.
    """
    q, k, v, g = (x.transpose(1, 2).double() for x in (q, k, v, g))
    b, h, t, kk = q.shape
    state = q.new_zeros(b, h, kk, v.shape[-1])
    if initial_state is not None:
        state = state + initial_state.double()
    o = q.new_zeros(b, h, t, v.shape[-1])
    for i in range(t):
        state = state * g[:, :, i].exp()[..., None]
        state = state + k[:, :, i][..., None] * v[:, :, i][..., None, :]
        o[:, :, i] = ((q[:, :, i] * scale)[..., None] * state).sum(-2)
    return o.transpose(1, 2)


# --------------------------------------------------------------- correctness

def test_causal_matches_fla_reference(monkeypatch):
    """Primary causal check: parity with the implementation this op mirrors.

    This is what makes the later NKI kernel's correctness argument transitive —
    kernel matches CPU, CPU matches FLA. Skips if `fla` is not importable; the
    fp64 reference test below covers causal correctness unconditionally.
    """
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    #naive = pytest.importorskip(
    #    "fla.ops.gla.naive",
    #    reason="install flash-linear-attention, or put the clone on PYTHONPATH",
    #)
    try:
        from fla.ops.gla import naive
    except (ImportError, OSError):
        pytest.skip("flash-linear-attention is not importable")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    ref, _ = naive.naive_recurrent_gla(q, k, v, g)
    out, _ = gla_attention(q, k, v, g, scale=SCALE, causal=True)

    assert out.shape == (B, T, H, V)
    assert torch.allclose(ref.float(), out.float(), **TOL)


def test_causal_matches_fp64_reference(monkeypatch):
    """Second, independent opinion at fp64. Runs with or without `fla`."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    out, state = gla_attention(
        q, k, v, g, scale=SCALE, causal=True, output_final_state=True
    )
    ref = _reference_causal(q, k, v, g, SCALE)

    assert out.shape == (B, T, H, V)
    assert state.shape == (B, H, K, V)
    assert torch.allclose(ref.float(), out.float(), **TOL)


def test_causal_state_composes_across_a_split(monkeypatch):
    """Splitting a sequence and threading the state must reproduce one pass.

    This is what `initial_state` and `output_final_state` exist for, and it
    tests them against the op's own full-sequence result rather than against a
    reference that threads state the same way the implementation does.
    """
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()
    half = T // 2

    full, _ = gla_attention(q, k, v, g, scale=SCALE, causal=True)

    first, state = gla_attention(
        q[:, :half], k[:, :half], v[:, :half], g[:, :half],
        scale=SCALE, causal=True, output_final_state=True,
    )
    second, _ = gla_attention(
        q[:, half:], k[:, half:], v[:, half:], g[:, half:],
        scale=SCALE, causal=True, initial_state=state,
    )

    stitched = torch.cat([first, second], dim=1)
    assert torch.allclose(full.float(), stitched.float(), **TOL)


def test_non_causal_is_permutation_invariant(monkeypatch):
    """Non-causal mode has no ordering, so shuffling tokens shuffles outputs."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()
    perm = torch.randperm(T)

    out, _ = gla_attention(q, k, v, g, causal=False)
    out_perm, _ = gla_attention(
        q[:, perm], k[:, perm], v[:, perm], g[:, perm], causal=False
    )

    assert torch.allclose(out[:, perm].float(), out_perm.float(), **TOL)


def test_non_causal_ignores_the_gate(monkeypatch):
    """Non-causal mode is ungated by design, so `g` must not change the output.

    Pins the decision rather than leaving it implicit: if a gate is later wired
    into the non-causal branch, this fails instead of silently changing the
    numerics the Trainium kernel is validated against.
    """
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    gated, _ = gla_attention(q, k, v, g, scale=SCALE, causal=False)
    zeroed, _ = gla_attention(
        q, k, v, torch.zeros_like(g), scale=SCALE, causal=False
    )

    assert torch.allclose(gated, zeroed, **TOL)


def test_non_causal_matches_plain_linear_attention(monkeypatch):
    """States what non-causal computes: q @ sum_j (k_j v_j^T), no gate."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    out, _ = gla_attention(q, k, v, g, scale=SCALE, causal=False)

    qd, kd, vd = (x.transpose(1, 2).double() for x in (q, k, v))
    state = torch.einsum("bhtk,bhtv->bhkv", kd, vd)
    ref = torch.einsum("bhtk,bhkv->bhtv", qd * SCALE, state).transpose(1, 2)

    assert torch.allclose(ref.float(), out.float(), **TOL)


# ------------------------------------------------------- degenerate identity

def test_zero_gate_is_ungated_linear_attention(monkeypatch):
    """g=0 gives decay 1, so the state never forgets: plain linear attention.

    Exact, and computed here by a different algorithm — a parallel masked
    matmul against the implementation's sequential loop.
    """
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, _ = _inputs()
    g = torch.zeros(B, T, H, K)

    out, _ = gla_attention(q, k, v, g, scale=SCALE, causal=True)

    qd, kd, vd = (x.transpose(1, 2).double() for x in (q, k, v))
    scores = torch.einsum("bhtk,bhsk->bhts", qd * SCALE, kd)
    scores = scores * torch.tril(torch.ones(T, T, dtype=torch.float64))
    ref = torch.einsum("bhts,bhsv->bhtv", scores, vd).transpose(1, 2)

    assert torch.allclose(ref.float(), out.float(), **TOL)


# --------------------------------------------------------------- mode guard

def test_causal_and_non_causal_differ(monkeypatch):
    """Guards against a mode flag that is accepted and then ignored."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    causal_out, _ = gla_attention(q, k, v, g, causal=True)
    non_causal_out, _ = gla_attention(q, k, v, g, causal=False)

    assert not torch.allclose(causal_out, non_causal_out, atol=1e-3)


# ------------------------------------------------------------ negative tests

def test_rejects_mismatched_shapes(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    with pytest.raises(ValueError):
        gla_attention(q, k, v, g[..., :-1])
    with pytest.raises(ValueError):
        gla_attention(q, k, v[:, :-1], g)


def test_rejects_initial_state_when_non_causal(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import gla_attention

    torch.manual_seed(0)
    q, k, v, g = _inputs()

    with pytest.raises(ValueError):
        gla_attention(
            q, k, v, g, causal=False, initial_state=torch.randn(B, H, K, V)
        )
