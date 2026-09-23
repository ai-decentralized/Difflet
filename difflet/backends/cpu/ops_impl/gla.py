"""Torch-native gated linear attention for CPU numerical checks.

Reference implementation. Deliberately the recurrent form: it is the definition
of the operator, it is unambiguous, and it is what the chunkwise form and the
Trainium kernel are validated against. It is O(T) sequential and slow by design.

Layout is `[B, T, H, K]` / `[B, T, H, V]`, mirroring flash-linear-attention's
`chunk_gla`, rather than the `[B*H, S, D]` used by `difflet.ops.attention`.
Callers reshape at the boundary.

GATE CONVENTION — the gate `g` is the NATURAL LOG of the per-channel decay,
i.e. the output of `F.logsigmoid(...)`, always <= 0. It is NOT a multiplier in
(0, 1). This matches FLA (`fla/ops/gla/naive.py` exponentiates with `.exp()`)
and is the single easiest thing to get backwards here: passing multipliers
produces a result that runs, has the right shape, and is silently wrong.

NON-CAUSAL MODE IS UNGATED. `g` is accepted and shape-checked but not applied
when `causal=False`. The gate is a forget gate over sequence order; with no
order there is no principled place to put it, and any placement is a modelling
choice rather than a consequence of the definition. Decision recorded in
`design-note-causal-flag`; revisit if a gated bidirectional variant is adopted.
"""

from __future__ import annotations

import torch


def gla_attention(
    q,
    k,
    v,
    g,
    *,
    scale: float | None = None,
    causal: bool = False,
    initial_state=None,
    output_final_state: bool = False,
    **kwargs,
):
    """Gated linear attention.

    Args:
        q, k: `[B, T, H, K]`
        v: `[B, T, H, V]`
        g: `[B, T, H, K]`, natural-log per-channel gate (<= 0). Applied only
            when `causal=True`; shape-checked but unused otherwise.
        scale: applied to `q`. Defaults to `K ** -0.5`.
        causal: if True, a left-to-right recurrence in which `g` acts as a
            forget gate on the running state. If False, a single global state
            summed over all tokens with no gate; the result is
            permutation-invariant over the sequence.
        initial_state: `[B, H, K, V]`, causal mode only.
        output_final_state: return the state alongside the output.

    Returns:
        `(o, final_state)` where `o` is `[B, T, H, V]` and `final_state` is
        `[B, H, K, V]` or None.

    Non-causal mode reduces to ungated linear attention: every query reads the
    same global state, so tokens are differentiated only by their own `q`. That
    is weak in isolation and is intended for hybrid stacks that interleave
    softmax attention layers.
    """
    del kwargs

    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(
            f"q, k, g must have the same shape; got {tuple(q.shape)}, "
            f"{tuple(k.shape)}, {tuple(g.shape)}"
        )
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must be [B, T, H, V] matching q's leading dims; got "
            f"{tuple(v.shape)} against {tuple(q.shape)}"
        )
    if not causal and initial_state is not None:
        raise ValueError("initial_state is only meaningful when causal=True")

    out_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    # [B, T, H, D] -> [B, H, T, D], fp32 throughout: this is a reference.
    q, k, v, g = (x.transpose(1, 2).float() for x in (q, k, v, g))

    if causal:
        state = q.new_zeros(B, H, K, V)
        if initial_state is not None:
            state = state + initial_state.float()
        o = q.new_zeros(B, H, T, V)
        for t in range(T):
            # Decay the running state, then write this token in undecayed.
            state = state * g[:, :, t].exp()[..., None]
            state = state + k[:, :, t][..., None] * v[:, :, t][..., None, :]
            o[:, :, t] = ((q[:, :, t] * scale)[..., None] * state).sum(-2)
    else:
        # Order-free and ungated: the state is a plain sum of outer products.
        # `g` is intentionally unused here; see the module docstring.
        state = torch.einsum("bhtk,bhtv->bhkv", k, v)
        o = torch.einsum("bhtk,bhkv->bhtv", q * scale, state)

    o = o.transpose(1, 2).to(out_dtype)
    return o, (state if output_final_state else None)


__all__ = ["gla_attention"]
