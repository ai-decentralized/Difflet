"""Translate a contiguous-key-pad attention_mask -> attention_cte bound_min/bound_max.

attention_cte's ``bound_min``/``bound_max`` express a per-query CONTIGUOUS ``[lo, hi)``
KV range ("sequence packing", attention_cte.py docstring section 6). A key-padding /
packed / block-diagonal mask whose per-query valid keys form a contiguous range is
**losslessly** expressible this way -- proven on device in
``tests/numerical/test_attention_cte_bound_mask.py`` (cosine 0.99986, 1.16x vs SDPA).
An arbitrary / sparse mask is not -> this returns ``None`` so the caller falls back to
the slow XLA SDPA path.

Pure torch (no nki / neuron import) so it is unit-testable on CPU and the routing
decision is made at trace time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Additive masks use a large negative for "masked"; treat anything <= this as masked.
# Set below the common diffusers/HF -10000 sentinel AFTER low-precision rounding (bf16
# rounds -10000 to -9984) yet well above genuine soft attention biases (|bias| <~ 100),
# so hard masks resolve and soft biases fall in the ambiguous band -> None (SDPA).
_MASKED_THRESHOLD = -1e3


def mask_to_contiguous_bounds(
    attention_mask: torch.Tensor, num_heads: int, seq_q: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(bound_min, bound_max)`` of shape ``(B*num_heads, seq_q, 1)`` int32 if
    every query's valid-key set is a contiguous ``[lo, hi)`` range; else ``None``.

    ``attention_mask``: additive float (0 attend, ``<= -1e4`` masked) OR bool
    (``True`` = attend), broadcastable to ``[B, H?, S_q, S_kv]``. A head dim, if present
    and ``!= 1``, must be uniform across heads (key-pad masks do not vary by head). A
    query dim of 1 (key-pad broadcast over queries) is expanded to ``seq_q``.
    """
    m = attention_mask
    if m.dim() == 2:            # [S_q, S_kv]
        m = m.unsqueeze(0)     # [1, S_q, S_kv]
    elif m.dim() == 4:         # [B, H, S_q, S_kv]
        if m.shape[1] != 1 and not bool((m[:, :1] == m).all()):
            return None        # heads differ -> not a uniform key-pad mask
        m = m[:, 0]            # [B, S_q, S_kv]
    elif m.dim() != 3:         # not a recognized layout
        return None

    if m.dtype == torch.bool:
        valid = m                       # bool convention: True = attend
    elif not m.is_floating_point():
        valid = m != 0                  # integer validity: nonzero = attend, 0 = pad/mask
    else:
        # A float mask must be a HARD additive mask: values in {~0 = attend,
        # <= _MASKED_THRESHOLD = masked}. Refuse to GUESS on anything ambiguous
        # (return None -> caller falls back to SDPA) rather than risk a silent
        # misread:
        #   - any positive value  -> looks like a float-validity mask, not additive;
        #   - any mildly-negative value (between threshold and ~0) -> a soft attention
        #     bias, which cannot be a hard [lo, hi) range.
        if bool((m > 1e-4).any()):
            return None
        if bool(((m < -1e-4) & (m > _MASKED_THRESHOLD)).any()):
            return None
        valid = m > _MASKED_THRESHOLD
    B, sq_mask, skv = valid.shape

    idx = torch.arange(skv, device=valid.device)
    lo = torch.where(valid, idx, torch.full_like(idx, skv)).amin(dim=-1)     # [B, sq_mask]
    hi = torch.where(valid, idx, torch.full_like(idx, -1)).amax(dim=-1) + 1  # [B, sq_mask]
    cnt = valid.sum(dim=-1)                                                  # [B, sq_mask]
    if not bool(((cnt == (hi - lo)) & (cnt > 0)).all()):
        return None            # some query's valid keys are empty or non-contiguous

    if sq_mask == 1 and seq_q != 1:
        lo = lo.expand(B, seq_q)
        hi = hi.expand(B, seq_q)
    elif sq_mask != seq_q:
        return None            # mask query dim does not match the real query length

    # (B, seq_q) -> (B, num_heads, seq_q, 1) -> (B*num_heads, seq_q, 1)
    def _per_head(t: torch.Tensor) -> torch.Tensor:
        # (B, seq_q) -> (B, 1, seq_q, 1) -> (B, num_heads, seq_q, 1) -> (B*num_heads, seq_q, 1)
        return (
            t.to(torch.int32)
            .unsqueeze(-1)
            .unsqueeze(1)
            .expand(B, num_heads, seq_q, 1)
            .reshape(B * num_heads, seq_q, 1)
            .contiguous()
        )

    return _per_head(lo), _per_head(hi)
