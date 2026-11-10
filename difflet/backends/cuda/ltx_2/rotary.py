"""BF16 rotary position embedding for the opt-in ``ltx_official_bf16`` LTX-2 profile.

The cos/sin tables are rounded to the activation dtype before use, and every product and sum
rounds in that dtype exactly where the native LTX-2 engine rounds, so the rotated queries and
keys are bit-identical to it.
"""

from __future__ import annotations

import torch

__all__ = ["apply_official_rotary"]


def apply_official_rotary(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
    *,
    rope_type: str,
    head_dim: int,
) -> torch.Tensor:
    """Rotate ``x`` by the ``(cos, sin)`` pair in ``x.dtype`` arithmetic.

    ``interleaved`` rotates adjacent (real, imag) element pairs of a ``(B, T, D)`` tensor.
    ``split`` rotates the first half of each head against its second half; a flat
    ``(B, T, H * head_dim)`` input with per-head ``(B, H, T, head_dim / 2)`` tables is
    returned in its flat layout.
    """
    cos, sin = (table.to(x.dtype) for table in freqs)
    if rope_type == "interleaved":
        return _rotate_interleaved(x, cos, sin)
    if rope_type == "split":
        return _rotate_split(x, cos, sin, head_dim)
    raise ValueError(f"unsupported LTX-2 rope_type {rope_type!r}")


def _rotate_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    real, imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(-2)
    # Two products and a sum as separate kernels: each rounds to x.dtype on its own.
    return x * cos + rotated * sin


def _rotate_split(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    per_head = x.ndim != 4 and cos.ndim == 4
    if per_head:
        batch = x.shape[0]
        _, heads, tokens, _ = cos.shape
        x = x.reshape(batch, tokens, heads, head_dim).swapaxes(1, 2)
    width = x.shape[-1]
    if width % 2:
        raise ValueError("split rotary requires an even head dimension")
    halves = x.unflatten(-1, (2, width // 2))
    first, second = halves.unbind(-2)
    out = halves * cos.unsqueeze(-2)
    out_first, out_second = out.unbind(-2)
    # addcmul_ fuses the cross term into the cos product with a single rounding.
    out_first.addcmul_(-sin, second)
    out_second.addcmul_(sin, first)
    out = out.flatten(-2)
    if per_head:
        out = out.swapaxes(1, 2).reshape(batch, tokens, heads * head_dim)
    return out
