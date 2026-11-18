"""Static K/V-tile masks for the attention fork's ``kv_tiles`` (``mx_attention_kernel.py``; E2 of
``docs/plans/2026-11-18-attn-constmax-sparse-experiments.md``), and the masked fp32 reference.

Tokens are in Wan's frame-major raster order (``modeling_wan.py``: frame, then row, then column;
wan_81f is 21 latent frames of 30 x 52). A query block attends a key block when some query of it
and some key of it are within the window, so every in-window key is kept (the tile mask is a
superset of the token window). A 512-key tile spans whole rows (512 > 52 columns), so the window
has no column extent: ``wt`` frames and, for the STA-like family, ``wh`` rows.
"""

from __future__ import annotations

Q_GROUP, K_TILE = 128, 512  # the kernel's query group and key tile


def _cells(start: int, stop: int, rows: int, cols: int) -> set[tuple[int, int]]:
    """The (frame, row) cells tokens ``[start, stop)`` touch."""
    return {(i // (rows * cols), i // cols % rows) for i in range(start, stop)}


def window_tiles(frames: int, rows: int, cols: int, *, wt: int, wh: int | None = None,
                 q_block: int = Q_GROUP, k_block: int = K_TILE) -> tuple[tuple[int, ...], ...]:
    """Per ``q_block``-token query block, the sorted ``k_block``-token key blocks holding a key
    within ``wt`` frames and (``wh`` given) ``wh`` rows of one of its queries: ``kv_tiles``.
    ``wh=None`` is the temporal window (whole frames)."""
    n = frames * rows * cols

    def blocks(size):
        return [_cells(s, min(s + size, n), rows, cols) for s in range(0, n, size)]

    def near(a, b):
        return abs(a[0] - b[0]) <= wt and (wh is None or abs(a[1] - b[1]) <= wh)

    keys = blocks(k_block)
    return tuple(tuple(t for t, kc in enumerate(keys) if any(near(a, b) for a in qc for b in kc))
                 for qc in blocks(q_block))


def density(kv_tiles, num_k_tiles: int) -> float:
    """Kept (query block, key tile) pairs over all pairs."""
    return sum(map(len, kv_tiles)) / (len(kv_tiles) * num_k_tiles)


def masked_attention_reference(q, k, v, *, scale: float, kv_tiles=None, chunk: int = 1024):
    """fp32 ``softmax(scale q k^T) v`` on ``[BH, S, D]``, keys outside query group g's
    ``kv_tiles[g]`` at -inf (None: dense), ``chunk`` queries (a multiple of 128) at a time."""
    import torch

    qf, kf, vf = q.float(), k.float(), v.float()
    s_q, s_k = q.shape[1], k.shape[1]
    out = torch.empty(q.shape[0], s_q, v.shape[2])
    for c0 in range(0, s_q, chunk):
        c1 = min(c0 + chunk, s_q)
        scores = (qf[:, c0:c1] @ kf.transpose(1, 2)) * scale
        if kv_tiles is not None:
            keep = torch.zeros(c1 - c0, s_k, dtype=torch.bool)
            for g in range(c0 // Q_GROUP, (c1 + Q_GROUP - 1) // Q_GROUP):
                rows = slice(g * Q_GROUP - c0, (g + 1) * Q_GROUP - c0)
                for t in kv_tiles[g]:
                    keep[rows, t * K_TILE:(t + 1) * K_TILE] = True
            scores = scores.masked_fill(~keep, float("-inf"))
        out[:, c0:c1] = scores.softmax(-1) @ vf
    return out
