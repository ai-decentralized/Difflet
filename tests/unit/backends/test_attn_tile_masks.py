"""The kv_tiles mask builder (``ops_impl/attn_tile_masks.py``): pure Python, CPU only."""

import pytest

from difflet.backends.neuron.ops_impl.attn_tile_masks import (
    density,
    masked_attention_reference,
    window_tiles,
)

WAN_81F = (21, 30, 52)  # latent frames, rows, cols at 480 x 832 x 81: 32,760 tokens
#: The E2 masks at wan_81f and their tile densities (256 groups x 64 tiles).
WAN_MASKS = {"temporal w=2": (dict(wt=2), 0.240), "temporal w=6": (dict(wt=6), 0.534),
             "temporal w=10": (dict(wt=10), 0.756), "sta t=3 h=6": (dict(wt=3, wh=6), 0.243),
             "sta t=8 h=6": (dict(wt=8, wh=6), 0.511), "sta t=16 h=6": (dict(wt=16, wh=6), 0.753)}


@pytest.mark.parametrize("name", sorted(WAN_MASKS))
def test_wan_81f_masks_have_their_density_and_rows(name):
    kw, expected = WAN_MASKS[name]
    tiles = window_tiles(*WAN_81F, **kw)
    assert len(tiles) == 256 and all(tiles) and all(0 <= t < 64 for row in tiles for t in row)
    assert all(row == tuple(sorted(set(row))) for row in tiles)
    assert density(tiles, 64) == pytest.approx(expected, abs=5e-4)


def test_density_grows_with_the_window_to_full():
    temporal = [density(window_tiles(*WAN_81F, wt=w), 64) for w in range(21)]
    assert temporal == sorted(temporal) and temporal[-1] == 1.0
    assert density(window_tiles(*WAN_81F, wt=20, wh=29), 64) == 1.0


@pytest.mark.parametrize("kw", [dict(wt=2), dict(wt=3, wh=6), dict(wt=0, wh=0)])
def test_the_mask_is_symmetric_and_its_groups_refine_the_tiles(kw):
    """With 512-token query blocks the relation is symmetric; each 512 block's row is the union of
    its four 128-query groups' rows."""
    square = window_tiles(*WAN_81F, q_block=512, **kw)
    assert all((i in square[j]) == (j in square[i]) for i in range(64) for j in range(64))
    groups = window_tiles(*WAN_81F, **kw)
    for b in range(64):
        assert set().union(*groups[4 * b:4 * b + 4]) == set(square[b])


@pytest.mark.parametrize("wh", [None, 1])
def test_every_in_window_token_pair_is_kept(wh):
    """Brute force on a small video (4 frames of 5 x 6, blocks of 8 queries and 16 keys): a key
    within the window of a query is in a kept tile, and every kept tile holds one."""
    frames, rows, cols, wt = 4, 5, 6, 1
    tiles = window_tiles(frames, rows, cols, wt=wt, wh=wh, q_block=8, k_block=16)
    n = frames * rows * cols

    def near(i, j):
        fi, ri, fj, rj = i // (rows * cols), i // cols % rows, j // (rows * cols), j // cols % rows
        return abs(fi - fj) <= wt and (wh is None or abs(ri - rj) <= wh)

    for g, row in enumerate(tiles):
        queries = range(8 * g, min(8 * g + 8, n))
        wanted = {j // 16 for i in queries for j in range(n) if near(i, j)}
        assert set(row) == wanted


def test_the_reference_masks_skipped_tiles_and_equals_attention_unmasked():
    import torch

    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 640, 128, generator=g) for _ in range(3))
    tiles = ((0,), (1,), (0, 1), (1,), (0,))
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(masked_attention_reference(q, k, v, scale=128 ** -0.5, chunk=256), dense,
                          atol=1e-5)
    out = masked_attention_reference(q, k, v, scale=128 ** -0.5, kv_tiles=tiles, chunk=256)
    v2 = v.clone()
    v2[:, 512:] = 1e3  # tile 1: skipped by groups 0 and 4 only
    out2 = masked_attention_reference(q, k, v2, scale=128 ** -0.5, kv_tiles=tiles, chunk=256)
    assert torch.equal(out[:, :128], out2[:, :128]) and torch.equal(out[:, 512:], out2[:, 512:])
    assert not torch.allclose(out[:, 128:512], out2[:, 128:512])
