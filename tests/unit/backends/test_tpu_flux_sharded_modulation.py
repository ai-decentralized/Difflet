"""FLUX SP's sharded AdaLN modulation (backends/tpu/flux/sp.py) equals diffusers'
replicated AdaLayerNormZero / AdaLayerNormZeroSingle, on the CPU.

Each of ``tp`` simulated ranks keeps its row shard (``_shard_rows_``); the
all-gather is replaced by concatenating the ranks' local vectors along dim 1, as
the TPU all-gather does in rank order.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle  # noqa: E402

from difflet.backends.tpu.flux import sp  # noqa: E402

DIM = 64


@pytest.mark.parametrize("tp", [2, 4])
def test_sharded_modulation_matches_replicated(tp):
    torch.manual_seed(0)
    norms = [AdaLayerNormZero(DIM).double(), AdaLayerNormZero(DIM).double(),
             AdaLayerNormZeroSingle(DIM).double()]
    temb = torch.randn(2, DIM, dtype=torch.float64)
    x = torch.randn(2, 5, DIM, dtype=torch.float64)
    expected = [norm(x, emb=temb) for norm in norms]

    ranks = []
    for rank in range(tp):
        shard = copy.deepcopy(norms)
        for norm in shard:
            sp._shard_rows_(norm.linear, rank, tp)
        ranks.append(shard)

    def gather(local, *, dim):  # every rank's local vector, in rank order
        act = torch.nn.functional.silu(temb)
        return torch.cat([torch.cat([n.linear(act) for n in shard], dim=1) for shard in ranks],
                         dim=dim)

    for rank in range(tp):
        mods = sp._modulations(ranks[rank], temb, tp, gather=gather)
        for norm, mod, want in zip(norms, mods, expected):
            got = sp._adaln(norm, x, temb, mod)
            assert len(got) == len(want)
            for g, w in zip(got, want):
                torch.testing.assert_close(g, w, rtol=0, atol=1e-12)


def test_adaln_without_modulation_is_the_norm():
    norm = AdaLayerNormZeroSingle(DIM)
    temb, x = torch.randn(1, DIM), torch.randn(1, 3, DIM)
    for g, w in zip(sp._adaln(norm, x, temb, None), norm(x, emb=temb)):
        torch.testing.assert_close(g, w, rtol=0, atol=0)
