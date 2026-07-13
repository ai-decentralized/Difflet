"""Ulysses (all-to-all) CP attention: CPU degenerate path + the layout algebra.

The CPU backend only ever sees cp_degree == 1, so it can prove the op *reduces* to
plain attention but not that the all-to-all layout swap is correct. The correctness
of that swap is what actually makes Ulysses exact, so it is proven here by simulating
the cp ranks in-process: ``_sim_all_to_all`` reproduces XLA AllToAll's semantics
exactly (split into ``cp`` chunks along ``split_dim``, chunk ``j`` to rank ``j``;
concatenate what arrives along ``concat_dim`` in replica-group order), and the tests
run the real Ulysses algorithm over it. If the head/sequence chunk ordering were
wrong in either direction, the reconstructed output would not match a full dense
attention — that is the assertion.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch


# tests/conftest.py mocks torch for the default logic-only unit run; these are
# real-torch (CPU) numerical tests, so skip when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)


def _set_cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def _dense(q, k, v, scale):
    """Reference dense attention over [B, H, S, d]."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


# --------------------------------------------------------------------------
# CPU backend: cp_degree == 1, so both all-to-alls are identity.


def test_ulysses_attention_cpu_matches_plain_attention_cp1(monkeypatch):
    _set_cpu_backend(monkeypatch)
    from difflet.ops import ulysses_attention

    torch.manual_seed(0)
    b, h, s, d = 1, 2, 128, 64
    q, k, v = (torch.randn(b, h, s, d) for _ in range(3))
    scale = 1.0 / math.sqrt(d)

    out = ulysses_attention(q, k, v, scale=scale, causal=False)

    assert out.shape == (b, h, s, d)
    assert torch.allclose(_dense(q, k, v, scale), out.float(), atol=1e-4, rtol=1e-4)


def test_joint_ulysses_attention_cpu_matches_full_joint_attention_cp1(monkeypatch):
    _set_cpu_backend(monkeypatch)
    from difflet.ops import joint_ulysses_attention

    torch.manual_seed(0)
    b, h, s_img, s_txt, d = 1, 2, 128, 32, 64
    q_img, image_k, image_v = (torch.randn(b, h, s_img, d) for _ in range(3))
    q_txt, text_k, text_v = (torch.randn(b, h, s_txt, d) for _ in range(3))
    scale = 1.0 / math.sqrt(d)

    img_out, txt_out = joint_ulysses_attention(
        q_img, q_txt, image_k, image_v, text_k, text_v, scale=scale, causal=False
    )

    ref = _dense(
        torch.cat([q_img, q_txt], dim=2),
        torch.cat([image_k, text_k], dim=2),
        torch.cat([image_v, text_v], dim=2),
        scale,
    )
    assert img_out.shape == (b, h, s_img, d)
    assert txt_out.shape == (b, h, s_txt, d)
    assert torch.allclose(ref[:, :, :s_img], img_out.float(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref[:, :, s_img:], txt_out.float(), atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# The layout algebra, simulated across cp ranks (device-free).


def _sim_all_to_all(per_rank, *, split_dim, concat_dim):
    """XLA AllToAll over a full replica group, as a list-of-per-rank-tensors.

    Rank ``r`` sends chunk ``j`` (along ``split_dim``) to rank ``j``, and concatenates
    everything it receives along ``concat_dim`` in rank order.
    """
    cp = len(per_rank)
    chunks = [list(torch.chunk(t, cp, dim=split_dim)) for t in per_rank]
    return [
        torch.cat([chunks[src][r] for src in range(cp)], dim=concat_dim)
        for r in range(cp)
    ]


def _sim_all_gather(per_rank, *, dim):
    return [torch.cat(per_rank, dim=dim) for _ in per_rank]


def _shard_seq(t, cp, *, dim=2):
    return list(torch.chunk(t, cp, dim=dim))


@pytest.mark.parametrize("cp", [2, 4])
def test_ulysses_layout_swap_reconstructs_full_attention(cp):
    """The two all-to-alls must reproduce a full dense attention exactly."""
    torch.manual_seed(0)
    b, h_local, s, d = 1, 8, 256, 64  # h_local divisible by cp
    q, k, v = (torch.randn(b, h_local, s, d, dtype=torch.float64) for _ in range(3))
    scale = 1.0 / math.sqrt(d)

    # Each rank starts with its contiguous sequence block — how CP scatter shards it.
    q_r, k_r, v_r = _shard_seq(q, cp), _shard_seq(k, cp), _shard_seq(v, cp)

    # heads → seq
    q_f = _sim_all_to_all(q_r, split_dim=1, concat_dim=2)
    k_f = _sim_all_to_all(k_r, split_dim=1, concat_dim=2)
    v_f = _sim_all_to_all(v_r, split_dim=1, concat_dim=2)
    assert q_f[0].shape == (b, h_local // cp, s, d)

    out_f = [_dense(q_f[r], k_f[r], v_f[r], scale) for r in range(cp)]

    # seq → heads
    out_r = _sim_all_to_all(out_f, split_dim=2, concat_dim=1)
    assert out_r[0].shape == (b, h_local, s // cp, d)

    # Reassemble the sequence across ranks and compare to one dense attention.
    got = torch.cat(out_r, dim=2)
    assert torch.allclose(_dense(q, k, v, scale), got, atol=1e-10)


@pytest.mark.parametrize("cp", [2, 4])
def test_joint_ulysses_layout_reconstructs_full_joint_attention(cp):
    """Sharded image + replicated text must reconstruct full joint attention.

    Also pins the trick the device impl leans on: all-to-all'ing the *replicated*
    text yields ``cp`` identical copies of this rank's head block, so slicing the
    first ``S_txt`` is a valid rank-agnostic way to get the local head block.
    """
    torch.manual_seed(0)
    b, h_local, s_img, s_txt, d = 1, 8, 256, 32, 64
    q_img, k_img, v_img = (torch.randn(b, h_local, s_img, d, dtype=torch.float64) for _ in range(3))
    q_txt, k_txt, v_txt = (torch.randn(b, h_local, s_txt, d, dtype=torch.float64) for _ in range(3))
    scale = 1.0 / math.sqrt(d)

    # Image is sequence-sharded; text is replicated (identical on every rank).
    qi_r, ki_r, vi_r = _shard_seq(q_img, cp), _shard_seq(k_img, cp), _shard_seq(v_img, cp)
    qt_r, kt_r, vt_r = ([t] * cp for t in (q_txt, k_txt, v_txt))

    def to_head_shard(sharded, replicated):
        full = _sim_all_to_all(sharded, split_dim=1, concat_dim=2)
        tiled = _sim_all_to_all(replicated, split_dim=1, concat_dim=2)
        # Every rank held identical text, so `tiled` is cp copies of our head block.
        for r in range(cp):
            assert tiled[r].shape == (b, h_local // cp, cp * s_txt, d)
            for c in range(1, cp):
                assert torch.equal(
                    tiled[r].narrow(2, 0, s_txt), tiled[r].narrow(2, c * s_txt, s_txt)
                )
        return full, [t.narrow(2, 0, s_txt) for t in tiled]

    qi_f, qt_f = to_head_shard(qi_r, qt_r)
    ki_f, kt_f = to_head_shard(ki_r, kt_r)
    vi_f, vt_f = to_head_shard(vi_r, vt_r)

    out_f = [
        _dense(
            torch.cat([qi_f[r], qt_f[r]], dim=2),
            torch.cat([ki_f[r], kt_f[r]], dim=2),
            torch.cat([vi_f[r], vt_f[r]], dim=2),
            scale,
        )
        for r in range(cp)
    ]
    img_f = [o.narrow(2, 0, s_img) for o in out_f]
    txt_f = [o.narrow(2, s_img, s_txt) for o in out_f]

    img_r = _sim_all_to_all(img_f, split_dim=2, concat_dim=1)
    txt_r = _sim_all_gather(txt_f, dim=1)

    ref = _dense(
        torch.cat([q_img, q_txt], dim=2),
        torch.cat([k_img, k_txt], dim=2),
        torch.cat([v_img, v_txt], dim=2),
        scale,
    )
    got_img = torch.cat(img_r, dim=2)
    assert got_img.shape == (b, h_local, s_img, d)
    assert torch.allclose(ref[:, :, :s_img], got_img, atol=1e-10)

    # Text output is re-replicated: every rank holds the same full-head-count result.
    for r in range(cp):
        assert txt_r[r].shape == (b, h_local, s_txt, d)
        assert torch.allclose(ref[:, :, s_img:], txt_r[r], atol=1e-10)


def test_ulysses_rejects_head_count_not_divisible_by_cp():
    from difflet.backends.trainium.ops_impl.attention import _ulysses_check_heads

    _ulysses_check_heads(8, 2)  # fine
    with pytest.raises(ValueError, match="divisible by cp_degree"):
        _ulysses_check_heads(6, 4)


def test_ulysses_rejects_causal():
    # Causal would be silently wrong, not merely unimplemented: the all-to-all
    # reconstructs the full token set but not always in global sequence order.
    import math

    from difflet.backends.trainium.ops_impl.attention import ulysses_attention as dev_ulysses

    q = k = v = torch.zeros(1, 4, 8, 16)
    with pytest.raises(NotImplementedError, match="non-causal"):
        dev_ulysses(q, k, v, scale=1.0 / math.sqrt(16), causal=True)
