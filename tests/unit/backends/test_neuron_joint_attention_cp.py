"""Joint (text + image) context-parallel attention on the neuron backend; CPU only, no device.

``joint_ring_attention`` and ``joint_ulysses_attention`` take a sequence-sharded image stream
and a replicated text stream (Qwen-Image, HunyuanVideo). The gloo tests run them on 2 and 4
CPU ranks, eagerly and under ``torch.compile(backend="aot_eager", fullgraph=True)``, and hold
every rank's output to one dense softmax over the full joint sequence; the single-process
tests cover the cp == 1 fallbacks and the rejected inputs.
"""

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.ops_impl import attention as A  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh as pm  # noqa: E402
from difflet.pipeline.parallel_mesh import MeshSpec  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_workers import (  # noqa: E402
    JOINT_D,
    JOINT_HEADS,
    JOINT_S_IMG,
    JOINT_S_TXT,
    joint_cp_worker,
)

TOL = 1e-5  # fp32 against an fp64 reference


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def _streams(heads=2, s_txt=5, s_img=7, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    txt = [torch.randn(1, heads, s_txt, d, generator=g) for _ in range(3)]
    img = [torch.randn(1, heads, s_img, d, generator=g) for _ in range(3)]
    return txt, img


def _dense(q, k, v, *, scale, valid=None):
    scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    if valid is not None:
        scores[..., valid:] = float("-inf")
    return (torch.softmax(scores, -1) @ v.double()).float()


# ---- gloo ranks: each rank's shard against the full joint softmax ---------------------


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize(("world_size", "tp"), [(2, 1), (4, 1), (4, 2)],
                         ids=["cp2", "cp4", "tp2xcp2"])
def test_joint_cp_attention_matches_dense_joint_softmax(world_size, tp, mode):
    cp = world_size // tp
    hl = JOINT_HEADS // tp
    for rank, r in enumerate(run_ranks(joint_cp_worker, tp, mode, world_size=world_size)):
        assert r["cp"] == cp
        assert r["ring_shape"] == [1, hl, JOINT_S_TXT + JOINT_S_IMG // cp, JOINT_D]
        assert r["ulysses_shapes"] == [[1, hl, JOINT_S_IMG // cp, JOINT_D],
                                       [1, hl, JOINT_S_TXT, JOINT_D]]
        errors = {name: r[name] for name in ("ring_err", "ulysses_img_err", "ulysses_txt_err",
                                             "masked_img_err", "masked_txt_err")}
        assert all(e < TOL for e in errors.values()), (rank, errors)
        assert "divisible by cp_degree" in r["head_error"], (rank, r["head_error"])


# ---- single process: cp == 1 fallbacks and rejected inputs ----------------------------


def test_joint_ring_attention_without_context_parallelism_is_dense_joint_attention():
    (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams()
    q = torch.cat([q_t, q_i], 2)
    out = A.joint_ring_attention(q, k_i, v_i, k_t, v_t, scale=0.3)
    torch.testing.assert_close(out, _dense(q, torch.cat([k_t, k_i], 2), torch.cat([v_t, v_i], 2),
                                           scale=0.3))


def test_joint_ulysses_attention_without_context_parallelism_is_dense_joint_attention():
    (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams()
    img_out, txt_out = A.joint_ulysses_attention(q_i, q_t, k_i, v_i, k_t, v_t, scale=0.3)
    ref = _dense(torch.cat([q_i, q_t], 2), torch.cat([k_i, k_t], 2), torch.cat([v_i, v_t], 2),
                 scale=0.3)
    torch.testing.assert_close(img_out, ref[:, :, :7])
    torch.testing.assert_close(txt_out, ref[:, :, 7:])


def test_joint_ulysses_key_valid_len_masks_trailing_text_keys_at_cp1():
    (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams()
    img_out, txt_out = A.joint_ulysses_attention(
        q_i, q_t, k_i, v_i, k_t, v_t, scale=0.3, key_valid_len=torch.tensor([9], dtype=torch.int32))
    ref = _dense(torch.cat([q_i, q_t], 2), torch.cat([k_i, k_t], 2), torch.cat([v_i, v_t], 2),
                 scale=0.3, valid=9)
    torch.testing.assert_close(img_out, ref[:, :, :7])
    torch.testing.assert_close(txt_out, ref[:, :, 7:])


def test_joint_variants_reject_causal():
    (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams()
    with pytest.raises(NotImplementedError, match="non-causal"):
        A.joint_ring_attention(torch.cat([q_t, q_i], 2), k_i, v_i, k_t, v_t, scale=0.3,
                               causal=True)
    with pytest.raises(NotImplementedError, match="non-causal"):
        A.joint_ulysses_attention(q_i, q_t, k_i, v_i, k_t, v_t, scale=0.3, causal=True)


def test_joint_ulysses_rejects_heads_not_divisible_by_cp():
    # A spec-only cp=2 mesh: the head check runs before any collective, so no process group.
    pm.init_parallel_mesh(MeshSpec(cp=2))
    (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams(heads=3)
    with pytest.raises(ValueError, match=r"head count \(3\) divisible by cp_degree \(2\)"):
        A.joint_ulysses_attention(q_i, q_t, k_i, v_i, k_t, v_t, scale=0.3)


def test_joint_variants_dispatch_through_difflet_ops(monkeypatch):
    """The ops dispatcher passes ``key_valid_len`` to every backend's joint Ulysses; the neuron
    signature must take it, and both joint names must resolve to these functions."""
    from difflet.backends import registry
    from difflet.ops._dispatch import load_backend_attr

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    registry._get_backend_by_name.cache_clear()
    try:
        assert load_backend_attr("attention", "joint_ring_attention") is A.joint_ring_attention
        assert load_backend_attr("attention", "joint_ulysses_attention") is \
            A.joint_ulysses_attention
        from difflet.ops import joint_ulysses_attention

        (q_t, k_t, v_t), (q_i, k_i, v_i) = _streams()
        img_out, _ = joint_ulysses_attention(q_i, q_t, k_i, v_i, k_t, v_t, scale=0.3)
        assert img_out.shape == q_i.shape
    finally:
        registry._get_backend_by_name.cache_clear()
