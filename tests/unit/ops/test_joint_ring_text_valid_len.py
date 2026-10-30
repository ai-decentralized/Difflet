"""``joint_ring_attention``'s ``text_valid_len``: a right-padded text stream given as the count
of valid text keys per batch row (HunyuanVideo's padded-Llama mask under ring CP)."""

import math

import pytest
import torch
import torch.nn.functional as F


def _inputs(b, h=2, s_img=12, s_txt=8, d=16):
    torch.manual_seed(0)
    q = torch.randn(b, h, s_img + s_txt, d)
    image_k, image_v = torch.randn(b, h, s_img, d), torch.randn(b, h, s_img, d)
    text_k, text_v = torch.randn(b, h, s_txt, d), torch.randn(b, h, s_txt, d)
    return q, image_k, image_v, text_k, text_v


@pytest.mark.parametrize("counts", [[5], [5, 2]])
def test_cpu_reference_masks_the_padded_text_keys(monkeypatch, counts):
    """Each row equals dense SDPA over ``[image ‖ text[:count]]``: text keys at or beyond the
    count take no weight from any query, image or text."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import joint_ring_attention

    q, image_k, image_v, text_k, text_v = _inputs(len(counts))
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = joint_ring_attention(q, image_k, image_v, text_k, text_v, scale=scale,
                               text_valid_len=torch.tensor(counts, dtype=torch.int32))
    assert out.shape == q.shape
    for b, n in enumerate(counts):
        ref = F.scaled_dot_product_attention(
            q[b:b + 1], torch.cat([image_k[b:b + 1], text_k[b:b + 1, :, :n]], dim=2),
            torch.cat([image_v[b:b + 1], text_v[b:b + 1, :, :n]], dim=2), scale=scale)
        assert torch.allclose(out[b:b + 1], ref, atol=1e-5, rtol=1e-4)


def test_cpu_reference_without_a_count_attends_every_text_key(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import joint_ring_attention

    q, image_k, image_v, text_k, text_v = _inputs(1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    ref = F.scaled_dot_product_attention(q, torch.cat([image_k, text_k], dim=2),
                                         torch.cat([image_v, text_v], dim=2), scale=scale)
    out = joint_ring_attention(q, image_k, image_v, text_k, text_v, scale=scale,
                               text_valid_len=None)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_trainium_refuses_a_text_count():
    """The NxD ring has no text mask: a count is refused, as the mask was before."""
    pytest.importorskip("nkilib")
    from difflet.backends.trainium.ops_impl.attention import joint_ring_attention

    q, image_k, image_v, text_k, text_v = _inputs(1)
    with pytest.raises(NotImplementedError, match="text_valid_len"):
        joint_ring_attention(q, image_k, image_v, text_k, text_v, scale=0.25,
                             text_valid_len=torch.tensor([5], dtype=torch.int32))
