import math
import os

import torch


def test_joint_ring_attention_cpu_matches_full_joint_attention():
    os.environ["DIFFLET_BACKEND"] = "cpu"
    from difflet.ops import attention, joint_ring_attention

    torch.manual_seed(0)
    b, h, s_img, s_txt, d = 1, 2, 128, 64, 64
    q = torch.randn(b, h, s_img + s_txt, d)
    image_k = torch.randn(b, h, s_img, d)
    image_v = torch.randn(b, h, s_img, d)
    text_k = torch.randn(b, h, s_txt, d)
    text_v = torch.randn(b, h, s_txt, d)
    scale = 1.0 / math.sqrt(d)

    full_k = torch.cat([image_k, text_k], dim=2)
    full_v = torch.cat([image_v, text_v], dim=2)
    ref = attention(
        q.reshape(b * h, s_img + s_txt, d),
        full_k.reshape(b * h, s_img + s_txt, d),
        full_v.reshape(b * h, s_img + s_txt, d),
        scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
    ).reshape(b, h, s_img + s_txt, d)

    out = joint_ring_attention(q, image_k, image_v, text_k, text_v, scale=scale, causal=False)
    assert out.shape == (b, h, s_img + s_txt, d)
    assert torch.allclose(ref.float(), out.float(), atol=1e-4, rtol=1e-4)
