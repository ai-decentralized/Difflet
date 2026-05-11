import math

import pytest
import torch


def test_hunyuan_video_transformer3d_model_matches_diffusers_tiny(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel as DiffusersHunyuanVideoTransformer3DModel,
    )

    from nova.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformer3DModel

    kwargs = {
        "in_channels": 2,
        "out_channels": 2,
        "num_attention_heads": 2,
        "attention_head_dim": 6,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_refiner_layers": 1,
        "mlp_ratio": 2.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "guidance_embeds": True,
        "text_embed_dim": 6,
        "pooled_projection_dim": 5,
        "rope_axes_dim": (2, 2, 2),
    }

    torch.manual_seed(20)
    ref = DiffusersHunyuanVideoTransformer3DModel(**kwargs).eval()
    actual = HunyuanVideoTransformer3DModel(**kwargs).eval()
    assert set(actual.state_dict()) == set(ref.state_dict())
    actual.load_state_dict(ref.state_dict())

    torch.manual_seed(21)
    hidden_states = torch.randn(1, 2, 2, 4, 4)
    timestep = torch.tensor([7], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 4, 6)
    encoder_attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    pooled_projections = torch.randn(1, 5)
    guidance = torch.tensor([3], dtype=torch.long)

    with torch.no_grad():
        ref_out = ref(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            return_dict=False,
        )[0]
        actual_out = actual(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            return_dict=False,
        )[0]

    assert torch.allclose(actual_out, ref_out, atol=1e-6, rtol=1e-6)


def test_hunyuan_video_transformer3d_model_matches_diffusers_production_heads(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel as DiffusersHunyuanVideoTransformer3DModel,
    )

    from nova.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformer3DModel

    kwargs = {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 1,
        "num_single_layers": 1,
        "num_refiner_layers": 1,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_axes_dim": (16, 56, 56),
    }

    torch.manual_seed(30)
    ref = DiffusersHunyuanVideoTransformer3DModel(**kwargs).eval()
    actual = HunyuanVideoTransformer3DModel(**kwargs).eval()
    assert set(actual.state_dict()) == set(ref.state_dict())
    actual.load_state_dict(ref.state_dict())

    torch.manual_seed(31)
    hidden_states = torch.randn(1, 16, 2, 8, 8)
    timestep = torch.tensor([17], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 16, 4096)
    encoder_attention_mask = torch.ones(1, 16, dtype=torch.long)
    encoder_attention_mask[:, -3:] = 0
    pooled_projections = torch.randn(1, 768)
    guidance = torch.tensor([5], dtype=torch.long)

    with torch.no_grad():
        ref_out = ref(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            return_dict=False,
        )[0]
        actual_out = actual(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            return_dict=False,
        )[0]

    assert torch.allclose(actual_out, ref_out, atol=1e-5, rtol=1e-4)


def test_hunyuan_video_transformer3d_token_replace_is_explicitly_out_of_scope(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from nova.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformer3DModel

    with pytest.raises(NotImplementedError, match="token_replace"):
        HunyuanVideoTransformer3DModel(image_condition_type="token_replace")


def test_hunyuan_video_transformer_block_matches_diffusers(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformerBlock as DiffusersHunyuanVideoTransformerBlock,
    )

    from nova.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformerBlock

    torch.manual_seed(10)
    batch, latent_seq, context_seq, heads, head_dim = 2, 3, 2, 2, 4
    hidden_size = heads * head_dim
    hidden_states = torch.randn(batch, latent_seq, hidden_size)
    encoder_hidden_states = torch.randn(batch, context_seq, hidden_size)
    temb = torch.randn(batch, hidden_size)
    mask = torch.ones(batch, 1, 1, latent_seq + context_seq, dtype=torch.bool)
    mask[1, :, :, -1] = False
    rotary = get_1d_rotary_pos_embed(
        head_dim,
        torch.arange(latent_seq, dtype=torch.float32),
        theta=256.0,
        use_real=True,
    )

    torch.manual_seed(11)
    ref = DiffusersHunyuanVideoTransformerBlock(heads, head_dim, mlp_ratio=2.0).eval()
    actual = HunyuanVideoTransformerBlock(heads, head_dim, mlp_ratio=2.0).eval()
    assert set(actual.state_dict()) == set(ref.state_dict())
    actual.load_state_dict(ref.state_dict())

    with torch.no_grad():
        ref_out = ref(hidden_states, encoder_hidden_states, temb, mask, rotary)
        actual_out = actual(hidden_states, encoder_hidden_states, temb, mask, rotary)

    for ref_tensor, actual_tensor in zip(ref_out, actual_out):
        assert torch.allclose(actual_tensor, ref_tensor, atol=1e-6, rtol=1e-6)


def test_hunyuan_video_single_transformer_block_matches_diffusers(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock as DiffusersHunyuanVideoSingleTransformerBlock,
    )

    from nova.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoSingleTransformerBlock

    torch.manual_seed(12)
    batch, latent_seq, context_seq, heads, head_dim = 2, 3, 2, 2, 4
    hidden_size = heads * head_dim
    hidden_states = torch.randn(batch, latent_seq, hidden_size)
    encoder_hidden_states = torch.randn(batch, context_seq, hidden_size)
    temb = torch.randn(batch, hidden_size)
    mask = torch.ones(batch, 1, 1, latent_seq + context_seq, dtype=torch.bool)
    mask[1, :, :, -1] = False
    rotary = get_1d_rotary_pos_embed(
        head_dim,
        torch.arange(latent_seq, dtype=torch.float32),
        theta=256.0,
        use_real=True,
    )

    torch.manual_seed(13)
    ref = DiffusersHunyuanVideoSingleTransformerBlock(heads, head_dim, mlp_ratio=2.0).eval()
    actual = HunyuanVideoSingleTransformerBlock(heads, head_dim, mlp_ratio=2.0).eval()
    assert set(actual.state_dict()) == set(ref.state_dict())
    actual.load_state_dict(ref.state_dict())

    with torch.no_grad():
        ref_out = ref(hidden_states, encoder_hidden_states, temb, mask, rotary)
        actual_out = actual(hidden_states, encoder_hidden_states, temb, mask, rotary)

    for ref_tensor, actual_tensor in zip(ref_out, actual_out):
        assert torch.allclose(actual_tensor, ref_tensor, atol=1e-6, rtol=1e-6)


def test_dual_stream_attention_matches_concat_attention_with_mask(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from nova.models.hunyuan_video.modeling_hunyuan_video import dual_stream_attention

    torch.manual_seed(0)
    batch, latent_seq, context_seq, heads, head_dim = 2, 3, 2, 4, 5
    latent_q = torch.randn(batch, latent_seq, heads, head_dim)
    latent_k = torch.randn(batch, latent_seq, heads, head_dim)
    latent_v = torch.randn(batch, latent_seq, heads, head_dim)
    context_q = torch.randn(batch, context_seq, heads, head_dim)
    context_k = torch.randn(batch, context_seq, heads, head_dim)
    context_v = torch.randn(batch, context_seq, heads, head_dim)
    mask = torch.ones(batch, 1, 1, latent_seq + context_seq, dtype=torch.bool)
    mask[1, :, :, -1] = False

    latent_out, context_out = dual_stream_attention(
        latent_q,
        latent_k,
        latent_v,
        context_q,
        context_k,
        context_v,
        attention_mask=mask,
    )

    expected = _reference_concat_attention(
        torch.cat([latent_q, context_q], dim=1),
        torch.cat([latent_k, context_k], dim=1),
        torch.cat([latent_v, context_v], dim=1),
        mask,
    )
    assert torch.allclose(latent_out, expected[:, :latent_seq], atol=1e-6, rtol=1e-6)
    assert torch.allclose(context_out, expected[:, latent_seq:], atol=1e-6, rtol=1e-6)


def test_trainium_masked_attention_uses_sdpa_fallback(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import attention

    torch.manual_seed(1)
    q = torch.randn(2, 4, 3)
    k = torch.randn(2, 4, 3)
    v = torch.randn(2, 4, 3)
    mask = torch.ones(2, 1, 4, dtype=torch.bool)
    mask[1, :, -1] = False

    out = attention(q, k, v, scale=0.25, attention_mask=mask, tp_q=True, tp_k=True)

    expected = _reference_flat_attention(q, k, v, scale=0.25, mask=mask)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_trainium_masked_attention_sdpa_fallback_respects_cte_layout_flags(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import attention

    torch.manual_seed(2)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    mask[:, :, -1] = False

    out = attention(q, k, v, scale=0.5, attention_mask=mask, tp_out=True)

    q_ref = q.transpose(-1, -2)
    k_ref = k.transpose(-1, -2)
    expected = _reference_flat_attention(q_ref, k_ref, v, scale=0.5, mask=mask)
    assert torch.allclose(out.transpose(-1, -2), expected, atol=1e-6, rtol=1e-6)


def test_trainium_masked_attention_sdpa_fallback_combines_causal_mask(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import attention

    torch.manual_seed(3)
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    v = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mask[:, :, -1] = False
    causal = torch.ones(4, 4, dtype=torch.bool).tril()

    out = attention(q, k, v, scale=0.25, causal=True, attention_mask=mask, tp_q=True, tp_k=True)

    expected = _reference_flat_attention(q, k, v, scale=0.25, mask=mask & causal)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_trainium_masked_attention_sdpa_fallback_combines_additive_and_causal_mask(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import attention

    torch.manual_seed(4)
    q = torch.randn(1, 4, 3)
    k = torch.randn(1, 4, 3)
    v = torch.randn(1, 4, 3)
    keep = torch.ones(1, 4, 4, dtype=torch.bool)
    keep[:, :, -1] = False
    additive_mask = torch.zeros(1, 4, 4)
    additive_mask = additive_mask.masked_fill(~keep, torch.finfo(additive_mask.dtype).min)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()

    out = attention(
        q,
        k,
        v,
        scale=0.25,
        causal=True,
        attention_mask=additive_mask,
        tp_q=True,
        tp_k=True,
    )

    expected = _reference_flat_attention(q, k, v, scale=0.25, mask=keep & causal)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def _reference_concat_attention(q, k, v, mask):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.matmul(probs, v)
    return out.permute(0, 2, 1, 3)


def _reference_flat_attention(q, k, v, *, scale, mask):
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)
