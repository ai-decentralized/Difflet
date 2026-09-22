"""CPU forward-path coverage for HunyuanVideo modeling layers.

These tests run the real ``nn.Module`` forwards on the CPU op backend with tiny
dimensions. They are shape/finiteness checks (not diffusers-parity, which is
covered on device); the goal is to exercise the layer/block/model forward code
paths that the AST-only ``test_modeling_hunyuan_video.py`` cannot reach.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Force the CPU op backend for the whole module and restore afterwards."""
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    import importlib

    import difflet.ops as ops

    importlib.reload(ops)
    import difflet.models.hunyuan_video.modeling_hunyuan_video as modeling

    importlib.reload(modeling)
    yield modeling
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _rotary(modeling, seq, head_dim):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    return get_1d_rotary_pos_embed(
        head_dim,
        torch.arange(seq, dtype=torch.float32),
        theta=256.0,
        use_real=True,
    )


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #
def test_config_inner_dim_and_from_diffusers_dict(_cpu_backend):
    m = _cpu_backend
    cfg = m.HunyuanVideoTransformerConfig(num_attention_heads=3, attention_head_dim=4)
    assert cfg.inner_dim == 12

    raw = {
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "rope_axes_dim": [2, 2, 4],
        "unknown_field": 99,
    }
    parsed = m.HunyuanVideoTransformerConfig.from_diffusers_dict(raw)
    assert parsed.num_attention_heads == 2
    assert parsed.rope_axes_dim == (2, 2, 4)
    assert not hasattr(parsed, "unknown_field")


def test_config_rejects_unsupported_qk_norm(_cpu_backend):
    m = _cpu_backend
    with pytest.raises(NotImplementedError, match="rms_norm"):
        m.HunyuanVideoTransformerConfig(qk_norm="layer_norm")


def test_config_rejects_bad_image_condition_type(_cpu_backend):
    m = _cpu_backend
    with pytest.raises(ValueError, match="image_condition_type"):
        m.HunyuanVideoTransformerConfig(image_condition_type="bogus")


def test_config_token_replace_not_implemented(_cpu_backend):
    m = _cpu_backend
    with pytest.raises(NotImplementedError, match="token_replace"):
        m.HunyuanVideoTransformerConfig(image_condition_type="token_replace")


def test_config_latent_concat_is_allowed(_cpu_backend):
    m = _cpu_backend
    cfg = m.HunyuanVideoTransformerConfig(image_condition_type="latent_concat")
    assert cfg.image_condition_type == "latent_concat"


# --------------------------------------------------------------------------- #
# Small leaf modules
# --------------------------------------------------------------------------- #
def test_patch_embed_int_patch_size(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    pe = m.HunyuanVideoPatchEmbed(patch_size=2, in_chans=2, embed_dim=8)
    out = pe(torch.randn(1, 2, 2, 4, 4))
    # (frames/2)*(h/2)*(w/2) = 1*2*2 = 4 tokens, embed dim 8
    assert out.shape == (1, 4, 8)


def test_ada_norm_forward(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    ada = m.HunyuanVideoAdaNorm(8)
    gate_msa, gate_mlp = ada(torch.randn(2, 8))
    assert gate_msa.shape == (2, 1, 8)
    assert gate_mlp.shape == (2, 1, 8)


def test_gelu_and_linear_activation_feedforwards(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    ff_gelu = m.HunyuanVideoFeedForward(8, mult=2.0, activation_fn="gelu-approximate")
    assert ff_gelu(x).shape == (2, 3, 8)
    ff_silu = m.HunyuanVideoFeedForward(8, mult=2.0, activation_fn="linear-silu")
    assert ff_silu(x).shape == (2, 3, 8)


def test_feedforward_rejects_unknown_activation(_cpu_backend):
    m = _cpu_backend
    with pytest.raises(NotImplementedError, match="activation"):
        m.HunyuanVideoFeedForward(8, activation_fn="relu")


def test_self_attention_forward(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    sa = m.HunyuanVideoSelfAttention(query_dim=8, heads=2, dim_head=4)
    out = sa(torch.randn(2, 3, 8))
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()


def test_self_attention_rejects_encoder_hidden_states(_cpu_backend):
    m = _cpu_backend
    sa = m.HunyuanVideoSelfAttention(query_dim=8, heads=2, dim_head=4)
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        sa(torch.randn(2, 3, 8), encoder_hidden_states=torch.randn(2, 3, 8))


# --------------------------------------------------------------------------- #
# Joint attention + transformer blocks (no mask -> CPU plain attention path)
# --------------------------------------------------------------------------- #
def test_joint_attention_dual_stream(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    heads, hd = 2, 4
    attn = m.HunyuanVideoAttention(
        hidden_size=heads * hd,
        num_attention_heads=heads,
        attention_head_dim=hd,
        added_kv_proj_dim=heads * hd,
        context_pre_only=False,
        pre_only=False,
    ).eval()
    hs = torch.randn(2, 3, heads * hd)
    enc = torch.randn(2, 2, heads * hd)
    rot = _rotary(m, 3, hd)
    with torch.no_grad():
        out, ctx = attn(hidden_states=hs, encoder_hidden_states=enc, image_rotary_emb=rot)
    assert out.shape == (2, 3, heads * hd)
    assert ctx.shape == (2, 2, heads * hd)


def test_joint_attention_pre_only_requires_encoder(_cpu_backend):
    m = _cpu_backend
    attn = m.HunyuanVideoAttention(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=4,
        added_kv_proj_dim=None,
        context_pre_only=None,
        pre_only=True,
    ).eval()
    with pytest.raises(ValueError, match="dual-stream"):
        attn(hidden_states=torch.randn(2, 3, 8), encoder_hidden_states=None)


def test_transformer_block_forward(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    heads, hd = 2, 4
    blk = m.HunyuanVideoTransformerBlock(heads, hd, mlp_ratio=2.0).eval()
    hs = torch.randn(2, 3, heads * hd)
    enc = torch.randn(2, 2, heads * hd)
    temb = torch.randn(2, heads * hd)
    rot = _rotary(m, 3, hd)
    with torch.no_grad():
        out, ctx = blk(hs, enc, temb, None, rot)
    assert out.shape == hs.shape
    assert ctx.shape == enc.shape
    assert torch.isfinite(out).all() and torch.isfinite(ctx).all()


def test_single_transformer_block_forward(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    heads, hd = 2, 4
    blk = m.HunyuanVideoSingleTransformerBlock(heads, hd, mlp_ratio=2.0).eval()
    hs = torch.randn(2, 3, heads * hd)
    enc = torch.randn(2, 2, heads * hd)
    temb = torch.randn(2, heads * hd)
    rot = _rotary(m, 3, hd)
    with torch.no_grad():
        out, ctx = blk(hs, enc, temb, None, rot)
    assert out.shape == hs.shape
    assert ctx.shape == enc.shape


# --------------------------------------------------------------------------- #
# Condition embedding / token refiner / rope
# --------------------------------------------------------------------------- #
def test_condition_embedding_with_guidance(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    ce = m.HunyuanVideoConditionEmbedding(8, 5, guidance_embeds=True).eval()
    with torch.no_grad():
        cond, token_replace = ce(torch.tensor([3, 4]), torch.randn(2, 5), torch.tensor([1, 2]))
    assert cond.shape == (2, 8)
    assert token_replace is None


def test_condition_embedding_without_guidance(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    ce = m.HunyuanVideoConditionEmbedding(8, 5, guidance_embeds=False).eval()
    with torch.no_grad():
        cond, token_replace = ce(torch.tensor([3, 4]), torch.randn(2, 5))
    assert cond.shape == (2, 8)
    assert token_replace is None


def test_condition_embedding_requires_guidance_value(_cpu_backend):
    m = _cpu_backend
    ce = m.HunyuanVideoConditionEmbedding(8, 5, guidance_embeds=True).eval()
    with pytest.raises(ValueError, match="guidance"):
        ce(torch.tensor([3]), torch.randn(1, 5), guidance=None)


def test_condition_embedding_token_replace_branch(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    ce = m.HunyuanVideoConditionEmbedding(
        8, 5, guidance_embeds=False, image_condition_type="token_replace"
    ).eval()
    with torch.no_grad():
        cond, token_replace = ce(torch.tensor([3, 4]), torch.randn(2, 5))
    assert cond.shape == (2, 8)
    assert token_replace is not None
    assert token_replace.shape == (2, 8)


def test_individual_refiner_block_custom_drop_rate(_cpu_backend):
    m = _cpu_backend
    block = m.HunyuanVideoIndividualTokenRefinerBlock(2, 4, mlp_drop_rate=0.1)
    assert isinstance(block.ff.net[1], torch.nn.Dropout)
    assert block.ff.net[1].p == 0.1


def test_attention_added_kv_without_context_query(_cpu_backend):
    """added_kv_proj_dim set with context_pre_only=None -> add_q_proj is None.

    The joint-concat path normalises latent+context together while folding the
    extra add_k/add_v projections into the norm_added_* layers.
    """
    m = _cpu_backend
    torch.manual_seed(0)
    attn = m.HunyuanVideoAttention(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=4,
        added_kv_proj_dim=8,
        context_pre_only=None,
        pre_only=True,
    ).eval()
    assert attn.add_q_proj is None
    assert attn.norm_added_q is not None
    rot = _rotary(m, 3, 4)
    with torch.no_grad():
        out, ctx = attn(
            hidden_states=torch.randn(2, 3, 8),
            encoder_hidden_states=torch.randn(2, 2, 8),
            image_rotary_emb=rot,
        )
    assert out.shape == (2, 3, 8)
    assert ctx.shape == (2, 2, 8)


def test_token_refiner_forward_with_and_without_mask(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    heads, hd = 2, 4
    refiner = m.HunyuanVideoTokenRefiner(6, heads, hd, num_layers=1).eval()
    hs = torch.randn(2, 4, 6)
    ts = torch.tensor([3, 4])
    with torch.no_grad():
        out_no_mask = refiner(hs, ts)
    assert out_no_mask.shape == (2, 4, heads * hd)

    mask = torch.ones(2, 4, dtype=torch.long)
    mask[1, -1] = 0
    with torch.no_grad():
        out_mask = refiner(hs, ts, mask)
    assert out_mask.shape == (2, 4, heads * hd)
    assert torch.isfinite(out_mask).all()


def test_rotary_pos_embed_forward(_cpu_backend):
    m = _cpu_backend
    rope = m.HunyuanVideoRotaryPosEmbed(patch_size=2, patch_size_t=1, rope_dim=(2, 2, 2))
    cos, sin = rope(torch.randn(1, 2, 2, 4, 4))
    # 2 * 2 * 2 tokens, sum(rope_dim) = 6 channels
    assert cos.shape == (8, 6)
    assert sin.shape == (8, 6)


# --------------------------------------------------------------------------- #
# Full model forward + teacache hook
# --------------------------------------------------------------------------- #
def _tiny_model(m):
    torch.manual_seed(0)
    return m.HunyuanVideoTransformer3DModel(
        in_channels=2,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=2,
        patch_size_t=1,
        guidance_embeds=True,
        text_embed_dim=6,
        pooled_projection_dim=5,
        rope_axes_dim=(2, 2, 0),
    ).eval()


def _tiny_inputs():
    torch.manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, 2, 2, 4, 4),
        timestep=torch.tensor([7], dtype=torch.long),
        encoder_hidden_states=torch.randn(1, 4, 6),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        pooled_projections=torch.randn(1, 5),
        guidance=torch.tensor([3], dtype=torch.long),
    )


def test_model_construction_from_config_object(_cpu_backend):
    m = _cpu_backend
    cfg = m.HunyuanVideoTransformerConfig(
        in_channels=2,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        text_embed_dim=6,
        pooled_projection_dim=5,
        rope_axes_dim=(2, 2, 0),
    )
    model = m.HunyuanVideoTransformer3DModel(cfg)
    assert model.config.in_channels == 2
    assert len(model.transformer_blocks) == 1
    assert len(model.single_transformer_blocks) == 1


def test_model_full_forward_return_dict_and_tuple(_cpu_backend):
    m = _cpu_backend
    model = _tiny_model(m)
    inputs = _tiny_inputs()
    with torch.no_grad():
        out_dict = model(**inputs, return_dict=True)
        out_tuple = model(**inputs, return_dict=False)
    assert out_dict.sample.shape == (1, 2, 2, 4, 4)
    assert out_tuple[0].shape == (1, 2, 2, 4, 4)
    assert torch.isfinite(out_dict.sample).all()


def test_model_teacache_mod_input(_cpu_backend):
    m = _cpu_backend
    model = _tiny_model(m)
    inputs = _tiny_inputs()
    with torch.no_grad():
        mod = model.teacache_mod_input(
            inputs["hidden_states"],
            inputs["timestep"],
            inputs["encoder_hidden_states"],
            inputs["encoder_attention_mask"],
            inputs["pooled_projections"],
            inputs["guidance"],
        )
    # 4 latent tokens, inner_dim = 2*4 = 8
    assert mod.shape == (1, 8, 8)


# --------------------------------------------------------------------------- #
# dual_stream_attention + module-level helpers
# --------------------------------------------------------------------------- #
def test_dual_stream_attention_unmasked_and_masked(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    b, ls, cs, h, d = 2, 3, 2, 2, 4
    lq = torch.randn(b, ls, h, d)
    lk = torch.randn(b, ls, h, d)
    lv = torch.randn(b, ls, h, d)
    cq = torch.randn(b, cs, h, d)
    ck = torch.randn(b, cs, h, d)
    cv = torch.randn(b, cs, h, d)
    latent, context = m.dual_stream_attention(lq, lk, lv, cq, ck, cv)
    assert latent.shape == (b, ls, h, d)
    assert context.shape == (b, cs, h, d)

    mask = torch.ones(b, 1, 1, ls + cs, dtype=torch.bool)
    mask[1, :, :, -1] = False
    latent_m, context_m = m.dual_stream_attention(lq, lk, lv, cq, ck, cv, attention_mask=mask)
    assert torch.isfinite(latent_m).all() and torch.isfinite(context_m).all()


def test_dual_stream_attention_validates_shapes(_cpu_backend):
    m = _cpu_backend
    good = torch.randn(2, 3, 2, 4)
    ctx = torch.randn(2, 2, 2, 4)
    # latent K/V mismatch
    with pytest.raises(ValueError, match="latent K and V"):
        m.dual_stream_attention(good, good, torch.randn(2, 4, 2, 4), ctx, ctx, ctx)
    # latent Q not 4D
    with pytest.raises(ValueError, match="must be 4D"):
        m.dual_stream_attention(torch.randn(2, 3, 8), good, good, ctx, ctx, ctx)
    # latent vs context head/dim mismatch
    bad_ctx = torch.randn(2, 2, 4, 4)
    with pytest.raises(ValueError, match="batch, heads"):
        m.dual_stream_attention(good, good, good, bad_ctx, bad_ctx, bad_ctx)


def test_check_stream_shapes_errors(_cpu_backend):
    m = _cpu_backend
    q = torch.randn(2, 3, 2, 4)
    with pytest.raises(ValueError, match="must be 4D"):
        m._check_stream_shapes(torch.randn(2, 3, 8), q, q, name="context")
    with pytest.raises(ValueError, match="shapes must match"):
        m._check_stream_shapes(q, q, torch.randn(2, 4, 2, 4), name="context")


def test_flatten_attention_mask(_cpu_backend):
    m = _cpu_backend
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    flat = m._flatten_attention_mask(mask, batch=2, heads=3)
    assert flat.shape == (6, 1, 5)
    assert m._flatten_attention_mask(None, batch=2, heads=3) is None
    with pytest.raises(ValueError, match="diffusers HunyuanVideo mask"):
        m._flatten_attention_mask(torch.ones(3, 5), batch=2, heads=3)


def test_apply_hunyuan_rotary_expands_2d(_cpu_backend):
    m = _cpu_backend
    torch.manual_seed(0)
    hs = torch.randn(1, 4, 2, 6)
    cos = torch.randn(4, 6)
    sin = torch.randn(4, 6)
    out = m._apply_hunyuan_rotary(hs, (cos, sin))
    assert out.shape == hs.shape


def test_keypad_bounds_from_mask(_cpu_backend):
    m = _cpu_backend
    mask = torch.tensor([[[True, True, False]]])  # (1, 1, 3)
    bound_min, bound_max = m._keypad_bounds_from_mask(mask, q_len=2)
    assert bound_min.shape == (1, 2, 1)
    assert bound_max.shape == (1, 2, 1)
    assert torch.equal(bound_min, torch.zeros(1, 2, 1, dtype=torch.int32))
    assert int(bound_max[0, 0, 0]) == 2


@pytest.mark.parametrize("key_len", [10496, 20096])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_keypad_bounds_long_prefixes(_cpu_backend, key_len, dtype):
    m = _cpu_backend
    counts = torch.tensor([0, 18, key_len - 238, key_len])
    mask = (torch.arange(key_len)[None, None, :] < counts[:, None, None]).to(dtype)
    lo, hi = m._keypad_bounds_from_mask(mask, q_len=3)
    assert lo.dtype == hi.dtype == torch.int32
    assert torch.equal(lo, torch.zeros(4, 3, 1, dtype=torch.int32))
    assert torch.equal(hi, counts.to(torch.int32)[:, None, None].expand(4, 3, 1))


# --------------------------------------------------------------------------- #
# Ulysses CP + key-padding mask (cp == 1 on the CPU backend, so the all-to-alls
# are identities and the ulysses branch must equal the masked gather_kv branch)
# --------------------------------------------------------------------------- #
def _hv_attention_pair(m, *, pre_only: bool):
    heads, hd = 2, 4
    kw = dict(hidden_size=heads * hd, num_attention_heads=heads, attention_head_dim=hd,
              added_kv_proj_dim=heads * hd,
              context_pre_only=None if pre_only else False, pre_only=pre_only)
    torch.manual_seed(3)
    base = m.HunyuanVideoAttention(**kw).eval()                       # gather_kv, no CP
    cp = m.HunyuanVideoAttention(**kw, context_parallel_enabled=True,
                                 cp_mode="ulysses").eval()
    cp.load_state_dict(base.state_dict())
    return base, cp


@pytest.mark.parametrize("pre_only", [False, True], ids=["dual_stream", "single_stream"])
def test_attention_ulysses_honours_key_padding_mask(_cpu_backend, pre_only):
    """Under ulysses a padded text mask used to raise NotImplementedError; it now
    routes through the joint valid-key count and must match the masked
    dual_stream_attention (gather_kv) result on the same weights and inputs."""
    m = _cpu_backend
    base, cp = _hv_attention_pair(m, pre_only=pre_only)
    torch.manual_seed(4)
    b, ls, cs = 2, 6, 4
    hs = torch.randn(b, ls, 8)
    enc = torch.randn(b, cs, 8)
    rot = _rotary(m, ls, 4)
    mask = torch.ones(b, 1, 1, ls + cs, dtype=torch.bool)
    mask[1, :, :, ls + 1:] = False        # row 1: only 1 of 4 text tokens is valid
    with torch.no_grad():
        ref_h, ref_c = base(hidden_states=hs, encoder_hidden_states=enc,
                            attention_mask=mask, image_rotary_emb=rot)
        got_h, got_c = cp(hidden_states=hs, encoder_hidden_states=enc,
                          attention_mask=mask, image_rotary_emb=rot)
        # the mask must matter: unmasked ulysses differs on the padded row
        un_h, _ = cp(hidden_states=hs, encoder_hidden_states=enc, image_rotary_emb=rot)
    assert torch.allclose(got_h, ref_h, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got_c, ref_c, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(un_h[1], got_h[1], atol=1e-5)
    assert torch.allclose(un_h[0], got_h[0], atol=1e-5)


def test_attention_ulysses_key_valid_len_is_the_mask_row_sum(_cpu_backend, monkeypatch):
    """The count handed to the op is the trace-safe row sum of the joint mask."""
    m = _cpu_backend
    _, cp = _hv_attention_pair(m, pre_only=False)
    seen = {}
    real = m.joint_ulysses_attention

    def spy(*a, **k):
        seen["key_valid_len"] = k.get("key_valid_len")
        return real(*a, **k)

    monkeypatch.setattr(m, "joint_ulysses_attention", spy)
    b, ls, cs = 2, 6, 4
    mask = torch.ones(b, 1, 1, ls + cs, dtype=torch.bool)
    mask[1, :, :, ls + 1:] = False
    with torch.no_grad():
        cp(hidden_states=torch.randn(b, ls, 8), encoder_hidden_states=torch.randn(b, cs, 8),
           attention_mask=mask)
    assert seen["key_valid_len"].dtype == torch.int32
    assert seen["key_valid_len"].tolist() == [ls + cs, ls + 1]
    with torch.no_grad():
        cp(hidden_states=torch.randn(b, ls, 8), encoder_hidden_states=torch.randn(b, cs, 8))
    assert seen["key_valid_len"] is None
