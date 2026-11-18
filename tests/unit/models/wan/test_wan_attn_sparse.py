"""Experiment E2's routing in modeling_wan (DIFFLET_EXP_ATTN_SPARSE): on the CPU op backend, only
the self-attention (attn1) goes to ``block_sparse_attention``, with the call's grid in its spec;
unset, nothing changes."""

from __future__ import annotations

import importlib
import os

import pytest
import torch

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.modeling_wan as wan  # noqa: E402

importlib.reload(wan)

if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND

ENV = "DIFFLET_EXP_ATTN_SPARSE"


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(wan, "_block_sparse_attention", None)  # restored after each test


def _config(**overrides):
    return wan.WanTransformerConfig(**{**dict(patch_size=(1, 2, 2), num_attention_heads=4,
                                              attention_head_dim=16, in_channels=4, out_channels=4,
                                              text_dim=24, freq_dim=32, ffn_dim=48, num_layers=2),
                                       **overrides})


def _inputs():
    g = torch.Generator().manual_seed(0)
    # 4 frames of 16 x 16 tokens: 1,024 tokens, 8 query groups, 2 key tiles of 2 frames each
    return (torch.randn(1, 4, 4, 32, 32, generator=g), torch.tensor([3.0]),
            torch.randn(1, 5, 24, generator=g))


def _model(monkeypatch, spec=None, state=None):
    if spec is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, spec)
    torch.manual_seed(0)
    model = wan.WanTransformer3DModel(_config())
    if state is not None:
        model.load_state_dict(state)
    return model


def _spy(monkeypatch):
    calls, real = [], wan._block_sparse_attention

    def spy(q, k, v, *, scale, kv_mask):
        calls.append((tuple(q.shape), kv_mask))
        return real(q, k, v, scale=scale, kv_mask=kv_mask)

    monkeypatch.setattr(wan, "_block_sparse_attention", spy)
    return calls


def test_unset_routes_nothing(monkeypatch):
    model = _model(monkeypatch)
    assert model.attn_sparse is None and wan._block_sparse_attention is None
    model(*_inputs())
    assert all(b.attn1.kv_mask is None and b.attn2.kv_mask is None for b in model.blocks)


def test_set_routes_only_the_self_attention_with_the_calls_grid(monkeypatch):
    dense = _model(monkeypatch)
    sparse = _model(monkeypatch, "temporal:0", dense.state_dict())
    calls = _spy(monkeypatch)
    out = sparse(*_inputs())
    assert calls == [((4, 1024, 16), "temporal:0@4x16x16")] * 2  # attn1 of both blocks only
    assert all(b.attn1.kv_mask == "temporal:0@4x16x16" and b.attn2.kv_mask is None
               for b in sparse.blocks)
    assert not torch.allclose(out, dense(*_inputs()), atol=1e-3)  # half the tiles skipped


def test_a_window_over_every_frame_is_dense_attention(monkeypatch):
    dense = _model(monkeypatch)
    full = _model(monkeypatch, "temporal:3", dense.state_dict())
    torch.testing.assert_close(full(*_inputs()), dense(*_inputs()), rtol=1e-4, atol=1e-5)


def test_context_parallel_is_refused(monkeypatch):
    monkeypatch.setenv(ENV, "temporal:1")
    cfg = _config()
    cfg.context_parallel_enabled = True
    with pytest.raises(ValueError, match="context parallelism"):
        wan.WanTransformer3DModel(cfg)


def test_a_grid_off_the_call_is_refused(monkeypatch):
    model = _model(monkeypatch, "temporal:0")
    for block in model.blocks:
        block.attn1.kv_mask = "temporal:0@4x16x15"
    with pytest.raises(ValueError, match="960 tokens, the attention call 1024"):
        model.blocks[0].attn1(torch.randn(1, 1024, 64),
                              rotary_emb=model.rope(_inputs()[0], "interleaved"))
