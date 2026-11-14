"""LTX-2 prompt encoding over the real tokens (backends/tpu/ltx_2/text_encode.py)
against the stock padded computation, on tiny random models."""

from __future__ import annotations

import types

import pytest
import torch

from difflet.backends.tpu.ltx_2 import text_encode as te

L = 16


def _gemma():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=L,
        max_position_embeddings=64, layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    return Gemma3ForCausalLM(cfg).eval().double()


def _connectors():
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    torch.manual_seed(1)
    return LTX2TextConnectors(
        caption_channels=32, text_proj_in_factor=4, video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=16, video_connector_num_layers=1,
        video_connector_num_learnable_registers=4, audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=16, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=4, rope_type="split",
    ).eval().double()


def _left_padded(lengths):
    ids = torch.zeros(len(lengths), L, dtype=torch.int64)
    mask = torch.zeros(len(lengths), L, dtype=torch.int64)
    g = torch.Generator().manual_seed(2)
    for row, n in enumerate(lengths):
        ids[row, L - n:] = torch.randint(3, 64, (n,), generator=g)
        mask[row, L - n:] = 1
    return ids, mask


def test_gemma_on_real_tokens_matches_the_padded_call():
    model = _gemma()
    ids, mask = _left_padded([5, 9])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    stock = torch.stack(out.hidden_states, dim=-1).flatten(2, 3)
    trimmed = te.gemma_prompt_embeds(model, ids, mask)
    valid = mask.bool()
    assert trimmed.shape == stock.shape
    torch.testing.assert_close(trimmed[valid], stock[valid], rtol=1e-10, atol=1e-10)
    assert not trimmed[~valid].any()


def test_connectors_on_real_rows_match_the_stock_forward():
    conn = _connectors()
    ids, mask = _left_padded([5, 9])
    hidden = torch.randn(2, L, 32 * 4, dtype=torch.float64)
    garbage = hidden.clone()
    garbage[~mask.bool()] = 1e3 * torch.randn(int((~mask.bool()).sum()), 32 * 4, dtype=torch.float64)
    with torch.no_grad():
        stock = conn(garbage, mask, padding_side="left")
    zeroed = hidden.clone()
    zeroed[~mask.bool()] = 0
    ours = te.connectors_forward(conn, zeroed, mask)
    for a, b in zip(ours, stock):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_right_padding_is_refused():
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    with pytest.raises(ValueError, match="left-padded"):
        te.real_token_window(mask)


class _Ring:
    """Two simulated ranks: rank 0's broadcast payload is what rank 1 receives."""

    def __init__(self):
        self.sent = None

    def broadcast(self, tensors, src=0):
        if self.sent is None:
            self.sent = [t.clone() for t in tensors]
        else:
            for t, s in zip(tensors, self.sent):
                t.copy_(s)


def _pipe(is_encoder, monkeypatch, ring, gemma=None):
    from difflet.backends.tpu.ops_impl import device_runtime
    from difflet.models.ltx_2 import tpu_application as mod

    monkeypatch.setattr(device_runtime, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(device_runtime, "broadcast", ring.broadcast)

    class Tok:
        padding_side, pad_token, eos_token = "left", "<pad>", "<eos>"

        def __call__(self, prompts, max_length, **_):
            lengths = [len(p.split()) + 1 for p in prompts]
            ids, mask = _left_padded(lengths)
            return types.SimpleNamespace(input_ids=ids[:, :max_length], attention_mask=mask[:, :max_length])

    conn = _connectors().float()
    pipe = types.SimpleNamespace(tokenizer=Tok(), text_encoder=gemma, connectors=conn,
                                 _get_gemma_prompt_embeds=None)
    mod._install_rank0_text_conditioning(pipe, is_encoder=is_encoder)
    return pipe


def _condition(pipe, prompts):
    embeds, mask = pipe._get_gemma_prompt_embeds(prompts, max_sequence_length=L, device="cpu",
                                                 dtype=torch.bfloat16)
    return pipe.connectors(embeds, mask, padding_side="left")


def test_rank0_conditioning_reaches_every_rank(monkeypatch):
    ring = _Ring()
    gemma = _gemma().float()
    rank0 = _condition(_pipe(True, monkeypatch, ring, gemma), ["a red fox", "snow"])
    rank1_pipe = _pipe(False, monkeypatch, ring)
    embeds, _ = rank1_pipe._get_gemma_prompt_embeds(["a red fox", "snow"], max_sequence_length=L)
    assert embeds.shape == (2, L, 0)  # no Gemma off rank 0
    rank1 = rank1_pipe.connectors(embeds, torch.zeros(2, L, dtype=torch.int64))
    for a, b in zip(rank0, rank1):
        assert torch.equal(a, b)
    assert rank0[0].shape == (2, L, 32) and rank0[2].dtype is torch.int64


def test_rank0_failure_raises_on_every_rank(monkeypatch):
    ring = _Ring()

    def broken(**_):
        raise OSError("disk")

    with pytest.raises(RuntimeError, match="encoder rank") as info:
        _condition(_pipe(True, monkeypatch, ring, gemma=broken), ["a fox"])
    assert isinstance(info.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="encoder rank"):
        _condition(_pipe(False, monkeypatch, ring), ["a fox"])
