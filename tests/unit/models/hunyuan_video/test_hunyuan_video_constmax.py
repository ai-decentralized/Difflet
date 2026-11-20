"""Experiment E1 (DIFFLET_EXP_ATTN_CONSTMAX=1), lane HV1: the joint attentions that pass the table's
gap rule carry their layer's sigma as a non-persistent [128, 1] buffer and pass it to the backend's
attention; the layers that fail it (``transformer_blocks.1.attn`` here) and the token refiner pass
nothing; with the env unset, no buffer and no kwarg anywhere. CPU backend, tiny model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("env", ["1", None])
def test_constmax_env_threads_the_table_sigma(monkeypatch, env):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    if env:
        monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", env)
    else:
        monkeypatch.delenv("DIFFLET_EXP_ATTN_CONSTMAX", raising=False)
    import difflet.models.hunyuan_video.modeling_hunyuan_video as m

    seen, real = [], m.attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("constmax_sigma", "absent"))
        return real(*args, **kwargs)

    monkeypatch.setattr(m, "attention", spy)
    torch.manual_seed(0)
    model = m.HunyuanVideoTransformer3DModel(
        in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=4, num_layers=2,
        num_single_layers=1, num_refiner_layers=1, mlp_ratio=2.0, text_embed_dim=6,
        pooled_projection_dim=5, rope_axes_dim=(2, 2, 0),
    ).eval()
    with torch.no_grad():
        out = model(
            hidden_states=torch.randn(1, 2, 2, 4, 4), timestep=torch.tensor([7]),
            encoder_hidden_states=torch.randn(1, 4, 6),
            encoder_attention_mask=torch.tensor([[1, 1, 1, 0]]),
            pooled_projections=torch.randn(1, 5), guidance=torch.tensor([3]), return_dict=False,
        )[0]
    assert torch.isfinite(out).all()
    attns = {"transformer_blocks.0.attn": model.transformer_blocks[0].attn,
             "transformer_blocks.1.attn": model.transformer_blocks[1].attn,
             "single_transformer_blocks.0.attn": model.single_transformer_blocks[0].attn}
    assert len(seen) == 4  # token refiner, dual 0, dual 1, single 0
    if not env:
        assert not any(hasattr(a, "constmax_sigma") for a in attns.values())
        assert seen == ["absent"] * 4
        return
    table = json.loads(Path(m.__file__).with_name("attn_constmax.json").read_text())["layers"]
    assert not table["transformer_blocks.1.attn"]["ok"]  # the allowlist's fallback
    assert not hasattr(attns["transformer_blocks.1.attn"], "constmax_sigma")
    for name in ("transformer_blocks.0.attn", "single_transformer_blocks.0.attn"):
        sigma = attns[name].constmax_sigma
        assert sigma.shape == (128, 1)
        assert torch.all(sigma == torch.tensor(table[name]["sigma"]))
        assert "constmax_sigma" not in attns[name].state_dict()  # not a checkpoint entry
    assert seen[0] == "absent" and seen[2] == "absent"
    assert seen[1] is attns["transformer_blocks.0.attn"].constmax_sigma
    assert seen[3] is attns["single_transformer_blocks.0.attn"].constmax_sigma
