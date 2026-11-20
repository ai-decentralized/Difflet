"""Experiment E1: Qwen-Image's const-max table and per-layer allowlist (DIFFLET_EXP_ATTN_CONSTMAX).

Tiny dims at the real depth (60 blocks), CPU: with the env set, each layer the table routes
carries its sigma as a non-persistent [128, 1] buffer and both joint-attention calls (key-bounded
and unbounded) pass it as ``constmax_sigma``; with the env unset, neither.
"""

import json
import os

import pytest
import torch

_PINNED = "DIFFLET_BACKEND" not in os.environ
os.environ.setdefault("DIFFLET_BACKEND", "cpu")

import difflet.models.qwen_image.modeling_qwen_image as mq  # noqa: E402
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig  # noqa: E402

if _PINNED:
    del os.environ["DIFFLET_BACKEND"]

_CFG = dict(patch_size=2, in_channels=16, out_channels=4, num_layers=60, attention_head_dim=32,
            num_attention_heads=2, joint_attention_dim=64, guidance_embeds=False,
            axes_dims_rope=[8, 12, 12])
_TABLE = json.loads(mq.Path(mq.__file__).with_name("attn_constmax.json").read_text())["layers"]


@pytest.fixture
def config(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_CFG))
    return QwenImageDiTConfig.from_pretrained(tmp_path / "transformer", height=64, width=64,
                                              text_seq_len=8, torch_dtype=torch.float32)


def _run(config, monkeypatch, env, *, masked):
    if env:
        monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", env)
    else:
        monkeypatch.delenv("DIFFLET_EXP_ATTN_CONSTMAX", raising=False)
    seen, real = [], mq.attention

    def spy(*args, **kwargs):
        seen.append((kwargs.get("bound_max") is not None, kwargs.get("constmax_sigma", "absent")))
        return real(*args, **kwargs)

    monkeypatch.setattr(mq, "attention", spy)
    module = mq._QwenImageTransformerTraceModule(config).eval()
    torch.manual_seed(0)
    hidden = torch.randn(1, config.image_seq_len, config.in_channels)
    text = torch.randn(1, config.text_seq_len, config.joint_attention_dim)
    with torch.no_grad():
        module(hidden, torch.full((1,), 0.5), text, None, torch.zeros(1),
               text_valid_len=torch.tensor([5]) if masked else None)
    return module.transformer.transformer_blocks, seen


@pytest.mark.parametrize("masked", [True, False])
def test_routed_layers_carry_and_pass_the_table_sigma(config, monkeypatch, masked):
    blocks, seen = _run(config, monkeypatch, "1", masked=masked)
    routed = [i for i, b in enumerate(blocks) if hasattr(b.attn, "constmax_sigma")]
    assert routed == [i for i in range(60) if i not in (0, 52, 58)]  # the table's 3 failures
    assert [bounded for bounded, _ in seen] == [masked] * 60
    for i, (_, sigma) in enumerate(seen):
        if i not in routed:
            assert sigma == "absent"
            continue
        assert sigma is blocks[i].attn.constmax_sigma and sigma.shape == (128, 1)
        assert torch.all(sigma == torch.tensor(_TABLE[f"transformer_blocks.{i}.attn"]["sigma"]))
        assert "constmax_sigma" not in blocks[i].attn.state_dict()  # not a checkpoint entry


@pytest.mark.parametrize("masked", [True, False])
def test_env_unset_adds_nothing(config, monkeypatch, masked):
    blocks, seen = _run(config, monkeypatch, None, masked=masked)
    assert not any(hasattr(b.attn, "constmax_sigma") for b in blocks)
    assert len(seen) == 60 and all(isinstance(s, str) and s == "absent" for _, s in seen)
