"""scripts/calib_models/qwen_image.py on a tiny random diffusers Qwen-Image DiT (no real weights)."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(num_layers: int = 1):
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(0)
    return QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=num_layers, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=32, guidance_embeds=False, axes_dims_rope=(4, 6, 6),
    ).to(torch.bfloat16).eval()


def _model_dir(tmp_path: Path) -> Path:
    from diffusers import FlowMatchEulerDiscreteScheduler

    # The real snapshot's scheduler config (dynamic shifting -> the CLI's mu).
    FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9, shift=1.0,
        shift_terminal=0.02, time_shift_type="exponential", use_dynamic_shifting=True,
    ).save_pretrained(str(tmp_path / "scheduler"))
    return tmp_path


def _args(tmp_path, **kw):
    base = dict(model_dir=_model_dir(tmp_path), model_type="qwen_image", prompt="a red fox", height=64, width=64,
                num_frames=1, steps=3, max_steps=None, guidance_scale=4.0, seed=42, text_seq_len=512,
                threads=2, text_pt=None, out=tmp_path / "out.json")
    base.update(kw)
    return argparse.Namespace(**base)


def _install(records, calls):
    os.environ["DIFFLET_BACKEND"] = "cpu"
    driver = _load("ptq_calibrate_activations_q", ROOT / "scripts" / "ptq_calibrate_activations.py")
    from difflet.quant.spec import QuantSpec

    spec = QuantSpec.for_model("qwen_image")

    def install_hooks(model):
        n = driver._hook_targets(model, spec, records, {"calls": 0}, None)
        model.register_forward_pre_hook(lambda m, a, k: calls.append(k), with_kwargs=True)
        return n
    return install_hooks


def test_run_records_hf_named_targets_per_step(tmp_path, monkeypatch):
    plugin = _load("calib_qwen_image", ROOT / "scripts" / "calib_models" / "qwen_image.py")
    seq = plugin._device_text_seq_len()
    assert seq == 1024
    embeds = torch.randn(1, 7, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(plugin, "_load_transformer", lambda d: _tiny_model())
    monkeypatch.setattr(plugin, "_encode_prompt",
                        lambda d, p, n: plugin._pad(embeds, torch.ones(1, 7, dtype=torch.bool), n))

    records, calls = {}, []
    elapsed = plugin.run(_args(tmp_path), _install(records, calls))

    assert elapsed >= 0
    assert "transformer_blocks.0.attn.to_q" in records
    assert "transformer_blocks.0.img_mlp.net.2" in records
    assert not any(n.startswith("transformer.") for n in records)
    assert len(records) == 12
    assert all(len(v) == 3 for v in records.values())  # one DiT call per step (no true CFG on device)
    assert len(calls) == 3
    kw = calls[0]
    assert kw["encoder_hidden_states_mask"] is None  # dropped, as on device
    assert kw["encoder_hidden_states"].shape == (1, seq, 32)
    assert torch.all(kw["encoder_hidden_states"][0, 7:] == 0)  # zero padding to TEXT_SEQ_LEN
    assert kw["hidden_states"].shape == (1, 16, 64)  # (64/16)^2 packed tokens
    # timestep/1000 of the mu-shifted schedule: first sigma is 1.0
    assert float(calls[0]["timestep"][0]) == pytest.approx(1.0, abs=1e-2)
    assert float(calls[1]["timestep"][0]) < float(calls[0]["timestep"][0])


def test_text_pt_skips_encoder_and_wrong_count_raises(tmp_path, monkeypatch):
    plugin = _load("calib_qwen_image2", ROOT / "scripts" / "calib_models" / "qwen_image.py")
    text_pt = tmp_path / "text.pt"
    mask = torch.zeros(1, 1024, dtype=torch.bool)
    mask[:, :5] = True
    torch.save({"encoder_hidden_states": torch.randn(1, 1024, 32, dtype=torch.bfloat16),
                "encoder_hidden_states_mask": mask}, text_pt)
    monkeypatch.setattr(plugin, "_encode_prompt", lambda *a: pytest.fail("encoder must not load with --text-pt"))
    monkeypatch.setattr(plugin, "_load_transformer", lambda d: _tiny_model())
    records, calls = {}, []
    plugin.run(_args(tmp_path, text_pt=text_pt, steps=2), _install(records, calls))
    assert len(calls) == 2 and all(len(v) == 2 for v in records.values())

    with pytest.raises(RuntimeError, match="expected 12"):
        plugin.run(_args(tmp_path, text_pt=text_pt, steps=1), lambda m: 11)
