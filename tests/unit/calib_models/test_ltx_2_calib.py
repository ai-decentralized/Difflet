"""The LTX-2 calibration plugin (scripts/calib_models/ltx_2.py) on a tiny random transformer.

No real weights, no Gemma: the loader, scheduler and conditioning are monkeypatched.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

# LTX-2's scheduler/scheduler_config.json (snapshot dfcc2108).
_SCHEDULER = dict(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True, base_shift=0.95,
                  max_shift=2.05, base_image_seq_len=1024, max_image_seq_len=4096, shift_terminal=0.1,
                  time_shift_type="exponential")
_CAPTION = 12
_SEQ = 8


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(num_layers: int = 2):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    return LTX2VideoTransformer3DModel(
        in_channels=8, out_channels=8, num_attention_heads=2, attention_head_dim=12, cross_attention_dim=24,
        audio_in_channels=16, audio_out_channels=16, audio_num_attention_heads=2, audio_attention_head_dim=8,
        audio_cross_attention_dim=16, num_layers=num_layers, caption_channels=_CAPTION, rope_type="split",
    ).to(torch.bfloat16).eval()


def _cond(batch: int) -> dict[str, torch.Tensor]:
    mask = torch.ones(batch, _SEQ, dtype=torch.bool)
    mask[:, : _SEQ // 4] = False  # left padding, as Gemma's tokenizer pads
    return {"encoder_hidden_states": torch.randn(batch, _SEQ, _CAPTION, dtype=torch.bfloat16),
            "audio_encoder_hidden_states": torch.randn(batch, _SEQ, _CAPTION, dtype=torch.bfloat16),
            "encoder_attention_mask": mask, "audio_encoder_attention_mask": mask.clone()}


@pytest.fixture
def plugin(monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler

    ltx = _load(ROOT / "scripts" / "calib_models" / "ltx_2.py", "calib_models_ltx_2_test")
    model = _tiny_model()
    calls = {"forward": 0, "encode": 0}
    model.register_forward_pre_hook(lambda m, a, k: calls.__setitem__("forward", calls["forward"] + 1),
                                    with_kwargs=True)

    def fake_conditioning(args, dtype=torch.bfloat16):
        calls["encode"] += 1
        return _cond(2 if args.guidance_scale > 1 else 1)

    monkeypatch.setattr(ltx, "compute_conditioning", fake_conditioning)
    monkeypatch.setattr(ltx, "load_transformer", lambda d, dtype=torch.bfloat16: model)
    monkeypatch.setattr(ltx, "load_scheduler", lambda d: FlowMatchEulerDiscreteScheduler(**_SCHEDULER))
    return ltx, calls


def _args(tmp_path, **kw):
    base = dict(model_dir=tmp_path / "snap", model_type="ltx_2", prompt="a red fox", height=64, width=64,
                num_frames=9, steps=3, max_steps=None, guidance_scale=1.0, seed=42, text_seq_len=_SEQ,
                threads=torch.get_num_threads(), text_pt=None, out=tmp_path / "act_calibration.json")
    base.update(kw)
    return SimpleNamespace(**base)


def _driver_hooks(args):
    """The driver's own install_hooks (real QuantSpec matching) and its records."""
    calib = _load(ROOT / "scripts" / "ptq_calibrate_activations.py", "ptq_calibrate_activations_ltx_2_test")
    from difflet.quant.spec import QuantSpec

    spec = QuantSpec.for_model("ltx_2")
    records: dict[str, list[float]] = {}
    counter = {"calls": 0}
    return records, lambda model: calib._hook_targets(model, spec, records, counter, args.max_steps)


def test_run_records_unprefixed_hf_names_for_every_target(plugin, tmp_path):
    ltx, calls = plugin
    args = _args(tmp_path)
    records, install_hooks = _driver_hooks(args)
    elapsed = ltx.run(args, install_hooks)

    assert elapsed > 0
    assert calls == {"forward": 3, "encode": 1}  # one pass per step at guidance 1
    assert "transformer_blocks.0.attn1.to_q" in records
    assert "transformer_blocks.0.ff.net.2" in records
    assert "transformer_blocks.1.audio_to_video_attn.to_out.0" in records
    assert "transformer_blocks.1.audio_ff.net.0.proj" in records
    assert not any(n.startswith("transformer.") for n in records)
    assert len(records) == 2 * 28  # blocks x LTX_2 targets: every hooked linear fired
    assert all(len(v) == 3 and all(x > 0 for x in v) for v in records.values())
    # the computed conditioning is kept for --text-pt reruns
    saved = torch.load(tmp_path / "act_calibration_conditioning.pt")
    assert set(saved) == set(ltx.COND_KEYS)


def test_run_cfg_two_passes_per_step_from_text_pt(plugin, tmp_path):
    ltx, calls = plugin
    cond_path = tmp_path / "cond.pt"
    torch.save({k: v for k, v in _cond(2).items() if k != "audio_encoder_attention_mask"}, cond_path)
    args = _args(tmp_path, guidance_scale=3.0, steps=2, text_pt=cond_path)
    records, install_hooks = _driver_hooks(args)
    ltx.run(args, install_hooks)

    assert calls == {"forward": 4, "encode": 0}  # uncond + cond per step, Gemma skipped
    assert all(len(v) == 4 for v in records.values())


def test_text_pt_accepts_cache_dit_inputs_bundle(plugin, tmp_path):
    from safetensors.torch import save_file

    ltx, _ = plugin
    tensors = dict(_cond(1), latents_init=torch.zeros(1, 8, 8))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp_path / "bundle.safetensors"))
    cond = ltx.load_conditioning_file(tmp_path / "bundle.safetensors")
    assert set(cond) == set(ltx.COND_KEYS)
    with pytest.raises(ValueError, match="batch 2"):
        ltx._validate_conditioning(cond, guidance_scale=3.0, text_seq_len=_SEQ)


def test_hook_count_mismatch_is_fatal(plugin, tmp_path):
    ltx, _ = plugin
    with pytest.raises(RuntimeError, match="expected 56"):
        ltx.run(_args(tmp_path), lambda model: 55)


def test_real_checkpoint_keys_match_targets():
    """Smoke check against the real snapshot's key index only (skipped when absent)."""
    import json

    snap = Path(os.environ.get(
        "LTX2_SNAPSHOT", "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/"
        "dfcc2108383fe1aaa0584bdf55d368a4bdadd90c"))
    index = snap / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if not index.exists():
        pytest.skip("LTX-2 snapshot not present")
    from difflet.quant.spec import QuantSpec

    spec = QuantSpec.for_model("ltx_2")
    linears = {k[: -len(".weight")] for k in json.loads(index.read_text())["weight_map"] if k.endswith(".weight")}
    hits = sorted(n for n in linears if spec.matches(n))
    layers = json.loads((snap / "transformer" / "config.json").read_text())["num_layers"]
    assert len(hits) == layers * 28 == 1344
    assert "transformer_blocks.3.attn1.to_q" in hits and "transformer_blocks.3.ff.net.2" in hits
