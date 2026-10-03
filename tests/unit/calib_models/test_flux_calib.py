"""The FLUX calibration plugin (scripts/calib_models/flux.py) on a tiny random transformer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin(monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

    flux = _load(ROOT / "scripts" / "calib_models" / "flux.py", "calib_models_flux_test")
    torch.manual_seed(0)
    model = FluxTransformer2DModel(
        patch_size=1, in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=24, guidance_embeds=True,
        axes_dims_rope=(4, 6, 6),
    ).to(torch.bfloat16).eval()
    calls = {"forward": 0, "encode": 0}
    model.register_forward_pre_hook(lambda m, a: calls.__setitem__("forward", calls["forward"] + 1))

    def fake_encode(model_dir, prompt, seq_len):
        calls["encode"] += 1
        return {"prompt_embeds": torch.randn(1, seq_len, 32), "pooled_prompt_embeds": torch.randn(1, 24)}

    monkeypatch.setattr(flux, "encode_text", fake_encode)
    monkeypatch.setattr(flux, "load_transformer", lambda d: model)
    # FLUX.1-dev's scheduler/scheduler_config.json
    monkeypatch.setattr(flux, "load_scheduler", lambda d: FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=4096))
    return flux, calls


def _args(**kw):
    base = dict(model_dir=Path("/nonexistent"), model_type="flux", prompt="a red fox", height=32, width=32,
                num_frames=None, steps=3, max_steps=None, guidance_scale=3.5, seed=42, text_seq_len=8,
                threads=torch.get_num_threads(), text_pt=None, out=Path("/nonexistent/out.json"))
    base.update(kw)
    return SimpleNamespace(**base)


def _driver_hooks(args):
    """The driver's own install_hooks (real QuantSpec matching)."""
    calib = _load(ROOT / "scripts" / "ptq_calibrate_activations.py", "ptq_calibrate_activations_flux_test")
    from difflet.quant.spec import QuantSpec

    records: dict[str, list[float]] = {}
    counter = {"calls": 0}
    spec = QuantSpec.for_model("flux")
    return calib, records, (lambda m: calib._hook_targets(m, spec, records, counter, args.max_steps))


def test_run_records_hf_named_targets_once_per_step(plugin):
    flux, calls = plugin
    args = _args()
    calib, records, install = _driver_hooks(args)
    elapsed = flux.run(args, install)

    assert elapsed > 0
    assert calls["encode"] == 1
    assert calls["forward"] == args.steps  # guidance-distilled: one pass per step
    for name in ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.add_q_proj",
                 "transformer_blocks.0.ff_context.net.2", "single_transformer_blocks.0.attn.to_q",
                 "single_transformer_blocks.0.proj_mlp", "single_transformer_blocks.0.proj_out"):
        assert name in records, name
    assert len(records) == flux.LINEARS_PER_DOUBLE + flux.LINEARS_PER_SINGLE
    assert all(len(v) == args.steps and all(x > 0 for x in v) for v in records.values())
    assert not any(n.startswith("transformer.") or n == "proj_out" for n in records)

    # Names resolve through the checkpoint lookup the static-scale build uses.
    from difflet.quant.checkpoint import calibrated_amax

    layers = {n: max(v) for n, v in records.items()}
    assert calibrated_amax(layers, "single_transformer_blocks.0.proj_out") == layers["single_transformer_blocks.0.proj_out"]


def test_expected_hook_count_matches_flux1_dev():
    flux = _load(ROOT / "scripts" / "calib_models" / "flux.py", "calib_models_flux_count")
    assert flux.expected_hooks(SimpleNamespace(num_layers=19, num_single_layers=38)) == 418


def test_max_steps_stops_early_and_keeps_elapsed(plugin):
    flux, calls = plugin
    args = _args(steps=4, max_steps=2)
    calib, records, install = _driver_hooks(args)
    elapsed = flux.run(args, install)
    assert elapsed == elapsed and elapsed > 0  # not NaN
    assert calls["forward"] == 3  # third call trips the stop hook
    # The driver's stop hook sits after the record hook on the first target, which so
    # records the aborted third call; every other target holds exactly max_steps values.
    lengths = sorted(len(v) for v in records.values())
    assert lengths[0] == 2 and lengths[-2] == 2 and lengths[-1] == 3


def test_text_pt_skips_encoders(plugin, tmp_path):
    flux, calls = plugin
    pt = tmp_path / "flux_text.pt"
    torch.save({"prompt_embeds": torch.randn(1, 8, 32), "pooled_prompt_embeds": torch.randn(1, 24)}, pt)
    args = _args(steps=2, text_pt=pt)
    _, records, install = _driver_hooks(args)
    flux.run(args, install)
    assert calls["encode"] == 0
    assert calls["forward"] == 2
