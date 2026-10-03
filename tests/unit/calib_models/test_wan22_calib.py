"""The Wan 2.2 A14B calibration plugin (scripts/calib_models/wan22.py) on two tiny experts.

A tiny on-disk snapshot (model_index.json with boundary_ratio, the real UniPC
scheduler config, transformer/ + transformer_2/ safetensors in diffusers naming)
drives the real WanOrchestrator loop through the plugin's real loader.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.modeling_wan as wan  # noqa: E402

if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND

# scripts/ptq_fp8_device_probe.py::TINY_CONFIG
TINY_CONFIG = {
    "_class_name": "WanTransformer3DModel", "patch_size": [1, 2, 2], "num_attention_heads": 4,
    "attention_head_dim": 32, "in_channels": 16, "out_channels": 16, "text_dim": 64, "freq_dim": 64,
    "ffn_dim": 256, "num_layers": 2, "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024, "eps": 1e-6,
}
# Wan2.2-T2V-A14B-Diffusers scheduler/scheduler_config.json (the fields that differ from defaults).
SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler", "beta_end": 0.02, "beta_schedule": "linear", "beta_start": 0.0001,
    "flow_shift": 3.0, "num_train_timesteps": 1000, "predict_x0": True, "prediction_type": "flow_prediction",
    "solver_order": 2, "solver_type": "bh2", "use_flow_sigmas": True, "final_sigmas_type": "zero",
    "lower_order_final": True, "timestep_spacing": "linspace",
}
_TO_DIFFUSERS = ((".ffn.net_in.", ".ffn.net.0.proj."), (".ffn.net_out.", ".ffn.net.2."))
TEXT_SEQ, TEXT_DIM = 8, 64


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    monkeypatch.syspath_prepend(str(SCRIPTS))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_expert(path: Path, seed: int) -> None:
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    model = wan.WanTransformer3DModel(wan.WanTransformerConfig.from_diffusers_dict(dict(TINY_CONFIG)))
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("rope."):
            continue  # non-persistent in practice; never in a real checkpoint
        for ours, hf in _TO_DIFFUSERS:
            key = key.replace(ours, hf)
        state[key] = value.float().contiguous()  # the real shards are fp32
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    save_file(state, str(path / "diffusion_pytorch_model.safetensors"))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("wan22_tiny")
    (root / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline", "boundary_ratio": 0.875}))
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SCHEDULER_CONFIG))
    _save_expert(root / "transformer", seed=1)
    _save_expert(root / "transformer_2", seed=2)
    return root


def _high_steps(steps: int) -> int:
    from diffusers import UniPCMultistepScheduler

    sched = UniPCMultistepScheduler.from_config({k: v for k, v in SCHEDULER_CONFIG.items() if k != "_class_name"})
    sched.set_timesteps(steps)
    return sum(float(t) >= 875.0 for t in sched.timesteps)


def _args(snapshot, **kw):
    base = dict(model_dir=snapshot, model_type="wan22", prompt="a red fox", height=32, width=32, num_frames=5,
                steps=8, max_steps=None, guidance_scale=1.0, seed=42, text_seq_len=TEXT_SEQ, threads=1,
                text_pt=None, out=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _harness(max_steps=None):
    """The driver's own install_hooks (QuantSpec for 'wan') plus per-expert forward counters."""
    driver = importlib.import_module("ptq_calibrate_activations")
    from difflet.quant.spec import QuantSpec

    spec = QuantSpec.for_model("wan")
    records, counter, hooked = {}, {"calls": 0}, []

    def install_hooks(model):
        n = driver._hook_targets(model, spec, records, counter, max_steps)
        hooked.append(n)
        return n

    return driver, install_hooks, records, hooked


def _count_forwards(plugin, monkeypatch):
    calls = {"transformer": 0, "transformer_2": 0}
    real = plugin._load_expert

    def load(expert_dir, dtype=torch.bfloat16):
        model = real(expert_dir, dtype)
        name = Path(expert_dir).name
        model.register_forward_pre_hook(lambda m, a: calls.__setitem__(name, calls[name] + 1))
        return model

    monkeypatch.setattr(plugin, "_load_expert", load)
    return calls


def test_loader_matches_checkpoint_in_bf16(snapshot):
    plugin = _load(SCRIPTS / "calib_models" / "wan22.py", "calib_models_wan22_test")
    model = plugin._load_expert(snapshot / "transformer_2")
    ref = wan.WanTransformer3DModel(wan.WanTransformerConfig.from_diffusers_dict(dict(TINY_CONFIG))).to(torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 and not p.is_meta for p in model.parameters())
    assert torch.equal(model.rope.freqs_cos, ref.rope.freqs_cos)  # recomputed, not left on meta
    from safetensors.torch import load_file

    raw = load_file(str(snapshot / "transformer_2" / "diffusion_pytorch_model.safetensors"))
    assert torch.equal(model.blocks[1].ffn.net_out.weight, raw["blocks.1.ffn.net.2.weight"].to(torch.bfloat16))


def test_run_records_both_experts_under_distinct_names(snapshot, tmp_path, monkeypatch):
    plugin = _load(SCRIPTS / "calib_models" / "wan22.py", "calib_models_wan22_test")
    calls = _count_forwards(plugin, monkeypatch)
    text_pt = tmp_path / "text.pt"
    torch.save({"prompt_embeds": torch.randn(1, TEXT_SEQ, TEXT_DIM)}, text_pt)
    _, install_hooks, records, hooked = _harness()
    args = _args(snapshot, text_pt=text_pt)

    elapsed = plugin.run(args, install_hooks)

    n_high = _high_steps(args.steps)
    assert 0 < n_high < args.steps
    assert calls == {"transformer": n_high, "transformer_2": args.steps - n_high}
    assert hooked[0] == hooked[1] > 0
    assert len(records) == hooked[0] + hooked[1]
    assert len(records["blocks.0.attn1.to_q"]) == n_high
    assert len(records["transformer_2.blocks.0.attn1.to_q"]) == args.steps - n_high
    assert len(records["transformer_2.blocks.1.ffn.net_out"]) == args.steps - n_high
    assert sum(k.startswith("transformer_2.") for k in records) == hooked[1]
    assert elapsed > 0


def test_cfg_encodes_the_empty_negative_prompt_and_doubles_the_calls(snapshot, monkeypatch):
    plugin = _load(SCRIPTS / "calib_models" / "wan22.py", "calib_models_wan22_test")
    calls = _count_forwards(plugin, monkeypatch)
    driver, install_hooks, records, _ = _harness()
    prompts = []

    def fake_embeds(model_dir, prompt, seq_len, dtype):
        prompts.append(prompt)
        return torch.randn(1, seq_len, TEXT_DIM).to(dtype)

    monkeypatch.setattr(driver, "_prompt_embeds", fake_embeds)
    args = _args(snapshot, steps=6, guidance_scale=3.0)
    plugin.run(args, install_hooks)

    n_high = _high_steps(args.steps)
    assert prompts == ["a red fox", ""]
    assert calls == {"transformer": 2 * n_high, "transformer_2": 2 * (args.steps - n_high)}
    assert len(records["transformer_2.blocks.0.attn1.to_q"]) == 2 * (args.steps - n_high)


def test_max_steps_counts_across_the_expert_switch(snapshot, tmp_path, monkeypatch):
    plugin = _load(SCRIPTS / "calib_models" / "wan22.py", "calib_models_wan22_test")
    text_pt = tmp_path / "text.pt"
    torch.save({"prompt_embeds": torch.randn(1, TEXT_SEQ, TEXT_DIM)}, text_pt)
    args = _args(snapshot, text_pt=text_pt)
    n_high = _high_steps(args.steps)
    driver, install_hooks, records, _ = _harness(max_steps=n_high + 2)
    with pytest.raises(driver._Stop):
        plugin.run(args, install_hooks)
    # n_high high-noise steps, then 2 low-noise steps; the aborted 3rd low-noise call
    # only reaches the first hooked linear (driver behaviour), so check a later one.
    assert len(records["blocks.0.ffn.net_out"]) == n_high
    assert len(records["transformer_2.blocks.0.ffn.net_out"]) == 2
