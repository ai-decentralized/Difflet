"""bf16 copies of fp32 checkpoints: bit-identical to casting at load, used only when valid and when
the component runs in bf16, skipped when nothing is fp32."""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file

from difflet.backends.neuron.core import weights_bf16 as wb


def _checkpoint(root, dtype=torch.float32):
    comp = root / "snap123" / "transformer"
    comp.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    shards = {"model-00001-of-00002.safetensors": {"a.weight": torch.randn(64, 32, generator=g).to(dtype),
                                                   "a.step": torch.arange(4)},
              "model-00002-of-00002.safetensors": {"b.weight": torch.randn(16, 8, generator=g).to(dtype)}}
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(comp / name))
        weight_map.update({k: name for k in tensors})
    (comp / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (comp / "config.json").write_text(json.dumps({"_class_name": "Toy"}))
    return comp, shards


def test_the_copy_is_bit_identical_to_casting_at_load(tmp_path):
    comp, shards = _checkpoint(tmp_path)
    copy = wb.ensure_bf16_copy(comp, cache_root=tmp_path / "cache")
    assert copy == tmp_path / "cache" / "weights-bf16" / "snap123" / "transformer"
    for name, tensors in shards.items():
        got = load_file(str(copy / name))
        for key, original in tensors.items():
            expected = original.to(torch.bfloat16) if original.dtype == torch.float32 else original
            assert got[key].dtype == expected.dtype and torch.equal(got[key], expected)
    assert (copy / "config.json").exists() and (copy / "diffusion_pytorch_model.safetensors.index.json").exists()


def test_resolution_prefers_a_valid_copy_for_bf16_only(tmp_path):
    comp, _ = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    assert wb.resolve_component_dir(comp, torch.bfloat16, cache) == str(comp)  # no copy yet
    copy = wb.ensure_bf16_copy(comp, cache_root=cache)
    assert wb.resolve_component_dir(comp, torch.bfloat16, cache) == str(copy)
    assert wb.resolve_component_dir(comp, torch.float32, cache) == str(comp)
    assert wb.ensure_bf16_copy(comp, cache_root=cache) == copy  # reused, not rebuilt


def test_a_changed_source_invalidates_the_copy(tmp_path):
    comp, _ = _checkpoint(tmp_path)
    cache = tmp_path / "cache"
    wb.ensure_bf16_copy(comp, cache_root=cache)
    cfg = comp / "config.json"
    cfg.write_text(json.dumps({"_class_name": "Toy", "changed": True}))
    os.utime(cfg, ns=(1, 1))
    assert wb.resolve_component_dir(comp, torch.bfloat16, cache) == str(comp)


def test_a_checkpoint_without_fp32_needs_no_copy(tmp_path):
    comp, _ = _checkpoint(tmp_path, dtype=torch.bfloat16)
    assert wb.ensure_bf16_copy(comp, cache_root=tmp_path / "cache") is None
    assert wb.resolve_component_dir(comp, torch.bfloat16, tmp_path / "cache") == str(comp)
