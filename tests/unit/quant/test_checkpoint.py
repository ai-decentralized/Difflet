"""The quantized checkpoint directory, written from the tiny diffusers-layout Qwen-Image DiT."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from difflet.backends._torch_megatron.checkpoint import build_weight_map
from difflet.quant import fp8
from difflet.quant.calibration import ActivationCalibrator, read_calibration, write_calibration
from difflet.quant.checkpoint import (
    MANIFEST,
    check_quantized_dir,
    ensure_quantized_dir,
    quantized_dir,
    quantized_linear_names,
    source_fingerprint,
    write_quantized_dir,
)
from difflet.quant.fake_linear import FakeQuantLinear, fake_quantize_model_
from difflet.quant.targets import default_spec

pytest.importorskip("diffusers")

from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    tiny_qwen_inputs,
    write_tiny_qwen_model,
)

#: 2 blocks x (6 attention projections + 2 attention outputs + 4 MLP linears)
TINY_TARGET_COUNT = 2 * 12


def _diffusers_model(transformer_dir):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval().requires_grad_(False)


def _forward(model, dtype=torch.float32):
    hidden, timestep, text = tiny_qwen_inputs(dtype=dtype)
    with torch.no_grad():
        return model(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text,
                     encoder_hidden_states_mask=None, img_shapes=[[(1, *TINY_PACKED)]],
                     return_dict=False)[0]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_qwen")
    reference = write_tiny_qwen_model(root)
    transformer_dir = root / "transformer"
    spec = default_spec("qwen_image")
    model = _diffusers_model(transformer_dir)
    with ActivationCalibrator(model, spec) as cal:
        _forward(model)
    calibration = read_calibration(write_calibration(
        root / "calibration.json", amax=cal.amax(), model="qwen_image",
        checkpoint={"path": str(transformer_dir)}, prompts=["toy"], steps=1,
        shape={"height": 64, "width": 96}, calls=cal.calls()))
    spec = dataclasses.replace(spec, calibration_sha256=calibration.sha256)
    return {"root": root, "transformer_dir": transformer_dir, "spec": spec,
            "calibration": calibration, "reference": reference}


def test_calibration_covers_every_target_linear(tiny):
    assert len(tiny["calibration"].amax) == TINY_TARGET_COUNT


def test_write_quantized_dir_layout_and_manifest(tiny, tmp_path):
    out = tmp_path / "fp8"
    manifest = write_quantized_dir(tiny["transformer_dir"], out, tiny["spec"],
                                   tiny["calibration"], shard_bytes=64 * 1024)
    assert (out / MANIFEST).is_file() and (out / "config.json").is_file()
    assert manifest["tensors"]["quantized"] == TINY_TARGET_COUNT
    assert len(manifest["files"]) > 1, "the small shard size must produce several shards"
    assert manifest["calibration"]["sha256"] == tiny["calibration"].sha256
    assert manifest["spec"] == tiny["spec"].to_dict()
    assert quantized_linear_names(out) == manifest["quantized_linears"]
    assert manifest["bytes"] < manifest["source"]["bytes"]

    weight_map = build_weight_map(out)
    source_map = build_weight_map(tiny["transformer_dir"])
    index = json.loads((out / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(weight_map)
    # Every source key is present; every target weight brings its two scales.
    assert set(source_map) <= set(weight_map)
    extra = set(weight_map) - set(source_map)
    assert len(extra) == 2 * TINY_TARGET_COUNT
    assert all(k.endswith((".weight_scale", ".input_scale")) for k in extra)


def test_quantized_tensors_match_the_fake_quant_reference(tiny, tmp_path):
    from safetensors import safe_open

    out = tmp_path / "fp8"
    write_quantized_dir(tiny["transformer_dir"], out, tiny["spec"], tiny["calibration"])
    model = _diffusers_model(tiny["transformer_dir"])
    plain = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    fake_quantize_model_(model, tiny["spec"], tiny["calibration"])
    fakes = {n: m for n, m in model.named_modules() if isinstance(m, FakeQuantLinear)}
    assert len(fakes) == TINY_TARGET_COUNT

    weight_map = build_weight_map(out)
    handles = {}
    try:
        def get(key):
            path = weight_map[key]
            if path not in handles:
                handles[path] = safe_open(path, framework="pt").__enter__()
            return handles[path].get_tensor(key)

        for name, fake in fakes.items():
            w = get(f"{name}.weight")
            assert w.dtype == fp8.FP8_DTYPE and torch.equal(w, fake.weight)
            assert torch.equal(get(f"{name}.weight_scale"), fake.weight_scale)
            assert torch.equal(get(f"{name}.input_scale"), fake.input_scale)
            assert torch.equal(get(f"{name}.bias"), plain[name].bias.detach())
        # A kept tensor is copied byte for byte, in its own dtype.
        mod = get("transformer_blocks.0.img_mod.1.weight")
        assert mod.dtype == torch.float32
        assert torch.equal(mod, plain["transformer_blocks.0.img_mod.1"].weight.detach())
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)


def test_fake_quant_model_stays_close_to_the_reference(tiny):
    model = _diffusers_model(tiny["transformer_dir"])
    fake_quantize_model_(model, tiny["spec"], tiny["calibration"])
    out = _forward(model)
    ref = tiny["reference"]
    cosine = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cosine > 0.99, cosine
    assert not torch.equal(out, ref)


def test_check_and_ensure(tiny, tmp_path):
    cache = tmp_path / "cache"
    spec, cal, src = tiny["spec"], tiny["calibration"], tiny["transformer_dir"]
    target = quantized_dir(cache, src, spec)
    assert target.parent.name == "transformer" and target.name.startswith("fp8-")
    with pytest.raises(ValueError, match="never written"):
        ensure_quantized_dir(cache, src, spec, cal, create=False)
    made = ensure_quantized_dir(cache, src, spec, cal, create=True)
    assert made == target and (made / MANIFEST).is_file()
    # Idempotent: a second call finds the valid directory and writes nothing.
    stamp = (made / MANIFEST).stat().st_mtime_ns
    assert ensure_quantized_dir(cache, src, spec, cal, create=False) == made
    assert (made / MANIFEST).stat().st_mtime_ns == stamp

    check_quantized_dir(made, src, spec, cal.sha256)
    # The compute path does not change the weights: dequant and nki share the directory.
    other = dataclasses.replace(spec, compute="dequant")
    assert quantized_dir(cache, src, other) == target
    check_quantized_dir(made, src, other, cal.sha256)
    with pytest.raises(ValueError, match="another calibration"):
        check_quantized_dir(made, src, spec, "0" * 64)
    with pytest.raises(ValueError, match="another quantization spec"):
        check_quantized_dir(made, src, dataclasses.replace(spec, fp8_max=240.0), cal.sha256)
    assert quantized_dir(cache, src, dataclasses.replace(spec, fp8_max=240.0)) != target
    with pytest.raises(FileExistsError):
        write_quantized_dir(src, made, spec, cal)


def test_source_fingerprint_ignores_mtime_but_not_content(tiny, tmp_path):
    import os
    import shutil

    src = tiny["transformer_dir"]
    before = source_fingerprint(src)
    copy = tmp_path / "copy"
    shutil.copytree(src, copy)
    for path in copy.iterdir():
        os.utime(path, (0, 0))
    assert source_fingerprint(copy) == before
    shard = sorted(copy.glob("*.safetensors"))[0]
    shard.write_bytes(shard.read_bytes() + b"\0")
    assert source_fingerprint(copy) != before
