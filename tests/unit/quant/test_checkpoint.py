"""Offline checkpoint quantizer: state-dict rewrite, on-disk layout, reuse rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec


def _wan_like_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "patch_embedding.weight": torch.randn(8, 4, 1, 2, 2, dtype=torch.bfloat16),
        "condition_embedder.time_embedder.linear_1.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "blocks.0.attn1.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "blocks.0.attn1.to_q.bias": torch.randn(8, dtype=torch.bfloat16),
        "blocks.0.attn1.to_out.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "blocks.0.ffn.net.0.proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "blocks.0.ffn.net.2.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "blocks.0.scale_shift_table": torch.randn(1, 6, 8, dtype=torch.bfloat16),
        "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16),
    }


def test_quantize_state_dict_targets_only_the_fastvideo_layer_set():
    sd = _wan_like_state_dict()
    out, report = ckpt.quantize_state_dict(sd, QuantSpec(weight_granularity="channel"))

    quantized = {
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_out.0",
        "blocks.0.ffn.net.0.proj",
        "blocks.0.ffn.net.2",
    }
    assert set(report["quantized"]) == quantized
    assert report["num_quantized"] == 4
    for prefix in quantized:
        assert out[f"{prefix}.weight"].dtype == torch.float8_e4m3fn
        scale = out[f"{prefix}.weight_scale"]
        assert scale.dtype == torch.float32
        assert scale.shape == (sd[f"{prefix}.weight"].shape[0], 1)
    # Untouched tensors are passed through as the same objects, biases included.
    for key in (
        "patch_embedding.weight",
        "condition_embedder.time_embedder.linear_1.weight",
        "blocks.0.attn1.to_q.bias",
        "blocks.0.scale_shift_table",
        "proj_out.weight",
    ):
        assert out[key] is sd[key]
    assert "proj_out.weight_scale" not in out
    assert report["bytes_after"] < report["bytes_before"]

    per_tensor, _ = ckpt.quantize_state_dict(sd, QuantSpec(weight_granularity="tensor"))
    assert per_tensor["blocks.0.attn1.to_q.weight_scale"].shape == (1,)


def _write_source(model_dir: Path) -> Path:
    src = model_dir / "transformer"
    src.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in _wan_like_state_dict().items()},
              str(src / "diffusion_pytorch_model.safetensors"))
    (src / "config.json").write_text(json.dumps({"_class_name": "WanTransformer3DModel"}))
    return src


def test_quantize_checkpoint_dir_round_trips_through_the_vendored_loader(tmp_path):
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict

    src = _write_source(tmp_path / "model")
    spec = QuantSpec()
    dest = ckpt.quantized_checkpoint_dir(tmp_path / "cache", src, spec)
    assert dest.parent.name == "transformer"
    assert dest.name.startswith("fp8-tensor-")

    manifest = ckpt.quantize_checkpoint_dir(src, dest, spec)
    assert manifest["schema_version"] == ckpt.MANIFEST_SCHEMA_VERSION
    assert manifest["report"]["num_quantized"] == 4
    assert (dest / "config.json").is_file()
    assert (dest / ckpt.MANIFEST_FILENAME).is_file()
    assert ckpt.is_valid_quantized_checkpoint(dest, spec, src)

    loaded = load_state_dict(str(dest))
    assert loaded["blocks.0.attn1.to_q.weight"].dtype == torch.float8_e4m3fn
    assert loaded["blocks.0.attn1.to_q.weight_scale"].dtype == torch.float32
    assert loaded["proj_out.weight"].dtype == torch.bfloat16
    assert set(loaded) == set(_wan_like_state_dict()) | {
        "blocks.0.attn1.to_q.weight_scale",
        "blocks.0.attn1.to_out.0.weight_scale",
        "blocks.0.ffn.net.0.proj.weight_scale",
        "blocks.0.ffn.net.2.weight_scale",
    }

    # A valid checkpoint is reused, not rebuilt.
    stamp = (dest / ckpt.MANIFEST_FILENAME).stat().st_mtime_ns
    ckpt.quantize_checkpoint_dir(src, dest, spec)
    assert (dest / ckpt.MANIFEST_FILENAME).stat().st_mtime_ns == stamp


def test_ensure_checkpoint_respects_create_flag_and_spec_identity(tmp_path):
    src = _write_source(tmp_path / "model")
    spec = QuantSpec()
    dest = ckpt.quantized_checkpoint_dir(tmp_path / "cache", src, spec)

    with pytest.raises(FileNotFoundError, match="difflet quantize"):
        ckpt.ensure_quantized_checkpoint(src, dest, spec, create=False)
    assert ckpt.ensure_quantized_checkpoint(src, dest, spec, create=True) == dest
    # The activation mode does not change the weights: same checkpoint serves both.
    assert ckpt.ensure_quantized_checkpoint(src, dest, QuantSpec(activation="none"), create=False) == dest
    # A different weight granularity is a different checkpoint.
    channel = QuantSpec(weight_granularity="channel")
    assert not ckpt.is_valid_quantized_checkpoint(dest, channel, src)
    assert ckpt.quantized_checkpoint_dir(tmp_path / "cache", src, channel) != dest


def test_checkpoint_dir_is_keyed_by_source_and_uses_hf_cache_slug(tmp_path):
    spec = QuantSpec()
    hf = tmp_path / "hub" / "models--Wan-AI--Wan2.1-T2V-14B-Diffusers" / "snapshots" / "abc" / "transformer"
    hf.mkdir(parents=True)
    path = ckpt.quantized_checkpoint_dir(tmp_path / "cache", hf, spec)
    assert path.parts[-3] == "Wan-AI--Wan2.1-T2V-14B-Diffusers"
    assert path.parts[-2] == "transformer"
    other = tmp_path / "hub" / "models--Wan-AI--Wan2.1-T2V-14B-Diffusers" / "snapshots" / "def" / "transformer"
    other.mkdir(parents=True)
    assert ckpt.quantized_checkpoint_dir(tmp_path / "cache", other, spec) != path


def test_quantized_hf_checkpoint_maps_onto_difflet_wan_names(tmp_path):
    """HF-layout fp8 checkpoint -> vendored loader renames -> Difflet Wan keys.

    Mirrors what NeuronApplicationBase.get_state_dict does on device
    (``.weight_scale`` -> ``.scale``) followed by convert_backbone_state_dict,
    so the quantized FFN keys land on ``ffn.net_in`` / ``ffn.net_out``.
    """
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
    from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict

    src = _write_source(tmp_path / "model")
    dest = tmp_path / "q"
    ckpt.quantize_checkpoint_dir(src, dest, QuantSpec(weight_granularity="channel"))
    loaded = load_state_dict(str(dest))
    renamed = {
        (k[: -len(".weight_scale")] + ".scale" if k.endswith(".weight_scale") else k): v
        for k, v in loaded.items()
    }
    converted = convert_backbone_state_dict(renamed)
    assert converted["blocks.0.ffn.net_in.weight"].dtype == torch.float8_e4m3fn
    assert converted["blocks.0.ffn.net_in.scale"].shape == (16, 1)
    assert converted["blocks.0.ffn.net_out.scale"].shape == (8, 1)
    assert converted["blocks.0.attn1.to_out.0.scale"].dtype == torch.float32
    assert "blocks.0.ffn.net.0.proj.weight" not in converted
    assert converted["proj_out.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_split_fused_proj_out_copies_scale_for_both_granularities(granularity):
    """FLUX / HunyuanVideo split the single-block proj_out along the input dim at
    load; the fp8 weight splits the same way and the scale (per-tensor [1] or
    per-channel [out, 1]) is valid for both halves, so it is copied (Review Focus 3)."""
    from difflet.quant.fp8 import quantize_weight

    w = torch.randn(8, 12, dtype=torch.bfloat16)
    q, scale = quantize_weight(w, granularity)
    sd = {"blk.0.proj_out.weight": q, "blk.0.proj_out.scale": scale, "blk.0.proj_out.bias": torch.zeros(8)}
    ckpt.split_fused_proj_out(sd, "blk.0.proj_out", attn_name="blk.0.proj_out_attn",
                              mlp_name="blk.0.proj_out_mlp", cols=4)
    assert "blk.0.proj_out.weight" not in sd and "blk.0.proj_out.scale" not in sd
    attn, mlp = sd["blk.0.proj_out_attn.weight"], sd["blk.0.proj_out_mlp.weight"]
    assert attn.shape == (8, 4) and attn.dtype == torch.float8_e4m3fn and mlp.shape == (8, 8)
    assert torch.equal(attn.float(), q[:, :4].float()) and torch.equal(mlp.float(), q[:, 4:].float())
    assert torch.equal(sd["blk.0.proj_out_attn.scale"], scale)
    assert torch.equal(sd["blk.0.proj_out_mlp.scale"], scale)
    assert "blk.0.proj_out_attn.bias" in sd and "blk.0.proj_out_mlp.bias" not in sd


def test_split_fused_proj_out_without_scale_keeps_the_bf16_path():
    sd = {"blk.0.proj_out.weight": torch.randn(8, 12), "blk.0.proj_out.bias": torch.zeros(8)}
    ckpt.split_fused_proj_out(sd, "blk.0.proj_out", attn_name="a", mlp_name="m", cols=4)
    assert set(sd) == {"a.weight", "a.bias", "m.weight"}
    assert sd["a.weight"].shape == (8, 4) and sd["m.weight"].shape == (8, 8)
