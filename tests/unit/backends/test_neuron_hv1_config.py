"""HunyuanVideo 1.0 on the neuron backend (CPU only): the shape/topology config and the HBM
estimate the CLI checks before torchrun.

``NeuronHunyuanVideoConfig`` must describe the real snapshot by default (the latent and token
arithmetic at the registry's 320x512x61), refuse what HunyuanVideo or the neuron backend does not
run (CFG parallelism on a guidance-distilled model, unaligned shapes, ragged TP/CP/SP/Ulysses
splits) before anything loads, and carry every attribute ``HunyuanVideoTransformer3DModel``
reads from its config. ``hbm.dit_fits`` reads the safetensors headers only; the fake snapshot
declares the real checkpoint's 12.8 B parameters without any data.
"""

import json
import struct
from dataclasses import fields

import pytest

from difflet.backends.neuron import hbm as neuron_hbm
from difflet.backends.neuron.hunyuan_video import hbm
from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig
from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformerConfig

#: hunyuanvideo-community/HunyuanVideo's transformer/config.json.
SNAPSHOT_CONFIG = {
    "_class_name": "HunyuanVideoTransformer3DModel",
    "_diffusers_version": "0.32.0.dev0",
    "attention_head_dim": 128,
    "guidance_embeds": True,
    "in_channels": 16,
    "mlp_ratio": 4.0,
    "num_attention_heads": 24,
    "num_layers": 20,
    "num_refiner_layers": 2,
    "num_single_layers": 40,
    "out_channels": 16,
    "patch_size": 2,
    "patch_size_t": 1,
    "pooled_projection_dim": 768,
    "qk_norm": "rms_norm",
    "rope_axes_dim": [16, 56, 56],
    "rope_theta": 256.0,
    "text_embed_dim": 4096,
}


def test_defaults_match_the_snapshot_shape():
    c = NeuronHunyuanVideoConfig()
    assert (c.latent_frames, c.latent_height, c.latent_width) == (16, 40, 64)
    assert c.num_image_tokens == 16 * 20 * 32 == 10240
    assert c.inner_dim == 24 * 128
    assert (c.height, c.width, c.num_frames, c.batch_size, c.text_seq_len) == (
        320, 512, 61, 1, 256)
    c.validate()


def test_carries_every_attribute_the_port_reads():
    """``HunyuanVideoTransformer3DModel(config)`` reads these by name; the defaults are the
    snapshot's, as are the port's own."""
    ours = NeuronHunyuanVideoConfig()
    for f in fields(HunyuanVideoTransformerConfig):
        assert getattr(ours, f.name) == getattr(HunyuanVideoTransformerConfig(), f.name), f.name


def test_from_pretrained_reads_the_snapshot_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(SNAPSHOT_CONFIG))
    c = NeuronHunyuanVideoConfig.from_pretrained(tmp_path, tp_degree=2, num_frames=33)
    assert c.rope_axes_dim == (16, 56, 56) and isinstance(c.rope_axes_dim, tuple)
    assert (c.num_layers, c.num_single_layers, c.tp_degree, c.num_frames) == (20, 40, 2, 33)
    assert c == NeuronHunyuanVideoConfig(tp_degree=2, num_frames=33)
    with pytest.raises(ValueError, match="tp=5"):
        NeuronHunyuanVideoConfig.from_pretrained(tmp_path, tp_degree=5)


def test_validate_refuses_what_hunyuan_video_does_not_support():
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        NeuronHunyuanVideoConfig(cfg_parallel_enabled=True).validate()
    with pytest.raises(ValueError, match="4k\\+1"):
        NeuronHunyuanVideoConfig(num_frames=60).validate()
    with pytest.raises(ValueError, match="multiple of 16"):
        NeuronHunyuanVideoConfig(height=328).validate()
    with pytest.raises(ValueError, match="tp=5"):
        NeuronHunyuanVideoConfig(tp_degree=5).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        NeuronHunyuanVideoConfig(tp_degree=2, sp_enabled=True, cp_degree=2,
                                 context_parallel_enabled=True).validate()
    with pytest.raises(NotImplementedError, match="image_condition_type"):
        NeuronHunyuanVideoConfig(image_condition_type="latent_concat").validate()
    with pytest.raises(ValueError, match="context_parallel_enabled=True"):
        NeuronHunyuanVideoConfig(cp_degree=2).validate()
    with pytest.raises(ValueError, match="cp_mode"):
        NeuronHunyuanVideoConfig(cp_degree=2, context_parallel_enabled=True,
                                 cp_mode="striped").validate()


def test_validate_names_an_unsplittable_token_count():
    c = NeuronHunyuanVideoConfig(height=480, width=848, num_frames=33, cp_degree=4,
                                 context_parallel_enabled=True)
    with pytest.raises(ValueError, match="14310"):
        c.validate()
    sp = NeuronHunyuanVideoConfig(height=480, width=848, num_frames=33, tp_degree=4,
                                  sp_enabled=True)
    with pytest.raises(ValueError, match="14310"):
        sp.validate()


def test_supported_layouts_validate():
    for kwargs in (
        {"tp_degree": 2},
        {"tp_degree": 4},
        {"tp_degree": 2, "sp_enabled": True},
        {"tp_degree": 4, "sp_enabled": True},
        {"tp_degree": 2, "cp_degree": 2, "context_parallel_enabled": True},
        {"tp_degree": 2, "cp_degree": 2, "context_parallel_enabled": True, "cp_mode": "ring"},
    ):
        NeuronHunyuanVideoConfig(**kwargs).validate()


def test_ulysses_needs_heads_per_rank_divisible_by_cp():
    NeuronHunyuanVideoConfig(tp_degree=2, cp_degree=2, context_parallel_enabled=True,
                             cp_mode="ulysses").validate()          # 12 heads / 2: ok
    with pytest.raises(ValueError, match="ulysses"):
        NeuronHunyuanVideoConfig(tp_degree=8, cp_degree=4, context_parallel_enabled=True,
                                 cp_mode="ulysses").validate()      # 3 heads / 4


def _header_only(path, shapes, dtype="BF16"):
    """A safetensors file with a header and no data: ``safetensors_numel`` reads no further."""
    header = {name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_dit_fits_refuses_tp1_and_accepts_tp2(tmp_path, monkeypatch):
    # a fake snapshot whose safetensors header declares 12.8e9 bf16 params (23.8 GiB)
    snapshot = tmp_path / "HunyuanVideo"
    (snapshot / "transformer").mkdir(parents=True)
    _header_only(snapshot / "transformer" / "diffusion_pytorch_model.safetensors",
                 {"w": (128_000, 100_000)})
    monkeypatch.setenv(neuron_hbm.HBM_ENV, "24")
    fits, estimate, limit = hbm.dit_fits(str(snapshot), tp_degree=1)
    assert fits is False and estimate == 12_800_000_000 * 2 > limit
    assert hbm.dit_fits(str(snapshot), tp_degree=2)[0] is True
    assert hbm.dit_bytes(str(snapshot), tp_degree=4) == 12_800_000_000 * 2 // 4
