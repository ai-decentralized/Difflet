"""The backend-neutral Flux DiT surface (CUDA F5 Task 2).

``difflet.models.flux.dit_config.FluxDiTConfig`` is the duck-typed config the neutral
``NeuronFluxTransformer2DModel`` runs from off Trainium, ``convert_flux_hf_state_dict``
is the HF -> Difflet key conversion every backend shares, and the Trainium wrapper
classes live in ``difflet.models.flux.trainium_backbone`` so the modeling module
imports nothing from the Trainium backend.
"""

import ast
import json
from pathlib import Path

import torch

from difflet.models.flux.dit_config import FluxDiTConfig
from difflet.models.flux.modeling_flux import convert_flux_hf_state_dict

REPO = Path(__file__).resolve().parents[4]


def test_flux_dit_config_from_pretrained_reads_diffusers_fields(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxTransformer2DModel",
                "patch_size": 1,
                "in_channels": 64,
                "num_layers": 19,
                "num_single_layers": 38,
                "attention_head_dim": 128,
                "num_attention_heads": 24,
                "joint_attention_dim": 4096,
                "pooled_projection_dim": 768,
                "guidance_embeds": True,
                "axes_dims_rope": [16, 56, 56],
            }
        )
    )
    cfg = FluxDiTConfig.from_pretrained(tmp_path, height=64, width=64, torch_dtype=torch.float32)
    assert (cfg.patch_size, cfg.in_channels, cfg.num_layers, cfg.num_single_layers) == (
        1,
        64,
        19,
        38,
    )
    assert (cfg.attention_head_dim, cfg.num_attention_heads) == (128, 24)
    assert (cfg.joint_attention_dim, cfg.pooled_projection_dim) == (4096, 768)
    assert cfg.guidance_embeds is True
    assert cfg.axes_dims_rope == (16, 56, 56)
    assert cfg.out_channels is None  # absent -> the modeling falls back to in_channels
    assert cfg.inner_dim == 24 * 128
    assert cfg.image_seq_len == 16  # 64*64 // (2*8)**2
    assert cfg.neuron_config.torch_dtype is torch.float32
    assert (cfg.neuron_config.world_size, cfg.neuron_config.tp_degree) == (1, 1)
    # Runtime defaults the modeling reads through getattr.
    assert cfg.cfg_parallel_enabled is False and cfg.context_parallel_enabled is False
    assert cfg.cp_mode == "gather_kv" and cfg.sp_enabled is False


def test_convert_flux_hf_state_dict_splits_proj_out_at_inner_dim():
    heads, head_dim = 2, 8
    inner = heads * head_dim  # 16
    mlp_hidden = 4 * inner  # 64
    weight = torch.randn(inner, inner + mlp_hidden)
    bias = torch.randn(inner)
    other = torch.randn(3, 3)
    sd = {
        "single_transformer_blocks.0.proj_out.weight": weight,
        "single_transformer_blocks.0.proj_out.bias": bias,
        "x_embedder.weight": other,
    }

    out = convert_flux_hf_state_dict(sd, num_single_layers=1, inner_dim=inner)

    attn_w = out["single_transformer_blocks.0.proj_out_attn.weight"]
    mlp_w = out["single_transformer_blocks.0.proj_out_mlp.weight"]
    assert attn_w.shape == (16, 16) and mlp_w.shape == (16, 64)
    assert torch.equal(attn_w, weight[:, :inner]) and torch.equal(mlp_w, weight[:, inner:])
    assert torch.equal(out["single_transformer_blocks.0.proj_out_attn.bias"], bias)
    assert "single_transformer_blocks.0.proj_out_mlp.bias" not in out
    # The fused HF keys are consumed, so a strict load sees only Difflet names.
    assert "single_transformer_blocks.0.proj_out.weight" not in out
    assert "single_transformer_blocks.0.proj_out.bias" not in out
    # The SPMD rank buffer is Trainium state; the Trainium converter adds it, not this one.
    assert "global_rank.rank" not in out
    assert torch.equal(out["x_embedder.weight"], other)
    # Contiguous clones: the result never aliases the caller's tensors.
    assert all(t.is_contiguous() for t in out.values())
    assert out["x_embedder.weight"].data_ptr() != other.data_ptr()
    assert attn_w.data_ptr() != weight.data_ptr()
    # The caller's dict is left as it was.
    assert set(sd) == {
        "single_transformer_blocks.0.proj_out.weight",
        "single_transformer_blocks.0.proj_out.bias",
        "x_embedder.weight",
    }


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_trainium_backbone_module_is_the_only_trainium_importer():
    flux = REPO / "difflet/models/flux"
    for neutral in ("modeling_flux.py", "dit_config.py"):
        offenders = [
            m
            for m in _imported_modules(flux / neutral)
            if m == "difflet.backends.trainium" or m.startswith("difflet.backends.trainium.")
        ]
        assert offenders == [], f"{neutral} imports {offenders}"
    backbone = _imported_modules(flux / "trainium_backbone.py")
    assert any(m.startswith("difflet.backends.trainium.") for m in backbone)
