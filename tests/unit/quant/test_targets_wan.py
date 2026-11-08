"""The Wan MXFP8 target table against diffusers' own module names, in checkpoint-key spelling
(``ffn.net.0.proj`` / ``ffn.net.2``, what the neuron application's ``checkpoint_key`` maps its
``ffn.net_in`` / ``net_out`` to): the cross-attention q/k/v and the FFN are quantized, the
self-attention q/k/v, every output projection, the modulation tables, the embedders and the
output layer stay bf16, and the row-parallel set is what the TP layer shards along K. One table
serves Wan 2.1's DiT and both experts of Wan 2.2-A14B (one block graph)."""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from difflet.cli.quant import QUANT_MODELS, quant_spec
from difflet.quant.spec import LAYOUT_K
from difflet.quant.targets import (
    ROW_PARALLEL_BY_MODEL,
    TARGETS_BY_MODEL,
    WAN_KEPT_BF16,
    WAN_ROW_PARALLEL,
    WAN_TARGETS,
    default_spec,
)

pytest.importorskip("diffusers")

from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_FP8_WAN,
    TINY_WAN_TRANSFORMER,
    tiny_wan_config,
    write_tiny_wan22_model,
    write_tiny_wan_model,
)

#: What TINY_FP8_WAN changes against the W1 toy: 32-wide heads (inner 128), a 512-wide FFN and
#: 128-wide text, so every target's K is a multiple of the MX layout's tile.
FP8_GEOMETRY = ("attention_head_dim", "ffn_dim", "text_dim")

#: Per block: the cross-attention's q/k/v and the two FFN linears (the self-attention's
#: q/k/v stay bf16 on measurement, difflet/quant/targets.py).
TARGETS_PER_BLOCK = 5

WAN_IDS = ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")


@pytest.fixture(scope="module")
def model():
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel(**TINY_FP8_WAN)


def _linears(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def test_default_targets_cover_the_cross_attention_qkv_and_the_ffn_only(model):
    spec = default_spec("wan")
    names = [n for n in _linears(model) if spec.matches(n)]
    per_block = {n.split(".", 2)[2] for n in names}
    assert per_block == {"attn2.to_q", "attn2.to_k", "attn2.to_v",
                         "ffn.net.0.proj", "ffn.net.2"}
    assert len(names) == TARGETS_PER_BLOCK * TINY_FP8_WAN["num_layers"]
    assert all(spec.is_row_parallel(n) == n.endswith(("to_out.0", "ffn.net.2"))
               for n, _ in model.named_modules())
    assert not any(spec.matches(n) for n in (
        "proj_out", "condition_embedder.text_embedder.linear_1", "blocks.0.attn1.to_out.0",
        "blocks.0.attn2.to_out.0", "blocks.0.attn1.to_q", "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v"))
    # The 917d087 default's order: the targets key the quantized directory, so the directories
    # written for that set are this one's.
    assert WAN_TARGETS == (r"blocks\.\d+\.attn2\.(to_q|to_k|to_v)",
                           r"blocks\.\d+\.ffn\.net\.0\.proj", r"blocks\.\d+\.ffn\.net\.2")


def test_the_targets_are_spelled_as_the_checkpoint_keys():
    """The neuron module names its FFN ``net_in`` / ``net_out``; the table is written against
    the checkpoint keys, which the application's ``checkpoint_key`` maps a parameter's name to,
    so the spec applied through it (``quantize_module_(key_fn=...)``: a module name goes through
    its weight's key) quantizes the FFN, while a module-spelled name matches nothing."""
    from difflet.backends.neuron.wan.transformer import checkpoint_key

    def key(module_name):
        return checkpoint_key(f"{module_name}.weight").removesuffix(".weight")

    spec = default_spec("wan", tp_degree=4)
    for module_name in ("blocks.0.ffn.net_in", "blocks.39.ffn.net_out"):
        assert not spec.matches(module_name)
        assert spec.matches(key(module_name))
    assert spec.k_shards(key("blocks.39.ffn.net_out")) == 4
    assert spec.k_shards(key("blocks.39.attn2.to_out.0")) == 4
    assert spec.k_shards(key("blocks.39.ffn.net_in")) == 1


def test_every_linear_is_a_target_or_documented_bf16(model):
    """The kept-bf16 table is documentation: with the targets it covers every linear of the
    model exactly once, so a reader can tell each layer's precision from the table alone; the
    modulation tables it names are parameters, not linears, so they match no module."""
    target = default_spec("wan")
    kept = default_spec("wan", targets=WAN_KEPT_BF16)
    for name in _linears(model):
        assert target.matches(name) != kept.matches(name), name
    assert all(kept.matches(f"blocks.0.attn1.{p}") for p in ("to_q", "to_k", "to_v"))
    tables = [n for n, _ in model.named_parameters() if n.endswith("scale_shift_table")]
    assert len(tables) == 1 + TINY_FP8_WAN["num_layers"]
    assert all(kept.matches(n) and not target.matches(n) for n in tables)


def test_the_fp8_toy_is_written_and_configured_with_its_own_geometry(model, tmp_path):
    """Every default target's K in the fp8 toy is a multiple of the MX layout's tile (``net.2``
    per rank: 512 / tp), and the toy writers carry that geometry through to the saved weights,
    the saved ``config.json`` of every expert and to ``tiny_wan_config`` (not the W1 toy's
    16-wide heads, which would build a 64-wide module against 128-wide weights)."""
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("transformers")
    from safetensors import safe_open

    spec = default_spec("wan")
    for name in _linears(model):
        if spec.matches(name):
            assert model.get_submodule(name).in_features % LAYOUT_K == 0, name
    assert all(TINY_FP8_WAN[k] != TINY_WAN_TRANSFORMER[k] for k in FP8_GEOMETRY)
    write_tiny_wan22_model(tmp_path, config=TINY_FP8_WAN)
    for expert in ("transformer", "transformer_2"):
        saved = json.loads((tmp_path / expert / "config.json").read_text())
        assert all(saved[k] == TINY_FP8_WAN[k] for k in FP8_GEOMETRY), expert
        with safe_open(str(tmp_path / expert / "diffusion_pytorch_model.safetensors"),
                       framework="pt") as f:
            assert f.get_slice("blocks.0.ffn.net.2.weight").get_shape() == [128, 512]
            assert f.get_slice("blocks.0.attn2.to_k.weight").get_shape() == [128, 128]
    reference = write_tiny_wan_model(tmp_path / "wan21", config=TINY_FP8_WAN)
    assert reference.shape[1] == TINY_FP8_WAN["out_channels"]
    config = tiny_wan_config(config=TINY_FP8_WAN)
    assert all(getattr(config, k) == TINY_FP8_WAN[k] for k in FP8_GEOMETRY)
    assert tiny_wan_config().attention_head_dim == TINY_WAN_TRANSFORMER["attention_head_dim"]


def test_the_registries_and_the_cli_ids_name_the_table():
    assert TARGETS_BY_MODEL["wan"] is WAN_TARGETS
    assert ROW_PARALLEL_BY_MODEL["wan"] is WAN_ROW_PARALLEL
    for model_id in WAN_IDS:
        assert QUANT_MODELS[model_id] == "wan"
        args = argparse.Namespace(quant="mxfp8", quant_compute=None, model_id=model_id,
                                  tp_degree=None)
        assert quant_spec(args) == default_spec("wan", compute="nki", tp_degree=4)
