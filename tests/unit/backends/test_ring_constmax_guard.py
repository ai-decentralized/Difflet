"""The ring CP const-max guard: layers past their const-max table's gap limit keep the running-max
ring, since the Cauchy-Schwarz shift overshooting the row maximum by that much underflows."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3] / "difflet"
TABLES = {
    "flux": ROOT / "models/flux/attn_constmax.json",
    "hunyuan_video": ROOT / "models/hunyuan_video/attn_constmax.json",
    "hunyuan_video_15": ROOT / "backends/neuron/hunyuan_video15/attn_constmax.json",
    "qwen_image": ROOT / "models/qwen_image/attn_constmax.json",
}


@pytest.mark.parametrize("model", sorted(TABLES))
def test_ok_follows_the_gap_limit(model):
    table = json.loads(TABLES[model].read_text())
    for name, row in table["layers"].items():
        assert row["ok"] == (row["hard_bound"] - row["softmax_max"] <= table["gap_limit"]), name


def test_qwen_image_keeps_its_outlier_layers_off_the_const_max_ring():
    layers = json.loads(TABLES["qwen_image"].read_text())["layers"]
    assert sorted(k for k, v in layers.items() if not v["ok"]) == [
        "transformer_blocks.0.attn", "transformer_blocks.52.attn", "transformer_blocks.58.attn"]


def test_hunyuan_video_15_installs_each_layers_flag():
    from difflet.backends.neuron.hunyuan_video15.transformer import _install_ring_constmax_ok

    layers = json.loads(TABLES["hunyuan_video_15"].read_text())["layers"]
    blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in layers]
    model = SimpleNamespace(transformer_blocks=blocks)
    _install_ring_constmax_ok(model)
    got = [b.attn.ring_constmax_ok for b in model.transformer_blocks]
    assert got == [layers[f"transformer_blocks.{i}.attn"]["ok"] for i in range(len(layers))]
    assert not all(got)


def test_wan_installs_its_experts_flags_and_defaults_without_a_table():
    from difflet.models.wan.modeling_wan import add_ring_constmax_ok_

    tables = json.loads((ROOT / "models/wan/attn_constmax.json").read_text())["tables"]
    key = "5be7df96/transformer_2"
    layers = tables[key]["layers"]
    model = SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace()) for _ in layers])
    revision, expert = key.split("/")
    path = f"/hf/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/{revision}0000/{expert}"
    kept_off = add_ring_constmax_ok_(model, path)
    want = [layers[f"blocks.{i}.attn1"]["ok"] for i in range(len(layers))]
    assert [b.attn1.ring_constmax_ok for b in model.blocks] == want
    assert kept_off == want.count(False) > 0
    other = SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace())])
    assert add_ring_constmax_ok_(other, "/somewhere/else/transformer") == 0
    assert not hasattr(other.blocks[0].attn1, "ring_constmax_ok")
