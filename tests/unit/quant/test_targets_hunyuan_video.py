"""The HunyuanVideo MXFP8 target table against diffusers' own module names (the checkpoint keys):
the dual-stream q/k/v of both streams and both MLPs, the single-stream q/k/v, MLP-up and the
two halves of the split ``proj_out`` are quantized; the dual blocks' output projections, the
modulation, the token refiner, the embedders and the output layer stay bf16, and the fused
``proj_out`` is never a target (the spec stores it as its halves); the row-parallel set is
every layer the TP layer shards along K, the token refiner's and the halves included, with an
MX layout at tp 1, 2, 4 and 8 on the real geometry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest
import torch

from difflet.cli.quant import QUANT_MODELS, quant_spec
from difflet.quant.spec import LAYOUT_K
from difflet.quant.targets import (
    HUNYUAN_VIDEO_KEPT_BF16,
    HUNYUAN_VIDEO_ROW_PARALLEL,
    HUNYUAN_VIDEO_TARGETS,
    ROW_PARALLEL_BY_MODEL,
    SINGLE_BLOCK_SPLIT_LINEARS,
    SPLITS_BY_MODEL,
    TARGETS_BY_MODEL,
    default_spec,
)

pytest.importorskip("diffusers")

from diffusers import HunyuanVideoTransformer3DModel  # noqa: E402

from tests.unit.backends._neuron_hv1_toy import (  # noqa: E402
    TINY_FP8_HV1,
    TINY_FP8_HV1_TP2,
    write_tiny_hv1,
)

HV1 = "hunyuanvideo-community/HunyuanVideo"

#: Per dual-stream block: the text stream's q/k/v and both MLPs (the image q/k/v stay bf16).
PER_DUAL = {"attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2"}
#: Per single-stream block: the MLP-up and the halves of the split ``proj_out`` (the joint q/k/v
#: stay bf16).
PER_SINGLE = {"proj_mlp", "proj_out_attn", "proj_out_mlp"}


@pytest.fixture(scope="module")
def model():
    return HunyuanVideoTransformer3DModel(**TINY_FP8_HV1)


def _linears(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _stored_linears(model, spec) -> dict[str, int]:
    """The linears of the quantized directory under ``spec`` with their K: a split linear
    ``[N, K]`` as its halves ``<name>_attn`` (K = N) and ``<name>_mlp`` (K - N), as the writer
    stores them and the neuron module names them, every other one as itself."""
    linears = {}
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if spec.is_split(n):
            linears[f"{n}_attn"] = m.out_features
            linears[f"{n}_mlp"] = m.in_features - m.out_features
        else:
            linears[n] = m.in_features
    return linears


def test_default_targets_cover_the_text_qkv_the_mlps_and_the_proj_out_halves_only(model):
    spec = default_spec("hunyuan_video")
    names = [n for n in _stored_linears(model, spec) if spec.matches(n)]
    assert {n.split(".", 2)[2] for n in names if n.startswith("transformer_blocks.")} == PER_DUAL
    assert {n.split(".", 2)[2] for n in names if n.startswith("single_")} == PER_SINGLE
    assert len(names) == (7 * TINY_FP8_HV1["num_layers"]
                          + 3 * TINY_FP8_HV1["num_single_layers"])
    assert not spec.matches("transformer_blocks.0.attn.to_q")
    assert not spec.matches("single_transformer_blocks.0.attn.to_v")
    assert not any(n.startswith("context_embedder.") for n in names)  # the refiner's own ff/attn


def test_the_single_blocks_proj_out_is_stored_and_quantized_as_its_halves():
    """HunyuanVideo's spec splits the single blocks' fused ``proj_out`` (phase D's writer), so
    the targets name the halves the neuron module runs, never the fused linear, and each half
    is row-parallel: quantized per rank K range of its own."""
    assert SPLITS_BY_MODEL["hunyuan_video"] is SINGLE_BLOCK_SPLIT_LINEARS
    assert default_spec("hunyuan_video").split_linears == SINGLE_BLOCK_SPLIT_LINEARS
    spec = default_spec("hunyuan_video", tp_degree=4)
    assert spec.is_split("single_transformer_blocks.0.proj_out")
    assert spec.matches("single_transformer_blocks.0.proj_out_attn")
    assert spec.matches("single_transformer_blocks.39.proj_out_mlp")
    assert not spec.matches("single_transformer_blocks.0.proj_out")
    assert (spec.k_shards("single_transformer_blocks.0.proj_out_attn")
            == spec.k_shards("single_transformer_blocks.0.proj_out_mlp") == 4)


#: The neuron module's row layers (``RowParallelLinear`` in
#: ``difflet/models/hunyuan_video/modeling_hunyuan_video.py``) with a checkpoint key of their
#: own in the quantized directory, by owner: the dual blocks' output projections and MLP-downs,
#: the single blocks' ``proj_out`` halves (tensors of their own: the spec splits the fused
#: weight), and the token refiner's attention output and MLP-down.
ROW_LAYERS = {
    "transformer_blocks.": ("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"),
    "single_transformer_blocks.": ("proj_out_attn", "proj_out_mlp"),
    "context_embedder.token_refiner.": ("attn.to_out.0", "ff.net.2"),
}


def test_the_row_parallel_set_is_every_row_layer_with_a_key_of_its_own(model):
    """The FP8 twin of a row layer is picked by the module's type and shards the stored weight
    along K, but the writer lays a weight out per rank K range only where ``row_parallel`` names
    it. A row layer the set missed would be written as one K range, and wherever a rank's range
    is not whole 512-wide tiles (the refiner's ``attn.to_out.0``, K 3072, at tp 4 and 8) the
    ranks would read other block memberships than the weight was scaled in, with no error. The
    refiner's row layers are not default targets: they are listed for overrides."""
    spec = default_spec("hunyuan_video", tp_degree=4)
    stored = _stored_linears(model, spec)
    row = {n for n in stored if spec.is_row_parallel(n)}
    assert row == {n for n in stored for owner, layers in ROW_LAYERS.items()
                   if n.startswith(owner) and n.endswith(layers)}
    assert spec.k_shards("context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0") == 4


def test_targets_and_kept_bf16_cover_every_linear_once(model):
    """The kept-bf16 table is documentation: with the targets it covers every linear of the
    model exactly once, so a reader can tell each layer's precision from the table alone; the
    fused ``proj_out`` by the kept table, since the targets name its halves."""
    for n in _linears(model):
        hits = [any(re.fullmatch(p, n) for p in table)
                for table in (HUNYUAN_VIDEO_TARGETS, HUNYUAN_VIDEO_KEPT_BF16)]
        assert hits.count(True) == 1, n


def test_every_default_target_has_an_mx_layout_at_tp_1_2_4_8():
    """On the real geometry (diffusers' defaults are HunyuanVideo's own: 20 + 40 blocks of 24
    heads x 128) the 260 default targets have a K per rank that is a multiple of the MX layout's
    tile at tp 1, 2, 4 and 8, so the default set refuses no tp: the column targets have K 3072,
    the MLP-downs 12288 / tp, and the ``proj_out`` halves 3072 / tp and 12288 / tp (768 at tp 4:
    one 512-wide tile and a partial one of 256)."""
    accelerate = pytest.importorskip("accelerate")
    with accelerate.init_empty_weights():
        real = HunyuanVideoTransformer3DModel()
    for tp in (1, 2, 4, 8):
        spec = default_spec("hunyuan_video", tp_degree=tp)
        targets = {n: k for n, k in _stored_linears(real, spec).items() if spec.matches(n)}
        assert len(targets) == 260
        halves = {n: k for n, k in targets.items()
                  if n.endswith(("proj_out_attn", "proj_out_mlp"))}
        assert len(halves) == 80 and set(halves.values()) == {3072, 12288}
        assert all((k // spec.k_shards(n)) % LAYOUT_K == 0 for n, k in targets.items()), tp


def test_the_cli_id_resolves_to_the_table_at_the_launchs_tp(monkeypatch):
    monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)
    assert QUANT_MODELS[HV1] == "hunyuan_video"
    assert TARGETS_BY_MODEL["hunyuan_video"] is HUNYUAN_VIDEO_TARGETS
    assert ROW_PARALLEL_BY_MODEL["hunyuan_video"] is HUNYUAN_VIDEO_ROW_PARALLEL
    args = argparse.Namespace(model_id=HV1, quant="mxfp8", quant_compute=None, tp_degree=None)
    assert quant_spec(args) == default_spec("hunyuan_video", compute="nki", tp_degree=4)
    assert quant_spec(argparse.Namespace(**{**vars(args), "tp_degree": 2})).tp_degree == 2


def _saved_config(path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def test_the_fp8_toy_has_the_mx_geometry(tmp_path):
    fp8 = _saved_config(write_tiny_hv1(tmp_path / "a", config=TINY_FP8_HV1))
    assert (fp8["attention_head_dim"], fp8["rope_axes_dim"]) == (32, [8, 12, 12])
    tp2 = _saved_config(write_tiny_hv1(tmp_path / "c", config=TINY_FP8_HV1_TP2))
    assert (tp2["attention_head_dim"], tp2["rope_axes_dim"]) == (64, [16, 24, 24])
    inner = tp2["num_attention_heads"] * tp2["attention_head_dim"]  # 256: proj_out_attn K
    assert all(k % (2 * LAYOUT_K) == 0 for k in (inner, 4 * inner))  # per-rank K at tp2
    assert _saved_config(write_tiny_hv1(tmp_path / "b"))["attention_head_dim"] == 16
