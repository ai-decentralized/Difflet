"""The HunyuanVideo MXFP8 target table against diffusers' own module names (the checkpoint keys):
the dual-stream q/k/v of both streams and both MLPs, the single-stream q/k/v and MLP-up are
quantized; the output projections, the modulation, the token refiner, the embedders, the output
layer and the single blocks' fused ``proj_out`` stay bf16 (until the spec splits it); the
row-parallel set is every layer the TP layer shards along K, the token refiner's included, with
an MX layout at tp 1, 2, 4 and 8 on the real geometry. A spec naming the fused ``proj_out`` is
found by ``fused_proj_out_targets``, which the application and the driver refuse it with."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest
import torch

from difflet.cli.quant import QUANT_MODELS, quant_spec
from difflet.quant.spec import LAYOUT_K, QuantSpec
from difflet.quant.targets import (
    FUSED_PROJ_OUT_REASON,
    HUNYUAN_VIDEO_KEPT_BF16,
    HUNYUAN_VIDEO_ROW_PARALLEL,
    HUNYUAN_VIDEO_TARGETS,
    ROW_PARALLEL_BY_MODEL,
    TARGETS_BY_MODEL,
    default_spec,
    fused_proj_out_targets,
)

pytest.importorskip("diffusers")

from diffusers import HunyuanVideoTransformer3DModel  # noqa: E402

from tests.unit.backends._neuron_hv1_toy import (  # noqa: E402
    TINY_FP8_HV1,
    TINY_FP8_HV1_TP2,
    write_tiny_hv1,
)

HV1 = "hunyuanvideo-community/HunyuanVideo"

#: Per dual-stream block: both streams' q/k/v and both MLPs.
PER_DUAL = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
            "attn.add_v_proj", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj",
            "ff_context.net.2"}
#: Per single-stream block: the joint q/k/v and the MLP-up (``proj_out`` waits for its split).
PER_SINGLE = {"attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"}


@pytest.fixture(scope="module")
def model():
    return HunyuanVideoTransformer3DModel(**TINY_FP8_HV1)


def _linears(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def test_default_targets_cover_the_block_qkv_and_mlps_only(model):
    spec = default_spec("hunyuan_video")
    names = [n for n in _linears(model) if spec.matches(n)]
    assert {n.split(".", 2)[2] for n in names if n.startswith("transformer_blocks.")} == PER_DUAL
    assert {n.split(".", 2)[2] for n in names if n.startswith("single_")} == PER_SINGLE
    assert len(names) == (10 * TINY_FP8_HV1["num_layers"]
                          + 4 * TINY_FP8_HV1["num_single_layers"])
    assert not any(n.startswith("context_embedder.") for n in names)  # the refiner's own ff/attn


#: The neuron module's row layers (``RowParallelLinear`` in
#: ``difflet/models/hunyuan_video/modeling_hunyuan_video.py``) with a checkpoint key of their
#: own, by owner: the dual blocks' output projections and MLP-downs, and the token refiner's
#: attention output and MLP-down. The single blocks' ``proj_out_attn`` / ``proj_out_mlp`` are
#: row layers too, read as column ranges of the fused ``proj_out`` until the spec splits it.
ROW_LAYERS = {
    "transformer_blocks.": ("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"),
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
    row = {n for n in _linears(model) if spec.is_row_parallel(n)}
    assert row == {n for n in _linears(model) for owner, layers in ROW_LAYERS.items()
                   if n.startswith(owner) and n.endswith(layers)}
    assert spec.k_shards("context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0") == 4


def test_targets_and_kept_bf16_cover_every_linear_once(model):
    """The kept-bf16 table is documentation: with the targets it covers every linear of the
    model exactly once, so a reader can tell each layer's precision from the table alone."""
    for n in _linears(model):
        hits = [any(re.fullmatch(p, n) for p in table)
                for table in (HUNYUAN_VIDEO_TARGETS, HUNYUAN_VIDEO_KEPT_BF16)]
        assert hits.count(True) == 1, n


def test_every_default_target_has_an_mx_layout_at_tp_1_2_4_8():
    """On the real geometry (diffusers' defaults are HunyuanVideo's own: 20 + 40 blocks of 24
    heads x 128) the 360 default targets have a K per rank that is a multiple of the MX layout's
    tile at tp 1, 2, 4 and 8, so the default set refuses no tp: the column targets have K 3072
    and the MLP-downs 12288 / tp."""
    accelerate = pytest.importorskip("accelerate")
    with accelerate.init_empty_weights():
        real = HunyuanVideoTransformer3DModel()
    for tp in (1, 2, 4, 8):
        spec = default_spec("hunyuan_video", tp_degree=tp)
        targets = [(n, m) for n, m in real.named_modules()
                   if isinstance(m, torch.nn.Linear) and spec.matches(n)]
        assert len(targets) == 360
        assert all((m.in_features // spec.k_shards(n)) % LAYOUT_K == 0 for n, m in targets), tp


def test_fused_proj_out_targets_name_the_single_blocks_proj_out():
    assert fused_proj_out_targets(default_spec("hunyuan_video"), 40) == []
    every = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out",))
    assert fused_proj_out_targets(every, 2) == ["single_transformer_blocks.0.proj_out",
                                                "single_transformer_blocks.1.proj_out"]
    one = QuantSpec(targets=(r"single_transformer_blocks\.1\.proj_out",))
    assert fused_proj_out_targets(one, 2) == ["single_transformer_blocks.1.proj_out"]
    # module spelling: no checkpoint key, so the writer's dead-pattern refusal owns it
    split = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out_attn",))
    assert fused_proj_out_targets(split, 2) == []


def test_the_fused_proj_out_reason_reads_after_the_callers_prefix():
    """The application puts ``HunyuanVideo: <name>`` before the reason and the driver
    ``--quant: <name>``, so the reason names the model no second time, its clause on the
    checkpoint is about ``proj_out`` (not the neuron module), and it keeps the substrings the
    application's refusal test matches."""
    message = f"HunyuanVideo: single_transformer_blocks.1.proj_out {FUSED_PROJ_OUT_REASON}"
    assert re.search(r"single_transformer_blocks\.1\.proj_out is read as two column ranges "
                     r"\(proj_out_attn / proj_out_mlp\) by the neuron module, and .*phase D",
                     message)
    assert message.count("HunyuanVideo") == 1
    assert ", which " not in FUSED_PROJ_OUT_REASON


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
