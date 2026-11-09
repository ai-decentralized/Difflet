"""The HunyuanVideo-1.5 MXFP8 tables against diffusers' own module names (the checkpoint keys:
the neuron module is diffusers' class): the model has no default set (Task F8, the user's
decision of 2026-11-09), so ``"mxfp8"`` alone is refused; the four device-bisection groups are
the override table, and G1 + G3 (the dual-stream blocks' image q/k/v and image MLP) through
``DIFFLET_QUANT_TARGETS`` is the very spec the device arms measured (so it resolves their
quantized directory); every linear is in one group or is documented bf16 (the output
projections, the modulation, the token refiner, the embedders and the output layer); the
row-parallel set is every row layer the TP layer shards along K, and every group has an MX
layout at tp 1 to 16 on the real geometry. The FP8 toy
(``_neuron_hv15_toy``) carries the MX geometry to a snapshot whose row-parallel weights
quantize differently per rank than whole."""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from difflet.cli.quant import QUANT_MODELS, quant_spec
from difflet.quant import mx
from difflet.quant.spec import LAYOUT_K
from difflet.quant.targets import (
    HUNYUAN_VIDEO_15_GROUPS,
    HUNYUAN_VIDEO_15_KEPT_BF16,
    HUNYUAN_VIDEO_15_ROW_PARALLEL,
    HUNYUAN_VIDEO_15_TARGETS,
    ROW_PARALLEL_BY_MODEL,
    TARGETS_BY_MODEL,
    default_spec,
)

pytest.importorskip("diffusers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from difflet.backends.neuron.core import weights_bf16  # noqa: E402
from tests.unit.backends._neuron_hv15_toy import (  # noqa: E402
    HV15_ROW_TARGETS,
    TINY_FP8_HV15,
    HV15_MEASURED_TARGETS,
    TINY_TARGETS_PER_BLOCK,
    hv15_spec,
    write_tiny_hv15_snapshot,
)

HV15 = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
#: The set F6's device arms ran as ``DIFFLET_QUANT_TARGETS`` (G1 + G3, mx_final, 2026-11-08),
#: verbatim: the quantized directory is keyed by the ordered pattern strings, so the override
#: must give this spec to resolve the directory those arms wrote.
MEASURED_TARGETS_ENV = (r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);"
                        r"transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)")

#: The neuron module's row layers (``_install_tensor_parallel`` in
#: ``difflet/backends/neuron/hunyuan_video15/transformer.py``), by owner: the dual blocks'
#: output projections and MLP-downs, and the token refiner's attention output and MLP-down,
#: listed for overrides (the FP8 row twin shards its weight along K whatever the set says).
ROW_LAYERS = {
    "transformer_blocks.": ("attn.to_out.0", "attn.to_add_out", "ff.net.2", "ff_context.net.2"),
    "context_embedder.token_refiner.": ("attn.to_out.0", "ff.net.2"),
}


@pytest.fixture(autouse=True)
def _no_quant_targets_override(monkeypatch):
    monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)


@pytest.fixture(scope="module")
def model():
    from diffusers import HunyuanVideo15Transformer3DModel

    return HunyuanVideo15Transformer3DModel(**TINY_FP8_HV15)


def _linears(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _is_row_layer(name: str) -> bool:
    return any(name.startswith(owner) and name.endswith(layers)
               for owner, layers in ROW_LAYERS.items())


def test_the_measured_set_covers_the_image_streams_projections_and_mlp_only(model):
    spec = hv15_spec()
    names = [n for n in _linears(model) if spec.matches(n)]
    assert {n.split(".", 2)[2] for n in names} == TINY_TARGETS_PER_BLOCK
    assert len(names) == len(TINY_TARGETS_PER_BLOCK) * TINY_FP8_HV15["num_layers"]  # 10
    assert all(spec.is_row_parallel(n) == _is_row_layer(n) for n in _linears(model))
    assert not any(spec.matches(n) for n in (
        "transformer_blocks.0.attn.add_q_proj", "transformer_blocks.1.attn.add_v_proj",
        "transformer_blocks.0.ff_context.net.0.proj", "transformer_blocks.1.ff_context.net.2",
        "context_embedder.token_refiner.refiner_blocks.0.attn.to_q",
        "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.norm1.linear", "transformer_blocks.0.attn.to_add_out", "proj_out"))


def test_the_override_gives_the_spec_the_device_measured(monkeypatch):
    args = argparse.Namespace(quant="mxfp8", quant_compute=None, model_id=HV15, tp_degree=None)
    monkeypatch.setenv("DIFFLET_QUANT_TARGETS", MEASURED_TARGETS_ENV)
    assert quant_spec(args, default_tp_degree=2) == hv15_spec(compute="nki", tp_degree=2)
    assert HV15_MEASURED_TARGETS == tuple(MEASURED_TARGETS_ENV.split(";"))
    assert list(HUNYUAN_VIDEO_15_GROUPS) == ["G1", "G2", "G3", "G4"]
    assert HUNYUAN_VIDEO_15_GROUPS["G2"] == (
        r"transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj)")
    assert HUNYUAN_VIDEO_15_GROUPS["G4"] == r"transformer_blocks\.\d+\.ff_context\.net\.(0\.proj|2)"


def test_hunyuan_video_15_has_no_default_set():
    """Task F8 (the user's decision, 2026-11-09): every measured set costs HunyuanVideo-1.5
    4.7-7.5 dB against the CPU fp32 reference beyond bf16 and buys neither speed nor peak HBM
    at 121 frames, so nothing is quantized unless the user names the layers: the table is
    empty, and the default spec and the CLI's spec are refused, naming the override."""
    assert HUNYUAN_VIDEO_15_TARGETS == ()
    with pytest.raises(ValueError, match=r"'hunyuan_video_15' has no default MXFP8 target set"
                                         r".*DIFFLET_QUANT_TARGETS"):
        default_spec("hunyuan_video_15")
    args = argparse.Namespace(quant="mxfp8", quant_compute=None, model_id=HV15, tp_degree=None)
    with pytest.raises(ValueError, match="DIFFLET_QUANT_TARGETS"):
        quant_spec(args, default_tp_degree=2)


def test_every_linear_is_in_one_group_or_documented_bf16(model):
    tables = {**{g: (p,) for g, p in HUNYUAN_VIDEO_15_GROUPS.items()},
              "bf16": HUNYUAN_VIDEO_15_KEPT_BF16}
    specs = {k: default_spec("hunyuan_video_15", targets=t) for k, t in tables.items()}
    for name in _linears(model):
        assert [k for k, spec in specs.items() if spec.matches(name)] in (
            [g] for g in tables), name


def test_the_fp8_toy_has_the_mx_geometry(model, tmp_path):
    spec = hv15_spec()
    for n in _linears(model):
        if spec.matches(n):
            k = model.get_submodule(n).in_features
            assert all((k // tp if spec.is_row_parallel(n) else k) % LAYOUT_K == 0
                       for tp in (1, 2, 4)), n
    root = write_tiny_hv15_snapshot(tmp_path)
    saved = json.loads((root / "transformer" / "config.json").read_text())
    assert saved["attention_head_dim"] == 32
    assert weights_bf16._has_fp32(root / "transformer")  # the copy path is exercised


def test_the_toy_row_parallel_targets_quantize_differently_per_rank(tmp_path):
    # The spread makes a whole-K row-parallel quantization visible (Review Focus 4, F2).
    root = write_tiny_hv15_snapshot(tmp_path)
    weights = safetensors_torch.load_file(next((root / "transformer").glob("*.safetensors")))
    for name in HV15_ROW_TARGETS:
        w = weights[f"{name}.weight"]
        differ = (mx.fake_quantize_mx(w) != mx.fake_quantize_mx_sharded(w, 2)).float().mean()
        assert differ > 0.3, name  # 0.374 / 0.378


def test_every_group_has_an_mx_layout_at_every_tp():
    # diffusers' defaults with task_type="t2v" are the 480p_t2v geometry (54 blocks of 16 x 128,
    # MLP ratio 4; F6 Step 1 checks the downloaded config.json against them).
    accelerate = pytest.importorskip("accelerate")
    from diffusers import HunyuanVideo15Transformer3DModel

    with accelerate.init_empty_weights():
        real = HunyuanVideo15Transformer3DModel(task_type="t2v")
    for tp in (1, 2, 4, 8, 16):
        spec = default_spec("hunyuan_video_15", targets=tuple(HUNYUAN_VIDEO_15_GROUPS.values()),
                            tp_degree=tp)
        targets = [(n, real.get_submodule(n)) for n in _linears(real) if spec.matches(n)]
        assert len(targets) == 54 * 10, tp  # 3 + 3 q/k/v and 2 + 2 MLP linears per block
        assert all((m.in_features // spec.k_shards(n)) % LAYOUT_K == 0 for n, m in targets), tp
    assert spec.k_shards("context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0") == 16


def test_the_registries_and_the_cli_id_name_the_table():
    assert TARGETS_BY_MODEL["hunyuan_video_15"] is HUNYUAN_VIDEO_15_TARGETS
    assert ROW_PARALLEL_BY_MODEL["hunyuan_video_15"] is HUNYUAN_VIDEO_15_ROW_PARALLEL
    assert QUANT_MODELS[HV15] == "hunyuan_video_15"
