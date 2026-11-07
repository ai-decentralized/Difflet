"""The MXFP8 HunyuanVideo DiT on the neuron backend, on CPU: the application takes ``quant`` as a
keyword of its own (never a config override, with or without ``config=``), quantizes the dual
blocks' q/k/v of both streams and both MLPs and the single blocks' joint q/k/v, ``proj_mlp`` and
the two halves of ``proj_out``, loads the quantized directory written straight from the source
``transformer`` (the fused ``proj_out`` stored as its halves, read by name) and, in dequant
mode, matches diffusers' model run with the weights the directory holds dequantized, at tp1,
tp2, tp2 with SP (the image stream's unreduced row-parallel partials) and cp2; the dual
blocks' output projections, the modulation and the token refiner stay bf16. A target set
without the halves loads them in the source's dtype by name, a target on the fused ``proj_out``
is a dead pattern the writer refuses, and eager mode and a spec quantized for another tp are
refused. The pipeline application resolves the quantized directory from the source
``transformer`` at the launch's tp and refuses what the shared resolver refuses.

Every run of the Difflet port is in a fresh gloo process (``hv1_dit_worker``): importing the
port binds ``difflet.ops`` to the backend of the moment, so it must not be imported here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("safetensors.torch")

from difflet.backends.neuron.hunyuan_video.config import NeuronHunyuanVideoConfig  # noqa: E402
from difflet.backends.neuron.hunyuan_video.transformer import (  # noqa: E402
    NeuronHunyuanVideoTransformerApplication,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.fake_linear import SplitLinear, split_linear_columns  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402
from difflet.quant.targets import (  # noqa: E402
    HUNYUAN_VIDEO_TARGETS,
    SINGLE_BLOCK_SPLIT_LINEARS,
    default_spec,
)
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_hv1_toy import (  # noqa: E402
    TINY_FP8_HV1_TP2,
    TINY_SHAPE,
    tiny_inputs,
    write_tiny_hv1,
)
from tests.unit.backends._neuron_workers import hv1_dit_worker  # noqa: E402

#: Per dual block: both streams' q/k/v and both MLPs' up and down (10); per single block: the
#: joint q/k/v, ``proj_mlp`` and the two halves of ``proj_out`` (6).
TINY_TARGETS_HV1 = (10 * TINY_FP8_HV1_TP2["num_layers"]
                    + 6 * TINY_FP8_HV1_TP2["num_single_layers"])  # 22
#: Valid text tokens of the toy call the runs and the references share.
VALID_TEXT = 9


def _fake_quant_reference(transformer_dir: str, spec: QuantSpec) -> torch.Tensor:
    """diffusers' model with every weight ``spec`` targets fake-quantized, a row-parallel one
    per rank K range (``spec.k_shards``), and a split linear (the single blocks' ``proj_out``)
    run as its two column ranges, each quantized on its own by the half's own shard count: what
    dequant mode computes, since it multiplies the unquantized activations by the weights the
    directory holds, dequantized. diffusers' module names are the checkpoint keys, so the table
    matches them as it is."""
    from diffusers import HunyuanVideoTransformer3DModel

    model = HunyuanVideoTransformer3DModel.from_pretrained(transformer_dir).eval()
    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        for name, module in linears:
            if spec.is_split(name):
                halves = split_linear_columns(module)
                for h, half in zip(("attn", "mlp"), halves):
                    if spec.matches(f"{name}_{h}"):
                        half.weight.copy_(mx.fake_quantize_mx_sharded(
                            half.weight, spec.k_shards(f"{name}_{h}")))
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent), leaf,
                        SplitLinear(*halves, split=module.out_features))
            elif spec.matches(name):
                module.weight.copy_(mx.fake_quantize_mx_sharded(module.weight,
                                                                spec.k_shards(name)))
        return model(**tiny_inputs(VALID_TEXT), return_dict=False)[0]


def _tensors(directory) -> dict[str, torch.Tensor]:
    """Every tensor of a directory's safetensors files, by key."""
    from safetensors.torch import load_file

    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tp2-geometry toy (``TINY_FP8_HV1_TP2``: every K per rank a multiple of 128 at tp2)
    with, for tp 1 and 2, the default dequant spec, its quantized directory under the root's
    cache (written from the source ``transformer``, as HunyuanVideo's hub DiT is already bf16)
    and the reference that spec computes."""
    root = tmp_path_factory.mktemp("tiny_hv1_fp8")
    transformer_dir = write_tiny_hv1(root, config=TINY_FP8_HV1_TP2)
    specs = {tp: default_spec("hunyuan_video", compute="dequant", tp_degree=tp) for tp in (1, 2)}
    quantized = {tp: prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                                  cache_root=root / "cache")
                 for tp, spec in specs.items()}
    reference = {tp: _fake_quant_reference(transformer_dir, spec) for tp, spec in specs.items()}
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, specs=specs,
                           quantized=quantized, reference=reference)


def _run(tiny, parallel: dict, world: int, tp: int) -> list[dict]:
    return run_ranks(hv1_dit_worker, tiny.transformer_dir, parallel, "compile", VALID_TEXT, None,
                     tiny.specs[tp].to_dict(), str(tiny.quantized[tp]), world_size=world)


def _app(**kwargs) -> NeuronHunyuanVideoTransformerApplication:
    """Constructed only: nothing is built or loaded, so the port is never imported here."""
    return NeuronHunyuanVideoTransformerApplication(dtype=torch.float32, device="cpu", **kwargs)


def test_the_quantized_directory_holds_the_default_targets_only(tiny):
    """The default quantizes the single blocks' ``proj_out`` as its two halves, each fp8 with
    its scales; the bias stays the fused one and the fused weight is never stored."""
    source = _tensors(tiny.transformer_dir)
    for tp in (1, 2):
        manifest = read_manifest(tiny.quantized[tp])
        names = set(manifest["quantized_linears"])
        assert manifest["tensors"]["quantized"] == TINY_TARGETS_HV1 == len(names)
        assert not any(n.endswith(("to_out.0", "to_add_out", "proj_out", ".linear"))
                       for n in names)
        stored = _tensors(tiny.quantized[tp])
        for i in range(TINY_FP8_HV1_TP2["num_single_layers"]):
            p = f"single_transformer_blocks.{i}"
            assert {f"{p}.proj_out_attn", f"{p}.proj_out_mlp"} <= names
            assert stored[f"{p}.proj_out_attn.weight"].dtype == mx.FP8_DTYPE
            assert stored[f"{p}.proj_out_mlp.weight_scale"].dtype == torch.uint8
            assert torch.equal(stored[f"{p}.proj_out.bias"], source[f"{p}.proj_out.bias"])
            assert f"{p}.proj_out.weight" not in stored


def test_quant_is_the_base_class_keyword_not_a_config_override(tiny):
    """``quant=`` reaches the base, with ``config=`` (where an override is refused as one) and
    without (where the overrides build the config): it is never a config field."""
    spec = tiny.specs[1]
    common = dict(model_path=tiny.quantized[1], exec_mode="compile")
    config = NeuronHunyuanVideoConfig.from_pretrained(tiny.transformer_dir, **TINY_SHAPE)
    assert _app(config=config, quant=spec, **common).quant == spec
    app = _app(quant=spec.to_dict(), **TINY_SHAPE, **common)
    assert app.quant == spec and app.config.num_frames == TINY_SHAPE["num_frames"]
    assert not hasattr(app.config, "quant")


def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    [r] = _run(tiny, {"tp_degree": 1}, 1, 1)
    assert r["report"] == {"missing": [], "unexpected": []} and r["unwarmed_shapes"] == []
    assert sorted(r["fp8"]) == sorted(r["quantized_linears"])
    assert len(r["fp8"]) == TINY_TARGETS_HV1
    assert r["fp8"]["transformer_blocks.0.attn.add_v_proj"] == "FP8ColumnParallelLinear"
    assert r["fp8"]["transformer_blocks.0.ff_context.net.2"] == "FP8RowParallelLinear"
    assert r["fp8"]["single_transformer_blocks.1.proj_mlp"] == "FP8ColumnParallelLinear"
    assert (r["fp8"]["single_transformer_blocks.0.proj_out_attn"]
            == r["fp8"]["single_transformer_blocks.1.proj_out_mlp"] == "FP8RowParallelLinear")
    assert not any(n.endswith(("to_out.0", "to_add_out", ".linear"))
                   or n.startswith("context_embedder.") for n in r["fp8"])
    torch.testing.assert_close(torch.tensor(r["out"]), tiny.reference[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("parallel", "world", "tp"), [
    ({"tp_degree": 2}, 2, 2),
    ({"tp_degree": 2, "sp_enabled": True}, 2, 2),
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "gather_kv"}, 2, 1),
], ids=["tp2", "tp2-sp", "cp2-gather_kv"])
def test_parallel_layouts_run_the_fp8_twins_as_tp_does(tiny, parallel, world, tp):
    """At tp2 each rank reads its K range of the MLP-downs and of each ``proj_out`` half,
    quantized from exactly those columns; the attention half's twin keeps the block's one
    merged all-reduce and adds the bias once (``skip_bias_add``, no ``reduce_output``). Under SP
    the image stream's ``ff.net.2`` and ``to_out.0`` return their unreduced partial plus the
    whole bias (``reduce_output=False``, ``_sp_unbias``): the FP8 row twin keeps the flag, so
    ``--sp --quant`` gives the ``--quant`` output."""
    for r in _run(tiny, parallel, world, tp):
        assert r["report"] == {"missing": [], "unexpected": []}
        assert len(r["quantized_linears"]) == TINY_TARGETS_HV1
        assert r["attn_twin"] == ["FP8RowParallelLinear", True, False]
        torch.testing.assert_close(torch.tensor(r["out"]), tiny.reference[tp],
                                   atol=1e-4, rtol=1e-4)


def test_eager_and_a_spec_for_another_tp_are_refused(tiny):
    common = dict(model_path=tiny.quantized[1], **TINY_SHAPE)
    with pytest.raises(ValueError, match="D43"):
        _app(exec_mode="eager", quant=tiny.specs[1], **common)
    with pytest.raises(ValueError,
                       match=r"the HunyuanVideo FP8 DiT spec .*tp_degree=2 .*tp_degree=1"):
        _app(exec_mode="compile", quant=tiny.specs[2], **common)


def test_a_target_on_the_fused_proj_out_is_a_dead_pattern_for_the_split_writer(tiny, tmp_path):
    """The directory stores the halves, not the fused weight, so a target on the fused name
    (a spec written before the split) matches nothing the writer can quantize: refused before
    anything is written, naming the halves to target instead."""
    default = default_spec("hunyuan_video", compute="dequant", tp_degree=1)
    fused = r"single_transformer_blocks\.\d+\.proj_out"
    spec = default_spec("hunyuan_video", compute="dequant", tp_degree=1,
                        targets=(*default.targets, fused))
    with pytest.raises(ValueError, match="match no linear weight") as exc:
        prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                     cache_root=tmp_path / "cache")
    assert str(exc.value).endswith(f": {fused}")
    assert ("name single_transformer_blocks.0.proj_out_attn / "
            "single_transformer_blocks.0.proj_out_mlp") in str(exc.value)
    assert not (tmp_path / "cache").exists()


def test_a_target_set_without_the_halves_loads_them_in_bf16_by_name(tiny, tmp_path):
    """The split is the model's checkpoint layout, not a target: a set without the halves
    still stores them as tensors of their own, the source's column ranges in its dtype, and
    the application reads them by name (no slice: there is no fused weight to slice, and the
    load report is clean) into plain row layers."""
    no_halves = tuple(p for p in HUNYUAN_VIDEO_TARGETS if "proj_out" not in p)
    spec = default_spec("hunyuan_video", compute="dequant", tp_degree=1, targets=no_halves)
    assert spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS
    quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    stored, source = _tensors(quantized), _tensors(tiny.transformer_dir)
    inner = TINY_FP8_HV1_TP2["num_attention_heads"] * TINY_FP8_HV1_TP2["attention_head_dim"]
    for i in range(TINY_FP8_HV1_TP2["num_single_layers"]):
        p = f"single_transformer_blocks.{i}"
        w = source[f"{p}.proj_out.weight"]
        assert f"{p}.proj_out.weight" not in stored  # nothing to slice
        assert torch.equal(stored[f"{p}.proj_out_attn.weight"], w[:, :inner])
        assert torch.equal(stored[f"{p}.proj_out_mlp.weight"], w[:, inner:])
        assert f"{p}.proj_out_attn.weight_scale" not in stored  # not quantized
    [r] = run_ranks(hv1_dit_worker, tiny.transformer_dir, {"tp_degree": 1}, "compile",
                    VALID_TEXT, None, spec.to_dict(), str(quantized), world_size=1)
    assert r["report"] == {"missing": [], "unexpected": []}  # a slice would miss proj_out.weight
    assert r["attn_twin"][0] == "RowParallelLinear"
    assert torch.equal(torch.tensor(r["attn_weight"]),
                       source["single_transformer_blocks.0.proj_out.weight"][:, :inner])
    torch.testing.assert_close(torch.tensor(r["out"]),
                               _fake_quant_reference(tiny.transformer_dir, spec),
                               atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# the pipeline application (TorchNeuronHunyuanVideoApplication)
# --------------------------------------------------------------------------


def _pipeline_app(tiny, **kw):
    """The composed application on the CPU, constructed only: the DiT is built (not loaded) and
    the quant options resolved. Construction reads ``transformer/`` and nothing else, so the
    root needs no text encoders or tokenizers."""
    from difflet.models.hunyuan_video.neuron_application import (
        TorchNeuronHunyuanVideoApplication,
    )

    return TorchNeuronHunyuanVideoApplication(
        model_path=str(tiny.root), parallel=kw.pop("parallel", DiffletParallelConfig()),
        dtype=torch.float32, shape=dict(height=64, width=64, num_frames=17), text_seq_len=16,
        device="cpu", exec_mode="compile", **kw)


def test_pipeline_application_resolves_the_quantized_directory(tiny, monkeypatch):
    """``"mxfp8"`` is the default recipe at the launch's tp (the kernel compute; dequant shares
    the directory, whose bytes do not depend on the compute path), and the DiT loads the
    directory `difflet compile` wrote under the cache from the source ``transformer``."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    app = _pipeline_app(tiny, quant="mxfp8")
    assert app.quant == default_spec("hunyuan_video", tp_degree=1)
    assert app.transformer.quant is app.quant
    assert app.transformer.model_path == str(tiny.quantized[1])
    tp2 = _pipeline_app(tiny, quant="mxfp8", parallel=DiffletParallelConfig(tp_degree=2))
    assert tp2.quant.tp_degree == 2 and tp2.transformer.model_path == str(tiny.quantized[2])
    plain = _pipeline_app(tiny)
    assert plain.quant is None and plain.transformer.quant is None
    assert plain.transformer.model_path == str(tiny.transformer_dir)
    explicit = _pipeline_app(tiny, quant=tiny.specs[1].to_dict(),
                             quant_checkpoint=str(tiny.quantized[1]))
    assert explicit.transformer.model_path == str(tiny.quantized[1])


def test_pipeline_application_option_refusals(tiny, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="quant_checkpoint needs quant"):
        _pipeline_app(tiny, quant_checkpoint=str(tiny.quantized[1]))
    with pytest.raises(ValueError, match="quant must be"):
        _pipeline_app(tiny, quant="fp8")
    with pytest.raises(ValueError, match="built with another quantization spec"):
        _pipeline_app(tiny, quant=tiny.specs[1], quant_checkpoint=str(tiny.quantized[2]))
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        _pipeline_app(tiny, quant="mxfp8")
