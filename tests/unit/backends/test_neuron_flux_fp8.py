"""The FP8 (mxfp8) FLUX DiT on the neuron backend, on CPU: the application quantizes the
model's target linears, loads the quantized checkpoint directory and, in dequant mode, matches
diffusers' model run with the dequantized weights; the pipeline application resolves the
quantized directory from the spec; the output projections stay bf16 by default. The single
blocks' fused ``proj_out`` is stored as its attention and MLP halves, read by name: unquantized
under a target set without them, each quantized per rank K range of its own (tp2, gloo) when
targeted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from difflet.backends.neuron.flux.config import NeuronFluxConfig  # noqa: E402
from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import (  # noqa: E402
    FP8ColumnParallelLinear,
    FP8RowParallelLinear,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.fake_linear import SplitLinear, split_linear_columns  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402
from difflet.quant.targets import (  # noqa: E402
    FLUX_SPLIT_TARGETS,
    SINGLE_BLOCK_SPLIT_LINEARS,
    default_spec,
)
from tests.unit.backends._neuron_flux_toy import (  # noqa: E402
    TINY,
    TINY_FP8,
    TINY_FP8_TP2,
    TINY_SHAPE,
    first,
    tiny_inputs,
    write_tiny_flux,
)

# per dual block: q/k/v, add_q/k/v, ff up/down, ff_context up/down = 10; per single block: q/k/v,
# proj_mlp and proj_out's two halves = 6 (to_out, to_add_out stay bf16)
TINY_TARGETS = TINY["num_layers"] * 10 + TINY["num_single_layers"] * 6
#: FLUX's targets before the split (``FLUX_TARGETS`` as of 917d087), written out so that the
#: tests of the halves keep their sets when the default changes.
PRE_D = (
    r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)",
    r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.0\.proj",
    r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.2",
    r"single_transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v)",
    r"single_transformer_blocks\.\d+\.proj_mlp",
)
WITH_HALVES = PRE_D + FLUX_SPLIT_TARGETS
# and per single block the two halves of proj_out
TINY_TARGETS_HALVES = TINY["num_layers"] * 10 + TINY["num_single_layers"] * 6


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def _fake_quant_reference(transformer_dir: str, spec: QuantSpec, config) -> torch.Tensor:
    """Diffusers' model with every weight the spec targets fake-quantized, a split linear run
    as its two column ranges, each quantized on its own (per rank K range of the half, by the
    half's own shard count): what dequant mode computes, since it multiplies the unquantized
    activations by the dequantized weights."""
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel.from_pretrained(transformer_dir).eval()
    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        for name, module in linears:
            if spec.is_split(name):
                attn, mlp = split_linear_columns(module)
                for h, half in (("attn", attn), ("mlp", mlp)):
                    if spec.matches(f"{name}_{h}"):
                        half.weight.data = mx.fake_quantize_mx_sharded(
                            half.weight.data, spec.k_shards(f"{name}_{h}"))
                parent_path, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent_path), leaf,
                        SplitLinear(attn, mlp, split=module.out_features))
            elif spec.matches(name):
                module.weight.data = mx.fake_quantize_mx_sharded(module.weight.data,
                                                                 spec.k_shards(name))
        return model(**tiny_inputs(config), return_dict=False)[0]


def _quantized_fixture(root, spec: QuantSpec) -> SimpleNamespace:
    transformer_dir = write_tiny_flux(root, config=TINY_FP8)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    config = NeuronFluxConfig.from_pretrained(transformer_dir, **TINY_SHAPE)
    return SimpleNamespace(transformer_dir=transformer_dir, quantized=quantized, spec=spec,
                           config=config,
                           reference=_fake_quant_reference(transformer_dir, spec, config))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _quantized_fixture(tmp_path_factory.mktemp("tiny_flux_fp8"),
                              default_spec("flux", compute="dequant", tp_degree=1))


@pytest.fixture(scope="module")
def tiny_halves(tmp_path_factory):
    return _quantized_fixture(tmp_path_factory.mktemp("tiny_flux_fp8_halves"),
                              default_spec("flux", compute="dequant", tp_degree=1,
                                           targets=WITH_HALVES))


def _tensors(directory) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


def _app(model_path, config, spec: QuantSpec) -> NeuronFluxTransformerApplication:
    app = NeuronFluxTransformerApplication(
        model_path=model_path, config=config, parallel=DiffletParallelConfig(),
        dtype=torch.float32, exec_mode="compile", device="cpu", quant=spec)
    app.load()
    return app


def test_the_quantized_directory_holds_the_default_targets_with_the_halves_quantized(tiny):
    """FLUX's default quantizes the single blocks' proj_out as its two halves (D5's
    measurement); the bias stays the fused one and the fused weight is never stored."""
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS
    stored, source = _tensors(tiny.quantized), _tensors(tiny.transformer_dir)
    for i in range(TINY["num_single_layers"]):
        p = f"single_transformer_blocks.{i}"
        for half in ("proj_out_attn", "proj_out_mlp"):
            assert stored[f"{p}.{half}.weight"].dtype == mx.FP8_DTYPE
            assert stored[f"{p}.{half}.weight_scale"].dtype == torch.uint8
        assert torch.equal(stored[f"{p}.proj_out.bias"], source[f"{p}.proj_out.bias"])
        assert f"{p}.proj_out.weight" not in stored


def test_the_quantized_directory_holds_the_halves_not_the_fused_weight(tiny_halves):
    stored = _tensors(tiny_halves.quantized)
    for i in range(TINY["num_single_layers"]):
        p = f"single_transformer_blocks.{i}"
        for half in ("proj_out_attn", "proj_out_mlp"):
            assert stored[f"{p}.{half}.weight"].dtype == mx.FP8_DTYPE
            assert stored[f"{p}.{half}.weight_scale"].dtype == torch.uint8
        assert f"{p}.proj_out.bias" in stored
        assert f"{p}.proj_out.weight" not in stored
    assert read_manifest(tiny_halves.quantized)["tensors"]["quantized"] == TINY_TARGETS_HALVES


def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    app = _app(tiny.quantized, tiny.config, tiny.spec)
    assert len(app.quantized_linears) == TINY_TARGETS
    assert app.load_report == {"missing": [], "unexpected": []}
    dual = app.module.transformer_blocks[0]
    single = app.module.single_transformer_blocks[0]
    assert isinstance(dual.attn.to_q, FP8ColumnParallelLinear)
    assert isinstance(single.proj_mlp, FP8ColumnParallelLinear)
    assert not isinstance(dual.attn.to_out[0], FP8ColumnParallelLinear)
    assert isinstance(single.proj_out_attn, FP8RowParallelLinear)
    assert isinstance(single.proj_out_mlp, FP8RowParallelLinear)
    assert type(dual.attn.to_add_out).__name__ == "RowParallelLinear"
    out = first(app(**tiny_inputs(tiny.config)))
    torch.testing.assert_close(out, tiny.reference, atol=1e-4, rtol=1e-4)


def test_application_quantizes_the_halves_and_matches_the_reference(tiny_halves):
    app = _app(tiny_halves.quantized, tiny_halves.config, tiny_halves.spec)
    single = app.module.single_transformer_blocks[0]
    assert isinstance(single.proj_out_attn, FP8RowParallelLinear)
    assert isinstance(single.proj_out_mlp, FP8RowParallelLinear)
    assert single.proj_out_attn.skip_bias_add is True
    assert len(app.quantized_linears) == TINY_TARGETS_HALVES
    assert app.load_report == {"missing": [], "unexpected": []}
    out = first(app(**tiny_inputs(tiny_halves.config)))
    torch.testing.assert_close(out, tiny_halves.reference, atol=1e-4, rtol=1e-4)


def test_a_target_set_without_the_halves_loads_them_unquantized_by_name(tiny, tmp_path):
    spec = default_spec("flux", compute="dequant", tp_degree=1, targets=PRE_D)
    assert spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS
    quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    app = _app(quantized, tiny.config, spec)
    assert app.load_report == {"missing": [], "unexpected": []}
    single = app.module.single_transformer_blocks[0]
    assert type(single.proj_out_attn).__name__ == "RowParallelLinear"
    fused = _tensors(tiny.transformer_dir)["single_transformer_blocks.0.proj_out.weight"]
    assert torch.equal(single.proj_out_attn.weight, fused[:, :tiny.config.inner_dim])
    out = first(app(**tiny_inputs(tiny.config)))
    reference = _fake_quant_reference(tiny.transformer_dir, spec, tiny.config)
    torch.testing.assert_close(out, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.multirank
def test_tp2_quantized_halves_match_the_per_rank_reference(tmp_path):
    """At tp2 each rank reads its K range of each half, quantized from exactly those columns;
    the attention half's twin keeps the merged all-reduce and adds the bias once."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import flux_dit_fp8_worker

    transformer_dir = write_tiny_flux(tmp_path, config=TINY_FP8_TP2)
    spec = default_spec("flux", compute="dequant", tp_degree=2, targets=WITH_HALVES)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    reference = _fake_quant_reference(
        transformer_dir, spec, NeuronFluxConfig.from_pretrained(transformer_dir, **TINY_SHAPE))
    results = run_ranks(flux_dit_fp8_worker, str(quantized), spec.to_dict(), world_size=2)
    for r in results:
        assert r["report"] == {"missing": [], "unexpected": []}
        assert r["attn_twin"] == ["FP8RowParallelLinear", True, False]
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-4, rtol=1e-4)


def test_application_accepts_the_spec_as_a_dict_and_refuses_eager(tiny):
    kwargs = dict(model_path=tiny.quantized, config=tiny.config, parallel=DiffletParallelConfig(),
                  dtype=torch.float32, device="cpu")
    app = NeuronFluxTransformerApplication(exec_mode="compile", quant=tiny.spec.to_dict(), **kwargs)
    assert app.quant == tiny.spec
    with pytest.raises(ValueError, match="D43"):
        NeuronFluxTransformerApplication(exec_mode="eager", quant=tiny.spec, **kwargs)


def test_pipeline_application_resolves_the_quantized_directory(tiny, monkeypatch):
    from difflet.models.flux.neuron_application import TorchNeuronFluxApplication

    app = TorchNeuronFluxApplication.__new__(TorchNeuronFluxApplication)
    app.parallel = DiffletParallelConfig()
    # <cache_root>/quantized/<label>/<model_dir.name>/<format>-<digest>
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.quantized.parents[3]))
    spec, path = app._resolve_quant({"quant": tiny.spec, "quant_checkpoint": None},
                                    tiny.transformer_dir)
    assert spec == tiny.spec and path == str(tiny.quantized)
    spec, path = app._resolve_quant({"quant": None, "quant_checkpoint": None}, tiny.transformer_dir)
    assert spec is None and path == tiny.transformer_dir
    with pytest.raises(ValueError, match="quant must be"):
        app._resolve_quant({"quant": "fp8", "quant_checkpoint": None}, tiny.transformer_dir)
    explicit, path = app._resolve_quant({"quant": tiny.spec, "quant_checkpoint": str(tiny.quantized)},
                                        tiny.transformer_dir)
    assert path == str(tiny.quantized)
