"""The FP8 (mxfp8) FLUX DiT on the neuron backend, on CPU: the application quantizes the
model's target linears, loads the quantized checkpoint directory and, in dequant mode, matches
diffusers' model run with the dequantized weights; the pipeline application resolves the
quantized directory from the spec; the output projections stay bf16.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from difflet.backends.neuron.flux.config import NeuronFluxConfig  # noqa: E402
from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import FP8ColumnParallelLinear  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402
from tests.unit.backends._neuron_flux_toy import TINY, TINY_SHAPE, first, tiny_inputs  # noqa: E402

#: The toy's dims with a 128-wide inner dimension: the MX block layout wants K in multiples of
#: 128 (``difflet/quant/mx.swizzle_perm``), the toy's 32 is below that.
TINY_FP8 = {**TINY, "attention_head_dim": 32, "axes_dims_rope": (8, 12, 12)}

# per dual block: q/k/v, add_q/k/v, ff up/down, ff_context up/down = 10; per single block: q/k/v,
# proj_mlp = 4 (to_out, to_add_out, proj_out stay bf16)
TINY_TARGETS = TINY["num_layers"] * 10 + TINY["num_single_layers"] * 4


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import FluxTransformer2DModel

    root = tmp_path_factory.mktemp("tiny_flux_fp8")
    torch.manual_seed(0)
    transformer_dir = f"{root}/transformer"
    FluxTransformer2DModel(**TINY_FP8).save_pretrained(transformer_dir, safe_serialization=True)
    spec = default_spec("flux", format="mxfp8", compute="dequant", tp_degree=1)
    quantized, spec = prepare_quantized_checkpoint(transformer_dir, spec, None, create=True,
                                                   cache_root=root / "cache")
    config = NeuronFluxConfig.from_pretrained(transformer_dir, **TINY_SHAPE)
    # diffusers' own model with the dequantized mxfp8 weights: what dequant mode computes.
    model = FluxTransformer2DModel.from_pretrained(transformer_dir).eval()
    for name, module in model.named_modules():
        if spec.matches(name) and isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.copy_(mx.dequantize_mx(*mx.quantize_mx(module.weight.detach())))
    with torch.no_grad():
        reference = model(**tiny_inputs(config), return_dict=False)[0]
    return SimpleNamespace(transformer_dir=transformer_dir, quantized=quantized, spec=spec,
                           config=config, reference=reference)


def test_the_quantized_directory_holds_the_default_targets_only(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS


def test_application_quantizes_the_targets_and_matches_the_weight_only_reference(tiny):
    app = NeuronFluxTransformerApplication(
        model_path=tiny.quantized, config=tiny.config, parallel=DiffletParallelConfig(),
        dtype=torch.float32, exec_mode="compile", device="cpu", quant=tiny.spec)
    app.load()
    assert len(app.quantized_linears) == TINY_TARGETS
    assert app.load_report == {"missing": [], "unexpected": []}
    dual = app.module.transformer_blocks[0]
    single = app.module.single_transformer_blocks[0]
    assert isinstance(dual.attn.to_q, FP8ColumnParallelLinear)
    assert isinstance(single.proj_mlp, FP8ColumnParallelLinear)
    assert not isinstance(dual.attn.to_out[0], FP8ColumnParallelLinear)
    assert type(single.proj_out_attn).__name__ == "RowParallelLinear"
    assert type(dual.attn.to_add_out).__name__ == "RowParallelLinear"
    out = first(app(**tiny_inputs(tiny.config)))
    torch.testing.assert_close(out, tiny.reference, atol=1e-4, rtol=1e-4)


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
    with pytest.raises(ValueError, match="calibration"):
        app._resolve_quant({"quant": "fp8", "quant_checkpoint": None}, tiny.transformer_dir)
    explicit, path = app._resolve_quant({"quant": tiny.spec, "quant_checkpoint": str(tiny.quantized)},
                                        tiny.transformer_dir)
    assert path == str(tiny.quantized)
