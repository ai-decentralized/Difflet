"""The MXFP8 Wan DiT on the neuron backend, on CPU: the application quantizes the attention
q/k/v and the FFN through ``checkpoint_key`` (the table is written against the checkpoint keys
``ffn.net.0.proj`` / ``ffn.net.2``; the module names its FFN ``net_in`` / ``net_out``), loads the
quantized checkpoint directory and, in dequant mode, matches diffusers' model run with the
dequantized fp8 weights; every output projection, the modulation tables and the top level stay
as they were; eager mode and a spec quantized for another tp are refused. The pipeline
application resolves one quantized directory per expert from the expert's bf16 copy with one
spec for both of Wan 2.2's experts, refuses ``quant_checkpoint`` with two experts, and counts
both quantized experts as stored in the HBM check beside the device text encoder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")
pytest.importorskip("safetensors.torch")
pytest.importorskip("transformers")

from difflet.backends.neuron.core.weights_bf16 import ensure_bf16_copy  # noqa: E402
from difflet.backends.neuron.hbm import FIT_FRACTION, GIB, HbmBudgetError  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import (  # noqa: E402
    FP8ColumnParallelLinear,
    FP8RowParallelLinear,
    _FP8LinearBase,
)
from difflet.backends.neuron.wan.hbm import device_text_encoder_bytes  # noqa: E402
from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.targets import default_spec  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_FP8_WAN,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    tiny_wan_config,
    tiny_wan_inputs,
    write_tiny_wan22_model,
    write_tiny_wan_model,
)

#: Per block: q/k/v of the self- and the cross-attention and the FFN's up and down; every
#: to_out.0, the modulation tables, the embedders and proj_out stay bf16.
TARGETS_PER_BLOCK = 8
TINY_TARGETS_WAN = TARGETS_PER_BLOCK * TINY_FP8_WAN["num_layers"]
#: The targets in the module's spelling (``quantized_linears`` addresses the module) ...
MODULE_TARGETS = {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k",
                  "attn2.to_v", "ffn.net_in", "ffn.net_out"}
#: ... and in the checkpoint's (what the table and the quantized directory name).
CHECKPOINT_TARGETS = (MODULE_TARGETS - {"ffn.net_in", "ffn.net_out"}) | {"ffn.net.0.proj",
                                                                         "ffn.net.2"}


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import WanTransformer3DModel

    root = tmp_path_factory.mktemp("tiny_wan_fp8")
    write_tiny_wan_model(root, config=TINY_FP8_WAN)
    transformer_dir = root / "transformer"
    spec = default_spec("wan", compute="dequant", tp_degree=1)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    # diffusers' own model with the dequantized mxfp8 weights: what dequant mode computes. Its
    # module names are the checkpoint keys, so the table matches them as it is.
    model = WanTransformer3DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval()
    for name, module in model.named_modules():
        if spec.matches(name) and isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.copy_(mx.dequantize_mx(*mx.quantize_mx(module.weight.detach())))
    latents, timestep, text = tiny_wan_inputs(config=TINY_FP8_WAN)
    with torch.no_grad():
        reference = model(latents, timestep, text, return_dict=False)[0]
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, quantized=quantized,
                           spec=spec, reference=reference)


def _app(tiny, **overrides):
    kwargs = dict(model_path=tiny.quantized, config=tiny_wan_config(config=TINY_FP8_WAN),
                  parallel=DiffletParallelConfig(), dtype=torch.float32, exec_mode="compile",
                  device="cpu", quant=tiny.spec)
    kwargs.update(overrides)
    return NeuronWanTransformerApplication(**kwargs)


def test_the_quantized_directory_holds_every_target(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS_WAN
    assert manifest["spec"]["format"] == "mxfp8"
    # The writer matched the checkpoint keys: the FFN is there under its on-disk names.
    names = manifest["quantized_linears"]
    assert len(names) == TINY_TARGETS_WAN
    assert {n.split(".", 2)[2] for n in names} == CHECKPOINT_TARGETS


def test_application_quantizes_through_the_checkpoint_key(tiny):
    """The spec's targets are matched by checkpoint key, so the FFN's ``net_in`` / ``net_out``
    (``net.0.proj`` / ``net.2`` on disk) are quantized with the attention q/k/v and loaded
    from the directory cleanly, while the names recorded are the module's own. The module's
    linears are the selected backend's TP layers (the CPU backend's here, ``difflet.ops``), so
    the twins are the column and the row twin, at tp=1 as on the device at tp4."""
    app = _app(tiny)
    app.load()
    names = app.quantized_linears
    assert len(names) == TINY_TARGETS_WAN
    assert {n.split(".", 2)[2] for n in names} == MODULE_TARGETS
    assert app.load_report == {"missing": [], "unexpected": []}
    block = app.module.blocks[0]
    assert isinstance(block.ffn.net_in, FP8ColumnParallelLinear)
    assert isinstance(block.ffn.net_out, FP8RowParallelLinear)
    assert isinstance(block.attn1.to_q, FP8ColumnParallelLinear)
    assert isinstance(block.attn2.to_v, FP8ColumnParallelLinear)
    assert block.ffn.net_in.weight is None and block.ffn.net_in.weight_bf16 is not None  # prepared
    # Every output projection and the top level are the linears they were; the modulation
    # tables are parameters in the module's dtype.
    for linear in (block.attn1.to_out[0], block.attn2.to_out[0], app.module.proj_out):
        assert isinstance(linear, torch.nn.Linear) and not isinstance(linear, _FP8LinearBase)
    assert not any(isinstance(m, _FP8LinearBase) for m in app.module.condition_embedder.modules())
    assert block.scale_shift_table.dtype == app.module.scale_shift_table.dtype == torch.float32
    out = app(*tiny_wan_inputs(config=TINY_FP8_WAN))
    # Exact up to fp32 summation order: a dequantized fp8 value is bf16-representable, so the
    # prepared bf16 weight copy is the reference's weight.
    torch.testing.assert_close(out, tiny.reference, atol=1e-4, rtol=1e-4)


def test_eager_is_refused(tiny):
    assert _app(tiny, quant=tiny.spec.to_dict()).quant == tiny.spec
    with pytest.raises(ValueError, match="D43"):
        _app(tiny, exec_mode="eager")


def test_a_spec_for_another_tp_is_refused(tiny):
    """A3's check, the Wan instance: ``ffn.net.2`` is row-parallel, quantized per rank K
    range, so a directory written for tp2 holds scales grouped for tp2's K halves, which this
    tp1 launch would read as the wrong blocks; refused at construction, naming both tps."""
    with pytest.raises(ValueError, match=r"the Wan FP8 DiT spec .*tp_degree=2 .*tp_degree=1"):
        _app(tiny, quant=default_spec("wan", compute="dequant", tp_degree=2))


# --------------------------------------------------------------------------
# the pipeline application (TorchNeuronWanApplication): two experts, one spec
# --------------------------------------------------------------------------

EXPERTS = ("transformer", "transformer_2")


@pytest.fixture(scope="module")
def tiny22(tmp_path_factory):
    """The Wan 2.2 toy with what ``difflet compile`` leaves under one cache root for a
    ``--quant`` launch: each expert's bf16 copy (the hub DiTs are fp32, read once by
    ``ensure_bf16_copy``) and, built from that copy, the expert's quantized directory. The
    tests point ``DIFFLET_COMPILE_CACHE`` at the root, where the application looks for both."""
    root = tmp_path_factory.mktemp("tiny_wan22_fp8")
    write_tiny_wan22_model(root, config=TINY_FP8_WAN)
    cache = root / "cache"
    spec = default_spec("wan", tp_degree=1)
    copies, quantized = {}, {}
    for name in EXPERTS:
        copies[name] = ensure_bf16_copy(root / name, cache)
        quantized[name] = prepare_quantized_checkpoint(copies[name], spec, create=True,
                                                       cache_root=cache)
    return SimpleNamespace(root=root, cache=cache, spec=spec, copies=copies,
                           quantized=quantized)


def _pipeline_app(tiny, **overrides):
    """The composed application on the CPU with the text encoder off: the experts are built
    (not loaded) and the quant options resolved."""
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication

    kwargs = dict(model_path=str(tiny.root), parallel=DiffletParallelConfig(),
                  dtype=torch.float32, shape=dict(TINY_SHAPE), text_seq_len=TINY_TEXT_SEQ_LEN,
                  exec_mode="compile", device="cpu", enable_text_encoder=False)
    kwargs.update(overrides)
    return TorchNeuronWanApplication(**kwargs)


def test_pipeline_resolves_one_quantized_directory_per_expert(tiny, tiny22, monkeypatch):
    """Both experts take the one spec (one block graph: one target table at the launch's tp)
    and each loads its own quantized directory, resolved from the directory the expert would
    load in bf16, its copy (``resolve_component_dir``), so the writer read bf16 tensors and the
    hub's fp32 files never; nothing names a second directory, so a launch that quantizes one
    expert and not the other cannot be built."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny22.cache))
    app = _pipeline_app(tiny22, dtype=torch.bfloat16, enable_transformer_2=True, quant="mxfp8")
    assert app.quant == default_spec("wan", tp_degree=1) and app.quant.compute == "nki"
    assert app.transformer.quant is app.quant and app.transformer_2.quant is app.quant
    expected = {name: str(path) for name, path in tiny22.quantized.items()}
    assert app.quantized_dits == expected
    assert app.transformer.model_path == expected["transformer"]
    assert app.transformer_2.model_path == expected["transformer_2"]
    assert expected["transformer"] != expected["transformer_2"]
    # Each directory was built from the expert's bf16 copy and names its fp32 source (C2).
    for name, path in tiny22.quantized.items():
        assert read_manifest(path)["source"]["bf16_copy_of"] == str((tiny22.root / name).resolve())
    # Without quant the experts load their bf16 copies and there is no quantized directory.
    app = _pipeline_app(tiny22, dtype=torch.bfloat16, enable_transformer_2=True)
    assert app.quant is None and app.quantized_dits == {}
    assert app.transformer.model_path == str(tiny22.copies["transformer"])
    assert app.transformer_2.model_path == str(tiny22.copies["transformer_2"])
    # One expert (Wan 2.1) in fp32: no copy is loaded, so the directory is the fp32 source's.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny.root / "cache"))
    app = _pipeline_app(tiny, quant="mxfp8")
    assert app.quantized_dits == {"transformer": str(tiny.quantized)}
    assert app.transformer.model_path == str(tiny.quantized) and app.transformer_2 is None


def test_quant_checkpoint_is_refused_with_two_experts(tiny, tiny22, monkeypatch, tmp_path):
    """``quant_checkpoint`` names one directory and Wan 2.2's experts each need their own, so
    with the low-noise expert it is refused before anything is read; one expert takes it as
    the other pipeline applications do, and the resolver's refusals reach the caller."""
    one = tiny22.quantized["transformer"]
    with pytest.raises(ValueError, match="quant_checkpoint names one directory"):
        _pipeline_app(tiny22, dtype=torch.bfloat16, enable_transformer_2=True, quant="mxfp8",
                      quant_checkpoint=str(one))
    app = _pipeline_app(tiny, quant=tiny.spec.to_dict(), quant_checkpoint=str(tiny.quantized))
    assert app.quant == tiny.spec and app.quantized_dits == {"transformer": str(tiny.quantized)}
    with pytest.raises(ValueError, match="needs quant"):
        _pipeline_app(tiny, quant_checkpoint=str(tiny.quantized))
    with pytest.raises(ValueError, match="quant must be"):
        _pipeline_app(tiny, quant="fp8")
    # A cache without the copies: the experts resolve from the fp32 source, whose quantized
    # directory `difflet compile` has not written either.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="difflet compile"):
        _pipeline_app(tiny22, dtype=torch.bfloat16, enable_transformer_2=True, quant="mxfp8")


def test_hbm_counts_both_quantized_experts(tiny22, monkeypatch):
    """umT5 on the device beside Wan 2.2's two experts: a quantized expert is counted as its
    directory stores it (fp8 weights, uint8 block scales, the bf16 rest), and both of them
    are. At a budget between the two bf16 experts and the two quantized ones, one bf16 expert
    and the quantized pair are admitted and the bf16 pair refused; 1 KiB under the quantized
    pair's bytes refuses it, so both directories were read."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tiny22.cache))

    def estimate(experts, dit_dirs=None):
        return device_text_encoder_bytes(tiny22.root, tp_degree=1, experts=experts,
                                         dit_dtype_bytes=2, encoder_dtype_bytes=2,
                                         dit_dirs=dit_dirs)

    two_bf16 = estimate(EXPERTS)
    two_quantized = estimate(EXPERTS, dit_dirs=tiny22.quantized)
    assert estimate(EXPERTS[:1]) < two_quantized < two_bf16

    def budget(limit_bytes: int) -> None:
        """The env budget whose ``FIT_FRACTION`` is ``limit_bytes``."""
        monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", repr(limit_bytes / FIT_FRACTION / GIB))

    def build(**overrides):
        return _pipeline_app(tiny22, dtype=torch.bfloat16, enable_text_encoder=True,
                             text_encoder_device="neuron", **overrides)

    budget((two_quantized + two_bf16) // 2)
    build()
    with pytest.raises(HbmBudgetError, match=r"transformer, transformer_2\) on one core"):
        build(enable_transformer_2=True)
    assert len(build(enable_transformer_2=True, quant="mxfp8").quantized_dits) == 2
    budget(two_quantized - 1024)
    with pytest.raises(HbmBudgetError, match=r"umT5 .* beside the Wan DiT \(transformer, "):
        build(enable_transformer_2=True, quant="mxfp8")
