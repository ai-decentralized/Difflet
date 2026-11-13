"""The MXFP8 Wan DiT on the neuron backend, on CPU: the application quantizes the attention
q/k/v and the FFN through ``checkpoint_key`` (the table is written against the checkpoint keys
``ffn.net.0.proj`` / ``ffn.net.2``; the module names its FFN ``net_in`` / ``net_out``), loads the
quantized checkpoint directory and, in dequant mode, matches diffusers' model run with the
dequantized fp8 weights; every output projection, the modulation tables and the top level stay
as they were; eager mode and a spec quantized for another tp are refused. A quantized
self-attention q or k puts every self-attention of the module in the half-split rotary layout
(the q/k rows reordered before the twins are prepared), on either compute path and at tp2, and
the output is still the interleaved reference's; a set without them keeps the interleaved
layout. The pipeline application resolves one quantized directory per expert from the expert's
bf16 copy with one spec for both of Wan 2.2's experts, refuses ``quant_checkpoint`` with two
experts, and counts both quantized experts as stored in the HBM check beside the device text
encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
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
from difflet.cli.quant import QUANT_TARGETS_ENV, quant_spec  # noqa: E402
from difflet.models.wan.modeling_wan import half_split_index  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402
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

#: Per block: the q/k/v of the self- and the cross-attention and the FFN's up and down; every
#: to_out.0, the modulation tables, the embedders and proj_out stay bf16.
TARGETS_PER_BLOCK = 8
TINY_TARGETS_WAN = TARGETS_PER_BLOCK * TINY_FP8_WAN["num_layers"]
#: The targets in the module's spelling (``quantized_linears`` addresses the module) ...
MODULE_TARGETS = {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q", "attn2.to_k",
                  "attn2.to_v", "ffn.net_in", "ffn.net_out"}
#: ... and in the checkpoint's (what the table and the quantized directory name).
CHECKPOINT_TARGETS = (MODULE_TARGETS - {"ffn.net_in", "ffn.net_out"}) | {"ffn.net.0.proj",
                                                                         "ffn.net.2"}
#: The 917d087 default: the cross-attention's q/k/v and the FFN, the self-attention's in bf16.
CROSS_ATTENTION_AND_FFN = (r"blocks\.\d+\.attn2\.(to_q|to_k|to_v)",
                           r"blocks\.\d+\.ffn\.net\.0\.proj", r"blocks\.\d+\.ffn\.net\.2")
REORDER_LINE = "[wan] self-attention q/k reordered for the half-split rotary"


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def _reference(transformer_dir, spec: QuantSpec) -> torch.Tensor:
    """diffusers' own model with every weight ``spec`` targets dequantized from its mxfp8 form
    (a row-parallel one per rank K range): what dequant mode computes, in the checkpoint's
    channel order and with the interleaved rotary. Its module names are the checkpoint keys, so
    the table matches them as it is."""
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if spec.matches(name) and isinstance(module, torch.nn.Linear):
                module.weight.copy_(mx.fake_quantize_mx_sharded(module.weight,
                                                                spec.k_shards(name)))
        return model(*tiny_wan_inputs(config=TINY_FP8_WAN), return_dict=False)[0]


def _train_self_attention_qk(transformer_dir: Path) -> None:
    """Per-channel self-attention q/k norm weights and q/k rows of unlike magnitude, as a
    checkpoint has. The toy writes the norm weights as ones and q/k rows of one magnitude (one
    E8M0 block scale for every row), which the half-split reorder leaves as they are, so a
    reorder that missed the norms or the block scales (or a rank that sliced the wrong heads of
    them) would go unseen. A power-of-two factor per row keeps each row's fp8 codes and moves
    its block scale only."""
    from safetensors.torch import load_file, save_file

    path = transformer_dir / "diffusion_pytorch_model.safetensors"
    state = load_file(str(path))
    gen = torch.Generator().manual_seed(7)
    for key in sorted(k for k in state if ".attn1.norm_q." in k or ".attn1.norm_k." in k):
        state[key] = 0.5 + torch.rand(state[key].shape, generator=gen)
    for key in (k for k in state if k.endswith((".attn1.to_q.weight", ".attn1.to_k.weight"))):
        rows = state[key].shape[0]
        state[key] = state[key] * (2.0 ** (torch.arange(rows) % 4))[:, None]
    save_file(state, str(path))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_wan_fp8")
    write_tiny_wan_model(root, config=TINY_FP8_WAN)
    transformer_dir = root / "transformer"
    _train_self_attention_qk(transformer_dir)
    spec = default_spec("wan", compute="dequant", tp_degree=1)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, quantized=quantized,
                           spec=spec, reference=_reference(transformer_dir, spec))


def _app(tiny, **overrides):
    kwargs = dict(model_path=tiny.quantized, config=tiny_wan_config(config=TINY_FP8_WAN),
                  parallel=DiffletParallelConfig(), dtype=torch.float32, exec_mode="compile",
                  device="cpu", quant=tiny.spec)
    kwargs.update(overrides)
    return NeuronWanTransformerApplication(**kwargs)


def _override(monkeypatch, *targets: str) -> QuantSpec:
    """The tp1 dequant spec the driver builds with ``targets`` as ``DIFFLET_QUANT_TARGETS``."""
    monkeypatch.setenv(QUANT_TARGETS_ENV, ";".join(targets))
    args = argparse.Namespace(quant="mxfp8", quant_compute="dequant", tp_degree=1,
                              model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    return quant_spec(args)


def _loaded(tiny, spec: QuantSpec, tmp_path):
    """The application of ``spec`` on the toy's quantized directory for it, loaded."""
    quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    app = _app(tiny, model_path=quantized, quant=spec)
    app.load()
    return app


def _rope_layouts(module) -> set[str]:
    """The model's rotary layout and every self-attention's: one value when they agree."""
    return {module.rope_layout, *(block.attn1.rope_layout for block in module.blocks)}


def test_the_quantized_directory_holds_every_target(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == TINY_TARGETS_WAN
    assert manifest["spec"]["format"] == "mxfp8"
    # The writer matched the checkpoint keys: the FFN is there under its on-disk names.
    names = manifest["quantized_linears"]
    assert len(names) == TINY_TARGETS_WAN
    assert {n.split(".", 2)[2] for n in names} == CHECKPOINT_TARGETS


def test_application_quantizes_through_the_checkpoint_key(tiny, capsys):
    """The spec's targets are matched by checkpoint key, so the FFN's ``net_in`` / ``net_out``
    (``net.0.proj`` / ``net.2`` on disk) are quantized with the attention q/k/v and loaded
    from the directory cleanly, while the names recorded are the module's own. The module's
    linears are the selected backend's TP layers (the CPU backend's here, ``difflet.ops``), so
    the twins are the column and the row twin, at tp=1 as on the device at tp4. The quantized
    self-attention q/k put the model and every self-attention in the half-split rotary layout,
    said in one line, and the output is still the interleaved reference's."""
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
    assert app.module.rope_layout == "half_split"
    assert all(b.attn1.rope_layout == "half_split" for b in app.module.blocks)
    assert capsys.readouterr().out.count(REORDER_LINE) == 1
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


def test_a_set_without_self_attention_qk_keeps_the_interleaved_rotary(tiny, monkeypatch,
                                                                       tmp_path, capsys):
    """The half split is for q/k from the MXFP8 kernel (bf16 q/k run 12% slower with it): a set
    whose self-attention q/k stay bf16, the 917d087 default here, keeps the checkpoint's order
    and the interleaved rotary."""
    spec = _override(monkeypatch, *CROSS_ATTENTION_AND_FFN)
    app = _loaded(tiny, spec, tmp_path)
    assert {n.split(".", 2)[2] for n in app.quantized_linears} == (
        MODULE_TARGETS - {"attn1.to_q", "attn1.to_k", "attn1.to_v"})
    assert _rope_layouts(app.module) == {"interleaved"}
    assert REORDER_LINE not in capsys.readouterr().out
    torch.testing.assert_close(app(*tiny_wan_inputs(config=TINY_FP8_WAN)),
                               _reference(tiny.transformer_dir, spec), atol=1e-4, rtol=1e-4)


def test_quantizing_one_block_qk_switches_every_self_attention(tiny, monkeypatch, tmp_path):
    """The rotary tables are the model's, one layout per forward: one quantized q (block 0's)
    reorders the bf16 q/k of every self-attention with it."""
    spec = _override(monkeypatch, r"blocks\.0\.attn1\.to_q")
    app = _loaded(tiny, spec, tmp_path)
    assert app.quantized_linears == ["blocks.0.attn1.to_q"]
    assert _rope_layouts(app.module) == {"half_split"}
    torch.testing.assert_close(app(*tiny_wan_inputs(config=TINY_FP8_WAN)),
                               _reference(tiny.transformer_dir, spec), atol=1e-4, rtol=1e-4)


def test_the_dequant_path_uses_the_same_layout(tiny):
    """The layout follows what is quantized, not how it multiplies: the dequant path (the
    fixture's spec) switches as the kernel path (the device's) does, and both prepare the
    reordered rows. The kernel path's packed q/k are the kernel layout of the directory's fp8
    rows and block scales in the half-split order, so the reorder ran before the packing; it
    loads without its warm-up forward, since the kernel needs the Neuron SDK. The dequant
    path's output is the interleaved reference's. The fixture's q/k rows differ in block scale
    within a head, so scales left in the checkpoint's order fail this test, the tp2 case and
    every reference match whose set quantizes them."""
    from safetensors.torch import load_file

    assert tiny.spec.compute == "dequant"
    dequant = _app(tiny)
    dequant.load()
    # One directory serves both paths: the compute path is not in its digest.
    kernel = _app(tiny, quant=default_spec("wan", compute="nki", tp_degree=1))
    kernel.load(skip_warmup=True)
    assert kernel.load_report == {"missing": [], "unexpected": []}
    assert _rope_layouts(dequant.module) == _rope_layouts(kernel.module) == {"half_split"}
    stored = {}
    for path in Path(tiny.quantized).glob("*.safetensors"):
        stored.update(load_file(str(path)))
    head_dim = TINY_FP8_WAN["attention_head_dim"]
    for i, block in enumerate(kernel.module.blocks):
        for proj in ("to_q", "to_k"):
            weight = stored[f"blocks.{i}.attn1.{proj}.weight"]
            scale = stored[f"blocks.{i}.attn1.{proj}.weight_scale"]
            perm = half_split_index(weight.shape[0], head_dim)
            assert not torch.equal(scale.index_select(0, perm), scale), (i, proj)
            data, scales = mx.kernel_layout(weight.index_select(0, perm),
                                            scale.index_select(0, perm))
            twin = getattr(block.attn1, proj)
            assert torch.equal(twin.weight_packed, data), (i, proj)
            assert torch.equal(twin.weight_scales_hw, scales), (i, proj)
    torch.testing.assert_close(dequant(*tiny_wan_inputs(config=TINY_FP8_WAN)), tiny.reference,
                               atol=1e-4, rtol=1e-4)


@pytest.mark.multirank
def test_tp2_matches_the_reference(tiny, tmp_path):
    """At tp2 each rank reorders the q/k rows of its own heads (a column shard is whole heads),
    slices its heads of the reordered q/k norm weights and reads ``ffn.net.2`` per rank K
    range: every rank is in the half-split layout and its output is the reference's."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import wan_fp8_worker

    spec = default_spec("wan", compute="dequant", tp_degree=2)
    quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    reference = _reference(tiny.transformer_dir, spec)
    results = run_ranks(wan_fp8_worker, str(quantized), spec.to_dict(), world_size=2)
    for r in results:
        assert r["report"] == {"missing": [], "unexpected": []}
        assert r["rope_layouts"] == ["half_split"]
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-4, rtol=1e-4)


def test_a_weight_store_hit_is_reordered_once(tiny, tmp_path, monkeypatch):
    """The weight store is written while the checkpoint loads, before the reorder, so it keeps
    the checkpoint's order: the next application reading it reorders the rows once, as the
    first did, and computes the same output (a store written after the reorder would be
    reordered twice and miss the reference). A second ``load()`` of a loaded application
    returns at once, and its module, in the half-split layout already, reorders nothing."""
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")  # an fp32 toy saves nothing
    inputs = tiny_wan_inputs(config=TINY_FP8_WAN)
    first = _app(tiny)
    first.load()
    assert first.weight_store.status == "written"
    parallel_mesh.destroy_parallel_mesh()
    second = _app(tiny)
    second.load()
    assert second.weight_store.status == "hit"
    out = second(*inputs)
    assert torch.equal(out, first(*inputs))
    second.load()
    assert second.module.use_half_split_rope_() == 0
    assert torch.equal(second(*inputs), out)
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
