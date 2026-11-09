"""The MXFP8 HunyuanVideo-1.5 DiT on the neuron backend, on CPU: the application quantizes the
set the device measured, the dual-stream blocks' image q/k/v and image MLP (an override since
Task F8: the model has no default set; diffusers' module names are the checkpoint keys, so the
patterns match them as they are), loads the quantized directory and, in dequant mode,
matches diffusers' model run with the weights the directory holds dequantized, at tp1,
tp2 x CFG2, tp2 with SP and tp2 x CP2. Each rank's row-parallel weights are its own K range
quantized as such, compared exactly, since the outputs cannot tell. The text stream's q/k/v
and MLP, the output projections, the modulation, the token refiner and the top level stay as
they were; eager mode and a spec quantized for another tp are refused. The native path has no
stage cache key, so what keeps a quantized and a bf16 DiT apart is checked here: their block
graphs differ (the kernel call per target, which the NEFF cache keys on through the HLO) and so
do their weight-store entries, by layout as well as by source. The bytes a loaded DiT holds,
which the transformer stage reports, count the kernel's scale layout that the pre-load estimate
omits.
The pipeline resolves the quantized directory from the bf16 copy (the hub DiT is fp32 and
``difflet compile`` quantizes the copy) at the spec's tp, loads the copy without ``quant``,
refuses a launch whose directory was never written, and refuses ``quant="mxfp8"``, which names
no layers for this model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from difflet.backends.neuron.core.weights_bf16 import ensure_bf16_copy  # noqa: E402
from difflet.backends.neuron.hunyuan_video15.pipeline import (  # noqa: E402
    TorchNeuronHunyuanVideo15Pipeline,
)
from difflet.backends.neuron.hunyuan_video15.transformer import (  # noqa: E402
    NeuronHunyuanVideo15TransformerApplication,
)
from difflet.backends.neuron.ops_impl import fp8_kernels, parallel_mesh  # noqa: E402
from difflet.backends.neuron.ops_impl.fp8_linear import FP8Linear, _FP8LinearBase  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.quant import mx  # noqa: E402
from difflet.quant.checkpoint import prepare_quantized_checkpoint, read_manifest  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_hv15_toy import (  # noqa: E402
    HV15_ROW_TARGETS,
    TINY_FP8_HV15,
    TINY_SHAPE_HV15,
    TINY_TARGETS_PER_BLOCK,
    hv15_reference,
    hv15_spec,
    tiny_hv15_inputs,
    write_tiny_hv15_snapshot,
)
from tests.unit.backends._neuron_workers import _recording_backend, hv15_fp8_worker  # noqa: E402

#: The toy's default targets: five per dual-stream block.
N_TARGETS = len(TINY_TARGETS_PER_BLOCK) * TINY_FP8_HV15["num_layers"]  # 10


@torch.library.custom_op("difflet_hv15_test::mx_linear_stub", mutates_args=())
def _mx_linear_stub(x: torch.Tensor, data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The MX kernel op's signature and output shape without the Neuron SDK: one node per
    kernel call in a traced graph, which is all the graph test reads."""
    return torch.zeros(x.shape[0], data.shape[1], dtype=torch.bfloat16, device=x.device)


@_mx_linear_stub.register_fake
def _mx_linear_stub_fake(x, data, scales):
    return x.new_empty(x.shape[0], data.shape[1], dtype=torch.bfloat16)


def _stub_calls(graphs) -> list[int]:
    """The kernel calls in each recorded graph."""
    return [sum("mx_linear_stub" in str(node.target) for node in gm.graph.nodes) for gm in graphs]


def _store_key(app) -> dict:
    """The key inputs the weight-store entry ``app`` wrote, as its manifest records them (one
    data file at tp1: everything is replicated)."""
    (manifest,) = Path(app.weight_store.directory).glob("*.manifest.json")
    return json.loads(manifest.read_text())["key"]


@pytest.fixture(autouse=True)
def _reset_mesh():
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The FP8 toy snapshot, the measured set's dequant spec at tp1, its quantized directory under
    the root's cache (written from the fp32 ``transformer/``: dequant mode computes in the
    application's dtype, fp32 here) and the reference that spec computes."""
    root = write_tiny_hv15_snapshot(tmp_path_factory.mktemp("tiny_hv15_fp8"))
    transformer_dir = root / "transformer"
    spec = hv15_spec(compute="dequant", tp_degree=1)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    reference = hv15_reference(transformer_dir, tiny_hv15_inputs(), spec=spec)
    return SimpleNamespace(root=root, transformer_dir=transformer_dir, spec=spec,
                           quantized=quantized, reference=reference)


def _app(tiny, **overrides) -> NeuronHunyuanVideo15TransformerApplication:
    kwargs = dict(model_path=tiny.quantized, dtype=torch.float32, exec_mode="compile",
                  device="cpu", quant=tiny.spec, **TINY_SHAPE_HV15)
    kwargs.update(overrides)
    return NeuronHunyuanVideo15TransformerApplication(**kwargs)


def test_the_quantized_directory_holds_every_target(tiny):
    manifest = read_manifest(tiny.quantized)
    assert manifest["tensors"]["quantized"] == N_TARGETS
    assert {n.split(".", 2)[2] for n in manifest["quantized_linears"]} == TINY_TARGETS_PER_BLOCK


def test_application_quantizes_the_image_streams_projections_and_mlp(tiny):
    """At tp1 the module's linears are diffusers' own, so every twin is the plain one; the text
    stream's q/k/v and MLP, the output projections, the modulation, the token refiner and
    ``proj_out`` stay as they were."""
    app = _app(tiny)
    app.load()
    assert len(app.quantized_linears) == N_TARGETS
    assert app.load_report == {"missing": [], "unexpected": []}
    fp8 = {n: m for n, m in app.module.named_modules() if isinstance(m, _FP8LinearBase)}
    assert sorted(fp8) == sorted(app.quantized_linears)
    assert {n.split(".", 2)[2] for n in fp8} == TINY_TARGETS_PER_BLOCK
    assert all(type(m) is FP8Linear for m in fp8.values())
    for block in app.module.transformer_blocks:
        for linear in (block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj,
                       block.ff_context.net[0].proj, block.ff_context.net[2],
                       block.attn.to_out[0], block.attn.to_add_out, block.norm1.linear,
                       block.norm1_context.linear):
            assert type(linear) is torch.nn.Linear
    assert type(app.module.proj_out) is torch.nn.Linear
    assert not any(isinstance(m, _FP8LinearBase) for m in app.module.context_embedder.modules())
    torch.testing.assert_close(app(**tiny_hv15_inputs())[0], tiny.reference,
                               atol=1e-4, rtol=1e-4)


def test_eager_is_refused(tiny):
    assert _app(tiny, quant=tiny.spec.to_dict()).quant == tiny.spec
    with pytest.raises(ValueError, match="D43"):
        _app(tiny, exec_mode="eager")


def test_a_spec_for_another_tp_is_refused(tiny):
    """``net.2`` is row-parallel, quantized per rank K range: a tp2 directory's scales group
    tp2's K halves, which a tp1 launch would read as the wrong blocks."""
    with pytest.raises(ValueError,
                       match=r"the HunyuanVideo-1.5 FP8 DiT spec .*tp_degree=2 .*tp_degree=1"):
        _app(tiny, quant=hv15_spec(compute="dequant", tp_degree=2))


@pytest.mark.parametrize(("tp", "cfg", "sp", "cp", "world"),
                         [(2, True, False, 1, 4), (2, False, True, 1, 2), (2, False, False, 2, 4)],
                         ids=["tp2-cfg2", "tp2-sp", "tp2-cp2"])
def test_quantized_layouts_match_the_dequantized_reference(tiny, tmp_path, tp, cfg, sp, cp,
                                                           world):
    """Review Focus 4. Under CFG2 each group runs its branch of the stacked inputs; under SP the
    image MLP's row twin returns its unreduced partial and the bias (the worker checks the
    flags); under CP2 (F6's device layout adds it, Ulysses) both CP replicas of a TP rank load
    that rank's weights. Every rank's ``net.2`` weights must be its own K range quantized as
    such. A whole-K quantization, dequantized and sliced, puts other values in rank 0's range
    (the toy's scaled-down half of K) while the outputs move only 6e-7, far inside the
    tolerance: only the weight check sees it."""
    spec = hv15_spec(compute="dequant", tp_degree=tp)
    quantized = prepare_quantized_checkpoint(tiny.transformer_dir, spec, create=True,
                                             cache_root=tmp_path / "cache")
    branches = ([tiny_hv15_inputs(seed=2, valid_text=3), tiny_hv15_inputs()] if cfg
                else [tiny_hv15_inputs()])
    inputs = {k: torch.cat([b[k] for b in branches]) for k in branches[0]}
    want = torch.cat([hv15_reference(tiny.transformer_dir, b, spec=spec) for b in branches])
    source = safetensors_torch.load_file(next(tiny.transformer_dir.glob("*.safetensors")))
    results = run_ranks(hv15_fp8_worker, str(quantized), TINY_SHAPE_HV15, inputs,
                        spec.to_dict(), tp, cfg, sp, cp, world_size=world)
    assert {r["tp_rank"] for r in results} == set(range(tp))
    for name in HV15_ROW_TARGETS:
        w = source[f"{name}.weight"]
        assert not torch.equal(mx.fake_quantize_mx(w), mx.fake_quantize_mx_sharded(w, tp)), name
    for r in results:
        torch.testing.assert_close(torch.tensor(r["out"]), want, atol=1e-4, rtol=1e-4)
        for name in HV15_ROW_TARGETS:
            w = source[f"{name}.weight"]
            k = w.shape[1] // tp
            mine = mx.fake_quantize_mx_sharded(w, tp)[:, r["tp_rank"] * k:(r["tp_rank"] + 1) * k]
            assert torch.equal(torch.tensor(r["row_weights"][name]), mine), (name, r["tp_rank"])


def test_a_quantized_and_a_bf16_dit_share_no_block_graph_and_no_store_entry(tiny, tmp_path,
                                                                             monkeypatch):
    """The native path has no stage cache key: NEFFs are keyed by the HLO, the weight store by
    content and layout. So the quantized DiT's block graph must hold one kernel call per
    target and the bf16 one none, and the two must write different store entries. Their
    sources differ, which alone names two directories, so the entries' recorded keys must also
    differ in the module layout half (the fp8 data and scales against bf16 weights). The kernel
    is a stub op (no Neuron SDK here), both DiTs are bf16 as on the device, and the directory
    is built from the bf16 copy, as ``difflet compile`` builds it."""
    monkeypatch.setattr(fp8_kernels, "_MX_OP", _mx_linear_stub)
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")  # a bf16 source saves nothing
    cache = tmp_path / "cache"
    copy = ensure_bf16_copy(tiny.transformer_dir, cache)
    spec = hv15_spec(tp_degree=1)
    quantized = prepare_quantized_checkpoint(copy, spec, create=True, cache_root=cache)

    def loaded(model_path, quant):
        graphs = []
        app = NeuronHunyuanVideo15TransformerApplication(
            model_path=model_path, dtype=torch.bfloat16, exec_mode="compile", device="cpu",
            quant=quant, **TINY_SHAPE_HV15)
        app.compile_backend = _recording_backend(graphs)
        torch._dynamo.reset()
        app.load()
        return app, graphs

    bf16_app, bf16_graphs = loaded(copy, None)
    quant_app, quant_graphs = loaded(quantized, spec)
    assert _stub_calls(quant_graphs)
    assert all(c == len(TINY_TARGETS_PER_BLOCK) for c in _stub_calls(quant_graphs))
    assert _stub_calls(bf16_graphs) and not any(_stub_calls(bf16_graphs))
    assert bf16_app.weight_store.status == quant_app.weight_store.status == "written"
    assert bf16_app.weight_store.directory != quant_app.weight_store.directory
    bf16_key, quant_key = (_store_key(app) for app in (bf16_app, quant_app))
    assert bf16_key["layout_sha256"] != quant_key["layout_sha256"]


def test_resident_bytes_count_the_kernel_layout(snap):
    """A regression pin on the arithmetic the transformer stage's memory line reports. After
    ``_after_load`` an ``nki`` target holds its fp8 data packed as int32 ``[K/4, N]`` (N·K
    bytes), the stored scales (N·K/32) and the kernel's uint8 ``[K/4, N]`` scale layout
    (N·K/4): 41/32 = 1.28 bytes per weight against bf16's 2. The pre-load estimate
    (``weight_bytes``) omits the kernel layout, and ``dequant`` keeps a bf16 copy beside the
    stored scales. ``nk`` is counted on diffusers' own model, not on the application."""
    from diffusers import HunyuanVideo15Transformer3DModel

    from difflet.backends.neuron.core.application_base import rank_weight_bytes

    def loaded(model_path, quant):
        app = NeuronHunyuanVideo15TransformerApplication(
            model_path=model_path, dtype=torch.bfloat16, exec_mode="compile", device="cpu",
            quant=quant, **TINY_SHAPE_HV15)
        app.load(skip_warmup=True)
        return app

    spec = hv15_spec(tp_degree=1)
    with torch.device("meta"):
        model = HunyuanVideo15Transformer3DModel(**TINY_FP8_HV15)
    targets = [m for name, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and spec.matches(name)]
    assert len(targets) == N_TARGETS
    nk = sum(m.weight.numel() for m in targets)
    bf16 = loaded(snap.copy, None)
    nki = loaded(snap.quantized[1], spec)
    dq = loaded(snap.quantized[1], hv15_spec(compute="dequant",
                                                tp_degree=1))
    bf16_bytes, nki_bytes, dq_bytes = (rank_weight_bytes(app.module, app.dtype)
                                       for app in (bf16, nki, dq))
    assert bf16_bytes - nki_bytes == nk * 23 // 32
    assert nki_bytes - nki.weight_bytes == nk // 4
    assert dq_bytes - bf16_bytes == nk // 32


@pytest.fixture(scope="module")
def snap(tmp_path_factory):
    """The toy with what ``difflet compile`` leaves under one cache root for a ``--quant``
    launch: the bf16 copy of the fp32 DiT and, built from that copy, the measured set's
    quantized directory at tp1 and at tp2. The tests point ``DIFFLET_COMPILE_CACHE`` at the
    cache, where the pipeline looks for both."""
    root = write_tiny_hv15_snapshot(tmp_path_factory.mktemp("tiny_hv15_pipe_fp8"))
    cache = root / "cache"
    copy = ensure_bf16_copy(root / "transformer", cache)
    quantized = {tp: prepare_quantized_checkpoint(
        copy, hv15_spec(tp_degree=tp), create=True, cache_root=cache)
        for tp in (1, 2)}
    return SimpleNamespace(root=root, cache=cache, copy=copy, quantized=quantized)


def _pipe(model_path, **overrides) -> TorchNeuronHunyuanVideo15Pipeline:
    """The pipeline in bf16 on the CPU at the toy's shape: the DiT is built (not loaded) and
    the quant options resolved."""
    kwargs = dict(model_path=model_path, dtype=torch.bfloat16, exec_mode="compile",
                  device="cpu", **TINY_SHAPE_HV15)
    kwargs.update(overrides)
    return TorchNeuronHunyuanVideo15Pipeline(**kwargs)


def test_pipeline_resolves_the_quantized_directory_from_the_bf16_copy(snap, monkeypatch):
    """The directory is resolved from what the DiT would load in bf16, the copy, so the writer
    read bf16 tensors and never the hub's fp32 files; the stage passes the spec as a dict."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(snap.cache))
    spec = hv15_spec(tp_degree=1)
    pipe = _pipe(snap.root, quant=spec.to_dict())
    assert pipe.dit.model_path == pipe.quantized_dit == str(snap.quantized[1])
    assert pipe.quant == pipe.dit.quant == spec
    assert _pipe(snap.root, quant=spec).quant == spec


def test_the_pipeline_refuses_mxfp8_without_a_target_set(snap, monkeypatch):
    """Task F8: HunyuanVideo-1.5 has no default table, so ``quant="mxfp8"`` names no layers and
    is refused, naming the override; a spec (or its dict, as the stage passes it) still runs."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(snap.cache))
    with pytest.raises(ValueError, match="no default MXFP8 target set"):
        _pipe(snap.root, quant="mxfp8")


def test_the_pipeline_resolves_the_directory_of_the_spec_tp(snap, monkeypatch):
    """``net.2`` is quantized per rank K range, so the directory is the spec's tp's: the stage
    passes the driver's spec, built at the launch's tp."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(snap.cache))
    pipe = _pipe(snap.root, quant=hv15_spec(tp_degree=2).to_dict(),
                 parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True))
    assert pipe.quant.tp_degree == 2
    assert pipe.quant == pipe.dit.quant == hv15_spec(tp_degree=2)
    assert pipe.dit.model_path == pipe.quantized_dit == str(snap.quantized[2])


def test_bf16_loads_the_copy_and_holds_no_spec(snap, monkeypatch):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(snap.cache))
    pipe = _pipe(snap.root)
    assert pipe.dit.model_path == str(snap.copy)
    assert pipe.quant is None and pipe.dit.quant is None
    assert pipe.quantized_dit is None


def test_without_the_copy_the_quantized_directory_is_missing(snap, monkeypatch, tmp_path):
    """A snapshot ``difflet compile`` never ran on has no copy, so the directory is resolved
    from the fp32 source, which nothing quantized: refused, naming the fix, and nothing is
    written (every rank of a ``generate`` resolves, none writes)."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(snap.cache))
    bare = write_tiny_hv15_snapshot(tmp_path / "bare")
    written = lambda: (sorted(snap.cache.rglob("*")), sorted(bare.rglob("*")))  # noqa: E731
    before = written()
    with pytest.raises(ValueError, match="run `difflet compile`"):
        _pipe(bare, quant=hv15_spec(tp_degree=1).to_dict())
    assert written() == before
    with pytest.raises(ValueError, match="quant_checkpoint needs quant"):
        _pipe(snap.root, quant_checkpoint="/q")
