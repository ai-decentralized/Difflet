"""C8 unit tests (CPU only): rank-0 host I/O primitives and the non-AoT application base.

Multi-rank cases run on 4 gloo processes through tests/unit/backends/_neuron_gloo.py; the
device counterpart is tests/manual/check_neuron_application_c8.py.
"""

import pytest

# As in test_neuron_attention_c3.py: importorskip keeps torch_neuronx's import-time nki.jit
# DeprecationWarning out of the report when this module is collected first.
torch = pytest.importorskip("torch")

from difflet.backends.neuron.core import distributed as nd  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_workers import c8_distributed_worker  # noqa: E402

HEADER_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
HEADER_SHAPES = [(2, 3), (), (0, 4), (1, 2, 1, 2, 1, 2, 1, 2)]
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def test_world_info_without_process_group():
    assert nd.world_info() == (0, 1)
    assert nd.is_rank0() is True


def test_sync_status_is_a_noop_on_one_rank():
    nd.sync_status(True, device="cpu", what="healthy")
    nd.sync_status(False, device="cpu", what="failing rank")  # never raises on the failing rank


def test_collective_phase_reraises_the_original_error():
    with pytest.raises(ValueError, match="boom"):
        with nd.collective_phase("phase", device="cpu"):
            raise ValueError("boom")


def test_collective_phase_clean_exit_runs_the_body():
    ran = []
    with nd.collective_phase("phase", device="cpu"):
        ran.append(True)
    assert ran == [True]


def test_rank0_call_returns_the_result_on_rank0():
    assert nd.rank0_call(lambda a, *, b: a + b, 40, b=2, device="cpu", what="add") == 42


@pytest.mark.parametrize("dtype", HEADER_DTYPES, ids=str)
@pytest.mark.parametrize("shape", HEADER_SHAPES, ids=str)
def test_header_round_trip(dtype, shape):
    header = nd._encode_header(torch.zeros(shape, dtype=dtype))
    assert header.dtype == torch.int32 and tuple(header.shape) == (nd._HEADER_LEN,)
    assert nd._decode_header(header, src=0) == (dtype, shape)


def test_header_rejects_unsupported_inputs():
    with pytest.raises(TypeError, match="must pass a tensor"):
        nd._encode_header(None)
    with pytest.raises(TypeError, match="unsupported dtype"):
        nd._encode_header(torch.zeros(2, dtype=torch.complex64))
    with pytest.raises(ValueError, match="at most 8 dims"):
        nd._encode_header(torch.zeros([1] * 9))


def test_failed_header_raises_rank_failure_on_receivers():
    with pytest.raises(nd.RankFailureError, match="source rank 0 failed"):
        nd._decode_header(nd._failed_header(), src=0)


def test_broadcast_tensor_on_one_rank():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert torch.equal(nd.broadcast_tensor(t, device="cpu"), t)
    with pytest.raises(TypeError, match="must pass a tensor"):
        nd.broadcast_tensor(None, device="cpu")


@pytest.mark.parametrize(
    "values, device, fits",
    [
        ([1, 2**31], "neuron", False),
        ([-(2**31) - 1, 0], "neuron", False),
        ([INT32_MIN, INT32_MAX], "neuron", True),
        ([], "neuron", True),
        ([2**40, -(2**40)], "cpu", True),  # gloo carries int64 as int64
    ],
    ids=["above", "below", "edges", "empty", "cpu"],
)
def test_int64_payload_must_fit_int32_on_the_neuron_device(values, device, fits):
    # Pure host check: nothing here touches the neuron device.
    payload = torch.tensor(values, dtype=torch.int64)
    if fits:
        nd._check_payload(payload, device)
    else:
        with pytest.raises(ValueError, match="outside the int32 range"):
            nd._check_payload(payload, device)


def test_broadcast_tensor_on_one_rank_applies_the_int64_rule(monkeypatch):
    # The single-rank path validates like the multi-rank one, so TP1 runs catch it too.
    monkeypatch.setattr(nd, "_narrows_int64", lambda device: True)
    with pytest.raises(ValueError, match="outside the int32 range"):
        nd.broadcast_tensor(torch.tensor([2**31], dtype=torch.int64), device="cpu")
    edges = torch.tensor([INT32_MIN, INT32_MAX], dtype=torch.int64)
    assert torch.equal(nd.broadcast_tensor(edges, device="cpu"), edges)


def test_distributed_primitives_on_four_gloo_ranks():
    results = run_ranks(c8_distributed_worker, world_size=4)
    for rank, out in enumerate(results):
        assert out["world"] == (rank, 4)
        assert out["is_rank0"] is (rank == 0)
        if rank == 2:
            assert out["phase"] == "ValueError: boom on rank 2"
        else:
            assert out["phase"].startswith("RankFailureError: rank-2 phase: failed on rank 2")
        assert out["rank0_call"] == (42 if rank == 0 else None)
        assert out["broadcast_mismatches"] == []
        assert out["bad_root"] == ("TypeError" if rank == 0 else "RankFailureError")
        if rank == 0:
            assert out["int64_out_of_range"].startswith("ValueError: broadcast_tensor: int64")
        else:
            assert out["int64_out_of_range"].startswith(
                "RankFailureError: broadcast_tensor: source rank 0 failed"
            )
        assert out["int64_edges"] == ("torch.int64", [[INT32_MIN], [INT32_MAX]])
        assert out["int64_scalar"] == ("torch.int64", (), -5)
        # Where int64 is narrowed (neuron), it travels as int32, and every payload travels flat:
        # torch_neuronx's own narrowing inside the collective, and widening a 0-dim tensor on
        # the device, each record a CPU fallback.
        header = ("torch.int32", (nd._HEADER_LEN,))
        assert out["int64_wire"] == [
            header, ("torch.int32", (2,)), header, ("torch.int32", (1,))
        ]
        assert out["final"] == "ok"


# ------------------------------------------------------------- application base (part 2)

import inspect  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.registry import get_backend  # noqa: E402
from difflet.pipeline import difflet_pipeline as dp  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends import _neuron_toy  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    TOY_APP_BLOCKS,
    TOY_APP_DIM,
    TOY_APP_HIDDEN,
    ToyApplication,
    toy_blocks_input,
    toy_blocks_reference,
    write_toy_checkpoint,
)
from tests.unit.backends._neuron_workers import (  # noqa: E402
    c8_failure_worker,
    c8_forward_failure_worker,
    c8_lifecycle_worker,
)

BLOCK_NAMES = [f"blocks.{i}" for i in range(TOY_APP_BLOCKS)]
LOAD_PHASES = {
    "runtime init", "build on meta", "load checkpoint", "eval", "compile", "warmup forward"
}
# The status-synced (collective-free) steps of load(), in order; the warm-up forward is not one.
SYNCED_PHASES = [
    "runtime init", "build on meta", "load checkpoint", "eval", "compile", "example inputs"
]
APP_LOGGER = "difflet.backends.neuron.core.application_base"
FORWARD_FAILURE_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    x = toy_blocks_input(1, 16, TOY_APP_DIM)
    weights, ref, ref_bf16 = toy_blocks_reference(TOY_APP_BLOCKS, TOY_APP_DIM, TOY_APP_HIDDEN, x)
    directory = tmp_path_factory.mktemp("c8_ckpt")
    write_toy_checkpoint(directory, weights, num_files=2)
    return SimpleNamespace(dir=str(directory), x=x, ref=ref, ref_bf16=ref_bf16, weights=weights)


def _app(toy, **overrides):
    kwargs = dict(
        model_path=toy.dir,
        parallel=DiffletParallelConfig(),
        dtype=torch.float32,
        exec_mode="eager",
        device="cpu",
    )
    kwargs.update(overrides)
    return ToyApplication(**kwargs)


def _counting_aot_eager(graphs: list):
    aot_eager = torch._dynamo.lookup_backend("aot_eager")

    def backend(gm, example_inputs):
        graphs.append(gm)
        return aot_eager(gm, example_inputs)

    return backend


def _app_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == APP_LOGGER and r.levelno >= logging.WARNING
    ]


def _rank_sections(report: str) -> dict[int, str]:
    """run_ranks' AssertionError text, split into each failed rank's traceback or reason."""
    parts = re.split(r"^--- rank (\d+) ---$", report, flags=re.MULTILINE)
    return {int(rank): body.strip() for rank, body in zip(parts[1::2], parts[2::2])}


def test_contract_signatures_match_pipeline_reflection():
    # DiffletPipeline passes only the parameter NAMES it finds (difflet_pipeline.py:281-320).
    sig = inspect.signature
    assert list(sig(TorchNeuronApplicationBase.compile).parameters) == [
        "self", "compiled_model_path", "debug"]
    load_params = sig(TorchNeuronApplicationBase.load).parameters
    assert list(load_params) == [
        "self", "compiled_model_path", "start_rank_id", "local_ranks_size", "skip_warmup"]
    assert load_params["compiled_model_path"].default is None
    assert list(sig(TorchNeuronApplicationBase.has_compiled_artifacts).parameters) == [
        "self", "compiled_model_path"]


def test_pipeline_helpers_drive_the_lifecycle(toy, tmp_path):
    app = _app(toy)
    artifacts = tmp_path / "artifacts"
    dp._compile_app(app, artifacts, debug=True)
    assert not artifacts.exists()
    assert dp._compiled_artifacts_ready(app, artifacts) is True
    dp._load_app(app, artifacts, backend=get_backend("neuron"), start_rank_id=None,
                 local_ranks_size=None, skip_warmup=False)
    assert app.is_loaded
    with torch.no_grad():
        assert torch.equal(app(toy.x), toy.ref)


def test_eager_lifecycle_matches_tp1_reference(toy):
    app = _app(toy)
    app.load()
    assert app.is_loaded and app.exec_mode == "eager" and app.compiled_blocks == []
    assert app.warmup_shapes == [tuple(toy.x.shape)]
    assert app.load_report == {"missing": [], "unexpected": []}
    assert set(app.phase_seconds) == LOAD_PHASES
    params = list(app.module.parameters())
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in params)
    assert not any(p.requires_grad for p in params) and not app.module.training
    with torch.no_grad():
        assert torch.equal(app(toy.x), toy.ref)  # same weights, same CPU kernels


def test_compile_lifecycle_compiles_each_block_in_place(toy):
    torch._dynamo.reset()
    app = _app(toy, exec_mode="compile")
    app.load()
    assert app.compiled_blocks == BLOCK_NAMES
    assert all(block._compiled_call_impl is not None for block in app.module.blocks)
    assert not any("_orig_mod" in key for key in app.module.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(app(toy.x), toy.ref, rtol=1e-5, atol=1e-5)


def test_bf16_load_casts_the_fp32_checkpoint(toy):
    app = _app(toy, dtype="bfloat16")
    app.load()
    assert app.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in app.module.parameters())
    with torch.no_grad():
        out = app(toy.x.to(torch.bfloat16)).float()
    ours = (out - toy.ref).abs().mean().item()
    cpu_bf16 = (toy.ref_bf16 - toy.ref).abs().mean().item()
    assert ours <= 2 * cpu_bf16 + 1e-6


def test_mesh_is_initialized_before_build_module(toy):
    seen = {}

    class Probe(ToyApplication):
        def build_module(self):
            seen["mesh"] = parallel_mesh.is_mesh_initialized()
            seen["tp"] = parallel_mesh.get_tp_size()
            return super().build_module()

    Probe(model_path=toy.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
          exec_mode="eager", device="cpu").load(skip_warmup=True)
    assert seen == {"mesh": True, "tp": 1}


def test_example_input_failure_surfaces_and_leaves_app_unloaded(toy):
    app = _app(toy, inject_failure="example_inputs")
    with pytest.raises(RuntimeError, match="injected failure: example_inputs"):
        app.load()
    assert app.is_loaded is False
    with pytest.raises(RuntimeError, match="not loaded"):
        app(toy.x)


def test_skip_warmup_never_asks_for_example_inputs(toy):
    app = _app(toy, inject_failure="example_inputs")
    app.load(skip_warmup=True)
    assert app.is_loaded and app.warmup_shapes is None


def test_missing_checkpoint_dir_is_rejected():
    pytest.importorskip("accelerate")
    app = ToyApplication(parallel=DiffletParallelConfig(), dtype=torch.float32,
                         exec_mode="eager", device="cpu")
    with pytest.raises(ValueError, match="model_path"):
        app.load()


def test_forward_before_load_raises():
    app = ToyApplication(device="cpu", exec_mode="eager")
    with pytest.raises(RuntimeError, match="not loaded"):
        app(torch.zeros(1))


def test_compile_is_a_noop_and_artifacts_are_always_ready(tmp_path):
    app = ToyApplication(device="cpu", exec_mode="eager")
    app.compile(str(tmp_path / "artifacts"), debug=True)
    assert not (tmp_path / "artifacts").exists()
    assert app.has_compiled_artifacts(str(tmp_path / "artifacts")) is True
    assert app.module is None and app.is_loaded is False


def test_exec_mode_resolution(monkeypatch):
    assert ToyApplication(device="cpu").exec_mode == "compile"
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    assert ToyApplication(device="cpu").exec_mode == "eager"
    assert ToyApplication(device="cpu", exec_mode="compile").exec_mode == "compile"
    with pytest.raises(ValueError):
        ToyApplication(device="cpu", exec_mode="lazy")


@pytest.mark.parametrize(
    "parallel",
    [
        DiffletParallelConfig(cp_degree=2),
        DiffletParallelConfig(cfg_parallel_enabled=True),
        DiffletParallelConfig(sp_enabled=True),
        DiffletParallelConfig(dp_degree=2),
    ],
    ids=["cp", "cfg", "sp", "dp"],
)
def test_unsupported_parallel_modes_raise(parallel):
    app = ToyApplication(parallel=parallel, device="cpu", exec_mode="eager",
                         model_path="/nonexistent")
    with pytest.raises(NotImplementedError, match="tensor parallelism only"):
        app.load()
    assert parallel_mesh.is_mesh_initialized() is False


def test_multi_rank_config_without_process_group_is_rejected():
    app = ToyApplication(parallel=DiffletParallelConfig(tp_degree=4), device="cpu",
                         exec_mode="eager", model_path="/nonexistent")
    with pytest.raises(RuntimeError, match="no process group"):
        app.load()


def test_example_inputs_must_be_a_tuple_of_tensors(toy):
    class ListInputs(ToyApplication):
        def get_example_inputs(self):
            return [toy.x]

    app = ListInputs(model_path=toy.dir, parallel=DiffletParallelConfig(), dtype=torch.float32,
                     exec_mode="eager", device="cpu")
    with pytest.raises(TypeError, match="tuple of tensors"):
        app.load()


def test_load_is_idempotent(toy):
    app = _app(toy)
    app.load(skip_warmup=True)
    module = app.module
    app.load()
    assert app.module is module


def test_dtype_normalization():
    assert ToyApplication(device="cpu", dtype="torch.float16").dtype == torch.float16
    assert ToyApplication(device="cpu", dtype=torch.float32).dtype == torch.float32
    with pytest.raises(ValueError, match="unsupported dtype"):
        ToyApplication(device="cpu", dtype="float128")


def test_rank_and_world_size_without_process_group():
    app = ToyApplication(device="cpu")
    assert (app.rank, app.world_size, app.device) == (0, 1, torch.device("cpu"))
    assert app.parallel == DiffletParallelConfig() and app.dtype == torch.bfloat16


def test_a_forward_failure_is_never_status_synced(toy, monkeypatch):
    # Only collective-free steps are status-synced. A failing warm-up forward must not send a
    # status all-reduce: on a multi-rank run it would pair with a peer's in-forward all-reduce.
    calls = []
    monkeypatch.setattr(nd, "sync_status", lambda ok, *, device, what: calls.append((what, ok)))
    monkeypatch.setattr(_neuron_toy, "FORWARD_FAILURE_RANK", 0)
    app = _app(toy, inject_failure="forward")
    with pytest.raises(RuntimeError, match="injected failure: forward on rank 0"):
        app.load()
    assert calls == [(what, True) for what in SYNCED_PHASES]
    assert app.is_loaded is False and "warmup forward" not in app.phase_seconds


def test_compile_mode_flags_a_forward_at_an_unwarmed_shape(toy, caplog):
    torch._dynamo.reset()
    graphs = []
    app = _app(toy, exec_mode="compile")
    app.compile_backend = _counting_aot_eager(graphs)
    app.load()
    assert len(graphs) == 1 and app.unwarmed_shapes == []
    out = app(toy.x)  # grad enabled here: forward runs under no_grad, as the warm-up did
    assert len(graphs) == 1 and not out.requires_grad
    other = toy_blocks_input(1, 8, TOY_APP_DIM)
    with caplog.at_level(logging.WARNING, logger=APP_LOGGER), torch.no_grad():
        app(toy.x)  # the warm-up shape: already compiled
        assert len(graphs) == 1 and _app_warnings(caplog) == []
        app(other)
        app(other)  # flagged once per new shape
    assert len(graphs) == 2  # the blocks really were traced and compiled again
    assert app.unwarmed_shapes == [((1, 8, TOY_APP_DIM),)]
    [message] = _app_warnings(caplog)
    assert f"(1, 8, {TOY_APP_DIM})" in message and "warm-up" in message


def test_compile_mode_without_warmup_takes_the_first_forward_shape(toy, caplog):
    torch._dynamo.reset()
    app = _app(toy, exec_mode="compile")
    app.load(skip_warmup=True)
    with caplog.at_level(logging.WARNING, logger=APP_LOGGER), torch.no_grad():
        app(toy.x)
        app(toy.x)
    assert app.unwarmed_shapes == [] and _app_warnings(caplog) == []


def test_eager_mode_does_not_track_input_shapes(toy, caplog):
    app = _app(toy)
    app.load()
    with caplog.at_level(logging.WARNING, logger=APP_LOGGER), torch.no_grad():
        app(toy_blocks_input(1, 8, TOY_APP_DIM))
    assert app.unwarmed_shapes == [] and _app_warnings(caplog) == []


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_four_rank_lifecycle_matches_tp1(toy, exec_mode):
    results = run_ranks(c8_lifecycle_worker, toy.dir, exec_mode, "float32", world_size=4)
    full_numel = sum(t.numel() for t in toy.weights.values())
    expected_blocks = BLOCK_NAMES if exec_mode == "compile" else []
    outs = [torch.tensor(r["out"]) for r in results]
    for rank, (result, out) in enumerate(zip(results, outs)):
        assert (result["rank"], result["world_size"], result["is_loaded"]) == (rank, 4, True)
        assert result["compiled_blocks"] == expected_blocks
        assert result["warmup_shapes"] == [tuple(toy.x.shape)]
        assert result["param_numel"] < full_numel  # really sharded, not four TP1 copies
        assert result["param_shapes"] == results[0]["param_shapes"]
        assert torch.equal(out, outs[0])  # the row-parallel all-reduce output is replicated
        torch.testing.assert_close(out, toy.ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "inject, phase", [("example_inputs", "example inputs"), ("build_module", "build on meta")]
)
def test_rank0_failure_reaches_every_rank(toy, inject, phase):
    results = run_ranks(c8_failure_worker, toy.dir, inject, world_size=4)
    assert results[0]["outcome"] == "raised"
    assert f"injected failure: {inject}" in results[0]["message"]
    for result in results[1:]:
        assert result["outcome"] == "rank_failure"
        assert result["message"].startswith(f"{phase}: failed on rank 0")
    assert not any(result["is_loaded"] for result in results)


def test_forward_failure_on_one_rank_is_reported_without_a_hang(toy):
    # Rank 2 raises inside block 0 of the warm-up forward, between its two all-reduces. It must
    # leave with its own error (a launcher then tears down its peers, blocked in the MLP
    # all-reduce); a status collective there would pair with that all-reduce instead.
    start = time.monotonic()
    with pytest.raises(AssertionError) as excinfo:
        run_ranks(c8_forward_failure_worker, toy.dir, world_size=4, timeout=FORWARD_FAILURE_TIMEOUT)
    elapsed = time.monotonic() - start
    sections = _rank_sections(str(excinfo.value))
    assert set(sections) == {0, 1, 2, 3}  # no peer finished a forward rank 2 never joined
    assert sections[2].splitlines()[-1] == "RuntimeError: injected failure: forward on rank 2"
    assert elapsed < FORWARD_FAILURE_TIMEOUT  # returned on rank 2's report, not at the deadline
