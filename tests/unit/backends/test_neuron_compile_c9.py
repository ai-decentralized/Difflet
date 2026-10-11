"""Unit tests for the neuron backend's compile path: execution modes, the persistent NEFF cache
settings and per-block torch.compile. CPU only; compiled checks use the aot_eager backend."""

from __future__ import annotations

import functools
import logging
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import torch._dynamo  # noqa: E402
import torch.nn as nn  # noqa: E402

from difflet import envs  # noqa: E402
from difflet.backends.neuron import compile as neuron_compile  # noqa: E402
from difflet.backends.neuron import runtime  # noqa: E402
from difflet.backends.neuron.ops_impl import attention as neuron_attention  # noqa: E402
from difflet.backends.neuron.ops_impl.parallel_mesh import destroy_parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_toy import ToyBlock, ToyBlocksModel, ToyTPMLP  # noqa: E402

N_BLOCKS, DIM, HIDDEN, HEADS, SEQ = 6, 64, 128, 4, 16
WAN_DEPTH = 40  # Wan 2.x DiT depth
CACHE_VARS = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "NKI_ENABLE_TRACE_CACHE",
)
HLO_CACHE_MODULE = "torch_neuronx.python_ops.compilation.hlo_cache"
COMPILE_LOGGER = "difflet.backends.neuron.compile"


@pytest.fixture(autouse=True)
def _fresh_dynamo_and_mesh(monkeypatch):
    # Dynamo keeps compiled graphs on the block's code object: without a reset, one test's graph
    # would serve (or, past recompile_limit, block) the next test's compile.
    destroy_parallel_mesh()
    torch._dynamo.reset()
    # prepare_runtime records that it ran; restore the flag so no test sees another's call.
    monkeypatch.setattr(runtime, "_runtime_prepared", False)
    # compile_blocks keeps one neuron options dict per option set for the whole process, and a
    # backend edits it; start every test without one. (raising=False: the RED run of the final
    # fix uses the module from before the attribute existed.)
    monkeypatch.setattr(neuron_compile, "_shared_neuron_options", [], raising=False)
    yield
    torch._dynamo.reset()
    destroy_parallel_mesh()


class CountingBackend:
    """aot_eager, keeping every graph Dynamo hands it; extra keyword arguments are dropped."""

    def __init__(self):
        self.inner = torch._dynamo.lookup_backend("aot_eager")
        self.graphs = []

    def __call__(self, gm, example_inputs, **kwargs):
        self.graphs.append(gm)
        return self.inner(gm, example_inputs)


class OptionsEditingBackend(CountingBackend):
    """Behaves like the neuron backend, which adds "dynamic" to its options dict in place
    (``edits``: the keys it sets that way); records the options object of every graph."""

    def __init__(self, edits=(("dynamic", False),)):
        super().__init__()
        self.edits = tuple(edits)
        self.options = []

    def __call__(self, gm, example_inputs, *, options=None, **kwargs):
        self.options.append(options)
        if options is not None:
            for key, value in self.edits:
                options.setdefault(key, value)
        return super().__call__(gm, example_inputs)


@pytest.fixture
def neuron_backend_stub(monkeypatch):
    """Install an OptionsEditingBackend as Dynamo's "neuron" backend for one test, so
    ``compile_blocks(backend="neuron")`` (NEURON_COMPILE_OPTIONS, the shared options dict) runs
    on CPU. Returns a factory taking the backend's ``edits``."""
    from torch._dynamo.backends import registry

    registry._lazy_import()  # read the entry-point table now, so it cannot replace the stub later

    def install(edits=(("dynamic", False),)):
        backend = OptionsEditingBackend(edits)
        monkeypatch.setitem(registry._BACKENDS, "neuron", None)
        monkeypatch.setitem(registry._COMPILER_FNS, "neuron", backend)
        return backend

    return install


def _toy(n_blocks=N_BLOCKS, dtype=torch.float32):
    torch.manual_seed(0)
    return ToyBlocksModel(n_blocks, DIM, HIDDEN, heads=HEADS, dtype=dtype).eval()


def _input(dtype=torch.float32, seq=SEQ):
    return torch.randn(2, seq, DIM, generator=torch.Generator().manual_seed(1)).to(dtype)


# ---------------------------------------------------------------- exec mode


@pytest.mark.parametrize(
    "value, expected", [("eager", "eager"), ("compile", "compile"), (" Compile ", "compile")]
)
def test_resolve_exec_mode_prefers_the_explicit_value(monkeypatch, value, expected):
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    assert neuron_compile.resolve_exec_mode(value) == expected


def test_resolve_exec_mode_reads_the_env_then_the_default(monkeypatch):
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    assert envs.DIFFLET_EXEC_MODE == "eager"
    assert neuron_compile.resolve_exec_mode() == "eager"
    monkeypatch.delenv("DIFFLET_EXEC_MODE")
    assert envs.DIFFLET_EXEC_MODE is None
    assert neuron_compile.resolve_exec_mode() == neuron_compile.DEFAULT_EXEC_MODE == "compile"


@pytest.mark.parametrize("value", ["jit", "aot", "latency"])
def test_resolve_exec_mode_rejects_unknown_modes(value):
    with pytest.raises(ValueError, match="expected one of: eager, compile"):
        neuron_compile.resolve_exec_mode(value)


def test_exec_mode_is_a_registered_env_var():
    assert "DIFFLET_EXEC_MODE" in dir(envs)


# ---------------------------------------------------------------- NEFF cache settings


def test_default_neff_cache_dir_follows_difflet_compile_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    assert neuron_compile.default_neff_cache_dir() == tmp_path / "neuron" / "neff"


def test_configure_compile_cache_defaults_under_the_difflet_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    env = {}
    settings = neuron_compile.configure_compile_cache(env=env)
    assert settings == env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(tmp_path / "neuron" / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(tmp_path / "neuron" / "neff_local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(tmp_path / "neuron" / "hlo"),
        "NKI_ENABLE_TRACE_CACHE": "0",
    }
    assert not (tmp_path / "neuron").exists()  # torch-neuronx creates the directories itself


def test_configure_compile_cache_keeps_launcher_dirs_but_forces_the_trace_cache_off(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    env = {
        "TORCH_NEURONX_NEFF_CACHE_DIR": "/shared/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "/local/locks",
        "TORCH_NEURONX_HLO_CACHE_DIR": "/shared/hlo",
        "NKI_ENABLE_TRACE_CACHE": "1",  # what `import torch_neuronx` sets by default
    }
    neuron_compile.configure_compile_cache(env=env)
    assert env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": "/shared/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "/local/locks",
        "TORCH_NEURONX_HLO_CACHE_DIR": "/shared/hlo",
        "NKI_ENABLE_TRACE_CACHE": "0",
    }


def test_configure_compile_cache_inherits_the_hlo_dir_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    env = {"TORCH_NEURONX_HLO_CACHE_DIR": "/shared/hlo"}
    neuron_compile.configure_compile_cache(env=env)
    assert env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(tmp_path / "neuron" / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(tmp_path / "neuron" / "neff_local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": "/shared/hlo",
        "NKI_ENABLE_TRACE_CACHE": "0",
    }


def test_configure_compile_cache_explicit_arguments_win(tmp_path):
    env = {
        "TORCH_NEURONX_NEFF_CACHE_DIR": "/shared/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "/local/locks",
        "TORCH_NEURONX_HLO_CACHE_DIR": "/shared/hlo",
    }
    neuron_compile.configure_compile_cache(
        tmp_path / "neff",
        local_cache_dir=str(tmp_path / "locks"),
        hlo_cache_dir=tmp_path / "hlo",
        nki_trace_cache=True,
        env=env,
    )
    assert env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(tmp_path / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(tmp_path / "locks"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(tmp_path / "hlo"),
        "NKI_ENABLE_TRACE_CACHE": "1",
    }


def test_configure_compile_cache_explicit_neff_dirs_leave_the_hlo_dir_at_its_default(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "root"))
    env = {
        "TORCH_NEURONX_NEFF_CACHE_DIR": "/shared/neff",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "/local/locks",
    }
    neuron_compile.configure_compile_cache(
        tmp_path / "neff", local_cache_dir=str(tmp_path / "locks"), nki_trace_cache=True, env=env
    )
    assert env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(tmp_path / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(tmp_path / "locks"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(tmp_path / "root" / "neuron" / "hlo"),
        "NKI_ENABLE_TRACE_CACHE": "1",
    }


def test_configure_compile_cache_expands_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {}
    neuron_compile.configure_compile_cache(
        "~/neff", local_cache_dir="~/locks", hlo_cache_dir="~/hlo", env=env
    )
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neff")
    assert env["TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR"] == str(tmp_path / "locks")
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == str(tmp_path / "hlo")


def _isolate_cache_env(monkeypatch):
    """Unset the cache variables now and restore them after the test, even though
    configure_compile_cache assigns os.environ itself (delenv of an absent name records
    nothing to undo, so set first)."""
    for name in CACHE_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_configure_compile_cache_writes_the_process_environment(monkeypatch, tmp_path):
    _isolate_cache_env(monkeypatch)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    monkeypatch.setattr(neuron_compile, "_hlo_cache_warned", True)  # no log about the real one
    neuron_compile.configure_compile_cache()
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neuron" / "neff")
    assert os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] == str(tmp_path / "neuron" / "hlo")
    assert os.environ["NKI_ENABLE_TRACE_CACHE"] == "0"


@pytest.fixture
def fake_hlo_cache(monkeypatch, tmp_path):
    """torch-neuronx's hlo_cache module as already imported with CACHE_DIR=/tmp/hlo_cache (a
    stand-in in sys.modules; the real one, if any, comes back at teardown), no warning issued
    yet in this process, no cache variable set, DIFFLET_COMPILE_CACHE=tmp_path."""
    fake = types.ModuleType(HLO_CACHE_MODULE)
    fake.CACHE_DIR = "/tmp/hlo_cache"
    monkeypatch.setitem(sys.modules, HLO_CACHE_MODULE, fake)
    monkeypatch.setattr(neuron_compile, "_hlo_cache_warned", False)
    _isolate_cache_env(monkeypatch)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    return fake


def _compile_log_records(caplog):
    return [r for r in caplog.records if r.name == COMPILE_LOGGER]


def test_configure_compile_cache_logs_once_that_an_imported_torch_neuronx_keeps_its_hlo_locks(
    fake_hlo_cache, caplog, tmp_path
):
    """torch-neuronx's C++ HLO cache reads TORCH_NEURONX_HLO_CACHE_DIR at each lookup, but
    hlo_cache.CACHE_DIR, its lock directory, was read at import: the entries move, the locks
    do not."""
    with caplog.at_level(logging.WARNING, logger=COMPILE_LOGGER):
        neuron_compile.configure_compile_cache()
        neuron_compile.configure_compile_cache(hlo_cache_dir=tmp_path / "another")  # differs too
    records = _compile_log_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert f"TORCH_NEURONX_HLO_CACHE_DIR={tmp_path / 'neuron' / 'hlo'} was set" in message
    assert "HLO lowering locks stay in /tmp/hlo_cache" in message
    assert "processes started with the new value" in message


def test_configure_compile_cache_does_not_log_when_the_hlo_dir_already_matches(
    fake_hlo_cache, caplog, tmp_path
):
    fake_hlo_cache.CACHE_DIR = str(tmp_path / "neuron" / "hlo") + "/"  # the same directory
    with caplog.at_level(logging.WARNING, logger=COMPILE_LOGGER):
        neuron_compile.configure_compile_cache()
    assert _compile_log_records(caplog) == []
    assert neuron_compile._hlo_cache_warned is False  # still armed for a real mismatch


def test_configure_compile_cache_does_not_log_for_another_processes_env(fake_hlo_cache, caplog):
    """A launcher's mapping (run_stage's rank env) is for processes not started yet: this
    process's HLO directory is not what it sets."""
    with caplog.at_level(logging.WARNING, logger=COMPILE_LOGGER):
        neuron_compile.configure_compile_cache(env={})
    assert _compile_log_records(caplog) == []


def test_configure_compile_cache_does_not_import_torch_neuronx_to_check(
    fake_hlo_cache, monkeypatch, caplog
):
    monkeypatch.delitem(sys.modules, HLO_CACHE_MODULE)  # not imported in this process
    with caplog.at_level(logging.WARNING, logger=COMPILE_LOGGER):
        neuron_compile.configure_compile_cache()
    assert _compile_log_records(caplog) == []
    assert HLO_CACHE_MODULE not in sys.modules


def test_prepare_runtime_configures_the_cache_before_binding_a_core(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "configure_compile_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(runtime, "bind_core", lambda: calls.append("bind"))
    monkeypatch.setattr(runtime, "init_process_group", lambda: calls.append("pg"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())
    assert calls == ["cache", "bind", "pg"]


def test_prepare_runtime_leaves_the_env_alone_when_the_config_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "configure_compile_cache", lambda: calls.append("cache"))
    with pytest.raises(NotImplementedError, match="cp_degree=2"):
        runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig(tp_degree=2, cp_degree=2))
    assert calls == []


def _fake_torch_neuronx(monkeypatch, initialized):
    fake = types.SimpleNamespace(is_neuron_runtime_initialized=lambda: initialized)
    monkeypatch.setitem(sys.modules, "torch_neuronx", fake)


def test_neuron_runtime_initialized_reads_torch_neuronx_without_importing_it(monkeypatch):
    _fake_torch_neuronx(monkeypatch, True)
    assert runtime._neuron_runtime_initialized() is True
    _fake_torch_neuronx(monkeypatch, False)
    assert runtime._neuron_runtime_initialized() is False
    monkeypatch.delitem(sys.modules, "torch_neuronx")  # never imported: no runtime either
    assert runtime._neuron_runtime_initialized() is False
    assert "torch_neuronx" not in sys.modules


def test_prepare_runtime_refuses_a_runtime_started_before_it(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "configure_compile_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(runtime, "bind_core", lambda: calls.append("bind"))
    monkeypatch.setattr(runtime, "init_process_group", lambda: calls.append("pg"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _fake_torch_neuronx(monkeypatch, True)
    with pytest.raises(RuntimeError, match="Neuron runtime is already initialised"):
        runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())
    assert calls == []  # no core bound, no cache settings written


def test_prepare_runtime_may_run_again_once_it_has_prepared_the_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "configure_compile_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(runtime, "bind_core", lambda: calls.append("bind"))
    monkeypatch.setattr(runtime, "init_process_group", lambda: calls.append("pg"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _fake_torch_neuronx(monkeypatch, False)
    runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())
    _fake_torch_neuronx(monkeypatch, True)  # device work started after the first call
    runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())  # a second pipeline
    assert calls == ["cache", "bind", "pg"] * 2


# ---------------------------------------------------------------- per-block compile


def test_compile_blocks_shares_one_graph_and_matches_eager():
    model, x = _toy(), _input()
    keys = list(model.state_dict())
    with torch.no_grad():
        ref = model(x)
    backend = CountingBackend()
    names = neuron_compile.compile_blocks(model, backend=backend)
    with torch.no_grad():
        out = model(x)
        again = model(x)
    assert names == [f"blocks.{i}" for i in range(N_BLOCKS)]
    assert len(backend.graphs) == 1  # one graph for all blocks; the second pass compiles nothing
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(again, out, rtol=0, atol=0)
    assert list(model.state_dict()) == keys  # no _orig_mod. prefix


def test_compile_blocks_shares_one_graph_when_the_backend_edits_its_options():
    model, x = _toy(), _input()
    backend = OptionsEditingBackend()
    options = {"fallback_execution": False}
    neuron_compile.compile_blocks(model, backend=backend, options=options)
    with torch.no_grad():
        model(x)
    assert len(backend.graphs) == 1
    assert options == {"fallback_execution": False}  # the caller's dict is copied, not edited


def test_compile_blocks_defaults_to_static_fullgraph_neuron(monkeypatch):
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append((self, a, k)))
    model = _toy(n_blocks=3)
    assert neuron_compile.compile_blocks(model) == ["blocks.0", "blocks.1", "blocks.2"]
    assert [module for module, _, _ in seen] == list(model.blocks)
    expected = {
        "backend": "neuron",
        "fullgraph": True,
        "dynamic": False,
        "options": {"fallback_execution": False},
    }
    assert all(args == () and kwargs == expected for _, args, kwargs in seen)
    # One dict object for every block: the backend wrappers must compare equal to share a graph.
    assert len({id(kwargs["options"]) for _, _, kwargs in seen}) == 1


def test_compile_blocks_turns_neuron_fallback_execution_off_unless_asked(monkeypatch):
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append(k["options"]))
    model = _toy(n_blocks=1)
    neuron_compile.compile_blocks(model, options={"optlevel": 1})
    neuron_compile.compile_blocks(model, options={"fallback_execution": True})
    neuron_compile.compile_blocks(model, backend="aot_eager")
    neuron_compile.compile_blocks(model, backend="aot_eager", options={"x": 1})
    assert seen == [
        {"fallback_execution": False, "optlevel": 1},  # neuron: off by default, merged
        {"fallback_execution": True},  # an explicit opt-in is kept
        None,  # other backends get no neuron options
        {"x": 1},
    ]
    assert dict(neuron_compile.NEURON_COMPILE_OPTIONS) == {"fallback_execution": False}
    with pytest.raises(TypeError):
        neuron_compile.NEURON_COMPILE_OPTIONS["fallback_execution"] = True  # read-only


def test_compile_blocks_hands_the_neuron_backend_one_options_dict_per_option_set(monkeypatch):
    """Every call that resolves to the same options passes the same dict object, found by the
    options as requested even after the backend has edited it; another option set gets a dict
    of its own; the caller's dict is copied, never handed out or edited."""
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append(k["options"]))
    model = _toy(n_blocks=2)
    caller = {"optlevel": 1}
    neuron_compile.compile_blocks(model)  # seen[0:2]
    seen[0]["dynamic"] = False  # what the neuron backend does when it compiles the first graph
    neuron_compile.compile_blocks(model)  # seen[2:4]
    neuron_compile.compile_blocks(model, options=caller)  # seen[4:6]
    neuron_compile.compile_blocks(  # seen[6:8]: the same set, spelled out in another order
        model, options={"optlevel": 1, "fallback_execution": False}
    )
    neuron_compile.compile_blocks(model, options={"fallback_execution": True})  # seen[8:10]
    default, optlevel, fallback_on = seen[0], seen[4], seen[8]
    assert all(options is default for options in seen[0:4])
    assert all(options is optlevel for options in seen[4:8])
    assert all(options is fallback_on for options in seen[8:10])
    assert len({id(default), id(optlevel), id(fallback_on)}) == 3
    assert default == {"fallback_execution": False, "dynamic": False}
    assert optlevel == {"fallback_execution": False, "optlevel": 1}
    assert fallback_on == {"fallback_execution": True}
    assert caller == {"optlevel": 1} and all(options is not caller for options in seen)
    assert dict(neuron_compile.NEURON_COMPILE_OPTIONS) == {"fallback_execution": False}


def _other_toy(n_blocks=N_BLOCKS):
    """Same block class and shapes as _toy(), other weights."""
    torch.manual_seed(1)
    return ToyBlocksModel(n_blocks, DIM, HIDDEN, heads=HEADS).eval()


@pytest.mark.parametrize(
    "edits",
    [(("dynamic", False),), (("dynamic", False), ("added_later", 1))],
    ids=["dynamic", "dynamic+another-key"],
)
@pytest.mark.parametrize("order", ["load_then_load", "compile_both_then_run"])
def test_two_models_of_one_block_class_compile_one_graph(neuron_backend_stub, order, edits):
    """Wan 2.2 A14B's two transformers: one block class, one compile_blocks call per model (one
    per application load), backend="neuron". The neuron backend edits the options dict it gets,
    and Dynamo reuses a block's graph only while the backend wrappers' options compare equal, so
    the second model must be handed the first model's dict, edits included: one graph for both.
    That also holds if the backend edits another key too, which pre-seeding "dynamic" in
    NEURON_COMPILE_OPTIONS would not cover."""
    backend = neuron_backend_stub(edits)
    first, second, x = _toy(), _other_toy(), _input()
    with torch.no_grad():
        refs = [first(x), second(x)]
        if order == "load_then_load":
            neuron_compile.compile_blocks(first)
            outs = [first(x)]
            neuron_compile.compile_blocks(second)
            outs.append(second(x))
        else:
            neuron_compile.compile_blocks(first)
            neuron_compile.compile_blocks(second)
            outs = [first(x), second(x)]
        outs += [first(x), second(x)]  # every block of both models is a cache hit by now
    assert len(backend.graphs) == 1
    assert backend.options == [{"fallback_execution": False, **dict(edits)}]
    for out, ref in zip(outs, refs * 2):
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


class _TwoStacks(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Module()
        self.inner.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
        self.single = nn.Sequential(nn.Linear(4, 4))
        self.empty = nn.ModuleList()


def test_compile_blocks_walks_dotted_and_sequential_containers(monkeypatch):
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: None)
    names = neuron_compile.compile_blocks(_TwoStacks(), block_attrs=("inner.blocks", "single"))
    assert names == ["inner.blocks.0", "inner.blocks.1", "single.0"]


def test_compile_blocks_rejects_missing_empty_and_string_containers(monkeypatch):
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: None)
    model = _TwoStacks()
    with pytest.raises(AttributeError, match="no block container 'blocks'"):
        neuron_compile.compile_blocks(model)
    with pytest.raises(ValueError, match="'empty' .* is empty"):
        neuron_compile.compile_blocks(model, block_attrs=("empty",))
    with pytest.raises(TypeError, match="sequence of attribute names"):
        neuron_compile.compile_blocks(model, block_attrs="single")


# The colleague's fixtures (71aa987): one uniform stack, a mixed list, a one-block list, and a
# list nested in every block.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.inner = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])  # nested stack

    def forward(self, x):
        return self.proj(x)


class Other(nn.Module):
    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.mixed = nn.ModuleList([Block(), Other()])
        self.single = nn.ModuleList([Block()])


def test_finds_outermost_uniform_stacks_only():
    assert neuron_compile.repeated_block_lists(Model()) == ["blocks"]


def test_compile_blocks_without_block_attrs_compiles_the_discovered_stack(monkeypatch):
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append((self, k)))
    model, backend = Model(depth=3), CountingBackend()
    names = neuron_compile.compile_blocks(model, block_attrs=None, backend=backend)
    assert names == ["blocks.0", "blocks.1", "blocks.2"]
    assert [module for module, _ in seen] == list(model.blocks)  # not mixed, single or inner
    assert all(kwargs["backend"] is backend for _, kwargs in seen)


def test_a_bare_module_list_is_the_one_stack(monkeypatch):
    """The model itself is a uniform ModuleList (named ""): it is the stack, the lists nested
    in its blocks are not, and the compiled blocks are named "0", "1", ... (not ".0")."""
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append(self))
    stack = nn.ModuleList([Block() for _ in range(3)])
    assert neuron_compile.repeated_block_lists(stack) == [""]
    assert neuron_compile.compile_blocks(stack, block_attrs=None) == ["0", "1", "2"]
    assert seen == list(stack)
    mixed = nn.ModuleList([Block(), Other()])  # not uniform, and its blocks' lists stay inside
    assert neuron_compile.repeated_block_lists(mixed) == []


def test_compile_blocks_without_block_attrs_raises_when_no_stack_is_found(monkeypatch):
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: None)
    with pytest.raises(ValueError, match="no repeated block stack found; pass block_attrs"):
        neuron_compile.compile_blocks(nn.Sequential(nn.Linear(2, 2)), block_attrs=None)


def test_compile_blocks_still_defaults_to_the_blocks_container(monkeypatch):
    """block_attrs=None is opt-in: the default stays ("blocks",), never discovery."""
    seen = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, *a, **k: seen.append(self))
    model = _TwoStacks()  # inner.blocks is a uniform stack discovery would find
    assert neuron_compile.repeated_block_lists(model) == ["inner.blocks"]
    with pytest.raises(AttributeError, match="no block container 'blocks'"):
        neuron_compile.compile_blocks(model)
    assert seen == []


def test_compile_inplace_returns_the_module_with_its_state_dict_keys():
    block = _toy(n_blocks=1).blocks[0]
    keys = list(block.state_dict())
    backend = CountingBackend()
    assert neuron_compile.compile_inplace(block, backend=backend) is block
    with torch.no_grad():
        block(_input())
    assert len(backend.graphs) == 1
    assert list(block.state_dict()) == keys


def test_explain_graphs_counts_graphs_and_breaks():
    model, x = _toy(), _input()
    with torch.no_grad():
        assert neuron_compile.explain_graphs(model, x) == (1, 0)

    def broken(t):
        t = t + 1
        torch._dynamo.graph_break()
        return t * 2

    assert neuron_compile.explain_graphs(broken, x) == (2, 1)
    assert neuron_compile.explain_graphs(lambda t: t, x) == (0, 0)


def test_toy_block_rejects_heads_that_do_not_split():
    with pytest.raises(ValueError, match="heads=3"):
        ToyBlock(DIM, HIDDEN, heads=3)


def test_a_wan_depth_stack_of_40_blocks_compiles_to_one_graph():
    model, x = _toy(n_blocks=WAN_DEPTH), _input()
    with torch.no_grad():
        ref = model(x)
    backend = CountingBackend()
    names = neuron_compile.compile_blocks(model, backend=backend)
    with torch.no_grad():
        out = model(x)
    assert len(names) == WAN_DEPTH
    assert len(backend.graphs) == 1  # depth costs no extra graph, so no extra NEFF
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- Dynamo's recompile limit


class _MLPBlock(nn.Module):
    """A second block class with its own forward: a pre-norm residual ToyTPMLP."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ToyTPMLP(dim, hidden)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class _TwoClassModel(nn.Module):
    """Two stacks of different block classes, like FLUX's double- and single-stream blocks."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList(ToyBlock(DIM, HIDDEN, heads=HEADS) for _ in range(2))
        self.single_blocks = nn.ModuleList(_MLPBlock(DIM, HIDDEN) for _ in range(2))

    def forward(self, x):
        for block in [*self.blocks, *self.single_blocks]:
            x = block(x)
        return x


SHAPES = (8, 16, 24, 32, 40)  # sequence lengths


def test_two_block_classes_at_five_shapes_stay_within_the_recompile_limit():
    """10 static graphs, 5 per block class. Dynamo keeps each class's entries on that class's
    forward code object, so each class has its own recompile_limit (8): compile_blocks needs no
    higher limit and leaves the setting alone."""
    model = _TwoClassModel().eval()
    inputs = [_input(seq=seq) for seq in SHAPES]
    with torch.no_grad():
        refs = [model(x) for x in inputs]
    backend = CountingBackend()
    with torch._dynamo.config.patch(recompile_limit=8):
        names = neuron_compile.compile_blocks(
            model, block_attrs=("blocks", "single_blocks"), backend=backend
        )
        assert torch._dynamo.config.recompile_limit == 8
        with torch.no_grad():
            outs = [model(x) for x in inputs]
            again = [model(x) for x in inputs]  # every shape is now a cache hit
    assert len(names) == 4
    assert len(backend.graphs) == len(SHAPES) * 2
    for out, ref, rerun in zip(outs, refs, again):
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rerun, out, rtol=0, atol=0)


def test_a_ninth_shape_of_one_block_class_fails_hard_at_the_recompile_limit(caplog):
    """Past recompile_limit, fullgraph=True raises instead of silently running the block eagerly;
    Dynamo's warning names the limit, the block's forward and the guard that failed."""
    model = _toy(n_blocks=2)
    backend = CountingBackend()
    with torch._dynamo.config.patch(recompile_limit=8):
        neuron_compile.compile_blocks(model, backend=backend)
        with torch.no_grad():
            for seq in range(1, 9):
                model(_input(seq=seq))
            assert len(backend.graphs) == 8
            with caplog.at_level(logging.WARNING, logger="torch._dynamo"):
                with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
                    model(_input(seq=9))
    assert len(backend.graphs) == 8
    assert "recompile_limit (8)" in caplog.text
    assert "function: 'forward'" in caplog.text
    assert "size mismatch" in caplog.text


# ---------------------------------------------------------------- C3 kernel path under Dynamo

_STUB_CALLS: list[int] = []


@torch.library.custom_op("difflet_c9_test::flash_stub", mutates_args=())
def _flash_stub(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, scale: float
) -> torch.Tensor:
    _STUB_CALLS.append(1)
    weights = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1)
    return (weights @ v.float()).to(q.dtype)


@_flash_stub.register_fake
def _flash_stub_fake(q, k, v, is_causal, scale):
    return q.new_empty(q.shape)


def _stub_kernel(
    q, k, v, is_causal=False, dropout_p=0.0, scale=None, training=False, seed=None, lse=None
):
    """Same call signature as torch-neuronx's scaled_dot_product_attention_kernel."""
    return _flash_stub(q, k, v, is_causal, scale)


@functools.lru_cache(maxsize=None)
def _stub_kernel_getter():
    """Shaped like attention._flash_kernel: cached, with an import guarded by ImportError."""
    try:
        from difflet.backends.cpu.ops_impl import attention as _cpu_attention  # noqa: F401
    except ImportError:
        return None
    return _stub_kernel


def test_kernel_attention_path_traces_as_one_graph(monkeypatch):
    """The C3 direct-kernel branch (taken on the device for unmasked bf16) stays in one graph:
    the gating, the cached kernel getter and the kernel call, the kernel as one custom-op node."""
    monkeypatch.setattr(neuron_attention, "_on_neuron", lambda t: True)
    monkeypatch.setattr(neuron_attention, "_flash_kernel", _stub_kernel_getter)
    model, x = _toy(dtype=torch.bfloat16), _input(torch.bfloat16)
    _STUB_CALLS.clear()
    with torch.no_grad():
        ref = model(x)
    assert len(_STUB_CALLS) == N_BLOCKS  # eager: every block took the kernel branch, not SDPA
    with torch.no_grad():
        assert neuron_compile.explain_graphs(model, x) == (1, 0)
    backend = CountingBackend()
    neuron_compile.compile_blocks(model, backend=backend)
    with torch.no_grad():
        out = model(x)
    assert len(backend.graphs) == 1
    targets = [str(n.target) for n in backend.graphs[0].graph.nodes if n.op == "call_function"]
    assert "difflet_c9_test.flash_stub.default" in targets
    torch.testing.assert_close(out, ref)


def test_launcher_exports_the_cache_settings_to_every_rank(monkeypatch, tmp_path):
    """The HLO cache is read at torch_neuronx import, so the launcher sets it for the ranks."""
    from difflet.cli import runner

    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    env = {"TORCH_NEURONX_NEFF_CACHE_DIR": "/shared/neff"}
    runner._prepare_mpmd_env(env, 4, strict_environment=False)
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == "/shared/neff"
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == str(tmp_path / "neuron" / "hlo")
    assert env["NKI_ENABLE_TRACE_CACHE"] == "0"
