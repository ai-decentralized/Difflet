"""Unit tests for the neuron (TorchNeuron) backend runtime; no Neuron hardware needed."""

import datetime
import os
import sys
import types

import pytest
import torch.distributed as dist

from difflet import envs
from difflet.backends import registry
from difflet.backends.neuron import runtime
from difflet.pipeline.parallel_config import DiffletParallelConfig


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    registry._get_backend_by_name.cache_clear()
    yield
    registry._get_backend_by_name.cache_clear()


def test_resolve_and_get_neuron_backend():
    assert registry.resolve_backend_name("Neuron") == "neuron"
    backend = registry.get_backend("neuron")
    assert backend.name == "neuron"
    assert backend.capabilities.requires_aot is False
    assert backend.capabilities.single_process_multi_core is False
    assert backend.capabilities.supports_torchrun_mpmd is True


@pytest.mark.parametrize(
    "present, expected",
    [({"torch_neuronx"}, "neuron"), ({"torch_neuronx", "torch_xla"}, "trainium")],
)
def test_auto_detect_neuron_vs_trainium(monkeypatch, present, expected):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(
        registry.importlib.util, "find_spec", lambda name: object() if name in present else None
    )
    assert registry._auto_detect_backend() == expected


@pytest.mark.parametrize(
    "value, cores",
    [(None, []), ("", []), ("2", [2]), ("0-3", [0, 1, 2, 3]), ("0,2,3", [0, 2, 3]), ("0-1,4-5", [0, 1, 4, 5])],
)
def test_parse_core_list(value, cores):
    assert runtime.parse_core_list(value) == cores


def test_parse_core_list_rejects_reversed_range():
    with pytest.raises(ValueError, match="invalid core range"):
        runtime.parse_core_list("3-1")


def test_bind_core_single_process_is_noop():
    env = {}
    assert runtime.bind_core(env) is None
    assert "NEURON_RT_VISIBLE_CORES" not in env


def test_bind_core_uses_local_rank_when_unset():
    env = {"LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "2"}
    assert runtime.bind_core(env) == 2
    assert env["NEURON_RT_VISIBLE_CORES"] == "2"


def test_bind_core_indexes_the_launch_visible_cores():
    env = {"LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1", "NEURON_RT_VISIBLE_CORES": "2-3"}
    assert runtime.bind_core(env) == 3
    assert env["NEURON_RT_VISIBLE_CORES"] == "3"


def test_bind_core_keeps_an_existing_single_core_binding():
    env = {"LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "1", "NEURON_RT_VISIBLE_CORES": "5"}
    assert runtime.bind_core(env) == 5
    assert env["NEURON_RT_VISIBLE_CORES"] == "5"


def test_bind_core_rejects_more_ranks_than_cores():
    env = {"LOCAL_WORLD_SIZE": "4", "LOCAL_RANK": "0", "NEURON_RT_VISIBLE_CORES": "0-1"}
    with pytest.raises(ValueError, match="4 local ranks but only 2 cores"):
        runtime.bind_core(env)


def test_tensor_parallel_is_supported():
    runtime.check_parallel_supported(DiffletParallelConfig(tp_degree=4))


def test_cfg_parallel_is_supported():
    runtime.check_parallel_supported(DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True))


@pytest.mark.parametrize(
    "parallel, name",
    [
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), "cp_degree=2"),
        (DiffletParallelConfig(tp_degree=4, sp_enabled=True), "sp_enabled"),
        (DiffletParallelConfig(tp_degree=2, dp_degree=2), "dp_degree=2"),
    ],
)
def test_other_parallel_modes_are_rejected(parallel, name):
    with pytest.raises(NotImplementedError, match=name):
        runtime.check_parallel_supported(parallel)


def test_prepare_runtime_rejects_world_size_mismatch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="torchrun --nproc-per-node 4"):
        runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig(tp_degree=4))


def test_prepare_runtime_single_process_needs_no_device(monkeypatch, tmp_path):
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    # prepare_runtime exports the cache settings; setenv+delenv records them for restoration
    # (delenv alone records nothing for a variable that is not set).
    for name in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
                 "TORCH_NEURONX_HLO_CACHE_DIR", "NKI_ENABLE_TRACE_CACHE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    monkeypatch.setattr(runtime, "_runtime_prepared", False)  # restored after the call sets it
    # The real hlo_cache module is imported here, so the call may log its once-per-process
    # HLO warning and set this flag; restore it so no later test inherits that state.
    from difflet.backends.neuron import compile as neuron_compile

    monkeypatch.setattr(neuron_compile, "_hlo_cache_warned", False)
    runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neuron" / "neff")
    assert os.environ["TORCH_NEURONX_HLO_CACHE_DIR"] == str(tmp_path / "neuron" / "hlo")
    assert os.environ["NKI_ENABLE_TRACE_CACHE"] == "0"


# ------------------------------------------- process-group timeout (DIFFLET_NEURON_PG_TIMEOUT_S)


def _record_process_group_creation(monkeypatch, timeout_env):
    """A 4-rank launch whose ``init_process_group`` call is recorded instead of run: a stub
    ``torch_neuronx`` (the real import registers the device) and no existing group."""
    if timeout_env is None:
        monkeypatch.delenv("DIFFLET_NEURON_PG_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("DIFFLET_NEURON_PG_TIMEOUT_S", timeout_env)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setitem(sys.modules, "torch_neuronx", types.ModuleType("torch_neuronx"))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **kw: calls.append((a, kw)))
    return calls


@pytest.mark.parametrize("timeout_env", [None, "", "  "])
def test_init_process_group_keeps_the_torch_default_timeout_when_the_knob_is_unset(
    monkeypatch, timeout_env
):
    calls = _record_process_group_creation(monkeypatch, timeout_env)
    runtime.init_process_group()
    # No timeout argument at all, so torch's own default (30 min for non-NCCL groups) applies.
    assert calls == [(("neuron",), {})]


@pytest.mark.parametrize(
    "timeout_env, seconds", [("3600", 3600), (" 5400 ", 5400), ("90.5", 90.5)]
)
def test_init_process_group_passes_the_timeout_knob(monkeypatch, timeout_env, seconds):
    calls = _record_process_group_creation(monkeypatch, timeout_env)
    runtime.init_process_group()
    assert calls == [(("neuron",), {"timeout": datetime.timedelta(seconds=seconds)})]


@pytest.mark.parametrize("timeout_env", ["abc", "30m", "0", "-5", "nan", "inf"])
def test_init_process_group_rejects_an_invalid_timeout_before_creating_the_group(
    monkeypatch, timeout_env
):
    calls = _record_process_group_creation(monkeypatch, timeout_env)
    with pytest.raises(
        ValueError, match="DIFFLET_NEURON_PG_TIMEOUT_S=.*positive number of seconds"
    ):
        runtime.init_process_group()
    assert calls == []


def test_single_process_creates_no_group_whatever_the_timeout_knob(monkeypatch):
    calls = _record_process_group_creation(monkeypatch, "3600")
    monkeypatch.setenv("WORLD_SIZE", "1")
    runtime.init_process_group()
    assert calls == []


def test_pg_timeout_is_a_registered_env_var(monkeypatch):
    assert "DIFFLET_NEURON_PG_TIMEOUT_S" in dir(envs)
    monkeypatch.delenv("DIFFLET_NEURON_PG_TIMEOUT_S", raising=False)
    assert envs.DIFFLET_NEURON_PG_TIMEOUT_S is None
    monkeypatch.setenv("DIFFLET_NEURON_PG_TIMEOUT_S", "3600")
    assert envs.DIFFLET_NEURON_PG_TIMEOUT_S == "3600"
