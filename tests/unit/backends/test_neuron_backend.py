"""Unit tests for the neuron (TorchNeuron) backend runtime; no Neuron hardware needed."""

import os

import pytest

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


@pytest.mark.parametrize(
    "parallel, name",
    [
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), "cp_degree=2"),
        (DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), "cfg_parallel_enabled"),
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
    # prepare_runtime exports the NEFF cache settings; delenv records them for restoration.
    for name in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
                 "NKI_ENABLE_TRACE_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    monkeypatch.setattr(runtime, "_runtime_prepared", False)  # restored after the call sets it
    runtime.NeuronBackend().prepare_runtime(DiffletParallelConfig())
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neuron" / "neff")
    assert os.environ["NKI_ENABLE_TRACE_CACHE"] == "0"
