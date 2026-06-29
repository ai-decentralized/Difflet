"""Unit tests for difflet.backends.base contracts."""

import dataclasses

from difflet.backends.base import BackendCapabilities, BackendRuntime


def test_capabilities_is_frozen_dataclass():
    caps = BackendCapabilities(
        requires_aot=True,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )
    assert caps.requires_aot is True
    assert caps.single_process_multi_core is False
    assert caps.supports_torchrun_mpmd is True
    assert dataclasses.is_dataclass(caps)
    # frozen -> assignment raises
    try:
        caps.requires_aot = False
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_backend_runtime_default_prepare_runtime_returns_none():
    rt = BackendRuntime()
    assert rt.prepare_runtime(parallel=object()) is None


def test_backend_runtime_default_resolve_load_rank_range_passthrough():
    rt = BackendRuntime()
    out = rt.resolve_load_rank_range(start_rank_id=2, local_ranks_size=4)
    assert out == (2, 4)
    out_none = rt.resolve_load_rank_range(start_rank_id=None, local_ranks_size=None)
    assert out_none == (None, None)


def test_cpu_backend_runtime_prepare_runtime():
    from difflet.backends.cpu.runtime import CpuBackend, create_backend

    backend = create_backend()
    assert isinstance(backend, CpuBackend)
    assert backend.name == "cpu"
    assert backend.prepare_runtime(parallel=object()) is None


def test_backend_runtime_subclass_overrides():
    class MyRuntime(BackendRuntime):
        name = "my"
        capabilities = BackendCapabilities(False, False, False)

    rt = MyRuntime()
    assert rt.name == "my"
    assert rt.prepare_runtime(parallel=None) is None
