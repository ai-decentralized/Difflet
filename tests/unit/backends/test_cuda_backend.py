"""Unit tests for the CUDA backend runtime.

Runs on a host without CUDA: the device calls are monkeypatched. The real
``prepare_runtime`` on a GPU is exercised by ``scripts/cuda_smoke.sh``.
"""

import pytest
import torch

from difflet.backends import registry


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    registry._get_backend_by_name.cache_clear()
    yield
    registry._get_backend_by_name.cache_clear()


def test_resolve_backend_name_cuda():
    assert registry.resolve_backend_name("cuda") == "cuda"


def test_cuda_capabilities_match_decision_d5():
    caps = registry.get_backend("cuda").capabilities
    assert (caps.requires_aot, caps.single_process_multi_core, caps.supports_torchrun_mpmd) == (
        False,
        False,
        True,
    )


def test_prepare_runtime_without_cuda_raises_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available"):
        registry.get_backend("cuda").prepare_runtime(parallel=object())


def _fake_distributed(monkeypatch, *, initialised: bool):
    """Stand in for torch.distributed so the torchrun path runs on a CPU host."""
    import atexit

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(("set_device", d)))
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: calls.append(("atexit", fn)))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialised)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend=None, **kw: calls.append(("init_process_group", backend)),
    )
    return calls


def test_prepare_runtime_multi_process_initialises_nccl_and_binds_local_rank(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    registry.get_backend("cuda").prepare_runtime(parallel=object())
    # The device is bound before the NCCL communicator is created.
    assert calls[:2] == [("set_device", 1), ("init_process_group", "nccl")]


def test_prepare_runtime_destroys_the_process_group_it_created_at_exit(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    registry.get_backend("cuda").prepare_runtime(parallel=object())
    hooks = [fn for name, fn in calls if name == "atexit"]
    assert len(hooks) == 1
    (hook,) = hooks

    destroyed = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: destroyed.append(1))
    # Still initialised at exit: destroyed once, so the ranks exit cleanly.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    hook()
    assert destroyed == [1]
    # Someone already tore it down: the hook is a no-op rather than an error.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    hook()
    assert destroyed == [1]


def test_prepare_runtime_multi_process_reuses_an_existing_process_group(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    registry.get_backend("cuda").prepare_runtime(parallel=object())
    assert calls == [("set_device", 0)]


def test_prepare_runtime_single_process_sets_local_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))

    def _no_init(*a, **kw):
        raise AssertionError("world size 1 must not initialise torch.distributed")

    monkeypatch.setattr(torch.distributed, "init_process_group", _no_init)
    registry.get_backend("cuda").prepare_runtime(parallel=object())
    assert calls == [0]


def test_auto_detect_order_trainium_tpu_cuda_rocm(monkeypatch):
    # torch_neuronx present wins over a CUDA device (a Neuron venv can have both).
    present = {"torch_neuronx"}
    monkeypatch.setattr(
        registry.importlib.util, "find_spec", lambda n: object() if n in present else None
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert registry._auto_detect_backend() == "trainium"


def test_auto_detect_cuda_when_device_available(monkeypatch):
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert registry._auto_detect_backend() == "cuda"


def test_auto_detect_falls_back_to_trainium_without_accelerator(monkeypatch):
    # Pinned on purpose: a CPU-only host must set
    # DIFFLET_BACKEND=cpu explicitly; silently defaulting to cpu would be a decision.
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert registry._auto_detect_backend() == "trainium"
