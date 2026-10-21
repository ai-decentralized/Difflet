"""Unit tests for the CUDA backend scaffolding (CUDA plan Phase 1, task 1.3).

Runs on a host without CUDA: the device calls are monkeypatched. The real
``prepare_runtime`` on a GPU is exercised by ``scripts/cuda_smoke.sh`` in Phase 4.
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


def test_prepare_runtime_multi_process_raises_f2(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match="not supported"):
        registry.get_backend("cuda").prepare_runtime(parallel=object())


def test_prepare_runtime_single_process_sets_local_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))
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
    # Pinned on purpose (plan Review Focus 1): a CPU-only host must set
    # DIFFLET_BACKEND=cpu explicitly; silently defaulting to cpu would be a decision.
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert registry._auto_detect_backend() == "trainium"
