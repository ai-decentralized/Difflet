"""Unit tests for difflet.backends.registry."""

import pytest

from difflet.backends import registry


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    registry._get_backend_by_name.cache_clear()
    yield
    registry._get_backend_by_name.cache_clear()


def test_resolve_backend_name_explicit_arg():
    assert registry.resolve_backend_name("cpu") == "cpu"
    assert registry.resolve_backend_name("TRAINIUM") == "trainium"
    assert registry.resolve_backend_name("  Cuda  ") == "cuda"


def test_resolve_backend_name_env_override(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    assert registry.resolve_backend_name() == "cpu"


def test_resolve_backend_name_no_value_calls_auto_detect(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    # returns whatever the host auto-detects; must be a known backend name
    name = registry.resolve_backend_name()
    assert name in registry._BACKEND_FACTORIES


def test_resolve_backend_name_invalid_raises():
    with pytest.raises(ValueError, match="unknown Difflet backend"):
        registry.resolve_backend_name("nonsense")


def test_current_backend_uses_env(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    assert registry.current_backend() == "cpu"


def test_get_backend_returns_matching_runtime(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    backend = registry.get_backend()
    assert backend.name == "cpu"
    assert backend.capabilities.requires_aot is False


def test_get_backend_by_name_explicit():
    backend = registry.get_backend("cpu")
    assert backend.name == "cpu"


def test_get_backend_is_cached():
    a = registry.get_backend("cpu")
    b = registry.get_backend("cpu")
    assert a is b


def test_auto_detect_trainium_when_torch_neuronx_present(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(
        registry.importlib.util, "find_spec", lambda name: object()
    )
    assert registry._auto_detect_backend() == "trainium"


def test_auto_detect_cuda(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda name: None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert registry._auto_detect_backend() == "cuda"


def test_auto_detect_rocm(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda name: None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", "5.0", raising=False)
    assert registry._auto_detect_backend() == "rocm"


def test_auto_detect_falls_back_to_trainium(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(registry.importlib.util, "find_spec", lambda name: None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert registry._auto_detect_backend() == "trainium"
