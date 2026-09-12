"""Unit tests for the TPU backend scaffolding (plan Phase 1)."""

import pytest

from difflet.backends import registry
from difflet.pipeline.compile_cache import CacheSpec
from difflet.pipeline.parallel_config import DiffletParallelConfig


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    registry._get_backend_by_name.cache_clear()
    yield
    registry._get_backend_by_name.cache_clear()


def _spec(**overrides):
    kwargs = dict(
        model_id="Qwen/Qwen-Image",
        model_path="/tmp/does-not-matter",
        model_name="qwen_image",
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        height=1024,
        width=1024,
    )
    kwargs.update(overrides)
    return CacheSpec(**kwargs)


def test_resolve_backend_name_tpu():
    assert registry.resolve_backend_name("tpu") == "tpu"


def test_get_backend_tpu_runtime():
    runtime = registry.get_backend("tpu")
    assert runtime.name == "tpu"
    assert runtime.capabilities.single_process_multi_core is True


def test_tpu_prepare_runtime_is_explicitly_unimplemented():
    runtime = registry.get_backend("tpu")
    with pytest.raises(NotImplementedError, match="tpu"):
        runtime.prepare_runtime(parallel=object())


def test_auto_detect_prefers_trainium_over_tpu(monkeypatch):
    # Neuron venvs ship torch_xla too; torch_neuronx must win.
    present = {"torch_neuronx", "torch_xla", "libtpu"}
    monkeypatch.setattr(
        registry.importlib.util,
        "find_spec",
        lambda name: object() if name in present else None,
    )
    assert registry._auto_detect_backend() == "trainium"


def test_auto_detect_tpu_host(monkeypatch):
    present = {"torch_xla", "libtpu"}
    monkeypatch.setattr(
        registry.importlib.util,
        "find_spec",
        lambda name: object() if name in present else None,
    )
    assert registry._auto_detect_backend() == "tpu"


def test_cache_key_unchanged_for_default_and_trainium_backend():
    # Additive-only policy: None / "trainium" must be byte-identical to the
    # pre-backend key so existing Trainium caches stay valid.
    base = _spec().cache_inputs()
    assert _spec(backend=None).cache_inputs() == base
    assert _spec(backend="trainium").cache_inputs() == base
    assert "backend" not in base


def test_cache_key_differs_for_tpu_backend():
    base = _spec().cache_inputs()
    tpu = _spec(backend="tpu").cache_inputs()
    assert tpu != base
    assert tpu["backend"] == "tpu"
    assert "libtpu" in tpu["toolchain"]


def test_registry_models_reject_tpu_backend():
    # The guard must fire before any model code is imported. HunyuanVideo-1.5
    # is the one remaining Trainium-only registration (FLUX, the last of the
    # serving models, was ported in 2026-09; its TPU path never imports the
    # legacy NxDI fork).
    from difflet.registry import resolve_model

    entry = resolve_model("tencent/HunyuanVideo-1.5")
    with pytest.raises(ValueError, match="tpu"):
        entry.require_backend("tpu")
