"""Unit tests for the neuron backend's norm, rotary-embedding and platform ops; no Neuron hardware needed."""

import builtins

import pytest

from difflet.backends.neuron.ops_impl import platform


@pytest.fixture(autouse=True)
def _fresh_platform_cache():
    platform.get_platform_target.cache_clear()
    yield
    platform.get_platform_target.cache_clear()


def _without_torch_neuronx(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torch_neuronx"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_platform_override_wins(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", " TRN2 ")
    assert platform.get_platform_target() == "trn2"
    assert platform.hardware(platform.get_platform_target()) is platform.hardware.TRN2


def test_platform_from_instance_type(monkeypatch):
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    _without_torch_neuronx(monkeypatch)
    monkeypatch.setattr(platform, "_read_product_name", lambda: "trn2.3xlarge\n")
    assert platform.get_platform_target() == "trn2"


@pytest.mark.parametrize(
    "product, target",
    [("trn1.32xlarge", "trn1"), ("trn2u.48xlarge", "trn2"), ("m7i.xlarge", None), (None, None)],
)
def test_target_from_instance_type(product, target):
    assert platform._target_from_instance_type(product) == target


def test_platform_unknown_raises(monkeypatch):
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    _without_torch_neuronx(monkeypatch)
    monkeypatch.setattr(platform, "_read_product_name", lambda: None)
    with pytest.raises(RuntimeError, match="NEURON_PLATFORM_TARGET_OVERRIDE"):
        platform.get_platform_target()


def test_ops_dispatch_to_the_cpu_implementations(monkeypatch):
    pytest.importorskip("torch")
    from difflet.backends import registry
    from difflet.backends.cpu.ops_impl import embeddings as cpu_embeddings
    from difflet.backends.cpu.ops_impl import norm as cpu_norm
    from difflet.ops._dispatch import load_backend_attr

    registry._get_backend_by_name.cache_clear()
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    assert load_backend_attr("norm", "RMSNorm") is cpu_norm.RMSNorm
    assert load_backend_attr("norm", "LayerNorm") is cpu_norm.LayerNorm
    assert load_backend_attr("norm", "CustomRMSNorm") is cpu_norm.CustomRMSNorm
    assert load_backend_attr("embeddings", "apply_rotary_emb") is cpu_embeddings.apply_rotary_emb
    registry._get_backend_by_name.cache_clear()
