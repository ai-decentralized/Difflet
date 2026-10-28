"""The torch-native ``neuron`` platform: registered, resolved by name, detected apart from Trainium
(torch-neuronx without torch_xla), and identical to the neuron backend's runtime."""

from __future__ import annotations

import importlib.util

import pytest

from difflet.platforms import detect_platform_name, get_platform


def _fake_find_spec(present: set[str]):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in {"torch_neuronx", "torch_xla", "libtpu"}:
            return object() if name in present else None
        return real(name, *args, **kwargs)

    return find_spec


def test_neuron_resolves_by_name_and_is_the_neuron_backend_runtime():
    from difflet.backends.neuron.runtime import NeuronBackend
    from difflet.backends.registry import available_backends, get_backend, resolve_backend_name

    platform = get_platform("neuron")
    assert isinstance(platform, NeuronBackend) and platform.name == "neuron"
    assert get_backend("neuron") is platform
    assert resolve_backend_name("neuron") == "neuron" and "neuron" in available_backends()


def test_neuron_capabilities_describe_the_torchrun_path():
    caps = get_platform("neuron").capabilities
    assert caps.flow == "torchrun" and caps.supports_torchrun_mpmd
    assert not caps.requires_aot and not caps.single_process_multi_core


@pytest.mark.parametrize("present, expected", [
    ({"torch_neuronx"}, "neuron"),
    ({"torch_neuronx", "torch_xla"}, "trainium"),
])
def test_detection_tells_the_native_stack_from_trainium(monkeypatch, present, expected):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present))
    assert detect_platform_name() == expected
    assert type(get_platform("neuron")).detect() is (expected == "neuron")
    assert type(get_platform("trainium")).detect() is (expected == "trainium")


def test_neuron_layer_markers_are_the_identity():
    import torch

    from difflet.layers.markers import resolve_markers

    start, end = resolve_markers("neuron")
    x, y = torch.ones(2), torch.zeros(3)
    assert end()(x) is x
    assert start()(x, y) == (x, y)


def test_neuron_toolchain_versions_name_the_stack():
    assert set(get_platform("neuron").toolchain_versions()) == {"torch-neuronx", "neuronx-cc", "nki"}
