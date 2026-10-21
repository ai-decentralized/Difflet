"""Backend-resolved layer-boundary markers: identity off Trainium."""

import sys
import types

import torch

from difflet.layers import markers
from difflet.layers.markers import resolve_markers


def test_identity_markers_off_trainium():
    for backend in ("cuda", "cpu", "tpu"):
        start, end = resolve_markers(backend)
        x, y = torch.ones(2), torch.zeros(2)
        assert start()(x) is x and end()(x) is x
        assert start()(x, y) == (x, y)


def test_trainium_resolves_to_the_core_wrappers(monkeypatch):
    fake = types.ModuleType("difflet.backends.trainium.core.layer_boundary_marker")
    fake.ModuleMarkerStartWrapper, fake.ModuleMarkerEndWrapper = object, int
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    assert resolve_markers("trainium") == (object, int)


def test_normalization_uses_the_shared_markers():
    import difflet.layers.normalization as n

    assert n.ModuleMarkerStartWrapper is markers.ModuleMarkerStartWrapper
    assert n.ModuleMarkerEndWrapper is markers.ModuleMarkerEndWrapper
