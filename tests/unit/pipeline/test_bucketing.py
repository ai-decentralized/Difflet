"""``difflet.pipeline.bucketing``: the backend-neutral half of shape bucketing.

``canonicalize_shapes`` is pure Python + torch and is used by the pipeline, the
CLI, serving and every model entry, so it must import without the Trainium
backend (CUDA plan Phase 1, task 1.5). The Trainium bucketing module keeps the
NxD-specific half and re-exports these names, so on-device scripts that still
import the old path keep working.
"""

import sys

import pytest


def test_canonicalize_shapes_importable_without_trainium(monkeypatch):
    for name in [k for k in sys.modules if k.startswith("difflet.backends.trainium")]:
        monkeypatch.delitem(sys.modules, name)
    from difflet.pipeline.bucketing import canonicalize_shapes

    assert canonicalize_shapes([(512, 512), (1024, 1024)]) == ((1024, 1024, None), (512, 512, None))
    assert canonicalize_shapes([(320, 512, 33), (320, 512, 61)]) == (
        (320, 512, 61),
        (320, 512, 33),
    )
    assert not any(k.startswith("difflet.backends.trainium") for k in sys.modules)


def test_trainium_bucketing_reexports_same_object():
    trn = pytest.importorskip(
        "difflet.backends.trainium.core.bucketing", reason="Trainium module needs the Neuron SDK"
    )
    from difflet.pipeline import bucketing as neutral

    assert trn.canonicalize_shapes is neutral.canonicalize_shapes
    assert trn.CompileShape == neutral.CompileShape
