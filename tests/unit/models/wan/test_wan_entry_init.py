"""Coverage for difflet.models.wan.entry and the package __init__ lazy loader."""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.models.wan import entry


def test_create_wan_application_rejects_non_trainium_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium backend"):
        entry.create_wan_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(),
            dtype="bf16",
            shape={},
            backend="cuda",
        )


def test_create_wan_application_builds_skeleton_app(tmp_path):
    from difflet.models.wan.application import NeuronWanApplication

    app = entry.create_wan_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
    )
    assert isinstance(app, NeuronWanApplication)
    # No config.json in tmp_path → no active components.
    assert app.components() == []


def test_package_lazy_getattr_exports():
    import difflet.models.wan as wan_pkg

    assert wan_pkg.NeuronWanApplication.__name__ == "NeuronWanApplication"
    assert wan_pkg.WanOrchestrator.__name__ == "WanOrchestrator"
    assert wan_pkg.WanPipelineOutput.__name__ == "WanPipelineOutput"


def test_package_lazy_getattr_unknown_attribute_raises():
    import difflet.models.wan as wan_pkg

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = wan_pkg.NotARealThing
