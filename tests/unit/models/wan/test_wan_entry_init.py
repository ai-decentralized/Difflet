"""Coverage for difflet.models.wan.entry and the package __init__ lazy loader."""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.models.wan import entry


def test_create_wan_application_rejects_unsupported_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium and tpu"):
        entry.create_wan_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(),
            dtype="bf16",
            shape={},
            backend="cuda",
        )


def test_create_wan_application_builds_tpu_app(tmp_path):
    """The tpu branch must not fall through to the Trainium application.

    ``tmp_path`` has no ``transformer/config.json``, so the app is built
    without a DiT — enough to prove the dispatch without a device.
    """
    from difflet.models.wan.tpu_application import TpuWanApplication

    app = entry.create_wan_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
        backend="tpu",
    )
    assert isinstance(app, TpuWanApplication)
    assert app.transformer is None
    # Two resident experts do not fit a v5e chip at tp=4; opting in is the
    # caller's decision, never the default.
    assert app.enable_transformer_2 is False


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
