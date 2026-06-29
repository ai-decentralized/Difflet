"""CPU unit tests for difflet.models.flux.entry.create_flux_application.

Covers the backend guard and the config-flow wiring up to (but not including)
the NeuronFluxApplication runtime construction, which needs a real checkpoint.
"""

import importlib
import os

import pytest

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402,F401


def _reload(name):
    return importlib.reload(importlib.import_module(name))


entry = _reload("difflet.models.flux.entry")
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402

# Restore process-wide backend env so collecting other (trainium-only) test
# modules in the same session is unaffected.
if _PREV_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND


def test_create_flux_application_rejects_non_trainium_backend():
    with pytest.raises(NotImplementedError):
        entry.create_flux_application(
            model_path="/fake/model",
            parallel=DiffletParallelConfig(),
            dtype=torch.float32,
            shape={"height": 512, "width": 512},
            backend="cuda",
        )


def test_create_flux_application_wires_config_flow(monkeypatch):
    """The trainium path should compute world_size, build configs, and hand off
    to NeuronFluxApplication. We stub the application module's heavy pieces so
    only the entry wiring is exercised."""
    import difflet.models.flux.application as app

    captured = {}

    def fake_get_world_size(backbone_tp_degree, cp_degree=1, cfg_parallel_enabled=False):
        captured["world_size_args"] = (backbone_tp_degree, cp_degree, cfg_parallel_enabled)
        return backbone_tp_degree * cp_degree

    def fake_create_config(**kwargs):
        captured["config_kwargs"] = kwargs
        return ("clip", "t5", "backbone", "decoder")

    class FakeApplication:
        def __init__(self, model_path, *configs, height, width, **kwargs):
            captured["app"] = dict(
                model_path=model_path, configs=configs, height=height, width=width
            )

    monkeypatch.setattr(app, "get_flux_parallelism_config", fake_get_world_size)
    monkeypatch.setattr(app, "create_flux_config", fake_create_config)
    monkeypatch.setattr(app, "NeuronFluxApplication", FakeApplication)

    parallel = DiffletParallelConfig(tp_degree=2, cp_degree=1)
    result = entry.create_flux_application(
        model_path="/fake/model",
        parallel=parallel,
        dtype=torch.float32,
        shape={"height": None, "width": 256},
    )
    assert isinstance(result, FakeApplication)
    assert captured["world_size_args"][0] == 2
    # height falls back to default 1024 when None; width uses provided 256
    assert captured["config_kwargs"]["height"] == 1024
    assert captured["config_kwargs"]["width"] == 256
    assert captured["app"]["configs"] == ("clip", "t5", "backbone", "decoder")
