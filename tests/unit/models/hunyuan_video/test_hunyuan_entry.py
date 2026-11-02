"""CPU coverage for HunyuanVideo registry entry factories.

These factories validate backend/parallelism options and, on the happy path,
construct ``NeuronHunyuanVideoApplication`` against a model path with no
``config.json`` (so no Trainium backbone components are built).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import entry
from difflet.pipeline.parallel_config import DiffletParallelConfig


def _parallel(**kwargs):
    return DiffletParallelConfig(tp_degree=1, **kwargs)


def test_create_application_happy_path():
    with tempfile.TemporaryDirectory() as path:
        application = entry.create_hunyuan_video_application(
            model_path=path,
            parallel=_parallel(),
            dtype="bf16",
            shape={"height": None, "width": None, "num_frames": None},
        )
    assert type(application).__name__ == "NeuronHunyuanVideoApplication"
    assert application.model_version == "1.0"


def test_create_application15_sets_model_version():
    with tempfile.TemporaryDirectory() as path:
        application = entry.create_hunyuan_video15_application(
            model_path=path,
            parallel=_parallel(),
            dtype=torch.float32,
            shape={},
        )
    assert application.model_version == "1.5"


def test_create_application_rejects_backends_other_than_trainium_and_neuron():
    # The neuron backend is accepted (tests/unit/backends/test_neuron_hv1_app.py builds it).
    with pytest.raises(NotImplementedError, match="trainium and neuron"):
        entry.create_hunyuan_video_application(
            model_path="x", parallel=_parallel(), dtype="bf16", shape={}, backend="cpu"
        )


def test_create_application15_rejects_non_trainium_backend():
    with pytest.raises(NotImplementedError, match="trainium"):
        entry.create_hunyuan_video15_application(
            model_path="x", parallel=_parallel(), dtype="bf16", shape={}, backend="cuda"
        )


def test_create_application_rejects_cfg_parallel():
    with pytest.raises(NotImplementedError, match="CFG-parallel"):
        entry.create_hunyuan_video_application(
            model_path="x",
            parallel=_parallel(cfg_parallel_enabled=True),
            dtype="bf16",
            shape={},
        )


def test_create_application15_accepts_cfg_parallel():
    with tempfile.TemporaryDirectory() as path:
        application = entry.create_hunyuan_video15_application(
            model_path=path,
            parallel=_parallel(cfg_parallel_enabled=True),
            dtype=torch.float32,
            shape={},
        )
    # [uncond, cond] are stacked into batch 2 and scattered one branch per cfg rank.
    assert application.backbone_batch_size == 2
    assert application.batch_size == 1  # VAE and other components stay at the base batch


@pytest.mark.parametrize("cp_mode", ["ulysses", "gather_kv"])
def test_create_application15_accepts_context_parallel(cp_mode):
    with tempfile.TemporaryDirectory() as path:
        application = entry.create_hunyuan_video15_application(
            model_path=path,
            parallel=_parallel(cp_degree=2, cp_mode=cp_mode),
            dtype=torch.float32,
            shape={},
        )
    assert application.model_version == "1.5"
    assert application.parallel.world_size == 2


def test_ring_rejected_before_compile(monkeypatch):
    from difflet.models.hunyuan_video import application as app

    def built(**kwargs):
        raise AssertionError("ring must be rejected before the application is built")

    monkeypatch.setattr(app, "NeuronHunyuanVideoApplication", built)
    with pytest.raises(NotImplementedError, match="ulysses") as err:
        entry.create_hunyuan_video15_application(
            model_path="x",
            parallel=_parallel(cp_degree=2, cp_mode="ring"),
            dtype="bf16",
            shape={},
        )
    assert "text" in str(err.value)


def test_registry_keeps_ring_for_native():
    # Only the AOT entry rejects ring; the native backend's ring path stays registered.
    from difflet.registry import resolve_model

    model = resolve_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    assert "ring" in model.capabilities.cp_modes
