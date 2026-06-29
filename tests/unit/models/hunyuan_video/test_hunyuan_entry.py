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


def test_create_application_rejects_non_trainium_backend():
    with pytest.raises(NotImplementedError, match="trainium"):
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


def test_create_application15_rejects_cfg_parallel():
    with pytest.raises(NotImplementedError, match="CFG-parallel"):
        entry.create_hunyuan_video15_application(
            model_path="x",
            parallel=_parallel(cfg_parallel_enabled=True),
            dtype="bf16",
            shape={},
        )


def test_create_application15_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="CP is deferred"):
        entry.create_hunyuan_video15_application(
            model_path="x",
            parallel=_parallel(cp_degree=2),
            dtype="bf16",
            shape={},
        )
