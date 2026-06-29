"""CPU-only unit coverage for difflet.models.qwen_image.entry."""

import os


import pytest

from difflet.models.qwen_image.entry import create_qwen_image_application
from difflet.models.qwen_image.application import NeuronQwenImageApplication
from difflet.pipeline.parallel_config import DiffletParallelConfig


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def test_entry_rejects_non_trainium_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium backend"):
        create_qwen_image_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape={"height": 64, "width": 64},
            backend="cpu",
        )


def test_entry_rejects_cfg_parallel(tmp_path):
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        create_qwen_image_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True),
            dtype="bf16",
            shape={"height": 64, "width": 64},
        )


def test_entry_builds_application(tmp_path):
    app = create_qwen_image_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": 64, "width": 64},
    )
    assert isinstance(app, NeuronQwenImageApplication)
    assert app.transformer is None  # empty model dir -> no transformer/config.json
