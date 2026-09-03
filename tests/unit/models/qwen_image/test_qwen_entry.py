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


def test_entry_rejects_an_unsupported_backend(tmp_path):
    # Qwen-Image gained the tpu backend in plan Phase 4; everything else must
    # still be refused rather than silently falling through to Trainium.
    with pytest.raises(NotImplementedError, match="trainium and tpu"):
        create_qwen_image_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape={"height": 64, "width": 64},
            backend="cpu",
        )


def test_entry_builds_the_tpu_application(tmp_path):
    # No transformer/config.json under tmp_path, so the app builds with no
    # components — enough to prove the tpu branch is wired without needing a
    # checkpoint or a TPU.
    app = create_qwen_image_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        backend="tpu",
    )
    assert type(app).__name__ == "TpuQwenImageApplication"
    assert app.transformer is None
    assert app.has_compiled_artifacts(str(tmp_path)) is False


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
