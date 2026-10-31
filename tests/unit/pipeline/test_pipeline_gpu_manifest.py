"""GPU manifest preparation through the public factory, without model checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from difflet.backends.gpu.application import GpuComponent
from difflet.pipeline import difflet_pipeline as factory
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.parallel_mesh import MeshSpec


class TinyComponent(GpuComponent):
    def __init__(self):
        super().__init__(device="cpu", dtype=torch.float32)

    def _mesh(self):
        return MeshSpec(tp=1)

    def build_module(self):
        return torch.nn.Identity()


@pytest.fixture
def gpu_factory(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        name="cuda",
        capabilities=SimpleNamespace(flow="gpu", requires_aot=False),
        prepare_runtime=lambda parallel: None,
        resolve_load_rank_range=lambda **kwargs: (0, 1),
    )
    entry = SimpleNamespace(
        name="tiny_gpu",
        default_parallel=DiffletParallelConfig(tp_degree=1),
        download_patterns=(),
        require_backend=lambda backend: None,
        resolve_shape=lambda **kwargs: {"height": 16, "width": 16, "num_frames": None},
        create_application=lambda **kwargs: TinyComponent(),
    )
    monkeypatch.setattr(factory, "get_backend", lambda backend: runtime)
    monkeypatch.setattr(factory, "resolve_model", lambda *args, **kwargs: entry)
    monkeypatch.setattr(factory, "resolve_model_path", lambda *args, **kwargs: str(tmp_path))

    def create(**kwargs):
        return factory.DiffletPipeline.from_pretrained(
            "tiny-gpu",
            backend="cuda",
            dtype=torch.float32,
            compile_cache_dir=str(tmp_path / "cache"),
            skip_warmup=True,
            **kwargs,
        )

    return create


@pytest.mark.parametrize("manual", [False, True])
def test_gpu_manifest_is_prepared_before_real_component_load(gpu_factory, manual):
    pipe = gpu_factory(load=not manual, skip_compile=manual)
    if manual:
        assert not (pipe.compiled_path / "cuda_manifest.json").exists()
        pipe.compile()
        pipe.load(skip_warmup=True)
    assert pipe.app.is_loaded
    assert pipe.app.has_compiled_artifacts(pipe.compiled_path)
    value = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(pipe.app(value), value)


def test_missing_gpu_manifest_is_repaired_with_valid_pipeline_cache(gpu_factory):
    first = gpu_factory()
    (first.compiled_path / "cuda_manifest.json").unlink()
    second = gpu_factory()
    assert second.app.is_loaded
    assert second.app.has_compiled_artifacts(second.compiled_path)
