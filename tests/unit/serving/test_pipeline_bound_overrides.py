from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.pipeline.difflet_pipeline import DiffletPipeline
from difflet.pipeline.parallel_config import DiffletParallelConfig


class _Entry:
    name = "fake"
    default_parallel = DiffletParallelConfig()
    download_patterns = None

    def require_backend(self, backend):
        return None

    def resolve_shape(self, **kwargs):
        return {"height": kwargs["height"], "width": kwargs["width"], "num_frames": None}

    def create_application(self, **kwargs):
        return SimpleNamespace(created_with=kwargs)


class _Backend:
    name = "trainium"

    def prepare_runtime(self, parallel):
        return None

    def resolve_load_rank_range(self, **kwargs):
        return None, None


def _patch_runtime(monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.resolve_model", lambda *args, **kwargs: _Entry()
    )
    monkeypatch.setattr("difflet.pipeline.difflet_pipeline.get_backend", lambda *args: _Backend())
    monkeypatch.setattr("difflet.pipeline.difflet_pipeline._default_dtype", lambda: "bfloat16")


def test_bound_pipeline_uses_exact_source_and_compiled_path(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    model_path = tmp_path / "snapshots" / ("a" * 40)
    compiled_path = tmp_path / "generation"
    model_path.mkdir(parents=True)
    compiled_path.mkdir()
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.resolve_model_path",
        lambda *args, **kwargs: pytest.fail("bound mode must not resolve model path"),
    )
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.cache_path",
        lambda *args, **kwargs: pytest.fail("bound mode must not select cache path"),
    )

    pipe = DiffletPipeline.from_pretrained(
        "org/model",
        height=64,
        width=64,
        dtype="bfloat16",
        load=False,
        skip_compile=True,
        model_path_override=str(model_path),
        resolved_source_id="a" * 40,
        compiled_path_override=str(compiled_path),
    )

    assert pipe.model_path == str(model_path.resolve())
    assert pipe.compiled_path == compiled_path.resolve()
    assert pipe.cache_spec.revision == "a" * 40
    assert pipe.app.created_with["model_path"] == str(model_path.resolve())


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_path_override": "/tmp/model"},
        {"resolved_source_id": "a" * 40},
        {"compiled_path_override": "/tmp/compiled"},
        {
            "model_path_override": "/tmp/model",
            "resolved_source_id": "a" * 40,
        },
    ],
)
def test_bound_pipeline_rejects_partial_override_tuple(monkeypatch, overrides):
    _patch_runtime(monkeypatch)

    with pytest.raises(ValueError, match="must be provided together"):
        DiffletPipeline.from_pretrained(
            "org/model",
            height=64,
            width=64,
            dtype="bfloat16",
            load=False,
            skip_compile=True,
            **overrides,
        )


def test_bound_pipeline_cache_identity_keeps_only_teacache_probe_mode(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch)
    model_path = tmp_path / "snapshots" / ("a" * 40)
    compiled_path = tmp_path / "generation"
    model_path.mkdir(parents=True)
    compiled_path.mkdir()

    first = DiffletPipeline.from_pretrained(
        "org/model",
        height=64,
        width=64,
        dtype="bfloat16",
        load=False,
        skip_compile=True,
        teacache_speedup=1.3,
        teacache_calibration=object(),
        model_path_override=str(model_path),
        resolved_source_id="a" * 40,
        compiled_path_override=str(compiled_path),
    )
    second = DiffletPipeline.from_pretrained(
        "org/model",
        height=64,
        width=64,
        dtype="bfloat16",
        load=False,
        skip_compile=True,
        teacache_speedup=1.8,
        teacache_calibration=object(),
        model_path_override=str(model_path),
        resolved_source_id="a" * 40,
        compiled_path_override=str(compiled_path),
    )

    assert first.cache_spec.application_kwargs == {"teacache_probe_enabled": True}
    assert first.cache_spec.cache_inputs() == second.cache_spec.cache_inputs()
    assert first.app.created_with["application_kwargs"]["teacache_speedup"] == 1.3
    assert second.app.created_with["application_kwargs"]["teacache_speedup"] == 1.8
