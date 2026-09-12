"""The Flux serving adapter's TPU branch: no artifacts, eager application, tpu placement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.options import _validate_probe_free_teacache
from difflet.serving.orchestrators import flux as flux_serving
from difflet.serving.orchestrators.flux import (
    FluxServingArtifactPreparer,
    FluxServingStageAdapter,
    _runtime_plan,
)
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import ResolvedModelSource, ServingProfile


def _profile(**overrides) -> ServingProfile:
    base = dict(
        model_id="black-forest-labs/FLUX.1-dev", model_type="flux", height=1024, width=1024,
        num_frames=None, parallel=DiffletParallelConfig(tp_degree=4),
    )
    base.update(overrides)
    return ServingProfile(**base)


def _source(tmp_path: Path) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot", model_id="black-forest-labs/FLUX.1-dev", requested_revision=None,
        pinned_model_path=str(tmp_path / "snap"), resolved_source_id="a" * 40,
    )


def test_runtime_plan_without_specs_is_a_tpu_stage():
    pipeline = SimpleNamespace(stages=(SimpleNamespace(stage_id="pipeline"),))
    plan = _runtime_plan(_profile(), pipeline, ())
    stage = plan.stages[0]
    assert stage.placement == "tpu" and stage.artifact_id is None
    assert plan.profile_identity == "" and plan.mode == "resident"
    assert stage.topology.tp_degree == 4 and plan.allocations[0].world_size == 4


def test_prepare_runtime_on_tpu_skips_compile_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(flux_serving, "_backend_is_tpu", lambda: True)
    monkeypatch.setattr(flux_serving, "resolve_model", lambda *a, **k: SimpleNamespace(download_patterns=None))
    monkeypatch.setattr(flux_serving, "resolve_hf_model_source", lambda *a, **k: _source(tmp_path))
    monkeypatch.setattr(flux_serving.flux_common, "build_compile_plan",
                        lambda *a, **k: pytest.fail("TPU must not build a compile plan"))
    runtime = FluxServingArtifactPreparer().prepare_runtime(
        _profile(), download_policy=DownloadPolicy.AUTO, compile_policy=CompilePolicy.NEVER,
    )
    assert runtime.compile_specs == () and runtime.artifacts.bindings == ()
    assert runtime.runtime_plan.stages[0].placement == "tpu"


def test_create_loaded_runners_on_tpu_builds_the_eager_application(monkeypatch, tmp_path):
    monkeypatch.setattr(flux_serving, "_backend_is_tpu", lambda: True)
    seen = {}

    class _App:
        def load_eager(self):
            seen["loaded"] = True

    def fake_create(**kwargs):
        seen.update(kwargs)
        return _App()

    import difflet.models.flux.entry as entry

    monkeypatch.setattr(entry, "create_flux_application", fake_create)
    profile = _profile(teacache_cadence=2)
    pipeline = SimpleNamespace(stages=(SimpleNamespace(stage_id="pipeline"),))
    runtime = SimpleNamespace(profile=profile, source=_source(tmp_path), artifacts=None,
                              pipeline_definition=pipeline)
    runners = asyncio.run(FluxServingStageAdapter().create_loaded_runners(runtime))
    assert list(runners) == ["pipeline"] and seen["loaded"]
    assert seen["backend"] == "tpu" and seen["teacache_cadence"] == 2
    assert seen["teacache_online_delta_alpha"] is None
    assert seen["shape"] == {"height": 1024, "width": 1024, "num_frames": None}
    assert seen["model_path"] == str(tmp_path / "snap")


@pytest.mark.parametrize("backend, ok", [("tpu", True), ("trainium", False)])
def test_probe_free_teacache_for_flux_is_tpu_only(monkeypatch, backend, ok):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    kwargs = dict(model_type="flux", teacache_cadence=2, teacache_online_delta=None, teacache_speedup=None)
    if ok:
        _validate_probe_free_teacache(**kwargs)
    else:
        with pytest.raises(Exception, match="implemented for"):
            _validate_probe_free_teacache(**kwargs)


def test_registry_lists_tpu_for_flux():
    from difflet.registry import resolve_model

    assert "tpu" in resolve_model("black-forest-labs/FLUX.1-dev", model_type="flux").backends
