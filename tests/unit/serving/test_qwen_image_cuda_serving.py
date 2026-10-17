"""Qwen-Image serving on CUDA: preparer, validator and resident adapter, CPU-only.

DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.resident_worker import _apply_worker_runtime_environment
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import qwen_image_cuda as mod
from difflet.serving.types import ResolvedModelSource, ServingProfile

_SOURCE = ResolvedModelSource("hf_snapshot", "Qwen/Qwen-Image", None, "/snap/qwen", "a" * 40)


def _profile(**over) -> ServingProfile:
    d = dict(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=1),
        cache_dir="/c",
    )
    d.update(over)
    return ServingProfile(**d)


@pytest.fixture
def source(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mod, "resolve_hf_model_source", lambda *a, **k: calls.append((a, k)) or _SOURCE
    )
    return calls


def _prepare(profile, source_calls, policy=CompilePolicy.AUTO):
    return mod.CudaQwenImageServingArtifactPreparer().prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy
    )


# ------------------------------------------------------------------ preparer


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(source):
    bundle = _prepare(_profile(), source)
    assert source and source[0][0] == ("Qwen/Qwen-Image",)
    assert source[0][1] == {"revision": None, "download_policy": DownloadPolicy.AUTO}
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    stage = bundle.runtime_plan.stages[0]
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.profile_identity
    assert bundle.adapter_config["compile_policy"] is CompilePolicy.AUTO
    assert bundle.source is _SOURCE and bundle.profile is not None


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"shapes": ((1024, 1024, None), (768, 768, None))}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)}, "not supported"),
        ({"teacache_speedup": 1.5}, "not supported"),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(source, over, expected):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(_profile(**over), source)
    assert exc.value.code == "feature_not_supported" and expected in exc.value.message
    assert source == []


def test_runtime_plan_is_accepted_by_the_resident_worker_environment(source, monkeypatch):
    for name in ("NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES", "WORLD_SIZE", "RANK"):
        monkeypatch.delenv(name, raising=False)
    bundle = _prepare(_profile(), source)
    _apply_worker_runtime_environment(bundle)  # raises ValueError on a plan the worker rejects


def test_profile_identity_tracks_shape_and_source(source):
    a = _prepare(_profile(), source).runtime_plan.profile_identity
    b = _prepare(_profile(height=768, width=768), source).runtime_plan.profile_identity
    assert a != b and a == _prepare(_profile(), source).runtime_plan.profile_identity
