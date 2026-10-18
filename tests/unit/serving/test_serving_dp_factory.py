"""`--dp-degree N`: refused before any download off CUDA or above the GPU count; on CUDA the
factory builds one resident engine per GPU behind a ReplicatedServingEngine (plan F2b)."""

from __future__ import annotations

import pytest

from difflet.serving import factory as factory_mod
from difflet.serving.engines.replicated import ReplicatedServingEngine
from difflet.serving.engines.resident_worker import ResidentWorkerServingEngine
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import ServeOptions
from difflet.serving.orchestrators import qwen_image_cuda as cuda_mod
from difflet.serving.orchestrators.base import NoopServingRequestValidator
from difflet.serving.types import ResolvedModelSource

_SOURCE = ResolvedModelSource("hf_snapshot", "Qwen/Qwen-Image", None, "/snap/qwen", "a" * 40)


@pytest.fixture
def no_download(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cuda_mod, "resolve_hf_model_source", lambda *a, **k: calls.append((a, k)) or _SOURCE
    )
    monkeypatch.setattr(
        factory_mod,
        "load_request_validator_factory",
        lambda metadata: (lambda runtime: NoopServingRequestValidator(runtime)),
    )
    return calls


def _options(**over) -> ServeOptions:
    d = dict(model_id="Qwen/Qwen-Image", cache_dir="/c")
    d.update(over)
    return ServeOptions(**d)


def test_replicas_must_be_positive():
    with pytest.raises(ValueError):
        _options(replicas=0)


def test_dp_degree_is_refused_off_cuda_before_the_download(no_download):
    # conftest pins DIFFLET_BACKEND=cpu
    with pytest.raises(DiffletServingError) as exc:
        factory_mod.build_serving_stack(_options(replicas=2))
    assert exc.value.code == "feature_not_supported" and "F2b" in exc.value.message
    assert no_download == []


def test_dp_degree_above_the_gpu_count_is_refused_before_the_download(monkeypatch, no_download):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setattr("difflet.serving.model_registry._visible_gpu_count", lambda: 1)
    with pytest.raises(DiffletServingError) as exc:
        factory_mod.build_serving_stack(_options(replicas=2))
    assert exc.value.code == "feature_not_supported"
    assert "--dp-degree 2" in exc.value.message and "1 visible GPU" in exc.value.message
    assert no_download == []


def test_dp_degree_two_builds_two_replicas_one_gpu_each(monkeypatch, no_download):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setattr("difflet.serving.model_registry._visible_gpu_count", lambda: 2)
    stack = factory_mod.build_serving_stack(_options(replicas=2))
    engine = stack.engine
    assert isinstance(engine, ReplicatedServingEngine)
    assert [type(r) for r in engine.replicas] == [ResidentWorkerServingEngine] * 2
    assert [r.runtime.runtime_plan.environment.available_core_ids for r in engine.replicas] == [
        (0,),
        (1,),
    ]
    # Every replica is the same model at the same profile; one snapshot resolution.
    assert len({r.runtime.runtime_plan.profile_identity for r in engine.replicas}) == 1
    assert len(no_download) == 1
    assert stack.runtime is engine.replicas[0].runtime


def test_dp_degree_one_is_the_plain_resident_engine(monkeypatch, no_download):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    stack = factory_mod.build_serving_stack(_options())
    assert type(stack.engine) is ResidentWorkerServingEngine
