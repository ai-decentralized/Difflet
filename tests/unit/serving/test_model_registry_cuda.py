"""`difflet serve` picks the CUDA Qwen-Image serving adapter by backend name."""

from __future__ import annotations

import pytest

from difflet.common.registry import qwen_image as qwen_registry
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.options import ServeOptions
from difflet.serving.types import ParallelTopology, StageRuntimeSpec

_CUDA_MODULE = "difflet.serving.orchestrators.qwen_image_cuda"


def _resolve(monkeypatch, backend, **options):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    return resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", **options))


def test_cuda_routes_qwen_to_the_cuda_factories(monkeypatch):
    md = _resolve(monkeypatch, "cuda").metadata
    assert md.artifact_preparer_factory == f"{_CUDA_MODULE}:CudaQwenImageServingArtifactPreparer"
    assert md.orchestrator_factory == f"{_CUDA_MODULE}:CudaQwenImageServingStageAdapter"
    assert md.request_validator_factory == f"{_CUDA_MODULE}:CudaQwenImageServingRequestValidator"
    assert [
        (s.stage_id, s.kind, s.role, s.final_output) for s in md.pipeline_definition.stages
    ] == [("pipeline", "opaque_pipeline", "pipeline", True)]


def test_cuda_request_defaults_match_the_cuda_cli(monkeypatch):
    # The CLI defaults --steps to 50 on CUDA and the application fills guidance 4.0.
    md = _resolve(monkeypatch, "cuda").metadata
    assert md.default_steps == 50 and md.default_guidance_scale == 4.0
    assert md.checkpoint_ids == ("Qwen/Qwen-Image",) and md.output_mime_type == "image/png"
    assert md.model_type == "qwen_image" and md.output_modality == "image"


def test_cuda_defaults_tp_degree_to_one(monkeypatch):
    # Registry default is tp=4 (Trainium); the CUDA CLI uses `args.tp_degree or 1`.
    assert _resolve(monkeypatch, "cuda").profile.parallel.tp_degree == 1
    assert _resolve(monkeypatch, "cuda", tp_degree=1).profile.parallel.tp_degree == 1


def test_explicit_tp_degree_is_kept_for_the_preparer_to_refuse(monkeypatch):
    assert _resolve(monkeypatch, "cuda", tp_degree=2).profile.parallel.tp_degree == 2


@pytest.mark.parametrize("backend", ["cpu", "trainium"])
def test_other_backends_keep_the_trainium_metadata(monkeypatch, backend):
    resolved = _resolve(monkeypatch, backend)
    assert resolved.metadata == qwen_registry.serving_metadata()
    assert resolved.profile.parallel.tp_degree == 4


def test_other_models_on_cuda_are_refused_by_name(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(ValueError) as exc:
        resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))
    message = str(exc.value)
    # Wan and Flux *generate* on CUDA; only their serving is outstanding.
    assert "Qwen-Image only" in message
    assert "not supported" in message and "not supported" in message


def test_cuda_placement_needs_allocation_and_topology_but_no_artifact():
    topology = ParallelTopology(1, 1, 1)
    StageRuntimeSpec("pipeline", "qwen-cuda", topology, None, placement="cuda")
    with pytest.raises(ValueError):
        StageRuntimeSpec("pipeline", None, None, None, placement="cuda")


def test_legacy_serving_tests_run_on_the_cpu_backend_whatever_the_host_has():
    # Review finding: resolve_serving_model now branches on current_backend(), so the existing
    # serving tests (which pin no backend) failed 97 times under DIFFLET_BACKEND=cuda. The
    # serving conftest pins cpu; CUDA tests set their own backend after it.
    from difflet.backends.registry import current_backend

    assert current_backend() == "cpu"
