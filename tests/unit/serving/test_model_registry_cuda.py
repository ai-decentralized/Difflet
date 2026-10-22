"""`difflet serve` picks the generic GPU serving adapter on a GPU platform."""

from __future__ import annotations

import pytest

from difflet.common.registry import qwen_image as qwen_registry
from difflet.serving import model_registry
from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.options import ServeOptions
from difflet.serving.types import ParallelTopology, StageRuntimeSpec

_GPU_MODULE = "difflet.backends.gpu.serving"


def _resolve(monkeypatch, backend, **options):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    return resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", **options))


def _assert_gpu_factories(md):
    assert md.artifact_preparer_factory == f"{_GPU_MODULE}:GpuServingArtifactPreparer"
    assert md.orchestrator_factory == f"{_GPU_MODULE}:GpuServingStageAdapter"
    assert md.request_validator_factory == f"{_GPU_MODULE}:GpuServingRequestValidator"


def test_cuda_routes_qwen_to_the_gpu_factories(monkeypatch):
    md = _resolve(monkeypatch, "cuda").metadata
    _assert_gpu_factories(md)
    assert [
        (s.stage_id, s.kind, s.role, s.final_output) for s in md.pipeline_definition.stages
    ] == [("pipeline", "opaque_pipeline", "pipeline", True)]


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


_FLUX = "black-forest-labs/FLUX.1-dev"
_WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _resolve_model(monkeypatch, model_id, backend="cuda", **options):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    return resolve_serving_model(ServeOptions(model_id=model_id, **options))


def test_cuda_routes_flux_to_the_gpu_adapter_with_the_cli_defaults(monkeypatch):
    resolved = _resolve_model(monkeypatch, _FLUX)
    md = resolved.metadata
    _assert_gpu_factories(md)
    assert (md.default_steps, md.default_guidance_scale) == (28, 3.5)
    profile = resolved.profile
    assert profile.parallel.tp_degree == 1 and (profile.height, profile.width) == (1024, 1024)


def test_cuda_routes_wan22_to_the_gpu_adapter_with_the_cli_defaults(monkeypatch):
    resolved = _resolve_model(monkeypatch, _WAN22)
    md = resolved.metadata
    _assert_gpu_factories(md)
    assert (md.default_steps, md.default_guidance_scale) == (40, 4.0)
    assert md.output_modality == "video" and md.output_mime_type == "video/mp4"
    profile = resolved.profile
    assert profile.parallel.tp_degree == 1
    assert (profile.height, profile.width, profile.num_frames) == (480, 832, 9)
    assert profile.output_fps == 16 and profile.host_vae is False


def test_wan21_is_not_served_on_cuda_because_it_never_ran_on_a_gpu(monkeypatch):
    # A DiffletServingError, so `difflet serve` exits with the message, not a traceback.
    with pytest.raises(DiffletServingError) as exc:
        _resolve_model(monkeypatch, _WAN21)
    assert exc.value.code == "feature_not_supported"
    message = exc.value.message
    assert _WAN21 in message and "CUDA" in message and _WAN22 in message
    assert "not served on the CUDA backend yet" in message
    # On Trainium the Wan serving metadata keeps both checkpoints.
    assert _resolve_model(monkeypatch, _WAN21, backend="trainium").metadata.checkpoint_ids == (
        _WAN21,
        _WAN22,
    )


def test_video_serving_on_cuda_refuses_replicas_and_image_serving_keeps_them(monkeypatch):
    # The video reason wins over the GPU count: on a one-GPU host the user learns why Wan
    # cannot use replicas at all, not that the host is short of GPUs.
    for gpus in (1, 2):
        monkeypatch.setattr(
            model_registry, "_visible_gpu_count", lambda platform, gpus=gpus: gpus
        )
        with pytest.raises(DiffletServingError) as exc:
            _resolve_model(monkeypatch, _WAN22, replicas=2)
        assert exc.value.code == "feature_not_supported"
        assert "one video job at a time" in exc.value.message
    monkeypatch.setattr(model_registry, "_visible_gpu_count", lambda platform: 2)
    assert _resolve_model(monkeypatch, _FLUX, replicas=2).metadata.model_type == "flux"


def test_models_without_a_gpu_spec_name_the_three_that_have_one(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id="hunyuanvideo-community/HunyuanVideo"))
    assert exc.value.code == "feature_not_supported"
    message = exc.value.message
    assert "black-forest-labs/FLUX.1-dev, Wan-AI/Wan2.2-T2V-A14B-Diffusers" in message
    assert "trainium" in message


@pytest.mark.parametrize("model_id", [_FLUX, _WAN22])
def test_flux_and_wan_keep_the_trainium_metadata_off_cuda(monkeypatch, model_id):
    from difflet.common.registry import flux, wan

    expected = {_FLUX: flux, _WAN22: wan}[model_id].serving_metadata()
    assert _resolve_model(monkeypatch, model_id, backend="trainium").metadata == expected


def test_cuda_placement_needs_allocation_and_topology_but_no_artifact():
    topology = ParallelTopology(1, 1, 1)
    StageRuntimeSpec("pipeline", "qwen-cuda", topology, None, placement="cuda")
    with pytest.raises(ValueError):
        StageRuntimeSpec("pipeline", None, None, None, placement="cuda")


@pytest.mark.parametrize("model_id", ["Qwen/Qwen-Image", _FLUX, _WAN22])
def test_a_lone_teacache_calibration_is_refused_on_cuda_not_dropped(monkeypatch, model_id):
    # Without --teacache-speedup the profile builder keeps no calibration, so the file would
    # be ignored silently while the snapshot downloads; the GPU CLI refuses it by name.
    with pytest.raises(DiffletServingError) as exc:
        _resolve_model(monkeypatch, model_id, teacache_calibration="/cal.json")
    assert exc.value.code == "feature_not_supported"
    assert "--teacache-calibration" in exc.value.message
