"""`difflet serve` picks the generic GPU serving adapter on a GPU platform."""

from __future__ import annotations

import pytest

from difflet.backends.gpu.serving import gpu_serving_metadata
from difflet.common.registry import flux, qwen_image, wan
from difflet.models.spec import gpu_model_specs
from difflet.serving import model_registry
from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.options import ServeOptions
from difflet.serving.types import ParallelTopology, StageRuntimeSpec

_QWEN = "Qwen/Qwen-Image"
_FLUX = "black-forest-labs/FLUX.1-dev"
_WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _resolve_model(monkeypatch, model_id, backend="cuda", **options):
    monkeypatch.setenv("DIFFLET_BACKEND", backend)
    return resolve_serving_model(ServeOptions(model_id=model_id, **options))


@pytest.mark.parametrize("spec", gpu_model_specs(), ids=lambda spec: spec.model_type)
def test_cuda_routes_each_model_to_the_gpu_adapter_with_the_cli_defaults(monkeypatch, spec):
    resolved = _resolve_model(monkeypatch, spec.served_checkpoint_ids[0])
    assert resolved.metadata == gpu_serving_metadata(spec)
    profile = resolved.profile
    assert profile.parallel.tp_degree == 1 and profile.host_vae is False
    assert (profile.height, profile.width, profile.num_frames, profile.output_fps) == (
        spec.default_height,
        spec.default_width,
        spec.default_num_frames,
        spec.default_fps,
    )


def test_cuda_defaults_tp_degree_to_one(monkeypatch):
    # Registry default is tp=4 (Trainium); the CUDA CLI uses `args.tp_degree or 1`.
    assert _resolve_model(monkeypatch, _QWEN).profile.parallel.tp_degree == 1
    assert _resolve_model(monkeypatch, _QWEN, tp_degree=1).profile.parallel.tp_degree == 1


def test_explicit_tp_degree_is_kept_for_the_preparer_to_refuse(monkeypatch):
    assert _resolve_model(monkeypatch, _QWEN, tp_degree=2).profile.parallel.tp_degree == 2


@pytest.mark.parametrize("backend", ["cpu", "trainium"])
@pytest.mark.parametrize("model_id,registry", [(_QWEN, qwen_image), (_FLUX, flux), (_WAN22, wan)])
def test_other_backends_keep_the_trainium_metadata(monkeypatch, backend, model_id, registry):
    resolved = _resolve_model(monkeypatch, model_id, backend=backend)
    assert resolved.metadata == registry.serving_metadata()
    assert resolved.profile.parallel.tp_degree > 1  # the registry's, not the GPU flow's 1


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
        monkeypatch.setattr(model_registry, "_visible_gpu_count", lambda platform, gpus=gpus: gpus)
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


def test_cuda_placement_needs_allocation_and_topology_but_no_artifact():
    topology = ParallelTopology(1, 1, 1)
    StageRuntimeSpec("pipeline", "qwen-cuda", topology, None, placement="cuda")
    with pytest.raises(ValueError):
        StageRuntimeSpec("pipeline", None, None, None, placement="cuda")


@pytest.mark.parametrize("model_id", [_QWEN, _FLUX, _WAN22])
def test_a_lone_teacache_calibration_is_refused_on_cuda_not_dropped(monkeypatch, model_id):
    # Without --teacache-speedup the profile builder keeps no calibration, so the file would
    # be ignored silently while the snapshot downloads; the GPU CLI refuses it by name.
    with pytest.raises(DiffletServingError) as exc:
        _resolve_model(monkeypatch, model_id, teacache_calibration="/cal.json")
    assert exc.value.code == "feature_not_supported"
    assert "--teacache-calibration" in exc.value.message
