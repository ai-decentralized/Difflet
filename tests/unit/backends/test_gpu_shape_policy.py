"""Geometry refusals happen before downloads or resident worker loading."""

import subprocess
import sys

import pytest

from difflet.models.hunyuan_video.spec import SPEC


@pytest.mark.parametrize("shape", [(320, 512, 61), (32, 48, 1), (480, 832, 81)])
def test_hunyuan_valid_geometry(shape):
    from difflet.backends.gpu.policy import validate_gpu_shape

    validate_gpu_shape(SPEC, height=shape[0], width=shape[1], num_frames=shape[2])


@pytest.mark.parametrize(
    "shape", [(0, 512, 61), (321, 512, 61), (320, 510, 61), (320, 512, 60), (320, 512, 0)]
)
def test_hunyuan_invalid_geometry_is_refused(shape):
    from difflet.backends.gpu.policy import validate_gpu_shape

    with pytest.raises(ValueError):
        validate_gpu_shape(SPEC, height=shape[0], width=shape[1], num_frames=shape[2])


@pytest.mark.parametrize("override", [dict(height=321), dict(width=0), dict(num_frames=60)])
def test_cli_refuses_bad_geometry_before_resolving_checkpoint(monkeypatch, override):
    from tests.unit.cli.test_gpu_cli import _orch

    seen = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **kw: seen.append(a)
    )
    with pytest.raises((ValueError, SystemExit)):
        _orch(SPEC, **override).download()
    assert seen == []


def test_server_refuses_bad_profile_before_source_resolution(monkeypatch):
    from tests.unit.serving.test_gpu_serving import _prepare, _profile

    seen = []
    monkeypatch.setattr(
        "difflet.backends.gpu.serving.resolve_hf_model_source", lambda *a, **kw: seen.append(a)
    )
    with pytest.raises(Exception, match="multiple|divisible"):
        _prepare(SPEC, _profile(SPEC, height=33))
    assert seen == []


def test_original_hunyuan_is_a_gpu_spec_but_15_remains_separate():
    from difflet.models.spec import gpu_model_spec, gpu_spec_for_cli_model_id
    from difflet.registry import resolve_model

    assert gpu_model_spec("hunyuan_video") is SPEC
    assert gpu_spec_for_cli_model_id(SPEC.cli_model_ids[0]) is SPEC
    assert "cuda" in resolve_model(SPEC.cli_model_ids[0]).backends
    assert gpu_model_spec("hunyuan_video_15") is None
    assert (
        "cuda"
        not in resolve_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v").backends
    )


def test_hunyuan_spec_does_not_load_torch_or_trainium():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import difflet.models.hunyuan_video.spec; "
            "assert 'torch' not in sys.modules; "
            "assert 'difflet.models.hunyuan_video.application' not in sys.modules",
        ],
        check=True,
    )
