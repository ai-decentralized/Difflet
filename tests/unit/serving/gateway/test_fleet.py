from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from difflet.serving.gateway.fleet import FleetConfigError, load_fleet
from difflet.serving.model_registry import serving_output_modality

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "serving" / "fleet.yaml"
FLUX = "black-forest-labs/FLUX.1-dev"
HV = "hunyuanvideo-community/HunyuanVideo"


def _entry(model_id=FLUX, device=0, port=8100, serve_args=("--height", "1024", "--width", "1024"),
           **extra):
    return {"model_id": model_id, "device": device, "port": port,
            "cache_dir": f"/tmp/cache/{device}", "serve_args": list(serve_args), **extra}


def _load(tmp_path, *models, gateway=None):
    path = tmp_path / "fleet.yaml"
    path.write_text(yaml.safe_dump({"gateway": gateway or {"host": "0.0.0.0", "port": 8000},
                                    "runs_dir": str(tmp_path / "runs"),
                                    "models": list(models)}))
    return load_fleet(path)


def test_load_example_fleet():
    config = load_fleet(EXAMPLE)
    assert [e.device for e in config.models] == list(range(len(config.models)))
    assert {e.kind for e in config.models} <= {"image", "video"}
    assert all(e.shapes for e in config.models)
    assert (config.host, config.port) == ("0.0.0.0", 8000)


def test_image_shape_from_height_width(tmp_path):
    config = _load(tmp_path, _entry())
    (entry,) = config.models
    assert entry.shapes == ((1024, 1024, None),)
    assert entry.kind == "image"
    assert entry.base_url == "http://127.0.0.1:8100"
    assert entry.slug == "black-forest-labs-flux-1-dev"


def test_video_shapes_from_shapes_flag(tmp_path):
    config = _load(tmp_path, _entry(HV, serve_args=["--shapes", "320x512x61,320x512x33"]))
    assert config.models[0].shapes == ((320, 512, 61), (320, 512, 33))
    assert config.video_entries() == config.models


@pytest.mark.parametrize("field", ["device", "port", "model_id", "slug"])
def test_duplicate_device_port_or_model_rejected(tmp_path, field):
    first = _entry(slug="a")
    second = _entry(HV, device=1, port=8101, slug="b",
                    serve_args=["--height", "320", "--width", "512", "--num-frames", "61"])
    second[field] = first[field]
    with pytest.raises(FleetConfigError, match=field):
        _load(tmp_path, first, second)


def test_gateway_port_collision_rejected(tmp_path):
    with pytest.raises(FleetConfigError, match="port"):
        _load(tmp_path, _entry(port=8000))


@pytest.mark.parametrize("flag", ["--host", "--port", "--model-id", "--cache-dir"])
def test_owned_flag_in_serve_args_rejected(tmp_path, flag):
    with pytest.raises(FleetConfigError, match=flag):
        _load(tmp_path, _entry(serve_args=["--height", "1024", "--width", "1024", flag, "x"]))
    with pytest.raises(FleetConfigError, match=flag):
        _load(tmp_path, _entry(serve_args=["--height", "1024", "--width", "1024", f"{flag}=x"]))


@pytest.mark.parametrize("serve_args", [[], ["--height", "1024"], ["--shapes", "1024xabc"]])
def test_missing_shape_rejected(tmp_path, serve_args):
    with pytest.raises(FleetConfigError, match="shape|--width"):
        _load(tmp_path, _entry(serve_args=serve_args))


def test_video_shape_without_frames_rejected(tmp_path):
    with pytest.raises(FleetConfigError, match="frame"):
        _load(tmp_path, _entry(HV, serve_args=["--height", "320", "--width", "512"]))


def test_unservable_model_rejected(tmp_path):
    with pytest.raises(FleetConfigError, match="HunyuanVideo-1.5"):
        _load(tmp_path, _entry("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"))


def test_entry_lookup_ignores_trailing_slash(tmp_path):
    config = _load(tmp_path, _entry("Qwen/Qwen-Image/"))
    assert config.entry("Qwen/Qwen-Image/") is config.entry("Qwen/Qwen-Image")
    assert config.entry("Qwen/Qwen-Image").model_id == "Qwen/Qwen-Image"
    assert config.entry("nope") is None


def test_slug_override_names_run_dir(tmp_path):
    config = _load(tmp_path, _entry(slug="flux"))
    (entry,) = config.models
    assert entry.slug == "flux"
    assert config.run_dir(entry) == tmp_path / "runs" / "flux"
    assert config.pid_file(entry) == tmp_path / "runs" / "flux" / "serve.pid"
    assert config.log_file(entry) == tmp_path / "runs" / "flux" / "serve.out"


@pytest.mark.parametrize("slug", ["../x", "a/b", "", "."])
def test_unsafe_slug_rejected(tmp_path, slug):
    with pytest.raises(FleetConfigError, match="slug"):
        _load(tmp_path, _entry(slug=slug))


def test_entry_env_loaded(tmp_path):
    config = _load(tmp_path, _entry(env={"HF_HUB_CACHE": "/hub", "N": 1}))
    assert dict(config.models[0].env) == {"HF_HUB_CACHE": "/hub", "N": "1"}
    assert dict(_load(tmp_path, _entry()).models[0].env) == {}


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(FleetConfigError, match="serve_arg"):
        _load(tmp_path, _entry(serve_arg=[]))


@pytest.mark.parametrize("model_id", [FLUX, "Qwen/Qwen-Image", "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                                      HV, "Lightricks/LTX-2", "MiniMaxAI/MiniMax-H3"])
def test_serving_output_modality(model_id):
    expected = "image" if model_id in (FLUX, "Qwen/Qwen-Image") else "video"
    assert serving_output_modality(model_id) == expected
    assert serving_output_modality(model_id + "/") == expected


def test_serving_output_modality_unknown():
    with pytest.raises(ValueError):
        serving_output_modality("nobody/nothing")
