from __future__ import annotations

import pytest

from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import load_request_validator_factory, resolve_serving_model
from difflet.serving.options import ServeOptions


def test_flux_serving_profile_uses_registry_defaults():
    resolved = resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))

    assert resolved.metadata.model_type == "flux"
    assert resolved.profile.height == 1024
    assert resolved.profile.width == 1024
    assert resolved.profile.num_frames is None
    assert resolved.profile.parallel.tp_degree == 8


def test_serving_registry_loads_request_validator_factory():
    resolved = resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))

    factory = load_request_validator_factory(resolved.metadata)

    assert factory is not None
    assert factory.__name__ == "FluxServingRequestValidator"


def test_serving_registry_exposes_stage_metadata():
    flux = resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))
    qwen = resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image"))

    assert [(stage.stage_id, stage.role) for stage in flux.metadata.stages] == [
        ("pipeline", "pipeline")
    ]
    assert [(stage.stage_id, stage.role) for stage in qwen.metadata.stages] == [
        ("text", "prompt_encoder"),
        ("generate", "denoiser"),
        ("vae", "decoder"),
    ]


def test_flux_schnell_is_not_enabled_for_p0_serving():
    with pytest.raises(ValueError, match="not enabled for P0 serving"):
        resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-schnell"))


def test_qwen_serving_rejects_cp_degree_above_one():
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", cp_degree=2))

    assert exc.value.code == "invalid_extra_body"
    assert "cp_degree=1" in exc.value.message


def test_image_serving_rejects_startup_num_frames():
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(
            ServeOptions(model_id="black-forest-labs/FLUX.1-dev", num_frames=1)
        )

    assert exc.value.code == "invalid_extra_body"
    assert "--num-frames" in exc.value.message
