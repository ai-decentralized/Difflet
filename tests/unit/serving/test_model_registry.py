from __future__ import annotations

import json

import pytest

from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import load_request_validator_factory, resolve_serving_model
from difflet.serving.options import ServeOptions


@pytest.fixture(autouse=True)
def _ambient_trainium(monkeypatch):
    """resolve_serving_model gates on the current backend; the tests below
    describe Trainium profiles, and a torch_xla venv would otherwise
    auto-detect tpu and reject FLUX before the assertion under test. The
    TPU tests set the variable themselves."""
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")


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


def test_serving_registry_exposes_pipeline_definition():
    flux = resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))
    qwen = resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image"))

    assert [
        (stage.stage_id, stage.kind, stage.role)
        for stage in flux.metadata.pipeline_definition.stages
    ] == [("pipeline", "opaque_pipeline", "pipeline")]
    assert [
        (stage.stage_id, stage.kind, stage.role)
        for stage in qwen.metadata.pipeline_definition.stages
    ] == [
        ("text", "extracted", "prompt_encoder"),
        ("generate", "extracted", "denoiser"),
        ("vae", "extracted", "decoder"),
    ]


def test_hunyuan_placement_defaults_to_neuron_vae_with_host_clip():
    resolved = resolve_serving_model(ServeOptions(model_id="hunyuanvideo-community/HunyuanVideo"))

    assert resolved.profile.clip_placement == "host"
    assert resolved.profile.vae_placement == "neuron"
    assert resolved.profile.host_vae is False


def test_hunyuan_explicit_neuron_clip_preserves_default_neuron_vae():
    resolved = resolve_serving_model(
        ServeOptions(
            model_id="hunyuanvideo-community/HunyuanVideo",
            clip_placement="neuron",
        )
    )

    assert resolved.profile.clip_placement == "neuron"
    assert resolved.profile.vae_placement == "neuron"
    assert resolved.profile.host_vae is False


def test_wan_defaults_to_neuron_vae():
    resolved = resolve_serving_model(
        ServeOptions(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    )

    assert resolved.profile.vae_placement == "neuron"
    assert resolved.profile.host_vae is False


def test_ltx_keeps_required_host_vae_default():
    resolved = resolve_serving_model(ServeOptions(model_id="Lightricks/LTX-2"))

    assert resolved.profile.vae_placement == "host"
    assert resolved.profile.host_vae is True


def test_host_vae_preserves_host_placement():
    resolved = resolve_serving_model(
        ServeOptions(
            model_id="hunyuanvideo-community/HunyuanVideo",
            host_vae=True,
        )
    )

    assert resolved.profile.vae_placement == "host"

    wan = resolve_serving_model(
        ServeOptions(
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            host_vae=True,
        )
    )
    assert wan.profile.vae_placement == "host"


def test_image_and_non_hunyuan_models_reject_placement_overrides():
    with pytest.raises(DiffletServingError) as image_exc:
        resolve_serving_model(
            ServeOptions(
                model_id="Qwen/Qwen-Image",
                host_vae=True,
            )
        )
    assert image_exc.value.code == "invalid_extra_body"

    with pytest.raises(DiffletServingError) as clip_exc:
        resolve_serving_model(
            ServeOptions(
                model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                clip_placement="neuron",
            )
        )
    assert clip_exc.value.code == "invalid_extra_body"

def test_flux_schnell_is_not_enabled_for_p0_serving():
    with pytest.raises(ValueError, match="not enabled for P0 serving"):
        resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-schnell"))


def test_qwen_serving_rejects_cp_degree_above_one():
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", cp_degree=2))

    assert exc.value.code == "invalid_extra_body"
    assert "cp_degree=1" in exc.value.message


def test_qwen_serving_accepts_rectangular_profile():
    resolved = resolve_serving_model(
        ServeOptions(model_id="Qwen/Qwen-Image", height=512, width=1024)
    )

    assert resolved.profile.height == 512
    assert resolved.profile.width == 1024


def test_image_serving_rejects_startup_num_frames():
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev", num_frames=1))

    assert exc.value.code == "invalid_extra_body"
    assert "--num-frames" in exc.value.message


def test_flux_serving_profile_accepts_sequence_parallelism():
    resolved = resolve_serving_model(
        ServeOptions(model_id="black-forest-labs/FLUX.1-dev", sp_enabled=True)
    )

    assert resolved.profile.parallel.sp_enabled is True


def test_qwen_serving_profile_accepts_sequence_parallelism():
    resolved = resolve_serving_model(
        ServeOptions(model_id="Qwen/Qwen-Image", sp_enabled=True)
    )

    assert resolved.profile.parallel.sp_enabled is True


def test_serving_profile_rejects_cfg_parallel_without_true_cfg_contract():
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(
            ServeOptions(model_id="black-forest-labs/FLUX.1-dev", cfg_parallel=True)
        )

    assert exc.value.code == "invalid_extra_body"
    assert "--cfg-parallel" in exc.value.message


def test_flux_serving_profile_carries_frozen_adaptive_teacache(tmp_path):
    calibration_path = tmp_path / "flux-calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema": "difflet-m9-teacache-calibration-v1",
                "model": "flux",
                "shape_label": "1024x1024",
                "num_steps": 50,
                "poly_coef": [0.0, 1.0],
                "threshold": 0.1,
                "target_speedup": 1.5,
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_serving_model(
        ServeOptions(
            model_id="black-forest-labs/FLUX.1-dev",
            teacache_speedup=1.5,
            teacache_calibration=str(calibration_path),
        )
    )

    assert resolved.profile.teacache_speedup == 1.5
    assert resolved.profile.teacache_calibration is None
    assert resolved.profile.teacache_calibration_data.model == "flux"
    assert resolved.profile.teacache_calibration_data.shape_label == "1024x1024"

    calibration_path.unlink()
    assert resolved.profile.teacache_calibration_data.target_speedup == 1.5


@pytest.mark.parametrize(
    ("model_id", "options", "cadence", "alpha"),
    [
        ("Qwen/Qwen-Image", {"teacache_cadence": 2}, 2, None),
        ("Qwen/Qwen-Image", {"teacache_online_delta": 0.6}, None, 0.6),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", {"teacache_cadence": 3}, 3, None),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", {"teacache_online_delta": 0.5}, None, 0.5),
    ],
)
def test_serving_profile_carries_probe_free_teacache_modes(model_id, options, cadence, alpha):
    """Fixed cadence / online-delta are host-side controller state in the Qwen
    and Wan pipelines (no probe graph, no calibration), so the profile carries
    them through to the adapter — on Trainium and on the eager TPU backend."""
    resolved = resolve_serving_model(ServeOptions(model_id=model_id, **options))

    assert resolved.profile.teacache_cadence == cadence
    assert resolved.profile.teacache_online_delta == alpha
    # Probe-free never means adaptive: no frozen calibration is synthesized.
    assert resolved.profile.teacache_speedup is None
    assert resolved.profile.teacache_calibration_data is None


@pytest.mark.parametrize(
    "model_id",
    ["black-forest-labs/FLUX.1-dev"],
)
def test_serving_profile_rejects_probe_free_teacache_for_unwired_adapters(model_id):
    """Adapters that do not wire the probe-free controller fail at option
    resolution, not after a worker has loaded the weights."""
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id=model_id, teacache_cadence=2))

    assert exc.value.code == "invalid_extra_body"
    assert "--teacache-cadence" in exc.value.message
    assert "hunyuan_video, ltx_2, qwen_image, wan" in exc.value.message


def test_serving_profile_carries_probe_free_teacache_for_hunyuan_on_tpu(monkeypatch):
    """The option layer lets the modes through for HunyuanVideo; the adapter
    then accepts them on TPU and rejects them on Trainium."""
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    resolved = resolve_serving_model(
        ServeOptions(model_id="hunyuanvideo-community/HunyuanVideo", teacache_cadence=2)
    )
    assert resolved.profile.teacache_cadence == 2


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"teacache_cadence": 2, "teacache_online_delta": 0.6}, "mutually exclusive"),
        (
            {
                "teacache_cadence": 2,
                "teacache_speedup": 1.5,
                "teacache_calibration": "/unused.json",
            },
            "mutually exclusive",
        ),
        ({"teacache_cadence": 1}, ">= 2"),
        ({"teacache_cadence": 0}, ">= 2"),
        ({"teacache_cadence": True}, ">= 2"),
        ({"teacache_online_delta": 0.0}, "finite positive"),
        ({"teacache_online_delta": -0.5}, "finite positive"),
        ({"teacache_online_delta": float("inf")}, "finite positive"),
    ],
)
def test_serving_profile_rejects_invalid_probe_free_teacache(options, match):
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(ServeOptions(model_id="Qwen/Qwen-Image", **options))

    assert exc.value.code == "invalid_extra_body"
    assert match in exc.value.message


def test_serving_profile_ignores_calibration_without_speedup():
    resolved = resolve_serving_model(
        ServeOptions(
            model_id="Qwen/Qwen-Image",
            teacache_calibration="/path/is/not/read.json",
        )
    )

    assert resolved.profile.teacache_speedup is None
    assert resolved.profile.teacache_calibration is None
    assert resolved.profile.teacache_calibration_data is None


def test_serving_profile_validates_calibration_when_teacache_is_enabled(tmp_path):
    calibration_path = tmp_path / "invalid.json"
    calibration_path.write_text("{}", encoding="utf-8")

    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(
            ServeOptions(
                model_id="Qwen/Qwen-Image",
                teacache_speedup=1.5,
                teacache_calibration=str(calibration_path),
            )
        )

    assert exc.value.code == "invalid_extra_body"
    assert "invalid TeaCache calibration" in exc.value.message


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"cadence": 2},
        {"online_delta_alpha": 0.1},
        {"poly_coef": [0.0, float("nan")]},
        {"poly_coef": [0.0, 10**400]},
        {"num_steps": 10},
        {"skip_run_length": 0},
        {"accumulate": "false"},
        {"mod_input_source": "hidden_states_proxy"},
    ],
)
def test_serving_profile_rejects_semantically_invalid_calibration(tmp_path, invalid_field):
    calibration = {
        "schema": "difflet-m9-teacache-calibration-v1",
        "model": "flux",
        "shape_label": "1024x1024",
        "num_steps": 50,
        "poly_coef": [0.0, 1.0],
        "threshold": 0.1,
        "target_speedup": 1.5,
    }
    calibration.update(invalid_field)
    calibration_path = tmp_path / "invalid-calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(
            ServeOptions(
                model_id="black-forest-labs/FLUX.1-dev",
                teacache_speedup=1.5,
                teacache_calibration=str(calibration_path),
            )
        )

    assert exc.value.code == "invalid_extra_body"
    assert "invalid TeaCache calibration" in exc.value.message


@pytest.mark.parametrize(
    "options,message",
    [
        (ServeOptions(model_id="black-forest-labs/FLUX.1-dev", tp_degree=0), "tp_degree"),
        (
            ServeOptions(
                model_id="black-forest-labs/FLUX.1-dev",
                cp_degree=1,
                cp_mode="ring",
            ),
            "requires cp_degree > 1",
        ),
    ],
)
def test_serving_profile_reports_invalid_parallel_values(options, message):
    with pytest.raises(DiffletServingError) as exc:
        resolve_serving_model(options)

    assert exc.value.code == "invalid_extra_body"
    assert message in exc.value.message


@pytest.mark.parametrize(
    "model_id",
    ["black-forest-labs/FLUX.1-dev"],
)
def test_serving_rejects_unported_models_on_tpu_before_touching_weights(monkeypatch, model_id):
    """Seen on a v5e: `difflet serve` for HunyuanVideo resolved the snapshot and
    then died in _build_llama_app with ModuleNotFoundError('neuronx_distributed_inference');
    FLUX got as far as a gated-repo 401. The registry's backend list must gate
    serving the way it gates DiffletPipeline."""
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")

    with pytest.raises(ValueError, match="does not support backend 'tpu'"):
        resolve_serving_model(ServeOptions(model_id=model_id))


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen-Image",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "hunyuanvideo-community/HunyuanVideo",
        "Lightricks/LTX-2",
    ],
)
def test_serving_accepts_the_tpu_ported_models(monkeypatch, model_id):
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")

    assert resolve_serving_model(ServeOptions(model_id=model_id)).model_id == model_id
