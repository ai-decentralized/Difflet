import json

import pytest
import torch

from nova.pipeline.teacache import (
    CALIBRATION_SCHEMA,
    TeaCacheCalibration,
    TeaCacheController,
    load_teacache_calibration_or_raise,
)


def _calibration(**overrides):
    data = {
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 4,
        "poly_coef": (1.0, 2.0, 3.0),
        "threshold": 10.0,
        "warmup_steps": 1,
        "cooldown_steps": 1,
        "target_speedup": 1.5,
        "fit_r2": 0.95,
        "n_samples": 32,
        "mod_input_source": "block0_modulated_input",
    }
    data.update(overrides)
    return TeaCacheCalibration(**data)


def test_teacache_calibration_roundtrips_and_predicts_delta(tmp_path):
    calibration = _calibration()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

    loaded = TeaCacheCalibration.from_json(path)

    assert loaded.to_dict()["schema"] == CALIBRATION_SCHEMA
    assert loaded.poly_coef == (1.0, 2.0, 3.0)
    assert loaded.predict_delta(2.0) == pytest.approx(17.0)
    assert loaded.n_samples == 32
    assert loaded.mod_input_source == "block0_modulated_input"


def test_teacache_controller_protects_warmup_cooldown_and_advances_skip_state():
    calibration = _calibration(poly_coef=(0.0,), threshold=1.0)
    controller = TeaCacheController(calibration)
    mod = torch.zeros((1, 2), dtype=torch.float32)

    assert controller.should_skip(0, mod) is False
    controller.record_full_step(torch.ones((1, 2), dtype=torch.float32), mod)

    assert controller.should_skip(1, mod) is False
    controller.record_full_step(torch.ones((1, 2), dtype=torch.float32) * 3.0, mod)

    assert controller.should_skip(2, mod + 0.1) is True
    assert torch.allclose(controller.skip_noise_pred(mod + 0.1), torch.full((1, 2), 5.0))
    assert controller.should_skip(3, mod + 0.2) is False
    assert controller.stats()["full_steps"] == 2
    assert controller.stats()["skipped_steps"] == 1


def test_teacache_loader_fails_fast_without_calibration():
    with pytest.raises(FileNotFoundError, match="scripts/calibrate_teacache.py"):
        load_teacache_calibration_or_raise(
            None,
            model="hunyuan_video",
            shape_label="320x512x61",
        )


def test_teacache_loader_rejects_model_and_shape_mismatch(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_calibration(model="flux").to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="model mismatch"):
        load_teacache_calibration_or_raise(
            path,
            model="hunyuan_video",
            shape_label="320x512x61",
        )

    path.write_text(json.dumps(_calibration(shape_label="1x1x1").to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_teacache_calibration_or_raise(
            path,
            model="hunyuan_video",
            shape_label="320x512x61",
        )


def test_should_skip_accepts_precomputed_diff_norm():
    """cclog 72: when the probe NEFF supplies a precomputed delta scalar,
    the controller must use it instead of doing its own host-side diff."""
    import torch
    from nova.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    calibration = TeaCacheCalibration(
        model="test",
        shape_label="test_shape",
        num_steps=10,
        poly_coef=(0.0, 1.0),
        threshold=0.5,
        warmup_steps=0,
        cooldown_steps=0,
    )
    controller = TeaCacheController(calibration)
    # Seed the controller with non-None prev state so should_skip can return True.
    controller.prev_mod_input = torch.zeros(3)
    controller.prev_noise_pred = torch.zeros(3)
    controller.cached_residual = torch.zeros(3)

    # Precomputed delta below threshold → skip.
    assert controller.should_skip(step_index=5, mod_input_now=torch.zeros(3), diff_norm=0.1) is True
    assert controller.last_delta_estimate == 0.1

    # Precomputed delta above threshold → don't skip.
    assert controller.should_skip(step_index=5, mod_input_now=torch.zeros(3), diff_norm=0.9) is False
    assert controller.last_delta_estimate == 0.9

    # When diff_norm=None, fall back to host-side computation.
    controller.prev_mod_input = torch.zeros(3)
    assert controller.should_skip(step_index=5, mod_input_now=torch.ones(3)) is False  # ||1-0||=sqrt(3)>0.5
    assert controller.last_delta_estimate > 0.5
