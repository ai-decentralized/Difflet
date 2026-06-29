import json

import pytest
import torch

from scripts.audit_m9_gate import audit
from scripts.calibrate_teacache import (
    FOREGROUND_ACK,
    _collect_hunyuan_video_pairs,
    _fit_poly,
    _poly_design,
    _r2_score,
    _require_hv_teacache_mod_input,
)
from scripts.verify_teacache_speedup import (
    FOREGROUND_ACK as T0B_FOREGROUND_ACK,
    _collect_hunyuan_video_candidates,
    verify,
)


def test_calibrate_teacache_polynomial_fit_matches_holdout():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    y = 1.0 + 2.0 * x

    coef = _fit_poly(x, y, degree=1)
    pred = _poly_design(x, degree=1).matmul(coef)

    assert coef.tolist() == pytest.approx([1.0, 2.0])
    assert _r2_score(y, pred) == pytest.approx(1.0)


def test_calibrate_teacache_collection_requires_explicit_hardware_ack():
    class Args:
        allow_hardware = False
        foreground_ack = FOREGROUND_ACK
        source_dir = "source"
        compiled_dir = "compiled"
        bundle = ["bundle.safetensors"]
        holdout_bundle = []

    with pytest.raises(RuntimeError, match="--allow-hardware"):
        _collect_hunyuan_video_pairs(Args())


def test_calibrate_teacache_preflight_rejects_missing_trainium_mod_input_hook():
    class App:
        transformer = object()

    with pytest.raises(RuntimeError, match="does not expose teacache_mod_input"):
        _require_hv_teacache_mod_input(App())


def test_verify_teacache_speedup_marks_gate_crossing_candidate():
    result = verify(
        {
            "schema": "difflet-m9-teacache-speedup-candidates-v1",
            "model": "hunyuan_video",
            "shape_label": "320x512x61",
            "num_steps": 50,
            "candidates": [
                {
                    "target_speedup": 1.5,
                    "measured_speedup": 1.51,
                    "trajectory_cosine": 0.99991,
                    "final_cosine": 0.9996,
                    "hardware_measured": True,
                }
            ],
        },
        min_speedup=1.5,
        min_trajectory_cosine=0.9999,
        min_final_cosine=0.9995,
    )

    assert result["can_unlock_t1"] is True
    assert result["candidates"][0]["passes_gate"] is True


def test_verify_teacache_collection_requires_explicit_hardware_ack():
    class Args:
        allow_hardware = False
        foreground_ack = T0B_FOREGROUND_ACK
        source_dir = "source"
        compiled_dir = "compiled"
        bundle = ["bundle.safetensors"]
        candidate_calibration = ["1.5:calibration.json"]

    with pytest.raises(RuntimeError, match="--allow-hardware"):
        _collect_hunyuan_video_candidates(Args())


def test_audit_m9_rejects_hidden_state_proxy_calibration(tmp_path):
    calibration = {
        "schema": "difflet-m9-teacache-calibration-v1",
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 50,
        "poly_coef": [0.0, 1.0],
        "threshold": 1.0,
        "fit_r2": 0.99,
        "n_samples": 400,
        "mod_input_source": "hidden_states_proxy",
        "hardware_measured": True,
    }
    speedup = {
        "schema": "difflet-m9-teacache-speedup-curve-v1",
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 50,
        "candidates": [
            {
                "target_speedup": 1.5,
                "measured_speedup": 1.6,
                "trajectory_cosine": 0.99995,
                "final_cosine": 0.9996,
                "hardware_measured": True,
            }
        ],
    }
    calibration_path = tmp_path / "calibration.json"
    speedup_path = tmp_path / "speedup.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    speedup_path.write_text(json.dumps(speedup), encoding="utf-8")

    result = audit(
        calibration_paths=[calibration_path],
        speedup_paths=[speedup_path],
        integration_paths=[],
        required_labels=["hunyuan_video"],
    )

    assert result["can_unlock_t1"] is False
    assert result["decision"] == "remain_gated_partial_evidence"
    assert "unsupported_mod_input_source:hidden_states_proxy" in result["rows"][0][
        "invalid_reasons"
    ]


def test_audit_m9_requires_calibration_speedup_and_integration_to_close(tmp_path):
    calibration = {
        "schema": "difflet-m9-teacache-calibration-v1",
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 50,
        "poly_coef": [0.0, 1.0],
        "threshold": 1.0,
        "fit_r2": 0.99,
        "n_samples": 400,
        "mod_input_source": "block0_modulated_input",
        "hardware_measured": True,
    }
    speedup = {
        "schema": "difflet-m9-teacache-speedup-curve-v1",
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 50,
        "candidates": [
            {
                "target_speedup": 1.5,
                "measured_speedup": 1.6,
                "trajectory_cosine": 0.99995,
                "final_cosine": 0.9996,
                "hardware_measured": True,
            }
        ],
    }
    integration = {
        "schema": "difflet-m9-teacache-integration-v1",
        "model": "hunyuan_video",
        "wallclock_speedup": 1.55,
        "trajectory_cosine": 0.99995,
        "final_cosine": 0.9996,
        "default_off": True,
        "hardware_measured": True,
    }
    paths = {}
    for name, doc in (
        ("calibration", calibration),
        ("speedup", speedup),
        ("integration", integration),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        paths[name] = path

    result = audit(
        calibration_paths=[paths["calibration"]],
        speedup_paths=[paths["speedup"]],
        integration_paths=[paths["integration"]],
        required_labels=["hunyuan_video"],
    )

    assert result["can_unlock_t1"] is True
    assert result["can_close_t1"] is True
    assert result["decision"] == "close_t1"
