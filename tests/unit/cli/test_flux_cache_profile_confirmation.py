from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.flux_cache_profile_confirmation import load_registration


ROOT = Path(__file__).resolve().parents[3]
REGISTRATION_PATH = (
    ROOT / "benchmark/flux_cache/brake-only-methodology-v1-confirmation.json"
)


def test_checked_in_brake_only_confirmation_is_frozen_and_valid():
    registration = load_registration(REGISTRATION_PATH)

    assert registration["status"] == "registered_not_collected"
    assert registration["candidate"]["policy"]["allow_acceleration"] is False
    assert registration["candidate"]["policy"]["tighten_error"] == pytest.approx(1.19)
    assert registration["candidate"]["policy"]["recovery_error"] == pytest.approx(1.5)
    assert registration["prompt_suite"]["prompt_count"] == 32
    assert registration["prompt_suite"]["seeds"] == [0]
    assert registration["novelty_audit"]["exact_overlap_count"] == 0
    assert registration["statistical_gate"]["required_failures"] == 0
    assert registration["statistical_gate"]["required_upper_bound"] == pytest.approx(
        0.0893681989862648
    )
    assert registration["confirmation_claim"]["pass_does_not_permit"] == (
        "serving qualification"
    )


def test_registration_rejects_post_freeze_threshold_edit(tmp_path):
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    registration["candidate"]["policy"]["tighten_error"] = 1.20
    tampered = tmp_path / "tampered-registration.json"
    tampered.write_text(json.dumps(registration), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 does not match"):
        load_registration(tampered)
