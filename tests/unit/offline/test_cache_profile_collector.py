from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

from difflet.offline.cache_profile.collector import (
    build_manifests,
    run_image_sample,
    sample_matrix,
)
from difflet.pipeline.cache.profile import PhasedCandidateArm


def _arm(tmp_path: Path) -> PhasedCandidateArm:
    return PhasedCandidateArm(
        source_path=tmp_path / "candidate.json",
        candidate_id="qualified-static-plus-brake",
        policy=MappingProxyType(
            {
                "type": "phased_static_plus_brake",
                "num_steps": 4,
                "static_anchor_steps": (0, 1, 3),
                "warmup_steps": 1,
                "cooldown_steps": 1,
                "require_final_anchor": True,
                "dynamic_budget": 1,
                "invalid_measurement_fail_closed": True,
                "plastic_window": (1, 2),
                "tighten_error": 1.19,
                "recovery_error": 1.5,
                "recovery_steps": 1,
                "disable_after_recoveries": 1,
                "tighten_rule": "bisect_next_static_gap",
                "allow_acceleration": False,
            }
        ),
        order=1,
        coord="index",
        horizon_ref=MappingProxyType({"path": "horizon.json", "sha256": "a" * 64}),
        quality_contract_ref=MappingProxyType({"path": "contract.json", "sha256": "b" * 64}),
        content_sha256="c" * 64,
        file_sha256="d" * 64,
    )


def test_sample_matrix_is_prompt_major_and_rejects_duplicate_seeds():
    rows = sample_matrix(("first", "second"), (3, 7))

    assert [(row["sample_id"], row["prompt"], row["seed"]) for row in rows] == [
        ("p000-s3", "first", 3),
        ("p000-s7", "first", 7),
        ("p001-s3", "second", 3),
        ("p001-s7", "second", 7),
    ]
    with pytest.raises(ValueError, match="duplicates"):
        sample_matrix(("first",), (3, 3))


def test_image_collection_persists_only_gate_input(tmp_path):
    class FakePipeline:
        def __call__(self, **_kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), "red")])

    row = run_image_sample(
        FakePipeline(),
        sample={"sample_id": "p000-s0", "prompt": "red", "seed": 0},
        num_steps=4,
        height=8,
        width=8,
        guidance_scale=3.5,
        artifact_dir=tmp_path / "artifacts" / "baseline",
        output_root=tmp_path,
    )

    assert set(row["artifacts"]) == {"image", "image_sha256"}
    assert (tmp_path / row["artifacts"]["image"]).is_file()
    assert not tuple(tmp_path.rglob("*.pt"))


def test_manifest_builder_emits_exactly_one_candidate(tmp_path):
    samples = sample_matrix(("prompt",), (0,))
    baseline = [
        {
            "sample_id": "p000-s0",
            "elapsed_s": 4.0,
            "artifacts": {"image": "baseline.png", "image_sha256": "a" * 64},
        }
    ]
    candidate = [
        {
            "sample_id": "p000-s0",
            "elapsed_s": 2.0,
            "artifacts": {"image": "candidate.png", "image_sha256": "b" * 64},
            "runner_stats": {
                "full_steps": 3,
                "skipped_steps": 1,
                "consecutive_skip_vetoes": 0,
            },
        }
    ]

    quality, speed = build_manifests(
        identity={"num_steps": 4, "model": "flux"},
        samples=samples,
        arm=_arm(tmp_path),
        baseline_runs=baseline,
        candidate_runs=candidate,
        started_at="2026-08-08T00:00:00Z",
        completed_at="2026-08-08T00:01:00Z",
    )

    assert [row["candidate_id"] for row in quality["candidates"]] == ["qualified-static-plus-brake"]
    assert len(quality["comparisons"]) == 1
    assert speed["candidates"][0]["measured_speedup"] == pytest.approx(2.0)
    assert speed["candidates"][0]["hardware_measured"] is True
