from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

from difflet.offline.cache_profile.collector import (
    build_manifests,
    load_reusable_baseline,
    run_image_sample,
    sample_matrix,
)
from difflet.pipeline.cache.profile import PhasedCandidateArm


def _arm(
    tmp_path: Path,
    *,
    candidate_id: str = "qualified-static-plus-brake",
    static: bool = False,
    anchors: tuple[int, ...] = (0, 1, 3),
) -> PhasedCandidateArm:
    policy = {
        "type": "phased_static" if static else "phased_static_plus_brake",
        "num_steps": 4,
        "static_anchor_steps": anchors,
        "warmup_steps": 1,
        "cooldown_steps": 1,
        "require_final_anchor": True,
        "dynamic_budget": 0 if static else 1,
        "invalid_measurement_fail_closed": True,
    }
    if not static:
        policy.update(
            {
                "plastic_window": (1, 2),
                "tighten_error": 1.19,
                "recovery_error": 1.5,
                "recovery_steps": 1,
                "disable_after_recoveries": 1,
                "tighten_rule": "bisect_next_static_gap",
                "allow_acceleration": False,
            }
        )
    return PhasedCandidateArm(
        source_path=tmp_path / "candidate.json",
        candidate_id=candidate_id,
        policy=MappingProxyType(policy),
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


def test_trajectory_collection_persists_hash_bound_baseline_evidence(tmp_path):
    import torch

    flux_pipeline = SimpleNamespace(_tc_last_trajectory=[])

    class FakePipeline:
        app = SimpleNamespace(pipe=flux_pipeline)

        def __call__(self, **_kwargs):
            flux_pipeline._tc_last_trajectory = [
                torch.full((1, 2), float(index), dtype=torch.bfloat16) for index in range(4)
            ]
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), "blue")])

    row = run_image_sample(
        FakePipeline(),
        sample={"sample_id": "p000-s0", "prompt": "blue", "seed": 0},
        num_steps=4,
        height=8,
        width=8,
        guidance_scale=3.5,
        artifact_dir=tmp_path / "artifacts" / "baseline",
        output_root=tmp_path,
        save_trajectory=True,
    )

    artifacts = row["artifacts"]
    assert set(artifacts) == {
        "trajectory",
        "trajectory_sha256",
        "final_latent",
        "final_latent_sha256",
        "image",
        "image_sha256",
    }
    trajectory = torch.load(tmp_path / artifacts["trajectory"], weights_only=True)
    assert trajectory.shape == (4, 1, 2)
    assert torch.equal(
        torch.load(tmp_path / artifacts["final_latent"], weights_only=True),
        trajectory[-1],
    )


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


def test_manifest_builder_supports_multiple_static_frontier_candidates(tmp_path):
    samples = sample_matrix(("prompt",), (0,))
    arms = (
        _arm(tmp_path, candidate_id="static-3", static=True, anchors=(0, 1, 3)),
        _arm(tmp_path, candidate_id="static-4", static=True, anchors=(0, 1, 2, 3)),
    )
    baseline = [
        {
            "sample_id": "p000-s0",
            "elapsed_s": 4.0,
            "artifacts": {
                "trajectory": "baseline.pt",
                "trajectory_sha256": "a" * 64,
                "image": "baseline.png",
                "image_sha256": "b" * 64,
            },
        }
    ]
    candidate_runs = {
        "static-3": [
            {
                "sample_id": "p000-s0",
                "elapsed_s": 2.0,
                "artifacts": {"image": "static-3.png", "image_sha256": "c" * 64},
                "runner_stats": {
                    "full_steps": 3,
                    "skipped_steps": 1,
                    "consecutive_skip_vetoes": 0,
                },
            }
        ],
        "static-4": [
            {
                "sample_id": "p000-s0",
                "elapsed_s": 3.0,
                "artifacts": {"image": "static-4.png", "image_sha256": "d" * 64},
                "runner_stats": {
                    "full_steps": 4,
                    "skipped_steps": 0,
                    "consecutive_skip_vetoes": 0,
                },
            }
        ],
    }

    quality, speed = build_manifests(
        identity={"num_steps": 4, "model": "flux"},
        samples=samples,
        arm=arms,
        baseline_runs=baseline,
        candidate_runs=candidate_runs,
        started_at="2026-08-09T00:00:00Z",
        completed_at="2026-08-09T00:01:00Z",
    )

    assert [row["candidate_id"] for row in quality["candidates"]] == [
        "static-3",
        "static-4",
    ]
    assert len(quality["comparisons"]) == 2
    assert all(row["baseline"]["trajectory"] == "baseline.pt" for row in quality["comparisons"])
    assert [row["measured_speedup"] for row in speed["candidates"]] == pytest.approx(
        [2.0, 4.0 / 3.0]
    )


def test_reusable_baseline_is_hash_bound_and_rebased(tmp_path):
    import hashlib
    import json

    source_root = tmp_path / "first-rung"
    image_path = source_root / "artifacts" / "baseline" / "p000-s0.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"baseline")
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    protocol = {"sha256": "a" * 64}
    sample = {
        "sample_id": "p000-s0",
        "prompt_index": 0,
        "prompt": "prompt",
        "seed": 0,
    }
    quality_path = source_root / "quality-input-v2.json"
    quality_path.write_text(
        json.dumps(
            {
                "schema": "quality-input-v2",
                "hardware_measured": True,
                "protocol": protocol,
                "comparisons": [
                    {
                        **sample,
                        "candidate_id": "a10",
                        "baseline": {
                            "image": "artifacts/baseline/p000-s0.png",
                            "image_sha256": digest,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    speed_path = source_root / "speedup-candidates-v1.json"
    speed_path.write_text(
        json.dumps(
            {
                "schema": "speedup-candidates-v1",
                "hardware_measured": True,
                "protocol": protocol,
                "baseline": {
                    "total_s": 4.0,
                    "samples": [{"sample_id": "p000-s0", "elapsed_s": 4.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    next_root = tmp_path / "second-rung"

    runs = load_reusable_baseline(
        quality_path,
        speed_path,
        protocol=protocol,
        samples=(sample,),
        output_root=next_root,
    )

    assert runs[0]["elapsed_s"] == 4.0
    assert (next_root / runs[0]["artifacts"]["image"]).resolve() == image_path.resolve()
    assert runs[0]["artifacts"]["image_sha256"] == digest
