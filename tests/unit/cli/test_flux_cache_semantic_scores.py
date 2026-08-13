from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.evaluate_flux_cache_semantics as semantics


def _write_quality_manifest(
    root: Path,
    *,
    split: str,
    candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b"),
) -> Path:
    artifact_root = root / "artifacts"
    baseline_path = artifact_root / "baseline" / "p000-s0.png"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(f"{split}-baseline".encode())
    comparisons = []
    for candidate_id in candidate_ids:
        candidate_path = artifact_root / candidate_id / "p000-s0.png"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_bytes(f"{split}-{candidate_id}".encode())
        comparisons.append(
            {
                "sample_id": "p000-s0",
                "prompt_index": 0,
                "seed": 0,
                "prompt": f"{split} prompt",
                "candidate_id": candidate_id,
                "baseline": {"image": baseline_path.relative_to(root).as_posix()},
                "candidate": {"image": candidate_path.relative_to(root).as_posix()},
            }
        )
    path = root / "quality-input.json"
    path.write_text(
        json.dumps(
            {
                "protocol": {"prompt_selection": {"split": split}},
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_semantic_collector_deduplicates_shared_baseline(tmp_path):
    manifest = _write_quality_manifest(tmp_path, split="calibration")

    images, sources = semantics.collect_unique_images((manifest,))

    assert len(images) == 3
    assert sum(row["role"] == "baseline" for row in images) == 1
    assert sum(row["role"] == "candidate" for row in images) == 2
    assert sources[0]["split"] == "calibration"
    assert len(sources[0]["sha256"]) == 64


def test_semantic_collector_rejects_duplicate_split(tmp_path):
    first = _write_quality_manifest(tmp_path / "first", split="calibration")
    second = _write_quality_manifest(tmp_path / "second", split="calibration")

    with pytest.raises(ValueError, match="split is duplicated"):
        semantics.collect_unique_images((first, second))


def test_semantic_collector_rejects_conflicting_shared_identity(tmp_path):
    manifest = _write_quality_manifest(tmp_path, split="calibration")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["comparisons"][1]["prompt"] = "different prompt"
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting semantic image identity"):
        semantics.collect_unique_images((manifest,))


def test_semantic_evaluation_pairs_scores_and_resumes(tmp_path, monkeypatch):
    calibration = _write_quality_manifest(
        tmp_path / "calibration",
        split="calibration",
    )
    holdout = _write_quality_manifest(
        tmp_path / "holdout",
        split="holdout",
    )
    report_path = tmp_path / "semantic-scores.json"
    load_counts = {"image_reward": 0, "vqa_score": 0}

    def load_image_reward(cache_root):
        del cache_root
        load_counts["image_reward"] += 1

        def score(prompt, image_path):
            del prompt
            return 1.0 if "/baseline/" in image_path else 0.75

        return score, {"implementation": "test-image-reward"}

    def load_vqa_score(*, model_cache, huggingface_cache):
        del model_cache, huggingface_cache
        load_counts["vqa_score"] += 1

        def score(prompts, image_paths):
            del prompts
            return [0.9 if "/baseline/" in path else 0.8 for path in image_paths]

        return score, {"implementation": "test-vqa-score"}

    monkeypatch.setattr(semantics, "load_image_reward", load_image_reward)
    monkeypatch.setattr(semantics, "load_vqa_score", load_vqa_score)
    args = SimpleNamespace(
        quality_input=(str(calibration), str(holdout)),
        out=str(report_path),
        metrics=("image_reward", "vqa_score"),
        expected_images=6,
        image_reward_cache=str(tmp_path / "image-reward"),
        vqa_model_cache=str(tmp_path / "vqa-model"),
        huggingface_cache=str(tmp_path / "huggingface"),
        vqa_batch_size=2,
    )

    semantics.evaluate(args)
    first = json.loads(report_path.read_text(encoding="utf-8"))

    assert first["schema"] == semantics.REPORT_SCHEMA
    assert first["schema_revision"] == semantics.REPORT_SCHEMA_REVISION
    assert first["complete"] is True
    assert len(first["images"]) == 6
    assert len(first["comparisons"]) == 4
    assert len(first["summary"]) == 4
    assert all(
        row["candidate_minus_baseline"]
        == {"image_reward": -0.25, "vqa_score": pytest.approx(-0.1)}
        for row in first["comparisons"]
    )
    assert load_counts == {"image_reward": 1, "vqa_score": 1}

    semantics.evaluate(args)

    assert load_counts == {"image_reward": 1, "vqa_score": 1}


def test_semantic_resume_rejects_report_revision_drift(tmp_path):
    manifest = _write_quality_manifest(tmp_path, split="calibration")
    images, sources = semantics.collect_unique_images((manifest,))
    report_path = tmp_path / "semantic-scores.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": semantics.REPORT_SCHEMA,
                "schema_revision": semantics.REPORT_SCHEMA_REVISION + 1,
                "sources": sources,
                "images": images,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        semantics._restore_scores(report_path, images, sources)


def test_semantic_resume_accepts_append_only_quality_ladder_prefix(tmp_path):
    manifest = _write_quality_manifest(
        tmp_path,
        split="confirmation",
        candidate_ids=("candidate-a",),
    )
    first_images, first_sources = semantics.collect_unique_images((manifest,))
    for row in first_images:
        row["scores"] = {"image_reward": 0.5}
    report_path = tmp_path / "semantic-scores.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": semantics.REPORT_SCHEMA,
                "schema_revision": semantics.REPORT_SCHEMA_REVISION,
                "sources": first_sources,
                "images": first_images,
            }
        ),
        encoding="utf-8",
    )
    _write_quality_manifest(
        tmp_path,
        split="confirmation",
        candidate_ids=("candidate-a", "candidate-b"),
    )
    next_images, next_sources = semantics.collect_unique_images((manifest,))

    restored = semantics._restore_scores(report_path, next_images, next_sources)

    scores = {row["image_id"]: row["scores"] for row in restored["images"]}
    assert scores["confirmation:baseline:p000-s0"] == {"image_reward": 0.5}
    assert scores["confirmation:candidate-a:p000-s0"] == {"image_reward": 0.5}
    assert scores["confirmation:candidate-b:p000-s0"] == {}
