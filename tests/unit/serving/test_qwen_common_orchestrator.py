from __future__ import annotations

from pathlib import Path

import pytest

from difflet.common.orchestrators import qwen_image
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.options import CompilePolicy
from difflet.serving.types import ServingProfile


def _profile(cache_dir: Path) -> ServingProfile:
    return ServingProfile(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        height=1024,
        width=1024,
        num_frames=None,
        parallel=DiffletParallelConfig(tp_degree=4, cp_degree=1),
        cache_dir=str(cache_dir),
    )


def test_qwen_artifact_check_rejects_missing_and_empty_dirs(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        Path(spec.artifact_path).mkdir(parents=True)

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_artifact_check_rejects_manifest_only_dirs(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "manifest.json").write_text("{}", encoding="utf-8")

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_artifact_check_rejects_neff_without_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")

    missing = qwen_image.missing_artifacts(profile)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_artifact_check_accepts_neff_with_matching_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")

    qwen_image.write_serving_markers(profile)

    assert qwen_image.missing_artifacts(profile) == []


def test_qwen_artifact_check_rejects_mismatched_serving_marker(tmp_path):
    profile = _profile(tmp_path)
    for spec in qwen_image.compile_plan(profile):
        artifact_path = Path(spec.artifact_path)
        artifact_path.mkdir(parents=True)
        (artifact_path / "graph.neff").write_bytes(b"neff")
    qwen_image.write_serving_markers(profile)
    other = ServingProfile(
        model_id=profile.model_id,
        model_type=profile.model_type,
        height=512,
        width=512,
        num_frames=None,
        parallel=profile.parallel,
        cache_dir=profile.cache_dir,
    )

    missing = qwen_image.missing_artifacts(other)

    assert {spec.stage_id for spec in missing} == {"text", "generate", "vae"}


def test_qwen_ensure_artifacts_never_fails_for_missing_artifacts(tmp_path):
    profile = _profile(tmp_path)

    with pytest.raises(RuntimeError, match="missing Qwen compiled artifacts"):
        qwen_image.ensure_artifacts(profile, CompilePolicy.NEVER)
