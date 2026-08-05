"""Unit tests for difflet.pipeline.compile_cache."""

import torch

from difflet.pipeline.compile_cache import (
    MANIFEST_SCHEMA_VERSION,
    CacheSpec,
    cache_key,
    cache_path,
    has_valid_manifest,
    manifest_path,
    normalize_dtype,
    read_manifest,
    toolchain_versions,
    write_manifest,
)
from difflet.pipeline.parallel_config import CandidateConfig, DiffletParallelConfig


def _spec(**overrides):
    base = dict(
        model_id="org/Model",
        model_path="/local/path",
        model_name="model",
        parallel=DiffletParallelConfig(tp_degree=2),
        dtype=torch.bfloat16,
        height=512,
        width=512,
        num_frames=None,
    )
    base.update(overrides)
    return CacheSpec(**base)


def test_normalize_dtype_variants():
    assert normalize_dtype(torch.bfloat16) == "bfloat16"
    assert normalize_dtype("bf16") == "bfloat16"
    assert normalize_dtype("torch.float16") == "float16"
    assert normalize_dtype("fp8_e4m3") == "float8_e4m3fn"
    assert normalize_dtype(None) == "none"
    # Unknown string passes through lowercased.
    assert normalize_dtype("weird") == "weird"
    # Last-resort stringify for exotic objects.
    assert normalize_dtype(123) == "123"


def test_toolchain_versions_includes_python():
    versions = toolchain_versions()
    assert "python" in versions
    assert "torch" in versions


def test_cache_inputs_excludes_model_path():
    spec = _spec()
    inputs = spec.cache_inputs()
    assert "model_path" not in inputs
    assert inputs["model_id"] == "org/Model"
    assert inputs["dtype"] == "bfloat16"
    assert spec.manifest_metadata()["model_path"] == "/local/path"


def test_runtime_only_app_kwargs_excluded_from_key():
    spec_a = _spec(
        application_kwargs={
            "enable_host_pipeline": True,
            "foo": 1,
        }
    )
    spec_b = _spec(application_kwargs={"foo": 1})
    # Runtime-only kwargs do not change the cache key.
    assert cache_key(spec_a) == cache_key(spec_b)


def test_trivial_candidate_keeps_key_identical():
    spec_default = _spec()
    spec_trivial = _spec(candidate=CandidateConfig(max_candidates=1))
    assert cache_key(spec_default) == cache_key(spec_trivial)


def test_nontrivial_candidate_changes_key_and_injects():
    spec_default = _spec()
    spec_cand = _spec(candidate=CandidateConfig(max_candidates=4))
    assert cache_key(spec_default) != cache_key(spec_cand)
    assert "candidate" in spec_cand.cache_inputs()


def test_cache_path_layout(tmp_path):
    spec = _spec()
    path = cache_path(tmp_path, spec)
    assert path.parent.name == "model"
    assert path.name == cache_key(spec)


def test_cache_path_uses_default_dir_when_none():
    spec = _spec()
    path = cache_path(None, spec)
    assert path.name == cache_key(spec)


def test_write_and_validate_manifest(tmp_path):
    spec = _spec()
    write_manifest(tmp_path, spec)
    assert manifest_path(tmp_path).exists()
    manifest = read_manifest(tmp_path)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert has_valid_manifest(tmp_path, spec) is True


def test_has_valid_manifest_false_when_missing(tmp_path):
    assert read_manifest(tmp_path) is None
    assert has_valid_manifest(tmp_path, _spec()) is False


def test_has_valid_manifest_false_on_spec_mismatch(tmp_path):
    write_manifest(tmp_path, _spec(height=512))
    assert has_valid_manifest(tmp_path, _spec(height=1024)) is False


def test_read_manifest_handles_corrupt_file(tmp_path):
    manifest_path(tmp_path).write_text("{ not valid json", encoding="utf-8")
    assert read_manifest(tmp_path) is None


def test_manifest_precision_schedule_recorded(tmp_path):
    spec = _spec(precision_schedule={"assignments": {"0:q": "bf16"}})
    write_manifest(tmp_path, spec)
    manifest = read_manifest(tmp_path)
    assert manifest["precision_schedule"]["assignments"]["0:q"] == "bf16"


def test_manifest_precision_schedule_none():
    assert _spec().manifest_precision_schedule() is None
