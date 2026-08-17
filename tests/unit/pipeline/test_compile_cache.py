"""Unit tests for difflet.pipeline.compile_cache."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from difflet.pipeline.compile_cache import (
    MANIFEST_SCHEMA_VERSION,
    CacheSpec,
    build_policy_binding_receipt,
    cache_key,
    cache_path,
    executable_inputs_sha256,
    has_valid_manifest,
    issue_policy_binding_receipt,
    manifest_path,
    normalize_dtype,
    read_manifest,
    toolchain_versions,
    validate_policy_binding_receipt,
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


def test_policy_app_kwargs_are_not_runtime_only_and_split_from_executable_identity():
    baseline = _spec(application_kwargs={"teacache_probe_enabled": True})
    first = _spec(application_kwargs={"teacache_speedup": 1.2})
    second = _spec(application_kwargs={"teacache_speedup": 2.0})

    assert cache_key(first) == cache_key(second) == cache_key(baseline)
    first_receipt = build_policy_binding_receipt(first, first.application_kwargs)
    second_receipt = build_policy_binding_receipt(second, second.application_kwargs)
    assert first_receipt is not None
    assert second_receipt is not None
    assert first_receipt.executable_inputs_sha256 == second_receipt.executable_inputs_sha256
    assert first_receipt.sha256 != second_receipt.sha256


def test_disabled_probe_capability_is_the_baseline_executable_identity():
    assert cache_key(_spec(application_kwargs={"teacache_probe_enabled": False})) == cache_key(
        _spec()
    )


def test_runtime_only_app_kwargs_do_not_create_policy_receipt():
    spec = _spec(application_kwargs={"enable_host_pipeline": True})
    assert build_policy_binding_receipt(spec, spec.application_kwargs) is None


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


def test_manifest_cache_key_is_validated_not_just_cache_inputs(tmp_path):
    spec = _spec()
    write_manifest(tmp_path, spec)
    manifest = read_manifest(tmp_path)
    manifest["cache_key"] = "0" * 16
    manifest_path(tmp_path).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert has_valid_manifest(tmp_path, spec) is False


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


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_document(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _fake_qualified_profile(tmp_path, suffix="one"):
    profile_path = tmp_path / f"profile-{suffix}.json"
    qualification_path = tmp_path / f"qualification-{suffix}.json"
    profile_document = _content_document({"profile": suffix})
    qualification_document = _content_document({"qualification": suffix})
    profile_path.write_text(json.dumps(profile_document) + "\n", encoding="utf-8")
    qualification_path.write_text(
        json.dumps(qualification_document) + "\n", encoding="utf-8"
    )
    candidate = SimpleNamespace(
        source_path=profile_path.resolve(),
        file_sha256=_file_sha256(profile_path),
        content_sha256=profile_document["sha256"],
    )
    qualified = SimpleNamespace(
        candidate=candidate,
        candidate_id=f"candidate-{suffix}",
        qualification_path=qualification_path.resolve(),
        qualification_sha256=qualification_document["sha256"],
        build_id=f"build-{suffix}",
    )
    kwargs = {
        "cache_profile_file": str(profile_path),
        "cache_profile_qualification_file": str(qualification_path),
        "cache_runtime_model_id": "org/Model",
        "cache_runtime_model_revision": "revision",
        "enable_host_pipeline": True,
    }
    return qualified, kwargs


def test_qualified_policy_receipt_binds_artifact_hashes_and_executable(tmp_path):
    qualified, kwargs = _fake_qualified_profile(tmp_path)
    spec = _spec(application_kwargs=kwargs, revision="revision")
    compiled = tmp_path / "compiled"
    write_manifest(compiled, spec)

    receipt = issue_policy_binding_receipt(
        compiled,
        spec,
        kwargs,
        qualified_profile=qualified,
    )

    assert receipt is not None
    validate_policy_binding_receipt(receipt, spec)
    document = receipt.to_dict()
    assert document["executable"] == {
        "cache_key": cache_key(spec),
        "cache_inputs_sha256": executable_inputs_sha256(spec),
    }
    policy = document["policy"]
    assert policy["kind"] == "qualified_cache_profile"
    assert policy["candidate_id"] == "candidate-one"
    assert "enable_host_pipeline" not in policy["application_kwargs"]
    assert policy["artifacts"]["cache_profile_file"]["file_sha256"] == (
        qualified.candidate.file_sha256
    )
    assert policy["artifacts"]["cache_profile_qualification_file"][
        "content_sha256"
    ] == qualified.qualification_sha256


def test_policy_receipt_is_not_issued_before_executable_manifest_matches(tmp_path):
    qualified, kwargs = _fake_qualified_profile(tmp_path)
    spec = _spec(application_kwargs=kwargs, revision="revision")

    with pytest.raises(RuntimeError, match="cannot issue policy receipt"):
        issue_policy_binding_receipt(
            tmp_path / "missing",
            spec,
            kwargs,
            qualified_profile=qualified,
        )


def test_policy_receipt_rejects_qualification_changed_after_profile_load(tmp_path):
    qualified, kwargs = _fake_qualified_profile(tmp_path)
    spec = _spec(application_kwargs=kwargs, revision="revision")
    compiled = tmp_path / "compiled"
    write_manifest(compiled, spec)
    qualification_path = qualified.qualification_path
    qualification_path.write_text(
        json.dumps(_content_document({"qualification": "tampered"})) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed before policy receipt issuance"):
        issue_policy_binding_receipt(
            compiled,
            spec,
            kwargs,
            qualified_profile=qualified,
        )


def test_different_profiles_share_executable_but_have_different_receipts(tmp_path):
    first_profile, first_kwargs = _fake_qualified_profile(tmp_path, "first")
    second_profile, second_kwargs = _fake_qualified_profile(tmp_path, "second")
    first_spec = _spec(application_kwargs=first_kwargs, revision="revision")
    second_spec = _spec(application_kwargs=second_kwargs, revision="revision")
    assert cache_key(first_spec) == cache_key(second_spec)

    compiled = tmp_path / "compiled"
    write_manifest(compiled, first_spec)
    first_receipt = issue_policy_binding_receipt(
        compiled,
        first_spec,
        first_kwargs,
        qualified_profile=first_profile,
    )
    second_receipt = issue_policy_binding_receipt(
        compiled,
        second_spec,
        second_kwargs,
        qualified_profile=second_profile,
    )
    assert first_receipt is not None and second_receipt is not None
    assert first_receipt.executable_cache_key == second_receipt.executable_cache_key
    assert first_receipt.sha256 != second_receipt.sha256
