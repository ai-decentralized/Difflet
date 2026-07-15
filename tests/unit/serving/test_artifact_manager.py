from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from difflet.serving.artifact_manager import GENERATION_MANIFEST, ImmutableArtifactManager
from difflet.serving.options import CompilePolicy
from difflet.serving.types import CompileArtifactIdentity


def _identity(value: str = "profile") -> CompileArtifactIdentity:
    payload = json.dumps({"profile": value}, sort_keys=True, separators=(",", ":")).encode()
    return CompileArtifactIdentity(1, payload, hashlib.sha256(payload).hexdigest())


def _validate(path: Path) -> None:
    payload = path / "model.neff"
    if not payload.is_file() or not payload.read_bytes():
        raise ValueError("missing model.neff")


def test_auto_publishes_then_reuses_immutable_generation(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)
    calls = []

    def compile_artifact(target):
        calls.append(target.staging_path)
        (target.staging_path / "model.neff").write_bytes(b"compiled")

    first = manager.prepare(
        model_type="qwen_image",
        artifact_id="text",
        identity=_identity(),
        policy=CompilePolicy.AUTO,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )
    second = manager.prepare(
        model_type="qwen_image",
        artifact_id="text",
        identity=_identity(),
        policy=CompilePolicy.AUTO,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )

    assert first == second
    assert first.generation_id == "g0000000000000001"
    assert len(calls) == 1
    assert first.manifest_path.name == GENERATION_MANIFEST


def test_force_publishes_next_generation_and_never_reuses_newest(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)

    def compile_artifact(target):
        (target.staging_path / "model.neff").write_bytes(target.staging_path.name.encode())

    first = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.FORCE,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )
    second = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.FORCE,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )
    reused = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.NEVER,
        compile_artifact=lambda target: pytest.fail("NEVER must not compile"),
        validate_payload=_validate,
    )

    assert first.generation_id == "g0000000000000001"
    assert second.generation_id == "g0000000000000002"
    assert reused == second


def test_corrupt_newest_generation_falls_back_to_older_valid_generation(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)

    def compile_artifact(target):
        (target.staging_path / "model.neff").write_bytes(target.staging_path.name.encode())

    first = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.FORCE,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )
    second = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.FORCE,
        compile_artifact=compile_artifact,
        validate_payload=_validate,
    )
    (second.path / "model.neff").write_bytes(b"tampered")

    reused = manager.prepare(
        model_type="flux",
        artifact_id="pipeline",
        identity=_identity(),
        policy=CompilePolicy.NEVER,
        compile_artifact=lambda target: pytest.fail("NEVER must not compile"),
        validate_payload=_validate,
    )

    assert reused == first


def test_compile_failure_does_not_publish_generation(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)

    def fail_compile(target):
        (target.staging_path / "partial").write_bytes(b"partial")
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        manager.prepare(
            model_type="flux",
            artifact_id="pipeline",
            identity=_identity(),
            policy=CompilePolicy.AUTO,
            compile_artifact=fail_compile,
            validate_payload=_validate,
        )

    assert list(tmp_path.rglob("g[0-9]*")) == []


def test_payload_symlink_is_rejected(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)
    external = tmp_path / "external.neff"
    external.write_bytes(b"external")

    def compile_artifact(target):
        (target.staging_path / "model.neff").symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        manager.prepare(
            model_type="flux",
            artifact_id="pipeline",
            identity=_identity(),
            policy=CompilePolicy.AUTO,
            compile_artifact=compile_artifact,
            validate_payload=lambda path: None,
        )


def test_concurrent_auto_publish_compiles_once(tmp_path):
    manager = ImmutableArtifactManager(tmp_path)
    compile_count = 0
    count_lock = threading.Lock()
    bindings = []

    def compile_artifact(target):
        nonlocal compile_count
        with count_lock:
            compile_count += 1
        time.sleep(0.05)
        (target.staging_path / "model.neff").write_bytes(b"compiled")

    def prepare():
        bindings.append(
            manager.prepare(
                model_type="flux",
                artifact_id="pipeline",
                identity=_identity(),
                policy=CompilePolicy.AUTO,
                compile_artifact=compile_artifact,
                validate_payload=_validate,
            )
        )

    threads = [threading.Thread(target=prepare) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert compile_count == 1
    assert len(bindings) == 4
    assert len(set(bindings)) == 1
