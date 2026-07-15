"""Immutable compiled-artifact generation publication for serving."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from difflet.serving.options import CompilePolicy
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactPublishTarget,
    CompileArtifactIdentity,
)

GENERATION_MANIFEST = "difflet_generation_manifest.json"
_GENERATION_RE = re.compile(r"^g([0-9]{16})$")

CompileArtifact = Callable[[ArtifactPublishTarget], None]
ValidatePayload = Callable[[Path], None]


class ImmutableArtifactManager:
    def __init__(self, cache_root: str | os.PathLike[str], *, lock_timeout: float = 300.0) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.lock_timeout = float(lock_timeout)
        if self.lock_timeout <= 0:
            raise ValueError("lock_timeout must be greater than 0")
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        model_type: str,
        artifact_id: str,
        identity: CompileArtifactIdentity,
        policy: CompilePolicy,
        compile_artifact: CompileArtifact,
        validate_payload: ValidatePayload,
    ) -> ArtifactBinding:
        identity_root = self._identity_root(model_type, identity)
        self._ensure_directory(identity_root)
        lock_path = identity_root / ".publish.lock"
        with self._publication_lock(lock_path):
            staging_root = identity_root / "staging"
            generations_root = identity_root / "generations"
            self._ensure_directory(staging_root)
            self._ensure_directory(generations_root)
            self._remove_abandoned_staging(staging_root)

            if policy != CompilePolicy.FORCE:
                binding = self._newest_valid_binding(
                    generations_root,
                    artifact_id=artifact_id,
                    identity=identity,
                    validate_payload=validate_payload,
                )
                if binding is not None:
                    return binding
                if policy == CompilePolicy.NEVER:
                    raise RuntimeError(f"no valid compiled generation for artifact {artifact_id!r}")

            generation_id = self._allocate_generation_id(generations_root)
            staging_path = staging_root / f"{uuid.uuid4().hex}.tmp"
            self._ensure_confined(staging_path, identity_root)
            staging_path.mkdir()
            target = ArtifactPublishTarget(
                artifact_id=artifact_id,
                identity=identity,
                identity_root=identity_root,
                staging_path=staging_path,
            )
            try:
                compile_artifact(target)
                self._ensure_confined(staging_path, identity_root)
                if staging_path.is_symlink() or not staging_path.is_dir():
                    raise ValueError("compile callback replaced the managed staging directory")
                validate_payload(staging_path)
                inventory = self._inventory(staging_path)
                content_digest = self._content_digest(inventory)
                manifest = {
                    "schema_version": 1,
                    "artifact_id": artifact_id,
                    "generation_id": generation_id,
                    "identity": {
                        "schema_version": identity.schema_version,
                        "canonical_cache_inputs_json": identity.canonical_cache_inputs_json.decode(
                            "utf-8"
                        ),
                        "digest": identity.digest,
                    },
                    "content_digest": content_digest,
                    "files": inventory,
                }
                manifest_path = staging_path / GENERATION_MANIFEST
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._fsync_tree(staging_path)
                generation_path = generations_root / generation_id
                os.rename(staging_path, generation_path)
                self._fsync_directory(generations_root)
                self._fsync_directory(identity_root)
            except BaseException:
                shutil.rmtree(staging_path, ignore_errors=True)
                raise

            binding = self._validate_generation(
                generation_path,
                artifact_id=artifact_id,
                identity=identity,
                validate_payload=validate_payload,
            )
            if binding is None:
                raise RuntimeError(f"published generation {generation_id!r} failed validation")
            return binding

    def validate_binding(
        self,
        binding: ArtifactBinding,
        *,
        validate_payload: ValidatePayload,
    ) -> None:
        self._ensure_confined(binding.path, self.cache_root)
        validated = self._validate_generation(
            binding.path,
            artifact_id=binding.artifact_id,
            identity=binding.identity,
            validate_payload=validate_payload,
        )
        if validated != binding:
            raise RuntimeError(f"artifact binding {binding.artifact_id!r} is no longer valid")

    def _identity_root(
        self,
        model_type: str,
        identity: CompileArtifactIdentity,
    ) -> Path:
        if not model_type or "/" in model_type or model_type in {".", ".."}:
            raise ValueError(f"invalid model_type {model_type!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", identity.digest):
            raise ValueError("artifact identity digest must be 64 lowercase hex characters")
        actual_digest = hashlib.sha256(identity.canonical_cache_inputs_json).hexdigest()
        if actual_digest != identity.digest:
            raise ValueError("artifact identity digest does not match canonical inputs")
        root = self.cache_root / "serving" / model_type / identity.digest
        self._ensure_confined(root, self.cache_root)
        return root

    def _newest_valid_binding(
        self,
        generations_root: Path,
        *,
        artifact_id: str,
        identity: CompileArtifactIdentity,
        validate_payload: ValidatePayload,
    ) -> ArtifactBinding | None:
        candidates = sorted(
            (
                path
                for path in generations_root.iterdir()
                if path.is_dir() and not path.is_symlink() and _GENERATION_RE.fullmatch(path.name)
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            binding = self._validate_generation(
                candidate,
                artifact_id=artifact_id,
                identity=identity,
                validate_payload=validate_payload,
            )
            if binding is not None:
                return binding
        return None

    def _validate_generation(
        self,
        path: Path,
        *,
        artifact_id: str,
        identity: CompileArtifactIdentity,
        validate_payload: ValidatePayload,
    ) -> ArtifactBinding | None:
        try:
            if path.is_symlink() or not path.is_dir() or not _GENERATION_RE.fullmatch(path.name):
                return None
            manifest_path = path / GENERATION_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_identity = {
                "schema_version": identity.schema_version,
                "canonical_cache_inputs_json": identity.canonical_cache_inputs_json.decode("utf-8"),
                "digest": identity.digest,
            }
            inventory = self._inventory(path)
            content_digest = self._content_digest(inventory)
            if (
                manifest.get("schema_version") != 1
                or manifest.get("artifact_id") != artifact_id
                or manifest.get("generation_id") != path.name
                or manifest.get("identity") != expected_identity
                or manifest.get("files") != inventory
                or manifest.get("content_digest") != content_digest
            ):
                return None
            validate_payload(path)
            return ArtifactBinding(
                artifact_id=artifact_id,
                path=path,
                manifest_path=manifest_path,
                identity=identity,
                generation_id=path.name,
                content_digest=content_digest,
            )
        except (OSError, ValueError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _inventory(self, root: Path) -> list[dict[str, object]]:
        inventory: list[dict[str, object]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise ValueError(f"artifact payload may not contain symlink {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"artifact payload contains unsupported file type {path}")
            relative = path.relative_to(root).as_posix()
            if relative == GENERATION_MANIFEST:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            inventory.append(
                {"path": relative, "size": path.stat().st_size, "sha256": digest.hexdigest()}
            )
        return inventory

    @staticmethod
    def _content_digest(inventory: list[dict[str, object]]) -> str:
        payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _allocate_generation_id(generations_root: Path) -> str:
        numbers = [
            int(match.group(1))
            for path in generations_root.iterdir()
            if (match := _GENERATION_RE.fullmatch(path.name)) is not None
        ]
        return f"g{(max(numbers, default=0) + 1):016d}"

    @contextmanager
    def _publication_lock(self, lock_path: Path) -> Iterator[None]:
        self._ensure_confined(lock_path, self.cache_root)
        if lock_path.is_symlink():
            raise ValueError(f"artifact lock may not be a symlink: {lock_path}")
        with lock_path.open("a+b") as handle:
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out acquiring artifact lock {lock_path}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _ensure_directory(self, path: Path) -> None:
        self._ensure_confined(path, self.cache_root)
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"artifact path must be a real directory: {path}")
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_confined(path: Path, root: Path) -> None:
        root_resolved = root.resolve()
        parent = path.parent.resolve()
        if parent != root_resolved and root_resolved not in parent.parents:
            raise ValueError(f"artifact path escapes cache root: {path}")

    @staticmethod
    def _remove_abandoned_staging(staging_root: Path) -> None:
        for path in staging_root.iterdir():
            if path.name.endswith(".tmp") and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_tree(self, root: Path) -> None:
        directories = [root]
        for path in root.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            elif path.is_dir():
                directories.append(path)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            self._fsync_directory(directory)
