"""Hash-bound implementation manifests for offline profile workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

IMPLEMENTATION_BUNDLE_SCHEMA = "difflet-cache-profile-implementation-bundle"
IMPLEMENTATION_BUNDLE_SCHEMA_REVISION = 1


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value using the repository canonical encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_bundle(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    """Describe every first-party source file that can affect one workflow."""

    root = Path(root).resolve()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"implementation file is outside the source root: {path}") from error
        if relative in seen:
            raise ValueError(f"implementation file is duplicated: {relative}")
        if not path.is_file():
            raise ValueError(f"implementation file does not exist: {path}")
        seen.add(relative)
        rows.append({"path": relative, "file_sha256": sha256_file(path)})
    if not rows:
        raise ValueError("implementation bundle must contain at least one file")
    payload: dict[str, Any] = {
        "schema": IMPLEMENTATION_BUNDLE_SCHEMA,
        "schema_revision": IMPLEMENTATION_BUNDLE_SCHEMA_REVISION,
        "files": sorted(rows, key=lambda row: row["path"]),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def validate_implementation_bundle(
    document: Mapping[str, Any],
    *,
    root: Path,
    required_paths: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """Validate bundle structure, bundle digest, file hashes, and required files."""

    expected = {"schema", "schema_revision", "files", "sha256"}
    if not isinstance(document, Mapping) or set(document) != expected:
        raise ValueError("implementation bundle fields are invalid")
    if (
        document["schema"] != IMPLEMENTATION_BUNDLE_SCHEMA
        or document["schema_revision"] != IMPLEMENTATION_BUNDLE_SCHEMA_REVISION
    ):
        raise ValueError("implementation bundle schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("implementation bundle content hash is invalid")
    rows = document["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("implementation bundle files must be a non-empty list")

    root = Path(root).resolve()
    resolved: list[Path] = []
    relative_names: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "file_sha256"}:
            raise ValueError("implementation bundle file fields are invalid")
        relative = row["path"]
        digest = row["file_sha256"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("implementation bundle path must be repository-relative")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("implementation bundle path escapes the source root") from error
        if relative in relative_names:
            raise ValueError(f"implementation bundle path is duplicated: {relative}")
        if not path.is_file():
            raise ValueError(f"implementation bundle file does not exist: {relative}")
        if not isinstance(digest, str) or sha256_file(path) != digest:
            raise ValueError(f"implementation bundle hash differs for {relative}")
        relative_names.append(relative)
        resolved.append(path)
    if relative_names != sorted(relative_names):
        raise ValueError("implementation bundle files must be sorted by path")

    required = {Path(path).resolve() for path in required_paths}
    missing = required - set(resolved)
    if missing:
        names = [path.relative_to(root).as_posix() for path in sorted(missing)]
        raise ValueError(f"implementation bundle is missing required files: {names}")
    return tuple(resolved)


__all__ = [
    "IMPLEMENTATION_BUNDLE_SCHEMA",
    "IMPLEMENTATION_BUNDLE_SCHEMA_REVISION",
    "canonical_sha256",
    "implementation_bundle",
    "sha256_file",
    "validate_implementation_bundle",
]
