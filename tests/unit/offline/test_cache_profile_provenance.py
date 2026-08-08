from __future__ import annotations

from pathlib import Path

import pytest

from difflet.offline.cache_profile import provenance


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_implementation_bundle_is_sorted_and_validates_required_files(tmp_path):
    first = _write(tmp_path / "pkg" / "a.py", "A = 1\n")
    second = _write(tmp_path / "pkg" / "b.py", "B = 2\n")

    bundle = provenance.implementation_bundle([second, first], root=tmp_path)
    resolved = provenance.validate_implementation_bundle(
        bundle,
        root=tmp_path,
        required_paths=[first, second],
    )

    assert [row["path"] for row in bundle["files"]] == ["pkg/a.py", "pkg/b.py"]
    assert resolved == (first.resolve(), second.resolve())


def test_implementation_bundle_rejects_source_changes(tmp_path):
    source = _write(tmp_path / "implementation.py", "VALUE = 1\n")
    bundle = provenance.implementation_bundle([source], root=tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        provenance.validate_implementation_bundle(bundle, root=tmp_path)


def test_implementation_bundle_rejects_missing_required_file(tmp_path):
    first = _write(tmp_path / "first.py", "FIRST = True\n")
    second = _write(tmp_path / "second.py", "SECOND = True\n")
    bundle = provenance.implementation_bundle([first], root=tmp_path)

    with pytest.raises(ValueError, match="missing required files"):
        provenance.validate_implementation_bundle(
            bundle,
            root=tmp_path,
            required_paths=[first, second],
        )


def test_implementation_bundle_rejects_duplicate_paths(tmp_path):
    source = _write(tmp_path / "implementation.py", "VALUE = 1\n")

    with pytest.raises(ValueError, match="duplicated"):
        provenance.implementation_bundle([source, source], root=tmp_path)
