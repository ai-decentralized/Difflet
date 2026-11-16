"""Page-cache control on ordinary files (no root needed)."""
from __future__ import annotations

import os

from difflet.bench import pagecache as P


def test_files_under_resolves_symlinks_and_dedupes(tmp_path):
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"x" * 10)
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    os.symlink(blob, snap / "model.safetensors")
    os.symlink(blob, snap / "copy.safetensors")
    files = P.files_under([snap, tmp_path / "missing"])
    assert files == [blob.resolve()]
    assert P.total_bytes(files) == 10


def test_preread_evict_and_residency(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(os.urandom(1 << 20))
    assert P.preread([path]) == 1 << 20
    resident, total = P.residency([path])
    assert total == 1 << 20 and 0 < resident <= total
    assert P.evict([path]) == 1 << 20  # whether the kernel complied is what residency measures
    assert P.resident_bytes(tmp_path / "w.bin") <= total


def test_residency_of_an_empty_file(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert P.residency([path]) == (0, 0)
