from __future__ import annotations

from pathlib import Path

import pytest

from difflet.serving.options import DownloadPolicy
from difflet.serving.orchestrators.base import resolve_hf_model_source

_COMMIT = "a" * 40


def test_resolve_hf_model_source_pins_snapshot_commit(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / _COMMIT
    snapshot.mkdir(parents=True)
    captured = {}

    def fake_resolve(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return str(snapshot)

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", fake_resolve)

    source = resolve_hf_model_source(
        "org/model",
        revision="main",
        download_policy=DownloadPolicy.AUTO,
        allow_patterns=("*.json",),
    )

    assert source.resolved_source_id == _COMMIT
    assert source.pinned_model_path == str(snapshot.resolve())
    assert captured == {
        "model_id": "org/model",
        "revision": "main",
        "local_files_only": False,
        "allow_patterns": ("*.json",),
    }


def test_resolve_hf_model_source_never_uses_local_only(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / _COMMIT
    snapshot.mkdir(parents=True)
    captured = {}

    def fake_resolve(model_id, **kwargs):
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", fake_resolve)

    resolve_hf_model_source(
        "org/model",
        revision=None,
        download_policy=DownloadPolicy.NEVER,
    )

    assert captured["local_files_only"] is True


def test_resolve_hf_model_source_rejects_non_snapshot_path(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *args, **kwargs: str(model_dir),
    )

    with pytest.raises(ValueError, match="not commit-addressed"):
        resolve_hf_model_source(
            "org/model",
            revision=None,
            download_policy=DownloadPolicy.AUTO,
        )


def test_resolve_hf_model_source_rejects_local_model_directory(tmp_path):
    local_model = tmp_path / "model"
    local_model.mkdir()

    with pytest.raises(ValueError, match="requires a Hugging Face model ID"):
        resolve_hf_model_source(
            str(local_model),
            revision=None,
            download_policy=DownloadPolicy.AUTO,
        )
