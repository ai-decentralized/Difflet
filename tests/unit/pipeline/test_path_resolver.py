"""Unit tests for difflet.pipeline.path_resolver."""

import sys
import types

import pytest

from difflet.pipeline.path_resolver import (
    DEFAULT_DIFFUSERS_PATTERNS,
    resolve_model_path,
)


def test_local_path_returned_resolved(tmp_path):
    out = resolve_model_path(str(tmp_path))
    assert out == str(tmp_path.resolve())


def test_default_patterns_constant_contents():
    assert "transformer/*" in DEFAULT_DIFFUSERS_PATTERNS
    assert "*.json" in DEFAULT_DIFFUSERS_PATTERNS


def test_hf_download_uses_default_patterns(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/downloaded/path"

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    out = resolve_model_path("org/NotLocalModel-xyz")
    assert out == "/downloaded/path"
    assert captured["repo_id"] == "org/NotLocalModel-xyz"
    assert captured["allow_patterns"] == list(DEFAULT_DIFFUSERS_PATTERNS)


def test_hf_download_honors_explicit_patterns(monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/p"

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    resolve_model_path(
        "org/Model-not-here",
        revision="abc",
        local_files_only=True,
        allow_patterns=["custom/*"],
    )
    assert captured["allow_patterns"] == ["custom/*"]
    assert captured["revision"] == "abc"
    assert captured["local_files_only"] is True


def test_missing_huggingface_hub_raises(monkeypatch):
    # Simulate huggingface_hub not being importable.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError):
        resolve_model_path("org/Model-definitely-not-local-zzz")
