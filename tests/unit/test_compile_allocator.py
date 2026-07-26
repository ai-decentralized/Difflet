"""Unit tests for `difflet.backends.trainium.utils.compile_allocator`.

Cover hiding preloaded allocators from the `neuronx-cc` subprocess.
"""

from __future__ import annotations

import os

import pytest

from difflet.backends.trainium.utils.compile_allocator import (
    strip_allocator_preloads,
    without_allocator_preload,
)

JEMALLOC = "/opt/venv/lib/python3.12/site-packages/torch_neuronx/lib/libjemalloc.so"
OTHER = "/usr/lib/libsomethingelse.so"


def test_strip_removes_jemalloc() -> None:
    assert strip_allocator_preloads(JEMALLOC) == ""


def test_strip_preserves_other_preloads_in_order() -> None:
    value = f"{OTHER}:{JEMALLOC}:/usr/lib/libthird.so"
    assert strip_allocator_preloads(value) == f"{OTHER}:/usr/lib/libthird.so"


def test_strip_drops_empty_segments() -> None:
    assert strip_allocator_preloads(f"{OTHER}::") == OTHER


def test_strip_is_noop_without_allocator() -> None:
    assert strip_allocator_preloads(OTHER) == OTHER


def test_allocator_hidden_inside_block_and_restored_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = f"{OTHER}:{JEMALLOC}"
    monkeypatch.setenv("LD_PRELOAD", original)

    with without_allocator_preload():
        assert os.environ["LD_PRELOAD"] == OTHER

    assert os.environ["LD_PRELOAD"] == original


def test_ld_preload_unset_when_only_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_PRELOAD", JEMALLOC)

    with without_allocator_preload():
        assert "LD_PRELOAD" not in os.environ

    assert os.environ["LD_PRELOAD"] == JEMALLOC


def test_absent_ld_preload_is_not_created(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    with without_allocator_preload():
        assert "LD_PRELOAD" not in os.environ

    assert "LD_PRELOAD" not in os.environ


def test_nested_scope_is_reentrant(monkeypatch: pytest.MonkeyPatch) -> None:
    original = f"{OTHER}:{JEMALLOC}"
    monkeypatch.setenv("LD_PRELOAD", original)

    with without_allocator_preload():
        with without_allocator_preload():
            assert os.environ["LD_PRELOAD"] == OTHER
        # The inner scope must not have restored the allocator early.
        assert os.environ["LD_PRELOAD"] == OTHER

    assert os.environ["LD_PRELOAD"] == original


def test_ld_preload_restored_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    original = f"{OTHER}:{JEMALLOC}"
    monkeypatch.setenv("LD_PRELOAD", original)

    with pytest.raises(RuntimeError):
        with without_allocator_preload():
            raise RuntimeError("compile blew up")

    assert os.environ["LD_PRELOAD"] == original
