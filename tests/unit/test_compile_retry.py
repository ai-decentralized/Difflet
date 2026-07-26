"""Unit tests for `difflet.backends.trainium.utils.compile_retry`.

Cover forcing NxD to retry cached compile failures instead of replaying them.
"""

from __future__ import annotations

import pytest

from difflet.backends.trainium.utils import compile_retry
from difflet.backends.trainium.utils.compile_retry import retrying_cached_failures


@pytest.fixture()
def patched_model_builder(monkeypatch: pytest.MonkeyPatch):
    """Stand in for the compile entry points in NxD's `model_builder`."""
    model_builder = pytest.importorskip(
        "neuronx_distributed.trace.model_builder",
        reason="Neuron toolchain not installed",
    )
    calls: list[dict] = []

    def record(**kwargs):
        calls.append(kwargs)
        return "neff"

    for name in compile_retry._PATCHED_COMPILE_FNS:
        monkeypatch.setattr(model_builder, name, record, raising=False)

    return model_builder, calls


def test_retry_forced_inside_block(patched_model_builder) -> None:
    model_builder, calls = patched_model_builder

    with retrying_cached_failures():
        model_builder.neuron_xla_compile(retry_failed_compilation=False, cache_key="k")

    assert calls == [{"retry_failed_compilation": True, "cache_key": "k"}]


def test_wlo_compile_also_patched(patched_model_builder) -> None:
    model_builder, calls = patched_model_builder

    with retrying_cached_failures():
        model_builder.neuron_xla_wlo_compile(retry_failed_compilation=False)

    assert calls == [{"retry_failed_compilation": True}]


def test_originals_restored_after_block(patched_model_builder) -> None:
    model_builder, calls = patched_model_builder
    before = model_builder.neuron_xla_compile

    with retrying_cached_failures():
        assert model_builder.neuron_xla_compile is not before

    assert model_builder.neuron_xla_compile is before

    model_builder.neuron_xla_compile(retry_failed_compilation=False)
    assert calls == [{"retry_failed_compilation": False}]


def test_originals_restored_on_exception(patched_model_builder) -> None:
    model_builder, _calls = patched_model_builder
    before = model_builder.neuron_xla_compile

    with pytest.raises(RuntimeError):
        with retrying_cached_failures():
            raise RuntimeError("compile blew up")

    assert model_builder.neuron_xla_compile is before


def test_nesting_is_reentrant(patched_model_builder) -> None:
    """An inner compile scope must not double-wrap or restore a wrapper."""
    model_builder, calls = patched_model_builder
    before = model_builder.neuron_xla_compile

    with retrying_cached_failures():
        outer = model_builder.neuron_xla_compile
        with retrying_cached_failures():
            assert model_builder.neuron_xla_compile is outer
        # The inner scope must not have restored the original early.
        assert model_builder.neuron_xla_compile is outer
        model_builder.neuron_xla_compile(retry_failed_compilation=False)

    assert model_builder.neuron_xla_compile is before
    assert calls == [{"retry_failed_compilation": True}]
