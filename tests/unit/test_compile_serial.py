"""Concurrency and restoration checks for memory-bounded bucket compilation."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest

from difflet.backends.trainium.utils.compile_serial import serial_bucket_compilation


def test_bucket_jobs_are_serialized_and_restored(monkeypatch):
    from neuronx_distributed.trace import model_builder

    active = peak = 0
    counter_lock = Lock()

    def job(value):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with counter_lock:
            active -= 1
        return value * 2

    monkeypatch.setattr(model_builder, "neuron_xla_compile", job)
    with serial_bucket_compilation():
        wrapped = model_builder.neuron_xla_compile
        with serial_bucket_compilation():
            assert model_builder.neuron_xla_compile is wrapped
            with ThreadPoolExecutor(max_workers=3) as pool:
                assert list(pool.map(wrapped, range(3))) == [0, 2, 4]
        assert model_builder.neuron_xla_compile is wrapped
    assert peak == 1
    assert model_builder.neuron_xla_compile is job


def test_compile_failure_propagates_and_restores(monkeypatch):
    from neuronx_distributed.trace import model_builder

    def fail():
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(model_builder, "neuron_xla_compile", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        with serial_bucket_compilation():
            model_builder.neuron_xla_compile()
    assert model_builder.neuron_xla_compile is fail
