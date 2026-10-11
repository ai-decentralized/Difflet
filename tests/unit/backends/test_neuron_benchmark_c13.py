"""Unit tests for the benchmark harness with a fake clock and synchronisation; no torch or hardware needed."""

import json

from pytest import approx

from tests.numerical import neuron_benchmark as benchmark


class FakeDevice:
    """A clock that only advances when a case runs; each case has a duration per call."""

    def __init__(self):
        self.now = 0.0
        self.log = []
        self.syncs = 0

    def clock(self):
        return self.now

    def sync(self):
        self.syncs += 1

    def case(self, name, durations_ms):
        calls = iter(durations_ms) if not callable(durations_ms) else None

        def fn():
            self.log.append(name)
            ms = durations_ms(len([n for n in self.log if n == name])) if calls is None else next(calls)
            self.now += ms / 1e3

        return fn


def run(dev, cases, **kw):
    return benchmark.run(cases, sync=dev.sync, clock=dev.clock, **kw)


def test_first_call_is_separate_and_samples_are_iters_times_rounds():
    dev = FakeDevice()
    result, = run(dev, {"a": dev.case("a", lambda n: 500.0 if n == 1 else 10.0)}, warmup=2, iters=4, rounds=3)
    assert result.first_call_ms == approx(500.0)
    assert result.samples_ms == approx([10.0] * 12)
    assert result.median_ms == approx(10.0) and result.flags == []


def test_cases_are_interleaved_by_round():
    dev = FakeDevice()
    run(dev, {"a": dev.case("a", lambda n: 1.0), "b": dev.case("b", lambda n: 1.0)}, warmup=0, iters=2, rounds=2)
    timed = dev.log[2:]  # after one first call per case
    assert timed == ["a", "a", "b", "b", "a", "a", "b", "b"]


def test_every_timed_call_is_synchronised_on_both_sides():
    dev = FakeDevice()
    run(dev, {"a": dev.case("a", lambda n: 1.0)}, warmup=0, iters=3, rounds=2)
    timed_calls = 1 + 3 * 2
    assert dev.syncs == 2 * timed_calls + 1  # +1: the sync after warm-up


def test_drift_flag_when_the_device_slows_between_rounds():
    dev = FakeDevice()
    # call 1 = first call; then 3 rounds of 4 calls: 10 ms, 10 ms, then 20 ms (throttled)
    slow = lambda n: 20.0 if n > 9 else 10.0  # noqa: E731
    result, = run(dev, {"a": dev.case("a", slow)}, warmup=0, iters=4, rounds=3)
    assert result.round_medians_ms == approx([10.0, 10.0, 20.0])
    assert result.drift == approx(2.0) and any(f.startswith("drift") for f in result.flags)


def test_unstable_flag_on_wide_spread():
    dev = FakeDevice()
    noisy = lambda n: 10.0 if n % 2 else 20.0  # noqa: E731
    result, = run(dev, {"a": dev.case("a", noisy)}, warmup=0, iters=10, rounds=1)
    assert "unstable" in result.flags


def test_cache_release_and_peak_memory():
    dev = FakeDevice()
    releases = []
    peaks = iter([{"peak_bytes": 5, "current_bytes": 1}, {"peak_bytes": 9}, {"peak_bytes": 7}])
    result, = run(dev, {"a": dev.case("a", lambda n: 1.0)}, warmup=0, iters=1, rounds=3,
                  release_cache=lambda: releases.append(1), memory_stats=lambda: next(peaks))
    assert len(releases) == 1 + 3  # before the first call, then before each round
    assert result.peak_memory == {"peak_bytes": 9}


def test_report_records_build_config_and_samples(capsys):
    dev = FakeDevice()
    results = run(dev, {"a": dev.case("a", lambda n: 2.0)}, warmup=0, iters=2, rounds=1)
    record = json.loads(benchmark.report(results, {"mode": "compile", "tokens": 4680}))
    assert record["config"] == {"mode": "compile", "tokens": 4680}
    assert record["results"][0]["samples_ms"] == approx([2.0, 2.0])
    assert "python" in record["build"]
    assert "median" in capsys.readouterr().out
