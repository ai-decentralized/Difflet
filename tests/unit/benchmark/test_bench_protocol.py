"""The cross-device benchmark protocol, pinned.

Every device folder must be measured the same way: a page-cache-dropped cold
run in a fresh process, discarded cache-warming runs, N warm fresh-process runs,
then -- on adapters that keep the model resident -- synced and natural requests
whose synced deltas are the per-step figure. These tests run the harness
against fake adapters so the ORDER and the SOURCE of every metric is fixed
without a chip.
"""
from __future__ import annotations

import pytest

from benchmark import bench
from benchmark.harness import BackendAdapter


class _FreshOnlyAdapter(BackendAdapter):
    """trn2-shaped: every run is a fresh process, nothing stays resident."""

    name = "fake-fresh"

    def __init__(self):
        self.calls: list[str] = []
        self.n = 0

    def device_info(self):
        return "fake"

    def prepare(self, spec):
        self.calls.append("prepare")

    def compile(self, spec):
        self.calls.append("compile")
        return 1.0, {"x": 1.0}

    def run_generate(self, spec):
        self.n += 1
        self.calls.append("generate")
        return {"wall_seconds": 100.0 - self.n, "load_seconds": 30.0,
                "step_seconds": [0.5, 0.5, 0.6], "output": {"finite": True}}


class _ResidentAdapter(_FreshOnlyAdapter):
    """TPU-shaped: run_generate restarts the workers, run_request reuses them."""

    name = "fake-resident"
    supports_resident_mode = True
    supports_natural_mode = True

    def __init__(self):
        super().__init__()
        self.tags: list[str] = []
        self.shutdowns = 0

    def tag(self, name):
        self.tags.append(name)

    def run_generate(self, spec):
        out = super().run_generate(spec)
        # a cold process: step deltas past step 0 still carry compiles
        out["step_seconds"] = [2.0, 2.0, 2.0]
        out["e2e_breakdown"] = {"weights_load_total_s": 30.0}
        return out

    def run_request(self, spec, sync_steps=True):
        self.calls.append("request-synced" if sync_steps else "request-natural")
        if sync_steps:
            return {"wall_seconds": 10.0, "step_seconds": [0.2, 0.2], "step_basis": "synced",
                    "stage_seconds": {"denoise": 4.0}, "throughput_step_seconds": 0.2,
                    "output": {"finite": True}}
        return {"wall_seconds": 9.0, "step_seconds": [0.15, 0.15], "step_basis": "natural",
                "stage_seconds": {"denoise": 3.0}, "throughput_step_seconds": 0.15,
                "enqueue_step_seconds": [0.01, 0.01], "output": {"finite": True}}

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def no_drop(monkeypatch):
    dropped = []
    monkeypatch.setattr(bench, "drop_page_cache", lambda: dropped.append(1) or False)
    return dropped


def _run(monkeypatch, adapter, **kw):
    monkeypatch.setattr(bench, "_make_adapter", lambda name, save_dir=None: adapter)
    defaults = dict(skip_download=True, skip_compile=True, iters=3, natural_iters=2,
                    warm_discard=1, resident_iters=2, drop_caches=True)
    defaults.update(kw)
    return bench.run_one("qwen_image", "tpu", **defaults)


def test_order_cold_discard_warm_resident_natural(monkeypatch, no_drop):
    adapter = _ResidentAdapter()
    res = _run(monkeypatch, adapter)
    assert res.status == "ok", res.notes
    assert adapter.calls == (
        ["generate"]                      # cold
        + ["generate"]                    # discarded warm-up
        + ["generate"] * 3                # warm, fresh process each
        + ["request-synced"] * 2          # resident, synced
        + ["request-natural"] * 2         # resident, natural
    )
    assert adapter.tags == [
        "qwen_image_cold", "qwen_image_warmup0",
        "qwen_image_warm0", "qwen_image_warm1", "qwen_image_warm2",
        "qwen_image_resident0", "qwen_image_resident1",
        "qwen_image_natural0", "qwen_image_natural1",
    ]
    assert adapter.shutdowns == 1
    assert no_drop == [1]


def test_metric_sources(monkeypatch, no_drop):
    adapter = _ResidentAdapter()
    res = _run(monkeypatch, adapter)
    # cold = the first generate's wall; warm = the 3 reported fresh runs only
    assert res.e2e_cold_seconds == 99.0
    assert res.e2e_warm["n"] == 3
    assert res.e2e_warm["samples"] == [97.0, 96.0, 95.0]  # cold=99, discarded=98
    assert res.e2e_warm_breakdown == {"weights_load_total_s": 30.0}
    # per-step comes from the resident synced requests, never the cold process
    assert res.step_latency["n"] == 4
    assert res.step_latency["mean"] == pytest.approx(0.2)
    assert res.step_basis == "synced"
    assert res.throughput["steps/s"] == pytest.approx(5.0)
    assert res.e2e_warm_resident["n"] == 2 and res.e2e_warm_resident["mean"] == 10.0
    assert res.e2e_warm_natural["n"] == 2 and res.e2e_warm_natural["mean"] == 9.0
    assert res.step_latency_natural["mean"] == pytest.approx(0.15)
    # stage split + throughput from the last SYNCED request; the enqueue rate
    # (only meaningful unsynced) from the last natural one
    assert res.step_latency_alt == {"throughput": 0.2, "enqueue_mean": pytest.approx(0.01)}
    assert res.stage_seconds == {"denoise": 4.0}
    assert res.protocol == {
        "drop_page_cache_before_cold": True, "page_cache_dropped": False,
        "warm_discarded_runs": 1, "warm_iters": 3, "resident_iters": 2,
        "natural_iters": 2, "fresh_process_per_run": True,
    }
    assert any(n.startswith("e2e_cold = 99 s") and "sudo failed" in n for n in res.notes)
    assert any(n.startswith("e2e_warm = 96 s (n=3; reported after 1 discarded") for n in res.notes)


def test_fresh_only_adapter_keeps_trn2_sources(monkeypatch, no_drop):
    adapter = _FreshOnlyAdapter()
    res = _run(monkeypatch, adapter, iters=2, warm_discard=2, drop_caches=False)
    assert res.status == "ok", res.notes
    assert adapter.calls == ["generate"] * (1 + 2 + 2)
    assert no_drop == []                                  # --no-drop-caches honoured
    assert res.step_latency["n"] == 3                     # from the cold generate log
    assert res.e2e_warm["n"] == 2
    assert res.e2e_warm_resident is None and res.e2e_warm_natural is None
    assert res.protocol["resident_iters"] == 0 and res.protocol["natural_iters"] == 0
    assert any("cache not dropped (--no-drop-caches)" in n for n in res.notes)


def test_failure_is_recorded_and_workers_shut_down(monkeypatch, no_drop):
    class _Boom(_ResidentAdapter):
        def run_request(self, spec, sync_steps=True):
            raise RuntimeError("chip on fire")

    adapter = _Boom()
    res = _run(monkeypatch, adapter, iters=1)
    assert res.status == "failed"
    assert any("chip on fire" in n for n in res.notes)
    assert res.e2e_warm["n"] == 1                         # partial results kept
    assert adapter.shutdowns == 1
