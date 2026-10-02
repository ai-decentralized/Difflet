"""Per-step DiT timing: the two log lines the benchmark adapter parses, shared by every model."""

from __future__ import annotations

from difflet.pipeline.step_timing import DiTStepTimer, format_dit_step_report


def test_format_dit_step_report_matches_the_adapter_contract():
    report = format_dit_step_report("flux", [0.9, 0.5, 0.52])
    raw, summary = report.split("\n")
    assert raw == "[flux] dit-step-seconds: [0.9000, 0.5000, 0.5200]"
    assert summary.startswith("[flux] dit-step ms: n=2 mean=510.0 median=510.0 min=500.0 max=520.0")
    assert summary.endswith("(step 0 excluded)")
    assert format_dit_step_report("flux", [0.5]).endswith("no per-step stat)")
    assert format_dit_step_report("flux", []) == (
        "[flux] dit-step-seconds: []\n[flux] dit-step ms: n=0 (fewer than two DiT steps; no per-step stat)"
    )


def test_timer_records_one_sample_per_timed_step():
    timer = DiTStepTimer("qwen_image")
    for _ in range(3):
        with timer.step():
            pass
    assert len(timer.seconds) == 3 and all(s >= 0.0 for s in timer.seconds)
    assert timer.report().startswith("[qwen_image] dit-step-seconds: [")
    assert "[qwen_image] dit-step ms: n=2" in timer.report()


def test_adapter_parses_the_shared_report():
    from benchmark.adapters.trainium import parse_dit_step_seconds

    assert parse_dit_step_seconds(format_dit_step_report("ltx_2", [0.9, 0.5, 0.52])) == [0.5, 0.52]
