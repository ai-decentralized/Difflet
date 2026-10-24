from __future__ import annotations

import json

from difflet.utils import step_timing
from difflet.utils.step_timing import StepTimer


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("DIFFLET_STEP_TIMES_FILE", raising=False)
    assert StepTimer.from_env() is None


def test_records_step_and_forward_seconds(monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("DIFFLET_STEP_TIMES_FILE", str(path))
    clock = iter([10.0, 10.5, 11.25, 11.75, 12.0, 12.5, 13.0])
    monkeypatch.setattr(step_timing.time, "perf_counter", lambda: next(clock))

    timer = StepTimer.from_env()  # loop entry at t=10.0
    with timer.forward():  # 10.5 -> 11.25
        pass
    timer.step_done(0)  # 11.75
    with timer.forward():  # 12.0 -> 12.5
        pass
    timer.step_done(1)  # 13.0

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        {"step": 0, "step_s": 1.75, "forward_s": 0.75},
        {"step": 1, "step_s": 1.25, "forward_s": 0.5},
    ]


def test_appends_across_requests(monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("DIFFLET_STEP_TIMES_FILE", str(path))
    for _ in range(2):
        timer = StepTimer.from_env()
        timer.step_done(0)
    assert len(path.read_text().splitlines()) == 2
