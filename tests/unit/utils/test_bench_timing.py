"""The engine's opt-in bench timing events."""
from __future__ import annotations

import json

import pytest

from difflet.utils import bench_timing as T


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(T, "_handle", None)
    monkeypatch.setattr(T, "_handle_pid", None)
    monkeypatch.setattr(T, "_failed", False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)


def _events(directory):
    lines = []
    for path in sorted(directory.glob("*.jsonl")):
        lines += [json.loads(line) for line in path.read_text().splitlines()]
    return lines


def test_off_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(T.TIMING_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    T.event("step", index=0)
    with T.span("load"):
        pass
    T.evidence(exec_mode="compile")
    assert list(tmp_path.iterdir()) == []


def test_events_spans_steps_and_forwards(monkeypatch, tmp_path):
    monkeypatch.setenv(T.TIMING_ENV, str(tmp_path))
    with T.span("request", request=0):
        T.new_request()
        seen = []
        step = T.step_hook(lambda: seen.append("prev"))
        forward = T.forward_hook()
        forward(0.25)
        step()
        step()
    events = _events(tmp_path)
    kinds = [e["event"] for e in events]
    assert kinds == ["process", "span", "forward", "step", "step", "span"]
    assert [e["index"] for e in events if e["event"] == "step"] == [0, 1]
    assert seen == ["prev", "prev"]
    begin, end = events[1], events[-1]
    assert begin["phase"] == "begin" and end["phase"] == "end" and end["ok"] is True
    assert all(e["t_ns"] >= events[0]["t_ns"] for e in events)
    assert events[2]["seconds"] == 0.25


def test_span_end_is_written_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv(T.TIMING_ENV, str(tmp_path))
    with pytest.raises(RuntimeError):
        with T.span("denoise"):
            raise RuntimeError("boom")
    end = [e for e in _events(tmp_path) if e["event"] == "span"][-1]
    assert end["phase"] == "end" and end["ok"] is False


def test_a_write_failure_disables_timing_instead_of_failing_the_run(monkeypatch, tmp_path):
    target = tmp_path / "not-a-dir"
    target.write_text("x")
    monkeypatch.setenv(T.TIMING_ENV, str(target))
    T.event("step", index=0)  # must not raise
    assert T._failed is True


def test_output_records_the_file_hash(monkeypatch, tmp_path):
    monkeypatch.setenv(T.TIMING_ENV, str(tmp_path / "t"))
    out = tmp_path / "out.png"
    out.write_bytes(b"abc")
    T.output(str(out), "image")
    record = [e for e in _events(tmp_path / "t") if e["event"] == "output"][0]
    assert record["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
