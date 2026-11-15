"""Reading the engine's timing events back: spans, steps per request, the stage breakdown."""
from __future__ import annotations

import json

from difflet.bench import timing as T


def _write(directory, name, events):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / name, "w") as handle:
        for e in events:
            handle.write(json.dumps(e) + "\n")


def _ev(kind, t, pid=10, rank=0, **fields):
    return {"event": kind, "t_ns": int(t * 1e9), "pid": pid, "rank": rank, "world": 2, **fields}


def test_spans_steps_forwards_and_breakdown(tmp_path):
    rank0 = [
        _ev("process", 0.0, argv=["x"], role="rank"),
        _ev("span", 1.0, name="load", phase="begin", stage="qwen"),
        _ev("span", 11.0, name="load", phase="end", ok=True, stage="qwen"),
        _ev("span", 12.0, name="request", phase="begin", stage="qwen", request=0),
        _ev("span", 13.0, name="denoise", phase="begin", stage="qwen"),
        _ev("step", 13.5, index=0), _ev("forward", 13.4, seconds=0.4),
        _ev("step", 14.0, index=1), _ev("step", 14.5, index=2),
        _ev("span", 14.6, name="denoise", phase="end", ok=True, stage="qwen"),
        _ev("span", 15.0, name="request", phase="end", ok=True, stage="qwen", request=0),
        _ev("evidence", 15.1, stage="qwen", phase_seconds={"warmup forward": 2.5}),
    ]
    rank1 = [_ev("span", 12.0, pid=11, rank=1, name="request", phase="begin", stage="qwen",
                 request=0),
             _ev("step", 13.6, pid=11, rank=1, index=0),
             _ev("step", 14.2, pid=11, rank=1, index=1),
             _ev("span", 15.0, pid=11, rank=1, name="request", phase="end", ok=True,
                 stage="qwen", request=0)]
    driver = [_ev("args", 0.5, pid=9, args={"tp_degree": 4})]
    _write(tmp_path, "rank0-10.jsonl", rank0)
    _write(tmp_path, "rank1-11.jsonl", rank1)
    _write(tmp_path, "proc-9.jsonl", driver)
    tl = T.load(tmp_path)
    assert tl.args() == {"tp_degree": 4}
    assert tl.ranks() == [0, 1]
    steps = tl.steps(rank=0)
    assert len(steps) == 1 and len(steps[0]) == 3
    assert [round(d, 6) for d in T.deltas(steps[0])] == [0.5, 0.5]
    assert [round(d, 6) for d in T.deltas(tl.steps(rank=1)[0])] == [0.6]
    assert tl.forwards(rank=0) == [[0.4]]
    assert tl.stage_of(10) == "qwen"
    parts = T.stage_breakdown(tl)
    assert parts["qwen: load"] == 10.0
    assert parts["qwen: request"] == 3.0
    assert parts["qwen: before denoise"] == 1.0
    assert round(parts["qwen: denoise"], 6) == 1.6
    assert round(parts["qwen: after denoise"], 6) == 0.4
    assert parts["qwen: load includes warmup forward"] == 2.5


def test_a_torn_last_line_is_skipped(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "rank0-1.jsonl").write_text(json.dumps(_ev("step", 1.0, index=0)) + "\n{\"ev")
    assert len(T.load(tmp_path).events) == 1


def test_an_unclosed_span_has_no_seconds(tmp_path):
    _write(tmp_path, "a.jsonl", [_ev("span", 1.0, name="request", phase="begin", request=0)])
    span = T.load(tmp_path).spans_named("request")[0]
    assert span.end_ns is None and span.seconds is None
