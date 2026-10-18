import json

import pytest

from difflet.cli.dp.requests_io import (
    RequestSpec,
    load_requests_jsonl,
    read_manifest,
    request_path,
    write_manifest,
)


def _write_jsonl(tmp_path, lines):
    p = tmp_path / "requests.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return p


def test_load_minimal_jsonl(tmp_path):
    p = _write_jsonl(
        tmp_path,
        [
            {"prompt": "a cat", "output": "a.png"},
            {
                "prompt": "a dog",
                "output": "b.png",
                "seed": 7,
                "steps": 12,
                "guidance_scale": 5.0,
                "negative_prompt": "blurry",
            },
        ],
    )
    reqs = load_requests_jsonl(p)
    assert [r.index for r in reqs] == [0, 1]
    assert reqs[0].seed == 42 and reqs[0].steps is None
    assert reqs[1].seed == 7 and reqs[1].guidance_scale == 5.0
    assert reqs[1].negative_prompt == "blurry"


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text(
        '{"prompt": "a", "output": "a.png"}\n\n\n{"prompt": "b", "output": "b.png"}\n',
        encoding="utf-8",
    )
    assert [r.index for r in load_requests_jsonl(p)] == [0, 1]


def test_load_rejects_duplicate_outputs(tmp_path):
    p = _write_jsonl(
        tmp_path,
        [
            {"prompt": "x", "output": "same.png"},
            {"prompt": "y", "output": "same.png"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate output"):
        load_requests_jsonl(p)


def test_load_rejects_missing_fields_and_empty(tmp_path):
    with pytest.raises(ValueError, match="line 1"):
        load_requests_jsonl(_write_jsonl(tmp_path, [{"prompt": "no output"}]))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no requests"):
        load_requests_jsonl(empty)


def test_load_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"prompt": "ok", "output": "a.png"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_requests_jsonl(p)


def test_manifest_roundtrip(tmp_path):
    reqs = [
        RequestSpec(index=0, prompt="a", output="a.png", assigned_worker=0),
        RequestSpec(index=1, prompt="b", output="b.png", assigned_worker=1),
    ]
    write_manifest(reqs, tmp_path)
    assert request_path(tmp_path, 1).name == "req_0001.json"
    back = read_manifest(tmp_path)
    assert back == reqs


def test_write_manifest_starts_from_an_empty_queue(tmp_path):
    # The requests dir IS the queue: markers from an earlier run (claim/done/failed) and a
    # longer earlier manifest made a second `--dp N --requests` over the same work dir report
    # done=N in seconds without generating (pod f2b, 2026-10-18, F2b observation).
    from difflet.cli.dp.claims import summarize

    stale = [RequestSpec(index=i, prompt=f"old{i}", output=f"old{i}.png") for i in range(3)]
    write_manifest(stale, tmp_path)
    (tmp_path / "req_0000.claim").write_text("0")
    (tmp_path / "req_0000.done").touch()
    (tmp_path / "req_0001.failed").write_text("boom")
    (tmp_path / "unrelated.txt").write_text("keep me")

    fresh = [RequestSpec(index=i, prompt=f"new{i}", output=f"new{i}.png") for i in range(2)]
    write_manifest(fresh, tmp_path)

    assert [r.prompt for r in read_manifest(tmp_path)] == ["new0", "new1"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "req_0000.json",
        "req_0001.json",
        "unrelated.txt",
    ]
    s = summarize(tmp_path)
    assert s.unfinished == [0, 1] and not s.done and not s.failed
