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
    p = _write_jsonl(tmp_path, [
        {"prompt": "a cat", "output": "a.png"},
        {"prompt": "a dog", "output": "b.png", "seed": 7, "steps": 12,
         "guidance_scale": 5.0, "negative_prompt": "blurry"},
    ])
    reqs = load_requests_jsonl(p)
    assert [r.index for r in reqs] == [0, 1]
    assert reqs[0].seed == 42 and reqs[0].steps is None
    assert reqs[1].seed == 7 and reqs[1].guidance_scale == 5.0
    assert reqs[1].negative_prompt == "blurry"


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text('{"prompt": "a", "output": "a.png"}\n\n\n{"prompt": "b", "output": "b.png"}\n',
                 encoding="utf-8")
    assert [r.index for r in load_requests_jsonl(p)] == [0, 1]


def test_load_rejects_duplicate_outputs(tmp_path):
    p = _write_jsonl(tmp_path, [
        {"prompt": "x", "output": "same.png"},
        {"prompt": "y", "output": "same.png"},
    ])
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
