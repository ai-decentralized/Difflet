import argparse
from pathlib import Path

import pytest

from difflet.cli.dp import stage_loop
from difflet.cli.dp.claims import claim_next, mark_failed, summarize
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def _batch_args(tmp_path, worker=0, schedule="least_loaded"):
    return argparse.Namespace(
        requests_dir=str(tmp_path / "requests"), worker_index=worker,
        dp_schedule=schedule, work_dir=str(tmp_path / "work"),
        prompt=None, output=None, seed=42, steps=None, guidance_scale=None,
    )


def _legacy_args(tmp_path):
    return argparse.Namespace(
        requests_dir=None, worker_index=None, dp_schedule=None,
        work_dir=str(tmp_path / "work"), prompt="a cat", output="cat.png",
        seed=7, steps=3, guidance_scale=None,
    )


def test_legacy_single_request_and_paths(tmp_path):
    args = _legacy_args(tmp_path)
    assert not stage_loop.batch_mode(args)
    reqs = list(stage_loop.claim_requests(args))
    assert len(reqs) == 1
    assert reqs[0].prompt == "a cat" and reqs[0].seed == 7 and reqs[0].steps == 3
    assert stage_loop.work_file(args, reqs[0], "latents.pt") == Path(args.work_dir) / "latents.pt"


def test_legacy_scope_propagates_exceptions(tmp_path):
    args = _legacy_args(tmp_path)
    req = next(iter(stage_loop.claim_requests(args)))
    with pytest.raises(RuntimeError, match="boom"):
        with stage_loop.request_scope(args, req, final=True):
            raise RuntimeError("boom")


def test_batch_claims_and_marks(tmp_path):
    write_manifest([RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png")
                    for i in range(3)], tmp_path / "requests")
    args = _batch_args(tmp_path)
    seen = []
    for req in stage_loop.claim_requests(args):
        with stage_loop.request_scope(args, req, final=True):
            if req.index == 1:
                raise RuntimeError("bad request")
            seen.append(req.index)
    assert seen == [0, 2]
    s = summarize(tmp_path / "requests")
    assert s.done == [0, 2] and 1 in s.failed and "bad request" in s.failed[1]


def test_batch_nonfinal_scope_does_not_mark_done(tmp_path):
    write_manifest([RequestSpec(index=0, prompt="p", output="o.png")], tmp_path / "requests")
    args = _batch_args(tmp_path)
    for req in stage_loop.claim_requests(args):
        with stage_loop.request_scope(args, req, final=False):
            pass
    s = summarize(tmp_path / "requests")
    assert s.done == [] and s.unfinished == [0]


def test_claimed_requests_replays_claims_skipping_failed(tmp_path):
    write_manifest([RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png")
                    for i in range(4)], tmp_path / "requests")
    args0, args1 = _batch_args(tmp_path, 0), _batch_args(tmp_path, 1)
    claim_next(args0.requests_dir, 0, "least_loaded")   # req 0 -> worker 0
    claim_next(args1.requests_dir, 1, "least_loaded")   # req 1 -> worker 1
    claim_next(args0.requests_dir, 0, "least_loaded")   # req 2 -> worker 0
    mark_failed(args0.requests_dir, 2, "earlier stage failed")
    assert [r.index for r in stage_loop.claimed_requests(args0)] == [0]
    assert [r.index for r in stage_loop.claimed_requests(args1)] == [1]


def test_batch_work_file_is_request_scoped(tmp_path):
    args = _batch_args(tmp_path)
    req = RequestSpec(index=3, prompt="p", output="o.png")
    assert stage_loop.work_file(args, req, "latents.pt") == (
        Path(args.work_dir) / "latents_req0003.pt"
    )


def test_effective_override_chain(tmp_path):
    args = argparse.Namespace(steps=10)
    assert stage_loop.effective(RequestSpec(0, "p", "o", steps=5), args, "steps", 2) == 5
    assert stage_loop.effective(RequestSpec(0, "p", "o"), args, "steps", 2) == 10
    assert stage_loop.effective(None, argparse.Namespace(steps=None), "steps", 2) == 2
