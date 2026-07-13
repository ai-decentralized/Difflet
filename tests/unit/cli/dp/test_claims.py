from concurrent.futures import ThreadPoolExecutor

from difflet.cli.dp.claims import (
    claim_next,
    claimed_by,
    is_failed,
    mark_done,
    mark_failed,
    summarize,
)
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def _manifest(tmp_path, n, dp=None):
    reqs = [
        RequestSpec(index=i, prompt=f"p{i}", output=f"o{i}.png",
                    assigned_worker=(i % dp if dp else None))
        for i in range(n)
    ]
    write_manifest(reqs, tmp_path)
    return reqs


def test_round_robin_claims_only_own(tmp_path):
    _manifest(tmp_path, 5, dp=2)
    got = []
    while (req := claim_next(tmp_path, 0, "round_robin")) is not None:
        got.append(req.index)
    assert got == [0, 2, 4]
    assert claim_next(tmp_path, 0, "round_robin") is None
    assert [r.index for r in claimed_by(tmp_path, 0)] == [0, 2, 4]


def test_least_loaded_exhausts_queue(tmp_path):
    _manifest(tmp_path, 4)
    a = claim_next(tmp_path, 0, "least_loaded")
    b = claim_next(tmp_path, 1, "least_loaded")
    assert {a.index, b.index} == {0, 1}
    assert claim_next(tmp_path, 1, "least_loaded").index == 2
    assert claim_next(tmp_path, 0, "least_loaded").index == 3
    assert claim_next(tmp_path, 0, "least_loaded") is None


def test_least_loaded_claims_are_race_safe(tmp_path):
    _manifest(tmp_path, 64)

    def drain(w):
        got = []
        while (req := claim_next(tmp_path, w, "least_loaded")) is not None:
            got.append(req.index)
        return got

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(drain, range(8)))
    all_claims = [i for r in results for i in r]
    assert sorted(all_claims) == list(range(64))  # no dupes, no losses


def test_unknown_schedule_raises(tmp_path):
    import pytest

    _manifest(tmp_path, 1)
    with pytest.raises(ValueError, match="unknown schedule"):
        claim_next(tmp_path, 0, "fastest_first")


def test_markers_and_summary(tmp_path):
    _manifest(tmp_path, 3)
    for _ in range(3):
        claim_next(tmp_path, 0, "least_loaded")
    mark_done(tmp_path, 0)
    mark_failed(tmp_path, 1, "boom\ntrace")
    assert is_failed(tmp_path, 1) and not is_failed(tmp_path, 0)
    s = summarize(tmp_path)
    assert s.done == [0]
    assert s.failed == {1: "boom\ntrace"}
    assert s.unfinished == [2]
