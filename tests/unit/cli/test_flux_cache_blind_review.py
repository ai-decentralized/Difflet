from __future__ import annotations

import pytest

from scripts.prepare_flux_cache_blind_review import build_review_assignments


def _comparison(candidate_id: str, sample_id: str):
    return {
        "candidate_id": candidate_id,
        "sample_id": sample_id,
        "prompt_index": int(sample_id[1:4]),
        "seed": 0,
        "prompt": f"prompt {sample_id}",
        "baseline": {"image": f"baseline/{sample_id}.png"},
        "candidate": {"image": f"{candidate_id}/{sample_id}.png"},
    }


def test_blind_assignments_are_deterministic_and_hide_order():
    comparisons = [
        _comparison(candidate, f"p{index:03d}-s0")
        for index, candidate in enumerate(("candidate-a", "candidate-b", "candidate-c"))
    ]

    first = build_review_assignments(comparisons, seed=29)
    second = build_review_assignments(comparisons, seed=29)

    assert first == second
    assert {row["candidate_side"] for row in first}.issubset({"left", "right"})
    assert [row["pair_id"] for row in first] == ["pair-001", "pair-002", "pair-003"]
    assert {row["candidate_id"] for row in first} == {
        "candidate-a",
        "candidate-b",
        "candidate-c",
    }


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_blind_assignments_reject_invalid_seed(seed):
    with pytest.raises(ValueError, match="nonnegative integer"):
        build_review_assignments([_comparison("candidate", "p000-s0")], seed=seed)


def test_blind_assignments_reject_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        build_review_assignments([], seed=0)
