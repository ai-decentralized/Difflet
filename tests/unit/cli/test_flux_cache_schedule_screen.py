from __future__ import annotations

from pathlib import Path

import pytest

from scripts.flux_cache_protocol import load_prompt_suite
from scripts.flux_cache_schedule_screen import _paired_bootstrap_lower

ROOT = Path(__file__).resolve().parents[3]


def test_screen_prompt_suite_has_frozen_balanced_matrix():
    selection = load_prompt_suite(
        ROOT / "benchmark" / "flux_cache" / "derived-schedule-screen-prompt-suite.json",
        "derived_schedule_development_screen",
    )
    categories = {}
    for row in selection.descriptor["prompts"]:
        categories[row["category"]] = categories.get(row["category"], 0) + 1

    assert len(selection.prompts) == 32
    assert len(categories) == 8
    assert set(categories.values()) == {4}


def test_paired_bootstrap_lower_bound_is_deterministic_for_exact_ratios():
    comparator = [4.0, 6.0, 8.0, 10.0]

    equal = _paired_bootstrap_lower(
        comparator,
        comparator,
        repetitions=1000,
        seed=17,
        quantile=0.025,
    )
    twice_as_fast = _paired_bootstrap_lower(
        comparator,
        [value / 2.0 for value in comparator],
        repetitions=1000,
        seed=17,
        quantile=0.025,
    )

    assert equal == pytest.approx(1.0)
    assert twice_as_fast == pytest.approx(2.0)


def test_paired_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="equal and non-empty"):
        _paired_bootstrap_lower(
            [1.0, 2.0],
            [1.0],
            repetitions=100,
            seed=3,
            quantile=0.025,
        )
