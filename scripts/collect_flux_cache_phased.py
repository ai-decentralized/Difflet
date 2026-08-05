#!/usr/bin/env python3
"""Collect FLUX A/B artifacts for frozen phase-aware cache candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_flux_cache_ab as base_collector
from scripts.flux_cache_phased_candidate import load_phased_candidates


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    phase_parser = argparse.ArgumentParser(add_help=False)
    phase_parser.add_argument(
        "--phased-candidate",
        action="append",
        required=True,
        help="frozen phase-aware candidate JSON; repeat to compare candidates",
    )
    phase_args, remaining = phase_parser.parse_known_args(argv)
    args = base_collector._parse_args(remaining)
    if (
        args.candidate_ladder is not None
        or args.adaptive_candidate
        or args.adaptive_only
        or args.warmup_steps is not None
        or args.anchor_intervals is not None
        or args.orders is not None
        or args.coord is not None
    ):
        raise ValueError(
            "phase-aware collection cannot be combined with legacy candidate flags"
        )
    args.phased_candidate = tuple(phase_args.phased_candidate)
    return args


def collect(args: argparse.Namespace) -> tuple[Path, Path]:
    """Reuse the frozen A/B execution path with an independently loaded arm set."""

    arms = load_phased_candidates(args.phased_candidate)
    original_selector = base_collector.select_candidate_arms
    base_collector.select_candidate_arms = lambda _args: arms
    try:
        return base_collector.collect(args)
    finally:
        base_collector.select_candidate_arms = original_selector


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        collect(args)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
