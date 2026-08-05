#!/usr/bin/env python3
"""Collect the exact arm matrix from a frozen derived-schedule screen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_flux_cache_ab as base_collector  # noqa: E402
from scripts.flux_cache_schedule_screen import (  # noqa: E402
    load_registered_arms,
    load_registration,
)

FOREGROUND_ACK = "I am running the registered derived schedule screen"


def collect(args: argparse.Namespace) -> tuple[Path, Path]:
    registration_path = Path(args.registration).expanduser().resolve()
    registration = load_registration(registration_path)
    if not args.allow_hardware or args.foreground_ack != FOREGROUND_ACK:
        raise ValueError("registered screen hardware acknowledgement is missing")
    collection = registration["collection"]
    base_args = base_collector._parse_args(
        [
            "--out-dir",
            collection["output_directory"],
            "--model-id",
            registration["controlled_generation"]["model_id"],
            "--model-revision",
            registration["controlled_generation"]["model_revision"],
            "--prompt-suite",
            registration["prompt_suite"]["path"],
            "--prompt-split",
            registration["prompt_suite"]["split"],
            "--seed",
            str(registration["prompt_suite"]["seeds"][0]),
            "--num-steps",
            str(registration["controlled_generation"]["num_steps"]),
            "--height",
            str(registration["controlled_generation"]["height"]),
            "--width",
            str(registration["controlled_generation"]["width"]),
            "--guidance-scale",
            str(registration["controlled_generation"]["guidance_scale"]),
            "--tp-degree",
            str(registration["controlled_generation"]["tp_degree"]),
            "--dtype",
            registration["controlled_generation"]["dtype"],
            "--allow-hardware",
            "--foreground-ack",
            base_collector.FOREGROUND_ACK,
        ]
    )
    arms = load_registered_arms(registration)
    original_selector = base_collector.select_candidate_arms
    base_collector.select_candidate_arms = lambda _args: arms
    try:
        return base_collector.collect(base_args)
    finally:
        base_collector.select_candidate_arms = original_selector


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        collect(_parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
