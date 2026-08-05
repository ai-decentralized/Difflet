#!/usr/bin/env python3
"""Collect the registered source profiles with neutral artifact names."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_flux_cache_ab import _parse_args, collect  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        quality_path, _ = collect(args)
        neutral_path = quality_path.with_name("quality-input.json")
        quality_path.replace(neutral_path)
        print(f"[phase-schedule-source] quality manifest: {neutral_path}", flush=True)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
