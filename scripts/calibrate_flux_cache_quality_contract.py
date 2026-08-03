#!/usr/bin/env python3
"""Freeze automatic semantic margins from normal no-cache seed variation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_quality_contract import calibrate_contract, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--semantic-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    try:
        document = calibrate_contract(
            Path(args.protocol).expanduser().resolve(),
            Path(args.semantic_report).expanduser().resolve(),
        )
        write_json(out, document)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        "[automatic-quality] margins "
        f"ImageReward={document['margins']['image_reward']:.6f} "
        f"VQAScore={document['margins']['vqa_score']:.6f} -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
