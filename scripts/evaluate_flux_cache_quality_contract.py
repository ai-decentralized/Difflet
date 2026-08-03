#!/usr/bin/env python3
"""Apply a frozen automatic semantic-quality contract to cache comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_quality_contract import (
    evaluate_contract,
    evaluate_profile_holdout,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--semantic-report", required=True)
    parser.add_argument("--profile-holdout-registration", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    try:
        contract_path = Path(args.contract).expanduser().resolve()
        semantic_path = Path(args.semantic_report).expanduser().resolve()
        if args.profile_holdout_registration is None:
            document = evaluate_contract(contract_path, semantic_path)
            summaries = document["candidate_summaries"]
        else:
            document = evaluate_profile_holdout(
                Path(args.profile_holdout_registration).expanduser().resolve(),
                contract_path,
                semantic_path,
            )
            summaries = [document["candidate_summary"]]
        write_json(out, document)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    for row in summaries:
        print(
            "[automatic-quality] "
            f"{row['candidate_id']} failures={row['failure_count']}/{row['sample_count']} "
            f"upper={row['failure_rate_upper_bound']:.6f} "
            f"pass={row['passes_statistical_gate']}",
            flush=True,
        )
    print(f"[automatic-quality] report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
