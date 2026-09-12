"""Re-parse a cell's saved cold/warm generate logs into its e2e breakdowns.

    python -m benchmark.reparse_e2e --config tp4 [--model flux_1_dev ...]

For each (model, config) cell with a result JSON, parse
benchmark/<device>/logs/{cold,warm}/<stem>_generate.log with parse_generate
(wall = the JSON's recorded e2e_cold_seconds / e2e_warm.mean, which are
authoritative), relabel the stages as the MATRIX entry says, and patch
e2e_breakdown / e2e_warm_breakdown / load_seconds; re-render the md.
Idempotent; needed when the parser learns a new log line after a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.adapters.trainium import spec_slug
from benchmark.cell_status import CAMPAIGN_MODELS
from benchmark.models import CONFIGS, UNSUPPORTED, json_path, logs_dir, report_path, resolve
from benchmark.parse_generate import parse, relabel


def reparse(slug: str, config: str) -> str:
    cfg = resolve(slug, config)
    jp = Path(json_path(cfg.config_slug))
    if not jp.exists():
        return "no result file"
    d = json.loads(jp.read_text())
    if d.get("status") == "skipped":
        return "N/A"
    stem = spec_slug(cfg)
    changed = []
    for kind, wall in (("cold", d.get("e2e_cold_seconds")),
                       ("warm", (d.get("e2e_warm") or {}).get("mean"))):
        log = Path(logs_dir()) / kind / f"{stem}_generate.log"
        if not log.exists() or wall is None:
            continue
        eb = relabel(parse(log.read_text(errors="ignore"), wall),
                     cfg.stage_names, cfg.e2e_host_note)
        if not eb.get("stages"):
            continue
        key = "e2e_breakdown" if kind == "cold" else "e2e_warm_breakdown"
        d[key] = eb
        if kind == "cold":
            d["load_seconds"] = eb["weights_load_total_s"]
        changed.append(f"{kind} load {eb['weights_load_total_s']:.1f}s/{len(eb['stages'])} stages")
    if changed:
        jp.write_text(json.dumps(d, indent=2))
        Path(report_path(cfg.config_slug)).write_text(report.render(d))
    return "; ".join(changed) or "no load lines found"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tp4", choices=sorted(CONFIGS))
    p.add_argument("--model", nargs="*", default=None)
    a = p.parse_args()
    for slug in a.model or CAMPAIGN_MODELS:
        if (slug, a.config) in UNSUPPORTED:
            continue
        print(f"[reparse] {slug}/{a.config}: {reparse(slug, a.config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
