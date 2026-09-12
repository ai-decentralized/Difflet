"""Is a (model, config) cell complete? The campaign driver's resume check.

    python -m benchmark.cell_status --model flux_1_dev --config tp2cp2
      -> exit 0 and "complete" when all four metrics are present and sane,
         exit 1 and the list of what is missing otherwise;
    python -m benchmark.cell_status --summary [--config tp4]
      -> one line per cell of the campaign matrix.

"Complete" means, in the cell's result JSON:
  compile_seconds > 0, e2e_cold_seconds, e2e_warm.mean, and step_latency with
  n >= steps - 1 samples (the real-loop rule: step 0 excluded), all finite;
or status == "skipped" (a by-design N/A cell written by benchmark.mark_na).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from benchmark.models import CONFIGS, MATRIX, UNSUPPORTED, json_path, resolve

CAMPAIGN_MODELS = ["flux_1_dev", "qwen_image", "ltx_2", "hunyuan_video", "wan_2_1"]


def _pos(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def missing(slug: str, config: str) -> list[str]:
    """What a cell still lacks; empty list == complete."""
    cfg = resolve(slug, config)
    jp = Path(json_path(cfg.config_slug))
    if not jp.exists():
        return ["result file"]
    d = json.loads(jp.read_text())
    if "config" not in d:
        # A pre-campaign file (the historical tp4 <slug>.json has no `config`
        # field) is history, not a measurement of this campaign: bench
        # --compile-only rewrites it wholesale and the later steps patch that.
        return ["result file (pre-campaign history, not this run)"]
    if d.get("status") == "skipped":
        return []
    out: list[str] = []
    if not _pos(d.get("compile_seconds")):
        out.append("compile_seconds")
    if not _pos(d.get("e2e_cold_seconds")):
        out.append("e2e_cold_seconds")
    if not _pos((d.get("e2e_warm") or {}).get("mean")):
        out.append("e2e_warm")
    st = d.get("step_latency") or {}
    if not _pos(st.get("mean")):
        out.append("step_latency")
    elif int(st.get("n") or 0) < cfg.steps - 1:
        out.append(f"step_latency.n={st.get('n')} < {cfg.steps - 1}")
    if d.get("status") not in ("ok", "compiled"):
        out.append(f"status={d.get('status')!r}")
    # the parallel record must be the topology the label names
    want = cfg.parallel_dict()
    got = d.get("parallel") or {}
    if any(got.get(k) != v for k, v in want.items()):
        out.append(f"parallel mismatch: {got} != {want}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--config", default=None, choices=sorted(CONFIGS))
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        configs = [a.config] if a.config else list(CONFIGS)
        for config in configs:
            for slug in CAMPAIGN_MODELS:
                if (slug, config) in UNSUPPORTED:
                    tag = "N/A" if not missing(slug, config) else "N/A (unrecorded)"
                else:
                    m = missing(slug, config)
                    tag = "complete" if not m else "missing: " + ", ".join(m)
                print(f"{config:7s} {slug:15s} {tag}")
        return 0
    if not a.model:
        raise SystemExit("--model is required without --summary")
    m = missing(a.model, a.config or "tp4")
    if m:
        print(f"[cell_status] {a.model}/{a.config or 'tp4'} missing: {', '.join(m)}")
        return 1
    print(f"[cell_status] {a.model}/{a.config or 'tp4'} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
