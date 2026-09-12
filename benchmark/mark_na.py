"""Record a by-design unsupported (model, config) cell as a result file.

The campaign driver calls this for every cell in ``benchmark.models.UNSUPPORTED``
so the report has an explicit N/A row with the reason, instead of a missing file
that could be read as "not measured yet".

    python -m benchmark.mark_na --model ltx_2 --config tp2cp2

Writes benchmark/<device>/<slug>_<config>.json (status="skipped", skip_reason)
and renders the md. Refuses to overwrite a measured (status != skipped) file.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.models import (DEVICE, UNSUPPORTED, add_config_arg, json_path,
                              report_path, resolve, results_dir)


def skipped_record(slug: str, config: str, reason: str) -> dict:
    cfg = resolve(slug, config)
    return {
        "model_id": cfg.model_id, "model_type": cfg.model_type,
        "backend": "trainium", "device": "", "dtype": cfg.dtype,
        "parallel": cfg.parallel_dict(),
        "shape": {"height": cfg.height, "width": cfg.width, "num_frames": cfg.num_frames},
        "steps": cfg.steps,
        "status": "skipped", "skip_reason": reason,
        "config_slug": cfg.config_slug, "model_slug": slug, "config": config,
        "device_slug": DEVICE, "revision": cfg.revision, "seed": cfg.seed,
        "prompt": cfg.prompt, "guidance_scale": cfg.guidance_scale,
        "output_kind": cfg.output_kind, "config_label": cfg.config_label,
        "toolchain": {}, "throughput": {}, "compile_breakdown": {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%d %H:%M UTC"),
        "notes": [f"N/A by design ({config}): {reason}"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    add_config_arg(p)
    p.add_argument("--force", action="store_true",
                   help="overwrite even if a measured result exists")
    a = p.parse_args()
    key = (a.model, a.config)
    if key not in UNSUPPORTED:
        print(f"[mark_na] {a.model}/{a.config} is not in UNSUPPORTED; refusing to "
              "declare it N/A (measure it instead)", file=sys.stderr)
        return 2
    cfg = resolve(a.model, a.config)
    jp = Path(json_path(cfg.config_slug))
    if jp.exists() and not a.force:
        cur = json.loads(jp.read_text())
        if cur.get("status") not in (None, "skipped"):
            print(f"[mark_na] {jp} holds a status={cur.get('status')!r} result; "
                  "not overwriting (use --force)", file=sys.stderr)
            return 3
    Path(results_dir()).mkdir(parents=True, exist_ok=True)
    d = skipped_record(a.model, a.config, UNSUPPORTED[key])
    jp.write_text(json.dumps(d, indent=2))
    Path(report_path(cfg.config_slug)).write_text(report.render(d))
    print(f"[mark_na] {cfg.config_slug}: status=skipped -> {jp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
