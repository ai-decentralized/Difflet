"""Measure a TRUE cold-start e2e and a warm e2e for a model, back to back.

Per model:
  1. drop the OS page cache (sudo sync + /proc/sys/vm/drop_caches) so the load
     genuinely reads weights from disk -> records e2e_cold (real cold start).
  2. immediately run again with the cache now hot -> records e2e_warm.

Both are measured in the same session under controlled cache state, so the
cold>warm ordering is honest (the previously-stored e2e_cold was a lucky run with
the host weights already cached -> 8 s host load instead of a true cold read).
Patches benchmark/<device>/<slug>.json after EACH run (real-time) and re-renders
the md. Cold/warm use separate log dirs so their breakdowns don't clobber.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.cold_warm_e2e --model ltx_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.adapters.trainium import TrainiumAdapter
from benchmark.harness import Stats, drop_page_cache
from benchmark.models import MATRIX


def drop_caches() -> bool:
    return drop_page_cache()


def _patch(slug: str, note: str | None = None, **kv) -> dict:
    """Update result JSON fields, drop any stale e2e_cold/e2e_warm note, optionally
    append a fresh one, persist, and re-render the md."""
    from benchmark.models import json_path
    jp = Path(json_path(slug))
    d = json.loads(jp.read_text())
    d.update(kv)
    notes = [n for n in d.get("notes", []) if not n.startswith("FAILED:")]
    if note:
        # only drop a prior note of the SAME kind, so cold + warm notes coexist
        pref = "e2e_cold =" if note.startswith("e2e_cold") else \
               "e2e_warm =" if note.startswith("e2e_warm") else None
        if pref:
            notes = [n for n in notes if not n.startswith(pref)]
        notes.append(note)
    d["notes"] = notes
    jp.write_text(json.dumps(d, indent=2))
    from benchmark.models import report_path
    Path(report_path(slug)).write_text(report.render(d))
    return d


def run(slug: str) -> int:
    from benchmark.models import logs_dir
    cfg = MATRIX[slug]
    cold_ad = TrainiumAdapter(log_dir=f"{logs_dir()}/cold")
    warm_ad = TrainiumAdapter(log_dir=f"{logs_dir()}/warm")

    print(f"[cold] {slug}: dropping page cache...", flush=True)
    cold = drop_caches()
    if not cold:
        print(f"[cold] {slug}: WARNING drop_caches failed (no sudo) -> not a true cold start",
              flush=True)
    gc = cold_ad.run_generate(cfg)
    print(f"[cold] {slug}: {gc['wall_seconds']:.1f}s (load {gc.get('load_seconds')})", flush=True)
    _patch(slug,
           note=(f"e2e_cold = {gc['wall_seconds']:.0f} s — "
                 + ("TRUE cold start (OS page cache dropped before the run), so the "
                    "weight load is a real cold disk read." if cold else
                    "cache NOT dropped (sudo failed) — not a guaranteed cold start.")),
           e2e_cold_seconds=round(gc["wall_seconds"], 3),
           load_seconds=gc.get("load_seconds"),
           e2e_breakdown=gc.get("e2e_breakdown"),
           output=gc.get("output"),
           status="ok")

    gw = warm_ad.run_generate(cfg)
    print(f"[warm] {slug}: {gw['wall_seconds']:.1f}s (load {gw.get('load_seconds')})", flush=True)
    st = Stats.from_samples([gw["wall_seconds"]])
    _patch(slug,
           note=(f"e2e_warm = {gw['wall_seconds']:.0f} s (n=1, warm OS page cache from "
                 f"the immediately-preceding cold run; same session as the "
                 f"{gc['wall_seconds']:.0f} s cold start). difflet reloads weights every "
                 "process, so warm = warm disk cache -> faster load, not a resident model."),
           e2e_warm=st.__dict__,
           e2e_warm_breakdown=gw.get("e2e_breakdown"))

    print(f"[done] {slug}: cold {gc['wall_seconds']:.0f}s -> warm {gw['wall_seconds']:.0f}s", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    a = p.parse_args()
    return run(a.model)


if __name__ == "__main__":
    sys.exit(main())
