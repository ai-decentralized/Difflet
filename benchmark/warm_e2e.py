"""Measure e2e_warm — a steady-state generate with the OS page cache warm.

The difflet CLI starts a fresh process per generate, so it always reloads weights;
e2e is load-dominated and that load swings wildly with OS page-cache warmth (a
single fresh run can be SLOWER than the recorded cold run — observed LTX-2 764 s
when the cache was evicted). So "warm" must mean the cache is genuinely warm: we
run ``--warmups`` discarded cache-warming generates first, then ``--iters``
measured ones, and report only the measured runs. Host RAM (124 GB, ~100 GB page
cache) holds a single model's weights (e.g. Qwen 54 GB), so the warm run reads
from cache. Writes e2e_warm into benchmark/<device>/<slug>.json and re-renders the
md; uses an isolated log dir so the cold-run logs (and their breakdowns) survive.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.warm_e2e --model qwen_image [--warmups 1] [--iters 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.adapters.trainium import TrainiumAdapter
from benchmark.harness import Stats
from benchmark.models import MATRIX, json_path, report_path, logs_dir


def run(slug: str, warmups: int, iters: int) -> int:
    cfg = MATRIX[slug]
    ad = TrainiumAdapter(log_dir=f"{logs_dir()}/warm")
    for i in range(warmups):  # discarded — purpose is to populate the OS page cache
        g = ad.run_generate(cfg)
        print(f"[warm] {slug} cache-warmer {i+1}/{warmups}: {g['wall_seconds']:.1f}s "
              f"(load {g.get('load_seconds')}) [discarded]", flush=True)
    samples: list[float] = []
    last: dict = {}
    for i in range(iters):
        last = ad.run_generate(cfg)
        samples.append(last["wall_seconds"])
        print(f"[warm] {slug} iter {i+1}/{iters}: {last['wall_seconds']:.1f}s "
              f"(load {last.get('load_seconds')})", flush=True)
    st = Stats.from_samples(samples)
    jp = Path(json_path(slug))
    d = json.loads(jp.read_text())
    d["e2e_warm"] = st.__dict__
    if last.get("e2e_breakdown"):
        d["e2e_warm_breakdown"] = last["e2e_breakdown"]
    d.setdefault("notes", []).append(
        f"e2e_warm = {st.mean:.0f} s (n={st.n}; reported after {warmups} discarded "
        "cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads "
        "weights every process, so 'warm' = warm disk cache -> faster load, not a "
        "resident model; cf. e2e cold and the load/compute breakdown.")
    jp.write_text(json.dumps(d, indent=2))
    Path(report_path(slug)).write_text(report.render(d))
    print(f"[warm] {slug}: mean {st.mean:.1f}s median {st.median:.1f}s n={st.n} -> patched", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--iters", type=int, default=1)
    a = p.parse_args()
    return run(a.model, a.warmups, a.iters)


if __name__ == "__main__":
    sys.exit(main())
