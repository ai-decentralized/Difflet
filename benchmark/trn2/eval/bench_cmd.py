#!/usr/bin/env python
"""Print the exact `difflet compile` / `difflet generate` command the benchmark
harness runs for <model> <config>, optionally against another cache dir.

    python benchmark/trn2/eval/bench_cmd.py generate wan_2_1 tp4 --cache-dir D --output O
    python benchmark/trn2/eval/bench_cmd.py compile wan_2_1 tp4 --cache-dir D

Used by the eval queues for runs the harness has no switch for (a cache clone
without presharded weights, a compile into empty caches).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("compile", "generate"))
    p.add_argument("model")
    p.add_argument("config")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()

    from benchmark.adapters.trainium import TrainiumAdapter
    from benchmark.models import resolve

    cfg = resolve(a.model, a.config)
    ad = TrainiumAdapter(log_dir=str(Path("/tmp") / "bench_cmd_logs"))
    if a.cache_dir:
        ad.cache_dir = a.cache_dir
    captured: dict = {}

    class _Captured(Exception):
        pass

    def fake_run(cmd, log, timeout=0):
        captured["cmd"] = cmd
        raise _Captured

    ad._run = fake_run
    try:
        (ad.compile if a.kind == "compile" else ad.run_generate)(cfg)
    except _Captured:
        pass
    cmd = list(captured["cmd"])
    if a.output and "--output" in cmd:
        cmd[cmd.index("--output") + 1] = a.output
    print(shlex.join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
