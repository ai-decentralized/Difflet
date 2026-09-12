"""Measure e2e_warm — a steady-state generate with the cache warm — for any backend.

Backend-generic (the harness is, so this is too): pick the adapter with --backend.

* trainium: the difflet CLI starts a fresh process per generate, so it always reloads
  weights; e2e is load-dominated and that load swings wildly with OS page-cache warmth
  (a single fresh run can be SLOWER than the recorded cold run — observed LTX-2 764 s
  when the cache was evicted). So "warm" must mean the cache is genuinely warm: run
  ``--warmups`` discarded cache-warming generates first, then ``--iters`` measured ones.
  Host RAM (124 GB, ~100 GB page cache) holds a single model's weights (e.g. Qwen 54 GB),
  so the warm run reads from cache.

* cuda (diffusers reference): used to add e2e_warm WITHOUT the in-process ``bench
  --iters 1`` path, which OOMs an 80 GB H100 — the cold generate's allocator memory
  isn't released before the warm load, so two generates can't coexist. Run this in its
  OWN process right after the cold ``bench --iters 0``: weights are already on disk, so
  the load reads a WARM disk cache (the numeric equivalent of B300's in-process second
  iter). ``--warmups`` defaults to 0 here (the cold run already warmed the disk).

Writes e2e_warm into benchmark/<device>/<slug>.json and re-renders the md; uses an
isolated log dir so the cold-run logs (and their breakdowns) survive.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.warm_e2e --model qwen_image [--warmups 1] [--iters 1]
    # CUDA/diffusers reference (e.g. on an H100), cold run done first:
    DIFFLET_BENCH_DEVICE=h100 \
        python -m benchmark.warm_e2e --backend cuda --model qwen_image
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import report
from benchmark.harness import Stats
from benchmark.models import add_config_arg, json_path, report_path, logs_dir, resolve


def _make_adapter(backend: str):
    if backend == "trainium":
        from benchmark.adapters.trainium import TrainiumAdapter
        return TrainiumAdapter(log_dir=f"{logs_dir()}/warm")
    if backend in ("cuda", "diffusers", "cpu"):
        from benchmark.adapters.diffusers_ref import DiffusersRefAdapter
        return DiffusersRefAdapter(device="cuda" if backend == "cuda" else "cpu")
    raise SystemExit(f"unknown backend: {backend}")


def _note(backend: str, st: Stats, warmups: int) -> str:
    if backend == "trainium":
        return (f"e2e_warm = {st.mean:.0f} s (n={st.n}; reported after {warmups} "
                "discarded cache-warming run(s) so the OS page cache is warm). The "
                "difflet CLI reloads weights every process, so 'warm' = warm disk "
                "cache -> faster load, not a resident model; cf. e2e cold and the "
                "load/compute breakdown.")
    return (f"e2e_warm = {st.mean:.0f} s (n={st.n}) — warm-cache generate(s) run in a "
            "SEPARATE process (an 80 GB H100 can't hold a second in-process generate "
            "after the cold run; the numeric equivalent of B300's in-process second "
            "iter). Weights were on disk from the immediately-preceding cold run, so "
            "warm = warm disk cache -> faster load, not a resident model.")


def run(model: str, backend: str, warmups: int, iters: int, config: str = "tp4") -> int:
    cfg = resolve(model, config)
    slug = cfg.config_slug   # result-file stem: <model>[_<config>]
    ad = _make_adapter(backend)
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
    notes = d.setdefault("notes", [])
    d["notes"] = [n for n in notes if not n.startswith("e2e_warm =")]
    d["notes"].append(_note(backend, st, warmups))
    jp.write_text(json.dumps(d, indent=2))
    Path(report_path(slug)).write_text(report.render(d))
    print(f"[warm] {slug}: mean {st.mean:.1f}s median {st.median:.1f}s n={st.n} -> patched", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="trainium",
                   help="trainium (difflet CLI) | cuda/cpu (diffusers reference)")
    p.add_argument("--warmups", type=int, default=None,
                   help="discarded cache-warming runs (default: 1 for trainium, 0 for "
                        "cuda/cpu — the cold run already warmed the disk)")
    p.add_argument("--iters", type=int, default=1)
    add_config_arg(p)
    a = p.parse_args()
    warmups = a.warmups if a.warmups is not None else (1 if a.backend == "trainium" else 0)
    return run(a.model, a.backend, warmups, a.iters, a.config)


if __name__ == "__main__":
    sys.exit(main())
