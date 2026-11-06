"""Layer 3 of the TPU profile: which tracer actually sees torch_tpu's device ops?

Unknown until tried on a chip: torch_xla had ``torch_xla.debug.profiler``;
torch_tpu 0.2.0 documents none we have read. This runs a small, recognisable
workload (a bf16 matmul chain plus an SDPA) under each candidate and reports
whether the trace it leaves contains device-side events, so the real Wan trace
can use whichever one works:

A. ``torch.profiler`` with every ProfilerActivity this build offers;
B. ``jax.profiler`` (torch_tpu runs on libtpu, which jax's tracer also drives);
C. discovery: anything named ``*profil*`` / ``*trace*`` on ``torch.tpu`` and
   ``torch_tpu``, printed for a human to follow up.

    python -m benchmark.tpu_profile.profiler_probe \\
        --out-dir /mnt/models/profile/trace_probe
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def _workload(torch, dev):
    import torch.nn.functional as F

    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    q = torch.randn(1, 10, 4680, 128, dtype=torch.bfloat16, device=dev)

    def run():
        x = a
        for _ in range(4):
            x = x @ a
        return x.sum() + F.scaled_dot_product_attention(q, q, q).sum()

    return run


def _try_torch_profiler(torch, run, wait, out_dir: Path) -> dict:
    from torch.profiler import ProfilerActivity, profile

    # A pybind11 enum: iterate its members, not the type.
    activities = list(ProfilerActivity.__members__.values())
    result = {"activities": [a.name for a in activities]}
    with profile(activities=activities) as prof:
        run()
        wait()
    path = out_dir / "torch_profiler.json"
    prof.export_chrome_trace(str(path))
    events = prof.events()
    device_events = [e for e in events if str(getattr(e, "device_type", "")).split(".")[-1] != "CPU"]
    result.update(trace=str(path), events=len(events), device_events=len(device_events),
                  device_examples=sorted({e.name for e in device_events})[:15])
    return result


def _try_jax_profiler(run, wait, out_dir: Path) -> dict:
    import jax

    trace_dir = out_dir / "jax_trace"
    jax.profiler.start_trace(str(trace_dir))
    try:
        run()
        wait()
    finally:
        jax.profiler.stop_trace()
    files = [p for p in trace_dir.rglob("*") if p.is_file()]
    # An xplane that saw the chips carries "/device:TPU" planes.
    saw_tpu = any(b"/device:TPU" in p.read_bytes() for p in files if p.suffix == ".pb")
    return {"trace_dir": str(trace_dir), "files": [str(p) for p in files],
            "contains_tpu_device_plane": saw_tpu}


def _discover(torch) -> dict:
    found = {}
    modules = {"torch.tpu": getattr(torch, "tpu", None)}
    try:
        import torch_tpu

        modules["torch_tpu"] = torch_tpu
    except Exception:  # noqa: BLE001
        pass
    for name, module in modules.items():
        if module is None:
            continue
        found[name] = sorted(attr for attr in dir(module)
                             if "profil" in attr.lower() or "trace" in attr.lower())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.prepare_launch(1)
    device_runtime.initialize_replica(0, 1)

    import torch

    run = _workload(torch, device_runtime.device())
    with torch.no_grad():
        run()                       # warm, so the trace holds steady-state ops
        device_runtime.wait()
        report = {"runtime": "torch_tpu", "discovered": _discover(torch)}
        for name, attempt in (
            ("torch_profiler", lambda: _try_torch_profiler(torch, run, device_runtime.wait, args.out_dir)),
            ("jax_profiler", lambda: _try_jax_profiler(run, device_runtime.wait, args.out_dir)),
        ):
            try:
                report[name] = {"status": "ok", **attempt()}
            except Exception as exc:  # noqa: BLE001 -- which ones fail is the finding
                report[name] = {"status": "error", "error": repr(exc),
                                "traceback": traceback.format_exc()}
            print(f"[profiler probe] {name}: {json.dumps(report[name])[:400]}", flush=True)
    print(f"[profiler probe] discovered: {report['discovered']}", flush=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
