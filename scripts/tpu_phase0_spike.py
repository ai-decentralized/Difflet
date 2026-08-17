#!/usr/bin/env python3
"""Phase 0 spike: which compile/load model fits TPU?

Throwaway probe for docs/plans/2026-08-16-tpu-backend-support.md — run it on
a TPU VM with torch_xla installed; it is never imported by the package. It
answers, per direction:

  A (AOT StableHLO export)   — can a graph *containing a collective* be
                               exported, saved, and re-loaded in a fresh
                               process without recompiling?
  B (lazy + persistent cache)— does the persistent compilation cache make a
                               second process skip compilation?
  C (torchax / torch-on-JAX) — is the interop stack even installed/usable?

The probe module deliberately includes an all_gather: exporting graphs with
collectives is exactly where AOT paths tend to break, and the real models
are full of them. A single-participant collective (1 device) still exercises
the op's lowering, but is weaker than a real mesh — re-run with >=2 visible
chips before trusting a PASS.

Usage (run twice; the second run answers the cross-process questions):
    python scripts/tpu_phase0_spike.py --direction all --work-dir /tmp/tpu_spike
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _report(direction: str, step: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{direction}] {mark:4s} {step}" + (f" — {detail}" if detail else ""), flush=True)


def _build_module():
    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    class Probe(nn.Module):
        """Linear -> all_gather -> Linear: minimal graph with a collective."""

        def __init__(self) -> None:
            super().__init__()
            self.fc_in = nn.Linear(64, 64, bias=True)
            self.fc_out = nn.Linear(64, 16, bias=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.fc_in(x)
            # Single-participant group still lowers to an HLO AllGather.
            # torch_xla 2.9 dropped xm.get_ordinal(); it lives on xr now.
            h = xm.all_gather(h, dim=0, groups=[[xr.global_ordinal()]], pin_layout=False)
            return self.fc_out(h)

    return Probe()


def direction_a(work_dir: Path) -> None:
    """AOT export: torch.export -> StableHLO -> save -> load."""
    import torch

    export_dir = work_dir / "stablehlo"
    model = _build_module().eval()
    example = (torch.randn(4, 64),)

    try:
        exported = torch.export.export(model, example)
        _report("A", "torch.export with collective in graph", True)
    except Exception as exc:  # noqa: BLE001 — probe: report, don't crash
        _report("A", "torch.export with collective in graph", False, repr(exc))
        return

    try:
        from torch_xla import stablehlo as xla_stablehlo

        shlo = xla_stablehlo.exported_program_to_stablehlo(exported)
        _report("A", "exported_program_to_stablehlo", True)
    except Exception as exc:  # noqa: BLE001
        _report("A", "exported_program_to_stablehlo", False, repr(exc))
        return

    try:
        xla_stablehlo.save_torch_model_as_stablehlo(model, example, str(export_dir))
        _report("A", f"save to {export_dir}", True)
    except Exception as exc:  # noqa: BLE001
        _report("A", "save_torch_model_as_stablehlo", False, repr(exc))
        return

    try:
        loaded = xla_stablehlo.StableHLOGraphModule.load(str(export_dir))
        start = time.monotonic()
        out = loaded(*example)
        elapsed = time.monotonic() - start
        _report("A", "load + execute in this process", True, f"{elapsed:.2f}s, out {tuple(out.shape)}")
        print("[A] NOTE: re-run this script in a fresh process; if this step is fast the "
              "second time too, the artifact genuinely survives a process boundary.")
    except Exception as exc:  # noqa: BLE001
        _report("A", "StableHLOGraphModule.load/execute", False, repr(exc))
    del shlo  # noqa: F821 — keep the object alive until here for its side effects


def direction_b(work_dir: Path) -> None:
    """Lazy execution + persistent compilation cache across processes."""
    import torch
    import torch_xla.core.xla_model as xm

    cache_dir = work_dir / "xla_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch_xla.runtime as xr

        xr.initialize_cache(str(cache_dir), readonly=False)
        _report("B", "xr.initialize_cache", True, str(cache_dir))
    except Exception as exc:  # noqa: BLE001
        _report("B", "xr.initialize_cache", False,
                f"{exc!r} — fall back to XLA_PERSISTENT_CACHE_PATH env var if supported")

    device = xm.xla_device()
    model = _build_module().to(device)
    x = torch.randn(4, 64, device=device)

    timings = []
    for i in range(2):
        start = time.monotonic()
        _ = model(x)
        xm.mark_step()
        xm.wait_device_ops()
        timings.append(time.monotonic() - start)
    n_entries = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
    _report("B", "two in-process steps", True,
            f"step1 {timings[0]:.2f}s vs step2 {timings[1]:.2f}s; cache files: {n_entries}")

    marker = work_dir / "b_first_run_done"
    if marker.exists():
        print(f"[B] VERDICT input: compare step1 above with the first run's step1 "
              f"(recorded {marker.read_text().strip()}s). A large drop = the persistent "
              "cache works across processes; no drop = Direction B fails its core promise.")
    else:
        marker.write_text(f"{timings[0]:.2f}")
        print("[B] First run recorded. Re-run the script now: the second process's step1 "
              "timing tells you whether compilation was actually skipped.")


def direction_c(_: Path) -> None:
    """torchax availability probe only — no deep test until A/B verdicts land."""
    for name in ("torchax", "torch_xla2"):
        try:
            mod = __import__(name)
            _report("C", f"import {name}", True, getattr(mod, "__version__", "unknown version"))
            return
        except ImportError:
            continue
    _report("C", "import torchax / torch_xla2", False,
            "not installed — pip install torchax on the TPU VM to evaluate Direction C")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("a", "b", "c", "all"), default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/tpu_phase0_spike"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    versions = {}
    for pkg in ("torch", "torch_xla", "libtpu"):
        try:
            from importlib import metadata

            versions[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    print(f"toolchain: {json.dumps(versions)}", flush=True)

    if args.direction in ("a", "all"):
        direction_a(args.work_dir)
    if args.direction in ("b", "all"):
        direction_b(args.work_dir)
    if args.direction in ("c", "all"):
        direction_c(args.work_dir)


if __name__ == "__main__":
    main()
