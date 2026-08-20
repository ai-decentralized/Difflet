#!/usr/bin/env python3
"""Build difflet/planner/data/measurements.json from benchmark/<device>/*.json.

The planner ships measured step latencies so its rankings rest on real numbers
where real numbers exist. Those numbers are produced by ``benchmark/bench.py``
and committed under ``benchmark/<device>/``; this script distills them into the
small, stable file the planner loads at runtime, so an installed wheel does not
depend on the repository layout.

Run it after adding or refreshing a benchmark result:

    python scripts/seed_planner_measurements.py            # every device dir
    python scripts/seed_planner_measurements.py --device trn2
    python scripts/seed_planner_measurements.py --check    # CI: fail if stale

``--check`` regenerates in memory and diffs, so a benchmark landing without a
reseed is caught rather than silently ignored.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from difflet.planner.feasibility import config_label  # noqa: E402
from difflet.planner.measurements import SCHEMA_VERSION  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402

BENCHMARK_ROOT = REPO / "benchmark"
OUTPUT = REPO / "difflet" / "planner" / "data" / "measurements.json"

# benchmark/<slug>.json uses report slugs; the planner keys on registry names.
# The flux_<label> entries come from scripts/flux_parallel_sweep.py, the
# one-config-per-file parallel sweep that anchors the planner's rankings on
# real per-step measurements.
SLUG_TO_MODEL = {
    "flux_1_dev": "flux",
    "flux_tp4": "flux",
    "flux_tp4sp": "flux",
    "flux_tp2cp2": "flux",
    "flux_tp2cp2ring": "flux",
    "flux_tp2cp2ulysses": "flux",
    "flux_dp2tp2": "flux",
    "flux_dp2tp2sp": "flux",
    "qwen_image": "qwen_image",
    "wan_2_1": "wan",
    "wan_2_2": "wan",
    "hunyuan_video": "hunyuan_video",
    "hunyuan_video_15": "hunyuan_video_15",
    "ltx_2": "ltx_2",
}


def device_dirs(device: str | None) -> list[pathlib.Path]:
    if device:
        path = BENCHMARK_ROOT / device
        return [path] if path.is_dir() else []
    return sorted(p for p in BENCHMARK_ROOT.iterdir() if p.is_dir() and p.name != "logs")


# The planner models Neuron hosts only, so GPU reproductions are skipped rather
# than ingested under a meaningless key -- benchmark/h100 and benchmark/b300
# report their device as "CUDA / NVIDIA H100 PCIe", whose first field is "CUDA".
NEURON_INSTANCE_TYPE = re.compile(r"^(?:trn|inf)\d[\w.]*\.[\w]+$")


def instance_type_of(payload: dict) -> str | None:
    """Recover the EC2 instance type from the benchmark's device string.

    ``"trn2.3xlarge / 4 NeuronCores / 96 GB/device"`` -> ``"trn2.3xlarge"``. The
    planner keys measurements by instance type because a step latency measured on
    one shape of host says nothing about another. Returns ``None`` for anything
    that is not a Neuron instance type.
    """

    device = payload.get("device")
    if not isinstance(device, str) or not device.strip():
        return None
    head = device.split("/", 1)[0].strip()
    return head if NEURON_INSTANCE_TYPE.match(head) else None


def label_of(payload: dict) -> str | None:
    parallel = payload.get("parallel")
    if not isinstance(parallel, dict):
        return None
    try:
        config = DiffletParallelConfig(
            tp_degree=int(parallel.get("tp_degree", 1)),
            cp_degree=int(parallel.get("cp_degree", 1)),
            cp_mode=str(parallel.get("cp_mode", "gather_kv")),
            cfg_parallel_enabled=bool(parallel.get("cfg_parallel_enabled", False)),
            sp_enabled=bool(parallel.get("sp_enabled", False)),
            dp_degree=int(parallel.get("dp_degree", 1)),
        )
    except (TypeError, ValueError):
        return None
    return config_label(config)


def collect(device: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for directory in device_dirs(device):
        for path in sorted(directory.glob("*.json")):
            slug = path.stem
            model = SLUG_TO_MODEL.get(slug)
            if model is None:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            instance_type = instance_type_of(payload)
            label = label_of(payload)
            if not instance_type or not label:
                continue
            shape = payload.get("shape") or {}
            step = payload.get("step_latency") or {}
            warm = payload.get("e2e_warm") or {}
            rows.append(
                {
                    "instance_type": instance_type,
                    "model": model,
                    "model_id": payload.get("model_id"),
                    "label": label,
                    "height": shape.get("height"),
                    "width": shape.get("width"),
                    "num_frames": shape.get("num_frames"),
                    "steps": payload.get("steps"),
                    "step_latency_seconds": step.get("median") or step.get("mean"),
                    "e2e_warm_seconds": warm.get("median") or warm.get("mean"),
                    "compile_seconds": payload.get("compile_seconds"),
                    "source": f"benchmark/{directory.name}/{path.name}",
                }
            )
    # Deterministic order: the file is committed, so churn should mean new data.
    rows.sort(
        key=lambda row: (row["instance_type"], row["model"], row["model_id"] or "", row["label"])
    )
    return rows


def build(device: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "note": (
            "Generated by scripts/seed_planner_measurements.py from "
            "benchmark/<device>/*.json. Do not edit by hand."
        ),
        "measurements": collect(device),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Only this benchmark/<device>/ directory")
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if the committed file is stale"
    )
    args = parser.parse_args(argv)

    payload = build(args.device)
    rendered = json.dumps(payload, indent=2) + "\n"

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print(
                f"{OUTPUT.relative_to(REPO)} is stale; "
                "run python scripts/seed_planner_measurements.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT.relative_to(REPO)} is up to date ({len(payload['measurements'])} rows)")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(payload['measurements'])} measurements)")
    for row in payload["measurements"]:
        step = row["step_latency_seconds"]
        rendered_step = f"{step:.4f}s" if step else "-"
        print(f"  {row['instance_type']:<16} {row['model_id'] or row['model']:<52} "
              f"{row['label']:<8} step={rendered_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
