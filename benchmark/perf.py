"""Perf records: one JSON file per benchmark run, and the perf page data built from them.

Every published result is one file in ``benchmark/results/``. The perf page
(``docs/perf/index.html``, deployed to GitHub Pages) reads ``perf.json``, which
``build`` makes from those files by keeping the newest record for each
(model, platform, backend).

    python -m benchmark.perf add benchmark/h100/flux_1_dev.json      # bench.py JSON -> record
    python -m benchmark.perf add <bench.json> --backend native --psnr 22.66 --ssim 0.885
    python -m benchmark.perf check                                    # validate every record
    python -m benchmark.perf build --out docs/perf/perf.json          # page data

``benchmark.bench`` writes a record automatically after each run. For a run made
outside ``bench.py``, copy an existing record, edit it, and run ``check``.

Standard library only: the Pages workflow runs this without installing anything.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

SCHEMA = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BACKENDS = ("native", "aot", "cuda", "tpu")
STATUSES = ("ok", "failed", "oom")
METRICS = ("dit_step_ms", "warm_e2e_s", "cold_e2e_s", "compile_s", "psnr_db", "ssim")
REQUIRED = (
    "schema",
    "model",
    "platform",
    "backend",
    "device",
    "parallel",
    "shape",
    "steps",
    "toolchain",
    "metrics",
    "status",
    "date",
)

# bench.py's adapter name -> perf backend.
ADAPTER_BACKEND = {"trainium": "aot", "neuron": "native", "cuda": "cuda", "tpu": "tpu"}

# Display order and names. Unknown values still work: they sort last and show as-is.
PLATFORMS = {"b300": "B300", "h100": "H100", "trn2": "trn2", "trn3": "trn3", "v5e": "TPU v5e"}
BACKEND_LABELS = {
    "native": "PyTorch Native",
    "aot": "AOT",
    "cuda": "CUDA (diffusers eager)",
    "tpu": "PyTorch/XLA",
}
MODELS = {
    "qwen_image": "Qwen-Image",
    "wan_2_1": "Wan2.1 14B",
    "hunyuan_video": "HunyuanVideo 1.0",
    "ltx_2": "LTX-2",
    "flux_1_dev": "FLUX.1-dev",
    "wan_2_2": "Wan2.2 A14B",
    "hunyuan_video_15": "HunyuanVideo 1.5",
}


def _order(table: dict, key: str) -> tuple:
    keys = list(table)
    return (keys.index(key), "") if key in keys else (len(keys), key)


def config_label(platform: str, backend: str) -> str:
    """Short name of a platform + backend, e.g. ``trn2 Native``, ``H100``."""
    name = PLATFORMS.get(platform, platform)
    if backend == "native":
        return f"{name} Native"
    if backend == "aot":
        return f"{name} AOT"
    return name


# ---------------------------------------------------------------- records


def validate(rec: Any) -> list[str]:
    """Return the problems with a record; an empty list means it is valid."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    errs = [f"missing field '{k}'" for k in REQUIRED if k not in rec]
    if errs:
        return errs
    if rec["schema"] != SCHEMA:
        errs.append(f"schema is {rec['schema']!r}, expected {SCHEMA}")
    if rec["backend"] not in BACKENDS:
        errs.append(f"backend {rec['backend']!r} is not one of {', '.join(BACKENDS)}")
    if rec["status"] not in STATUSES:
        errs.append(f"status {rec['status']!r} is not one of {', '.join(STATUSES)}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}( UTC)?)?", str(rec["date"])):
        errs.append(f"date {rec['date']!r} is not 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM UTC'")
    if not isinstance(rec["steps"], int) or rec["steps"] <= 0:
        errs.append("steps must be a positive integer")
    shape = rec["shape"]
    if not isinstance(shape, dict) or not shape.get("height") or not shape.get("width"):
        errs.append("shape needs height and width")
    if not isinstance(rec["toolchain"], dict):
        errs.append("toolchain must be an object")
    metrics = rec["metrics"]
    if not isinstance(metrics, dict):
        errs.append("metrics must be an object")
    else:
        for k, v in metrics.items():
            if k not in METRICS:
                errs.append(f"unknown metric '{k}' (known: {', '.join(METRICS)})")
            elif v is not None and (
                not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0
            ):
                errs.append(f"metric '{k}' must be a non-negative number or null")
        if rec["status"] == "ok" and all(metrics.get(k) is None for k in METRICS):
            errs.append("status is 'ok' but every metric is null")
    if not isinstance(rec.get("notes", []), list):
        errs.append("notes must be a list of strings")
    return errs


def _parallel(d: dict, backend: str) -> str:
    if backend == "cuda":
        return "1 GPU"  # the CUDA reference adapter is single-GPU dense whatever config_label says
    p = d.get("parallel") or {}
    out = f"TP{p.get('tp_degree', 1)}"
    if (p.get("cp_degree") or 1) > 1:
        out += f" CP{p['cp_degree']}"
    return out


def _device(d: dict) -> str:
    dev = str(d.get("device") or "")
    return dev.split(" / ", 1)[1] if dev.startswith("CUDA / ") else dev


def record_from_bench(d: dict, backend: Optional[str] = None, source: Optional[str] = None) -> dict:
    """Convert a ``benchmark.bench`` result JSON into a perf record."""
    adapter = d.get("backend")
    backend = backend or ADAPTER_BACKEND.get(adapter)
    if backend is None:
        raise ValueError(f"unknown adapter {adapter!r}: pass --backend")
    sl, warm = d.get("step_latency") or {}, d.get("e2e_warm") or {}
    compile_s = d.get("compile_seconds")
    status = d.get("status")
    # bench.py's notes are methodology for its own report; the page only gets the failure reason.
    failures = [n for n in d.get("notes") or [] if isinstance(n, str) and n.startswith("FAILED")]
    if status == "failed" and any("out of memory" in n.lower() for n in failures):
        status = "oom"
    elif status not in STATUSES:
        status = "failed"
    notes = [n.split(". ")[0][:160] for n in failures] if status != "ok" else []
    ts = d.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    return {
        "schema": SCHEMA,
        "model": d.get("config_slug"),
        "model_id": d.get("model_id"),
        "platform": d.get("device_slug"),
        "device": _device(d),
        "backend": backend,
        "parallel": _parallel(d, backend),
        "shape": d.get("shape"),
        "steps": d.get("steps"),
        "seed": d.get("seed"),
        "guidance_scale": d.get("guidance_scale"),
        "dtype": d.get("dtype"),
        "toolchain": d.get("toolchain") or {},
        "metrics": {
            # mean, as the device RESULTS.md tables report it
            "dit_step_ms": round(sl["mean"] * 1000, 2) if sl.get("mean") is not None else None,
            "warm_e2e_s": round(warm["mean"], 2) if warm.get("mean") is not None else None,
            "cold_e2e_s": (
                round(d["e2e_cold_seconds"], 2) if d.get("e2e_cold_seconds") is not None else None
            ),
            # eager backends have nothing to compile; bench.py writes 0.0 for them
            "compile_s": (
                round(compile_s, 1) if compile_s and backend in ("native", "aot") else None
            ),
            "psnr_db": None,
            "ssim": None,
        },
        "status": status,
        "notes": notes,
        "date": ts,
        "commit": d.get("commit"),
        "source": source,
    }


def record_filename(rec: dict) -> str:
    stamp = str(rec["date"]).replace(" UTC", "").replace(":", "").replace(" ", "_")
    return f"{stamp}_{rec['platform']}_{rec['backend']}_{rec['model']}.json"


def write_record(rec: dict, results_dir: Path = RESULTS_DIR, force: bool = False) -> Path:
    errs = validate(rec)
    if errs:
        raise ValueError("invalid perf record: " + "; ".join(errs))
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / record_filename(rec)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to replace it)")
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def load_records(results_dir: Path = RESULTS_DIR) -> list[tuple[Path, dict]]:
    """Load every record, raising ValueError listing all invalid files."""
    out, bad = [], []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            bad.append(f"{path.name}: not valid JSON ({e})")
            continue
        errs = validate(rec)
        if errs:
            bad.append(f"{path.name}: " + "; ".join(errs))
        else:
            out.append((path, rec))
    if bad:
        raise ValueError("invalid perf records:\n  " + "\n  ".join(bad))
    return out


# ---------------------------------------------------------------- page data


def _shape_text(shape: dict) -> str:
    dims = [shape["height"], shape["width"]] + (
        [shape["num_frames"]] if shape.get("num_frames") else []
    )
    return "×".join(str(x) for x in dims)


def build(results_dir: Path = RESULTS_DIR, repo_root: Optional[Path] = None) -> dict:
    """The perf page data: the newest record per (model, platform, backend)."""
    repo_root = repo_root or results_dir.parent.parent
    latest: dict[tuple, tuple[Path, dict]] = {}
    for path, rec in load_records(results_dir):
        key = (rec["model"], rec["platform"], rec["backend"])
        if key not in latest or (str(rec["date"]), path.name) > (
            str(latest[key][1]["date"]),
            latest[key][0].name,
        ):
            latest[key] = (path, rec)

    configs, models, results = {}, {}, []
    for (model, platform, backend), (path, rec) in latest.items():
        ckey = f"{platform}/{backend}"
        configs.setdefault(
            ckey,
            {
                "key": ckey,
                "platform": platform,
                "backend": backend,
                "label": config_label(platform, backend),
                "backend_label": BACKEND_LABELS.get(backend, backend),
                "device": rec["device"],
            },
        )
        models.setdefault(
            model,
            {"slug": model, "name": MODELS.get(model, model), "shape": _shape_text(rec["shape"])},
        )
        try:
            file = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            file = path.name
        results.append(
            {
                "model": model,
                "config": ckey,
                "status": rec["status"],
                "steps": rec["steps"],
                "parallel": rec["parallel"],
                "shape": _shape_text(rec["shape"]),
                "torch": rec["toolchain"].get("torch"),
                "toolchain": rec["toolchain"],
                "metrics": {k: rec["metrics"].get(k) for k in METRICS},
                "notes": rec.get("notes", []),
                "date": str(rec["date"])[:10],
                "commit": rec.get("commit"),
                "source": rec.get("source") or file,
                "record": file,
            }
        )

    def cfg_order(c):
        return _order(PLATFORMS, c["platform"]) + _order(BACKEND_LABELS, c["backend"])

    return {
        "schema": SCHEMA,
        "generated": max((r["date"] for r in results), default=None),
        "configs": sorted(configs.values(), key=cfg_order),
        "models": sorted(models.values(), key=lambda m: _order(MODELS, m["slug"])),
        "results": sorted(results, key=lambda r: (_order(MODELS, r["model"]), r["config"])),
    }


# ---------------------------------------------------------------- CLI


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.perf", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="convert benchmark.bench result JSON files into perf records")
    a.add_argument("bench_json", nargs="+", type=Path)
    a.add_argument(
        "--backend", choices=BACKENDS, help="override the backend inferred from the adapter"
    )
    a.add_argument("--model", help="override the model slug (e.g. a variant run of wan_2_1)")
    a.add_argument(
        "--source", help="report path to link from the page (default: the record itself)"
    )
    a.add_argument("--psnr", type=float, help="PSNR in dB against the reference")
    a.add_argument("--ssim", type=float, help="SSIM against the reference")
    a.add_argument(
        "--note", action="append", default=[], help="note shown next to the result (repeatable)"
    )
    a.add_argument("--force", action="store_true", help="replace a record with the same name")

    sub.add_parser("check", help="validate every record")

    b = sub.add_parser("build", help="write the perf page data")
    b.add_argument("--out", type=Path, default=Path("docs/perf/perf.json"))

    args = p.parse_args(argv)
    if args.cmd == "add":
        if (args.psnr is not None or args.ssim is not None) and len(args.bench_json) > 1:
            p.error("--psnr/--ssim apply to one run: pass a single bench JSON")
        for f in args.bench_json:
            rec = record_from_bench(
                json.loads(f.read_text()), backend=args.backend, source=args.source
            )
            rec["model"] = args.model or rec["model"]
            rec["metrics"]["psnr_db"] = args.psnr
            rec["metrics"]["ssim"] = args.ssim
            rec["notes"] += args.note
            print(f"[perf] wrote {write_record(rec, args.results_dir, force=args.force)}")
    elif args.cmd == "check":
        n = len(load_records(args.results_dir))
        print(f"[perf] {n} records OK in {args.results_dir}")
    elif args.cmd == "build":
        data = build(args.results_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(
            f"[perf] {len(data['results'])} results, {len(data['configs'])} platforms -> {args.out}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as e:
        sys.exit(f"[perf] {e}")
