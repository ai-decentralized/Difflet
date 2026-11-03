"""Perf records: one JSON file per benchmark run, and the perf page data built from them.

Every published result is one file in ``benchmark/results/``. The perf page
(``docs/perf/index.html``, deployed to GitHub Pages) reads ``perf.json``, which
``build`` makes from those files by keeping the newest record for each
(model, platform, backend).

    python -m benchmark.perf add benchmark/h100/flux_1_dev.json      # bench.py JSON -> record
    python -m benchmark.perf add <bench.json> --backend native --psnr 22.66 --ssim 0.885
    python -m benchmark.perf check                                    # validate every record
    python -m benchmark.perf build --out docs/perf/perf.json          # page data

``benchmark.bench`` writes a record automatically after each run, and the scripts that
later patch its result JSON (``cold_warm_e2e``, ``warm_e2e``, ``step_latency``,
``step_realloop``, ``backfill_breakdowns``) re-sync that record in place, keeping PSNR/SSIM
and notes added by hand. ``neuron_cold_warm`` writes the record of a completed run (not a
stopped job, a ``--steps`` smoke run, or one with ``--out-json``/``--out-md``). For a run
made outside these scripts, copy an existing record, edit it, and run ``check``.

Standard library only: the Pages workflow runs this without installing anything.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
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
# MATRIX slugs that are a backend-specific variant of a model the page already shows.
MODEL_ALIASES = {"wan_2_1_neuron": "wan_2_1"}
# Measured outside bench.py and added by hand; kept when a run's record is re-synced.
MANUAL_METRICS = ("psnr_db", "ssim")


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
            elif v is not None and not _finite_non_negative(v):
                # NaN/Infinity would also break perf.json for the page (JSON has no such values)
                errs.append(f"metric '{k}' must be a finite non-negative number or null")
        if rec["status"] == "ok" and all(metrics.get(k) is None for k in METRICS):
            errs.append("status is 'ok' but every metric is null")
    g = rec.get("guidance_scale")
    if g is not None and not _finite_non_negative(g):
        errs.append("guidance_scale must be a finite non-negative number or null")
    if not isinstance(rec.get("notes", []), list):
        errs.append("notes must be a list of strings")
    return errs


def _finite_non_negative(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


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
    # bench.py's notes are methodology for its own report. The page gets only the reason a run
    # failed, or why a completed run was flagged (neuron_cold_warm: soft checks only, the
    # numbers stand).
    raw = [n for n in d.get("notes") or [] if isinstance(n, str)]
    failures = [n for n in raw if n.startswith("FAILED")]
    flags = [n for n in raw if n.startswith("FLAGGED")]
    if status == "flagged":
        status = "ok"
    elif status == "failed" and any("out of memory" in n.lower() for n in failures):
        status = "oom"
    elif status not in STATUSES:
        status = "failed"
    notes = [n.split(". ")[0][:160] for n in (flags if status == "ok" else failures)]
    if compile_s and _cache_hit(d, backend):
        compile_s = None
        notes.append(CACHE_HIT_NOTE)
    ts = d.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    return {
        "schema": SCHEMA,
        "model": MODEL_ALIASES.get(d.get("config_slug"), d.get("config_slug")),
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


CACHE_HIT_NOTE = "Compile was a cache hit (no component was built), so there is no compile time"


def _cache_hit(d: dict, backend: str) -> bool:
    """An AOT compile that built nothing: its breakdown has only the wall total, no component
    (the rule backfill_breakdowns uses for cache-hit logs). Its seconds are not a compile time."""
    cb = d.get("compile_breakdown")
    return backend == "aot" and isinstance(cb, dict) and set(cb) == {"wall_total_s"}


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


def keep_manual_fields(rec: dict, old: dict, keep_notes: bool = True) -> dict:
    """Carry what was added by hand to an earlier version of the same record into ``rec``:
    PSNR/SSIM that ``rec`` lacks, and (with ``keep_notes``) its notes, except the generated
    failure/flag reasons, which ``rec`` brings fresh. The old source link is kept only when
    ``rec`` has none: a script re-sync links its own report."""
    old_metrics = old.get("metrics") or {}
    for k in MANUAL_METRICS:
        if rec["metrics"].get(k) is None and old_metrics.get(k) is not None:
            rec["metrics"][k] = old_metrics[k]
    if keep_notes:
        generated = ("FAILED", "FLAGGED", CACHE_HIT_NOTE)
        kept = [
            n for n in old.get("notes") or [] if isinstance(n, str) and not n.startswith(generated)
        ]
        rec["notes"] = kept + [n for n in rec["notes"] if n not in kept]
    if not rec.get("source") and old.get("source"):
        rec["source"] = old["source"]
    return rec


def _old_record(path: Path) -> Optional[dict]:
    """The existing record at ``path`` to merge from, or None when there is none or it cannot be
    read (e.g. left with merge-conflict markers): then it is simply replaced."""
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"[perf] replacing unreadable {path}: {e}", file=sys.stderr, flush=True)
        return None
    return old if isinstance(old, dict) else None


def sync_record(
    bench_json: Path | str, source: Optional[str] = None, results_dir: Path = RESULTS_DIR
) -> Path:
    """Write (or rewrite in place) the record of a ``benchmark.bench`` result JSON.

    The file name comes from the run's timestamp, which the patch scripts leave alone, so
    re-syncing after a patch replaces the same record instead of adding a stale twin."""
    rec = record_from_bench(json.loads(Path(bench_json).read_text()), source=source)
    old = _old_record(results_dir / record_filename(rec))
    if old is not None:
        keep_manual_fields(rec, old)
    return write_record(rec, results_dir, force=True)


def sync_quietly(
    bench_json: Path | str, source: Optional[str] = None, results_dir: Path = RESULTS_DIR
) -> Optional[Path]:
    """``sync_record`` for the benchmark scripts: report, never raise. A perf record must
    not fail a run that already wrote its results."""
    try:
        path = sync_record(bench_json, source=source, results_dir=results_dir)
    except Exception as e:  # noqa: BLE001 - see docstring
        print(
            f"[perf] no perf record written for {bench_json}: {type(e).__name__}: {e}", flush=True
        )
        return None
    print(
        f"[perf] perf record -> {path} (commit it to publish; see benchmark/README.md)", flush=True
    )
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
                "device": rec["device"],  # per result: one platform's runs can differ (e.g. LNC)
                # with shape and steps, the workload: the page warns when compared rows differ
                "guidance_scale": rec.get("guidance_scale"),
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
        "--note",
        action="append",
        default=[],
        help="note shown next to the result (repeatable); with --force, replaces the old notes",
    )
    a.add_argument(
        "--force",
        action="store_true",
        help="replace a record with the same name, keeping its PSNR/SSIM and notes unless given",
    )

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
            old = _old_record(args.results_dir / record_filename(rec)) if args.force else None
            if old is not None:
                # given --note values replace the old notes; given --psnr/--ssim replace too
                keep_manual_fields(rec, old, keep_notes=not args.note)
            print(f"[perf] wrote {write_record(rec, args.results_dir, force=args.force)}")
    elif args.cmd == "check":
        n = len(load_records(args.results_dir))
        print(f"[perf] {n} records OK in {args.results_dir}")
    elif args.cmd == "build":
        data = build(args.results_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        print(
            f"[perf] {len(data['results'])} results, {len(data['configs'])} platforms -> {args.out}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as e:
        sys.exit(f"[perf] {e}")
