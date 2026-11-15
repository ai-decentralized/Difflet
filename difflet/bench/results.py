"""Result records: append-only JSON, one per ``bench run`` of one recipe (design §9).

Layout::

    <results>/<device>/<recipe dir>/<run_id>.input.json   written before anything is launched
    <results>/<device>/<recipe dir>/<run_id>.json         written when the run ends (or fails)

``<results>`` is ``benchmark/results`` in the checkout bench runs from (so results are committed
next to the code that produced them), else ``$DIFFLET_BENCH_ROOT/results``. The recipe dir is the
recipe name, ``<recipe>@<suite>.<variant>`` for a suite variant, or ``<recipe>+<fingerprint>`` for a
run with ``--`` overrides. Files are never overwritten: a run id is unique, and every write goes to
a temporary name first and is renamed into place.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import statistics
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

#: The record layout. Bump when a field changes meaning.
SCHEMA_VERSION = 1
#: The measurement method (design §7). Bump when any metric's definition changes;
#: ``compare`` refuses records with different method versions.
METHOD_VERSION = 1


def results_root() -> Path:
    from difflet.bench.host import bench_root, source_root

    checkout = source_root() / "benchmark"
    return checkout / "results" if checkout.is_dir() else bench_root() / "results"


def host_id() -> str:
    """A stable, non-identifying id of this machine (hash of /etc/machine-id), to tell people's
    hosts apart in results without recording a host name."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw:
            return hashlib.sha256(raw.encode()).hexdigest()[:12]
    return "unknown"


def new_run_id(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{host_id()[:6]}-{uuid.uuid4().hex[:6]}"


def recipe_dir_name(recipe: str, *, suite: str | None = None, variant: str | None = None,
                    fingerprint: str | None = None) -> str:
    name = recipe
    if suite and variant:
        name += f"@{suite}.{variant}"
    if fingerprint:
        name += f"+{fingerprint}"
    return name


def result_paths(root: Path, device: str, dir_name: str, run_id: str) -> tuple[Path, Path]:
    base = root / device / dir_name
    return base / f"{run_id}.input.json", base / f"{run_id}.json"


def finite(value: Any) -> Any:
    """``value`` with every non-finite float as a string ("inf", "-inf", "nan"): an identical
    frame has an infinite PSNR, and strict JSON cannot hold it. Readers format numbers through
    :func:`fmt`, which passes these strings through."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def fmt(value: Any, spec: str) -> str:
    """``format(value, spec)`` for a number, the value itself for "inf"/"nan" strings, "" for
    None."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    return str(value)


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` atomically; refuses to replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}: results are append-only")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(finite(data), indent=2, sort_keys=False, ensure_ascii=False,
                              allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_results(root: Path) -> Iterable[Path]:
    """Every final result under ``root`` (input records excluded)."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json")
                  if not p.name.endswith(".input.json") and not p.name.startswith("."))


# ------------------------------------------------------------------------------------ stats


def _nearest_rank(sorted_samples: Sequence[float], p: float) -> float:
    """benchmark/harness.Stats' percentile rule, kept so old and new numbers mean the same."""
    n = len(sorted_samples)
    idx = min(n - 1, max(0, int(round(p * (n - 1)))))
    return sorted_samples[idx]


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """n, mean, median, min, max, p90, std and every raw sample (design §7)."""
    values = [float(v) for v in samples if v is not None and not math.isnan(float(v))]
    if not values:
        return {"n": 0, "samples": []}
    s = sorted(values)
    return {"n": len(s), "mean": statistics.fmean(s), "median": statistics.median(s),
            "min": s[0], "max": s[-1], "p90": _nearest_rank(s, 0.90),
            "std": statistics.pstdev(s) if len(s) > 1 else 0.0, "samples": values}


def metric_record(*, samples: Sequence[float] | None = None, unit: str = "s",
                  invalid: Sequence[str] = (), not_applicable: str | None = None,
                  **extra: Any) -> dict[str, Any]:
    """One metric's entry: validity, summary over its samples, and anything metric-specific."""
    record: dict[str, Any] = {"valid": not invalid and not_applicable is None,
                              "invalid_reasons": list(invalid), "not_applicable": not_applicable,
                              "unit": unit}
    if samples is not None:
        record["summary"] = summarize(samples)
    record.update(extra)
    return record
