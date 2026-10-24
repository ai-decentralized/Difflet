"""Reuse committed benchmark results instead of re-measuring them on every new instance.

Measured results are kept in their own folder, one JSON per model, committed to git so they
outlive the instance:

    benchmark/cached_results/<slug>.json
    {
      "schema": 1,
      "slug": "flux_1_dev",
      "entries": {                                    # one per config x platform
        "<key: sha256 of config + platform, 16 hex>": {
          "measured_at": "2026-10-24T12:00:00Z",
          "fingerprint": {"config": {...}, "platform": {...}},
          "code": {"git_head": ..., "difflet_tree": ..., "dirty": ...},
          "result": {...the full result JSON as measured...},
          "last_reused": {"at": ..., "instance_type": ..., "git_head": ...}   # once reused
        }
      }
    }

A trn2 entry and a trn3 entry of the same model live side by side; re-measuring replaces the
entry with the same key, and drops the model's entries that have expired.

A run reuses an entry, without touching the device, when all of these hold:

* the entry's result status is ``ok`` or ``flagged`` (a failed, partial or stopped run is never
  stored), and it holds every guidance value the run asks for (neuron_cold_warm);
* the config fingerprint matches (model, revision, shape, steps, parallel layout, prompt, seed,
  flags, backend and the run options such as ``--iters`` / ``--skip-compile``);
* the platform matches: the instance type (EC2 metadata, or ``DIFFLET_INSTANCE_TYPE`` off EC2;
  an unknown host never reuses), the accelerator ``device_info()`` reports, and the SDK (the
  adapter's ``toolchain()`` packages plus the OS-level ``aws-neuronx-*`` packages or the NVIDIA
  driver);
* ``measured_at`` is younger than ``max_age_days`` (30 by default). Reusing does not refresh
  the clock: only a re-measurement does.

The code identity (git HEAD, ``difflet/`` tree) is recorded and shown, but a change only forces
a re-run under ``--rerun-on-code-change``.

On reuse the report files ``benchmark/<device>/<slug>.{json,md}`` are rewritten from the entry
and say so (``**REUSED:**``). Those reports carry a short ``cache`` summary for that banner; the
scripts that patch a report in place (step_latency.py, warm_e2e.py, ...) drop it
(``invalidate``), since the report then no longer is the stored measurement.

    python -m benchmark.result_cache                 # every stored entry: fresh / expired
    python -m benchmark.result_cache --max-age-days 14
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

SCHEMA = 1
MAX_AGE_DAYS = 30
#: Completed measurements. ``flagged`` is neuron_cold_warm's "statistical check flagged, not
#: enforced": a finished run whose report already says so. partial / failed never are.
REUSABLE_STATUSES = ("ok", "flagged")
REPO_ROOT = Path(__file__).resolve().parent.parent
#: The store: one JSON per model (``<slug>.json``).
STORE_DIR = REPO_ROOT / "benchmark" / "cached_results"

#: BenchConfig fields that only describe or label a result; they do not change what runs.
PRESENTATION_FIELDS = ("config_label", "notes", "stage_names", "e2e_host_note", "backends")

_IMDS = "http://169.254.169.254/latest"


# ------------------------------------------------------------------------------ time


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t: datetime.datetime) -> str:
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)


def _days(td: datetime.timedelta) -> str:
    return f"{td.total_seconds() / 86400:.1f} days"


# ------------------------------------------------------------------------------ identity


def config_identity(cfg: Any, *, backend: str, run_options: Mapping[str, Any]) -> dict:
    """What was measured: the MATRIX entry minus its labels, the backend, the run options."""
    fields = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    for name in PRESENTATION_FIELDS:
        fields.pop(name, None)
    return {"backend": backend, **fields, "run_options": dict(run_options)}


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001 - absent tool: the fact is just not recorded
        return ""


def instance_type() -> str:
    """The EC2 instance type (IMDSv2, then IMDSv1); ``unknown`` off EC2, where
    ``DIFFLET_INSTANCE_TYPE`` names the host instead (an unknown host never reuses)."""
    override = os.environ.get("DIFFLET_INSTANCE_TYPE", "").strip()
    if override:
        return override
    try:
        req = urllib.request.Request(f"{_IMDS}/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request(f"{_IMDS}/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=1).read().decode().strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass
    try:
        return urllib.request.urlopen(f"{_IMDS}/meta-data/instance-type",
                                      timeout=1).read().decode().strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def system_sdk() -> dict[str, str]:
    """OS-level accelerator stack the Python packages do not show: the Neuron runtime, driver
    and collectives packages, and the NVIDIA driver."""
    sdk: dict[str, str] = {}
    out = _run(["dpkg-query", "-W", "-f", "${Package}=${Version}\n", "aws-neuronx-*"])
    out += _run(["rpm", "-qa", "--qf", "%{NAME}=%{VERSION}-%{RELEASE}\n", "aws-neuronx-*"])
    for line in out.splitlines():
        name, _, version = line.partition("=")
        if name and version:
            sdk[name.strip()] = version.strip()
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver.strip():
        sdk["nvidia-driver"] = driver.splitlines()[0].strip()
    return sdk


def platform_identity(adapter: Any, *, instance: Optional[str] = None,
                      system: Optional[Mapping[str, str]] = None) -> dict:
    """Where it was measured: instance type, accelerator and SDK versions."""
    sdk = dict(adapter.toolchain() or {})
    sdk.update(system_sdk() if system is None else system)
    return {"instance_type": instance or instance_type(), "device": adapter.device_info(),
            "sdk": dict(sorted(sdk.items()))}


def code_identity(repo: Path = REPO_ROOT) -> dict:
    """Which code measured it (recorded and shown; not part of the key)."""
    def git(*args: str) -> str:
        return _run(["git", "-C", str(repo), *args]).strip()

    return {"git_head": git("rev-parse", "HEAD") or None,
            "difflet_tree": git("rev-parse", "HEAD:difflet") or None,
            "dirty": bool(git("status", "--porcelain", "--", "difflet"))}


def fingerprint(config: Mapping[str, Any], platform: Mapping[str, Any]) -> dict:
    return {"config": dict(config), "platform": dict(platform)}


def key_of(fp: Mapping[str, Any]) -> str:
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------ the store


def store_path(slug: str, store: Optional[Path] = None) -> Path:
    return Path(store or STORE_DIR) / f"{slug}.json"


def load(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def entries(slug: str, store: Optional[Path] = None) -> dict[str, dict]:
    doc = load(store_path(slug, store))
    if not isinstance(doc, Mapping) or doc.get("schema") != SCHEMA:
        return {}
    return dict(doc.get("entries") or {})


def _write(slug: str, ents: Mapping[str, Any], store: Optional[Path] = None) -> None:
    path = store_path(slug, store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "slug": slug, "entries": dict(ents)},
                               indent=2, sort_keys=True) + "\n")


def _age(entry: Mapping[str, Any], now: datetime.datetime) -> datetime.timedelta:
    return now - parse_iso(entry["measured_at"])


def statuses(result: Mapping[str, Any]) -> list[Any]:
    """The result's status and, for a neuron_cold_warm job, each guidance value's."""
    return [result.get("status")] + [(r or {}).get("status") for r in
                                     (result.get("results_by_guidance") or {}).values()]


def completed(result: Mapping[str, Any]) -> bool:
    return all(st in REUSABLE_STATUSES for st in statuses(result))


def save(slug: str, fp: Mapping[str, Any], result: Mapping[str, Any], *,
         code: Mapping[str, Any], measured_at: datetime.datetime, now: datetime.datetime,
         max_age_days: int = MAX_AGE_DAYS, store: Optional[Path] = None) -> dict:
    """Store a completed measurement under its key (replacing the old one) and drop this
    model's expired entries. Returns the entry."""
    if not completed(result):
        raise ValueError(f"only a completed result is stored, not {statuses(result)!r}")
    limit = datetime.timedelta(days=max_age_days)
    ents = {k: e for k, e in entries(slug, store).items() if _age(e, now) <= limit}
    entry = {"measured_at": iso(measured_at), "fingerprint": dict(fp), "code": dict(code),
             "result": dict(result)}
    ents[key_of(fp)] = entry
    _write(slug, ents, store)
    return entry


def mark_reused(slug: str, key: str, *, now: datetime.datetime, platform: Mapping[str, Any],
                code: Mapping[str, Any], store: Optional[Path] = None) -> dict:
    """Record that an entry was reused (not re-measured); ``measured_at`` stays as it was."""
    ents = entries(slug, store)
    ents[key]["last_reused"] = {"at": iso(now), "instance_type": platform.get("instance_type"),
                                "git_head": code.get("git_head")}
    _write(slug, ents, store)
    return ents[key]


# ------------------------------------------------------------------------------ decision


@dataclasses.dataclass(frozen=True)
class Decision:
    reuse: bool
    reason: str
    key: Optional[str] = None


def _diff(path: str, old: Any, new: Any, out: list[str]) -> None:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for k in sorted(set(old) | set(new), key=str):
            _diff(f"{path}.{k}" if path else str(k), old.get(k, "<absent>"),
                  new.get(k, "<absent>"), out)
    elif old != new:
        out.append(f"{path}: {old!r} -> {new!r}")


def differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    _diff("", old, new, out)
    return out


def decide_entry(entry: Optional[Mapping[str, Any]], fp: Mapping[str, Any], *,
                 now: datetime.datetime, max_age_days: int = MAX_AGE_DAYS,
                 code: Optional[Mapping[str, Any]] = None,
                 rerun_on_code_change: bool = False) -> Decision:
    """Reuse this stored ``entry`` for a run whose fingerprint is ``fp``?"""
    if not isinstance(entry, Mapping):
        return Decision(False, "no stored result")
    plat = fp.get("platform") or {}
    if plat.get("instance_type") in (None, "", "unknown"):
        return Decision(False, "instance type could not be read on this host (no EC2 "
                               "metadata); set DIFFLET_INSTANCE_TYPE to name it")
    if not plat.get("sdk"):
        return Decision(False, "SDK versions could not be read on this host")
    result = entry.get("result") or {}
    bad = [st for st in statuses(result) if st not in REUSABLE_STATUSES]
    if bad:
        return Decision(False, f"stored result's status is {bad[0]!r}, not "
                               + " or ".join(map(repr, REUSABLE_STATUSES)))
    wanted = ((fp.get("config") or {}).get("run_options") or {}).get("guidance")
    if wanted and "results_by_guidance" in result:
        missing = sorted({f"{float(g):.1f}" for g in wanted}
                         - set(result["results_by_guidance"]))
        if missing:
            return Decision(False, f"stored result is missing guidance {', '.join(missing)}")
    old = entry.get("fingerprint") or {}
    platform = differences(old.get("platform") or {}, plat)
    if platform:
        what = ("instance type" if any(d.startswith(("instance_type", "device")) for d in platform)
                else "SDK")
        return Decision(False, f"{what} changed: " + "; ".join(platform))
    config = differences(old.get("config") or {}, fp.get("config") or {})
    if config:
        return Decision(False, "config changed: " + "; ".join(config))
    try:
        age = _age(entry, now)
    except (KeyError, ValueError):
        return Decision(False, "stored result has no valid measured_at")
    if age > datetime.timedelta(days=max_age_days):
        return Decision(False, f"expired: measured {_days(age)} ago "
                               f"(limit {max_age_days} days)")
    if rerun_on_code_change and code is not None:
        was = entry.get("code") or {}
        if code.get("dirty") or was.get("difflet_tree") != code.get("difflet_tree"):
            return Decision(False, "difflet/ code changed since the measurement")
    left = datetime.timedelta(days=max_age_days) - age
    return Decision(True, f"measured {entry['measured_at']} on {old['platform'].get('instance_type')}"
                          f" ({_days(age)} ago, expires in {_days(left)})", key_of(old))


def decide(slug: str, fp: Mapping[str, Any], *, now: datetime.datetime,
           max_age_days: int = MAX_AGE_DAYS, code: Optional[Mapping[str, Any]] = None,
           rerun_on_code_change: bool = False, store: Optional[Path] = None) -> Decision:
    """Is there a stored entry of ``slug`` this run can reuse? On a miss the reason compares
    against the closest entry: the same key, else the newest one."""
    ents = entries(slug, store)
    if not ents:
        return Decision(False, f"no stored result in {store_path(slug, store).name}")
    entry = ents.get(key_of(fp)) or max(ents.values(), key=lambda e: e.get("measured_at", ""))
    return decide_entry(entry, fp, now=now, max_age_days=max_age_days, code=code,
                        rerun_on_code_change=rerun_on_code_change)


# ------------------------------------------------------------------------------ the reports


def report_cache(entry: Mapping[str, Any]) -> dict:
    """The short ``cache`` summary a report JSON carries for its banner."""
    fp = entry.get("fingerprint") or {}
    out = {"key": key_of(fp), "measured_at": entry.get("measured_at"),
           "instance_type": (fp.get("platform") or {}).get("instance_type"),
           "git_head": (entry.get("code") or {}).get("git_head")}
    if entry.get("last_reused"):
        out["last_reused"] = dict(entry["last_reused"])
    return out


def report_doc(entry: Mapping[str, Any]) -> dict:
    """The report JSON of a stored entry: its result plus the banner summary."""
    return {**entry["result"], "cache": report_cache(entry)}


def invalidate(doc: dict) -> dict:
    """Drop a report's ``cache`` summary: a script re-measured some of its fields in place, so
    the report no longer is the stored measurement (the store itself is untouched)."""
    doc.pop("cache", None)
    return doc


def banner(doc: Mapping[str, Any]) -> list[str]:
    """Markdown lines saying when the numbers were measured and whether they were reused."""
    cache = doc.get("cache")
    if not isinstance(cache, Mapping):
        return []
    head = cache.get("git_head") or "?"
    try:
        until = iso(parse_iso(cache["measured_at"]) + datetime.timedelta(days=MAX_AGE_DAYS))
    except (KeyError, TypeError, ValueError):
        until = "?"
    lines = [f"**Measured:** {cache.get('measured_at')} on {cache.get('instance_type', '?')} "
             f"at `{head[:12]}` (cached in `benchmark/cached_results/`, key `{cache.get('key')}`;"
             f" reusable until {until} at the default {MAX_AGE_DAYS}-day limit)  "]
    reused = cache.get("last_reused")
    if reused:
        rhead = reused.get("git_head") or "?"
        lines.append(f"**REUSED:** these numbers were NOT re-measured; reused on {reused['at']} "
                     f"on {reused.get('instance_type', '?')} at `{rhead[:12]}`, same instance "
                     "type and SDK.  ")
    return lines


def try_reuse(slug: str, fp: Mapping[str, Any], *, now: datetime.datetime, max_age_days: int,
              code: Mapping[str, Any], rerun_on_code_change: bool,
              render: Callable[[dict], str], json_path: Path, md_path: Path,
              store: Optional[Path] = None) -> Decision:
    """Reuse a stored entry if one is still valid: mark it reused and rewrite the report files
    from it. Returns the decision either way; the caller measures when ``not reuse``."""
    d = decide(slug, fp, now=now, max_age_days=max_age_days, code=code,
               rerun_on_code_change=rerun_on_code_change, store=store)
    if d.reuse:
        entry = mark_reused(slug, d.key, now=now, platform=fp["platform"], code=code,
                            store=store)
        doc = report_doc(entry)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(doc, indent=2))
        Path(md_path).write_text(render(doc))
    return d


def record(slug: str, fp: Mapping[str, Any], result: dict, *, code: Mapping[str, Any],
           measured_at: datetime.datetime, max_age_days: int = MAX_AGE_DAYS,
           store: Optional[Path] = None) -> Optional[dict]:
    """After a measurement: store it when it completed and give the report its banner summary
    (``result`` gains ``cache``). A failed run is not stored. Returns the entry or None."""
    if not completed(result):
        return None
    entry = save(slug, fp, {k: v for k, v in result.items() if k != "cache"}, code=code,
                 measured_at=measured_at, now=utc_now(), max_age_days=max_age_days,
                 store=store)
    result["cache"] = report_cache(entry)
    return entry


# ------------------------------------------------------------------------------ status CLI


def status_rows(store: Optional[Path] = None, *, now: datetime.datetime,
                max_age_days: int = MAX_AGE_DAYS) -> list[tuple[str, str]]:
    rows = []
    for path in sorted(Path(store or STORE_DIR).glob("*.json")):
        for key, entry in sorted(entries(path.stem, path.parent).items()):
            plat = (entry.get("fingerprint") or {}).get("platform") or {}
            backend = ((entry.get("fingerprint") or {}).get("config") or {}).get("backend")
            age = _age(entry, now)
            where = f"{plat.get('instance_type', '?')}, {backend}, key {key}"
            if age > datetime.timedelta(days=max_age_days):
                rows.append((path.stem, f"EXPIRED ({_days(age)} old; {where})"))
            else:
                left = datetime.timedelta(days=max_age_days) - age
                reused = " reused" if entry.get("last_reused") else ""
                rows.append((path.stem, f"fresh{reused} ({_days(age)} old, expires in "
                                        f"{_days(left)}; {where})"))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Status of the stored benchmark results")
    p.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    args = p.parse_args(argv)
    rows = status_rows(now=utc_now(), max_age_days=args.max_age_days)
    if not rows:
        print(f"no stored results in {STORE_DIR}")
    for slug, state in rows:
        print(f"{slug:28s} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
