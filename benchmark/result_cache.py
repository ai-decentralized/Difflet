"""Reuse committed benchmark results instead of re-measuring them on every new instance.

The cache is the result file itself: ``benchmark/<device>/<slug>.json`` (committed to git, so
it outlives the instance) carries a ``cache`` block written when the numbers were measured:

    "cache": {
      "schema": 1,
      "key": "<sha256 of config + platform, 16 hex>",
      "measured_at": "2026-10-24T12:00:00Z",
      "expires_at":  "2026-11-23T12:00:00Z",
      "fingerprint": {"config": {...}, "platform": {...}},
      "code": {"git_head": ..., "difflet_tree": ..., "dirty": ...},
      "last_reused": {"at": ..., "instance_type": ..., "git_head": ...}   # only once reused
    }

A later run reuses the file, without touching the device, when all of these hold:

* the file's status is ``ok`` or ``flagged`` (a failed or partial run is never served);
* the config fingerprint matches (model, revision, shape, steps, parallel layout, prompt, seed,
  flags, backend and the run options such as ``--iters``);
* the platform matches: the instance type, the accelerator ``device_info()`` reports, and the
  SDK (the adapter's ``toolchain()`` packages plus the OS-level Neuron runtime/driver packages
  or the NVIDIA driver);
* ``measured_at`` is younger than ``max_age_days`` (30 by default). Reusing does not refresh
  the clock: only a re-measurement does.

The code identity (git HEAD, ``difflet/`` tree) is recorded and shown, but a change only
forces a re-run under ``--rerun-on-code-change``. Results written before this module have no
``cache`` block and are always re-measured.

    python -m benchmark.result_cache                 # status of every committed result
    python -m benchmark.result_cache --device trn2 --max-age-days 14
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
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
    """The EC2 instance type (IMDSv2, then IMDSv1); ``unknown`` off EC2."""
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


def cache_block(fp: Mapping[str, Any], *, code: Mapping[str, Any],
                measured_at: datetime.datetime, max_age_days: int = MAX_AGE_DAYS) -> dict:
    return {"schema": SCHEMA, "key": key_of(fp), "measured_at": iso(measured_at),
            "expires_at": iso(measured_at + datetime.timedelta(days=max_age_days)),
            "fingerprint": dict(fp), "code": dict(code)}


# ------------------------------------------------------------------------------ decision


@dataclasses.dataclass(frozen=True)
class Decision:
    reuse: bool
    reason: str


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


def _age(cache: Mapping[str, Any], now: datetime.datetime) -> datetime.timedelta:
    return now - parse_iso(cache["measured_at"])


def _days(td: datetime.timedelta) -> str:
    return f"{td.total_seconds() / 86400:.1f} days"


def decide(doc: Optional[Mapping[str, Any]], fp: Mapping[str, Any], *,
           now: datetime.datetime, max_age_days: int = MAX_AGE_DAYS,
           code: Optional[Mapping[str, Any]] = None,
           rerun_on_code_change: bool = False) -> Decision:
    """Reuse ``doc`` (a committed result) for a run whose fingerprint is ``fp``?"""
    if doc is None:
        return Decision(False, "no committed result")
    cache = doc.get("cache")
    if not isinstance(cache, Mapping) or cache.get("schema") != SCHEMA:
        return Decision(False, "committed result has no cache fingerprint "
                               "(written before the result cache)")
    statuses = [doc.get("status")] + [(r or {}).get("status") for r in
                                      (doc.get("results_by_guidance") or {}).values()]
    bad = [st for st in statuses if st not in REUSABLE_STATUSES]
    if bad:
        return Decision(False, f"committed result's status is {bad[0]!r}, not "
                               + " or ".join(map(repr, REUSABLE_STATUSES)))
    old = cache.get("fingerprint") or {}
    platform = differences(old.get("platform") or {}, fp.get("platform") or {})
    if platform:
        what = ("instance type" if any(d.startswith(("instance_type", "device")) for d in platform)
                else "SDK")
        return Decision(False, f"{what} changed: " + "; ".join(platform))
    if not (fp.get("platform") or {}).get("sdk"):
        return Decision(False, "SDK versions could not be read on this host")
    config = differences(old.get("config") or {}, fp.get("config") or {})
    if config:
        return Decision(False, "config changed: " + "; ".join(config))
    try:
        age = _age(cache, now)
    except (KeyError, ValueError):
        return Decision(False, "committed result has no valid measured_at")
    if age > datetime.timedelta(days=max_age_days):
        return Decision(False, f"expired: measured {_days(age)} ago "
                               f"(limit {max_age_days} days)")
    if rerun_on_code_change and code is not None:
        was = cache.get("code") or {}
        if code.get("dirty") or was.get("difflet_tree") != code.get("difflet_tree"):
            return Decision(False, "difflet/ code changed since the measurement")
    left = datetime.timedelta(days=max_age_days) - age
    return Decision(True, f"measured {cache['measured_at']} on "
                          f"{(old.get('platform') or {}).get('instance_type')} "
                          f"({_days(age)} ago, expires in {_days(left)})")


def mark_reused(doc: dict, *, now: datetime.datetime, platform: Mapping[str, Any],
                code: Mapping[str, Any]) -> dict:
    """Record that ``doc`` was reused (not re-measured); ``measured_at`` stays as it was."""
    doc["cache"]["last_reused"] = {"at": iso(now),
                                   "instance_type": platform.get("instance_type"),
                                   "git_head": code.get("git_head")}
    return doc


def banner(doc: Mapping[str, Any]) -> list[str]:
    """Markdown lines saying when the numbers were measured and whether they were reused."""
    cache = doc.get("cache")
    if not isinstance(cache, Mapping):
        return []
    plat = (cache.get("fingerprint") or {}).get("platform") or {}
    head = (cache.get("code") or {}).get("git_head") or "?"
    lines = [f"**Measured:** {cache.get('measured_at')} on {plat.get('instance_type', '?')} "
             f"at `{head[:12]}` (cache key `{cache.get('key')}`, reusable until "
             f"{cache.get('expires_at')})  "]
    reused = cache.get("last_reused")
    if reused:
        rhead = reused.get("git_head") or "?"
        lines.append(f"**REUSED:** these numbers were NOT re-measured; reused on {reused['at']} "
                     f"on {reused.get('instance_type', '?')} at `{rhead[:12]}`, same instance "
                     "type and SDK.  ")
    return lines


# ------------------------------------------------------------------------------ files


def load(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def try_reuse(path: Path, fp: Mapping[str, Any], *, now: datetime.datetime,
              max_age_days: int, code: Mapping[str, Any], rerun_on_code_change: bool,
              render: Callable[[dict], str], md_path: Path) -> Decision:
    """Reuse the result at ``path`` if it is still valid: mark it reused and re-render its
    report. Returns the decision either way; the caller measures when ``not reuse``."""
    doc = load(path)
    d = decide(doc, fp, now=now, max_age_days=max_age_days, code=code,
               rerun_on_code_change=rerun_on_code_change)
    if d.reuse:
        mark_reused(doc, now=now, platform=fp["platform"], code=code)
        Path(path).write_text(json.dumps(doc, indent=2))
        Path(md_path).write_text(render(doc))
    return d


# ------------------------------------------------------------------------------ status CLI


def status_rows(device_dir: Path, *, now: datetime.datetime,
                max_age_days: int = MAX_AGE_DAYS) -> list[tuple[str, str]]:
    rows = []
    for path in sorted(Path(device_dir).glob("*.json")):
        doc = load(path)
        cache = (doc or {}).get("cache")
        if not isinstance(cache, Mapping):
            rows.append((path.stem, "no fingerprint: re-measured on the next run"))
            continue
        age = _age(cache, now)
        plat = (cache.get("fingerprint") or {}).get("platform") or {}
        where = f"{plat.get('instance_type', '?')}, status {doc.get('status')}"
        if age > datetime.timedelta(days=max_age_days):
            rows.append((path.stem, f"EXPIRED ({_days(age)} old; {where})"))
        else:
            left = datetime.timedelta(days=max_age_days) - age
            reused = " reused" if cache.get("last_reused") else ""
            rows.append((path.stem, f"fresh{reused} ({_days(age)} old, expires in {_days(left)};"
                                    f" {where})"))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Status of the committed benchmark results")
    p.add_argument("--device", default=None, help="device folder (default: every folder)")
    p.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    args = p.parse_args(argv)
    root = REPO_ROOT / "benchmark"
    dirs = [root / args.device] if args.device else sorted(
        d for d in root.iterdir() if d.is_dir() and any(d.glob("*.json")))
    now = utc_now()
    for d in dirs:
        print(f"{d.name}/")
        for slug, state in status_rows(d, now=now, max_age_days=args.max_age_days):
            print(f"  {slug:28s} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
