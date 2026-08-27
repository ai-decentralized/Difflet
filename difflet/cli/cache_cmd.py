"""`difflet cache ls` — list compiled artifacts by reading their manifests.

Artifact dirs are pure hashes (schema v5); the manifests inside are the
authoritative record of what each dir contains. This command is the
human-readable index over them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _dir_size_bytes(path: Path) -> int:
    total = 0
    seen_inodes: set[int] = set()
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                stat = os.stat(os.path.join(root, name))
            except OSError:
                continue
            # Hardlinked weight shards are shared across artifacts; count each
            # inode once per artifact but flag nothing — sizes stay per-dir.
            if stat.st_ino in seen_inodes:
                continue
            seen_inodes.add(stat.st_ino)
            total += stat.st_size
    return total


def _fmt_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TiB"


def _shapes_token(cache_inputs: dict) -> str:
    shapes = cache_inputs.get("shapes")
    if not shapes:
        return "-"
    return "+".join(
        "x".join(str(dim) for dim in shape if dim is not None) for shape in shapes
    )


def _collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for manifest in sorted(root.glob("*/*/manifest.json")):
        try:
            with manifest.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        inputs = data.get("cache_inputs", {})
        artifact_dir = manifest.parent
        rows.append(
            {
                "dir": str(artifact_dir.relative_to(root)),
                "component": inputs.get("component") or inputs.get("model_name") or "-",
                "model": inputs.get("model_id", "-"),
                "shapes": _shapes_token(inputs),
                "tp": inputs.get("tp") or inputs.get("parallel", {}).get("tp_degree", "-"),
                "dtype": inputs.get("dtype", "-"),
                "schema": data.get("schema_version", "-"),
                "size_bytes": _dir_size_bytes(artifact_dir),
                "mtime": datetime.fromtimestamp(
                    manifest.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return rows


def run_cache_command(args) -> int:
    if args.cache_action != "ls":
        raise SystemExit(f"unknown cache action {args.cache_action!r}")
    from difflet import envs

    root = Path(args.cache_dir or envs.DIFFLET_COMPILE_CACHE).expanduser()
    rows = _collect(root)
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"no manifests found under {root}")
        return 0
    headers = ("DIR", "COMPONENT", "MODEL", "SHAPES", "TP", "DTYPE", "SCHEMA", "SIZE", "UPDATED (UTC)")
    table = [
        (
            row["dir"],
            str(row["component"]),
            str(row["model"]),
            row["shapes"],
            str(row["tp"]),
            str(row["dtype"]),
            str(row["schema"]),
            _fmt_size(row["size_bytes"]),
            row["mtime"],
        )
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(line[i]) for line in table)) for i in range(len(headers))]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    for line in table:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
    print(f"\n{len(rows)} artifact(s) under {root}")
    return 0
