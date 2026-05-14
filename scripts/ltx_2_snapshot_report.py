#!/usr/bin/env python3
"""Report or materialize the Nova LTX-2 diffusers snapshot subset.

By default this is a dry run: it queries Hugging Face metadata, applies Nova's
LTX-2 registry allow-list, and prints the exact files/bytes that would be
downloaded. Pass ``--download`` to call ``snapshot_download`` with the same
allow-list.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Lightricks/LTX-2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-type", default="ltx_2")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--list-files", action="store_true")
    return parser


def _prefix(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "."


def _gb(num_bytes: int) -> float:
    return num_bytes / 1_000_000_000.0


def _selected_filenames(filenames: list[str], allow_patterns: tuple[str, ...]) -> set[str]:
    from huggingface_hub.utils import filter_repo_objects

    return set(filter_repo_objects(filenames, allow_patterns=allow_patterns))


def _snapshot_report(
    *,
    model_id: str,
    revision: str | None,
    model_type: str,
) -> dict[str, Any]:
    from huggingface_hub import model_info

    from nova.registry import resolve_model

    entry = resolve_model(model_id, model_type=model_type)
    if entry.download_patterns is None:
        raise RuntimeError(f"Model {entry.name!r} does not define scoped download patterns.")

    info = model_info(model_id, revision=revision, files_metadata=True)
    siblings = sorted(info.siblings, key=lambda item: item.rfilename)
    selected_names = _selected_filenames(
        [item.rfilename for item in siblings],
        entry.download_patterns,
    )
    selected = [item for item in siblings if item.rfilename in selected_names]
    omitted = [item for item in siblings if item.rfilename not in selected_names]

    by_prefix: dict[str, int] = defaultdict(int)
    for item in selected:
        by_prefix[_prefix(item.rfilename)] += int(item.size or 0)

    return {
        "model_id": model_id,
        "revision": revision,
        "model_type": entry.name,
        "repo_file_count": len(siblings),
        "selected_file_count": len(selected),
        "omitted_file_count": len(omitted),
        "repo_bytes": sum(int(item.size or 0) for item in siblings),
        "selected_bytes": sum(int(item.size or 0) for item in selected),
        "omitted_bytes": sum(int(item.size or 0) for item in omitted),
        "selected_by_prefix": dict(sorted(by_prefix.items())),
        "selected_files": [
            {"path": item.rfilename, "bytes": int(item.size or 0)} for item in selected
        ],
        "omitted_files": [
            {"path": item.rfilename, "bytes": int(item.size or 0)} for item in omitted
        ],
        "allow_patterns": list(entry.download_patterns),
    }


def _download_snapshot(
    *,
    model_id: str,
    revision: str | None,
    model_type: str,
    local_files_only: bool,
) -> str:
    from huggingface_hub import snapshot_download

    from nova.registry import resolve_model

    entry = resolve_model(model_id, model_type=model_type)
    if entry.download_patterns is None:
        raise RuntimeError(f"Model {entry.name!r} does not define scoped download patterns.")
    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=list(entry.download_patterns),
    )


def main() -> int:
    args = build_parser().parse_args()
    report = _snapshot_report(
        model_id=args.model_id,
        revision=args.revision,
        model_type=args.model_type,
    )

    print(f"[ltx2-snapshot] model_id={report['model_id']}")
    print(
        "[ltx2-snapshot] selected "
        f"{report['selected_file_count']}/{report['repo_file_count']} files, "
        f"{_gb(report['selected_bytes']):.2f} GB "
        f"(repo total {_gb(report['repo_bytes']):.2f} GB)"
    )
    for prefix, num_bytes in report["selected_by_prefix"].items():
        print(f"[ltx2-snapshot] selected {prefix}: {_gb(num_bytes):.2f} GB")
    if args.list_files:
        for item in report["selected_files"]:
            print(f"[ltx2-snapshot] file {item['path']} {_gb(item['bytes']):.4f} GB")

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[ltx2-snapshot] wrote {output}")

    if args.download:
        path = _download_snapshot(
            model_id=args.model_id,
            revision=args.revision,
            model_type=args.model_type,
            local_files_only=args.local_files_only,
        )
        print(f"[ltx2-snapshot] downloaded {path}")
        return 0

    print("[ltx2-snapshot] dry run only; pass --download to materialize this subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
