#!/usr/bin/env python3
"""Record remote Hugging Face metadata for missing F3.0b artifacts.

This is a no-download audit. It captures repository size and gated/private
state so the local artifact inventory can distinguish "missing because not
downloaded / not accessible / too large for current disk" from negative F3.0b
performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPOS = {
    "ltx_2": "Lightricks/LTX-2",
    "flux": "black-forest-labs/FLUX.1-dev",
}


def audit_remote_repos(repos: dict[str, str]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    rows: list[dict[str, Any]] = []
    for label, repo_id in repos.items():
        try:
            info = api.model_info(repo_id, files_metadata=True)
            files = []
            total_bytes = 0
            for sibling in info.siblings:
                size = int(getattr(sibling, "size", None) or 0)
                total_bytes += size
                files.append({"path": sibling.rfilename, "size_bytes": size})
            rows.append(
                {
                    "label": label,
                    "repo_id": repo_id,
                    "ok": True,
                    "private": bool(info.private),
                    "gated": getattr(info, "gated", None),
                    "sha": info.sha,
                    "file_count": len(files),
                    "total_size_bytes": total_bytes,
                    "total_size_gb": round(total_bytes / 1e9, 2),
                    "largest_files": sorted(
                        files,
                        key=lambda item: item["size_bytes"],
                        reverse=True,
                    )[:12],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "label": label,
                    "repo_id": repo_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "schema": "difflet-f3-0b-remote-repo-audit-v1",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = audit_remote_repos(DEFAULT_REPOS)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
