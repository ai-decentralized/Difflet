#!/usr/bin/env python3
"""Inventory local artifacts needed to run F3.0b production profiles.

The gate audit answers "what did measured JSON prove?". This inventory answers
"which required production profiles can be run from the current workstation
state?" so missing artifacts do not get confused with negative evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = {
    "qwen_image": {
        "source": "/home/ubuntu/.cache/huggingface/hub/qwen-image-real",
        "compiled": (
            ".difflet-cache/qwen_image_transformer_full/qwen_image/"
            "14efb7c830dbb71e/transformer/model.pt"
        ),
        "bundle": ".difflet-cache/qwen_image_dit_inputs/full_1024_4step_active.safetensors",
    },
    "hunyuan_video": {
        "source": (
            ".difflet-cache/f3_hunyuan_n4_4d8s1r/source/transformer/"
            "diffusion_pytorch_model.safetensors"
        ),
        "compiled": ".difflet-cache/f3_hunyuan_n4_4d8s1r/compiled/transformer/model.pt",
        "bundle": ".difflet-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors",
    },
    "ltx_2": {
        "source": "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2",
        "compiled": (
            ".difflet-cache/ltx_2_transformer_full/ltx_2/"
            "7af2c71df74f264e/transformer/model.pt"
        ),
        "bundle": ".difflet-cache/ltx_2_dit_inputs/full_512x768x121_4step.safetensors",
    },
    "flux": {
        "source": "/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev",
        "compiled": ".difflet-cache/flux/transformer/model.pt",
        "bundle": None,
    },
}


def _exists(path: str | None, root: Path) -> bool:
    if path is None:
        return False
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.exists()


def audit_artifacts(root: Path, artifacts: dict[str, dict[str, str | None]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, spec in artifacts.items():
        checks = {
            name: {
                "path": value,
                "exists": _exists(value, root),
            }
            for name, value in spec.items()
        }
        missing = [name for name, item in checks.items() if not item["exists"]]
        rows.append(
            {
                "label": label,
                "checks": checks,
                "ready_for_f3_0b": not missing,
                "missing": missing,
            }
        )
    disk = shutil.disk_usage(root)
    return {
        "schema": "difflet-f3-0b-artifact-inventory-v1",
        "root": str(root),
        "disk_available_bytes": int(disk.free),
        "disk_available_gb": round(disk.free / 1e9, 2),
        "rows": rows,
        "ready_labels": sorted(row["label"] for row in rows if row["ready_for_f3_0b"]),
        "missing_labels": sorted(row["label"] for row in rows if not row["ready_for_f3_0b"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = audit_artifacts(Path(args.root).resolve(), DEFAULT_ARTIFACTS)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
