#!/usr/bin/env python3
"""Compare two HunyuanVideo 1.5 latents.pt files (or any two saved tensors of the same shape).

Prints (and with --json writes) cosine similarity, rel-L1 (sum|a-b| / sum|ref|), max-abs and mean-abs
difference, the NaN / Inf counts of both, and their sha256. ``ref`` is the second argument.

    python scripts/hv15_aot_compare_latents.py candidate/latents.pt reference/latents.pt [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch


def _load(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        t = t.get("latents", t.get("sample"))
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor, got {type(t)!r}")
    return t


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a: torch.Tensor, ref: torch.Tensor) -> dict:
    if a.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(ref.shape)}")
    a, ref = a.double(), ref.double()
    diff = (a - ref).abs()
    finite = torch.isfinite(a).all() and torch.isfinite(ref).all()
    return {
        "shape": list(a.shape),
        "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), ref.flatten(), dim=0)) if finite else None,
        "rel_l1": float(diff.sum() / ref.abs().sum()) if finite else None,
        "max_abs": float(diff.max()) if finite else None,
        "mean_abs": float(diff.mean()) if finite else None,
        "nan": [int(torch.isnan(a).sum()), int(torch.isnan(ref).sum())],
        "inf": [int(torch.isinf(a).sum()), int(torch.isinf(ref).sum())],
        "bit_identical": bool(torch.equal(a, ref)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidate")
    p.add_argument("reference")
    p.add_argument("--json", default=None, help="also write the result here")
    args = p.parse_args()
    out = {"candidate": args.candidate, "reference": args.reference,
           "sha256": [_sha256(args.candidate), _sha256(args.reference)],
           **compare(_load(args.candidate), _load(args.reference))}
    text = json.dumps(out, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
