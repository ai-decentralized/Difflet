#!/usr/bin/env python3
"""Map a raw VAE decode (1,3,F,H,W) in about [-1,1] to the layout quality_metrics.py reads.

The map is diffusers' own video postprocess, ``(x/2 + 0.5).clamp(0, 1)``, computed in float32
(a bf16 input is upcast exactly first), then permuted to (1,F,3,H,W). The CPU references' out.pt
is exactly this map of their out_raw.pt (checked bit for bit inside each reference run); this
script applies the same map to the device out.pt files and to Wan's A2 host_fp32_frames.pt, so
every frames input goes through one rule before quality_metrics' uint8 rule.

Usage: frames_map.py IN_raw.pt OUT_mapped.pt [--check-equal MAPPED_REF.pt]
Prints a JSON line: input sha256 / dtype / shape / range, output sha256 / shape.
"""
import hashlib
import json
import sys

import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


src, dst = sys.argv[1], sys.argv[2]
x = torch.load(src, map_location="cpu", weights_only=True)
if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 3:
    raise SystemExit(f"{src}: shape {tuple(x.shape)}, expected (1,3,F,H,W)")
xf = x.to(torch.float32)
y = (xf / 2 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4).contiguous()
torch.save(y, dst)
rec = {"in": src, "in_sha256": sha(src), "in_dtype": str(x.dtype), "in_shape": list(x.shape),
       "in_min": float(xf.min()), "in_max": float(xf.max()),
       "out": dst, "out_sha256": sha(dst), "out_shape": list(y.shape),
       "map": "(float32(x)/2 + 0.5).clamp(0,1).permute(0,2,1,3,4)"}
if len(sys.argv) > 4 and sys.argv[3] == "--check-equal":
    ref = torch.load(sys.argv[4], map_location="cpu", weights_only=True)
    rec["check_equal"] = sys.argv[4]
    rec["bit_identical"] = bool(ref.dtype == y.dtype and ref.shape == y.shape and torch.equal(ref, y))
print(json.dumps(rec))
