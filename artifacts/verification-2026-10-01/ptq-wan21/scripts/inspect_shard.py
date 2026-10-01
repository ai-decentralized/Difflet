"""Inspect fp8 probe checkpoints: dtypes, NaN/inf counts, scale values."""
import glob
import json
import sys

import torch
from safetensors.torch import load_file

for path in sys.argv[1:]:
    print("==", path)
    sd = load_file(path)
    for k in sorted(sd):
        t = sd[k]
        if t.dtype == torch.float8_e4m3fn:
            f = t.float()
            print(f"  {k:60s} {str(t.dtype):22s} {tuple(t.shape)!s:18s} nan={int(torch.isnan(f).sum())} "
                  f"absmax={float(f.abs().max()):.4g}")
        elif "scale" in k:
            print(f"  {k:60s} {str(t.dtype):22s} {tuple(t.shape)!s:18s} values={t.flatten()[:4].tolist()}")
        else:
            f = t.float()
            flag = "" if torch.isfinite(f).all() else "  NONFINITE"
            if flag or "blocks.0.attn1" in k or k.startswith("proj_out"):
                print(f"  {k:60s} {str(t.dtype):22s} {tuple(t.shape)!s:18s}{flag}")
    dtypes = {}
    for t in sd.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    print("  dtype histogram:", dtypes)
