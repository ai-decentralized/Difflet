"""Inspect a saved HunyuanVideo AoT frames tensor (out.pt): shape, dtype, range, per-frame std,
temporal diff; write the mid frame as a PNG (2026-10-26; not in repo). Read-only on out.pt.

Main's generate stage (difflet/cli/orchestrators/hunyuan_video.py, _stage_generate) saves
`output.frames` (the VAE decode, (1, C, T, H, W), nominally [-1, 1]) when --output is not .mp4.
The mp4 path (_save_video) maps it with clamp(-1, 1), (x + 1) / 2, clamp(0, 1); the same map is
used here for the statistics and the PNG, so they describe what the mp4 would have shown.
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

path, png = sys.argv[1], sys.argv[2]
x = torch.load(path, map_location="cpu", weights_only=True)
info = {"path": path, "shape": list(x.shape), "dtype": str(x.dtype)}
v = x.detach().float()
info.update({"raw_min": float(v.min()), "raw_max": float(v.max()), "raw_mean": float(v.mean())})
if v.ndim == 5:
    v = v[0]  # (C, T, H, W)
v = v.permute(1, 0, 2, 3)  # (T, C, H, W)
f = ((v.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1).numpy()
n = f.shape[0]
std = f.reshape(n, -1).std(axis=1)
tdiff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)) if n > 1 else np.array([0.0])
mid = (f[n // 2].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
Image.fromarray(mid).save(png)
info.update({"frames": n, "std_min": float(std.min()), "std_mean": float(std.mean()),
             "temporal_absdiff_mean": float(tdiff.mean()),
             "spatial_neighbour_absdiff": float(np.abs(np.diff(f, axis=3)).mean()), "png": png})
print("PT " + json.dumps(info))
