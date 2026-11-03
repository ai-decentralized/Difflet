"""Inspect a saved LTX-2 frames tensor (out.pt): shape, dtype, range, per-frame std, temporal diff;
write the mid frame as a PNG. Frames are (B, F, C, H, W) in [0, 1] (LTX-2's postprocess_video
layout, see difflet/cli/orchestrators/ltx_2.py _save_video). Read-only on out.pt."""
import json
import sys

import numpy as np
import torch
from PIL import Image

path, png = sys.argv[1], sys.argv[2]
x = torch.load(path, map_location="cpu", weights_only=True)
info = {"path": path, "shape": list(x.shape), "dtype": str(x.dtype)}
v = x.detach().float()
if v.ndim == 5:
    v = v[0]  # (F, C, H, W)
f = v.clamp(0, 1).numpy()
n = f.shape[0]
std = f.reshape(n, -1).std(axis=1)
tdiff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)) if n > 1 else np.array([0.0])
mid = (f[n // 2].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
Image.fromarray(mid).save(png)
info.update({"frames": n, "min": float(v.min()), "max": float(v.max()),
             "std_min": float(std.min()), "std_mean": float(std.mean()),
             "temporal_absdiff_mean": float(tdiff.mean()),
             "spatial_neighbour_absdiff": float(np.abs(np.diff(f, axis=3)).mean()), "png": png})
print("PT " + json.dumps(info))
