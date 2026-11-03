"""Decode an mp4, report frame count, size, per-frame std/mean, temporal diff; save the mid frame PNG."""
import json
import sys

import imageio.v3 as iio
import numpy as np

path = sys.argv[1]
png = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0] + "_mid.png"
frames = iio.imread(path, plugin="pyav") if True else None
frames = np.asarray(frames)
n, h, w, c = frames.shape
f = frames.astype(np.float32)
std = f.reshape(n, -1).std(axis=1)
mean = f.reshape(n, -1).mean(axis=1)
tdiff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)) if n > 1 else np.array([0.0])
# spatial smoothness: neighbour-pixel diff, low for natural images, ~high for noise
sp = np.abs(np.diff(f, axis=2)).mean()
iio.imwrite(png, frames[n // 2])
print("MP4 " + json.dumps({
    "path": path, "frames": n, "h": h, "w": w,
    "std_min": float(std.min()), "std_mean": float(std.mean()), "mean_mean": float(mean.mean()),
    "temporal_absdiff_mean": float(tdiff.mean()), "spatial_neighbour_absdiff": float(sp),
    "png": png}))
