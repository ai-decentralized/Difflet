"""Side-by-side frame grid (bf16 top row, fp8 bottom row) + |diff| row for visual inspection."""
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image

ref, test, out = sys.argv[1:4]
a = iio.imread(ref, plugin="pyav")
b = iio.imread(test, plugin="pyav")
n = min(len(a), len(b))
idx = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
w, h = a.shape[2], a.shape[1]
scale = 0.5
tw, th = int(w * scale), int(h * scale)
grid = Image.new("RGB", (tw * len(idx), th * 3), "black")
for col, i in enumerate(idx):
    fa, fb = a[i], b[i]
    diff = np.clip(np.abs(fa.astype(np.int16) - fb.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    for row, fr in enumerate((fa, fb, diff)):
        grid.paste(Image.fromarray(fr).resize((tw, th)), (col * tw, row * th))
grid.save(out)
print("frames", n, "shape", a.shape, "->", out, "diff max", int(np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16)).max()),
      "mean", round(float(np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16)).mean()), 3))
