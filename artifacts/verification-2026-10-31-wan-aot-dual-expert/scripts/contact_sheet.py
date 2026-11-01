"""Frames 0, 4 and 8 of each mp4, one row per video, into one PNG: contact_sheet.py OUT.png A.mp4 ..."""
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

rows = []
for path in sys.argv[2:]:
    frames = iio.imread(path, plugin="pyav")
    picks = [frames[i] for i in (0, len(frames) // 2, len(frames) - 1)]
    row = Image.fromarray(np.concatenate(picks, axis=1)).resize((3 * 416, 240))
    ImageDraw.Draw(row).text((8, 8), path.rsplit("/", 1)[-1], fill=(255, 255, 0))
    rows.append(np.asarray(row))
Image.fromarray(np.concatenate(rows, axis=0)).save(sys.argv[1])
print(sys.argv[1], [r.shape for r in rows])
