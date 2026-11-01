"""Frames 0, 1 and 8: CPU diffusers dual reference (row 1) vs AOT dual (row 2) -> PNG."""
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

C = "/home/ubuntu/campaigns/wan-dual-2026-10-31"
ref = torch.load(f"{C}/ref/diffusers_bf16_wan_dual_s42.pt", map_location="cpu")[0]  # [C, F, H, W]
ref = (ref.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()
aot = iio.imread(f"{C}/out/dual_gen1.mp4", plugin="pyav")
picks = (0, 1, 8)
rows = [np.concatenate([frames[i] for i in picks], axis=1) for frames in (ref, aot)]
Image.fromarray(np.concatenate(rows, axis=0)).resize((3 * 416, 2 * 240)).save(sys.argv[1])
print(sys.argv[1])
