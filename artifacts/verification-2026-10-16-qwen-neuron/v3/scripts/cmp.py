import json
import sys

import numpy as np
import torch
from PIL import Image

W = "/home/ubuntu/qwen-v3-work"


def img(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)


def lat(p):
    return torch.load(p, map_location="cpu").float()


res = {}
pairs = [("tp2cp2_ulysses_s50_vs_tp4_s50", "tp2cp2-ulysses-s50", "tp4-s50-gen1"),
         ("tp4_eager_s50_vs_tp4_compile_s50", "tp4-eager-s50", "tp4-s50-gen1"),
         ("tp2cp2_ulysses_s50_vs_tp4_eager_s50", "tp2cp2-ulysses-s50", "tp4-eager-s50"),
         ("tp4_s30_vs_tp4_s50", "tp4-s30", "tp4-s50-gen1")]
for name, a, b in pairs:
    A, B = img(f"{W}/{a}/out.png"), img(f"{W}/{b}/out.png")
    d = A - B
    mse = float((d ** 2).mean())
    r = {"png_mean_abs_diff_0_255": float(np.abs(d).mean()),
         "png_max_abs_diff": float(np.abs(d).max()),
         "png_psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf"),
         "png_frac_pixels_maxchan_diff_gt_16": float((np.abs(d).max(axis=2) > 16).mean())}
    la, lb = lat(f"{W}/{a}/work/latents.pt"), lat(f"{W}/{b}/work/latents.pt")
    dl = (la - lb).flatten()
    lbf = lb.flatten()
    r.update(latents_rel_l1=float(dl.abs().sum() / lbf.abs().sum()),
             latents_cosine=float(torch.nn.functional.cosine_similarity(la.flatten(), lbf, dim=0)),
             latents_max_abs=float(dl.abs().max()))
    res[name] = r
print(json.dumps(res, indent=2))
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(res, indent=2))
