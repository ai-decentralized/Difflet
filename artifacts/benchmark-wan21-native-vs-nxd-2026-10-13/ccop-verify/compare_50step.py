"""Fixed build (ccop overlap) vs baseline TorchNeuron at 50 steps, g=5 and g=1: final latents and
decoded mp4 frames. Writes ccop-verify/compare_50step.json."""
import glob
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

B = Path("/home/ubuntu/wan21-bench")
out = {}
for g in (5, 1):
    base = torch.load(glob.glob(str(B / f"neuron/warm1-g{g}/probe/latents-*.pt"))[0])
    fix = torch.load(glob.glob(str(B / f"ccop-verify/warm1-g{g}/probe/latents-*.pt"))[0])
    a, b = base["final"].flatten().double(), fix["final"].flatten().double()
    ua, ub = a - base["initial"].flatten().double(), b - fix["initial"].flatten().double()
    entry = {
        "initial_equal": bool(torch.equal(base["initial"], fix["initial"])),
        "final_bitwise_equal": bool(torch.equal(base["final"], fix["final"])),
        "final_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "final_rel_l2": float((a - b).norm() / a.norm()),
        "update_cosine": float(torch.nn.functional.cosine_similarity(ua, ub, dim=0)),
    }
    fb = iio.imread(B / f"neuron/warm1-g{g}/out.mp4").astype(np.float64)
    ff = iio.imread(B / f"ccop-verify/warm1-g{g}/out.mp4").astype(np.float64)
    mse = ((fb - ff) ** 2).reshape(fb.shape[0], -1).mean(1)
    entry["mp4_psnr_per_frame_db"] = [round(float(10 * np.log10(255 ** 2 / m)), 2) if m > 0 else None
                                      for m in mse]
    out[f"g{g}"] = entry
(B / "ccop-verify/compare_50step.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
