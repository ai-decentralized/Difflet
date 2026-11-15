"""Task 5b (c): latents finiteness/stats per arm, and image PSNR / latents cosine against Task 5's
10-step arms (linear = attention bf16; both = Task 5's mxfp8 switch, i.e. qk)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

B = Path("/mnt/nvme/difflet-mxattn/task5b")
T5 = Path("/mnt/nvme/difflet-mxattn/task5/smoke")


def img(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    m = ((img(a) - img(b)) ** 2).mean()
    return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


def lat(p):
    return torch.load(p).float().flatten()


arms = {f"quant_{v}": (B / f"smoke/quant_{v}/gen.png", B / f"smoke/quant_{v}/work/latents.pt")
        for v in ("qk", "pv", "qk+pv")}
if (B / "droot/gen_pv.png").exists():
    arms["noquant_pv"] = (B / "droot/gen_pv.png", B / "droot/work_pv/latents.pt")
ref = {n: (T5 / n / "gen.png", T5 / n / "work/latents.pt") for n in ("linear", "both")}
for name, (png, lt) in arms.items():
    x = lat(lt)
    row = f"{name}: finite={bool(x.isfinite().all())} mean={x.mean():.4f} std={x.std():.4f}"
    if name.startswith("quant"):
        for r, (rp, rl) in ref.items():
            y = lat(rl)
            row += (f" | vs t5 {r}: psnr={psnr(png, rp):.2f} dB "
                    f"cos={float(x @ y / (x.norm() * y.norm())):.4f}")
    print(row)
