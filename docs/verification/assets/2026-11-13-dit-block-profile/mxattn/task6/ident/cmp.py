"""BR37: bit identity (int16 views) of the fork with allow_vector_exp=True and no MX flags
(fvexp.<shape>.out.pt) against shipped attention_cte (cte.<shape>.out.pt), per shape; on a
difference, the count, the max |diff| and where. Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
Writes ident.json and prints it."""
import json
from pathlib import Path

import torch

D = Path("/mnt/nvme/difflet-mxattn/task6/ident")
out = {}
for shape in ("qwen_1024", "wan_81f", "hv1_L"):
    a, b = D / f"cte.{shape}.out.pt", D / f"fvexp.{shape}.out.pt"
    if not (a.exists() and b.exists()):
        out[shape] = "missing"
        continue
    x, y = torch.load(a), torch.load(b)
    r = {"shape": list(x.shape), "dtype": str(x.dtype), "same_shape": x.shape == y.shape}
    if x.shape == y.shape:
        ne = x.view(torch.int16) != y.view(torch.int16)
        r["differing"] = int(ne.sum())
        r["bit_identical"] = r["differing"] == 0
        r["nonfinite"] = [int((~torch.isfinite(t.float())).sum()) for t in (x, y)]
        if r["differing"]:
            d = (x.float() - y.float()).abs()
            i = int(d.argmax())
            r.update(max_abs_diff=float(d.max()), at=list(torch.unravel_index(torch.tensor(i), x.shape)),
                     heads_differing=sorted({int(h) for h in ne.nonzero()[:, 0].unique()})[:20])
    for arm in ("cte", "fvexp"):
        j = json.loads((D / f"{arm}.{shape}.json").read_text())
        r[arm] = {k: j.get(k) for k in ("impl", "qk_mx", "pv_mx", "fork_vector_exp", "rel_l2", "cosine")}
    out[shape] = r
(D / "ident.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
