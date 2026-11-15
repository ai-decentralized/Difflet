"""Task 5b device check (b) from kernel/: per variant and shape, bit identity of the routed op
(Difflet's attention() under DIFFLET_ATTN_QUANT=<variant>) against the probe's fork with the same
flags and the VectorE exp (int16 views), each arm's accuracy against the variant's Task 1 mirror,
and the flags each JSON records. Writes kernel/summary.json and prints it."""
import json
from pathlib import Path

import torch

K = Path("/mnt/nvme/difflet-mxattn/task5b/kernel")


def same(a: Path, b: Path) -> dict:
    x, y = torch.load(a), torch.load(b)
    eq = x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16))
    return {"bit_identical": bool(eq), "shape": list(x.shape),
            "differing": None if x.shape != y.shape
            else int((x.view(torch.int16) != y.view(torch.int16)).sum())}


out = {}
for shape in ("qwen_1024", "wan_81f"):
    for v in ("qk", "pv", "qk+pv"):
        names = {a: f"{K}/{a}.{v}.{shape}" for a in ("routed", "fvexp")}
        if not all(Path(p + ".json").exists() for p in names.values()):
            out[f"{shape} {v}"] = "missing"
            continue
        acc = {a: json.loads(Path(p + ".json").read_text()) for a, p in names.items()}
        out[f"{shape} {v}"] = {
            "bits routed vs fvexp": same(Path(names["routed"] + ".out.pt"),
                                         Path(names["fvexp"] + ".out.pt")),
            **{a: {k: r[k] for k in ("impl", "qk_mx", "pv_mx", "fork_vector_exp", "reference",
                                     "rel_l2", "cosine", "nonfinite", "first_call_s")}
               for a, r in acc.items()},
        }
(K / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
