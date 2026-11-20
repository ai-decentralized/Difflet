"""Experiment E1 for Wan: table.py's per-layer softmax_max rule on each Wan expert's
self-attention (``blocks.{i}.attn1``; cross-attention stays on attention_cte), at wan_81f's key
length Sk = 21 x 30 x 52 = 32,760 (480x832x81). Formulas unchanged from table.py / bounds.py:
  sigma = sqrt(sum(gamma_q^2 * gamma_k^2) / d)        (max over the 40 heads)
  softmax_max = 0.88 * sigma * sqrt(2 ln Sk)
  hard_bound = sqrt(d) * max|gamma_q| * max|gamma_k|  (max over heads)
  ok = hard_bound - softmax_max <= 80
One table per expert, keyed "<revision[:8]>/<expert dir>"; WanTransformer3DModel's
``add_constmax_sigma_`` picks the one whose key matches the directory it loads.

Cross-check: with BOUNDS_JSON (bounds.json of feat/mxfp8-sage-attention section 12), every Wan 2.1
layer must have its per-call sigma_max_over_heads, softmax_max_max_over_heads and call_gap to 1e-4.

usage: table_wan.py OUT_JSON [BOUNDS_JSON]
"""
import json
import math
import sys
from pathlib import Path

from safetensors import safe_open

from table import D, GAP_LIMIT, HUB

SK = 32760
EXPERTS = [  # (hub repo, revision, expert dir)
    ("models--Wan-AI--Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68", "transformer"),
    ("models--Wan-AI--Wan2.2-T2V-A14B-Diffusers", "5be7df9619b54f4e2667b2755bc6a756675b5cd7", "transformer"),
    ("models--Wan-AI--Wan2.2-T2V-A14B-Diffusers", "5be7df9619b54f4e2667b2755bc6a756675b5cd7", "transformer_2"),
]
BOUNDS_NAME = "Wan 2.1 T2V 14B (attn1, wan_81f)"


def expert_table(snap: Path, num_layers: int = 40) -> dict:
    idx = json.loads((snap / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    layers = {}
    for i in range(num_layers):
        layer = f"blocks.{i}.attn1"
        g = {}
        for kind in ("q", "k"):
            name = f"{layer}.norm_{kind}.weight"
            with safe_open(snap / idx[name], framework="pt") as h:
                g[kind] = h.get_tensor(name).float().view(-1, D)  # [40 heads, d]
        sigma = float(((g["q"] ** 2 * g["k"] ** 2).sum(1) / D).sqrt().max())
        hb = float((math.sqrt(D) * g["q"].abs().amax(1) * g["k"].abs().amax(1)).max())
        sm = 0.88 * sigma * math.sqrt(2 * math.log(SK))
        layers[layer] = {"sigma": sigma, "softmax_max": sm, "hard_bound": hb, "gap": hb - sm,
                         "ok": hb - sm <= GAP_LIMIT}
    return layers


def main(out: Path, bounds_json: Path | None) -> None:
    tables = {}
    for repo, rev, expert in EXPERTS:
        snap = HUB / repo / "snapshots" / rev / expert
        layers = expert_table(snap)
        tables[f"{rev[:8]}/{expert}"] = {"snapshot": str(snap), "layers": layers}
        bad = [k for k, r in layers.items() if not r["ok"]]
        print(f"{rev[:8]}/{expert}: sigma {min(r['sigma'] for r in layers.values()):.3f}.."
              f"{max(r['sigma'] for r in layers.values()):.3f}, max gap "
              f"{max(r['gap'] for r in layers.values()):.1f}; failing the gap rule: {bad}")
    if bounds_json is not None:
        rows = {r["layer"]: r for r in json.loads(bounds_json.read_text())[BOUNDS_NAME]["per_layer"]}
        mine = tables["38ec498c/transformer"]["layers"]
        assert set(rows) == set(mine), "bounds.json has other layers"
        for layer, row in mine.items():
            ref = rows[layer]
            for a, b in (("sigma", "sigma_max_over_heads"), ("softmax_max", "softmax_max_max_over_heads"),
                         ("gap", "call_gap")):
                assert abs(row[a] - ref[b]) < 1e-4, (layer, a, row[a], ref[b])
        print(f"cross-check against {bounds_json}: Wan 2.1's {len(mine)} layers match")
    out.write_text(json.dumps({
        "reference_sk": SK, "d": D, "gap_limit": GAP_LIMIT,
        "generator": "scripts/attn_constmax/table_wan.py",
        "rule": "softmax_max = 0.88 * sigma * sqrt(2 ln Sk); per call: max over heads",
        "tables": tables}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
