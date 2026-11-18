"""Experiment E1 (docs/plans/2026-11-18-attn-constmax-sparse-experiments.md): the per-layer
softmax_max table for nkilib's ``attention_const_max``, from the checkpoint's QK-RMSNorm weights.

This is ``/mnt/nvme/difflet-mxattn/probe/constmax/bounds.py`` (section 12 of
``docs/verification/2026-11-13-dit-block-profile.md`` on ``feat/mxfp8-sage-attention``) with its
formulas unchanged -- the kernel docstring's rule, per layer, head h and (q norm, k norm) pair:
  sigma = sqrt(sum(gamma_q^2 * gamma_k^2) / d)
  softmax_max = 0.88 * sigma * sqrt(2 ln Sk)
  hard_bound = sqrt(d) * max|gamma_q| * max|gamma_k|
-- and one change: bounds.py reports the worst head of the worst pair; one kernel call covers every
head and (for FLUX's joint blocks) every pair, so the table keeps the call's values, the max over
heads and pairs: ``sigma``, ``softmax_max`` (at the reference Sk), ``hard_bound`` and ``gap``
(hard_bound - softmax_max), and ``ok`` = gap <= 80 (the docstring's limit; a layer that fails it
falls back to attention_cte). The backend recomputes softmax_max from ``sigma`` at each call's own
Sk, so the reference Sk only sets ``gap``.

Cross-check: with BOUNDS_JSON given (bounds.json of the run above), every layer whose call has
one pair must have bounds.json's per-call numbers (sigma_max_over_heads, softmax_max_max_over_heads,
call_gap) to 1e-4.

usage: table.py {flux,ltx_2} OUT_JSON [BOUNDS_JSON]
"""
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HUB = Path.home() / ".cache/huggingface/hub"
D = 128
GAP_LIMIT = 80.0
MODELS = {  # key: (bounds.json name, snapshot dir, attention module regex, reference Sk)
    "flux": ("FLUX.1-dev (flux_1024)", "models--black-forest-labs--FLUX.1-dev",
             r"(?:single_)?transformer_blocks\.\d+\.attn\.", 4608),
    "ltx_2": ("LTX-2 (video attn1, ltx2_video)", "models--Lightricks--LTX-2",
              r"transformer_blocks\.\d+\.attn1\.", 6144),
}


def main(key: str, out: Path, bounds_json: Path | None) -> None:
    name, repo, pat, sk = MODELS[key]
    snap = next((HUB / repo / "snapshots").iterdir())
    idx = json.loads((snap / "transformer/diffusion_pytorch_model.safetensors.index.json")
                     .read_text())["weight_map"]
    names = [n for n in idx if re.match(pat, n) and re.search(r"\.norm_(added_)?[qk]\.weight$", n)]
    gammas: dict[str, dict[str, torch.Tensor]] = {}
    for n in names:
        with safe_open(snap / "transformer" / idx[n], framework="pt") as h:
            layer, kind = n.rsplit(".norm_", 1)
            gammas.setdefault(layer, {})[kind.split(".")[0]] = h.get_tensor(n).float()
    layers = {}
    for layer, g in sorted(gammas.items(), key=lambda kv: [int(t) if t.isdigit() else t
                                                           for t in kv[0].split(".")]):
        pairs = [("q", "k")] + ([("q", "added_k"), ("added_q", "k"), ("added_q", "added_k")]
                                if "added_q" in g else [])
        sigma = hb = 0.0
        for a, b in pairs:
            gq, gk = g[a].view(-1, D), g[b].view(-1, D)  # [heads in the norm, d]
            sigma = max(sigma, float(((gq ** 2 * gk ** 2).sum(1) / D).sqrt().max()))
            hb = max(hb, float((math.sqrt(D) * gq.abs().amax(1) * gk.abs().amax(1)).max()))
        sm = 0.88 * sigma * math.sqrt(2 * math.log(sk))
        layers[layer] = {"pairs": len(pairs), "sigma": sigma, "softmax_max": sm, "hard_bound": hb,
                         "gap": hb - sm, "ok": hb - sm <= GAP_LIMIT}
    if bounds_json is not None:
        rows = {r["layer"]: r for r in json.loads(bounds_json.read_text())[name]["per_layer"]}
        assert set(rows) == set(layers), "bounds.json has other layers"
        for layer, row in layers.items():
            if row["pairs"] == 1:
                ref = rows[layer]
                for mine, theirs in (("sigma", "sigma_max_over_heads"),
                                     ("softmax_max", "softmax_max_max_over_heads")):
                    assert abs(row[mine] - ref[theirs]) < 1e-4, (layer, mine, row[mine], ref[theirs])
                assert abs(row["gap"] - ref["call_gap"]) < 1e-4, (layer, row["gap"], ref["call_gap"])
        print(f"cross-check against {bounds_json}: {sum(r['pairs'] == 1 for r in layers.values())} "
              f"one-pair layers match")
    table = {"model": key, "snapshot": str(snap), "reference_sk": sk, "d": D,
             "gap_limit": GAP_LIMIT, "generator": "scripts/attn_constmax/table.py",
             "rule": "softmax_max = 0.88 * sigma * sqrt(2 ln Sk); per call: max over heads and "
                     "(q norm, k norm) pairs", "layers": layers}
    out.write_text(json.dumps(table, indent=1) + "\n")
    bad = [k for k, r in layers.items() if not r["ok"]]
    print(f"{key}: {len(layers)} layers, sigma {min(r['sigma'] for r in layers.values()):.3f}.."
          f"{max(r['sigma'] for r in layers.values()):.3f}, max gap "
          f"{max(r['gap'] for r in layers.values()):.1f}; failing the gap rule: {bad}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
