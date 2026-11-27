"""The per-layer softmax_max tables for nkilib's ``attention_const_max``
(``--attention const-max``), from each checkpoint's QK-RMSNorm weights. Experiment E1;
evidence at 713d53d9.

This is ``/mnt/nvme/difflet-mxattn/probe/constmax/bounds.py`` (section 12 of
``docs/verification/2026-11-13-dit-block-profile.md`` on ``feat/mxfp8-sage-attention``) with its
formulas unchanged -- the kernel docstring's rule, per layer, head h and (q norm, k norm) pair:
  sigma = sqrt(sum(gamma_q^2 * gamma_k^2) / d)
  softmax_max = 0.88 * sigma * sqrt(2 ln Sk)
  hard_bound = sqrt(d) * max|gamma_q| * max|gamma_k|
-- and one change: bounds.py reports the worst head of the worst pair; one kernel call covers every
head and (for the joint blocks) every pair, so the table keeps the call's values, the max over
heads and pairs: ``sigma``, ``softmax_max`` (at the reference Sk), ``hard_bound`` and ``gap``
(hard_bound - softmax_max), and ``ok`` = gap <= 80 (the docstring's limit; a layer that fails it
falls back to attention_cte). The backend recomputes softmax_max from ``sigma`` at each call's own
Sk, so the reference Sk only sets ``gap``.

Models and their reference Sk:
- flux, ltx_2 (video attn1), qwen_image: the shapes of the campaign (qwen_image's joint keys at
  1024x1024 are 4,096 image + the 1,024 padded text tokens; bounded calls).
- hunyuan_video: FLUX's joint layout (20 dual-stream blocks with four (q norm, k norm) pairs, 40
  single-stream with one); Sk at 544x960x129 = 33 x 34 x 60 image + 256 right-padded text tokens.
  The token refiner's self-attention has no QK norm and is not in the table.
- hunyuan_video_15: ``transformer_blocks.{i}.attn``, four pairs per key-bounded call; Sk at
  480x848x121 = 31 x 30 x 53 image + 1,985 text (1,000 Qwen2.5-VL + 256 ByT5 + 729 image-embedding).
- minimax_h3: the 50 single-stream blocks' self-attention (``transformer_blocks.{i}.attn``, one
  pair; the text-only token refiner's queries are under the kernel's 128 and stay on
  attention_cte); Sk at 256x448x124 = the packed t2va sequence padded to 5,120 rows (4,622 live
  keys, key-prefix bounds).
- wan: each expert's self-attention (``blocks.{i}.attn1``; cross-attention stays on attention_cte)
  at Sk = 21 x 30 x 52 = 32,760 (480x832x81), one table per expert keyed "<revision[:8]>/<expert
  dir>" (Wan 2.1 and both Wan 2.2-A14B experts, snapshots pinned); WanTransformer3DModel's
  ``add_constmax_sigma_`` picks the one whose key matches the directory it loads.

Cross-check: with BOUNDS_JSON given (bounds.json of the run above: flux, ltx_2, qwen_image, wan),
every layer whose call has one pair must have bounds.json's per-call numbers (sigma_max_over_heads,
softmax_max_max_over_heads, call_gap) to 1e-4 (for wan: Wan 2.1's layers).

usage: table.py {flux,ltx_2,qwen_image,hunyuan_video,hunyuan_video_15,minimax_h3,wan} OUT_JSON
       [BOUNDS_JSON]
(the hub is $HF_HUB_CACHE, else ~/.cache/huggingface/hub)
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HUB = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache/huggingface/hub")
D = 128
GAP_LIMIT = 80.0
GENERATOR = "scripts/attn_constmax/table.py"
MODELS = {  # key: (bounds.json name, snapshot dir, attention module regex, reference Sk)
    "flux": ("FLUX.1-dev (flux_1024)", "models--black-forest-labs--FLUX.1-dev",
             r"(?:single_)?transformer_blocks\.\d+\.attn\.", 4608),
    "ltx_2": ("LTX-2 (video attn1, ltx2_video)", "models--Lightricks--LTX-2",
              r"transformer_blocks\.\d+\.attn1\.", 6144),
    "qwen_image": ("Qwen-Image (joint, 1024x1024)", "models--Qwen--Qwen-Image",
                   r"transformer_blocks\.\d+\.attn\.", 5120),
    "hunyuan_video": (
        "HunyuanVideo (hunyuan_video_L)",
        "models--hunyuanvideo-community--HunyuanVideo",
        r"(?:single_)?transformer_blocks\.\d+\.attn\.",
        33 * 34 * 60 + 256,
    ),
    "hunyuan_video_15": ("HunyuanVideo-1.5 (joint, 480x848x121)",
                         "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v",
                         r"transformer_blocks\.\d+\.attn\.", 31 * 30 * 53 + 1000 + 256 + 729),
    "minimax_h3": ("MiniMax-H3 (t2va, 256x448x124)", "models--MiniMaxAI--MiniMax-H3",
                   r"transformer_blocks\.\d+\.attn\.", 5120),
}
WAN_SK = 32760
WAN_21 = (
    "models--Wan-AI--Wan2.1-T2V-14B-Diffusers",
    "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
)
WAN_22 = (
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers",
    "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
)
WAN_EXPERTS = [  # (hub repo, revision, expert dir)
    (*WAN_21, "transformer"),
    (*WAN_22, "transformer"),
    (*WAN_22, "transformer_2"),
]
WAN_BOUNDS_NAME = "Wan 2.1 T2V 14B (attn1, wan_81f)"


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
             "gap_limit": GAP_LIMIT, "generator": GENERATOR,
             "rule": "softmax_max = 0.88 * sigma * sqrt(2 ln Sk); per call: max over heads and "
                     "(q norm, k norm) pairs", "layers": layers}
    out.write_text(json.dumps(table, indent=1) + "\n")
    bad = [k for k, r in layers.items() if not r["ok"]]
    print(f"{key}: {len(layers)} layers, sigma {min(r['sigma'] for r in layers.values()):.3f}.."
          f"{max(r['sigma'] for r in layers.values()):.3f}, max gap "
          f"{max(r['gap'] for r in layers.values()):.1f}; failing the gap rule: {bad}")


def wan_expert_table(snap: Path, num_layers: int = 40) -> dict:
    index = snap / "diffusion_pytorch_model.safetensors.index.json"
    idx = json.loads(index.read_text())["weight_map"]
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
        sm = 0.88 * sigma * math.sqrt(2 * math.log(WAN_SK))
        layers[layer] = {"sigma": sigma, "softmax_max": sm, "hard_bound": hb, "gap": hb - sm,
                         "ok": hb - sm <= GAP_LIMIT}
    return layers


def main_wan(out: Path, bounds_json: Path | None) -> None:
    tables = {}
    for repo, rev, expert in WAN_EXPERTS:
        snap = HUB / repo / "snapshots" / rev / expert
        layers = wan_expert_table(snap)
        tables[f"{rev[:8]}/{expert}"] = {"snapshot": str(snap), "layers": layers}
        bad = [k for k, r in layers.items() if not r["ok"]]
        print(f"{rev[:8]}/{expert}: sigma {min(r['sigma'] for r in layers.values()):.3f}.."
              f"{max(r['sigma'] for r in layers.values()):.3f}, max gap "
              f"{max(r['gap'] for r in layers.values()):.1f}; failing the gap rule: {bad}")
    if bounds_json is not None:
        per_layer = json.loads(bounds_json.read_text())[WAN_BOUNDS_NAME]["per_layer"]
        rows = {r["layer"]: r for r in per_layer}
        mine = tables["38ec498c/transformer"]["layers"]
        assert set(rows) == set(mine), "bounds.json has other layers"
        for layer, row in mine.items():
            ref = rows[layer]
            for a, b in (
                ("sigma", "sigma_max_over_heads"),
                ("softmax_max", "softmax_max_max_over_heads"),
                ("gap", "call_gap"),
            ):
                assert abs(row[a] - ref[b]) < 1e-4, (layer, a, row[a], ref[b])
        print(f"cross-check against {bounds_json}: Wan 2.1's {len(mine)} layers match")
    out.write_text(json.dumps({
        "reference_sk": WAN_SK, "d": D, "gap_limit": GAP_LIMIT, "generator": GENERATOR,
        "rule": "softmax_max = 0.88 * sigma * sqrt(2 ln Sk); per call: max over heads",
        "tables": tables}, indent=1) + "\n")


if __name__ == "__main__":
    model, out_json = sys.argv[1], Path(sys.argv[2])
    bounds = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if model == "wan":
        main_wan(out_json, bounds)
    else:
        main(model, out_json, bounds)
