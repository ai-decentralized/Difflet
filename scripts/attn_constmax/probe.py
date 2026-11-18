"""Experiment E1 gate: the const-max route of ``ops_impl.attention`` (``DIFFLET_EXP_ATTN_CONSTMAX=1``)
against fp32 and against the shipped attention_cte route, at the model's tp4 per-rank shape (one
logical core): FLUX 6 heads x 4,608 x 128, LTX-2 video attn1 8 heads x 6,144 x 128, with q/k
scaled like the model's.

Inputs per layer: x ~ N(0, 1), RMS-normalized over d (as QK-RMSNorm leaves it), times the layer's
checkpoint gains (FLUX dual blocks: the first 512 rows, the text, take ``norm_added_q/k``; LTX-2's
norm spans all 32 heads: the 8 heads of the rank holding the layer's largest-sigma head);
v ~ N(0, 1).
Two families: ``indep`` (q and k from independent x) and ``self`` (q_i and k_i from the same x_i: a
diagonal score near sum(g_q g_k)/sqrt(d), the peaked case). RoPE is not applied. All bf16 on the
device; the reference is fp32 attention of the same bf16 inputs on the CPU.

usage: probe.py {flux,ltx_2} OUT_JSON   (env: DIFFLET_EXP_ATTN_CONSTMAX=1, NEURON_RT_VISIBLE_CORES=<one core>)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"
MODELS = {  # model: (snapshot dir, layers, heads per rank, S, text rows)
    "flux": ("models--black-forest-labs--FLUX.1-dev",
             ["single_transformer_blocks.36.attn",  # the largest gap
              "single_transformer_blocks.25.attn",  # the largest sigma
              "transformer_blocks.18.attn",         # joint, high sigma
              "transformer_blocks.0.attn"],         # joint, low sigma
             6, 4608, 512),
    "ltx_2": ("models--Lightricks--LTX-2",
              ["transformer_blocks.26.attn1",   # the largest gap of the routed layers
               "transformer_blocks.46.attn1",   # the largest sigma of the routed layers
               "transformer_blocks.45.attn1"],  # fails the gap rule, stays on cte: reference only
              8, 6144, 0),
}
MODEL = sys.argv[1] if len(sys.argv) > 2 else "flux"
SNAP_DIR, LAYERS, BH, S, TXT = MODELS[MODEL]
D = 128
CALLS = 20


def gains(layer: str) -> dict[str, torch.Tensor]:
    snap = next((HUB / SNAP_DIR / "snapshots").iterdir())
    idx = json.loads((snap / "transformer/diffusion_pytorch_model.safetensors.index.json")
                     .read_text())["weight_map"]
    out = {}
    for kind in ("q", "k", "added_q", "added_k"):
        name = f"{layer}.norm_{kind}.weight"
        if name in idx:
            with safe_open(snap / "transformer" / idx[name], framework="pt") as h:
                out[kind] = h.get_tensor(name).float()
    return out


def inputs(layer: str, family: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    gm = gains(layer)

    def unit(x):
        return x / x.square().mean(-1, keepdim=True).sqrt()

    def scaled(x, a):
        if gm[a].numel() > D:  # an across-heads norm: this rank's heads, [BH, 1, D]
            sig = ((gm["q"].view(-1, D) ** 2 * gm["k"].view(-1, D) ** 2).sum(1) / D).sqrt()
            r = int(sig.argmax()) // BH
            return (x * gm[a].view(-1, D)[r * BH:(r + 1) * BH, None]).bfloat16()
        row = gm[a].expand(S, D).clone()
        if f"added_{a}" in gm:
            row[:TXT] = gm[f"added_{a}"]
        return (x * row).bfloat16()

    xq = unit(torch.randn(BH, S, D, generator=g))
    xk = xq if family == "self" else unit(torch.randn(BH, S, D, generator=g))
    v = torch.randn(BH, S, D, generator=g).bfloat16()
    return scaled(xq, "q"), scaled(xk, "k"), v


def compare(out: torch.Tensor, q, k, v) -> dict:
    total = torch.zeros(2, dtype=torch.float64)
    heads = []
    for h in range(BH):
        ref = torch.softmax(q[h].float() @ k[h].float().T * D ** -0.5, -1) @ v[h].float()
        o = out[h].double()
        s = torch.stack([(o - ref.double()).square().sum(), ref.double().square().sum()])
        heads.append((s[0] / s[1]).sqrt().item())
        total += s
    return {"rel_l2": (total[0] / total[1]).sqrt().item(), "rel_l2_max_head": max(heads),
            "nonfinite": int((~torch.isfinite(out)).sum())}


def main(out_path: Path) -> None:
    assert os.environ.get("DIFFLET_EXP_ATTN_CONSTMAX") == "1", "set DIFFLET_EXP_ATTN_CONSTMAX=1"
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl import attention as attn

    assert attn._CONST_MAX_OP is not None

    class ConstMax(torch.nn.Module):
        def forward(self, q, k, v, sigma):
            return attn.attention(q, k, v, scale=D ** -0.5, tp_q=True, tp_k=True, constmax_sigma=sigma)

    class Cte(torch.nn.Module):
        def forward(self, q, k, v):
            return attn.attention(q, k, v, scale=D ** -0.5, tp_q=True, tp_k=True)

    table = json.loads((REPO / f"difflet/models/{MODEL}/attn_constmax.json").read_text())["layers"]
    regions = {"constmax": compile_region(ConstMax(), backend="neuron"),
               "cte": compile_region(Cte(), backend="neuron")}
    rows = []
    for li, layer in enumerate(LAYERS):
        sigma = torch.full((128, 1), table[layer]["sigma"])
        for family in ("indep", "self"):
            q, k, v = inputs(layer, family, seed=li)
            dev = [t.to("neuron") for t in (q, k, v)]
            row = {"layer": layer, "family": family, "sigma": table[layer]["sigma"],
                   "softmax_max": 0.88 * table[layer]["sigma"] * math.sqrt(2 * math.log(S)),
                   "score_max": float(max((q[h].float() @ k[h].float().T).max() for h in range(BH)))
                   * D ** -0.5}
            for arm, region in regions.items():
                args = dev + ([sigma.to("neuron")] if arm == "constmax" else [])
                t0 = time.perf_counter()
                out = region(*args).cpu()
                first = time.perf_counter() - t0
                region(*args)[0, 0, 0].item()
                t0 = time.perf_counter()
                for _ in range(CALLS):
                    o = region(*args)
                o[0, 0, 0].item()
                row[arm] = {**compare(out, q, k, v), "ms": 1000 * (time.perf_counter() - t0) / CALLS,
                            "first_call_s": first}
            rows.append(row)
            print(json.dumps(row), flush=True)
    out_path.write_text(json.dumps({"shape": [BH, S, D], "calls": CALLS, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[-1]))
