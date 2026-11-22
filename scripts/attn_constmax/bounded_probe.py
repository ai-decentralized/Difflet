#!/usr/bin/env python3
"""The key-bounded const-max route (``DIFFLET_EXP_ATTN_CONSTMAX=1``, a call with ``constmax_sigma``
and ``bound_min`` / ``bound_max``: ``ops_impl/attention_const_max_kernel.py``) against the shipped
bounded attention_cte route, at one model's tp per-rank joint-attention shape on one logical core.

Shapes (feat/mxfp8-sage-attention scripts/mx_attention/shapes.json; hi = its ``valid_k``, the bench
prompt's image + valid text keys):
  qwen_image        6 heads x 5,120 (4,096 image + 1,024 text), hi 4,609   (1024x1024 tp4)
  hunyuan_video     6 heads x 67,576 (67,320 + 256),            hi 67,449  (544x960x129 tp4)
  hunyuan_video_15  8 heads x 51,275 (49,290 + 1,985),          hi 50,283  (480x848x121 tp2)

Inputs per layer (E1's probe, scripts/attn_constmax/probe.py): x ~ N(0, 1) RMS-normalized over d,
times the layer's checkpoint QK-RMSNorm gains (image rows ``norm_q``/``norm_k``, text rows
``norm_added_q``/``norm_added_k`` where the layer has them); ``indep`` (q, k from independent x) and
``self`` (the same x: peaked); v ~ N(0, 1); no RoPE. Layers: the largest sigma and the largest gap
of those passing the gap rule (hard_bound - softmax_max <= 80), the table rule of
scripts/attn_constmax/table.py at Sk = the padded key length. Both arms call
``ops_impl.attention.attention`` exactly as the models do (tp_q / tp_k True, ``[B*H, S, d]``, int32
bounds ``[B*H, S_q, 1]``), in a compiled region; only const-max passes ``constmax_sigma``.

Accuracy (the first worker's outputs): rel L2 (float64) against fp32 attention of the same bf16
inputs with scores at keys >= hi set to -inf, overall and worst head, over every ``--ref-stride``-th
query row; non-finite count over every row.
Time: E2's method (scripts/block_sparse/probe.py, removed; at 02ff24b7): per worker process, a cache hit, one warm call,
``--calls`` calls and one sync; ``--runs`` processes per arm, the order alternated (cte, cm),
(cm, cte), ...; median and (max - min) / median per arm.

usage: bounded_probe.py MODEL --out FILE [--runs 3] [--calls N] [--accuracy-only]
env: NEURON_RT_VISIBLE_CORES=<one logical core>; pin CPUs with taskset around the command.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
if "--worker" not in sys.argv:  # the parent spawns and scores on the CPU only
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"
D = 128
GAP_LIMIT = 80.0
MODELS = {  # snapshot dir, heads per rank, image rows, S, hi, calls, ref stride
    "qwen_image": ("models--Qwen--Qwen-Image", 6, 4096, 5120, 4609, 100, 1),
    "hunyuan_video": ("models--hunyuanvideo-community--HunyuanVideo", 6, 67320, 67576, 67449, 10, 16),
    "hunyuan_video_15": ("models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v",
                         8, 49290, 51275, 50283, 10, 16),
}
SCALE = D ** -0.5


def checkpoint_gains(snap_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    snap = next((HUB / snap_dir / "snapshots").iterdir())
    idx = json.loads((snap / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, file in idx.items():
        m = re.match(r"(.*\.attn)\.norm_((?:added_)?[qk])\.weight$", name)
        if m:
            with safe_open(snap / "transformer" / file, framework="pt") as h:
                out.setdefault(m.group(1), {})[m.group(2)] = h.get_tensor(name).float()
    return out


def layer_row(g: dict[str, torch.Tensor], s_k: int) -> dict:
    """table.py's per-call numbers: max over heads and (q norm, k norm) pairs."""
    pairs = [("q", "k")] + ([("q", "added_k"), ("added_q", "k"), ("added_q", "added_k")] if "added_q" in g else [])
    sigma = hb = 0.0
    for a, b in pairs:
        gq, gk = g[a].view(-1, D), g[b].view(-1, D)
        sigma = max(sigma, float(((gq ** 2 * gk ** 2).sum(1) / D).sqrt().max()))
        hb = max(hb, float((math.sqrt(D) * gq.abs().amax(1) * gk.abs().amax(1)).max()))
    sm = 0.88 * sigma * math.sqrt(2 * math.log(s_k))
    return {"sigma": sigma, "softmax_max": sm, "hard_bound": hb, "gap": hb - sm, "ok": hb - sm <= GAP_LIMIT}


def pick_layers(model: str):
    snap, _, _, s, *_ = MODELS[model]
    gains = checkpoint_gains(snap)
    rows = {layer: layer_row(g, s) for layer, g in gains.items()}
    ok = {k: r for k, r in rows.items() if r["ok"]}
    picked = dict.fromkeys([max(ok, key=lambda k: ok[k]["sigma"]), max(ok, key=lambda k: ok[k]["gap"])])
    summary = {"layers": len(rows), "failing_gap_rule": sorted(k for k, r in rows.items() if not r["ok"]),
               "sigma_range": [min(r["sigma"] for r in rows.values()), max(r["sigma"] for r in rows.values())]}
    return [(layer, rows[layer], gains[layer]) for layer in picked], summary


def make_inputs(model: str, g: dict[str, torch.Tensor], family: str, seed: int):
    _, bh, n_img, s, *_ = MODELS[model]
    gen = torch.Generator().manual_seed(seed)

    def unit(x):
        return x / x.square().mean(-1, keepdim=True).sqrt()

    def scaled(x, a):
        row = g[a].expand(s, D).clone()
        if f"added_{a}" in g:
            row[n_img:] = g[f"added_{a}"]
        return (x * row).bfloat16()

    xq = unit(torch.randn(bh, s, D, generator=gen))
    xk = xq if family == "self" else unit(torch.randn(bh, s, D, generator=gen))
    v = torch.randn(bh, s, D, generator=gen).bfloat16()
    return scaled(xq, "q"), scaled(xk, "k"), v


def cases(model: str):
    layers, _ = pick_layers(model)
    return [(layer, row, g, family, li) for li, (layer, row, g) in enumerate(layers) for family in ("indep", "self")]


def require_idle_core() -> dict:
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not cores.isdigit():
        raise SystemExit(f"NEURON_RT_VISIBLE_CORES must name one logical core, got {cores!r}")
    device = int(cores) // 4
    done = subprocess.run(["fuser", f"/dev/neuron{device}"], capture_output=True)
    said = (done.stdout + done.stderr).decode(errors="replace").strip()
    if done.returncode != 1 or said:
        raise SystemExit(f"/dev/neuron{device} is not idle (fuser exit {done.returncode} {said})")
    return {"device": device, "core": int(cores), "at": datetime.datetime.now().isoformat(timespec="seconds")}


def worker(args) -> None:
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-exp/constmax-models/kernel/cache")
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1")
    if args.arm == "cm":
        os.environ["DIFFLET_EXP_ATTN_CONSTMAX"] = "1"
    else:
        os.environ.pop("DIFFLET_EXP_ATTN_CONSTMAX", None)
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl import attention as attn

    assert (attn._CONST_MAX_BOUNDED_OP is not None) == (args.arm == "cm")
    _, bh, _, s, hi, *_ = MODELS[args.model]

    class Region(torch.nn.Module):
        def forward(self, q, k, v, lo, hi_t, sigma=None):
            kw = {} if sigma is None else {"constmax_sigma": sigma}
            return attn.attention(q, k, v, scale=SCALE, causal=False, bound_min=lo, bound_max=hi_t,
                                  tp_q=True, tp_k=True, tp_out=False, **kw)

    region = compile_region(Region(), backend="neuron")
    fuser = require_idle_core()
    hi_t = torch.full((bh, s, 1), hi, dtype=torch.int32)
    bounds = [torch.zeros_like(hi_t).to("neuron"), hi_t.to("neuron")]
    rows, timed_args = [], None
    for layer, row, g, family, li in cases(args.model) if args.save_dir else cases(args.model)[:1]:
        dev = [t.to("neuron") for t in make_inputs(args.model, g, family, li)] + bounds
        if args.arm == "cm":
            dev.append(torch.full((128, 1), row["sigma"]).to("neuron"))
        t0 = time.perf_counter()
        out = region(*dev).cpu()
        first = time.perf_counter() - t0
        if args.save_dir:
            torch.save(out, args.save_dir / f"{args.arm}_{li}_{family}.pt")
        rows.append({"layer": layer, "family": family, "first_call_s": first,
                     "nonfinite": int((~out.isfinite()).sum())})
        timed_args = timed_args or dev
    result = {"rows": rows, "fuser": fuser}
    if not args.accuracy_only:
        region(*timed_args)[0, 0, 0].item()  # warm
        t0 = time.perf_counter()
        for _ in range(args.calls):
            out = region(*timed_args)
        out[0, 0, 0].item()  # one sync
        result["ms"] = 1000 * (time.perf_counter() - t0) / args.calls
    args.worker.write_text(json.dumps(result))


def reference_compare(model: str, out: torch.Tensor, q, k, v, stride: int) -> dict:
    """rel L2 against masked fp32 attention over query rows ::stride, and the max valid score."""
    _, bh, _, s, hi, *_ = MODELS[model]
    total, heads, score_max = torch.zeros(2, dtype=torch.float64), [], -math.inf
    kv = k[:, :hi].float(), v[:, :hi].float()
    for h in range(bh):
        acc = torch.zeros(2, dtype=torch.float64)
        for r0 in range(0, s, 4096 * stride):
            qs = q[h, r0:r0 + 4096 * stride:stride].float()
            sc = qs @ kv[0][h].T * SCALE
            score_max = max(score_max, float(sc.max()))
            ref = (torch.softmax(sc, -1) @ kv[1][h]).double()
            o = out[h, r0:r0 + 4096 * stride:stride].double()
            acc += torch.stack([(o - ref).square().sum(), ref.square().sum()])
        heads.append((acc[0] / acc[1]).sqrt().item())
        total += acc
    return {"rel_l2": (total[0] / total[1]).sqrt().item(), "rel_l2_max_head": max(heads),
            "nonfinite": int((~out.isfinite()).sum()), "score_max": score_max}


def checkout() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"path": str(REPO), "head": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "difflet/backends/neuron/ops_impl/"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--calls", type=int)
    p.add_argument("--ref-stride", type=int)
    p.add_argument("--accuracy-only", action="store_true", help="one worker per arm, no timing")
    p.add_argument("--arm", choices=["cte", "cm"], help=argparse.SUPPRESS)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--save-dir", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    *_, calls, stride = MODELS[args.model]
    args.calls = args.calls or calls
    stride = args.ref_stride or stride
    if args.worker:
        worker(args)
        return 0

    stem = args.out.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_dir = Path(f"{stem}.outputs")
    save_dir.mkdir(exist_ok=True)
    order = [("cte", "cm") if i % 2 == 0 else ("cm", "cte") for i in range(1 if args.accuracy_only else args.runs)]
    runs = {"cte": [], "cm": []}
    env = {k: v for k, v in os.environ.items() if k != "TORCH_DEVICE_BACKEND_AUTOLOAD"}
    for i, pair in enumerate(order):
        for arm in pair:
            run = Path(f"{stem}.{arm}.run{i}.json")
            extra = ["--save-dir", str(save_dir)] if i == 0 else []
            with Path(f"{stem}.{arm}.run{i}.log").open("w") as f:
                subprocess.run([sys.executable, __file__, args.model, "--out", str(args.out), "--arm", arm,
                                "--calls", str(args.calls), "--worker", str(run), *extra,
                                *(["--accuracy-only"] if args.accuracy_only else [])],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
            runs[arm].append(json.loads(run.read_text()))

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    layers, summary = pick_layers(args.model)
    rows = []
    for layer, row, g, family, li in cases(args.model):
        q, k, v = make_inputs(args.model, g, family, li)
        r = {"layer": layer, "family": family, **{k2: row[k2] for k2 in ("sigma", "softmax_max", "hard_bound", "gap")}}
        for arm in ("cm", "cte"):
            r[arm] = reference_compare(args.model, torch.load(save_dir / f"{arm}_{li}_{family}.pt").float(), q, k, v, stride)
        rows.append(r)
        print(json.dumps(r), flush=True)
    _, bh, n_img, s, hi, *_ = MODELS[args.model]
    result = {"model": args.model, "shape": [bh, s, D], "image_rows": n_img, "hi": hi, "scale": SCALE,
              "ref_stride": stride, "table": summary, "rows": rows, "calls": args.calls,
              "cpus": sorted(os.sched_getaffinity(0)), "tree": checkout(),
              "fuser": {a: [r["fuser"] for r in rs] for a, rs in runs.items()},
              "first_call_s": {a: [x["first_call_s"] for r in rs for x in r["rows"]] for a, rs in runs.items()}}
    if not args.accuracy_only:
        for arm, rs in runs.items():
            ms = [r["ms"] for r in rs]
            med = statistics.median(ms)
            result[arm] = {"runs_ms": ms, "median_ms": med, "spread": (max(ms) - min(ms)) / med}
        result["cm_over_cte"] = result["cm"]["median_ms"] / result["cte"]["median_ms"]
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("model", "cm_over_cte") if k in result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
