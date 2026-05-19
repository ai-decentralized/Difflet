#!/usr/bin/env python3
"""M5.D2 v0 — Latent-workload data-movement profiler.

Captures a Neuron Profiler (.ntff) trace for an already-compiled diffusion
NEFF, ingests the ``view --output-format summary-json`` metrics, and emits a
data-movement bottleneck readout: roofline position, per-engine utilisation,
DMA traffic breakdown (static vs dynamic, hw vs sw), HBM bandwidth
utilisation, and a single bottleneck verdict.

This is the v0 of the M5.D2 "Latent Workload Profiler" backlog item
(cclogs/m5-latent/44 §5.2). It deliberately profiles a single compiled NEFF
(the denoise-loop-dominant component), not the full host pipeline timeline.

Usage:
    python scripts/profile_dit_backbone.py --target hunyuan-attn
    python scripts/profile_dit_backbone.py --neff /path/graph.neff --label foo
    python scripts/profile_dit_backbone.py --target all --out /tmp/prof.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NEURON_PROFILE = "/opt/aws/neuron/bin/neuron-profile"

# Named presets: real compiled NEFFs already on this host.
TARGETS = {
    "hunyuan-attn": (
        "/tmp/nova_hunyuan_attention_neff_prod_long/compiler/graph.neff",
        "HunyuanVideo attention (production long sequence)",
    ),
    "hunyuan-vae-body": (
        "/tmp/nova_hunyuan_vae_body_act_split_neff/compiler/graph.neff",
        "HunyuanVideo VAE body (conv-heavy)",
    ),
    "wan-dit": (
        "/tmp/nxd_model_wan_bb/WanTransformer3DModel/_tp0_bk0/graph.neff",
        "Wan 2.2 DiT backbone (video transformer, tp0 shard)",
    ),
    "flux-dit": (
        "/tmp/nxd_model_flux/transformer/NeuronFluxTransformer2DModel/_tp0_bk0/graph.neff",
        "Flux.1-dev DiT transformer (image, 1024x1024, tp0 shard)",
    ),
    "flux-vae": (
        "/tmp/nxd_model_flux/decoder/Decoder/_tp0_bk0/graph.neff",
        "Flux.1-dev VAE decoder (image, 1024x1024, tp0 shard)",
    ),
}


def _run(cmd: list[str], timeout: int) -> str:
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", "replace")
    if p.returncode != 0 and "numerical error (NaN)" not in out:
        sys.stderr.write(p.stderr.decode("utf-8", "replace")[-2000:])
        p.check_returncode()
    return out


def capture_and_view(neff: str, workdir: str, world_size: int) -> dict:
    """capture -> .ntff -> view summary-json. Returns the metrics dict."""
    os.makedirs(workdir, exist_ok=True)
    prefix = os.path.join(workdir, "profile")
    cap = [
        NEURON_PROFILE, "capture", "-n", neff, "-s", prefix + ".ntff",
        "--num-exec", "2", "--profile-nth-exec", "2", "--ignore-exec-errors",
    ]
    if world_size > 1:
        cap += ["--collectives-workers-per-node", str(world_size),
                "--collectives-profile-id", "0"]
    _run(cap, timeout=900)

    # collectives runs name the file *_rank_0_exec_2.ntff; single-core is *_exec_2.ntff
    cands = [f"{prefix}_rank_0_exec_2.ntff", f"{prefix}_exec_2.ntff"]
    ntff = next((c for c in cands if os.path.exists(c)), None)
    if ntff is None:
        raise FileNotFoundError(f"no .ntff produced; looked for {cands}")

    out = _run([
        NEURON_PROFILE, "view", "-n", neff, "-s", ntff,
        "--output-format", "summary-json", "--ignore-nc-buf-usage",
    ], timeout=600)
    return list(json.loads(out).values())[0]


def analyze(m: dict) -> dict:
    """Derive a data-movement bottleneck verdict from raw summary metrics."""
    g = lambda k: float(m.get(k, 0.0) or 0.0)

    total_t = g("total_time")
    hbm_read = g("hbm_read_bytes")
    hbm_write = g("hbm_write_bytes")
    hbm_total = hbm_read + hbm_write

    engines = {
        "TensorE (matmul)": g("tensor_engine_active_time_percent"),
        "VectorE": g("vector_engine_active_time_percent"),
        "ScalarE": g("scalar_engine_active_time_percent"),
        "GpSimdE": g("gpsimd_engine_active_time_percent"),
        "SyncE": g("sync_engine_active_time_percent"),
        "CC-cores": g("cc_cores_instruction_active_time_percent"),
    }
    top_engine, top_engine_util = max(engines.items(), key=lambda kv: kv[1])

    arith_intensity = g("mm_arithmetic_intensity")
    balance = g("peak_flops_bandwidth_ratio")  # FLOP/byte machine balance
    mfu = g("mfu_estimated_percent")
    mbu = g("mbu_estimated_percent")

    # Roofline region: above the ridge point => compute side, below => bw side.
    roofline_side = "compute-side" if arith_intensity >= balance else "memory-bandwidth-side"

    dma = {
        "hw_dynamic_dma_active_pct": g("hardware_dynamic_dma_active_time_percent"),
        "sw_dynamic_dma_active_pct": g("software_dynamic_dma_active_time_percent"),
        "static_dma_active_pct": g("static_dma_active_time_percent"),
        "hw_dynamic_dma_bytes": g("hardware_dynamic_dma_size"),
        "sw_dynamic_dma_bytes": g("software_dynamic_dma_size"),
        "static_dma_bytes": g("static_dma_size"),
    }
    dma_active_total_pct = (dma["hw_dynamic_dma_active_pct"]
                            + dma["sw_dynamic_dma_active_pct"]
                            + dma["static_dma_active_pct"])

    # Verdict logic. An engine is "the bottleneck" if it is ~saturated while
    # the others (and HBM bandwidth) are not.
    if top_engine_util >= 0.90 and mbu < 0.60 and engines["TensorE (matmul)"] < 0.80:
        verdict = f"{top_engine}-bound"
        rationale = (f"{top_engine} active {top_engine_util:.1%} of runtime "
                     f"while matmul only {engines['TensorE (matmul)']:.1%} and "
                     f"HBM BW util (MBU) {mbu:.1%}; not DMA/HBM limited.")
    elif mbu >= 0.60:
        verdict = "memory-bandwidth-bound"
        rationale = f"MBU {mbu:.1%}: HBM bandwidth is the limiter."
    elif dma_active_total_pct >= 0.90 and top_engine_util < 0.60:
        verdict = "DMA / data-movement-bound"
        rationale = (f"DMA active {dma_active_total_pct:.1%} of runtime with "
                     f"no engine above {top_engine_util:.1%}.")
    elif engines["TensorE (matmul)"] >= 0.85:
        verdict = "compute-bound (matmul / TensorE)"
        rationale = f"TensorE active {engines['TensorE (matmul)']:.1%}."
    else:
        verdict = "latency / under-utilised (no single saturated resource)"
        rationale = (f"top engine {top_engine} {top_engine_util:.1%}, "
                     f"MFU {mfu:.1%}, MBU {mbu:.1%}.")

    return {
        "total_time_s": total_t,
        "engine_util_pct": engines,
        "top_engine": top_engine,
        "top_engine_util_pct": top_engine_util,
        "roofline": {
            "arithmetic_intensity_flop_per_byte": arith_intensity,
            "machine_balance_flop_per_byte": balance,
            "side": roofline_side,
            "mfu_pct": mfu,
            "mbu_pct": mbu,
            "hfu_pct": g("hfu_estimated_percent"),
        },
        "hbm": {
            "read_bytes": hbm_read,
            "write_bytes": hbm_write,
            "total_bytes": hbm_total,
            "read_GiB": hbm_read / 2**30,
            "write_GiB": hbm_write / 2**30,
            "effective_bandwidth_GBps": (hbm_total / total_t / 1e9) if total_t else 0.0,
        },
        "sbuf": {
            "read_bytes": g("sbuf_read_bytes"),
            "write_bytes": g("sbuf_write_bytes"),
        },
        "dma": dma,
        "dma_active_total_pct": dma_active_total_pct,
        "verdict": verdict,
        "rationale": rationale,
    }


def _fmt(a: dict, label: str) -> str:
    L = [f"\n{'='*72}", f" {label}", f"{'='*72}"]
    L.append(f" total exec time      : {a['total_time_s']*1e3:.3f} ms")
    L.append(" engine utilisation (% of runtime):")
    for name, v in sorted(a["engine_util_pct"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(v * 40))
        L.append(f"   {name:<18} {v:6.1%} |{bar}")
    r = a["roofline"]
    L.append(" roofline:")
    L.append(f"   arithmetic intensity : {r['arithmetic_intensity_flop_per_byte']:.1f} FLOP/byte")
    L.append(f"   machine balance      : {r['machine_balance_flop_per_byte']:.1f} FLOP/byte  ({r['side']})")
    L.append(f"   MFU={r['mfu_pct']:.1%}  MBU={r['mbu_pct']:.1%}  HFU={r['hfu_pct']:.1%}")
    h = a["hbm"]
    L.append(" HBM data movement:")
    L.append(f"   read {h['read_GiB']:.3f} GiB / write {h['write_GiB']:.3f} GiB")
    L.append(f"   effective HBM BW     : {h['effective_bandwidth_GBps']:.1f} GB/s")
    d = a["dma"]
    L.append(" DMA active time breakdown (% of runtime):")
    L.append(f"   hw-dynamic {d['hw_dynamic_dma_active_pct']:.1%}  "
             f"sw-dynamic {d['sw_dynamic_dma_active_pct']:.1%}  "
             f"static {d['static_dma_active_pct']:.1%}")
    L.append(f"\n  >>> VERDICT: {a['verdict']}")
    L.append(f"      {a['rationale']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=list(TARGETS) + ["all"],
                    help="named preset NEFF")
    ap.add_argument("--neff", help="explicit graph.neff path")
    ap.add_argument("--label", default="custom-neff")
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--workdir", default="/tmp/nova_profile_dit")
    ap.add_argument("--out", default="/tmp/nova_profile_dit/metrics.json")
    args = ap.parse_args()

    jobs: list[tuple[str, str, str]] = []  # (neff, label, subdir)
    if args.neff:
        jobs.append((args.neff, args.label, args.label))
    elif args.target == "all":
        jobs = [(p, lbl, k) for k, (p, lbl) in TARGETS.items()]
    elif args.target:
        p, lbl = TARGETS[args.target]
        jobs.append((p, lbl, args.target))
    else:
        ap.error("pass --target or --neff")

    results = {}
    for neff, label, sub in jobs:
        if not Path(neff).exists():
            print(f"[skip] {label}: NEFF missing {neff}")
            results[label] = {"error": f"NEFF missing: {neff}"}
            continue
        print(f"\n[profile] {label}\n  neff={neff}")
        m = capture_and_view(neff, os.path.join(args.workdir, sub), args.world_size)
        a = analyze(m)
        a["neff"] = neff
        a["raw_summary"] = m
        results[label] = a
        print(_fmt(a, label))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nmetrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
