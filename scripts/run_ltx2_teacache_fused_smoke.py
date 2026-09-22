#!/usr/bin/env python3
"""Compile + run the LTX-2 fused-A TeaCache probe on device (single mode).

Verifies, on real hardware:
  1. the probe compiles (alias on the prev_mod Parameter)
  2. it loads with no "Missing weight tensor with key ..." — the probe subclasses
     _LTX2TransformerTraceModule, so its keys are the backbone's `transformer.*`
  3. prev_mod persists and updates on device: a second call with the SAME input
     gives ~0, because prev_mod became mod_input on the first call
  4. the device rel-L1 scalar MATCHES the host CPU transformer path it replaces
  5. per-call latency, device probe vs host CPU transformer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = "/home/ubuntu/hf/hub/models--Lightricks--LTX-2/snapshots/*/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skip-host-parity", action="store_true",
                    help="skip the host CPU transformer comparison (saves a large host load)")
    ap.add_argument("--result", default=None)
    args = ap.parse_args()

    import torch

    from difflet.backends.trainium.ltx_2.teacache_probe_fused import (
        NeuronLTX2TeacacheProbeFusedApplication,
        probe_inner_dim,
    )
    from difflet.backends.trainium.ltx_2.transformer import NeuronLTX2TransformerApplication
    from difflet.models.ltx_2.application import create_ltx_2_transformer_config

    snap = args.model_path or sorted(glob.glob(DEFAULT_SNAP))[0]
    transformer_path = os.path.join(snap, "transformer")
    out_dir = args.out_dir or str(Path.home() / ".cache" / "difflet" / "ltx2_teacache_probe_fused")
    result_path = Path(args.result or (ROOT / "artifacts" / "ltx2_teacache_fused_smoke.json"))

    config = create_ltx_2_transformer_config(
        model_path=snap,
        world_size=args.tp_degree,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    seq, inner = int(config.video_seq_len), probe_inner_dim(config)
    print(f"[ltx2] snapshot   = {snap}", flush=True)
    print(f"[ltx2] shape      = {args.height}x{args.width} x {args.num_frames} frames", flush=True)
    print(f"[ltx2] seq_len    = {seq}  inner = {inner}  tp = {args.tp_degree}", flush=True)

    torch.manual_seed(0)
    lat_a = torch.randn(1, seq, config.in_channels, dtype=torch.bfloat16)
    lat_b = (lat_a + 0.15 * torch.randn_like(lat_a)).to(torch.bfloat16)
    ts_a = torch.full([1], 1.0, dtype=torch.bfloat16)
    ts_b = torch.full([1], 0.9, dtype=torch.bfloat16)

    app = NeuronLTX2TeacacheProbeFusedApplication(model_path=transformer_path, config=config)
    print("[ltx2] compiling probe...", flush=True)
    t = time.perf_counter()
    app.compile(out_dir)
    compile_s = time.perf_counter() - t
    print(f"[ltx2] compiled in {compile_s:.1f}s -> {out_dir}", flush=True)

    t = time.perf_counter()
    app.load(out_dir, skip_warmup=True)
    load_s = time.perf_counter() - t
    print(f"[ltx2] loaded in {load_s:.1f}s (no missing-weight error)", flush=True)

    def delta(latents, timestep):
        out = app.teacache_delta(latents, timestep)
        return float(out.detach().cpu().reshape(-1)[0].item())

    d_a = delta(lat_a, ts_a)
    d_same = delta(lat_a, ts_a)
    persisted = d_same < max(d_a, 1e-9) * 1e-3
    print(f"[ltx2] call 1 delta (zero prev_mod)    = {d_a:.6f}", flush=True)
    print(f"[ltx2] call 2 delta (same input)       = {d_same:.6e}", flush=True)
    print(f"[ltx2] PREV_MOD_PERSISTS_ON_DEVICE     = {persisted}", flush=True)

    d_ab_device = delta(lat_b, ts_b)
    print(f"[ltx2] device rel_l1(B vs A)           = {d_ab_device:.6f}", flush=True)

    dev_ms = []
    for _ in range(args.iters):
        t = time.perf_counter()
        delta(lat_b, ts_b)
        dev_ms.append((time.perf_counter() - t) * 1000.0)
    dev_median = sorted(dev_ms)[len(dev_ms) // 2]
    print(f"[ltx2] device probe median = {dev_median:.2f} ms/call", flush=True)

    d_ab_host = None
    rel_err = None
    host_median = None
    if not args.skip_host_parity:
        print("[ltx2] loading host CPU transformer for parity (large)...", flush=True)
        host = NeuronLTX2TransformerApplication(model_path=transformer_path, config=config)
        mod_a = host.teacache_mod_input(lat_a, ts_a).detach().float()
        mod_b = host.teacache_mod_input(lat_b, ts_b).detach().float()
        denom = mod_a.abs().mean().clamp_min(1e-8)
        d_ab_host = float((mod_b - mod_a).abs().mean() / denom)
        rel_err = abs(d_ab_device - d_ab_host) / max(abs(d_ab_host), 1e-9)
        print(f"[ltx2] host   rel_l1(B vs A)           = {d_ab_host:.6f}", flush=True)
        print(f"[ltx2] relative difference             = {rel_err:.4%}", flush=True)
        host_ms = []
        for _ in range(max(args.iters // 4, 1)):
            t = time.perf_counter()
            host.teacache_mod_input(lat_b, ts_b)
            host_ms.append((time.perf_counter() - t) * 1000.0)
        host_median = sorted(host_ms)[len(host_ms) // 2]
        print(f"[ltx2] host CPU median    = {host_median:.2f} ms/call", flush=True)

    result = {
        "schema": "difflet-ltx2-teacache-fused-smoke-v1",
        "model": "ltx_2",
        "transformer_mode": "single",
        "snapshot": snap,
        "shape": {"height": args.height, "width": args.width,
                  "num_frames": args.num_frames, "seq_len": seq},
        "tp_degree": args.tp_degree,
        "compile_s": compile_s,
        "load_s": load_s,
        "call1_delta_zero_prev": d_a,
        "call2_delta_same_input": d_same,
        "prev_mod_persists": persisted,
        "device_rel_l1_ab": d_ab_device,
        "host_rel_l1_ab": d_ab_host,
        "relative_difference": rel_err,
        "device_probe_median_ms": dev_median,
        "host_cpu_median_ms": host_median,
        "hardware_measured": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[ltx2] wrote {result_path}", flush=True)
    return 0 if persisted else 1


if __name__ == "__main__":
    sys.exit(main())
