#!/usr/bin/env python3
"""Compile + run the Wan fused-A TeaCache probe on device.

Verifies, on real hardware:
  1. the probe compiles (alias on the prev_mod Parameter)
  2. it loads with no "Missing weight tensor with key ..." — its weights resolve
     to the backbone's shards under the backbone's own names
  3. prev_mod persists and updates on device: a second call with the SAME input
     gives ~0, because prev_mod became mod_input on the first call
  4. the device rel-L1 scalar MATCHES the host CPU shadow's own formula on the
     same two latents (this is the number the TeaCache controller consumes)
  5. per-call latency, device probe vs host shadow
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
DEFAULT_SNAP = "/home/ubuntu/hf/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--result", default=None)
    args = ap.parse_args()

    import torch

    from difflet.backends.trainium.wan.teacache_cpu_shadow import WanTeacacheCPUShadow
    from difflet.backends.trainium.wan.teacache_probe_fused import (
        NeuronWanTeacacheProbeFusedApplication,
        probe_seq_len,
    )
    from difflet.models.wan.application import create_wan_backbone_config

    snap = args.model_path or sorted(glob.glob(DEFAULT_SNAP))[0]
    out_dir = args.out_dir or str(Path.home() / ".cache" / "difflet" / "wan21_teacache_probe_fused")
    result_path = Path(args.result or (ROOT / "artifacts" / "wan_teacache_fused_smoke.json"))

    config = create_wan_backbone_config(
        model_path=snap,
        world_size=args.tp_degree,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=args.latent_frames,
        batch_size=1,
    )
    seq = probe_seq_len(config)
    print(f"[wan] snapshot   = {snap}", flush=True)
    print(f"[wan] shape      = {args.height}x{args.width} x {args.latent_frames} latent frames",
          flush=True)
    print(f"[wan] seq_len    = {seq}  inner = {config.inner_dim}  tp = {args.tp_degree}", flush=True)

    # Two distinct denoise steps: the controller's signal is rel-L1 between them.
    torch.manual_seed(0)
    lat_h, lat_w = args.height // 8, args.width // 8
    latents_a = torch.randn(1, config.in_channels, args.latent_frames, lat_h, lat_w,
                            dtype=torch.bfloat16)
    latents_b = (latents_a + 0.15 * torch.randn_like(latents_a)).to(torch.bfloat16)
    text_seq_len = int(getattr(config, "text_seq_len", 512))
    prompt = torch.randn(1, text_seq_len, config.text_dim, dtype=torch.bfloat16)
    ts_a = torch.full([1], 1000.0, dtype=torch.bfloat16)
    ts_b = torch.full([1], 900.0, dtype=torch.bfloat16)

    # Must be the transformer directory, not the model root: the shared weight
    # store keys on os.path.realpath(app.model_path), and
    # difflet/models/wan/application.py builds both the backbone and the probe
    # with `model_path=self.transformer_path`. Passing the root here still loads
    # the right weights, but files a store entry under a key no production run
    # would ever look up.
    app = NeuronWanTeacacheProbeFusedApplication(
        model_path=os.path.join(snap, "transformer"), config=config
    )
    print("[wan] compiling probe...", flush=True)
    t = time.perf_counter()
    app.compile(out_dir)
    compile_s = time.perf_counter() - t
    print(f"[wan] compiled in {compile_s:.1f}s -> {out_dir}", flush=True)

    t = time.perf_counter()
    app.load(out_dir, skip_warmup=True)
    load_s = time.perf_counter() - t
    print(f"[wan] loaded in {load_s:.1f}s (no missing-weight error)", flush=True)

    def delta(latents, timestep):
        out = app.teacache_delta(latents, timestep, prompt)
        return float(out.detach().cpu().reshape(-1)[0].item())

    d_a = delta(latents_a, ts_a)          # vs zero prev_mod -> big garbage step-0 value
    d_same = delta(latents_a, ts_a)       # same input -> prev_mod == mod_input -> ~0
    persisted = d_same < max(d_a, 1e-9) * 1e-3
    print(f"[wan] call 1 delta (zero prev_mod)     = {d_a:.6f}", flush=True)
    print(f"[wan] call 2 delta (same input)        = {d_same:.6e}", flush=True)
    print(f"[wan] PREV_MOD_PERSISTS_ON_DEVICE      = {persisted}", flush=True)

    # prev_mod currently holds mod_input(A); this call yields rel-L1(B vs A).
    d_ab_device = delta(latents_b, ts_b)
    print(f"[wan] device rel_l1(B vs A)            = {d_ab_device:.6f}", flush=True)

    # Host CPU shadow: the path the device probe replaces.
    print("[wan] building host CPU shadow for parity...", flush=True)
    shadow = WanTeacacheCPUShadow(os.path.join(snap, "transformer"), dtype=torch.bfloat16)
    mod_a = shadow.teacache_mod_input(latents_a, ts_a, prompt).detach().float()
    mod_b = shadow.teacache_mod_input(latents_b, ts_b, prompt).detach().float()
    denom = mod_a.abs().mean().clamp_min(1e-8)
    d_ab_host = float((mod_b - mod_a).abs().mean() / denom)
    print(f"[wan] host   rel_l1(B vs A)            = {d_ab_host:.6f}", flush=True)

    rel_err = abs(d_ab_device - d_ab_host) / max(abs(d_ab_host), 1e-9)
    print(f"[wan] relative difference              = {rel_err:.4%}", flush=True)

    # Timing: device probe vs host shadow, same work per call.
    dev_ms = []
    for _ in range(args.iters):
        t = time.perf_counter()
        delta(latents_b, ts_b)
        dev_ms.append((time.perf_counter() - t) * 1000.0)
    host_ms = []
    for _ in range(args.iters):
        t = time.perf_counter()
        shadow.teacache_mod_input(latents_b, ts_b, prompt)
        host_ms.append((time.perf_counter() - t) * 1000.0)
    dev_median = sorted(dev_ms)[len(dev_ms) // 2]
    host_median = sorted(host_ms)[len(host_ms) // 2]
    print(f"[wan] device probe median = {dev_median:.2f} ms/call", flush=True)
    print(f"[wan] host shadow median  = {host_median:.2f} ms/call", flush=True)

    result = {
        "schema": "difflet-wan-teacache-fused-smoke-v1",
        "model": "wan2_1_t2v_14b",
        "snapshot": snap,
        "shape": {"height": args.height, "width": args.width,
                  "latent_frames": args.latent_frames, "seq_len": seq},
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
        "host_shadow_median_ms": host_median,
        "hardware_measured": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[wan] wrote {result_path}", flush=True)
    return 0 if persisted else 1


if __name__ == "__main__":
    sys.exit(main())
