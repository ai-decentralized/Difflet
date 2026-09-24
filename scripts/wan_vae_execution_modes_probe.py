#!/usr/bin/env python3
"""Which execution mode can decode the Wan VAE on Neuron without a giant graph?

The ahead-of-time path traces the decoder's per-latent-frame loop into one graph,
which exceeds neuronx-cc's 5,000,000-instruction ceiling past 3 latent frames
(NCC_EBVF030) and, even when it fits, takes ~111 minutes and over 120 GB of
compiler memory. Two other paths avoid building that graph at all:

  eager      every operator dispatches on its own; the Python loop stays a loop
  dynamo     torch.compile with the openxla backend, which inserts graph breaks
             where it cannot trace -- the mutating feat_cache list and its "Rep"
             string sentinel are exactly such points, so the decoder should split
             into many small graphs on its own

The only public Trainium2 Wan deployment we found reports both: "VAE decode
(eager on Neuron)" at 201.7 s and the same decode under torch.compile at 112.7 s,
at 768x1280 and 81 frames -- a larger shape than the one that fails to compile
here.

Each mode is run in its own subprocess: eager mode is process-global in
torch_xla, and a crash in one must not hide the others.

    python scripts/wan_vae_execution_modes_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import json, sys, time, os
sys.path.insert(0, {root!r})
os.environ.setdefault("DIFFLET_BACKEND", "trainium")
import torch

mode = {mode!r}
lat_d, lat_h, lat_w = {lat_d}, {lat_h}, {lat_w}
result = {{"mode": mode}}

try:
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

    torch.manual_seed(0)
    config = WanVAEDecoderConfig(
        base_dim=8, z_dim=4, dim_mult=[1, 2, 4], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[False, True], dropout=0.0, out_channels=3,
    )
    model = WanVAEDecoderModel(config).float().eval()
    z = torch.randn(1, config.z_dim, lat_d, lat_h, lat_w)

    with torch.no_grad():
        reference = model(z)
    result["reference_shape"] = list(reference.shape)

    if mode == "cpu":
        result["ok"] = True
        result["seconds"] = 0.0
        result["max_diff"] = 0.0
    else:
        import torch_xla
        import torch_xla.core.xla_model as xm

        if mode == "eager":
            from torch_xla.experimental import eager_mode, is_eager_mode
            eager_mode(True)
            result["eager_engaged"] = bool(is_eager_mode())

        device = xm.xla_device()
        result["device"] = str(device)
        dev_model = model.to(device)
        dev_z = z.to(device)

        if mode == "dynamo":
            dev_model = torch.compile(dev_model, backend="openxla")

        with torch.no_grad():
            t0 = time.time()
            out = dev_model(dev_z)
            out_cpu = out.cpu()
            result["seconds"] = round(time.time() - t0, 2)

        result["out_shape"] = list(out_cpu.shape)
        if out_cpu.shape == reference.shape:
            result["max_diff"] = float((out_cpu - reference).abs().max())
            result["ok"] = result["max_diff"] < 1e-2
        else:
            result["ok"] = False
            result["error"] = "shape mismatch"
except Exception as exc:
    import traceback
    result["ok"] = False
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    result["traceback"] = traceback.format_exc()[-1200:]

print("RESULT_JSON " + json.dumps(result))
'''


def run_mode(mode: str, lat: tuple[int, int, int], timeout: int) -> dict:
    code = CHILD.format(root=str(ROOT), mode=mode, lat_d=lat[0], lat_h=lat[1], lat_w=lat[2])
    env = dict(os.environ)
    env["PATH"] = f"{ROOT}/.venv/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("NEURON_RT_NUM_CORES", "1")
    try:
        proc = subprocess.run(
            [f"{ROOT}/.venv/bin/python", "-c", code],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"mode": mode, "ok": False, "error": f"timed out after {timeout}s"}
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    return {
        "mode": mode, "ok": False,
        "error": f"no result (rc={proc.returncode})",
        "tail": (proc.stdout + proc.stderr)[-700:],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent", type=int, nargs=3, default=(2, 8, 14), metavar=("D", "H", "W"))
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--modes", nargs="+", default=["cpu", "eager", "dynamo"])
    args = p.parse_args()

    print(f"decoder probe at latent {tuple(args.latent)}\n")
    results = []
    for mode in args.modes:
        print(f"=== {mode}", flush=True)
        r = run_mode(mode, tuple(args.latent), args.timeout)
        results.append(r)
        if r.get("ok"):
            extra = f" max|diff|={r['max_diff']:.2e}" if "max_diff" in r else ""
            print(f"    OK  {r.get('seconds', 0)}s  shape={r.get('out_shape', r.get('reference_shape'))}{extra}")
        else:
            print(f"    FAILED  {r.get('error')}")
            if r.get("traceback"):
                print("    " + r["traceback"].replace("\n", "\n    ")[-700:])
            elif r.get("tail"):
                print("    " + r["tail"].replace("\n", "\n    ")[-500:])

    working = [r["mode"] for r in results if r.get("ok") and r["mode"] != "cpu"]
    print(f"\non-device modes that work without compilation: {working or 'none'}")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
