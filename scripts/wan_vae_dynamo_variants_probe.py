#!/usr/bin/env python3
"""Can torch.compile be made to work for the Wan VAE decoder on Neuron?

A first attempt failed inside Dynamo's fake-tensor propagation:

    TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
    call_function silu(...): Expected all tensors in the given list to be XLA
    tensors. Element at index 0 is not an XLA tensor. Got: XLAFloatType

That names one operator, so it may be a narrow incompatibility rather than a
dead end. This tries the ways around it, cheapest first, each in its own
subprocess so one crash does not hide the rest:

  dynamo_default      torch.compile(backend="openxla") as-is, to reproduce
  dynamo_functional   nn.SiLU swapped for torch.nn.functional.silu, in case the
                      module wrapper is what confuses fake-tensor propagation
  dynamo_nofake       Dynamo's fake-tensor propagation disabled
  xla_compile         torch_xla.compile, which takes a different path into XLA
                      than Dynamo does
  dynamo_eager_fb     backend="openxla" with graph breaks allowed to fall back,
                      so unsupported regions run eagerly instead of failing

Whatever works here decides whether the VAE can skip the ahead-of-time compile
that otherwise costs ~111 minutes per graph and exceeds the 5,000,000-instruction
ceiling past 3 latent frames.

    python scripts/wan_vae_dynamo_variants_probe.py
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
import torch.nn as nn

variant = {variant!r}
lat_d, lat_h, lat_w = {lat_d}, {lat_h}, {lat_w}
result = {{"variant": variant}}

try:
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

    torch.manual_seed(0)
    config = WanVAEDecoderConfig(
        base_dim=8, z_dim=4, dim_mult=[1, 2, 4], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[False, True], dropout=0.0, out_channels=3,
    )
    model = WanVAEDecoderModel(config).float().eval()

    if variant == "dynamo_functional":
        # diffusers' get_activation returns an nn.SiLU module; swap in the
        # functional form in case the module wrapper is what trips fake tensors.
        class FunctionalSiLU(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        def swap(m):
            for name, child in list(m.named_children()):
                if isinstance(child, nn.SiLU):
                    setattr(m, name, FunctionalSiLU())
                else:
                    swap(child)
        swap(model)

    z = torch.randn(1, config.z_dim, lat_d, lat_h, lat_w)
    with torch.no_grad():
        reference = model(z)

    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    dev_model = model.to(device)
    dev_z = z.to(device)

    if variant == "dynamo_nofake":
        import torch._dynamo.config as dcfg
        dcfg.fake_tensor_propagation = False
        dev_model = torch.compile(dev_model, backend="openxla")
    elif variant == "dynamo_eager_fb":
        import torch._dynamo as dyn
        dyn.config.suppress_errors = True
        dev_model = torch.compile(dev_model, backend="openxla")
    elif variant == "xla_compile":
        dev_model = torch_xla.compile(dev_model)
    elif variant.startswith("dynamo"):
        dev_model = torch.compile(dev_model, backend="openxla")

    with torch.no_grad():
        t0 = time.time()
        out = dev_model(dev_z)
        out_cpu = out.cpu()
        result["seconds"] = round(time.time() - t0, 1)

    result["out_shape"] = list(out_cpu.shape)
    if out_cpu.shape == reference.shape:
        result["max_diff"] = float((out_cpu - reference).abs().max())
        result["ok"] = result["max_diff"] < 1e-2
    else:
        result["ok"] = False
        result["error"] = f"shape {{tuple(out_cpu.shape)}} vs {{tuple(reference.shape)}}"
except Exception as exc:
    result["ok"] = False
    result["error"] = f"{{type(exc).__name__}}: {{str(exc)[:300]}}"

print("RESULT_JSON " + json.dumps(result))
'''

VARIANTS = [
    "dynamo_default",
    "dynamo_functional",
    "dynamo_nofake",
    "xla_compile",
    "dynamo_eager_fb",
]


def run(variant: str, lat, timeout: int) -> dict:
    code = CHILD.format(root=str(ROOT), variant=variant, lat_d=lat[0], lat_h=lat[1], lat_w=lat[2])
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
        return {"variant": variant, "ok": False, "error": f"timed out after {timeout}s"}
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    return {"variant": variant, "ok": False, "error": f"no result (rc={proc.returncode})",
            "tail": (proc.stdout + proc.stderr)[-500:]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent", type=int, nargs=3, default=(2, 8, 14))
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    args = p.parse_args()

    print(f"torch.compile variants at latent {tuple(args.latent)}\n")
    winners = []
    for v in args.variants:
        print(f"=== {v}", flush=True)
        r = run(v, tuple(args.latent), args.timeout)
        if r.get("ok"):
            winners.append(v)
            print(f"    OK  {r.get('seconds')}s  max|diff|={r.get('max_diff', 0):.2e}")
        else:
            print(f"    FAILED  {r.get('error')}")
            if r.get("tail"):
                print("    " + r["tail"].replace("\n", "\n    ")[-400:])

    print(f"\nworking torch.compile paths: {winners or 'none'}")
    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())
