"""Manual check: neuron backend norm, rotary-embedding and platform ops on hardware.

Run on a trn2 host with the TorchNeuron stack (single process):

    python tests/manual/check_neuron_ops_c2.py

Each op is dispatched through difflet.ops with DIFFLET_BACKEND=neuron at Wan 2.2
A14B shapes. Pass criteria: fp32 on the device matches CPU fp32 to 1e-4
(relative to the output's magnitude); bf16 on the device is no further from the
CPU fp32 result than 1.5x the CPU bf16 error; no op falls back to CPU.
"""

from __future__ import annotations

import os
import sys

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from difflet import ops
from difflet.backends.neuron.runtime import track_fallbacks

S, D, HEADS, HEAD_DIM = 4680, 5120, 40, 128


def run(fn, inputs, module, device, dtype):
    if module is not None:
        module.to(device=device, dtype=dtype)
    args = [x.to(device=device, dtype=dtype) for x in inputs]
    with torch.no_grad():
        out = fn(*args)
    return out.to("cpu", torch.float64)


def check(name, fn, inputs, module=None):
    # fp32 runs first: casting the module to bf16 rounds its weights for good
    ref = run(fn, inputs, module, "cpu", torch.float32)
    with track_fallbacks() as fallbacks_fp32:
        dev_fp32 = run(fn, inputs, module, "neuron", torch.float32)
    cpu_bf16 = run(fn, inputs, module, "cpu", torch.bfloat16)
    with track_fallbacks() as fallbacks_bf16:
        dev_bf16 = run(fn, inputs, module, "neuron", torch.bfloat16)
    scale = max(1.0, ref.abs().max().item())
    fp32_err = (dev_fp32 - ref).abs().max().item() / scale
    cpu_bf16_err = (cpu_bf16 - ref).abs().mean().item()
    dev_bf16_err = (dev_bf16 - ref).abs().mean().item()
    fallbacks = fallbacks_fp32 + fallbacks_bf16
    ok = fp32_err <= 1e-4 and dev_bf16_err <= 1.5 * cpu_bf16_err + 1e-6 and not fallbacks
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} fp32 rel_err={fp32_err:.1e}  "
          f"bf16 mean_err device={dev_bf16_err:.2e} cpu={cpu_bf16_err:.2e}  fallbacks={fallbacks}",
          flush=True)
    return ok


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(1, S, D)
    results = []

    rms = ops.RMSNorm(D, eps=1e-6)
    rms.weight.data = 1 + 0.1 * torch.randn(D)
    results.append(check("RMSNorm", rms, [x], rms))

    for affine in (True, False):
        ln = ops.LayerNorm(D, eps=1e-6, elementwise_affine=affine)
        if affine:
            ln.weight.data = 1 + 0.1 * torch.randn(D)
            ln.bias.data = 0.1 * torch.randn(D)
        results.append(check(f"LayerNorm affine={affine}", ln, [x], ln))

    q = torch.randn(1, S, HEADS, HEAD_DIM)
    angles = torch.rand(1, S, 1, HEAD_DIM // 2).repeat_interleave(2, dim=-1) * 6.283
    results.append(check("apply_rotary_emb", ops.apply_rotary_emb, [q, angles.cos(), angles.sin()]))

    target = ops.get_platform_target()
    platform_ok = ops.hardware(target).value == target
    print(f"{'PASS' if platform_ok else 'FAIL'}  get_platform_target          {target!r}", flush=True)
    results.append(platform_ok)

    print(f"{sum(results)}/{len(results)} passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
