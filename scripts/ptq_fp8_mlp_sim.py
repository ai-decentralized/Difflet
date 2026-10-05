#!/usr/bin/env python3
"""CPU-simulate nkilib's fused MLP kernel in Wan's FFN configuration and compare with the CPU
reference of the same fp8 math:

    y = gelu_tanh(fp8(x / s1) @ W1_fp8 * (s1 * w1) + b1)  ->  fp8(h / s2) @ W2_fp8 * (s2 * w2) + b2

nkilib.core.mlp.mlp with skip_gate_proj=True (up -> act -> down), ActFnType.GELU_Tanh_Approx,
QuantizationType.STATIC (per-tensor weight + static input scales, double-row fp8 matmuls).

    NKI_FP8_E4M3_MODE=non_ocp python scripts/ptq_fp8_mlp_sim.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NKI_FP8_E4M3_MODE", "non_ocp")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import nki  # noqa: E402
import torch  # noqa: E402

FP8_MAX = 240.0


def metrics(ref, test):
    ref = ref.float().flatten(); test = test.float().flatten()
    err = (test - ref).pow(2).sum().item(); sig = ref.pow(2).sum().item()
    cos = torch.nn.functional.cosine_similarity(ref, test, dim=0).item()
    return round(cos, 6), round(10 * math.log10(sig / err), 2) if err > 0 else float("inf")


def q(x, s):
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def case(T, H, I, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, H, generator=g).to(torch.bfloat16)
    w1 = (torch.randn(H, I, generator=g) / math.sqrt(H)).to(torch.bfloat16)  # [H, I] (in, out)
    w2 = (torch.randn(I, H, generator=g) / math.sqrt(I)).to(torch.bfloat16)
    b1 = (torch.randn(1, I, generator=g) * 0.02).to(torch.bfloat16)
    b2 = (torch.randn(1, H, generator=g) * 0.02).to(torch.bfloat16)
    ws1 = w1.float().abs().max() / FP8_MAX
    ws2 = w2.float().abs().max() / FP8_MAX
    s1 = x.float().abs().max() * 1.25 / FP8_MAX
    h_ref_pre = (q(x, s1).float() @ q(w1, ws1).float()) * (s1 * ws1) + b1.float()
    h_ref = torch.nn.functional.gelu(h_ref_pre, approximate="tanh")
    s2 = h_ref.abs().max() * 1.25 / FP8_MAX
    y_ref = (q(h_ref, s2).float() @ q(w2, ws2).float()) * (s2 * ws2) + b2.float()
    y_bf16 = torch.nn.functional.gelu(x.float() @ w1.float() + b1.float(), approximate="tanh") @ w2.float() + b2.float()

    return x, (w1, ws1, b1), (w2, ws2, b2), (s1, s2), y_ref, y_bf16


def run(T, H, I):
    from nkilib.core.mlp.mlp import mlp
    from nkilib.core.utils.common_types import ActFnType, QuantizationType

    x, (w1, ws1, b1), (w2, ws2, b2), (s1, s2), y_ref, y_bf16 = case(T, H, I)
    col = lambda v: torch.full((128, 1), float(v), dtype=torch.float32)  # noqa: E731
    w1q, w2q = q(w1, ws1), q(w2, ws2)
    # CTE + STATIC takes the input already quantized (fp8 hidden, dequant scale s1).
    out = nki.simulate(mlp)(
        q(x, s1), w1q, w1q, w2q, output_dtype=torch.bfloat16,
        up_proj_bias_tensor=b1, down_proj_bias_tensor=b2,
        activation_fn=ActFnType.GELU_Tanh_Approx, skip_gate_proj=True,
        quantization_type=QuantizationType.STATIC,
        up_w_scale=col(ws1), gate_w_scale=col(ws1), down_w_scale=col(ws2),
        gate_up_in_scale=col(s1), down_in_scale=col(s2), force_cte_mode=True,
    )
    out = out[0] if isinstance(out, (list, tuple)) else out
    out = torch.as_tensor(out).reshape(y_ref.shape)
    print(f"T={T} H={H} I={I}: vs fp8 reference {metrics(y_ref, out)}, vs bf16 {metrics(y_bf16, out)}, "
          f"reference vs bf16 {metrics(y_bf16, y_ref)}", flush=True)


if __name__ == "__main__":
    shapes = [(512, 1024, 1024), (640, 1024, 1536)]
    for T, H, I in shapes:
        try:
            run(T, H, I)
        except Exception:
            traceback.print_exc()
