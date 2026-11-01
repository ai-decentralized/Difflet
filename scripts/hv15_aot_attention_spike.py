#!/usr/bin/env python3
"""hv15 AOT Task 0 spike: compile the bounded attention_cte call on ONE NeuronCore.

Traces ``difflet.backends.trainium.ops_impl.attention.attention`` (bound_min/bound_max
branch -> nkilib ``attention_cte[2]`` under LNC2) with ``torch_neuronx.trace``, runs one
forward and compares against a CPU fp32 masked-softmax reference. No collectives.

  bounded   : q, k, v = [heads, S, d], S = pad128(N + 1985)       (default / ulysses shape)
  gather_kv : q = [heads, pad128(N/cp + 1985), d], k = v = [heads, S, d]  (sharded-q vs full-k)

Every real query row attends keys [0, N + valid_text); padded query rows attend [0, 1).
Bounds are built in-graph with expand/cat only (no mask_to_contiguous_bounds).
"""

from __future__ import annotations

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")  # -> attention_cte[2] (LNC2 grid)

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_neuronx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from difflet.backends.trainium.ops_impl.attention import attention  # noqa: E402

TEXT_TOKENS = 1985  # hv15 fixed text-stream length (padded, per the plan)
D = 128
COMPILER_ARGS = (  # same flags as the trainium HV backbone (backbone.py get_compiler_args) + --lnc
    "--model-type=transformer -O1 --tensorizer-options='--enable-ccop-compute-overlap' "
    "--auto-cast=none --lnc=2 --internal-hlo2tensorizer-options='--verify-hlo=true'"
)


def pad128(n: int) -> int:
    return -(-n // 128) * 128


class BoundedAttn(nn.Module):
    def __init__(self, q_real: int, scale: float):
        super().__init__()
        self.q_real, self.scale = q_real, scale

    def forward(self, q, k, v, key_valid_len):
        bh, s_q, _ = q.shape
        # fp32 -> int32 (LNC2 long-int-reduction rule); expand/cat only.
        count = key_valid_len.to(torch.float32).to(torch.int32).reshape(1, 1, 1)
        real = count.expand(bh, self.q_real, 1)
        pad = torch.ones((bh, s_q - self.q_real, 1), dtype=torch.int32, device=q.device)
        bound_max = torch.cat([real, pad], dim=1).contiguous()
        bound_min = torch.zeros_like(bound_max)
        return attention(q, k, v, scale=self.scale, causal=False,
                         bound_min=bound_min, bound_max=bound_max,
                         tp_q=True, tp_k=True, tp_out=False)


def reference(q, k, v, q_real, kv_valid, scale, chunk=2048):
    """fp32 softmax(q k^T * scale) v over keys [0, kv_valid), real query rows only."""
    k = k[:, :kv_valid].float()
    v = v[:, :kv_valid].float()
    out = torch.empty(q.shape[0], q_real, q.shape[2])
    for h in range(q.shape[0]):
        for s in range(0, q_real, chunk):
            e = min(s + chunk, q_real)
            p = torch.softmax(q[h, s:e].float() @ k[h].T * scale, dim=-1)
            out[h, s:e] = p @ v[h]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bounded", "gather_kv"], required=True)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=61)
    ap.add_argument("--valid-text", type=int, default=37)
    ap.add_argument("--heads", type=int, default=8)  # 16 heads / tp2
    ap.add_argument("--cp", type=int, default=2)     # gather_kv query shard factor
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "hv15_aot_attention_spike"))
    args = ap.parse_args()

    n_img = ((args.num_frames - 1) // 4 + 1) * (args.height // 16) * (args.width // 16)
    kv_real = n_img + TEXT_TOKENS
    s_kv = pad128(kv_real)
    if args.mode == "bounded":
        q_real = kv_real
    else:
        if n_img % args.cp:
            raise SystemExit(f"N={n_img} not divisible by cp={args.cp}")
        q_real = n_img // args.cp + TEXT_TOKENS
    s_q = pad128(q_real)
    kv_valid = n_img + args.valid_text
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(args.seed)
    q = torch.randn(args.heads, s_q, D, dtype=torch.bfloat16)
    k = torch.randn(args.heads, s_kv, D, dtype=torch.bfloat16)
    v = torch.randn(args.heads, s_kv, D, dtype=torch.bfloat16)
    kvl = torch.tensor([kv_valid], dtype=torch.int32)

    info = {"mode": args.mode, "hwf": [args.height, args.width, args.num_frames],
            "N": n_img, "q_shape": list(q.shape), "kv_shape": list(k.shape),
            "q_real": q_real, "key_valid_len": kv_valid, "compiler_args": COMPILER_ARGS}
    print(json.dumps(info), flush=True)

    work = Path(args.work_dir) / f"{args.mode}_{args.height}x{args.width}x{args.num_frames}"
    work.mkdir(parents=True, exist_ok=True)
    model = BoundedAttn(q_real, scale).eval()
    saved, meta = work / "traced.pt", work / "compile.json"
    if saved.exists() and meta.exists():  # rerun the forward without recompiling
        traced = torch.jit.load(str(saved))
        info.update(json.loads(meta.read_text()), reused_neff=True)
    else:
        t0 = time.time()
        try:
            traced = torch_neuronx.trace(model, (q, k, v, kvl),
                                         compiler_workdir=str(work / "compiler"),
                                         compiler_args=COMPILER_ARGS)
        except Exception as exc:  # record the compiler failure verbatim
            info.update(compile_ok=False, compile_s=round(time.time() - t0, 1),
                        error=f"{type(exc).__name__}: {exc}")
            print("RESULT " + json.dumps(info), flush=True)
            return 2
        compiled = {"compile_ok": True, "compile_s": round(time.time() - t0, 1)}
        info.update(compiled)
        torch.jit.save(traced, str(saved))
        meta.write_text(json.dumps(compiled))
    print(f"compiled in {info['compile_s']}s", flush=True)

    with torch.no_grad():
        traced(q, k, v, kvl)  # warmup / load
        t0 = time.time()
        out = traced(q, k, v, kvl).float()
        info["forward_s"] = round(time.time() - t0, 3)

    real = out[:, :q_real]
    pad = out[:, q_real:]
    info["nan"] = bool(torch.isnan(out).any())
    info["inf"] = bool(torch.isinf(out).any())
    ref = reference(q, k, v, q_real, kv_valid, scale)
    diff = (real - ref).abs()
    info["rel_l1"] = float(diff.sum() / ref.abs().sum())
    info["max_abs"] = float(diff.max())
    # Negative control (head 0): an UNmasked reference over all keys must disagree,
    # proving the kernel honored bound_max rather than attending padded keys.
    ctrl = reference(q[:1], k[:1], v[:1], q_real, k.shape[1], scale)
    info["head0_rel_l1_vs_unmasked_ref"] = float((real[:1] - ctrl).abs().sum() / ctrl.abs().sum())
    if pad.shape[1]:  # padded query rows attend key 0 only -> v[:, 0]
        info["pad_rows_max_abs_vs_v0"] = float((pad - v[:, :1].float()).abs().max())
    print("RESULT " + json.dumps(info), flush=True)
    return 0 if not (info["nan"] or info["inf"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
