"""Where do FLUX's outputs start to differ when the AG kernel hands q/k/v heads
first (``DIFFLET_TPU_FLUX_KERNEL_HEADS``)? The kernel's values are bit-identical
(cm_pallas_bench --heads); this follows the same bf16 q through the ops after
it, on one chip, compiled, in both layouts:

    old: y = x + b on [B, S, H*D], then unflatten + transpose to [B, H, S, D]
    new: x arrives as [B, H, S, D] (a contiguous input, as the kernel writes it),
         y = x + b.view(H, 1, D)

and compares, stage by stage, bit for bit: the bias add, the RMSNorm's float32
mean of squares, the RMSNorm output (diffusers ``RMSNorm``, FLUX's eps and a
bf16 weight), and the split-halves RoPE after it.

    python -m benchmark.tpu_profile.norm_layout_probe [--seq 4608 --heads 6]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp


def _worker(args, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(0, 1)
        import torch
        from diffusers.models.normalization import RMSNorm

        from difflet.backends.tpu.flux.sp import _rope_split_halves

        dev = device_runtime.device()
        heads, d, seq = args.heads, 128, args.seq
        g = torch.Generator().manual_seed(0)
        x = (torch.randn(1, seq, heads * d, generator=g) * 2.0).to(torch.bfloat16)
        bias = (torch.randn(heads * d, generator=g) * 0.5).to(torch.bfloat16)
        norm = RMSNorm(d, eps=1e-6).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_((torch.rand(d, generator=g) + 0.5).to(torch.bfloat16))
        angles = torch.randn(seq, d, generator=g)
        cos, sin = angles.cos(), angles.sin()
        # What the kernel writes: (B, H, ring, 2, rows/2, d), reshaped to (B, H, S, d).
        x_heads = x.view(1, seq, heads, d).transpose(1, 2).contiguous()
        if not args.plain_4d:
            x_heads = x_heads.view(1, heads, args.ring, 2, seq // (2 * args.ring), d)

        norm = norm.to(dev)
        xd, xhd, bd = x.to(dev), x_heads.to(dev), bias.to(dev)
        cd, sd = cos.to(dev), sin.to(dev)

        def stages(y):  # y: [B, H, S, D] after the bias add
            meansq = y.to(torch.float32).pow(2).mean(-1, keepdim=True)
            normed = norm(y)
            roped = _rope_split_halves(normed, cd, sd)
            return y, meansq, normed, roped

        @torch.compile(backend="tpu")
        def old(x, b):
            y = (x + b).unflatten(-1, (heads, d)).transpose(1, 2)
            return stages(y)

        @torch.compile(backend="tpu")
        def new(xh, b):
            return stages(xh.reshape(1, heads, seq, d) + b.view(heads, 1, d))

        with torch.no_grad():
            a = [t.contiguous().cpu() for t in old(xd, bd)]
            n = [t.contiguous().cpu() for t in new(xhd, bd)]
        rows = []
        for name, u, v in zip(("bias add (bf16)", "mean of squares (f32)", "RMSNorm (bf16)",
                               "RoPE (bf16)"), a, n):
            diff = u != v
            row = {"stage": name, "elements": u.numel(), "differ": int(diff.sum())}
            if row["differ"]:
                uf, vf = u.float(), v.float()
                rel = ((uf - vf).abs() / vf.abs().clamp_min(1e-30))[diff]
                row["max_rel"] = float(rel.max())
                if u.dtype == torch.bfloat16:  # how many bf16 units in the last place apart
                    ulps = (u.view(torch.int16).int() - v.view(torch.int16).int()).abs()[diff]
                    row["max_ulp"] = int(ulps.max())
            rows.append(row)
        reply_q.put({"rows": rows})
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        reply_q.put({"error": repr(exc)})


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=4608)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ring", type=int, default=4)
    parser.add_argument("--plain-4d", action="store_true",
                        help="hand the new path a plain (B, H, S, d) input, no 6-D reshape")
    args = parser.parse_args()
    device_runtime.prepare_launch(1)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(args, reply_q))
    proc.start()
    result = reply_q.get(timeout=1800)
    proc.join(timeout=60)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
