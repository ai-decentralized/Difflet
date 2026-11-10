"""Minimal repro: collective-matmul rings across two transformer-like blocks, padded sequence.

    python -m benchmark.tpu_profile.cm_padded_repro [--seq 2310] [--variant full]

LTX-2's SP forward is right at 6144 video rows (no padding) and wrong -- or the
core halts -- at 2310 rows padded to 2312 once two blocks share one compiled
graph (2026-11-10). This strips the model away: random weights, a block of

    n = rms(x); q,k,v = AG(n); attn(q, k[:S], v[:S]); x += RS(o)
    n = rms(x); h = AG(n); x += RS(gelu(h))

run ``--blocks`` times inside one ``torch.compile`` graph on the chips, against
an fp32 CPU reference that rank 0 rebuilds from the same seeds (no TPU
collective in the reference). ``--variant`` drops pieces to bisect:
``full`` | ``noattn`` (o = q) | ``attnonly`` (no FFN) | ``ffonly`` | ``rsonly``
(x += RS(AG(x)), no norm or activation).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

HID, HEADS, HD, FF = 4096, 32, 128, 8192


def _weights(tp, rank, block, device=None, dtype=None):
    import torch

    g = torch.Generator().manual_seed(1000 * block + rank)
    loc = HID // tp
    ffl = FF // tp

    def w(o, i):
        t = torch.randn(o, i, generator=g) / i ** 0.5
        return t if device is None else t.to(device, dtype)

    return {"q": w(loc, HID), "k": w(loc, HID), "v": w(loc, HID), "o": w(HID, loc),
            "f1": w(ffl, HID), "f2": w(HID, ffl)}


def _worker(rank, world, args, reply_q):
    try:
        import os

        base = os.environ.get("REPRO_DUMP_BASE")
        if base:  # per-rank compiler dumps, set before the runtime starts
            os.environ["XLA_FLAGS"] = f"--xla_dump_to={base}/rank{rank}/xla --xla_dump_hlo_as_text"
            os.environ["TORCH_TPU_INTERNAL_DUMP_MLIR_PASSES_TO"] = f"{base}/rank{rank}/mlir"
            os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = f"{base}/rank{rank}/t3"
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)
        import torch
        import torch.nn.functional as F

        from difflet.backends.tpu.ltx_2 import sp
        from difflet.backends.tpu.ops_impl import collective_matmul as cm
        from difflet.backends.tpu.ops_impl import parallel_mesh
        from difflet.ops import attention, gather_from_sequence_parallel_region
        from difflet.pipeline.parallel_mesh import MeshSpec

        parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))
        cm.prepare(sp.LTX2_OPS)
        tp_rank = parallel_mesh.get_tp_rank()
        dev, bf = device_runtime.device(), torch.bfloat16
        seq = args.seq
        padded = sp.padded_length(seq, world) if args.pad else seq
        n = padded // world
        heads = HEADS // world

        class Lin:  # what the rings read: .weight / .bias
            def __init__(self, w):
                self.weight, self.bias = w, None

        blocks = [{k: Lin(v) for k, v in _weights(world, tp_rank, b, dev, bf).items()}
                  for b in range(args.blocks)]
        g = torch.Generator().manual_seed(7)
        x_full = torch.randn(1, seq, HID, generator=g)

        def rms(t):
            return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(t.dtype)

        def block_fn(x, w):
            if args.variant in ("full", "noattn", "attnonly"):
                q, k, v = cm.all_gather_matmul(rms(x), (w["q"], w["k"], w["v"]))
                if args.variant in ("full", "attnonly"):
                    b, s, inner = q.shape
                    k, v = k[:, :seq], v[:, :seq]

                    def bh(t):
                        return t.reshape(b, t.shape[1], heads, HD).permute(0, 2, 1, 3).reshape(
                            b * heads, t.shape[1], HD)

                    o = attention(bh(q), bh(k), bh(v), scale=HD ** -0.5, causal=False)
                    o = o.reshape(b, heads, s, HD).permute(0, 2, 1, 3).reshape(b, s, inner)
                else:
                    o = q
                x = x + cm.matmul_reduce_scatter(o, w["o"])
            if args.variant in ("full", "noattn", "ffonly"):
                (h,) = cm.all_gather_matmul(rms(x), (w["f1"],))
                x = x + cm.matmul_reduce_scatter(F.gelu(h, approximate="tanh"), w["f2"])
            if args.variant == "rsonly":
                (h,) = cm.all_gather_matmul(x, (w["f1"],))
                x = x + cm.matmul_reduce_scatter(h, w["f2"])
            if args.variant == "agonly":
                # AG ring only: this rank's rows of the gathered product, projected
                # back with a local (B, n, FF/tp) x (FF/tp, HID) -- no collective.
                (h,) = cm.all_gather_matmul(x, (w["f1"],))
                own = h[:, tp_rank * n:(tp_rank + 1) * n]
                x = x + own @ w["f2"].weight.T
            if args.variant == "rsonly2":
                # RS ring only: a full-sequence input built locally (the shard
                # tiled tp times), reduced and scattered back to this rank's rows.
                full = x.repeat(1, world, 1)
                part = full @ w["f1"].weight.T  # (B, S, FF/tp), local
                x = x + cm.matmul_reduce_scatter(part, w["f2"])
            return x

        def model(xf, off):
            if args.salt:
                # A distinct constant per run: a new graph hash, so every cache
                # layer misses and the compiler really runs (and dumps). XLA
                # folds the zero away; the optimized program is unchanged.
                xf = xf + xf.new_tensor(args.salt) * 0
            if args.pad_mode == "dynamic":
                # Padded in the graph as in "full", but the shard offset is a
                # runtime input, so every rank compiles the same program.
                xp = sp._pad_seq(xf, 1, padded, 0.0)
                x = torch.index_select(xp, 1, off + torch.arange(n, device=xf.device))
            elif args.pad_mode == "outside":
                # xf is already padded (eagerly, outside the graph): no cat in here.
                x = xf[:, tp_rank * n:(tp_rank + 1) * n]
            elif args.pad_mode == "shard":
                # Slice the shard off the graph input first, then pad only that
                # shard (only the last rank gets any rows).
                x = sp._pad_seq(xf[:, tp_rank * n:min((tp_rank + 1) * n, seq)], 1, n, 0.0)
            else:  # "full": pad the whole sequence, then slice (the original)
                x = sp._pad_seq(xf, 1, padded, 0.0)[:, tp_rank * n:(tp_rank + 1) * n]
            for w in blocks:
                x = block_fn(x, w)
            out = gather_from_sequence_parallel_region(x, dim=1)
            return out if args.pad_mode == "outside" else out[:, :seq]

        stopper = None
        if args.trace:
            # Started before the run; a timer thread stops it, so a hang still
            # leaves a trace of what each chip completed before it stopped.
            import threading

            import jax

            jax.profiler.start_trace(f"{args.trace}/rank{rank}")

            def _stop():
                print(f"[trace] rank{rank}: stopping trace after {args.trace_after}s", flush=True)
                jax.profiler.stop_trace()
                print(f"[trace] rank{rank}: trace written", flush=True)

            stopper = threading.Timer(args.trace_after, _stop)
            stopper.daemon = True
            stopper.start()
        with torch.no_grad():
            xin = x_full.to(dev, bf)
            if args.pad_mode == "outside":
                xin = sp._pad_seq(xin, 1, padded, 0.0)  # eager, before the compiled graph
            off = torch.tensor([tp_rank * n], dtype=torch.int64).to(dev)
            got = torch.compile(model, backend="tpu")(xin, off)
            got = got[:, :seq].float().cpu()  # unpad eagerly too
        if stopper is not None and stopper.is_alive():
            stopper.cancel()
            import jax

            jax.profiler.stop_trace()
        res = {"rank": rank, "finite": bool(torch.isfinite(got).all())}
        if rank == 0:
            # fp32 CPU reference: every rank's weights rebuilt from the seeds.
            ws = [[_weights(world, r, b) for r in range(world)] for b in range(args.blocks)]
            x = x_full.clone()

            def cat_rows(key, b):
                return torch.cat([ws[b][r][key] for r in range(world)], dim=0)

            def cat_cols(key, b):
                return torch.cat([ws[b][r][key] for r in range(world)], dim=1)

            def rms32(t):
                return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)

            xp_state = torch.cat([x, x.new_zeros(1, padded - seq, HID)], dim=1)
            for b in range(args.blocks):
                if args.variant in ("full", "noattn", "attnonly"):
                    nrm = rms32(x)
                    q, k, v = (nrm @ cat_rows(key, b).T for key in ("q", "k", "v"))
                    if args.variant in ("full", "attnonly"):
                        def heads_of(t):
                            return t.reshape(1, seq, HEADS, HD).transpose(1, 2)

                        o = F.scaled_dot_product_attention(heads_of(q), heads_of(k), heads_of(v),
                                                           scale=HD ** -0.5)
                        o = o.transpose(1, 2).reshape(1, seq, HID)
                    else:
                        o = q
                    x = x + o @ cat_cols("o", b).T
                if args.variant in ("full", "noattn", "ffonly"):
                    h = rms32(x) @ cat_rows("f1", b).T
                    x = x + F.gelu(h, approximate="tanh") @ cat_cols("f2", b).T
                if args.variant == "rsonly":
                    x = x + (x @ cat_rows("f1", b).T) @ cat_cols("f2", b).T
                if args.variant == "agonly":
                    # Each rank r keeps its own rows and only its f1 / f2 slice.
                    xp = xp_state
                    outs = []
                    for r in range(world):
                        rows = xp[:, r * n:(r + 1) * n]
                        outs.append(rows + (rows @ ws[b][r]["f1"].T) @ ws[b][r]["f2"].T)
                    xp_state = torch.cat(outs, dim=1)
                    x = xp_state[:, :seq]
                if args.variant == "rsonly2":
                    # The pad rows carry state here (they feed the tiled input).
                    xp = xp_state
                    shards = [xp[:, r * n:(r + 1) * n] for r in range(world)]
                    total = sum((shards[r].repeat(1, world, 1) @ ws[b][r]["f1"].T) @ ws[b][r]["f2"].T
                                for r in range(world))  # (1, padded, HID): the reduce
                    xp_state = torch.cat([shards[r] + total[:, r * n:(r + 1) * n]
                                          for r in range(world)], dim=1)
                    x = xp_state[:, :seq]
            err = (got - x).norm() / x.norm()
            rows = ((got - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6))[0]
            res.update(rel=round(float(err), 5),
                       worst_rows=[int(i) for i in rows.topk(6).indices],
                       worst_vals=[round(float(v), 3) for v in rows.topk(6).values])
        reply_q.put(res)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        reply_q.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}",
                     "tb": traceback.format_exc()[-2000:]})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=2310)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--pad-mode", default="full", choices=("full", "shard", "outside", "dynamic"))
    parser.add_argument("--salt", type=float, default=0.0,
                        help="nonzero: force a fresh compile (see model())")
    parser.add_argument("--trace", default="", help="jax.profiler trace dir (one subdir per rank)")
    parser.add_argument("--trace-after", type=float, default=90.0,
                        help="seconds after which a timer thread stops the trace (hang case)")
    parser.add_argument("--variant", default="full", choices=("full", "noattn", "ffonly", "rsonly", "attnonly", "agonly", "rsonly2"))
    args = parser.parse_args()
    from difflet.backends.tpu.ops_impl import device_runtime

    world = 4
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, args, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    results = sorted((reply_q.get(timeout=900) for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join(timeout=60)
    print(vars(args), results[0], [r.get("finite") for r in results], flush=True)
    for r in results:
        if "error" in r:
            print(f"rank {r['rank']} FAILED: {r['error']}\n{r['tb']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
