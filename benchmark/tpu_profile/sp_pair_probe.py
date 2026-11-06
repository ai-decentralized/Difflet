"""Does a sequence-parallel pair beat the all-reduce it replaces, on torch_tpu?

Megatron-SP splits each row-parallel all-reduce into a reduce-scatter (paired
with the row-parallel matmul) and an all-gather (paired with the NEXT
column-parallel matmul); everything between them -- residual, norm, adaLN --
is per token, so it runs on the 1/n of the sequence each chip holds.
collective_matmul_probe.py measured the RS half alone (J5rs); this probe
measures both halves of each pair in a Wan 14B block, against what runs today:

    pair            row-parallel (K/n x N)     next column-parallel (N x M/n)
    self_to_cross   self-attn out 1280x5120 -> cross-attn q 5120x1280
    cross_to_ffn    cross-attn out 1280x5120 -> FFN up 5120x3456
    ffn_to_qkv      FFN down 3456x5120      -> next self-attn q|k|v 5120x3840

Per pair, compiled, at S = 4680 tokens -- the benchmark's Wan shape (480x832,
9 frames: 3 x 30 x 52 latent patches, guidance 1.0 so one batch); inputs are
seeded random tensors at Wan's shapes, not the model's activations or weights:

    M      row matmul alone, no reduce               (compute floor, row half)
    T      row matmul + all-reduce, col matmul on the full x  (today)
    RS0    row matmul then psum_scatter              (no overlap)
    J5rs   bidirectional ring RS, overlapped         (collective_matmul_probe)
    AG0    all_gather x then col matmul              (no overlap)
    J5ag   bidirectional ring AG, overlapped: half of each chunk's rows go
           clockwise, half counter-clockwise; each hop's matmul runs while
           the next hop is in flight
    C      col matmul alone on the full x            (compute floor, col half)

The pair totals compared are T_row + C (today) against RS0 + AG0 and
J5rs + J5ag. Each SP output is checked against the torch reference rows.

    DIFFLET_TPU_RUNTIME=torch_tpu python -m benchmark.tpu_profile.sp_pair_probe

``SP_PAIRS=ffn_to_qkv`` runs a subset; ``SP_TRACE_DIR`` writes a jax.profiler
trace of rank 0 for each variant.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

S, D = 4680, 5120
#: (label, row K, col M): full (unsharded) inner dims of the two matmuls.
PAIRS = (("self_to_cross", 5120, 5120), ("cross_to_ffn", 5120, 13824),
         ("ffn_to_qkv", 13824, 3 * 5120))
R = 10


def worker(rank, world, reply_q):
    import faulthandler

    # A hung collective blocks in synchronize forever; dump every stack and exit.
    faulthandler.dump_traceback_later(int(os.environ.get("SP_WATCHDOG_S", "600")), exit=True)
    try:
        _worker(rank, world, reply_q)
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        import traceback

        reply_q.put({"rank": rank, "error": repr(exc), "traceback": traceback.format_exc()})


def _snake(devices):
    def key(d):
        x, y = d.coords[0], d.coords[1]
        return (y, x if y % 2 == 0 else -x)

    return sorted(devices, key=key)


def _worker(rank, world, reply_q):
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.initialize_replica(rank, world)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from jax.sharding import PartitionSpec as P

    from difflet.backends.tpu.ops_impl import collectives, parallel_mesh
    from difflet.pipeline.parallel_mesh import MeshSpec

    parallel_mesh.init_parallel_mesh(MeshSpec(tp=world))
    collectives.warm_process_groups()
    dev = device_runtime.device()
    shard_map = jax.shard_map if hasattr(jax, "shard_map") else \
        __import__("jax.experimental.shard_map", fromlist=["shard_map"]).shard_map

    rows = S // world
    half_rows = rows // 2
    cw = [(i, (i + 1) % world) for i in range(world)]
    ccw = [(i, (i - 1) % world) for i in range(world)]
    mesh = jax.sharding.Mesh(np.array(_snake(jax.devices())), ("tp",))
    pos = list(mesh.devices.flat).index(jax.local_devices()[0])

    def row_chunk(x, chunk):
        return jax.lax.dynamic_slice_in_dim(x, chunk * rows, rows, axis=0)

    # -- reduce-scatter side: x (S, K/n) @ w (K/n, D) -> this chip's rows ----
    def rs0(x, w, ids):
        return jax.lax.psum_scatter(x @ w, "tp", scatter_dimension=0, tiled=True)

    def j5rs(x, w, ids):
        j = ids[0]
        half = w.shape[1] // 2
        wl, wr = w[:, :half], w[:, half:]
        left = right = None
        for step in range(world):
            pl = row_chunk(x, (j - 1 - step) % world) @ wl
            pr = row_chunk(x, (j + 1 + step) % world) @ wr
            if left is None:
                left, right = pl, pr
            else:
                left = jax.lax.ppermute(left, "tp", cw) + pl
                right = jax.lax.ppermute(right, "tp", ccw) + pr
        return jnp.concatenate([left, right], axis=1)

    # -- all-gather side: x (S/n, D) @ w (D, M/n) -> (S, M/n) ----------------
    def ag0(x, w, ids):
        return jax.lax.all_gather(x, "tp", axis=0, tiled=True) @ w

    def j5ag(x, w, ids):
        # The top half of this chip's rows travels clockwise, the bottom half
        # counter-clockwise. After s hops the top half held is chip j-s's, the
        # bottom half chip j+s's; each lands at its owner's rows of the output.
        j = ids[0]
        out = jnp.zeros((S, w.shape[1]), dtype=x.dtype)
        top, bot = x[:half_rows], x[half_rows:]
        for step in range(world):
            if step:
                top = jax.lax.ppermute(top, "tp", cw)
                bot = jax.lax.ppermute(bot, "tp", ccw)
            out = jax.lax.dynamic_update_slice_in_dim(
                out, top @ w, ((j - step) % world) * rows, axis=0)
            out = jax.lax.dynamic_update_slice_in_dim(
                out, bot @ w, ((j + step) % world) * rows + half_rows, axis=0)
        return out

    def op(name, body, in_specs, out_spec):
        def fn(x, w, ids):
            return shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=out_spec,
                             check_vma=False)(x, w, ids)

        fn.__annotations__ = {"x": jax.Array, "w": jax.Array, "ids": jax.Array,
                              "return": jax.Array}
        # Output specs in a tuple: see collective_matmul_probe (PartitionSpec
        # is not a tuple in jax 0.12; a bare one is iterated by torch_tpu).
        jop = torch.tpu.pallas.jax_op(f"difflet::sp_pair_{name}", fn, mesh=mesh,
                                      input_partition_specs=in_specs,
                                      output_partition_specs=(out_spec,))
        torch._dynamo.reset()
        return torch.compile(lambda x, w, i: jop(x, w, i), backend="tpu")

    def tc(f):
        torch._dynamo.reset()
        return torch.compile(f, backend="tpu")

    def timed(fn, *args):
        out = fn(*args)
        device_runtime.wait()
        for _ in range(2):
            out = fn(*args)
        device_runtime.wait()
        mark = time.monotonic()
        for _ in range(R):
            out = fn(*args)
        device_runtime.wait()
        return out, (time.monotonic() - mark) / R * 1000

    ids = torch.tensor([pos], dtype=torch.int32).to(dev)
    results = {"rank": rank, "ring_pos": pos, "S": S,
               "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", "")}
    trace_dir = os.environ.get("SP_TRACE_DIR")
    wanted = os.environ.get("SP_PAIRS")
    #: Variants to skip (comma list of M,T_row,RS0,J5rs,C,AG0,J5ag): bisects a hang.
    skip = set(filter(None, os.environ.get("SP_SKIP", "").split(",")))
    pairs = [p for p in PAIRS if not wanted or p[0] in wanted.split(",")]

    rs_specs, rs_out = (P(None, "tp"), P("tp", None), P("tp")), P("tp", None)
    ag_specs, ag_out = (P("tp", None), P(None, "tp"), P("tp")), P(None, "tp")

    with torch.no_grad():
        for label, k, m in pairs:
            res = {}
            try:
                g = torch.Generator().manual_seed(rank)
                x_row = (torch.randn(S, k // world, generator=g) / 8).to(torch.bfloat16).to(dev)
                w_row = (torch.randn(k // world, D, generator=g) / 8).to(torch.bfloat16).to(dev)
                w_col = (torch.randn(D, m // world, generator=g) / 8).to(torch.bfloat16).to(dev)

                # Row half. T's output is the replicated sum; it is also the
                # col half's input, so both references chain as in a block.
                if "M" not in skip:
                    _, res["M"] = timed(tc(lambda x, w: F.linear(x, w.t().contiguous())), x_row, w_row)
                if "T_row" not in skip:
                    y_full, res["T_row"] = timed(
                        tc(lambda x, w: collectives.reduce_tp(F.linear(x, w.t().contiguous()))),
                        x_row, w_row)
                else:  # the same replicated activation on every rank, made on the host
                    y_full = (torch.randn(S, D, generator=torch.Generator().manual_seed(0)) / 8
                              ).to(torch.bfloat16).to(dev)
                # clone, not contiguous(): a row slice is already contiguous, so
                # contiguous() returns a view at a nonzero storage offset (every
                # rank but ring position 0), and J5ag under jax_op then hangs in
                # its first synchronize -- deterministically, torch_tpu 0.2.0
                # (2026-11-03). AG0 tolerates it; a fresh buffer fixes both.
                y_rows = y_full[pos * rows:(pos + 1) * rows].clone()
                ref_rows = y_rows.float().cpu()
                for name, body in (("RS0", rs0), ("J5rs", j5rs)):
                    if name in skip:
                        continue
                    out, res[name] = timed(op(f"{label}_{name}", body, rs_specs, rs_out),
                                           x_row, w_row, ids)
                    res[name + "_err"] = (out.float().cpu() - ref_rows).abs().max().item()

                # Col half: today it runs on the replicated y_full, no comm.
                z_ref, c_ms = timed(tc(lambda y, w: F.linear(y, w.t().contiguous())),
                                    y_full, w_col)
                if "C" not in skip:
                    res["C"] = c_ms
                z_ref = z_ref.float().cpu()
                for name, body in (("AG0", ag0), ("J5ag", j5ag)):
                    if name in skip:
                        continue
                    print(f"[rank {rank}] {label}: {name} start", file=sys.stderr, flush=True)
                    out, res[name] = timed(op(f"{label}_{name}", body, ag_specs, ag_out),
                                           y_rows, w_col, ids)
                    res[name + "_err"] = (out.float().cpu() - z_ref).abs().max().item()

                for total, parts in (("today", ("T_row", "C")), ("sp_plain", ("RS0", "AG0")),
                                     ("sp_overlap", ("J5rs", "J5ag")), ("floor", ("M", "C"))):
                    if all(p_ in res for p_ in parts):
                        res[total] = sum(res[p_] for p_ in parts)

                if trace_dir:
                    if rank == 0:
                        jax.profiler.start_trace(os.path.join(trace_dir, label))
                    for _ in range(3):
                        op(f"{label}_J5ag_t", j5ag, ag_specs, ag_out)(y_rows, w_col, ids)
                    device_runtime.wait()
                    if rank == 0:
                        jax.profiler.stop_trace()
            except Exception as exc:  # noqa: BLE001
                import traceback

                res["error"] = repr(exc)[:300]
                print(f"[rank {rank}] {label} failed:\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
            results[label] = {k_: (round(v, 4) if isinstance(v, float) else v)
                              for k_, v in res.items()}
    reply_q.put(results)


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    for _ in range(world):
        msg = reply_q.get(timeout=1500)
        if msg.get("rank") == 0 or "error" in msg:
            print(msg, flush=True)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
