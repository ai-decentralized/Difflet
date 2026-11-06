"""Can a row-parallel matmul overlap its reduce on torch_tpu, through jax_op?

TorchTPU runs every reduce of a compiled Wan block strictly after its matmul
(2.5 ms of 8.8 per block), and neither chunking, libtpu's async-collective
flags nor functional collectives changed that (benchmark/tpu_profile,
2026-10-27). ``jax_op`` takes a ``mesh`` and partition specs, treating each
process's tensor as its shard of one global array, so a JAX program spanning
all four chips runs inside a torch_tpu model -- compiled, at no measurable
bridge cost (J1 against T below).

At a row-parallel shape (x: S x K split by columns, w: K x N split by rows),
all compiled, each checked against T:

    T     torch: F.linear + the custom-op all-reduce (what runs today)
    J1    jax_op: x @ w, then psum
    J3    ring collective matmul over jax.devices() order, one direction
    J4    the same ring in physical order (2x2 snake), one direction
    J5    physical ring, both directions (half the columns each way)
    J5rs  J5 without the closing all_gather: the reduce-scatter half alone,
          the shape a sequence-parallel block would consume
    J6    jax_op: a plain global x @ w, no shard_map -- the SPMD partitioner
          places the reduce, so XLA's windowed einsum (collective matmul)
          can rewrite it when its libtpu flags are on
    J6rs  J6 with the output sharded by rows: the partitioner's reduce-scatter

J6 (2026-10-28, libtpu 0.0.51, torch_tpu 0.2.0): no partitioner runs. Eager
and compiled alike fail in call_custom_kernel with "invalid refinement for
argument 0 ... tensor<4680x5120xbf16> -> tensor<4680x1280xbf16>": the module
is refined straight to the local shard, sdy.sharding attrs and constraints
notwithstanding. Only per-device (shard_map) bodies run under jax_op, so
XLA's windowed einsum is out of reach there; the windowed-einsum flags
(reduce_scatter / all_gather / xla_jf_spmd_threshold) exist, no all_reduce
one. J6rs fails at the fake-tensor step exactly like J5rs.

First pass (2026-10-27): J3 was async (collective-permute-start/done) but
waited on every transfer -- two of its four hops were diagonal in the 2x2.

    DIFFLET_TPU_RUNTIME=torch_tpu python -m benchmark.tpu_profile.collective_matmul_probe

``CM_VARIANTS=J5,J6`` runs a subset (T always runs), ``CM_EAGER=1`` skips
torch.compile around the jax_op ops; ``LIBTPU_INIT_ARGS``
reaches every worker, so the windowed-einsum flags apply to J6 as passed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

S, N = 4680, 5120
#: (label, K): Wan's attention output projection and FFN down-projection.
SHAPES = (("attn_out", 5120), ("ffn_down", 13824))
R = 10


def worker(rank, world, reply_q):
    try:
        _worker(rank, world, reply_q)
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        import traceback

        reply_q.put({"rank": rank, "error": repr(exc), "traceback": traceback.format_exc()})


def _shard_map():
    import jax

    if hasattr(jax, "shard_map"):
        return jax.shard_map
    from jax.experimental.shard_map import shard_map

    return shard_map


def _snake(devices):
    """Devices in ring order on their physical grid: rows alternate direction."""
    def key(d):
        x, y = d.coords[0], d.coords[1]
        return (y, x if y % 2 == 0 else -x)

    return sorted(devices, key=key)


def _worker(rank, world, reply_q):
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.initialize_replica(rank, world)

    import jax
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
    shard_map = _shard_map()

    rows = S // world
    cw = [(i, (i + 1) % world) for i in range(world)]
    ccw = [(i, (i - 1) % world) for i in range(world)]
    meshes = {"devices": jax.sharding.Mesh(np.array(jax.devices()), ("tp",)),
              "ring": jax.sharding.Mesh(np.array(_snake(jax.devices())), ("tp",))}

    def row_chunk(x, chunk):
        return jax.lax.dynamic_slice_in_dim(x, chunk * rows, rows, axis=0)

    # ``ids`` carries this chip's position in the mesh as data: axis_index
    # lowers to partitionId, which jax_op's SPMD partitioning rejects, and the
    # position is JAX's, not the torch rank (rank 0 holds TPU_2).
    def j1(x, w, ids):
        return jax.lax.psum(x @ w, "tp")

    def ring_rs(x, w, ids):
        # Device j adds its partial for row-chunk (j - 1 - step) mod n at each
        # step, after receiving the accumulator from j - 1; after n steps it
        # holds chunk j summed over every device.
        j = ids[0]
        acc = None
        for step in range(world):
            part = row_chunk(x, (j - 1 - step) % world) @ w
            acc = part if acc is None else jax.lax.ppermute(acc, "tp", cw) + part
        return acc

    def bidir_rs(x, w, ids):
        # Columns split in two: the left half travels clockwise as above, the
        # right half counter-clockwise (chunk (j + 1 + step) mod n, received
        # from j + 1), so each link carries half the bytes in each direction.
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
        return jax.numpy.concatenate([left, right], axis=1)

    def gathered(rs):
        def body(x, w, ids):
            return jax.lax.all_gather(rs(x, w, ids), "tp", axis=0, tiled=True)

        return body

    def j6(mesh, out_spec):
        # Global arrays: x (S, K) and w (K, N) arrive split along K, so the
        # partitioner must reduce the product -- how is up to XLA. Without the
        # constraints the export carries no sharding and torch_tpu refines its
        # arguments to the local shard (4680x1280 against 4680x5120).
        def pin(a, spec):
            return jax.lax.with_sharding_constraint(a, jax.sharding.NamedSharding(mesh, spec))

        def body(x, w, ids):
            return pin(pin(x, P(None, "tp")) @ pin(w, P("tp", None)), out_spec)

        return body

    variants = (("J1", "devices", j1, P()), ("J3", "devices", gathered(ring_rs), P()),
                ("J4", "ring", gathered(ring_rs), P()), ("J5", "ring", gathered(bidir_rs), P()),
                ("J5rs", "ring", bidir_rs, P("tp", None)),
                ("J6", "ring", j6(meshes["ring"], P()), P()),
                ("J6rs", "ring", j6(meshes["ring"], P("tp", None)), P("tp", None)))
    wanted = os.environ.get("CM_VARIANTS")
    if wanted:
        variants = tuple(v for v in variants if v[0] in wanted.split(","))

    def wrap(name, mesh, body, out_spec):
        specs = (P(None, "tp"), P("tp", None), P("tp"))

        def fn(x, w, ids):
            if name.split("_")[-1].startswith("J6"):
                return body(x, w, ids)
            return shard_map(body, mesh=mesh, in_specs=specs, out_specs=out_spec,
                             check_vma=False)(x, w, ids)

        fn.__annotations__ = {"x": jax.Array, "w": jax.Array, "ids": jax.Array,
                              "return": jax.Array}
        # One spec per output, in a tuple: PartitionSpec is no longer a tuple
        # (jax 0.12), so torch_tpu 0.2.0 iterates a bare P("tp", None) and
        # hands the output the axis name "tp" as its spec -- KeyError 't' when
        # the fake kernel computes the local shape. A bare P() only worked
        # because it is empty, hence falsy.
        return torch.tpu.pallas.jax_op(f"difflet::cm_probe_{name}", fn, mesh=mesh,
                                       input_partition_specs=specs,
                                       output_partition_specs=(out_spec,))

    ids = {key: torch.tensor([list(m.devices.flat).index(jax.local_devices()[0])],
                             dtype=torch.int32).to(dev)
           for key, m in meshes.items()}
    results = {"rank": rank, "ring_order": [str(d) for d in meshes["ring"].devices.flat],
               "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", "")}
    trace_dir = os.environ.get("CM_TRACE_DIR")
    eager = os.environ.get("CM_EAGER") == "1"  # call the jax_op ops directly, no torch.compile
    results["eager"] = eager

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

    with torch.no_grad():
        for label, k in SHAPES:
            gen = torch.Generator().manual_seed(rank)
            x = (torch.randn(S, k // world, generator=gen) / 8).to(torch.bfloat16).to(dev)
            w = (torch.randn(k // world, N, generator=gen) / 8).to(torch.bfloat16).to(dev)
            torch._dynamo.reset()
            torch_path = torch.compile(
                lambda x, w: collectives.reduce_tp(F.linear(x, w.t().contiguous())), backend="tpu")
            ref, ms = timed(torch_path, x, w)
            ref_cpu = ref.float().cpu()
            shape = {"T": {"ms": ms}}
            for name, mesh_key, body, out_spec in variants:
                try:
                    op = wrap(f"{label}_{name}", meshes[mesh_key], body, out_spec)
                    torch._dynamo.reset()
                    fn = op if eager else torch.compile(lambda x, w, i, op=op: op(x, w, i),
                                                        backend="tpu")
                    out, ms = timed(fn, x, w, ids[mesh_key])
                    entry = {"ms": ms}
                    if name.endswith("rs"):
                        # This chip's rows of the full sum, in ring position.
                        pos = int(ids[mesh_key].cpu()[0])
                        entry["max_abs_vs_T"] = (out.float().cpu()
                                                 - ref_cpu[pos * rows:(pos + 1) * rows]).abs().max().item()
                    else:
                        entry["max_abs_vs_T"] = (out.float().cpu() - ref_cpu).abs().max().item()
                    if trace_dir and rank == 0:
                        jax.profiler.start_trace(os.path.join(trace_dir, f"{label}_{name}"))
                    if trace_dir:
                        for _ in range(3):
                            out = fn(x, w, ids[mesh_key])
                        device_runtime.wait()
                    if trace_dir and rank == 0:
                        jax.profiler.stop_trace()
                    shape[name] = entry
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    shape[name] = {"error": repr(exc)[:300], "traceback": traceback.format_exc()[-800:]}
                    # The summary above is cut; the full chain goes to the log.
                    print(f"[rank {rank}] {label}_{name} failed:\n{traceback.format_exc()}",
                          file=sys.stderr, flush=True)
            results[label] = shape
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
