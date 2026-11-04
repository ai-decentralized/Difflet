"""Can a Pallas kernel inside ``jax_op`` reach the other chips?

A collective-matmul kernel overlaps the row-parallel matmul with its reduce by
DMAing partial sums to the neighbouring chips from inside the kernel. Pallas
does that within ONE JAX program that sees every chip of the mesh. torch_tpu
gives each process one chip, so the question is what JAX sees from inside a
torch_tpu worker -- at module level and while ``jax_op`` traces its function.
If every rank sees only its own chip, the kernel cannot be written this way.

    python -m benchmark.tpu_profile.jax_devices_probe
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path


def worker(rank, world, reply_q):
    try:
        from difflet.backends.tpu.ops_impl import device_runtime

        device_runtime.initialize_replica(rank, world)

        import jax
        import torch
        import torch_tpu  # noqa: F401

        seen = {"rank": rank, "jax.devices": [str(d) for d in jax.devices()],
                "jax.local_devices": [str(d) for d in jax.local_devices()],
                "process_count": jax.process_count()}

        def probe(x):
            seen["inside_jax_op.devices"] = [str(d) for d in jax.devices()]
            return x * 2

        probe.__annotations__ = {"x": jax.Array, "return": jax.Array}
        op = torch.tpu.pallas.jax_op("difflet::devices_probe", probe)
        out = op(torch.ones(4, device=device_runtime.device()))
        device_runtime.wait()
        seen["op_result"] = out.cpu().tolist()
        reply_q.put(seen)
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        import traceback

        reply_q.put({"rank": rank, "error": repr(exc), "traceback": traceback.format_exc()})


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
        print(reply_q.get(timeout=600), flush=True)
    for p in procs:
        p.join(timeout=60)


if __name__ == "__main__":
    main()
