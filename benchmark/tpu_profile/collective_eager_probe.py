"""Which torch.distributed calls work in torch_tpu eager mode, and how fast?

The peer-stash probe hung four times (send/recv, a two-rank new_group, world
broadcast, then even the all-reduce every TP layer uses) until every buffer was
written first and one element of every result read back: under lazy execution
the source's collective was dropped as dead code while the receivers waited.
This re-tests each call that way, one method per process so a real hang
cannot hide the others, with no model loaded:

    sendrecv   dist.send 0 -> 1 / dist.recv
    broadcast  dist.broadcast from rank 0, world group
    rs_ag      dist.reduce_scatter_tensor (rank 0's bytes, zeros elsewhere),
               then dist.all_gather_into_tensor back -- the peer-stash pair

    python -m benchmark.tpu_profile.collective_eager_probe <method>
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

#: GiB per transfer; 6.984 is Wan's DiT shard at tp=4.
SIZES_GIB = (1.0, 6.984)
R = 3


def worker(rank, world, method, reply_q):
    try:
        reply_q.put(_worker(rank, world, method))
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        reply_q.put({"rank": rank, "error": repr(exc)[:400], "traceback": traceback.format_exc()[-1500:]})


def _worker(rank, world, method):
    from difflet.backends.tpu.ops_impl import device_runtime

    device_runtime.initialize_replica(rank, world)

    import torch
    import torch.distributed as dist

    dev = device_runtime.device()
    gib = 2**30
    out = {"rank": rank, "method": method}

    def read(t):
        return float(t.view(-1)[-1].cpu())  # forces execution

    for size in SIZES_GIB:
        n = int(size * gib) // 2 // world * world  # divisible by world for rs_ag
        entry = {}
        times = []
        for _ in range(R):
            buf = torch.empty(n, dtype=torch.bfloat16, device=dev)
            buf.fill_(1.0 if rank == 0 else 0.0)
            read(buf)
            dist.barrier()
            mark = time.monotonic()
            if method == "sendrecv":
                if rank == 0:
                    dist.send(buf, 1)
                elif rank == 1:
                    dist.recv(buf, 0)
                got = read(buf)
            elif method == "broadcast":
                dist.broadcast(buf, 0)
                got = read(buf)
            elif method == "a2a":
                # The peer stash without the waste: scatter (rank 0's quarters
                # to each rank) and gather (back to rank 0 only) as uneven
                # all_to_all_single. Every rank also gets one element from
                # every other, so it has an output to read -- an op with no
                # read result is the one lazy execution dropped (send/recv).
                #   scatter: rank 0 sends a quarter q to every rank; every other
                #            rank sends 1 element to every rank. All receive
                #            [q, 1, 1, 1] -- the quarter is out[:q].
                #   gather:  every rank sends its quarter to rank 0 and 1
                #            element to each other rank; rank 0 receives
                #            [q, q, q, q], the others [1, 1, 1, 1].
                q = n // world
                send = [q] * world if rank == 0 else [1] * world
                recv = [q] + [1] * (world - 1)
                inp = buf if rank == 0 else torch.ones(world, dtype=buf.dtype, device=dev)
                part = torch.empty(sum(recv), dtype=buf.dtype, device=dev)
                dist.all_to_all_single(part, inp, output_split_sizes=recv, input_split_sizes=send)
                got_rs = read(part[:q])
                t_rs = time.monotonic() - mark
                g_send = [q] + [1] * (world - 1)
                g_recv = [q] * world if rank == 0 else [1] * world
                g_inp = part[: sum(g_send)]
                full = torch.empty(sum(g_recv), dtype=buf.dtype, device=dev)
                mark2 = time.monotonic()
                dist.all_to_all_single(full, g_inp, output_split_sizes=g_recv, input_split_sizes=g_send)
                got = read(full)
                entry.setdefault("scatter_s", []).append(t_rs)
                entry.setdefault("gather_s", []).append(time.monotonic() - mark2)
                entry["scatter_last"] = got_rs
                del part, full, inp
            elif method == "rs_ag":
                part = torch.empty(n // world, dtype=torch.bfloat16, device=dev)
                dist.reduce_scatter_tensor(part, buf)
                got_rs = read(part)
                t_rs = time.monotonic() - mark
                buf.fill_(0.0)
                read(buf)
                mark2 = time.monotonic()
                dist.all_gather_into_tensor(buf, part)
                got = read(buf)
                entry.setdefault("rs_s", []).append(t_rs)
                entry.setdefault("ag_s", []).append(time.monotonic() - mark2)
                entry["rs_last"] = got_rs
                del part
            else:
                raise ValueError(method)
            times.append(time.monotonic() - mark)
            entry["last_elem"] = got
            dist.barrier()
            del buf
            device_runtime.wait()
        entry["s"] = times
        entry["GBps_best"] = size * gib / 1e9 / min(times)
        out[f"{size}GiB"] = entry
        print({"rank": rank, "method": method, "size_gib": size, **entry}, flush=True)
    return out


def main() -> None:
    from difflet.backends.tpu.ops_impl import device_runtime

    method = sys.argv[1]
    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    device_runtime.prepare_launch(world)
    ctx = mp.get_context("spawn")
    reply_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, world, method, reply_q)) for r in range(world)]
    for p in procs:
        p.start()
    for _ in range(world):
        print(reply_q.get(timeout=280), flush=True)
    for p in procs:
        p.join(timeout=30)


if __name__ == "__main__":
    main()
