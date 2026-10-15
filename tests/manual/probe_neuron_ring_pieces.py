"""Device probe: the building blocks of ring attention on the neuron backend.

    torchrun --standalone --nproc-per-node 4 tests/manual/probe_neuron_ring_pieces.py

1. flash_fwd_lse: an op returning (output, lse) for unmasked attention, registered with
   torch_neuronx's ``nki_op`` like its own SDPA kernel op, calling the same flash kernel with
   ``training=True`` (``cache_softmax=True``). That path requires the query length to be a
   multiple of 128 (kernel validation NCC_INKI016), so the query is zero-padded to one and the
   padded rows are dropped from the output and the lse (rows are independent without a mask).
   lse is natural-log logsumexp of the scaled scores. Checked against fp32 references, eager and
   torch.compile.
2. ring shift over CP groups {0,2} {1,3} with ``all_to_all_single`` (``permute_tensor`` fails on
   the neuron backend in both modes): this rank's block goes in the slot of its destination.
3. cost: the (output, lse) call against the plain kernel call at [1, 20, L, 128] for
   L = 2,340 (9-frame shard) and 16,380 (81-frame shard), median of 5 timed calls.

Prints PASS/FAIL lines and timings on each rank (timings: rank 0 only).
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", os.environ.get("LOCAL_RANK", "0"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as fc  # noqa: E402

from torch_neuronx.neuron_dynamo_backend.decompositions import _compute_attention_nki  # noqa: E402
from torch_neuronx.nki_hop import nki_op  # noqa: E402
from torch_neuronx.python_ops.nki_kernels.scaled_dot_product_attention import (  # noqa: E402
    scaled_dot_product_attention_kernel,
)


@nki_op("difflet_probe::flash_fwd_lse", mutates_args={})
def flash_fwd_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Unmasked attention and its natural-log logsumexp, [B, H, L, D] and [B, H, L] (fp32)."""
    length = q.shape[2]
    padded = -(-length // 128) * 128
    if padded != length:
        q = torch.nn.functional.pad(q, (0, 0, 0, padded - length))
    out, lse = _compute_attention_nki(q, k, v, False, scale, True)
    return out[:, :, :length].contiguous(), lse[..., :length].contiguous()


def reference(q, k, v, scale):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    return torch.softmax(s, -1) @ v.float(), torch.logsumexp(s, -1)


def check_lse_op(rank, device):
    ok_all = True
    g = torch.Generator().manual_seed(rank)
    for L, S in ((300, 300), (300, 520)):  # query lengths not multiples of 128; L != S too
        q, k, v = (torch.randn(1, 2, n, 128, generator=g, dtype=torch.float32).to(torch.bfloat16)
                   for n in (L, S, S))
        scale = 1 / math.sqrt(128)
        ref_out, ref_lse = reference(q, k, v, scale)
        fn = lambda a, b, c: flash_fwd_lse(a, b, c, scale)  # noqa: E731
        compiled = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        for mode, f in (("eager", fn), ("compile", compiled)):
            try:
                with torch.no_grad():
                    out, lse = f(q.to(device), k.to(device), v.to(device))
                out, lse = out.float().cpu(), lse.float().cpu()
                lse_err = (lse - ref_lse).abs().max().item()
                out_cos = torch.nn.functional.cosine_similarity(out.flatten(), ref_out.flatten(), 0).item()
                ok = tuple(lse.shape) == tuple(ref_lse.shape) and lse_err < 2e-2 and out_cos > 0.9999
                detail = f"lse shape {tuple(lse.shape)} max|err| {lse_err:.2e} out cos {out_cos:.6f}"
            except Exception as exc:  # report and keep going
                ok, detail = False, f"{type(exc).__name__}: {str(exc)[:160]}"
            ok_all &= ok
            print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} lse-op {mode:7s} L={L} S={S}: {detail}",
                  flush=True)
    return ok_all


def check_ring_shift(rank, device):
    groups = [dist.new_group(r) for r in ([0, 2], [1, 3])]
    group, cp_rank = groups[rank % 2], rank // 2
    x = torch.full((4, 8), float(rank), dtype=torch.bfloat16).to(device)
    expected = float([[2, 0], [3, 1]][rank % 2][cp_rank])  # the other member of this CP pair

    def shift(t):
        # cp = 2: slot j of the all-to-all input goes to group rank j; put t in the partner's
        # slot, and take what the partner put in ours.
        slots = [torch.zeros_like(t), torch.zeros_like(t)]
        slots[1 - cp_rank] = t
        out = fc.wait_tensor(fc.all_to_all_single(torch.stack(slots).contiguous(), None, None, group))
        return out[1 - cp_rank]

    compiled = torch.compile(shift, backend="neuron", fullgraph=True, dynamic=False)
    ok_all = True
    for mode, f in (("eager", shift), ("compile", compiled)):
        try:
            with torch.no_grad():
                got = f(x).float().cpu()
            ok = bool((got == expected).all())
            detail = f"received {got[0, 0].item()} expected {expected}"
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:160]}"
        ok_all &= ok
        print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} ring-shift {mode:7s}: {detail}", flush=True)
    return ok_all


def time_cost(rank, device):
    scale = 1 / math.sqrt(128)
    for L in (2340, 16380):
        q, k, v = (torch.randn(1, 20, L, 128, dtype=torch.bfloat16).to(device) for _ in range(3))

        def bench(f):
            with torch.no_grad():
                f(); torch.neuron.synchronize()  # noqa: E702  warm-up (first call loads)
                times = []
                for _ in range(5):
                    t = time.perf_counter(); f(); torch.neuron.synchronize()  # noqa: E702
                    times.append(time.perf_counter() - t)
            return statistics.median(times) * 1e3

        plain = bench(lambda: scaled_dot_product_attention_kernel(q, k, v, scale=scale))
        with_lse = bench(lambda: flash_fwd_lse(q, k, v, scale))
        if rank == 0:
            print(f"[rank 0] cost L={L}: plain {plain:.1f} ms, with lse {with_lse:.1f} ms "
                  f"({(with_lse / plain - 1) * 100:+.1f}%)", flush=True)


def main() -> int:
    import torch_neuronx  # noqa: F401

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    device = torch.device("neuron", int(os.environ.get("LOCAL_RANK", "0")))
    ok = check_lse_op(rank, device)
    ok &= check_ring_shift(rank, device)
    time_cost(rank, device)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
