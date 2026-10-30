"""Device probe: Ulysses' layout swaps with all_to_all_single over a CP subgroup, eager and compiled.

    torchrun --standalone --nproc-per-node 4 tests/manual/probe_neuron_all_to_all.py

Layout TP2 x CP2 (rank = cp * 2 + tp): CP groups {0,2} {1,3}. Each rank holds a sequence shard
q[B=1, H=4, S/cp=6, d=8] whose values encode (global position, head). The forward swap must give
[1, H/cp=2, S=12, 8]: this rank's head block, the whole sequence in order; the inverse swap must
restore the input exactly. Checked bit-exact in eager mode and inside
torch.compile(backend="neuron", fullgraph=True). Prints PASS/FAIL per check.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", os.environ.get("LOCAL_RANK", "0"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed._functional_collectives as fc  # noqa: E402

B, H, S_LOC, D, CP = 1, 4, 6, 8, 2


def seq_to_heads(t, group):
    """[B, H, S/cp, d] -> [B, H/cp, S, d]: send head block j to CP rank j; received blocks
    (from CP ranks 0..cp-1, i.e. sequence shards in order) concatenate along the sequence."""
    b, h, s, d = t.shape
    x = t.permute(1, 0, 2, 3).contiguous()                        # [H, B, S/cp, d]
    y = fc.wait_tensor(fc.all_to_all_single(x, None, None, group))  # [cp*(H/cp), B, S/cp, d]
    y = y.reshape(CP, h // CP, b, s, d)                             # [cp(src), H/cp, B, S/cp, d]
    return y.permute(2, 1, 0, 3, 4).reshape(b, h // CP, CP * s, d)


def heads_to_seq(t, group):
    """[B, H/cp, S, d] -> [B, H, S/cp, d]: the inverse swap."""
    b, hl, s, d = t.shape
    x = t.reshape(b, hl, CP, s // CP, d).permute(2, 0, 1, 3, 4).contiguous()  # [cp(dst), B, H/cp, S/cp, d]
    y = fc.wait_tensor(fc.all_to_all_single(x.reshape(CP * b, hl, s // CP, d), None, None, group))
    y = y.reshape(CP, b, hl, s // CP, d)                            # [cp(src), B, H/cp, S/cp, d]
    return y.permute(1, 0, 2, 3, 4).reshape(b, CP * hl, s // CP, d)


def main() -> int:
    import torch_neuronx  # noqa: F401

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    device = torch.device("neuron", int(os.environ.get("LOCAL_RANK", "0")))
    cp_groups = [dist.new_group(r) for r in ([0, 2], [1, 3])]
    group = cp_groups[rank % 2]
    cp_rank = rank // 2

    pos = torch.arange(cp_rank * S_LOC, (cp_rank + 1) * S_LOC, dtype=torch.float32)  # global positions
    heads = torch.arange(H, dtype=torch.float32)
    q = (pos.view(1, 1, S_LOC, 1) * 100 + heads.view(1, H, 1, 1)).expand(B, H, S_LOC, D).contiguous()
    q = q + 10000 * (rank % 2)  # tp coordinate, so the two CP groups carry different data
    full_pos = torch.arange(CP * S_LOC, dtype=torch.float32)
    my_heads = torch.arange(cp_rank * (H // CP), (cp_rank + 1) * (H // CP), dtype=torch.float32)
    expected = (full_pos.view(1, 1, -1, 1) * 100 + my_heads.view(1, -1, 1, 1)).expand(B, H // CP, CP * S_LOC, D)
    expected = expected + 10000 * (rank % 2)

    def round_trip(t):
        swapped = seq_to_heads(t, group)
        return swapped, heads_to_seq(swapped, group)

    compiled = torch.compile(round_trip, backend="neuron", fullgraph=True, dynamic=False)
    ok_all = True
    for mode, fn in (("eager", round_trip), ("compile", compiled)):
        with torch.no_grad():
            swapped, back = fn(q.to(device))
        ok_fwd = torch.equal(swapped.cpu(), expected)
        ok_inv = torch.equal(back.cpu(), q)
        ok_all = ok_all and ok_fwd and ok_inv
        print(f"[rank {rank}] {'PASS' if ok_fwd else 'FAIL'} {mode:7s} seq->heads  "
              f"{'PASS' if ok_inv else 'FAIL'} {mode:7s} heads->seq round trip", flush=True)
    dist.destroy_process_group()
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
