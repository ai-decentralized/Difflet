"""Two real ranks under gloo: the CUDA mesh's process groups and collectives (F2).

The CUDA backend uses NCCL on GPUs; the group bookkeeping and the collective
calls are backend-agnostic ``torch.distributed`` code, so two CPU processes under
gloo exercise the same path a ``torchrun --nproc-per-node 2`` launch takes.
"""

import datetime
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from difflet.backends.cuda.ops_impl import collectives as C
from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec

WORLD = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(WORLD),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD, timeout=datetime.timedelta(seconds=30)
    )
    try:
        pm.init_parallel_mesh(MeshSpec(cfg=2))
        group = pm.get_cfg_group()
        assert isinstance(group, dist.ProcessGroup), type(group)
        assert dist.get_world_size(group) == WORLD
        assert C._group_size(group) == WORLD and C._group_ranks(group) == [0, 1]
        assert pm.get_axis_rank("cfg") == rank
        # cp is trivial at cfg=2 on two ranks: it stays a replica list.
        assert pm.get_cp_group() == [[0], [1]]

        # gather: concatenation in rank order along the gather dim.
        local = torch.arange(6.0).reshape(2, 3) + 100.0 * rank
        gathered = C.gather_from_tensor_model_parallel_region_with_dim(
            local, 1, process_group=group
        )
        expected = torch.cat([torch.arange(6.0).reshape(2, 3) + 100.0 * r for r in range(WORLD)], 1)
        assert torch.equal(gathered, expected), gathered

        # scatter: this rank's slice (a local narrow, no communication).
        full = torch.arange(8.0).reshape(2, 4)
        shard = C.scatter_to_process_group_spmd(full, 1, rank, process_group=group)
        assert torch.equal(shard, full[:, 2 * rank : 2 * rank + 2]), shard

        # reduce: the sum over the group, on a copy (the input is untouched).
        mine = torch.full((2, 2), float(rank + 1))
        reduced = C.reduce_from_tensor_model_parallel_region(mine, process_group=group)
        assert torch.equal(reduced, torch.full((2, 2), 3.0)), reduced
        assert torch.equal(mine, torch.full((2, 2), float(rank + 1)))

        # The scatter/gather round trip is the identity.
        assert torch.equal(
            C.gather_from_tensor_model_parallel_region_with_dim(shard, 1, process_group=group),
            full,
        )

        pm.destroy_parallel_mesh()
        assert pm._AXIS_GROUPS == {}
    finally:
        dist.destroy_process_group()


@pytest.mark.multirank
def test_two_gloo_ranks_build_cfg_group_and_run_collectives(monkeypatch):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with gloo is unavailable in this torch build")
    # --import-mode=importlib names this module after its path under tests/
    # (``unit.backends.test_cuda_multirank``); spawned children unpickle ``_worker``
    # by that name and inherit the parent's sys.path, so put tests/ on it.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    mp.spawn(_worker, args=(_free_port(),), nprocs=WORLD, join=True)
    assert not dist.is_initialized()  # the parent process never joined a group
