"""CUDA implementations of the ``difflet.ops`` surface.

Release 1 (docs/plans/2026-10-16-cuda-backend-support.md): one process, one
device. Collectives are the identity at world size 1; at world size > 1 (F2,
under ``torchrun``) the group-taking cfg/cp/dp collectives run on NCCL, and the
tensor-parallel collectives are follow-up F6. Layers come from
``difflet.backends._torch_megatron`` bound to this package's collectives.
"""
