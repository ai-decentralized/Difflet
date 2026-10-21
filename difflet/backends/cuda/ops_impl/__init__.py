"""CUDA implementations of the ``difflet.ops`` surface.

One process per device. Collectives are the identity at world size 1; at world size > 1
(under ``torchrun``) the group-taking cfg/cp/dp collectives run on NCCL, and the
tensor-parallel collectives are not supported yet. Layers come from
``difflet.backends._torch_megatron`` bound to this package's collectives.
"""
