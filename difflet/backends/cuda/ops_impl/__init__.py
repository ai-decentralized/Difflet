"""CUDA implementations of the ``difflet.ops`` surface.

Release 1 (docs/plans/2026-10-16-cuda-backend-support.md): one process, one
device. Collectives are the identity at world size 1 and raise for anything
that would need a second rank. Layers come from
``difflet.backends._torch_megatron`` bound to this package's collectives.
"""
