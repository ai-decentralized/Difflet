"""Megatron-style layers, weight sharding and checkpoint loading shared by torch backends.

Pure torch. The TPU and CUDA backends both instantiate these against their own
collectives module (``linear.build_parallel_layers``), so a sharding bug is fixed once
(CUDA backend plan, decision D10). ``weights`` and ``checkpoint`` take ``tp_size`` /
``tp_rank`` as arguments and have no backend coupling at all.
"""
