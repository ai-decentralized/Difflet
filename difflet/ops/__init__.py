"""Hardware-dispatched operation surface for model code."""

_EXPORTS = {
    "ColumnParallelLinear": ("linear", "ColumnParallelLinear"),
    "ParallelEmbedding": ("linear", "ParallelEmbedding"),
    "RowParallelLinear": ("linear", "RowParallelLinear"),
    "LayerNorm": ("norm", "LayerNorm"),
    "RMSNorm": ("norm", "RMSNorm"),
    "CustomRMSNorm": ("norm", "CustomRMSNorm"),
    "SPMDRank": ("collectives", "SPMDRank"),
    "attention": ("attention", "attention"),
    "cross_attention": ("attention", "cross_attention"),
    "ring_attention": ("attention", "ring_attention"),
    "joint_ring_attention": ("attention", "joint_ring_attention"),
    "ulysses_attention": ("attention", "ulysses_attention"),
    "joint_ulysses_attention": ("attention", "joint_ulysses_attention"),
    "apply_rotary_emb": ("embeddings", "apply_rotary_emb"),
    "dequantize_mx": ("mx", "dequantize_mx"),
    "gather_tp_dim": ("collectives", "gather_tp_dim"),
    "init_parallel_mesh": ("collectives", "init_parallel_mesh"),
    "get_cfg_group": ("collectives", "get_cfg_group"),
    "get_cp_group": ("collectives", "get_cp_group"),
    "get_cfg_rank_spmd": ("collectives", "get_cfg_rank_spmd"),
    "get_cp_rank_spmd": ("collectives", "get_cp_rank_spmd"),
    "get_world_group": ("collectives", "get_world_group"),
    "gather_from_tensor_model_parallel_region_with_dim": (
        "collectives",
        "gather_from_tensor_model_parallel_region_with_dim",
    ),
    "reduce_tp": ("collectives", "reduce_tp"),
    "reduce_from_tensor_model_parallel_region": (
        "collectives",
        "reduce_from_tensor_model_parallel_region",
    ),
    "scatter_tp_dim": ("collectives", "scatter_tp_dim"),
    "scatter_to_tensor_model_parallel_region": (
        "collectives",
        "scatter_to_tensor_model_parallel_region",
    ),
    "scatter_to_process_group_spmd": (
        "collectives",
        "scatter_to_process_group_spmd",
    ),
    "scatter_to_sequence_parallel_region": (
        "collectives",
        "scatter_to_sequence_parallel_region",
    ),
    "gather_from_sequence_parallel_region": (
        "collectives",
        "gather_from_sequence_parallel_region",
    ),
    "reduce_scatter_to_sequence_parallel_region": (
        "collectives",
        "reduce_scatter_to_sequence_parallel_region",
    ),
    "get_platform_target": ("platform", "get_platform_target"),
    "hardware": ("platform", "hardware"),
    "linear_mx": ("mx", "linear_mx"),
    "matmul_mx": ("mx", "matmul_mx"),
    "quantize_mx": ("mx", "quantize_mx"),
    "get_tp_rank": ("collectives", "get_tp_rank"),
    "get_tp_size": ("collectives", "get_tp_size"),
    "get_tensor_model_parallel_rank": ("collectives", "get_tensor_model_parallel_rank"),
    "get_tensor_model_parallel_size": ("collectives", "get_tensor_model_parallel_size"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'difflet.ops' has no attribute {name!r}") from exc

    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr(module_name, attr_name)
