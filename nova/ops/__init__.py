"""Hardware-dispatched operation surface for model code."""

_EXPORTS = {
    "ColumnParallelLinear": ("linear", "ColumnParallelLinear"),
    "ParallelEmbedding": ("linear", "ParallelEmbedding"),
    "RowParallelLinear": ("linear", "RowParallelLinear"),
    "LayerNorm": ("norm", "LayerNorm"),
    "RMSNorm": ("norm", "RMSNorm"),
    "CustomRMSNorm": ("norm", "CustomRMSNorm"),
    "attention": ("attention", "attention"),
    "attention_cte": ("attention", "attention_cte"),
    "cross_attention": ("attention", "cross_attention"),
    "apply_rotary_emb": ("embeddings", "apply_rotary_emb"),
    "gather_tp_dim": ("collectives", "gather_tp_dim"),
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
        raise AttributeError(f"module 'nova.ops' has no attribute {name!r}") from exc

    from nova.ops._dispatch import load_backend_attr

    return load_backend_attr(module_name, attr_name)
