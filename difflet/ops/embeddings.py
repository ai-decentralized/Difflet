"""Embedding and positional encoding operations."""


def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    return _load("apply_rotary_emb")(hidden_states, freqs_cos, freqs_sin)


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("embeddings", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.embeddings' has no attribute {name!r}")
    return _load(name)
