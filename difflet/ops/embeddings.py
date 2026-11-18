"""Embedding and positional encoding operations."""


def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    return _load("apply_rotary_emb")(hidden_states, freqs_cos, freqs_sin)


def apply_rotary_emb_fp32(hidden_states, freqs_cos, freqs_sin):
    return _load("apply_rotary_emb_fp32")(hidden_states, freqs_cos, freqs_sin)


def apply_rotary_emb_slices(hidden_states, freqs_cos, freqs_sin):
    return load_apply_rotary_emb_slices()(hidden_states, freqs_cos, freqs_sin)


def load_apply_rotary_emb_slices():
    """The interleaved rotary computed with stride-2 slices: the backend's
    ``apply_rotary_emb_slices``, or its ``apply_rotary_emb`` where it has none, which is that law
    on every backend but neuron. Neuron's ``apply_rotary_emb`` swaps the pairs by a matmul
    (``x @ P``), and with it two bf16 Wan runs on the device differ; Wan's interleaved
    self-attention calls this op, HunyuanVideo keeps ``x @ P``."""
    try:
        return _load("apply_rotary_emb_slices")
    except NotImplementedError:
        return _load("apply_rotary_emb")


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("embeddings", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.embeddings' has no attribute {name!r}")
    return _load(name)
