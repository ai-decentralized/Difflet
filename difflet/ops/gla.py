""" Gated Linear Attention (GLA) """


def gla_attention(
    q,
    k,
    v,
    g,
    *,
    scale=None,
    causal=False,
    initial_state=None,
    output_final_state=False,
    **kwargs,
):

    return _load("gla_attention")(
        q,
        k,
        v,
        g,
        scale=scale,
        causal=causal,
        initial_state=initial_state,
        output_final_state=output_final_state,
        **kwargs,
    )


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("gla", name)

def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.gla' has no attribute{name!r}")
    return _load(name)
