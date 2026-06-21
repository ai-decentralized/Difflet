# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/utils/decorator_peeling.py
# Fork date: 2026-05-08
# Modifications: (none — verbatim copy; see git log for divergence)
# <<< NxDI fork banner <<<
def peel_decorations(decorated_function):
    undecorated_function = decorated_function
    while hasattr(undecorated_function, "__wrapped__"):
        undecorated_function = undecorated_function.__wrapped__

    return undecorated_function
