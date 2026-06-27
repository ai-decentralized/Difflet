"""Wan model integration."""

__all__ = ["NeuronWanApplication", "WanOrchestrator", "WanPipelineOutput"]

_MODULE_MAP = {
    "NeuronWanApplication": "difflet.models.wan.application",
    "WanOrchestrator": "difflet.models.wan.pipeline",
    "WanPipelineOutput": "difflet.models.wan.pipeline",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        from importlib import import_module

        module = import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module 'difflet.models.wan' has no attribute {name!r}")
