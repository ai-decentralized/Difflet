"""LTX-2 model support.

The names resolve lazily, as in ``difflet.models.wan``: ``application`` imports NxD at module
level, so importing this package must not import it (the neuron backend imports ``pipeline``,
``contract`` and ``modeling_ltx_2`` without NxD).
"""

__all__ = [
    "LTX2DiTInputBundle",
    "LTX2Orchestrator",
    "LTX2PipelineOutput",
    "NeuronLTX2Application",
    "create_ltx_2_transformer_config",
    "validate_ltx_2_dit_inputs",
]

_MODULE_MAP = {
    "LTX2DiTInputBundle": "difflet.models.ltx_2.contract",
    "validate_ltx_2_dit_inputs": "difflet.models.ltx_2.contract",
    "NeuronLTX2Application": "difflet.models.ltx_2.application",
    "create_ltx_2_transformer_config": "difflet.models.ltx_2.application",
    "LTX2Orchestrator": "difflet.models.ltx_2.pipeline",
    "LTX2PipelineOutput": "difflet.models.ltx_2.pipeline",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        from importlib import import_module

        return getattr(import_module(_MODULE_MAP[name]), name)
    raise AttributeError(f"module 'difflet.models.ltx_2' has no attribute {name!r}")
