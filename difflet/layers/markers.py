"""Backend-resolved layer-boundary markers.

On Trainium the markers are the Neuron custom-call wrappers that delimit layer
boundaries for the compiler. On every other backend they are identity modules, so
model code can wrap layers unconditionally without importing the Neuron SDK.
"""

from __future__ import annotations

from torch import nn

from difflet.backends.registry import current_backend


class _IdentityMarker(nn.Module):
    """Same call contract as the Trainium wrappers: one arg -> itself, many -> tuple."""

    def forward(self, *args):
        return args[0] if len(args) == 1 else args


def resolve_markers(backend: str) -> tuple[type, type]:
    """Return the (start, end) marker classes for ``backend``."""
    if backend == "trainium":
        from difflet.backends.trainium.core.layer_boundary_marker import (
            ModuleMarkerEndWrapper,
            ModuleMarkerStartWrapper,
        )

        return ModuleMarkerStartWrapper, ModuleMarkerEndWrapper
    return _IdentityMarker, _IdentityMarker


ModuleMarkerStartWrapper, ModuleMarkerEndWrapper = resolve_markers(current_backend())
