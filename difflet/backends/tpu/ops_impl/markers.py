"""Layer-boundary markers: a no-op on this backend.

The Trainium (AoT) compiler uses them as layer-boundary hints in its traced graph; here they
return their inputs unchanged (one tensor in, that tensor out; several in, the tuple out), so model
code can call ``difflet.ops.ModuleMarker*Wrapper`` on every backend.
"""

from torch import nn


class _Passthrough(nn.Module):
    def forward(self, *tensors):
        return tensors[0] if len(tensors) == 1 else tensors


class ModuleMarkerStartWrapper(_Passthrough):
    pass


class ModuleMarkerEndWrapper(_Passthrough):
    pass
